import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior
from dexmani_policy.agents.policies.lamp import LAMPPolicy
from dexmani_policy.common.checkpoint import (
    TrainingCheckpoint,
    load_training_checkpoint,
    restore_policy_agent,
    save_training_checkpoint,
)
from dexmani_policy.common.config import (
    load_policy_config,
    resolve_input_recipe,
    validate_source_recipe,
)
from dexmani_policy.common.normalization import AffineNormalizer, FieldNormalizer
from dexmani_policy.common.runtime import get_rng_state
from dexmani_policy.training.builder import (
    build_model_and_ema,
    build_optimizer_and_scheduler,
)
from dexmani_policy.training.common.hand_representation import describe_hand_data
from dexmani_policy.training.data import build_dataset_and_normalizer, build_train_loader
from dexmani_policy.training.resume import build_resume_contract, restore_training_state
from dexmani_policy.training.validation import validate_config


class LAMPBCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.kwargs = dict(
            horizon=1, n_obs_steps=1, n_action_steps=1, action_dim=19, prior_path="",
            history_steps=3, latent_dim=2, prior_hidden_dim=8, num_points=8, state_out_dim=8,
            pc_encoder_config=dict(
                channels=24, num_patches=2, group_size=4, depth=1, num_heads=4,
            ),
            fps_random_config={"use_random": False},
        )
        self.loss_config = dict(
            history_noise_std=0.01, hand_loss_weight=1.0,
            offset_reg_weight=0.0, drift_reg_weight=1.0,
        )
        self.batch = dict(
            obs={"joint_state": torch.randn(2, 1, 19), "point_cloud": torch.randn(2, 1, 8, 6)},
            action=torch.randn(2, 1, 19),
            action_context={"hand_history": torch.randn(2, 3, 12)},
        )

    def model(self, **weights):
        model = LAMPPolicy(**self.kwargs, **{**self.loss_config, **weights})
        scale, offset = torch.linspace(0.4, 2.5, 19), torch.linspace(-0.3, 0.2, 19)
        normalizer = AffineNormalizer()
        normalizer["action"] = FieldNormalizer.from_parameters(scale, offset)
        normalizer["joint_state"] = FieldNormalizer.from_parameters(torch.ones(19), torch.zeros(19))
        model.prior.hand_scale.copy_(scale[7:])
        model.prior.hand_offset.copy_(offset[7:])
        model.prior.ready.fill_(True)
        model.prior_initialized = True
        model.load_normalizer_from_dataset(normalizer)
        return model

    def test_history_noise_scale_modes_and_input_preservation(self):
        for training, std in ((True, 0.01), (False, 0.01), (True, 0.0)):
            with self.subTest(training=training, std=std):
                model = self.model(history_noise_std=std).train(training)
                self.assertFalse(model.prior.training)
                saved = copy.deepcopy(self.batch)
                history = self.batch["action_context"]["hand_history"]
                rng = torch.get_rng_state().clone()
                expected = model.prior.normalize_hand(history)
                if training and std:
                    expected = expected + torch.randn_like(history) * std
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                encoded = []
                handle = model.prior.encoder.register_forward_pre_hook(
                    lambda module, args: encoded.append(args[0].detach().clone())
                )
                try:
                    _, mu = model.encode_condition(self.batch["obs"], self.batch["action_context"])
                finally:
                    handle.remove()
                self.assertEqual(len(encoded), 1)
                torch.testing.assert_close(encoded[0], expected.flatten(1), rtol=1e-6, atol=1e-6)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                torch.testing.assert_close(self.batch, saved, rtol=0, atol=0)
                if not training or not std:
                    torch.testing.assert_close(
                        mu, model.prior.encode_history(history)[0], rtol=0, atol=0
                    )

    def test_loss_formula_and_decoder_gradient(self):
        for drift, latent, hand_weight in ((1.0, 0.0, 0.0), (0.0, 0.03, 0.7), (0.0, 0.0, 0.7)):
            with self.subTest(drift=drift, latent=latent):
                model = self.model(
                    drift_reg_weight=drift, offset_reg_weight=latent,
                    hand_loss_weight=hand_weight, history_noise_std=0.01,
                ).train()
                saved = copy.deepcopy(self.batch)
                prior_state = copy.deepcopy(model.prior.state_dict())
                captured = {}

                def capture_mu(module, inputs, output):
                    captured["mu"] = output.detach().clone()

                def capture_native(module, inputs, output):
                    captured["native"] = output
                    output.retain_grad()

                handles = [
                    model.prior.fc_mu.register_forward_hook(capture_mu),
                    model.action_decoder.register_forward_hook(capture_native),
                ]
                try:
                    loss, metrics = model.compute_loss(self.batch)
                finally:
                    for handle in handles:
                        handle.remove()
                native, mu = captured["native"], captured["mu"]
                offset = native[:, 7:].float()
                predicted = model.prior.decode(mu + offset)
                with torch.no_grad():
                    reference = model.prior.decode(mu)
                action = self.batch["action"][:, 0]
                hand_target = action[:, 7:] * model.prior.hand_scale + model.prior.hand_offset
                arm_target = model.normalizer["action"].normalize(action)[:, :7]
                hand_mse = (predicted - hand_target).square().mean()
                drift_mse = (predicted - reference).square().mean()
                expected = (
                    (native[:, :7] - arm_target).square().mean() + hand_weight * hand_mse
                    + drift * drift_mse + latent * offset.square().sum(-1).mean()
                )
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(metrics["hand_drift_mse"], drift_mse)
                torch.testing.assert_close(metrics["loss_drift"], drift * drift_mse)
                torch.testing.assert_close(
                    metrics["prior_hand_mse"], F.mse_loss(reference, hand_target)
                )
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.prior.parameters()))
                gradients = [p.grad for p in model.action_decoder.parameters()]
                self.assertTrue(all(g is not None and g.isfinite().all() for g in gradients))
                self.assertGreater(float(native.grad[:, 7:].abs().sum()), 0)
                torch.testing.assert_close(self.batch, saved, rtol=0, atol=0)
                torch.testing.assert_close(model.prior.state_dict(), prior_state, rtol=0, atol=0)

    def test_logging_invariance_and_decode_counts(self):
        for drift in (0.0, 1.0):
            model = self.model(
                drift_reg_weight=drift, offset_reg_weight=0.0, history_noise_std=0.01,
            ).train()
            rng = torch.get_rng_state().clone()
            results = []
            for log_metrics in (False, True):
                torch.set_rng_state(rng)
                model.zero_grad(set_to_none=True)
                with (
                    patch.object(model.prior, "decode", wraps=model.prior.decode) as decode,
                    patch.object(
                        model.prior, "encode_history", wraps=model.prior.encode_history
                    ) as encode,
                    torch.autocast("cpu", dtype=torch.bfloat16),
                ):
                    loss, metrics = model.compute_loss(self.batch, log_metrics=log_metrics)
                    self.assertEqual(decode.call_count, 2 if drift or log_metrics else 1)
                    self.assertEqual(encode.call_count, 1)
                    self.assertTrue(all(
                        call.args[0].dtype == torch.float32 for call in decode.call_args_list
                    ))
                loss.backward()
                results.append((loss.detach(), {
                    name: p.grad.clone()
                    for name, p in model.named_parameters() if p.grad is not None
                }))
                self.assertEqual(bool(metrics), log_metrics)
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
            model.eval()
            rng = torch.get_rng_state().clone()
            with patch.object(model.prior, "decode", wraps=model.prior.decode) as decode:
                prediction = model.predict_action(
                    self.batch["obs"], action_context=self.batch["action_context"]
                )
                self.assertEqual(decode.call_count, 1)
            self.assertTrue(prediction["control_action"].isfinite().all())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_parameters_and_new_config(self):
        cfg = load_policy_config("lamp")
        validate_config(cfg)
        fields = (
            "history_noise_std", "hand_loss_weight", "offset_reg_weight", "drift_reg_weight"
        )
        self.assertEqual([cfg.agent[key] for key in fields], [0.01, 1.0, 0.0, 1.0])
        for key in fields:
            incomplete = {**self.kwargs, **self.loss_config}
            del incomplete[key]
            with self.assertRaisesRegex(TypeError, key):
                LAMPPolicy(**incomplete)
            for value in (-1.0, float("inf"), float("nan")):
                with self.subTest(key=key, value=value):
                    weights = {**self.loss_config, key: value}
                    with self.assertRaisesRegex(ValueError, key):
                        LAMPPolicy(**self.kwargs, **weights)
                    invalid = copy.deepcopy(cfg)
                    invalid.agent[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        validate_config(invalid)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LAMPPolicy(**self.kwargs, **{**self.loss_config, "offset_reg_weight": 1e-3})
        cfg.agent.offset_reg_weight = 1e-3
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_config(cfg)

    def test_builder_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rng = np.random.default_rng(31)
            root = zarr.open_group(str(path / "data.zarr"), mode="w")
            root.attrs["dt"] = 0.1
            fields = (("action", (7, 19)), ("joint_state", (7, 19)), ("point_cloud", (7, 8, 6)))
            for key, shape in fields:
                root.create_dataset("data/" + key, data=rng.normal(size=shape).astype(np.float32))
            root.create_dataset("meta/episode_ends", data=np.array([7], dtype=np.int64))
            cfg = load_policy_config("lamp")
            with open_dict(cfg):
                cfg.workspace.output_dir = str(path / "run")
                cfg.workspace.wandb_cfg = None
                cfg.zarr_path = str(path / "data.zarr")
                cfg.lamp.history_steps = 3
                cfg.lamp.hidden_dim = 8
                cfg.lamp.prior_path = str(path / "prior.pt")
                cfg.agent.num_points = 8
                cfg.agent.state_out_dim = 8
                cfg.agent.pc_encoder_config = self.kwargs["pc_encoder_config"]
                cfg.agent.fps_random_config = self.kwargs["fps_random_config"]
                cfg.dataset.augmentation_cfg = None
                cfg.dataloader.num_workers = 0
                cfg.dataloader.persistent_workers = False
                cfg.dataloader.prefetch_factor = None
                cfg.dataloader.batch_size = 2
                cfg.training.device = "cpu"
                cfg.training.use_compile = False
                cfg.training.use_bfloat16 = False
            OmegaConf.resolve(cfg)
            validate_config(cfg)
            dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
            prior = HandMotionPrior(12, 3, 2, 8)
            prior.hand_scale.copy_(normalizer["action"].params_dict["scale"][7:])
            prior.hand_offset.copy_(normalizer["action"].params_dict["offset"][7:])
            prior.ready.fill_(True)
            torch.save(dict(
                model_config=prior.model_config, model_state_dict=prior.state_dict(),
                metadata={"hand_data": describe_hand_data(dataset)},
            ), path / "prior.pt")
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            model, ema, updater = build_model_and_ema(cfg, "cpu", normalizer, dataset=dataset)
            optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=False)
            loader = build_train_loader(cfg, dataset)
            model.train()
            optimizer.zero_grad()
            loss, metrics = model.compute_loss(batch)
            loss.backward()
            self.assertTrue(loss.isfinite())
            self.assertGreater(float(metrics["hand_drift_mse"]), 0)
            self.assertTrue(all(
                p.grad is not None and p.grad.isfinite().all()
                for p in model.action_decoder.parameters()
            ))
            optimizer.step()
            scheduler.step()
            updater.step(model)
            saved = OmegaConf.to_container(cfg, resolve=True)
            checkpoint = TrainingCheckpoint(
                epoch=0, global_step=1, next_micro_step=1,
                model_state=model.state_dict(), ema_model_state=ema.state_dict(),
                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                resume_contract=build_resume_contract(loader),
                ema_updater_step=updater.optimization_step,
                rng_states=[get_rng_state()], effective_config=saved,
            )
            checkpoint_path = save_training_checkpoint(path / "policy.pt", checkpoint)
            checkpoint = load_training_checkpoint(checkpoint_path)
            (path / "prior.pt").unlink()
            invocation = copy.deepcopy(cfg)
            with open_dict(invocation):
                invocation.resume_from = str(checkpoint_path)
            resumed_cfg = resolve_input_recipe(
                invocation, overrides=[], resume_checkpoint=checkpoint
            )
            resumed, resumed_ema, resumed_updater = build_model_and_ema(
                resumed_cfg, "cpu", None, dataset=dataset, checkpoint=checkpoint,
            )
            resumed_optimizer, resumed_scheduler = build_optimizer_and_scheduler(
                resumed_cfg, resumed, verbose=False,
            )
            validate_source_recipe(saved, resumed_cfg)
            components = SimpleNamespace(
                train_loader=loader, optimizer=resumed_optimizer, scheduler=resumed_scheduler,
                ema_updater=resumed_updater, device="cpu",
            )
            self.assertEqual(
                restore_training_state(checkpoint, components, effective_config=resumed_cfg),
                (1, 0, 1),
            )
            torch.testing.assert_close(
                resumed_optimizer.state_dict()["state"], checkpoint.optimizer_state["state"],
                rtol=0, atol=0,
            )
            self.assertEqual(
                resumed_optimizer.state_dict()["param_groups"],
                checkpoint.optimizer_state["param_groups"],
            )
            self.assertEqual(resumed_scheduler.state_dict(), checkpoint.scheduler_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), checkpoint.rng_states[0]["torch"]))
            for name, value in self.loss_config.items():
                self.assertEqual(getattr(resumed, name), value)
            for use_ema, expected, restored in ((False, model, resumed), (True, ema, resumed_ema)):
                torch.testing.assert_close(
                    expected.state_dict(), restored.state_dict(), rtol=0, atol=0
                )
                expected.eval()
                inference = restore_policy_agent(
                    saved, checkpoint_path, use_ema=use_ema, device="cpu"
                )
                kwargs = dict(action_context=batch["action_context"])
                torch.testing.assert_close(
                    expected.predict_action(batch["obs"], **kwargs),
                    inference.predict_action(batch["obs"], **kwargs), rtol=0, atol=0,
                )
            resumed.train()
            rng_state = torch.get_rng_state()
            expected_loss, _ = model.train().compute_loss(batch, log_metrics=False)
            torch.set_rng_state(rng_state)
            restored_loss, _ = resumed.compute_loss(batch, log_metrics=False)
            torch.testing.assert_close(expected_loss, restored_loss, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
