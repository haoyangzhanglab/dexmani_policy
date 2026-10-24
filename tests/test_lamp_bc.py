import copy
import json
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

from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior, load_motion_prior
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
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.training.builder import (
    build_model_and_ema,
    build_optimizer_and_scheduler,
)
from dexmani_policy.training.common.hand_representation import describe_hand_data
from dexmani_policy.datasets.common.episode_split import build_index_manifest
from dexmani_policy.training.builder import build_optimizer
from dexmani_policy.training.pretrain.hand_prior.lamp import perturb_history
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
            history_noise_std=0.10, hand_loss_weight=1.0,
            arm_state_noise_std=0.0, drift_reg_weight=1.0,
        )
        self.batch = dict(
            obs={"joint_state": torch.randn(2, 1, 19), "point_cloud": torch.randn(2, 1, 8, 6)},
            action=torch.randn(2, 1, 19),
            action_context={"hand_history": torch.randn(2, 3, 12)},
        )

    def model(self, **weights):
        model = LAMPPolicy(**self.kwargs, **{**self.loss_config, **weights})
        normalizer = AffineNormalizer()
        normalizer["joint_state"] = FieldNormalizer.from_parameters(torch.ones(19), torch.zeros(19))
        model.prior.ready.fill_(True)
        model.prior_initialized = True
        model.set_normalization_spec(
            {"action": "identity", "joint_state": "gaussian", "point_cloud": "identity"},
            {"action": 19, "joint_state": 19},
        )
        model.load_normalizer_from_dataset(normalizer)
        return model

    def test_history_noise_scale_modes_and_input_preservation(self):
        for training, std in ((True, 0.10), (False, 0.10), (True, 0.0)):
            with self.subTest(training=training, std=std):
                model = self.model(history_noise_std=std).train(training)
                self.assertFalse(model.prior.training)
                saved = copy.deepcopy(self.batch)
                history = self.batch["action_context"]["hand_history"]
                rng = torch.get_rng_state().clone()
                expected = history
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
        for drift, hand_weight in ((1.0, 0.0), (0.0, 0.7), (1.0, 1.0)):
            with self.subTest(drift=drift, hand_weight=hand_weight):
                model = self.model(
                    drift_reg_weight=drift,
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
                hand_target = action[:, 7:]
                arm_target = action[:, :7]
                hand_mse = (predicted - hand_target).square().mean()
                drift_mse = (predicted - reference).square().mean()
                expected = (
                    (native[:, :7] - arm_target).square().mean() + hand_weight * hand_mse
                    + drift * drift_mse
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
                drift_reg_weight=drift, history_noise_std=0.10, arm_state_noise_std=0.10,
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
                    self.assertEqual(decode.call_count, 2)
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
            "history_noise_std", "hand_loss_weight", "arm_state_noise_std", "drift_reg_weight"
        )
        self.assertEqual([cfg.agent[key] for key in fields], [0.10, 1.0, 0.10, 1.0])
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
        with self.assertRaisesRegex(TypeError, "offset_reg_weight"):
            LAMPPolicy(**self.kwargs, **{**self.loss_config, "offset_reg_weight": 1e-3})

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
                cfg.dataset.val_ratio = 0.0
                cfg.dataset.split_manifest = str(path / "split.json")
                cfg.agent.arm_state_noise_std = 0.10
                cfg.training.lr_warmup_steps = 2
                cfg.training.loop.total_train_steps = 6
                cfg.dataloader.num_workers = 0
                cfg.dataloader.persistent_workers = False
                cfg.dataloader.prefetch_factor = None
                cfg.dataloader.batch_size = 2
                cfg.training.device = "cpu"
                cfg.training.use_compile = False
                cfg.training.use_bfloat16 = False
            manifest = build_index_manifest(str(path / "data.zarr"), [7], dict(root.attrs), val_ratio=0)
            (path / "split.json").write_text(json.dumps(manifest))
            OmegaConf.resolve(cfg)
            validate_config(cfg)
            dataset, normalizer = build_dataset_and_normalizer(cfg, verbose=False)
            prior = HandMotionPrior(12, 3, 2, 8)
            prior.ready.fill_(True)
            torch.save(dict(
                format=prior.artifact_format, coordinate_space=prior.coordinate_space,
                model_config=prior.model_config, model_state_dict=prior.state_dict(),
                metadata={"hand_data": describe_hand_data(dataset), "split_manifest": manifest,
                          "hand_joint_order": prior.hand_joint_order},
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
            self.assertIsNone(updater)
            self.assertIsNone(ema)
            saved = OmegaConf.to_container(cfg, resolve=True)
            checkpoint = TrainingCheckpoint(
                epoch=0, global_step=1, next_micro_step=1,
                model_state=model.state_dict(), ema_model_state=None,
                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                resume_contract=build_resume_contract(loader),
                ema_updater_step=None,
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
                self.assertEqual(getattr(resumed, name), cfg.agent[name])
            torch.testing.assert_close(model.state_dict(), resumed.state_dict(), rtol=0, atol=0)
            model.eval()
            inference = restore_policy_agent(
                saved, checkpoint_path, use_ema=False, device="cpu"
            )
            kwargs = dict(action_context=batch["action_context"])
            torch.testing.assert_close(
                model.predict_action(batch["obs"], **kwargs),
                inference.predict_action(batch["obs"], **kwargs), rtol=0, atol=0,
            )
            resumed.train()
            rng_state = torch.get_rng_state()
            optimizer.zero_grad(set_to_none=True)
            resumed_optimizer.zero_grad(set_to_none=True)
            expected_loss, _ = model.train().compute_loss(batch, log_metrics=False)
            torch.set_rng_state(rng_state)
            restored_loss, _ = resumed.compute_loss(batch, log_metrics=False)
            torch.testing.assert_close(expected_loss, restored_loss, rtol=0, atol=0)
            expected_loss.backward()
            restored_loss.backward()
            optimizer.step()
            resumed_optimizer.step()
            scheduler.step()
            resumed_scheduler.step()
            torch.testing.assert_close(model.state_dict(), resumed.state_dict(), rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_prior_raw_math_and_noise(self):
        prior = HandMotionPrior(12, 8, 2, 256)
        with torch.no_grad():
            prior.fc_mu.weight.zero_()
            prior.fc_mu.bias.copy_(torch.tensor([1.0, -2.0]))
            prior.fc_log_var.weight.zero_()
            prior.fc_log_var.bias.copy_(torch.tensor([4.0, 0.25]).log())
            prior.decoder[-1].bias.fill_(3.0)
        history = torch.linspace(-3, 4, 2 * 8 * 12).reshape(2, 8, 12)
        target = torch.linspace(-2, 3, 24).reshape(2, 12)
        epsilon = torch.tensor([[0.25, -0.5], [1.0, 2.0]])
        z = torch.tensor([[1.5, -2.25], [3.0, -1.0]])
        prediction = prior.decode(z)
        expected_recon = (prediction - target).square().sum() / 24
        expected_kl = 0.5 * (torch.tensor([1.0, 4.0]) + torch.tensor([4.0, 0.25])
                             - 1 - torch.tensor([4.0, 0.25]).log()).mean()
        loss, recon, kl = prior.compute_loss(history, target, 0.001, epsilon=epsilon)
        torch.testing.assert_close(recon, expected_recon)
        torch.testing.assert_close(kl, expected_kl)
        torch.testing.assert_close(loss, expected_recon + 0.001 * expected_kl)
        self.assertTrue((prediction > 1).all())
        self.assertFalse(any("scale" in key or "offset" in key for key in prior.state_dict()))
        loss.backward()
        self.assertGreater(float(prior.fc_log_var.bias.grad.abs().sum()), 0)

        history = torch.full((4096, 8, 12), 2.5)
        for std in (0.0, 0.01):
            rng = torch.get_rng_state()
            expected = history + torch.randn_like(history) * std if std else history
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            noisy = perturb_history(history, std)
            torch.testing.assert_close(noisy, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertTrue(torch.equal(history, torch.full_like(history, 2.5)))
            if std:
                self.assertAlmostEqual(float((noisy - history).std()), std, delta=0.0001)

    def test_arm_noise_after_normalization(self):
        model = self.model(arm_state_noise_std=0.10)
        scale = torch.linspace(0.2, 3, 19)
        offset = torch.linspace(-1, 1, 19)
        model.normalizer["joint_state"] = FieldNormalizer.from_parameters(scale, offset)
        raw = torch.linspace(-2, 4, 4096 * 19).reshape(4096, 1, 19)
        original = raw.clone()
        for training, std in ((True, 0.10), (False, 0.10), (True, 0.0)):
            model.train(training)
            model.arm_state_noise_std = std
            rng = torch.get_rng_state()
            expected = (raw * scale + offset).flatten(0, 1)
            if training and std:
                expected[:, :7] += torch.randn_like(expected[:, :7]) * std
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = model.preprocess({"joint_state": raw})["joint_state"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[:, 7:], (raw * scale + offset)[:, 0, 7:])
            torch.testing.assert_close(raw, original, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_prior_features_and_optimizer_update(self):
        model = self.model(history_noise_std=0, arm_state_noise_std=0).train()
        captured = []
        handle = model.prior_feature_mlp.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        before = copy.deepcopy(model.state_dict())
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4, obs_lr=2e-3)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(params), len(set(map(id, params))))
        self.assertEqual(set(map(id, params)), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertTrue(all(g["weight_decay"] == 1e-4 for g in optimizer.param_groups))
        loss, _ = model.compute_loss(self.batch)
        handle.remove()
        mu, log_var = model.prior.encode_history(self.batch["action_context"]["hand_history"])
        torch.testing.assert_close(captured[0], torch.cat((mu, log_var), dim=-1))
        loss.backward()
        for module in (model.prior_feature_mlp, model.action_decoder, model.obs_encoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and g.isfinite().all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.prior.parameters()))
        optimizer.step()
        after = model.state_dict()
        for prefix in ("prior_feature_mlp.", "action_decoder.", "obs_encoder."):
            self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before
                                if k.startswith(prefix)))
        for k in before:
            if k.startswith("prior."):
                torch.testing.assert_close(before[k], after[k], rtol=0, atol=0)

    def test_prior_artifact_coordinates_and_joint_order(self):
        prior = HandMotionPrior()
        prior.ready.fill_(True)
        payload = dict(
            format=prior.artifact_format, coordinate_space=prior.coordinate_space,
            model_config=prior.model_config, model_state_dict=prior.state_dict(),
            metadata={"hand_joint_order": prior.hand_joint_order},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.pt"
            torch.save(payload, path)
            loaded, _ = load_motion_prior(path)
            history = torch.randn(2, 8, 12)
            torch.testing.assert_close(loaded.encode_history(history), prior.encode_history(history))
            for key, value in (("format", "old"), ("coordinate_space", "normalized")):
                torch.save({**payload, key: value}, path)
                with self.assertRaisesRegex(ValueError, "raw-radians"):
                    load_motion_prior(path)
            invalid = {**payload, "metadata": {"hand_joint_order": prior.hand_joint_order[::-1]}}
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "joint order"):
                load_motion_prior(path)

    def test_runner_history_records_executed_commands_and_resets(self):
        model = self.model().eval()
        runner = BaseRunner(1, ["joint_state", "point_cloud"])
        captured = []
        handle = model.prior.encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().reshape(3, 12).clone())
        )
        self.addCleanup(handle.remove)

        class CommandEnvironment:
            def reset(self, seed):
                self.action_cnt = 0
                self.initial = float(seed)
                return self.observation(), {}

            def observation(self):
                return {"joint_state": np.full(19, self.initial, dtype=np.float32),
                        "point_cloud": np.zeros((8, 6), dtype=np.float32)}

            def step(self, prediction):
                self.action_cnt += 1
                # Simulate execution-side clipping/overrides, distinct from the prediction.
                executed = np.full(19, self.initial + self.action_cnt, dtype=np.float32)
                return self.observation(), 0, self.action_cnt == 3, False, {
                    "executed_joint_action": executed, "success": False,
                    "success_condition": False,
                }

        environment = CommandEnvironment()
        for initial in (2, 9):
            runner.run_episode(model, environment, initial, action_steps=1)
            expected = torch.full((3, 12), float(initial))
            for i, actual in enumerate(captured[-3:]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected = torch.cat((expected[1:], torch.full((1, 12), initial + i + 1.0)))
            np.testing.assert_array_equal(runner.hand_history, expected.numpy())


if __name__ == "__main__":
    unittest.main()
