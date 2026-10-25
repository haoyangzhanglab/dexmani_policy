import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader

from dexmani_policy.agents.hand_prior.hand_vae import HandVAE, load_hand_vae
from dexmani_policy.common.checkpoint import (
    TrainingCheckpoint, load_training_checkpoint, restore_policy_agent, save_training_checkpoint,
)
from dexmani_policy.common.config import (
    load_policy_config, resolve_input_recipe, validate_source_recipe,
)
from dexmani_policy.common.normalization import FieldNormalizer
from dexmani_policy.common.runtime import get_rng_state
from dexmani_policy.datasets.common.episode_split import build_split_manifest
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset
from dexmani_policy.deployment.metadata import PolicyInfo
from dexmani_policy.deployment.runtime import PolicyRuntime
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.training.builder import (
    TrainingComponents, build_model_and_ema, build_optimizer_and_scheduler,
    build_trainer, build_training_components,
)
from dexmani_policy.training.common.ema import EMA
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS, validate_hand_chunk_compatibility
from dexmani_policy.training.data import build_dataset_and_normalizer, build_normalizer, build_train_loader
from dexmani_policy.training.pretrain.hand_prior import hand_vae as pretrain
from dexmani_policy.training.resume import build_resume_contract, restore_training_state
from dexmani_policy.training.validation import validate_config, validate_policy_layout


class LatentFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.zarr_path = self.path / "data.zarr"
        root = zarr.open_group(str(self.zarr_path), mode="w")
        root.attrs.update(dt=0.0625, action_space="joint", source_id="synthetic")
        root.create_dataset("meta/episode_ends", data=np.array([8, 16]))
        time = np.arange(8, dtype=np.float32)
        self.state = np.repeat(np.r_[1 + time * 0.5, 1001 + time * 0.5,][:, None], 19, axis=1)
        self.commands = np.repeat(np.r_[3 + time * time * 0.1, 2003 + time * time * 0.1][:, None], 19, axis=1)
        cloud = np.random.default_rng(7).normal(size=(16, 8, 6)).astype(np.float32)
        for key, value in (("action", self.commands), ("joint_state", self.state), ("point_cloud", cloud)):
            root.create_dataset("data/" + key, data=value, chunks=(4, *value.shape[1:]))
        manifest = build_split_manifest(self.zarr_path, [8, 16], dict(root.attrs), val_ratio=0)
        manifest.update(train_indices=[0], val_indices=[1], exclusions=[])
        self.split = self.path / "split.json"
        self.split.write_text(json.dumps(manifest))
        self.cfg = load_policy_config("latent_flow")
        with open_dict(self.cfg):
            self.cfg.zarr_path = str(self.zarr_path)
            self.cfg.horizon = 4
            self.cfg.n_action_steps = 4
            self.cfg.codec.path = str(self.path / "codec.pt")
            self.cfg.codec.hidden_dim = 16
            self.cfg.codec.n_layers = 1
            self.cfg.codec.n_heads = 4
            self.cfg.agent.num_points = 8
            self.cfg.agent.state_out_dim = 8
            self.cfg.agent.pc_encoder_config = dict(channels=24, num_patches=2, group_size=4, depth=1, num_heads=4)
            self.cfg.agent.fps_random_config = dict(use_random=False)
            self.cfg.agent.hidden_dim = 32
            self.cfg.agent.n_layers = 1
            self.cfg.agent.n_head = 4
            self.cfg.agent.p_drop_attn = 0
            self.cfg.dataset.split_manifest = str(self.split)
            self.cfg.dataset.val_ratio = 0
            self.cfg.dataset.max_train_episodes = None
            self.cfg.dataset.augmentation_cfg = None
            self.cfg.dataloader.batch_size = 2
            self.cfg.dataloader.num_workers = 0
            self.cfg.training.use_compile = False
            self.cfg.training.use_bfloat16 = False
            self.cfg.training.device = "cpu"
            self.cfg.training.lr_warmup_steps = 2
            self.cfg.training.loop.total_train_steps = 6
            self.cfg.workspace.output_dir = str(self.path / "run")
            self.cfg.workspace.wandb_cfg = None
        OmegaConf.resolve(self.cfg)

    def hand_dataset(self):
        return HandChunkDataset(
            str(self.zarr_path), data_semantics=HAND_ACTION_SEMANTICS, horizon=4,
            split_manifest=str(self.split), val_ratio=0,
        )

    def prepare_model(self):
        dataset, normalizer = build_dataset_and_normalizer(self.cfg, verbose=False)
        codec = HandVAE(horizon=4, hidden_dim=16, n_layers=1)
        codec.ready.fill_(True)
        codec.action_mean.fill_(0.7)
        codec.action_std.fill_(1.3)
        metadata = dict(
            model_config=codec.model_config, hand_data=dataset.hand_data,
            loss_reduction=dict(kl="mean_over_batch_time_latent"),
        )
        torch.save(dict(model_config=codec.model_config, state_dict=codec.state_dict(), metadata=metadata), self.cfg.codec.path)
        model, ema, updater = build_model_and_ema(self.cfg, "cpu", normalizer)
        return dataset, model, ema, updater

    def test_policy_dataset_pairing_and_mixed_action_origins(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.n_action_steps = cfg.agent.n_action_steps = 3
        cfg.agent._target_ = "dexmani_policy.agents.policies.dp3.DP3Policy"
        with self.assertRaisesRegex(ValueError, "must be used together"):
            validate_policy_layout(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg.dataset._target_ = "dexmani_policy.datasets.demonstration_dataset.DemonstrationDataset"
        with self.assertRaisesRegex(ValueError, "must be used together"):
            validate_policy_layout(cfg)
        future = hydra.utils.instantiate(self.cfg.dataset)
        historical = DemonstrationDataset(
            str(self.zarr_path), sensor_modalities=["joint_state", "point_cloud"],
            horizon=4, obs_horizon=2, pc_dim=6,
            split_manifest=str(self.split), val_ratio=0,
        )
        with self.assertRaisesRegex(ValueError, "time origin"):
            MultiTaskDataset([historical, future], ["historical", "future"])

    def test_timeline_boundaries_augmentation_and_full_execution(self):
        validate_config(self.cfg)
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.assertEqual(len(dataset), 5)
        for ds, base in ((dataset, 0), (dataset.get_validation_dataset(), 8)):
            for i in range(len(ds)):
                sample = ds[i]
                expected_rows = [max(base, base + i - 1), base + i]
                np.testing.assert_array_equal(sample["obs"]["joint_state"], self.state[expected_rows])
                np.testing.assert_array_equal(sample["action_context"]["reference_state"], self.state[base + i])
                np.testing.assert_array_equal(sample["action"], self.commands[base + i:base + i + 4])
        before = dataset[1]
        class AddNoise:
            prob = 1
            def augment(self, values):
                values += 100
        dataset.augmentors["joint_state"] = [AddNoise()]
        after = dataset[1]
        torch.testing.assert_close(before["action"], after["action"])
        torch.testing.assert_close(before["action_context"], after["action_context"])
        self.assertFalse(torch.equal(before["obs"]["joint_state"], after["obs"]["joint_state"]))
        np.testing.assert_array_equal(dataset.replay_buffer.read("joint_state", slice(0, 2)), self.state[:2])
        _, model, _, _ = self.prepare_model()
        sequence = torch.arange(76).reshape(1, 4, 19).float()
        torch.testing.assert_close(model.format_action_output(sequence)["future_actions"], sequence)
        with patch.object(model, "predict_action", return_value={"future_actions": sequence}):
            runner = SimRunner.__new__(SimRunner)
            np.testing.assert_array_equal(runner.predict_action_chunk({}, model, action_steps=4), sequence[0])

    def test_consumed_rows_relative_statistics_and_hand_only_eligibility(self):
        root = zarr.open_group(str(self.zarr_path), mode="a")
        root["data/action"][0] = np.nan
        root["data/joint_state"][7] = 1e8
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.assertEqual(len(dataset), 4)  # t=1 remains valid although its historical action is NaN.
        np.testing.assert_array_equal(dataset[0]["action"], self.commands[1:5])
        np.testing.assert_array_equal(dataset.sampler.observation_source_rows, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(dataset.sampler.action_source_rows, [1, 2, 3, 4, 5, 6, 7])
        normalizer = build_normalizer(dataset, {"joint_state": "gaussian", "action": "gaussian"}, "action")
        self.assertLess(float(normalizer["joint_state"].input_stats["max"].max()), 4)
        # Explicit first-coordinate targets: t=1 has q0=1.5; t=2 has q0=2.
        relative = np.concatenate(list(dataset.iter_normalization_data("action")))
        np.testing.assert_allclose(relative[:4, 0], [1.6, 1.9, 2.4, 3.1], atol=1e-6)
        np.testing.assert_allclose(relative[4:8, 0], [1.4, 1.9, 2.6, 3.5], atol=1e-6)
        self.assertNotEqual(relative[1, 0], relative[4, 0])
        self.assertLess(float(normalizer["action"].input_stats["max"].max()), 10)
        self.assertAlmostEqual(float(normalizer["action"].input_stats["mean"][0]), 2.6, places=5)
        with self.assertRaisesRegex(ValueError, "reuse"):
            list(dataset.get_validation_dataset().iter_normalization_data("action"))
        ordinary = DemonstrationDataset(
            str(self.zarr_path), sensor_modalities=["joint_state"], horizon=4, obs_horizon=2,
            split_manifest=str(self.split),
        )
        plain = np.concatenate(list(ordinary.iter_normalization_data("joint_state")))
        self.assertLess(plain.max(), 10)
        del root["data/point_cloud"]
        root["data/action"][0, :7] = np.nan
        root["data/action"][0, 7:] = self.commands[0, 7:]
        root["data/joint_state"][0, :7] = np.nan
        hand = self.hand_dataset()
        self.assertEqual(len(hand), 5)
        self.assertEqual(set(hand.replay_buffer.keys()), {"action", "joint_state"})
        mean, _, statistics = pretrain.fit_relative_statistics(hand)
        self.assertTrue(torch.isfinite(mean).all())
        self.assertAlmostEqual(float(mean[0]), 2.55, places=5)
        self.assertEqual(statistics["frames"], 20)

    def test_coordinates_and_decode_before_slice(self):
        dataset, model, _, _ = self.prepare_model()
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        model.eval()
        model.normalizer["action"] = FieldNormalizer.from_parameters(torch.full((19,), 2.), torch.full((19,), 3.))
        reference = batch["action_context"]["reference_state"]
        native = torch.ones(2, 4, 16)
        native[..., :7] = 7  # inverse affine gives relative arm=2.
        model.codec.decoder_output.weight.data.zero_()
        model.codec.decoder_output.bias.data.fill_(2)
        model.n_action_steps = 2
        with patch.object(model.action_decoder, "predict_action", return_value=native), patch.object(model.codec, "decode", wraps=model.codec.decode) as decode:
            result = model.predict_action(batch["obs"], action_context=batch["action_context"])
        torch.testing.assert_close(result["future_actions"][..., :7], reference[:, None, :7].expand(-1, 4, -1) + 2)
        torch.testing.assert_close(result["future_actions"][..., 7:], reference[:, None, 7:].expand(-1, 4, -1) + 3.3)
        self.assertEqual(decode.call_args.args[0].shape[1], 4)
        self.assertEqual(result["future_actions"].shape, (2, 4, 19))
        torch.testing.assert_close(model.resolve_reference_state(batch["obs"], None), reference)
        # An exact velocity oracle must give zero arm/latent FM and decode the true target estimate.
        model.arm_loss_weight, model.latent_loss_weight, model.decoded_loss_weight = 1, 1, 1
        mu, _ = model.codec.encode(batch["action"][..., 7:], reference[:, 7:])
        target = torch.cat(((batch["action"][..., :7] - reference[:, None, :7]) * 2 + 3, mu), -1)
        noise = torch.full_like(target, -0.4)
        def oracle(x, timestep, context, **kwargs):
            return target - noise
        with patch('torch.randn_like', return_value=noise), patch.object(model.action_decoder.model, "forward", side_effect=oracle), patch.object(model.codec, "decode", wraps=model.codec.decode) as decode:
            _, metrics = model.compute_loss(batch)
        self.assertEqual(float(metrics["loss_arm"]), 0)
        self.assertEqual(float(metrics["loss_latent"]), 0)
        torch.testing.assert_close(decode.call_args.args[0], mu, rtol=1e-5, atol=1e-6)
        # Oracle decoder is deliberately wrong: supervision must remain the demonstration.
        self.assertGreater(float(metrics["loss_decoded_hand"]), 0)
        self.assertIn("one_step_denoising_hand_mae", metrics)
        condition = model.encode_condition(batch["obs"])
        with patch('torch.randn_like', return_value=noise), patch.object(model.action_decoder.model, "forward", side_effect=oracle):
            sampled = model.action_decoder.predict_action(condition, target, inference_steps=10)
        torch.testing.assert_close(sampled, target, rtol=1e-5, atol=1e-6)

    def test_decoded_only_gradients_and_zero_weight_skip(self):
        dataset, model, _, _ = self.prepare_model()
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        model.arm_loss_weight = model.latent_loss_weight = 0
        model.train()
        self.assertFalse(model.codec.training)
        loss, _ = model.compute_loss(batch)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.action_decoder.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.codec.parameters()))
        model.decoded_loss_weight = 0
        model.arm_loss_weight = 1
        with patch.object(model.codec, "decode", side_effect=AssertionError("decoder executed")):
            loss, metrics = model.compute_loss(batch, log_metrics=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("loss_decoded_hand", metrics)

    def test_codec_motion_kl_and_heldout_selection(self):
        codec = HandVAE(horizon=4, hidden_dim=16, n_layers=1)
        codec.joint_weights[8:] = 0.5
        target = torch.arange(4).float()[None, :, None].expand(1, 4, 12)
        shifted = codec.reconstruction_terms(target + 1, target)
        self.assertAlmostEqual(float(shifted["position"]), 5 / 12, places=6)
        self.assertEqual(float(shifted["velocity"]), 0)
        self.assertEqual(float(shifted["acceleration"]), 0)
        constant = codec.reconstruction_terms(torch.zeros_like(target), target)
        self.assertGreater(float(constant["velocity"]), 0)
        self.assertEqual(float(constant["acceleration"]), 0)
        quadratic = codec.reconstruction_terms(torch.zeros_like(target), target.square())
        self.assertAlmostEqual(float(quadratic["acceleration"]), 1.25, places=6)
        codec.decoder_output.weight.data.zero_()
        codec.decoder_output.bias.data.zero_()
        metrics = pretrain.evaluate(codec, self.hand_dataset().get_validation_dataset(), 2)
        self.assertAlmostEqual(metrics["physical_mae"], 1002.55, places=4)
        for step, expected in ((0, 0), (4000, 0), (4001, 3e-4/4000), (6000, 1.5e-4), (8000, 3e-4), (30000, 3e-4)):
            self.assertAlmostEqual(pretrain.kl_coefficient(step, 3e-4, 4000, 4000), expected)
        cfg = load_policy_config("hand_vae", [
            f"zarr_path={self.zarr_path}", f"codec.path={self.path / 'trained_codec.pt'}",
            "horizon=4", f"dataset.split_manifest={self.split}", "dataset.val_ratio=0",
            "codec.hidden_dim=16", "codec.n_layers=1", "training.device=cpu",
            "training.batch_size=2", "training.steps=3", "training.kl_warmup=1",
            "training.kl_ramp=1", "training.eval_every=1",
        ])
        real_evaluate = pretrain.evaluate
        evaluated_parameters = []
        def ranked_evaluation(model, dataset, batch_size):
            self.assertTrue(dataset.is_validation)
            value = real_evaluate(model, dataset, batch_size)
            evaluated_parameters.append({
                name: p.detach().clone() for name, p in model.named_parameters()
            })
            value["physical_mae"] = float(len(evaluated_parameters))
            return value
        with patch.object(pretrain, "evaluate", side_effect=ranked_evaluation):
            payload = pretrain.train(cfg)
        self.assertEqual(payload["metadata"]["selection"]["step"], 1)
        self.assertEqual(payload["metadata"]["selection"]["beta"], 0)
        self.assertEqual(payload["metadata"]["completed_updates"], 3)
        self.assertEqual(len(evaluated_parameters), 3)
        disk = torch.load(cfg.codec.path, map_location="cpu", weights_only=False)
        self.assertEqual(disk["metadata"], payload["metadata"])
        self.assertEqual(disk["metrics"], payload["metrics"])
        torch.testing.assert_close(disk["state_dict"], payload["state_dict"], rtol=0, atol=0)
        loaded, _ = load_hand_vae(cfg.codec.path)
        torch.testing.assert_close(dict(loaded.named_parameters()), evaluated_parameters[0], rtol=0, atol=0)
        self.assertTrue(any(
            not torch.equal(evaluated_parameters[0][name], parameter)
            for name, parameter in evaluated_parameters[-1].items()
        ))
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertEqual(len(payload["metrics"]["physical_joint_mae"]), 12)
        cfg.dataset.split_manifest = None
        with self.assertRaisesRegex(ValueError, "held-out"):
            pretrain.build_codec_dataset(cfg)

    def test_fresh_compatibility_and_fixed_ema(self):
        data = self.hand_dataset().hand_data
        other = copy.deepcopy(data)
        other["provenance"].update(source_id="other", episode_ends=[100, 200, 300], train_episodes=[0, 1], val_episodes=[2])
        validate_hand_chunk_compatibility(data, other)
        for key, bad in (("dt", 0.1), ("units", "degrees"), ("horizon", 5), ("reference", "previous_state")):
            changed = copy.deepcopy(other)
            changed["contract"][key] = bad
            with self.assertRaises(ValueError):
                validate_hand_chunk_compatibility(data, changed)
        bad = copy.deepcopy(data)
        bad["provenance"].update(train_episodes=[1], val_episodes=[0])
        with self.assertRaisesRegex(ValueError, "split|leakage"):
            validate_hand_chunk_compatibility(data, bad)
        raw = nn.Linear(1, 1, bias=False)
        raw.weight.data.zero_()
        averaged = copy.deepcopy(raw)
        updater = EMA(averaged, fixed_decay=0.99)
        raw.weight.data.fill_(10)
        updater.step(raw)
        self.assertAlmostEqual(float(averaged.weight), 0.1, places=6)
        updater.step(raw)
        self.assertAlmostEqual(float(averaged.weight), 0.199, places=6)
        self.assertEqual(EMA(copy.deepcopy(raw)).get_decay(0), 0)

    def test_checkpoint_without_codec_inference_and_metadata(self):
        dataset, model, ema, updater = self.prepare_model()
        optimizer, scheduler = build_optimizer_and_scheduler(self.cfg, model, verbose=False)
        loader = build_train_loader(self.cfg, dataset)
        batch = next(iter(loader))
        model.train()
        loss, _ = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updater.step(model)
        saved = OmegaConf.to_container(self.cfg, resolve=True)
        checkpoint = TrainingCheckpoint(
            epoch=0, global_step=1, next_micro_step=1,
            model_state=model.state_dict(), ema_model_state=ema.state_dict(),
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
            resume_contract=build_resume_contract(loader), ema_updater_step=updater.optimization_step,
            rng_states=[get_rng_state()], effective_config=saved,
        )
        path = save_training_checkpoint(self.path / "policy.pt", checkpoint)
        Path(self.cfg.codec.path).unlink()
        for use_ema, expected in ((False, model), (True, ema)):
            restored = restore_policy_agent(saved, path, use_ema=use_ema, device="cpu")
            expected.eval()
            torch.manual_seed(100)
            prediction = expected.predict_action(batch["obs"])
            torch.manual_seed(100)
            torch.testing.assert_close(restored.predict_action(batch["obs"]), prediction, rtol=0, atol=0)
        info = PolicyInfo(self.path, path, self.cfg, "ema", 10)
        self.assertEqual(info.future_steps, 4)
        self.assertEqual(info.action_start_index, 0)
        runtime = PolicyRuntime(restored, info, device="cpu", seed=0)
        observation = {key: value[0].numpy() for key, value in batch["obs"].items()}
        reference = batch["action_context"]["reference_state"][:1]
        torch.manual_seed(51)
        expected = restored.predict_action(
            {key: value[:1] for key, value in batch["obs"].items()},
            action_context={"reference_state": reference},
        )["future_actions"][0].numpy()
        torch.manual_seed(51)
        actual = runtime.predict(observation, action_context={"reference_state": reference[0].numpy()})
        np.testing.assert_array_equal(actual, expected)
        other = copy.deepcopy(self.cfg)
        other.agent._target_ = "dexmani_policy.agents.policies.dp3.DP3Policy"
        self.assertEqual(PolicyInfo(self.path, path, other, "raw", 10).future_steps, 4)

    def test_relocated_dataset_resume_next_update(self):
        base_cfg = copy.deepcopy(self.cfg)
        for stable_source in (True, False):
            with self.subTest(stable_source=stable_source):
                self.cfg = copy.deepcopy(base_cfg)
                self.cfg.dataset.split_manifest = None
                self.cfg.dataset.val_ratio = 0.5
                self.cfg.training.loop.total_train_steps = 5
                self.cfg.workspace.output_dir = str(self.path / f"continuous_{stable_source}")
                if not stable_source:
                    del zarr.open_group(str(self.zarr_path), mode="a").attrs["source_id"]
                dataset, model, ema, updater = self.prepare_model()
                optimizer, scheduler = build_optimizer_and_scheduler(self.cfg, model, verbose=False)
                components = TrainingComponents(
                    device=torch.device("cpu"), model=model, ema_model=ema, ema_updater=updater,
                    optimizer=optimizer, scheduler=scheduler,
                    train_loader=build_train_loader(self.cfg, dataset),
                    workspace=hydra.utils.instantiate(self.cfg.workspace),
                )
                self.addCleanup(components.close)
                trainer = build_trainer(self.cfg, components)
                trainer.train(max_updates=2)
                continuous_rng = get_rng_state()
                checkpoint_path = next(components.workspace.checkpoint_dir.glob("*-step=00000001-*.pt"))
                checkpoint = load_training_checkpoint(checkpoint_path)
                Path(self.cfg.codec.path).unlink()
                source_config = copy.deepcopy(checkpoint.effective_config)
                self.assertEqual(
                    (checkpoint.global_step, checkpoint.epoch, checkpoint.next_micro_step), (1, 0, 1)
                )

                moved_path = self.path / f"moved_{stable_source}.zarr"
                shutil.copytree(self.zarr_path, moved_path)
                invocation = copy.deepcopy(self.cfg)
                with open_dict(invocation):
                    invocation.resume_from = str(checkpoint_path)
                cfg = resolve_input_recipe(
                    invocation, overrides=[
                        f"dataset.zarr_path={moved_path}",
                        f"workspace.output_dir={self.path / f'resumed_{stable_source}'}",
                    ], resume_checkpoint=checkpoint,
                )
                if not stable_source:
                    with self.assertRaisesRegex(hydra.errors.InstantiationException, "manifest data path/episode identity"):
                        build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint, verbose=False)
                    continue
                ds, normalizer = build_dataset_and_normalizer(
                    cfg, resume_checkpoint=checkpoint, verbose=False
                )
                self.assertIsNone(normalizer)
                resumed = build_training_components(cfg, "cpu", ds, normalizer, checkpoint=checkpoint)
                self.addCleanup(resumed.close)
                self.assertIs(resumed.train_loader.dataset, ds)
                self.assertEqual(cfg.hand_data.provenance.zarr_path, str(moved_path))
                self.assertEqual(checkpoint.effective_config, source_config)
                old_source = source_config["hand_data"]["provenance"]["source_id"]
                self.assertEqual(cfg.hand_data.provenance.source_id, old_source)
                state = restore_training_state(checkpoint, resumed, effective_config=cfg)
                self.assertEqual(state, (1, 0, 1))
                resumed_trainer = build_trainer(cfg, resumed)
                # Only restore_training_state restores the process RNG for this branch.
                resumed_trainer.train(resume_state=state, max_updates=1)
                resumed_rng = get_rng_state()
                for actual, expected in (
                    (resumed.model.state_dict(), model.state_dict()),
                    (resumed.ema_model.state_dict(), ema.state_dict()),
                    (resumed.optimizer.state_dict()["state"], optimizer.state_dict()["state"]),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(
                    resumed.optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"]
                )
                self.assertEqual(resumed.scheduler.state_dict(), scheduler.state_dict())
                self.assertEqual(resumed.ema_updater.optimization_step, updater.optimization_step)
                for loop in (trainer, resumed_trainer):
                    self.assertEqual((loop.global_step, loop.current_epoch, loop.next_micro_step), (2, 1, 0))
                self.assertEqual(resumed_rng["python"], continuous_rng["python"])
                np.testing.assert_equal(resumed_rng["numpy"], continuous_rng["numpy"])
                torch.testing.assert_close(resumed_rng["torch"], continuous_rng["torch"], rtol=0, atol=0)
                self.assertEqual(resumed_rng["torch_cuda"], continuous_rng["torch_cuda"])
                self.assertEqual(load_training_checkpoint(checkpoint_path).effective_config, source_config)

    def test_recipe_location_exclusion_is_narrow(self):
        self.prepare_model()
        saved = OmegaConf.to_container(self.cfg, resolve=True)
        current = copy.deepcopy(saved)
        current["dataset"]["zarr_path"] = "/relocated/data.zarr"
        current["hand_data"]["provenance"]["zarr_path"] = "/relocated/data.zarr"
        before = copy.deepcopy(current)
        validate_source_recipe(saved, current)
        self.assertEqual(current, before)
        self.assertEqual(saved, OmegaConf.to_container(self.cfg, resolve=True))
        for key, value in (
            ("hand_data.provenance.source_id", "different"),
            ("hand_data.provenance.episode_ids", ["first", "second"]),
            ("hand_data.provenance.episode_ends", [7, 16]),
            ("hand_data.provenance.train_episodes", [1]),
            ("hand_data.provenance.val_episodes", [0]),
            ("hand_data.train_windows", 4),
            ("hand_data.val_windows", 4),
            ("hand_data.contract.units", "degrees"),
            ("hand_data.contract.dt", 0.1),
            ("agent.codec_metadata.hand_data.provenance.zarr_path", "/changed/codec_source.zarr"),
            ("data_recipe.0.split_manifest.zarr_path", "/changed/manifest_source.zarr"),
            ("training.loop.total_train_steps", 7),
        ):
            with self.subTest(field=key):
                changed = OmegaConf.create(current)
                OmegaConf.update(changed, key, value, merge=False)
                with self.assertRaisesRegex(ValueError, "recipe mismatch"):
                    validate_source_recipe(saved, changed)


if __name__ == "__main__":
    unittest.main()
