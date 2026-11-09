"""XHand Real 尺度契约、辅助力损失与 CPU 策略 smoke tests。"""

import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.nn import functional as F
from torch.utils.data import default_collate

from dexmani_policy.agents.obs_encoder.interaction.future_tactile import (
    ForceDiffusion,
    ForceTransformer,
)
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandPatchEncoder
from dexmani_policy.agents.policies.interact_dex import InterActDex
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.deployment import load_policy
from dexmani_policy.test_tools.smoke_test import make_source, run_smoke
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import build_dataset, fit_normalizer


def small_interact(**kwargs):
    return InterActDex(
        horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19, future_force_horizon=2,
        num_points=8, pc_encoder_config={"num_patches": 4, "group_size": 2, "depth": 1},
        hidden_dim=48, n_head=4, n_layers=1, timestep_embed_dim=16, p_drop_attn=0,
        **kwargs,
    )


class EpisodeStoreScaleTest(unittest.TestCase):
    def test_real_requires_scale_for_each_selected_tactile_field(self):
        with tempfile.TemporaryDirectory() as directory:
            source = zarr.open_group(directory, mode="w")
            source.create_dataset("meta/episode_ends", data=np.array([2], dtype=np.int64))
            source.create_dataset("data/joint_state", data=np.zeros((2, 19), dtype=np.float32))
            for key, shape in (("finger_tactile_force", (5, 3)), ("taxel_tactile_force", (5, 120, 3))):
                source.create_dataset(f"data/{key}", data=np.full((2, *shape), 0.5, dtype=np.float32))
            source.attrs["format"] = "dexmani.real.canonical"
            for scale in (None, 1.0, 254.999, "255.0", 255.0):
                if scale is not None:
                    source.attrs["tactile_scale_divisor"] = scale
                for keys in (("finger_tactile_force",), ("taxel_tactile_force",), None):
                    with self.subTest(scale=scale, keys=keys):
                        if scale == 255.0:
                            store = EpisodeStore(directory, keys=keys)
                            for field in set(store.keys()) - {"joint_state"}:
                                np.testing.assert_array_equal(store.read(field, slice(None)), 0.5)
                        else:
                            with self.assertRaisesRegex(ValueError, "tactile_scale_divisor"):
                                EpisodeStore(directory, keys=keys)
                # 只选非触觉字段时，即使磁盘上另有触觉，也无需读取尺度属性。
                EpisodeStore(directory, keys=("joint_state",))
            del source.attrs["tactile_scale_divisor"]
            source.attrs["format"] = "simulation"
            EpisodeStore(directory)
            source.attrs["tactile_scale_divisor"] = 50.0
            EpisodeStore(directory)
            del source.attrs["format"]
            EpisodeStore(directory)
            del source["data/finger_tactile_force"]
            del source["data/taxel_tactile_force"]
            del source.attrs["tactile_scale_divisor"]
            source.attrs["format"] = "dexmani.real.canonical"
            EpisodeStore(directory)


class PatchScalingTest(unittest.TestCase):
    def test_descriptor_scale_covariance(self):
        generator = torch.Generator().manual_seed(7)
        native = torch.randn(2, 5, 120, 3, generator=generator) * 127.5
        native[0, 0] = 0
        native[1, :, ::3] = 0
        original = XHandPatchEncoder(activity_scale=1.0)
        scaled = XHandPatchEncoder()
        self.assertAlmostEqual(scaled.activity_scale, 1 / 255, delta=1e-9)
        expected = original._patch_descriptors(native)
        actual = scaled._patch_descriptors(native / 255)
        # FP32 散点归约允许舍入误差；Activity Fraction 无物理数值变化。
        torch.testing.assert_close(actual[..., 6], expected[..., 6], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual[..., :6], expected[..., :6] / 255, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(actual[..., 7], expected[..., 7] / 255, rtol=1e-6, atol=1e-7)
        output = scaled(native / 255)
        self.assertEqual(output.shape, (2, 5, 64))
        self.assertTrue(torch.isfinite(output).all())


class InterActLossTest(unittest.TestCase):
    def test_beta_config_validation_and_effective_config(self):
        default = small_interact()
        self.assertEqual(default.future_force_loss_beta, 1.0)
        cfg = load_policy_config("interact_dex")
        self.assertEqual(cfg.agent.future_force_loss_beta, 1 / 255)
        self.assertEqual(cfg.agent.future_force_loss_weight, 0.1)
        model = small_interact(future_force_loss_beta=cfg.agent.future_force_loss_beta)
        self.assertEqual(model.effective_config["future_force_loss_beta"], 1 / 255)
        self.assertEqual(model.effective_config["future_force_loss_weight"], 0.1)
        for beta in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "future_force_loss_beta"):
                small_interact(future_force_loss_beta=beta)

    def test_loss_gradient_metrics_and_invalid_mask_in_scaled_space(self):
        model = small_interact(future_force_loss_beta=1 / 255)
        target = torch.full((3, 5, 3), 127.5)
        errors = torch.tensor([-2., -1., -0.5, 0., 0.5, 1., 2., 255., -510.]).repeat(5).reshape(3, 5, 3)
        native_prediction = (target + errors).requires_grad_()
        prediction = (native_prediction.detach() / 255).requires_grad_()
        valid = torch.tensor([True, False, True])
        batch = {"future_force": (target / 255).double(), "future_force_valid": valid}
        loss, metrics = model._compute_future_force_loss(prediction, batch, log_metrics=True)
        unlogged, no_metrics = model._compute_future_force_loss(prediction, batch, log_metrics=False)
        torch.testing.assert_close(unlogged, loss, rtol=0, atol=0)
        self.assertEqual(no_metrics, {})
        native_loss = F.smooth_l1_loss(native_prediction, target, reduction="none", beta=1.0).mean((-1, -2))[valid].mean()
        torch.testing.assert_close(loss, native_loss / 255, rtol=1e-6, atol=1e-8)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        native_loss.backward()
        torch.testing.assert_close(prediction.grad, native_prediction.grad, rtol=1e-5, atol=1e-7)
        mae = (prediction.detach() - batch["future_force"].float()).abs().mean((-1, -2))[valid].mean()
        torch.testing.assert_close(metrics["future_force_mae_scaled"], mae)
        torch.testing.assert_close(metrics["future_force_mae_sdk"], mae * 255)
        self.assertNotIn("future_force_mae_raw", metrics)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        batch["future_force_valid"] = torch.zeros_like(valid)
        prediction = prediction.detach().requires_grad_()
        empty_loss, empty_metrics = model._compute_future_force_loss(prediction, batch, log_metrics=True)
        empty_loss.backward()
        self.assertEqual(empty_loss.item(), 0)
        self.assertTrue((prediction.grad == 0).all())
        self.assertTrue(all(value.item() == 0 for value in empty_metrics.values()))


class ForceDiffusionTest(unittest.TestCase):
    def test_standard_noise_sample_prediction_and_mse(self):
        diffusion = ForceDiffusion(ForceTransformer(3, 8, 15, depth=1, embed_dim=16, dropout=0), 8, 2)
        scheduler = diffusion.noise_scheduler
        self.assertIsInstance(scheduler, DDIMScheduler)
        self.assertEqual(scheduler.config.prediction_type, "sample")
        self.assertEqual(scheduler.config.beta_schedule, "squaredcos_cap_v2")
        self.assertEqual(scheduler.config.beta_start, 0.0001)
        self.assertEqual(scheduler.config.beta_end, 0.02)
        self.assertFalse(scheduler.config.clip_sample)
        force = torch.arange(90, dtype=torch.float32).reshape(2, 3, 15) / 255
        condition = torch.ones(2, 1, 8)
        valid = torch.tensor([[True, True, False], [True, False, False]])
        torch.manual_seed(23)
        expected_noise = torch.randn_like(force)
        expected_time = torch.randint(8, (2,))
        outputs = []
        hook = diffusion.model.register_forward_hook(lambda _module, _args, output: outputs.append(output))
        try:
            torch.manual_seed(23)
            with patch.object(scheduler, "add_noise", wraps=scheduler.add_noise) as add_noise:
                loss, prediction = diffusion.compute_loss_and_prediction(condition, force, valid)
            clean, noise, timestep = add_noise.call_args.args
            torch.testing.assert_close(clean, force, rtol=0, atol=0)
            torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)
            torch.testing.assert_close(timestep, expected_time, rtol=0, atol=0)
            torch.testing.assert_close(prediction, outputs[0], rtol=0, atol=0)
            torch.testing.assert_close(loss, (prediction - force).square()[valid].mean())
            loss.backward()
            self.assertTrue(torch.isfinite(diffusion.model.head.weight.grad).all())
        finally:
            hook.remove()


class TactilePolicySmokeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="xhand-policy-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "source.zarr"
        self.source = make_source(self.path)
        self.source.attrs.update(format="dexmani.real.canonical", tactile_scale_divisor=255.0, dt=0.05)
        for key, shape in (("finger_tactile_force", (5, 3)), ("taxel_tactile_force", (5, 120, 3))):
            native = np.arange(12 * np.prod(shape), dtype=np.float32).reshape(12, *shape) % 1021 - 510
            self.source.create_dataset(f"data/{key}", data=native / np.float32(255))
        self.source.create_dataset("data/eef_pose", data=np.tile(
            np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32), (12, 1),
        ))
        self.source.create_dataset("data/fingertip_xyz", data=np.zeros((12, 5, 3), dtype=np.float32))

    def config(self, policy, tactile="force"):
        overrides = [
            f"zarr_path={self.path}", "training.device=cpu", "horizon=4", "n_obs_steps=2", "n_action_steps=2",
            "agent.num_points=8", "agent.pc_encoder_config.num_patches=4", "agent.pc_encoder_config.group_size=2",
            "agent.pc_encoder_config.depth=1", "agent.num_inference_steps=2", "dataset.augmentation_cfg=null",
            "training.use_compile=false", "training.use_bfloat16=false", "training.lr_warmup_steps=0",
            "training.loop.total_train_steps=2", "training.loop.log_interval_steps=1",
            "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
        ]
        if policy == "interact_dex":
            overrides += ["agent.hidden_dim=48", "agent.n_head=4", "agent.n_layers=1", "agent.timestep_embed_dim=16"]
        else:
            overrides += [
                f"tactile={tactile}", "agent.feature_dim=32", "agent.state_out_dim=8",
                "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16", "agent.num_training_steps=8",
                "agent.force_num_inference_steps=2", "agent.force_predictor_config.depth=1",
                "agent.force_predictor_config.embed_dim=16", "agent.force_predictor_config.dropout=0",
            ]
        return load_policy_config(policy, overrides)

    def prepare(self, cfg):
        dataset, description = build_dataset(cfg)
        model = hydra.utils.instantiate(cfg.agent, _convert_="all")
        dims = description["field_dims"]
        model.configure_normalizer(dims, normalization_feature_shapes(dims, cfg.normalization))
        fit_normalizer(dataset, cfg.normalization, model.normalizer)
        field = next(key for key in dataset.sensor_modalities if "tactile_force" in key)
        self.assertEqual(cfg.normalization[field], "identity")
        self.assertNotIn(field, model.normalizer.fields)
        # 检查边界填充与真实未来标签均直接使用 Zarr 数值。
        for index in (0, len(dataset) - 1):
            sample = dataset[index]
            rows = dataset.sampler.sample_source_rows(index)
            torch.testing.assert_close(sample["obs"][field],
                                       torch.from_numpy(self.source[f"data/{field}"][:][rows[:2]]), rtol=0, atol=0)
            query = rows[dataset.action_start_index]
            end = 6 if query < 6 else 12
            offsets = np.array([cfg.future_force_horizon]) if cfg.policy_name == "interact-dex" else np.arange(1, 4)
            targets = query + offsets
            values = self.source[f"data/{field}"][:][np.minimum(targets, end - 1)]
            expected_valid = targets < end
            if cfg.policy_name == "interact-dex":
                values, expected_valid = values[0], expected_valid[0]
            torch.testing.assert_close(sample["future_force"], torch.as_tensor(values), rtol=0, atol=0)
            torch.testing.assert_close(sample["future_force_valid"], torch.from_numpy(np.asarray(expected_valid)))
        return model, default_collate([dataset[0], dataset[1]]), field

    def test_interact_forward_keeps_head_and_labels_in_scaled_space(self):
        model, batch, field = self.prepare(self.config("interact_dex"))
        seen = []
        hook = model.future_force_head.register_forward_hook(lambda _module, _args, output: seen.append(output))
        try:
            with patch.object(model, "_compute_future_force_loss", wraps=model._compute_future_force_loss) as loss_call:
                loss, metrics = model(batch)
            torch.testing.assert_close(loss_call.call_args.args[0], seen[0], rtol=0, atol=0)
            self.assertIs(loss_call.call_args.args[1]["future_force"], batch["future_force"])
            torch.testing.assert_close(model._preprocess_obs(batch["obs"])[field], batch["obs"][field].flatten(0, 1))
            torch.testing.assert_close(metrics["loss_future_force_weighted"], 0.1 * metrics["loss_future_force"])
            torch.testing.assert_close(loss, metrics["loss_action"] + metrics["loss_future_force_weighted"])
            loss.backward()
            self.assertTrue(torch.isfinite(model.future_force_head.mlp[-1].weight.grad).all())
        finally:
            hook.remove()

    def test_adaptac_forward_loss_fusion_and_sampling(self):
        for tactile in ("force", "taxel", "patch"):
            with self.subTest(tactile=tactile):
                cfg = self.config("adaptac", tactile)
                model, batch, field = self.prepare(cfg)
                self.assertEqual(model.force_loss_weight, 1.0)
                if tactile == "patch":
                    self.assertEqual(cfg.agent.tactile_encoder.activity_scale, 1 / 255)
                predictions: list[torch.Tensor] = []
                guides: list[torch.Tensor] = []
                prediction_hook = model.force_predictor.model.register_forward_hook(
                    lambda _module, _args, output, outputs=predictions: outputs.append(output))
                fusion_hook = model.force_fusion.register_forward_pre_hook(
                    lambda _module, args, inputs=guides: inputs.append(args[2]))
                try:
                    predictor = model.force_predictor
                    with patch.object(predictor, "compute_loss_and_prediction", wraps=predictor.compute_loss_and_prediction) as call:
                        loss, metrics = model(batch)
                    torch.testing.assert_close(call.call_args.args[1], batch["future_force"].flatten(2), rtol=0, atol=0)
                    guide = guides[0].reshape(2, 5, model.force_dim)
                    torch.testing.assert_close(guide[:, :2], batch["obs"][field].reshape(2, 2, -1), rtol=0, atol=0)
                    torch.testing.assert_close(guide[:, 2:], predictions[0], rtol=0, atol=0)
                    torch.testing.assert_close(metrics["loss_force_weighted"], metrics["loss_force"])
                    torch.testing.assert_close(loss, metrics["loss_action"] + metrics["loss_force_weighted"])
                    loss.backward()
                    self.assertTrue(torch.isfinite(predictor.model.head.weight.grad).all())
                    model.eval()
                    actions = model.predict_action(batch["obs"])["future_actions"]
                    self.assertEqual(actions.shape, (2, 3, 19))
                    self.assertTrue(torch.isfinite(actions).all())
                finally:
                    prediction_hook.remove()
                    fusion_hook.remove()

    def test_training_resume_and_deployment_preserve_force_settings(self):
        for policy in ("adaptac", "interact_dex"):
            with self.subTest(policy=policy):
                cfg = self.config(policy)
                run = self.root / policy
                self.assertEqual(run_smoke(cfg, run, max_updates=1, resume_updates=1), (1, 2))
                obs = {key: self.source[f"data/{key}"][:2] for key in cfg.dataset.sensor_modalities}
                with closing(load_policy(run, device="cpu")) as runtime:
                    prediction = runtime.predict(obs)
                    self.assertEqual(prediction.shape, (3, 19))
                    self.assertTrue(np.isfinite(prediction).all())
                    self.assertNotIn("finger_tactile_force", runtime.agent.normalizer.fields)
                    if policy == "interact_dex":
                        self.assertEqual(runtime.agent.future_force_loss_beta, 1 / 255)
                        self.assertEqual(runtime.agent.future_force_loss_weight, 0.1)
                    else:
                        self.assertEqual(runtime.agent.force_loss_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
