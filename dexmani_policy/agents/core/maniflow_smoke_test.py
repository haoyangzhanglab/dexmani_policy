"""Focused ManiFlow checks using synthetic data; no external data/weights/simulator.

Run in the policy environment:
    python -m dexmani_policy.agents.core.maniflow_smoke_test

CUDA-specific checks are skipped when CUDA is unavailable.
"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import numpy as np
import hydra
import torch
import torch.nn as nn
import zarr

from dexmani_policy.agents.core.maniflow import ManiFlowAgent, ManiFlowObsEncoder
from dexmani_policy.agents.action_decoders.backbone.consistency_ditx import ConsistencyDiTX
from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
from dexmani_policy.agents.obs_encoder.pointcloud import ops
from dexmani_policy.agents.position_encodings import NeRFSinusoidalPosEmb3D
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.augmentation import PointColorJitter, PointDropout
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer, build_model_and_ema, build_optimizer_and_scheduler,
)
from dexmani_policy.training.eval_utils import load_ckpt_for_inference
from dexmani_policy.training.resume import build_resume_contract, build_train_loader, restore_training_state


def small_config():
    cfg = load_config("maniflow")
    cfg.agent.num_points = 32
    cfg.agent.hidden_dim = 32
    cfg.agent.n_layers = 2
    cfg.agent.n_head = 4
    cfg.agent.timestep_embed_dim = 16
    cfg.agent.target_t_embed_dim = 16
    return cfg


def sample_batch(n=32):
    return {
        "obs": {
            "point_cloud": torch.rand(4, 2, n, 6),
            "joint_state": torch.randn(4, 2, 19),
        },
        "action": torch.randn(4, 16, 19),
    }


def make_models():
    cfg = small_config()
    batch = sample_batch()
    normalizer = LinearNormalizer()
    normalizer.fit({**batch["obs"], "action": batch["action"]}, mode="limits")
    model, ema, updater = build_model_and_ema(cfg, torch.device("cpu"), normalizer)
    return cfg, batch, model, ema, updater


class XYZEncodingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_authoritative_point_count(self):
        for nested in ({}, {"num_points": 32}, {"num_points": 64}):
            with self.subTest(config=nested):
                original = dict(nested)
                kwargs = dict(
                    encoder_type="pointnet_dense", pc_dim=6, state_dim=19,
                    num_points=32, n_obs_steps=2, pc_encoder_config=nested,
                )
                if nested.get("num_points") == 64:
                    with self.assertRaisesRegex(ValueError, "agent.num_points is authoritative"):
                        ManiFlowObsEncoder(**kwargs)
                else:
                    encoder = ManiFlowObsEncoder(**kwargs)
                    self.assertEqual(encoder.pc_encoder.out_shape[0], 32)
                self.assertEqual(nested, original)
        self.assertNotIn("num_points", load_config("maniflow").agent.pc_encoder_config)

    def test_nerf_formula_dtype_and_equivariance(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for dtype in (torch.float32, torch.float64, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    pe = NeRFSinusoidalPosEmb3D(8).to(device=device, dtype=dtype)
                    xyz = torch.rand(2, 17, 3, device=device, dtype=dtype)
                    result = pe(xyz)
                    self.assertEqual(result.shape, (2, 17, 48))
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.device, xyz.device)
                    self.assertEqual(list(pe.parameters()), [])
                    self.assertIn("frequencies", dict(pe.named_buffers()))
                    torch.testing.assert_close(pe.frequencies, xyz.new_tensor([1, 2, 4, 8, 16, 32, 64, 128]))
                    expected = torch.cat([
                        fn(xyz[..., axis:axis + 1] * pe.frequencies)
                        for axis in range(3) for fn in (torch.sin, torch.cos)
                    ], dim=-1)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    torch.testing.assert_close(pe(xyz), result, rtol=0, atol=0)
                    permutation = torch.randperm(17, device=device)
                    torch.testing.assert_close(pe(xyz[:, permutation]), result[:, permutation], rtol=0, atol=0)
        xyz = torch.rand(3, requires_grad=True)
        NeRFSinusoidalPosEmb3D()(xyz).sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                NeRFSinusoidalPosEmb3D(k)

    def test_encoder_shape_and_coordinate_source(self):
        _, _, model, _, _ = make_models()
        for training, n in ((False, 32), (False, 47), (True, 32), (True, 47)):
            model.train(training)
            with self.subTest(n=n, training=training):
                batch = sample_batch(n)
                # Exercise the real dataset augmentation before normalization/FPS.
                dataset = BaseDataset.__new__(BaseDataset)
                dataset.augmentation_cfg = load_config("maniflow").dataset.augmentation_cfg
                dataset._build_augmentors()
                for b in range(4):
                    augmented = dataset.apply_augmentation({
                        "obs": {key: value[b].numpy() for key, value in batch["obs"].items()},
                    })["obs"]
                    for key, value in augmented.items():
                        batch["obs"][key][b] = torch.from_numpy(value)
                captured = {}

                def capture(name):
                    def hook(module, args):
                        captured[name] = args[0].detach().clone()
                    return hook

                h1 = model.obs_encoder.pc_encoder.register_forward_pre_hook(capture("pc"))
                h2 = model.obs_encoder.xyz_pe.register_forward_pre_hook(capture("xyz"))
                rng_state = torch.get_rng_state()
                try:
                    with patch.object(ops, "farthest_point_sample", wraps=ops.farthest_point_sample) as fps:
                        cond, _ = model._build_cond(batch["obs"])
                        self.assertEqual(fps.call_count, int(n > 32))
                finally:
                    h1.remove()
                    h2.remove()
                self.assertEqual(cond.shape, (4, 64, 192))
                torch.testing.assert_close(captured["xyz"], captured["pc"][..., :3], rtol=0, atol=0)
                normalized = model.preprocess(batch["obs"])["point_cloud"]
                torch.set_rng_state(rng_state)
                expected = ops.preprocess_point_cloud(
                    normalized, 32, False, model.obs_encoder.fps_random_config, training=training,
                )
                torch.testing.assert_close(captured["pc"], expected, rtol=0, atol=0)
                self.assertFalse(torch.equal(captured["xyz"], batch["obs"]["point_cloud"].flatten(0, 1)[:, :32, :3]))

    def test_default_encoder_contract(self):
        cfg = load_config("maniflow")
        encoder = ManiFlowObsEncoder(
            encoder_type=cfg.agent.encoder_type, pc_dim=cfg.agent.pc_dim,
            state_dim=cfg.agent.state_dim, num_points=cfg.agent.num_points,
            n_obs_steps=cfg.n_obs_steps, pc_encoder_config=cfg.agent.pc_encoder_config,
        ).eval()
        with torch.no_grad():
            cond, _ = encoder({"point_cloud": torch.rand(2, 1024, 6), "joint_state": torch.rand(2, 19)})
        self.assertEqual(cond.shape, (1, 2048, 192))
        self.assertEqual(encoder.xyz_pe_proj[0].in_features, 48)
        self.assertEqual(encoder.xyz_pe_proj[0].out_features, 128)
        self.assertEqual(encoder.state_mlp.out_dim, 64)


def activate_observation_path(model):
    """Test-only nonzero gates/head: zero-init must not hide order dependence."""
    with torch.no_grad():
        for block in model.ditx_blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
        nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=0.1)
        nn.init.normal_(model.context_frame_pos_embed, std=0.1)


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_direct_constructor_defaults(self):
        backbone = ConsistencyDiTX(
            horizon=16, action_dim=19, n_obs_steps=2, obs_token_dim=192,
            hidden_dim=32, n_layers=2, n_head=4,
        )
        agent = ManiFlowAgent(
            horizon=16, action_dim=19, n_obs_steps=2, n_action_steps=8,
            encoder_type="pointnet_dense", pc_dim=6, state_dim=19,
            num_points=32, hidden_dim=32, n_layers=2, n_head=4,
        )
        for model in (backbone, agent.action_decoder.model):
            for block in model.ditx_blocks:
                self.assertIsInstance(block.self_attn, nn.MultiheadAttention)
                self.assertIsNotNone(block.self_attn.in_proj_bias)
                self.assertIsNotNone(block.self_attn.out_proj.bias)
                self.assertIsNone(block.cross_attn.q.bias)
                self.assertIsNone(block.cross_attn.kv.bias)
                self.assertIsInstance(block.cross_attn.q_norm, nn.Identity)
                self.assertIsInstance(block.cross_attn.k_norm, nn.Identity)

    def test_mha_and_cross_attention(self):
        _, _, agent, _, _ = make_models()
        model = agent.action_decoder.model
        for block in model.ditx_blocks:
            self.assertIsInstance(block.self_attn, nn.MultiheadAttention)
            self.assertIsNotNone(block.self_attn.in_proj_bias)
            self.assertEqual(torch.count_nonzero(block.self_attn.in_proj_bias).item(), 0)
            self.assertEqual(torch.count_nonzero(block.self_attn.out_proj.bias).item(), 0)
            self.assertEqual(block.self_attn.dropout, 0.1)
            self.assertIsNone(block.cross_attn.q.bias)
            self.assertIsNone(block.cross_attn.kv.bias)
            self.assertIsInstance(block.cross_attn.q_norm, nn.Identity)
            self.assertIsInstance(block.cross_attn.k_norm, nn.Identity)
            self.assertEqual(block.cross_attn.attn_drop.p, 0.0)
        self.assertEqual(torch.count_nonzero(model.final_layer.ffn_final.fc2.weight).item(), 0)

    def test_point_permutation_invariance_nonzero_velocity(self):
        _, batch, agent, _, _ = make_models()
        agent.eval()
        backbone = agent.action_decoder.model
        activate_observation_path(backbone)
        pc = batch["obs"]["point_cloud"]
        permutation = torch.rand(pc.shape[:3]).argsort(dim=2)
        permuted = {**batch["obs"], "point_cloud": pc.gather(2, permutation[..., None].expand_as(pc))}
        x, t, target_t = torch.randn(4, 16, 19), torch.rand(4), torch.rand(4)
        with torch.no_grad():
            cond, _ = agent._build_cond(batch["obs"])
            cond_p, _ = agent._build_cond(permuted)
            expected_cond = cond.reshape(4, 2, 32, 192).gather(2, permutation[..., None].expand(4, 2, 32, 192))
            torch.testing.assert_close(cond_p, expected_cond.flatten(1, 2), rtol=1e-5, atol=1e-6)
            # Exercise both existing cross-attention implementations.
            for fused in (False, True):
                for block in backbone.ditx_blocks:
                    block.cross_attn.fused_attn = fused
                original = backbone(x, t, target_t, cond)
                actual = backbone(x, t, target_t, cond_p)
                torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
                self.assertGreater(original.abs().max().item(), 1e-3)
                changed, _ = agent._build_cond({**batch["obs"], "point_cloud": pc + 0.3})
                self.assertGreater((backbone(x, t, target_t, changed) - original).abs().max().item(), 1e-5)


class ConstantVelocity(nn.Module):
    def forward(self, x, timestep, target_t, context):
        return torch.ones_like(x) * 0.25


class TimeGridTest(unittest.TestCase):
    def test_absolute_inference_euler_targets(self):
        decoder = ConsistencyFlowMatch(ConstantVelocity(), target_t_sample_mode="absolute")
        with patch.object(decoder.model, "forward", wraps=decoder.model.forward) as forward:
            decoder.predict_action(torch.zeros(2, 8, 192), torch.zeros(2, 16, 19), inference_steps=4)
        for step, call in enumerate(forward.call_args_list):
            torch.testing.assert_close(call.kwargs["timestep"], torch.full((2,), step / 4))
            torch.testing.assert_close(call.kwargs["target_t"], torch.full((2,), (step + 1) / 4))

    def test_absolute_and_relative_teacher_targets(self):
        actions = torch.randn(2, 16, 19)
        cond = torch.randn(2, 8, 192)
        t, dt = torch.full((2,), 0.9), torch.full((2,), 0.3)
        for mode in ("absolute", "relative"):
            with self.subTest(mode=mode):
                decoder = ConsistencyFlowMatch(ConstantVelocity(), target_t_sample_mode=mode)
                teacher = ConstantVelocity()
                with (
                    patch.object(decoder.time_sampler, "sample", side_effect=[t, dt]),
                    patch.object(teacher, "forward", wraps=teacher.forward) as forward,
                ):
                    targets = decoder.get_consistency_velocity(actions, cond, teacher)
                kwargs = forward.call_args.kwargs
                torch.testing.assert_close(kwargs["timestep"], torch.ones_like(t))
                torch.testing.assert_close(
                    kwargs["target_t"], torch.full_like(t, 1.3) if mode == "absolute" else dt,
                )
                torch.testing.assert_close(
                    targets["target_t"], torch.ones_like(t) if mode == "absolute" else dt,
                )
                torch.testing.assert_close(kwargs["x"], actions)
                torch.testing.assert_close(
                    targets["vt_target"], (actions - targets["xt"]) / (1 - t[:, None, None]),
                )

    def test_grid_and_targets_independent_of_inference_nfe(self):
        actions = torch.randn(10, 16, 19)
        cond = torch.randn(10, 8, 192)
        expected_targets = None
        expected_samples = None
        for nfe in (1, 2, 4, 10):
            decoder = ConsistencyFlowMatch(ConstantVelocity(), denoise_timesteps=10, num_inference_steps=nfe)
            self.assertEqual(decoder.time_sampler.num_steps, 10)
            torch.manual_seed(52)
            samples = decoder.time_sampler.sample(1024, "discrete", "cpu")
            torch.testing.assert_close(samples * 10, (samples * 10).round())
            self.assertEqual(set(samples.mul(10).round().int().tolist()), set(range(10)))
            if expected_samples is None:
                expected_samples = samples
            torch.testing.assert_close(samples, expected_samples, rtol=0, atol=0)
            # Include every grid time, especially t=0.9 where the old NFE clamp failed.
            t = torch.arange(10).float() / 10
            dt = torch.full((10,), 0.05)
            torch.manual_seed(53)
            with patch.object(decoder.time_sampler, "sample", side_effect=[t, dt]):
                targets = decoder.get_consistency_velocity(actions, cond, ConstantVelocity())
            t_view = t[:, None, None]
            x0 = (targets["xt"] - t_view * actions) / (1 - t_view)
            next_t = (t_view + dt[:, None, None]).clamp(max=1)
            endpoint = (1 - next_t) * x0 + next_t * actions + 0.25 * (1 - next_t)
            torch.testing.assert_close(targets["vt_target"], (endpoint - targets["xt"]) / (1 - t_view))
            torch.testing.assert_close(targets["target_t"], dt)
            if expected_targets is None:
                expected_targets = targets
            for key, value in targets.items():
                torch.testing.assert_close(value, expected_targets[key], rtol=0, atol=0)
            # Runtime override changes only the number of Euler evaluations.
            with patch.object(decoder.model, "forward", wraps=decoder.model.forward) as forward:
                torch.manual_seed(54)
                result = decoder.predict_action(cond, actions, inference_steps=4)
                self.assertEqual(forward.call_count, 4)
            torch.manual_seed(54)
            torch.testing.assert_close(result, torch.randn_like(actions) + 0.25)
            self.assertEqual(decoder.time_sampler.num_steps, 10)
            self.assertEqual(decoder.num_inference_steps, nfe)
            self.assertEqual(decoder.denoise_timesteps, 10)
        for steps in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "denoise_timesteps"):
                ConsistencyFlowMatch(ConstantVelocity(), denoise_timesteps=steps)


class AugmentationTest(unittest.TestCase):
    def test_contrast_global_mean_and_xyz_preservation(self):
        pc = np.linspace(0.15, 0.35, 2 * 17 * 6, dtype=np.float32).reshape(2, 17, 6)
        original = pc.copy()
        aug = PointColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0)
        with patch("numpy.random.uniform", return_value=1.25):
            aug._augment(pc)
        before = original[..., 3:]
        np.testing.assert_allclose(pc[..., 3:], (before - before.mean()) * 1.25 + before.mean(), rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(pc[..., 3:].mean()), float(before.mean()), places=6)
        np.testing.assert_array_equal(pc[..., :3], original[..., :3])
        self.assertGreater(float(np.abs(pc[..., 3:] - before).max()), 1e-3)

    def test_hue_and_clipping_remain_available(self):
        pc = np.zeros((2, 5, 6), dtype=np.float32)
        pc[..., 3:] = [0.8, 0.3, 0.1]
        original = pc.copy()
        with patch("numpy.random.uniform", return_value=0.06):
            PointColorJitter(brightness=0, contrast=0, saturation=0, hue=0.08)._augment(pc)
        self.assertFalse(np.array_equal(pc[..., 3:], original[..., 3:]))
        np.testing.assert_array_equal(pc[..., :3], original[..., :3])
        for delta in (-2.0, 2.0):
            with patch("numpy.random.uniform", return_value=delta):
                PointColorJitter(brightness=3, contrast=0, saturation=0, hue=0)._augment(pc)
            self.assertTrue(np.isfinite(pc).all())
            self.assertTrue(((pc[..., 3:] >= 0) & (pc[..., 3:] <= 1)).all())

    def test_point_dropout_per_frame_first_point_replacement(self):
        pc = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
        expected = pc.copy()
        expected[0, [0, 1]] = pc[0, 0]
        expected[1, [2, 3, 4]] = pc[1, 0]
        with (
            patch("numpy.random.uniform", side_effect=[0.2, 0.7]) as ratios,
            patch("numpy.random.random", side_effect=[
                np.array([0.1, 0.2, 0.25, 0.6, 0.95]),
                np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            ]) as masks,
        ):
            PointDropout()._augment(pc)
        self.assertEqual(ratios.call_args_list, [call(0.0, 0.8), call(0.0, 0.8)])
        self.assertEqual(masks.call_args_list, [call(5), call(5)])
        np.testing.assert_array_equal(pc, expected)

    def test_shared_policy_dataset_augmentation(self):
        for name in ("maniflow", "dp3", "r3d", "sat", "dqrise"):
            with self.subTest(policy=name):
                cfg = load_config(name)
                dataset = BaseDataset.__new__(BaseDataset)
                dataset.augmentation_cfg = cfg.dataset.augmentation_cfg
                dataset._build_augmentors()
                self.assertTrue(any(isinstance(a, PointColorJitter) for a in dataset.augmentors["point_cloud"]))
                pc = np.random.default_rng(12).uniform(0.2, 0.8, (2, 32, 6)).astype(np.float32)
                state = np.zeros((2, 19), dtype=np.float32)
                original = pc.copy()
                data = {"obs": {"point_cloud": pc, "joint_state": state}}
                np.random.seed(13)
                actual = dataset.apply_augmentation(data)["obs"]
                np.testing.assert_array_equal(pc, original)
                np.testing.assert_array_equal(state, 0)
                self.assertFalse(np.shares_memory(pc, actual["point_cloud"]))
                self.assertEqual(actual["point_cloud"].shape, pc.shape)
                self.assertTrue(np.isfinite(actual["point_cloud"]).all())
                xyz_reference = pc[..., :3]
                if name in ("dp3", "dqrise", "sat"):
                    # Dropout copies the augmented first row, including its XYZ noise.
                    copied = (actual["point_cloud"] == actual["point_cloud"][:, :1]).all(axis=-1)
                    xyz_reference = np.where(copied[..., None], pc[:, :1, :3], xyz_reference)
                self.assertLessEqual(np.abs(actual["point_cloud"][..., :3] - xyz_reference).max(), 0.004001)
                self.assertLessEqual(np.abs(actual["joint_state"]).max(), 0.000401)
                self.assertGreater(np.abs(actual["point_cloud"][..., 3:] - pc[..., 3:]).max(), 1e-4)
                self.assertTrue(((actual["point_cloud"][..., 3:] >= 0) & (actual["point_cloud"][..., 3:] <= 1)).all())
                self.assertEqual(cfg.dataset.augmentation_cfg.pc.color.hue, 0.0)
                if name in ("dp3", "dqrise", "sat"):
                    self.assertEqual(dict(cfg.dataset.augmentation_cfg.pc.dropout),
                                     dict(max_dropout_ratio=0.8, prob=1.0))
                    self.assertTrue(any(isinstance(a, PointDropout) for a in dataset.augmentors["point_cloud"]))
                else:
                    self.assertNotIn("dropout", cfg.dataset.augmentation_cfg.pc)


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        np.random.seed(61)

    def test_default_recipe_and_model_structure(self):
        cfg = load_config("maniflow")
        expected = {
            "horizon": 16, "n_obs_steps": 2, "n_action_steps": 8, "action_dim": 19,
            "pc_dim": 6, "state_dim": 19, "state_out_dim": 64, "num_points": 1024,
            "xyz_pe_num_frequencies": 8, "n_layers": 12, "hidden_dim": 768,
            "n_head": 8, "mlp_ratio": 4.0, "p_drop_attn": 0.1,
            "qkv_bias": False, "qk_norm": False, "pre_norm_modality": False,
            "denoise_timesteps": 10, "num_inference_steps": 10, "flow_batch_ratio": 0.75,
            "t_sample_mode_for_flow": "beta", "t_sample_mode_for_consistency": "discrete",
            "dt_sample_mode_for_consistency": "uniform", "target_t_sample_mode": "relative",
        }
        for key, value in expected.items():
            self.assertEqual(cfg.agent[key], value, key)
        self.assertNotIn("hidden_dims", cfg.agent.pc_encoder_config)
        self.assertEqual(cfg.eval.inference_steps, 4)
        self.assertTrue(cfg.training.use_compile)
        self.assertTrue(cfg.training.use_bfloat16)
        self.assertEqual(cfg.training.max_grad_norm, 1.0)
        self.assertEqual(cfg.training.loop.total_train_steps, 100000)
        self.assertEqual(cfg.optimizer.lr, 1e-4)
        self.assertEqual(cfg.optimizer.weight_decay, 1e-3)
        self.assertEqual(cfg.ema.power, 0.75)
        aug = cfg.dataset.augmentation_cfg
        self.assertEqual(aug.pc.coord_noise.noise_std, 0.002)
        self.assertEqual(aug.state.noise.noise_std, 0.0002)
        self.assertEqual(dict(aug.pc.color), dict(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.0, prob=1.0))
        # Construct the unscaled default architecture without allocating training-sized weights.
        with torch.device("meta"):
            agent = hydra.utils.instantiate(cfg.agent)
        backbone = agent.action_decoder.model
        self.assertEqual(len(backbone.ditx_blocks), 12)
        self.assertEqual(backbone.context_embedder.weight.shape, (768, 192))
        self.assertEqual(backbone.context_frame_pos_embed.shape, (1, 2, 768))
        self.assertEqual(backbone.input_embedder.weight.shape, (768, 19))
        for block in backbone.ditx_blocks:
            self.assertIsInstance(block.self_attn, nn.MultiheadAttention)
            self.assertEqual(block.self_attn.num_heads, 8)
            self.assertEqual(block.mlp.fc1.out_features, 3072)

    def test_dataset_training_ema_checkpoint_and_execution(self):
        cfg = small_config()
        cfg.training.device = "cpu"
        cfg.dataloader.batch_size = 4
        cfg.dataloader.num_workers = 0
        with tempfile.TemporaryDirectory(prefix="maniflow-smoke-") as tmp:
            cfg.dataset.zarr_path = str(Path(tmp) / "synthetic.zarr")
            root = zarr.open_group(cfg.dataset.zarr_path, mode="w")
            data = root.create_group("data")
            rng = np.random.default_rng(62)
            for key, shape in (("point_cloud", (64, 32, 6)), ("joint_state", (64, 19)), ("action", (64, 19))):
                data.create_dataset(key, data=rng.random(shape).astype(np.float32))
            root.create_group("meta").create_dataset("episode_ends", data=np.array([32, 64], dtype=np.int64))
            dataset, normalizer = build_dataset_and_normalizer(cfg)
            loader = build_train_loader(cfg, dataset)
            agent, ema, updater = build_model_and_ema(cfg, torch.device("cpu"), normalizer)
            optimizer, scheduler = build_optimizer_and_scheduler(cfg, agent, len(loader))
            params = [p for g in optimizer.param_groups for p in g["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params}, {id(p) for p in agent.parameters() if p.requires_grad})
            batch = next(iter(loader))
            self.assertEqual(batch["obs"]["point_cloud"].shape, (4, 2, 32, 6))
            self.assertEqual(batch["obs"]["joint_state"].shape, (4, 2, 19))
            self.assertEqual(batch["action"].shape, (4, 16, 19))
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with patch.object(agent.action_decoder.model, "forward", wraps=agent.action_decoder.model.forward) as online:
                    loss, metrics = agent.compute_loss(batch, **agent.get_training_loss_kwargs(ema))
                    self.assertEqual(online.call_count, 1)
                self.assertEqual(metrics["flow_batch_size"], 3)
                self.assertEqual(metrics["consistency_batch_size"], 1)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in agent.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(all(p.grad is None for p in ema.parameters()))
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                updater.step(agent)
            agent.eval()
            cond, _ = agent._build_cond(batch["obs"])
            self.assertEqual(cond.shape, (4, 64, 192))
            torch.manual_seed(63)
            result = agent.predict_action(batch["obs"], inference_steps=cfg.eval.inference_steps)
            self.assertEqual(result["pred_action"].shape, (4, 16, 19))
            self.assertEqual(result["control_action"].shape, (4, 8, 19))
            torch.testing.assert_close(result["control_action"], result["pred_action"][:, 1:9], rtol=0, atol=0)
            clone = copy.deepcopy(agent)
            clone.load_state_dict(agent.state_dict(), strict=True)
            contract = build_resume_contract(cfg, agent, loader)
            store = CheckpointStore(Path(tmp) / "checkpoints")
            checkpoint = TrainCheckpoint(
                epoch=0, global_step=2, next_micro_step=0,
                model_state=agent.state_dict(), ema_model_state=ema.state_dict(),
                optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                monitor={}, resume_contract=contract, ema_updater_step=updater.optimization_step,
                ema_decay=updater.decay, rng_states=[get_rng_state()],
            )
            checkpoint_path = store.save("smoke.pt", checkpoint)
            loaded = store.load(checkpoint_path)
            restored, restored_ema, restored_updater = build_model_and_ema(cfg, torch.device("cpu"), normalizer)
            restored_optimizer, restored_scheduler = build_optimizer_and_scheduler(cfg, restored, len(loader))
            self.assertEqual(restore_training_state(
                loaded, resume_contract=build_resume_contract(cfg, restored, loader),
                model=restored, ema_model=restored_ema, ema_updater=restored_updater,
                optimizer=restored_optimizer, scheduler=restored_scheduler, device=torch.device("cpu"),
            ), (2, 0, 0))
            for source, target in ((agent, restored), (ema, restored_ema)):
                for key, tensor in source.state_dict().items():
                    torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)
            self.assertEqual(restored_updater.optimization_step, updater.optimization_step)
            changed_contract = copy.deepcopy(contract)
            changed_contract["agent_config"]["denoise_timesteps"] = 4
            with self.assertRaisesRegex(ValueError, "denoise_timesteps"):
                restore_training_state(
                    loaded, resume_contract=changed_contract, model=restored,
                    ema_model=restored_ema, ema_updater=restored_updater,
                    optimizer=restored_optimizer, scheduler=restored_scheduler,
                    device=torch.device("cpu"),
                )
            # The checkpoint constructor, including training grid, owns evaluation.
            cfg.agent.denoise_timesteps = 3
            for use_ema, source in ((False, agent), (True, ema)):
                inference = load_ckpt_for_inference(store, checkpoint_path, use_ema, cfg=cfg).eval()
                self.assertEqual(inference.action_decoder.time_sampler.num_steps, 10)
                torch.manual_seed(64)
                expected = source.predict_action(batch["obs"], inference_steps=4)
                torch.manual_seed(64)
                actual = inference.predict_action(batch["obs"], inference_steps=4)
                torch.testing.assert_close(actual["pred_action"], expected["pred_action"], rtol=0, atol=0)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_bf16_training_and_ema(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                _, batch, agent, ema, updater = make_models()
                agent.to(device).train()
                ema.to(device)
                activate_observation_path(agent.action_decoder.model)
                ema.load_state_dict(agent.state_dict(), strict=True)
                batch = {
                    "obs": {key: value.to(device) for key, value in batch["obs"].items()},
                    "action": batch["action"].to(device),
                }
                projection_dtypes = []

                def record_projection_dtype(module, args, output):
                    projection_dtypes.append(output.dtype)

                hook = agent.obs_encoder.xyz_pe_proj[0].register_forward_hook(record_projection_dtype)
                try:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        cond, _ = agent._build_cond(batch["obs"])
                        # CUDA autocast runs LayerNorm in FP32; mixed-precision
                        # conditioning need not itself remain BF16.
                        self.assertIn(cond.dtype, (torch.float32, torch.bfloat16))
                        loss, _ = agent.compute_loss(batch, **agent.get_training_loss_kwargs(ema))
                finally:
                    hook.remove()
                self.assertEqual(projection_dtypes, [torch.bfloat16, torch.bfloat16])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in agent.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(agent.obs_encoder.xyz_pe_proj[0].weight.grad.abs().max().item(), 0)
                updater.step(agent)
                self.assertTrue(all(p.grad is None for p in ema.parameters()))

    def _check_compiled_backbone(self, device, backend):
        _, batch, agent, _, _ = make_models()
        agent.to(device).eval()
        eager = agent.action_decoder.model
        activate_observation_path(eager)
        compiled_source = copy.deepcopy(eager)
        compiled = torch.compile(compiled_source, backend=backend, fullgraph=True)
        obs = {key: value.to(device) for key, value in batch["obs"].items()}
        with torch.no_grad():
            cond, _ = agent._build_cond(obs)
        args = (torch.randn(4, 16, 19, device=device), torch.rand(4, device=device), torch.rand(4, device=device), cond)
        for bf16 in (False, True):
            with self.subTest(device=device, backend=backend, bf16=bf16):
                eager.zero_grad(set_to_none=True)
                compiled.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16):
                    expected = eager(*args)
                    actual = compiled(*args)
                tolerance = dict(rtol=0.03, atol=0.02) if bf16 else dict(rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(actual, expected, **tolerance)
                expected.float().square().mean().backward()
                actual.float().square().mean().backward()
                for (name, parameter), (_, other) in zip(eager.named_parameters(), compiled_source.named_parameters()):
                    self.assertTrue(torch.isfinite(other.grad).all(), name)
                    torch.testing.assert_close(other.grad, parameter.grad, **tolerance)

    def test_compile_cpu(self):
        self._check_compiled_backbone("cpu", "aot_eager")

    def test_compiled_policy_training_inference_and_ema(self):
        _, batch, agent, ema, updater = make_models()
        activate_observation_path(agent.action_decoder.model)
        ema.load_state_dict(agent.state_dict(), strict=True)
        agent.compile_backbone(backend="aot_eager")
        ema.compile_backbone(backend="aot_eager")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = agent.compute_loss(batch, **agent.get_training_loss_kwargs(ema))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(agent.obs_encoder.xyz_pe_proj[0].weight.grad).all())
        updater.step(agent)
        agent.eval()
        with torch.no_grad():
            result = agent.predict_action(batch["obs"], inference_steps=4)
            ema_result = ema.predict_action(batch["obs"], inference_steps=4)
        for prediction in (result, ema_result):
            self.assertEqual(prediction["control_action"].shape, (4, 8, 19))
            self.assertTrue(torch.isfinite(prediction["pred_action"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available; CUDA BF16/Inductor NOT VERIFIED")
    def test_compile_cuda(self):
        self._check_compiled_backbone("cuda", "inductor")


class FrameEncodingTest(unittest.TestCase):
    def test_frame_broadcast_and_parameter_names(self):
        _, _, agent, _, _ = make_models()
        model = agent.action_decoder.model.eval()
        for names in (model.state_dict(), dict(model.named_parameters())):
            self.assertNotIn("context_pos_embed", names)
        self.assertEqual(model.context_frame_pos_embed.shape, (1, 2, 32))
        self.assertEqual(torch.count_nonzero(model.context_frame_pos_embed).item(), 0)
        with torch.no_grad():
            model.context_frame_pos_embed[0, 0].fill_(1)
            model.context_frame_pos_embed[0, 1].fill_(3)
        captured = []

        def hook(module, args):
            captured.append(args[2].detach())

        handle = model.ditx_blocks[0].register_forward_pre_hook(hook)
        try:
            for count in (6, 10):
                context = torch.randn(2, count, 192)
                with torch.no_grad():
                    output = model(torch.randn(2, 16, 19), torch.rand(2), torch.rand(2), context)
                    added = captured[-1] - model.context_embedder(context)
                expected = torch.ones_like(added)
                expected[:, count // 2:] = 3
                torch.testing.assert_close(added, expected)
                self.assertEqual(output.shape, (2, 16, 19))
            for count in (0, 7):
                with self.assertRaisesRegex(ValueError, "divisible"):
                    model(torch.randn(2, 16, 19), torch.rand(2), torch.rand(2), torch.randn(2, count, 192))
        finally:
            handle.remove()
        groups = model.get_optim_groups()
        no_decay = {id(p) for group in groups if group["weight_decay"] == 0 for p in group["params"]}
        self.assertIn(id(model.context_frame_pos_embed), no_decay)


if __name__ == "__main__":
    unittest.main()
