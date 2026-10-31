"""CPU contracts for InterAct-Dex; native FPS/kNN is replaced only inside these tests."""

import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction.gated_tactile_fusion import GatedTactileFusion
from dexmani_policy.agents.obs_encoder.interaction.scene_aggregator import sample_region_seeds
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.common.checkpoint import Checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.config import validate_future_force_offsets
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.datasets.variants.tactile import InterActDexDataset, read_future_force
from dexmani_policy.training.common.config import load_policy_config, validate_config


def sample_cpu_patches(xyz, num_patches, group_size, *args):
    indices = sample_region_seeds(xyz, num_patches)
    centers = index_points(xyz, indices)
    neighbors = torch.cdist(centers, xyz).topk(group_size, largest=False).indices
    return centers, indices, neighbors


class InterActDexChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        stack = ExitStack()
        self.addCleanup(stack.close)
        module = "dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder"
        stack.enter_context(patch(f"{module}.load_pointcloud_ops"))
        stack.enter_context(patch(f"{module}.sample_point_patches", side_effect=sample_cpu_patches))

    def make_policy(self, *overrides):
        cfg = load_policy_config("interact_dex", [
            "horizon=6", "n_obs_steps=3", "n_action_steps=2", "agent.num_points=16",
            "agent.pc_encoder_config.channels=48", "agent.pc_encoder_config.num_patches=12",
            "agent.pc_encoder_config.group_size=4", "agent.pc_encoder_config.depth=1",
            "agent.pc_encoder_config.num_heads=4", "agent.hidden_dim=64", "agent.n_layers=1",
            "agent.p_drop_attn=0", "agent.num_inference_steps=2", *overrides,
        ])
        cfg.workspace.output_dir = str(Path.cwd())
        model = hydra.utils.instantiate(cfg.agent)
        dims = dict(action=19, joint_state=19, point_cloud=6, eef_pose=9,
                    fingertip_xyz=15, finger_tactile_force=15)
        model.configure_normalizer(dims, normalization_feature_shapes({key: dims[key] for key, mode in cfg.normalization.items() if mode != "identity"}))
        return model, cfg, dims

    def batch(self):
        pose = torch.zeros(2, 3, 9)
        pose[..., 3] = pose[..., 7] = 1
        return {
            "obs": {
                "joint_state": torch.randn(2, 3, 19),
                "point_cloud": torch.randn(2, 3, 16, 6) * 0.03,
                "eef_pose": pose,
                "fingertip_xyz": torch.randn(2, 3, 5, 3) * 0.01,
                "finger_tactile_force": torch.randn(2, 3, 5, 3),
            },
            "action": torch.randn(2, 6, 19),
            "future_force": torch.randn(2, 4, 5, 3),
            "future_force_valid": torch.ones(2, 4, dtype=torch.bool),
        }

    def test_grouped_flow_balances_arm_and_hand(self):
        class ConstantVelocity(nn.Module):
            def __init__(self):
                super().__init__()
                self.velocity = nn.Parameter(torch.tensor([1.0] * 7 + [2.0] * 12))

            def forward(self, x, **kwargs):
                return self.velocity.expand_as(x)

        decoder = RectifiedFlow(ConstantVelocity())
        actions = torch.zeros(2, 3, 19)
        with patch.object(decoder, "sample_training_pair", return_value=(actions, torch.zeros(2), actions)):
            loss, metrics = decoder.compute_loss(
                torch.zeros(2, 1), actions, dim_groups={"arm": (0, 7, 0.5), "hand": (7, 19, 0.5)},
            )
            self.assertEqual(loss.item(), 2.5)
            self.assertEqual(metrics["loss_arm"].item(), 1.0)
            self.assertEqual(metrics["loss_hand"].item(), 4.0)
            loss.backward()
            torch.testing.assert_close(decoder.model.velocity.grad[:7], torch.full((7,), 1 / 7))
            torch.testing.assert_close(decoder.model.velocity.grad[7:], torch.full((12,), 2 / 12))
            original_loss, metrics = decoder.compute_loss(torch.zeros(2, 1), actions, log_metrics=False)
            torch.testing.assert_close(original_loss, torch.tensor(55 / 19))
            self.assertEqual(metrics, {})

    def test_fusion_keeps_finger_correspondence_and_valid_zero_force(self):
        fusion = GatedTactileFusion()
        relation = dict(hand_query=torch.randn(2, 6, 192), relation_update=torch.randn(2, 6, 192),
                        geometry_descriptor=torch.randn(2, 6, 7))
        force = torch.randn(2, 5, 3)
        before = fusion(relation, tactile=force)["interaction_tokens"]
        changed = force.clone()
        changed[:, 2] += 3
        after = fusion(relation, tactile=changed)["interaction_tokens"]
        torch.testing.assert_close(before[:, [1, 2, 4, 5]], after[:, [1, 2, 4, 5]], rtol=0, atol=0)
        self.assertGreater((before[:, [0, 3]] - after[:, [0, 3]]).abs().sum().item(), 0)
        zeros = fusion(relation, tactile=torch.zeros_like(force))["interaction_tokens"]
        self.assertTrue(torch.isfinite(zeros).all())
        zeros.square().mean().backward()
        self.assertGreater(fusion.fusion_mlp[-1].weight.grad.abs().sum().item(), 0)

    def test_force_offsets_preserve_episode_boundaries_and_zero_labels(self):
        class Store:
            episode_ends = np.array([5, 11])
            force = np.broadcast_to(np.arange(11, dtype=np.float32)[:, None, None], (11, 5, 3)).copy()
            force[2] = 0

            def read(self, key, rows):
                return self.force[rows]

        store = Store()
        force, valid = read_future_force(store, 1, (1, 4, 8, 13))
        np.testing.assert_array_equal(force[:, 0, 0], [0, 4, 4, 4])
        np.testing.assert_array_equal(valid, [True, False, False, False])
        force, valid = read_future_force(store, 4, (1, 4, 8, 13))
        self.assertFalse(valid.any())
        self.assertTrue((force == 4).all())
        force, valid = read_future_force(store, 5, range(1, 8))
        np.testing.assert_array_equal(force[:, 0, 0], [6, 7, 8, 9, 10, 10, 10])
        np.testing.assert_array_equal(valid, [True] * 5 + [False] * 2)
        for offsets in ((), (0,), (True,), (2, 1), (1, 1)):
            with self.assertRaises(ValueError):
                validate_future_force_offsets(offsets)

    def test_dataset_uses_query_source_row_under_padding(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            group = zarr.open_group(directory, mode="w")
            group.attrs.update(dt=0.0625, physics_dt=1 / 240, frame_skip=15, action_dim=19,
                               action_space="joint", obs_alignment="obs[t]_before_action[t]")
            group.create_dataset("meta/episode_ends", data=np.array([6, 12]))
            group.create_dataset("data/action", data=np.zeros((12, 19), dtype=np.float32))
            group.create_dataset("data/finger_tactile_force", data=np.broadcast_to(
                np.arange(12, dtype=np.float32)[:, None, None], (12, 5, 3),
            ))
            dataset = InterActDexDataset(
                zarr_path=directory, sensor_modalities=["finger_tactile_force"], tactile_source="sim",
                horizon=4, obs_horizon=3, pad_before=2, pad_after=3, future_force_offsets=(1, 4, 8, 13),
            )
            saw_terminal = False
            for index in range(len(dataset)):
                batch = dataset[index]
                query = int(batch["obs"]["finger_tactile_force"][-1, 0, 0])
                end = 6 if query < 6 else 12
                expected_rows = np.minimum(query + np.array([1, 4, 8, 13]), end - 1)
                np.testing.assert_array_equal(batch["future_force"][:, 0, 0], expected_rows)
                np.testing.assert_array_equal(batch["future_force_valid"], query + np.array([1, 4, 8, 13]) < end)
                self.assertEqual(set(batch["obs"]), {"finger_tactile_force"})
                saw_terminal |= query == end - 1
            self.assertTrue(saw_terminal)

    def test_auxiliary_prediction_is_action_independent_and_trains_observation(self):
        model, _, _ = self.make_policy()
        model.eval()
        batch = self.batch()
        outputs, encodings = [], []
        handles = [model.future_force_head.register_forward_hook(lambda m, a, o: outputs.append(o)),
                   model.obs_encoder.register_forward_hook(lambda m, a, o: encodings.append(o))]
        for handle in handles:
            self.addCleanup(handle.remove)
        model.compute_loss(batch)
        changed = {**batch, "action": batch["action"] * 100}
        model.compute_loss(changed)
        self.assertEqual(len(encodings), 2)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        outputs[0].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.obs_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.action_decoder.parameters()))

    def test_sim_force_uses_observed_eef_frames(self):
        batch = self.batch()
        # A 90-degree world-from-EEF rotation about Z: world +X is EEF -Y.
        batch["obs"]["eef_pose"][..., 3:] = torch.tensor([0, 1, 0, -1, 0, 0])
        batch["obs"]["eef_pose"][:, 0, 3:] = torch.tensor([1, 0, 0, 0, 1, 0])
        batch["obs"]["eef_pose"][:, 1, 3:] = torch.tensor([-1, 0, 0, 0, -1, 0])
        batch["obs"]["finger_tactile_force"].zero_()[..., 0] = 1
        batch["future_force"].zero_()[..., 0] = 1
        for source in ("sim", "real"):
            model, _, _ = self.make_policy(f"tactile={source}_force")
            model.normalizer["finger_tactile_force"].scale.fill_(2)
            captured = []
            handle = model.obs_encoder.tactile_fusion.register_forward_pre_hook(
                lambda m, a, kw: captured.append(kw["tactile"].detach()), with_kwargs=True,
            )
            self.addCleanup(handle.remove)
            target = torch.tensor([0.0, -2.0, 0.0] if source == "sim" else [2.0, 0.0, 0.0])
            with torch.no_grad():
                model.future_force_head[-1].weight.zero_()
                model.future_force_head[-1].bias.copy_(target.repeat(4))
            _, metrics = model.compute_loss(batch)
            expected_history = torch.tensor([[2.0, 0, 0], [-2.0, 0, 0], [0, -2.0, 0]])
            if source == "real":
                expected_history = target.expand(3, 3)
            torch.testing.assert_close(
                captured[-1].reshape(2, 3, 5, 3), expected_history[None, :, None].expand(2, 3, 5, 3),
            )
            self.assertEqual(metrics["loss_future_force"].item(), 0)

    def test_scene_context_has_gradient_from_interaction_tokens(self):
        model, _, _ = self.make_policy()
        projected = []

        def capture(module, args, output):
            output.retain_grad()
            projected.append(output)

        handle = model.obs_encoder.scene_projection.register_forward_hook(capture)
        self.addCleanup(handle.remove)
        condition = model.encode_observation(self.batch()["obs"]).reshape(2, 3, 18, 192)
        interaction = condition[:, :, :6]
        (interaction * torch.randn_like(interaction)).sum().backward()
        self.assertGreater(projected[0].grad.abs().sum().item(), 0)

    def test_training_restore_and_inference_without_auxiliary_head_calls(self):
        model, cfg, dims = self.make_policy()
        batch = self.batch()
        groups = model.optimizer_groups(lr=1e-3, obs_lr=1e-3, weight_decay=0, obs_weight_decay=0)
        parameters = [p for group in groups for p in group["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters() if p.requires_grad})
        optimizer = torch.optim.AdamW(groups)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(metrics["loss_action"], 0.5 * (metrics["loss_arm"] + metrics["loss_hand"]))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
        optimizer.step()
        effective = OmegaConf.to_container(cfg, resolve=True)
        effective["agent"].update(model.effective_config)
        effective["normalization_dims"] = dims
        effective["normalizer_shapes"] = {key: list(field.scale.shape) for key, field in model.normalizer.fields.items()}
        checkpoint = Checkpoint(1, model.state_dict(), None, optimizer.state_dict(), {}, {}, effective)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = save_checkpoint(Path(directory) / "policy.pt", checkpoint)
            restored = restore_policy(path, use_ema=False, device="cpu")
        model.eval()
        with patch.object(model.future_force_head, "forward", side_effect=AssertionError("inference called auxiliary head")):
            expected = model.predict_action(batch["obs"], generator=torch.Generator().manual_seed(9))
        actual = restored.predict_action(batch["obs"], generator=torch.Generator().manual_seed(9))
        self.assertEqual(actual["future_actions"].shape, (2, 4, 19))
        torch.testing.assert_close(actual["future_actions"], expected["future_actions"], rtol=0, atol=0)

    def test_empty_future_supervision_and_disabled_head(self):
        batch = self.batch()
        batch["future_force_valid"].zero_()
        batch["future_force"].fill_(float("nan"))
        model, _, _ = self.make_policy()
        loss, metrics = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["loss_future_force"].item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and (p.grad == 0).all() for p in model.future_force_head.parameters()))
        disabled, _, _ = self.make_policy("agent.future_force_loss_weight=0")
        self.assertIsNone(disabled.future_force_head)
        _, metrics = disabled.compute_loss({"obs": batch["obs"], "action": batch["action"]})
        self.assertIn("loss_arm", metrics)
        self.assertIn("loss_hand", metrics)

    def test_config_assembles_offset_semantics(self):
        for source in ("sim", "real"):
            cfg = load_policy_config("interact_dex", [f"tactile={source}_force"])
            validate_config(cfg)
            self.assertEqual(list(cfg.dataset.future_force_offsets), list(cfg.future_force_offsets))
            self.assertEqual(list(cfg.agent.future_force_offsets), list(cfg.future_force_offsets))

    def test_single_camera_quality_and_outside_frustum(self):
        from dexmani_policy.agents.obs_encoder.pointcloud.feature_alignment import FeatureAlignment, build_patch_correspondence
        from dexmani_policy.agents.obs_encoder.pointcloud.projection import PointProjection

        indices = torch.tensor([[[0, 1]]])
        projection = PointProjection(torch.tensor([[[0., 0.], [2., 2.]]]), torch.tensor([[0.37, 1.]]))
        geometry = build_patch_correspondence(indices, projection, (2, 2))
        torch.testing.assert_close(geometry["coverage"], torch.tensor([[0.5]]))
        torch.testing.assert_close(geometry["confidence"], torch.tensor([[0.37]]))
        torch.testing.assert_close(geometry["pool_weight"], torch.full((1, 1, 4), 0.25))
        fusion = FeatureAlignment(8, 12)
        tokens = torch.randn(1, 1, 12)
        outside = PointProjection(torch.full((1, 2, 2), 2.), torch.ones(1, 2))
        actual = fusion(tokens, torch.randn(1, 2, 2, 8), outside, indices)
        torch.testing.assert_close(actual, tokens, rtol=0, atol=0)

    def test_check_data_rejects_fp32_overflow_and_empty_cloud(self):
        from dexmani_policy.datasets.tools.check_data import check_data

        with tempfile.TemporaryDirectory() as directory:
            group = zarr.open_group(directory, mode="w")
            group.create_dataset("meta/episode_ends", data=np.array([2]))
            group.create_dataset("data/point_cloud", data=np.full((2, 4, 3), 1e100, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "Nonfinite data:.*field=point_cloud.*row=0"):
                check_data(directory)
            del group["data/point_cloud"]
            group.create_dataset("data/point_cloud", shape=(2, 0, 3), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "Empty data field"):
                check_data(directory)

    def test_fresh_identity_pointcloud_records_consumed_channels(self):
        from dexmani_policy.training.common.data import build_dataset, fit_normalizer

        with tempfile.TemporaryDirectory() as directory:
            group = zarr.open_group(directory, mode="w")
            group.attrs.update(dt=0.0625, physics_dt=1 / 240, frame_skip=15, action_dim=19,
                               action_space="joint", obs_alignment="obs[t]_before_action[t]")
            group.create_dataset("meta/episode_ends", data=np.array([3, 6]))
            group.create_dataset("data/action", data=np.zeros((6, 19), dtype=np.float32))
            for key, values in self.batch()["obs"].items():
                group.create_dataset(f"data/{key}", data=values.flatten(0, 1).numpy())
            for channels in (3, 6):
                model, cfg, _ = self.make_policy(
                    f"zarr_path={directory}", f"agent.pc_dim={channels}", "dataset.augmentation_cfg=null",
                )
                dataset, description = build_dataset(cfg)
                dims = description["normalization_dims"]
                self.assertEqual(dims["point_cloud"], channels)
                model.configure_normalizer(dims, {})
                model.normalizer = fit_normalizer(dataset, cfg.normalization, feature_dims=dims)
                self.assertNotIn("point_cloud", model.normalizer.fields)
                obs = {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}
                self.assertEqual(model.preprocess(obs)["point_cloud"].shape[-1], channels)
                self.assertTrue(torch.isfinite(model.encode_observation(obs)).all())

    def test_runtime_converts_uint8_hwc_and_rejects_other_layouts(self):
        from types import SimpleNamespace
        from dexmani_policy.common.image_util import preprocess_rgb_eval
        from dexmani_policy.deployment.runtime import PolicyRuntime

        recipe = dict(resize_hw=None, center_crop_hw=None, keep_uint8=True)
        info = SimpleNamespace(observation_fields=("rgb",), n_obs_steps=2, rgb_preprocessing=recipe)
        runtime = PolicyRuntime(nn.Linear(1, 1), info, seed=7)
        rgb = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)[:, :, ::-1]
        actual = runtime.prepare_observations({"rgb": rgb})["rgb"]
        expected = preprocess_rgb_eval(rgb.copy(), **recipe).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (1, 2, 3, 4, 5))
        for wrong in (rgb.transpose(0, 3, 1, 2), rgb.astype(np.float32), np.zeros((2, 0, 5, 3), np.uint8)):
            with self.assertRaisesRegex(ValueError, "RGB input requires"):
                runtime.prepare_observations({"rgb": wrong})


if __name__ == "__main__":
    unittest.main()
