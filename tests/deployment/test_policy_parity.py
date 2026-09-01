"""Synthetic, real-agent coverage for direct/export deployment parity."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

import dexmani_policy.deployment.export as exporter
from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.deployment.qualify import (
    qualify_policy_parity,
    restore_direct_policy,
)

_COMMIT = "b" * 40

try:
    import pytorch3d.ops as torch3d_ops

    _ACTION_FLOW_POINT_OPS_AVAILABLE = all(
        hasattr(torch3d_ops, name)
        for name in ("sample_farthest_points", "knn_points", "ball_query")
    )
except ModuleNotFoundError:
    _ACTION_FLOW_POINT_OPS_AVAILABLE = False


def _zarr_attrs(task_name: str) -> dict[str, object]:
    return {
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 5,
        "domain": "real",
        "profile": "pointcloud",
        "task_name": task_name,
        "dt": 0.0625,
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_reference": "camera_source_monotonic_ns",
        "state_alignment": "camera_source_aligned_state",
        "max_observation_skew_s": 0.1,
        "action_semantics": "deployment_grid_rate_limited_target",
        "arm_max_delta_rad_per_tick": 0.1,
        "hand_max_delta_rad_per_tick": 0.3,
        "endpoint_delta_tolerance_rad": 1e-12,
        "deployment_equivalent": True,
        "point_cloud_frame": "xarm_base",
        "point_cloud_color_source": "mean_rgb_of_aligned_depth_pixels_per_voxel",
        "point_cloud_policy_id": "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9",
        "point_cloud_config_sha256": "a" * 64,
        "point_cloud_table_plane_abcd_json": "null",
        "point_cloud_sampling": "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad",
        "point_cloud_transform": (
            "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
            "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
            "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
            "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
            "coarse_voxel_stratified_hash_or_cyclic_pad"
        ),
    }


def _agent_config(policy: str, action_dim: int) -> dict[str, object]:
    common: dict[str, object] = {
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_dim": action_dim,
        "state_dim": 19,
        "pc_dim": 6,
        "num_points": 1024,
        "fps_random_config": {
            "use_random": True,
            "use_random_start": True,
            "random_noise_scale": 0.0,
            "use_shuffle_output": True,
        },
        "modality_dropout_probs": {"joint_state": 0.0},
    }
    if policy == "dp3":
        return {
            "_target_": "dexmani_policy.agents.core.dp3.DP3Agent",
            **common,
            "encoder_type": "dp3",
            "pc_out_dim": 8,
            "state_out_dim": 8,
            "diffusion_step_embed_dim": 16,
            "down_dims": [8],
            "kernel_size": 3,
            "n_groups": 8,
            "cond_predict_scale": True,
            "num_training_steps": 2,
            "num_inference_steps": 1,
            "prediction_type": "sample",
        }
    if policy == "action_flow":
        return {
            "_target_": "dexmani_policy.agents.core.action_flow.ActionFlowAgent",
            **common,
            "pc_encoder_config": {
                "num_patches": 2,
                "stem_channels": 8,
                "token_channels": 12,
                "patch_radii": [0.04, 0.08],
                "patch_neighbors": [4, 8],
                "use_patch_self_attn": False,
            },
            # GeoFormer uses 3-axis RoPE, so each 12-D attention head is a
            # valid 6*k width while retaining a genuinely tiny CPU model.
            "geo_hidden_dim": 24,
            "geo_depth": 1,
            "geo_num_heads": 2,
            "geo_ffn_hidden_dim": 32,
            "geo_qk_norm": True,
            "geo_use_3d_rope": True,
            "geo_attn_drop": 0.0,
            "geo_drop_path": 0.0,
            "hidden_dim": 24,
            "context_dim": 24,
            "depth": 1,
            "num_heads": 2,
            "ffn_hidden_dim": 32,
            "timestep_embed_dim": 8,
            "step_embed_dim": 8,
            "state_embed_hidden_dim": 16,
            "cond_bottleneck_dim": 12,
            "qk_norm": True,
            "attn_drop": 0.0,
            "use_step_conditioning": False,
            "denoise_steps": 1,
            "noise_shift_alpha": 3.0,
            "noise_shift_ratio": 0.75,
            "solver": "euler",
        }
    raise ValueError(f"unknown policy fixture: {policy}")


def _config(policy: str, action_key: str, *, use_ema: bool) -> dict[str, object]:
    action_dim = 21 if action_key == "action_ee" else 19
    task_name = f"tiny_{policy}_{action_key}"
    return {
        "policy_name": policy,
        "task_name": task_name,
        "zarr_path": "not-used-by-test.zarr",
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_key": action_key,
        "action_dim": action_dim,
        "use_aux_ee": False,
        # The direct restore must leave this training-only object untouched.
        "dataset": {
            "_target_": "dexmani_policy.datasets.pc_dataset.PCDataset",
            "sensor_modalities": ["joint_state", "point_cloud"],
            "action_key": action_key,
            "horizon": 4,
            "obs_horizon": 2,
            "pad_before": 1,
            "pad_after": 1,
            "use_aux_ee": False,
        },
        "agent": _agent_config(policy, action_dim),
        "eval": {
            "use_ema": use_ema,
            "denoise_steps": 1,
            "denoise_timesteps_list": None,
        },
    }


def _normalizer(action_dim: int) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.linspace(-0.9, 0.9, 2 * 4 * action_dim).reshape(
                2, 4, action_dim
            ),
            "joint_state": torch.linspace(-0.8, 0.8, 2 * 2 * 19).reshape(2, 2, 19),
            "point_cloud": torch.linspace(-0.7, 0.7, 2 * 2 * 1024 * 6).reshape(
                2, 2, 1024, 6
            ),
        }
    )
    return normalizer


def _train_params(action_key: str, action_dim: int) -> dict[str, object]:
    return {
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_dim": action_dim,
        "horizon": 4,
        "action_key": action_key,
        "tcp_dim": None,
        "hand_dim": None,
        "control_action_dim": action_dim,
        "num_training_steps": 2,
        "use_aux_ee": False,
    }


def _write_zarr(path: Path, task_name: str) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(_zarr_attrs(task_name))
    data = root.create_group("data")
    data.create_dataset(
        "point_cloud",
        data=np.zeros((2, 1024, 6), dtype=np.float32),
        shape=(2, 1024, 6),
    )
    data.create_dataset(
        "joint_state", data=np.zeros((2, 19), dtype=np.float32), shape=(2, 19)
    )
    data.create_dataset(
        "action", data=np.zeros((2, 19), dtype=np.float32), shape=(2, 19)
    )
    data.create_dataset(
        "action_ee", data=np.zeros((2, 21), dtype=np.float32), shape=(2, 21)
    )


def _write_experiment(
    root: Path, policy: str, action_key: str, *, use_ema: bool
) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    import hydra

    cfg = _config(policy, action_key, use_ema=use_ema)
    action_dim = int(cfg["action_dim"])
    agent = hydra.utils.instantiate(OmegaConf.create(cfg["agent"]))
    agent.action_key = action_key
    agent.load_normalizer_from_dataset(_normalizer(action_dim))
    model_state = dict(agent.state_dict())
    ema_state = copy.deepcopy(model_state)
    if use_ema:
        # Ensure the selected EMA branch is observably distinct without changing
        # any architecture or substituting a fake policy.
        first_float_key = next(
            key for key, value in ema_state.items() if value.is_floating_point()
        )
        ema_state[first_float_key] = ema_state[first_float_key] + 0.03125

    experiment = root / "experiment"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), experiment / "config.yaml")
    checkpoint_path = CheckpointStore(checkpoint_dir).save(
        "epoch=0000-step=00000000-100pct.pt",
        TrainCheckpoint(
            epoch=0,
            global_step=0,
            model_state=model_state,
            ema_model_state=ema_state if use_ema else None,
            optimizer_state={},
            scheduler_state={},
            monitor={},
            train_params=_train_params(action_key, action_dim),
        ),
    )
    (checkpoint_dir / "latest.pt").symlink_to(checkpoint_path.name)
    zarr_path = root / "policy.zarr"
    _write_zarr(zarr_path, str(cfg["task_name"]))
    return experiment, zarr_path, ema_state


class PolicyParityTest(unittest.TestCase):
    maxDiff = None

    def _assert_parity(self, policy: str, action_key: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, zarr_path, _ = _write_experiment(
                Path(directory), policy, action_key, use_ema=True
            )
            with patch.object(exporter, "_producer_provenance", return_value=_COMMIT):
                report = qualify_policy_parity(
                    experiment,
                    checkpoint_selector="latest",
                    zarr_path=zarr_path,
                    # Qualification itself always safe-reloads and strictly
                    # restores deployment-v2.  Skip the exporter's duplicate
                    # preflight to keep this four-case integration test small.
                    verify_export=False,
                    seed=19,
                    atol=0.0,
                    rtol=0.0,
                )
            expected_source_sha = _sha256_file(Path(report.selected_checkpoint))
        expected_dim = 21 if action_key == "action_ee" else 19
        self.assertEqual(report.action_key, action_key)
        self.assertEqual(report.action_dim, expected_dim)
        self.assertEqual(report.control_action_dim, expected_dim)
        self.assertTrue(report.use_ema)
        self.assertEqual(report.selected_weights, "ema_model")
        self.assertEqual(report.max_abs_diff, 0.0)
        self.assertEqual(len(report.source_checkpoint_sha256), 64)
        self.assertEqual(len(report.deployment_checkpoint_sha256), 64)
        self.assertEqual(
            report.source_checkpoint_sha256,
            expected_source_sha,
        )

    def test_tiny_actual_dp3_joint_and_ee_parity(self) -> None:
        """DP3 qualifies in both control spaces with its genuine agent class."""
        for action_key in ("action", "action_ee"):
            with self.subTest(action_key=action_key):
                self._assert_parity("dp3", action_key)

    @unittest.skipUnless(
        _ACTION_FLOW_POINT_OPS_AVAILABLE,
        "ActionFlow parity needs pytorch3d sample_farthest_points/knn_points/ball_query",
    )
    def test_tiny_actual_actionflow_joint_and_ee_parity(self) -> None:
        """ActionFlow retains state_dim=19 for joint and EE action layouts."""
        for action_key in ("action", "action_ee"):
            with self.subTest(action_key=action_key):
                self._assert_parity("action_flow", action_key)

    def test_direct_restore_uses_ema_and_never_instantiates_dataset_or_runner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, _, ema_state = _write_experiment(
                Path(directory), "dp3", "action", use_ema=True
            )
            import hydra

            real_instantiate = hydra.utils.instantiate
            seen_targets: list[str] = []

            def guarded_instantiate(config, *args, **kwargs):
                target = config.get("_target_", "")
                seen_targets.append(target)
                if (
                    "datasets." in target
                    or "env_runner." in target
                    or "dexmani_sim" in target
                ):
                    raise AssertionError(
                        f"direct restore instantiated forbidden target: {target}"
                    )
                return real_instantiate(config, *args, **kwargs)

            with patch("hydra.utils.instantiate", side_effect=guarded_instantiate):
                direct = restore_direct_policy(experiment, checkpoint_selector="latest")
            self.assertEqual(
                seen_targets,
                ["dexmani_policy.agents.core.dp3.DP3Agent"],
            )
            first_float_key = next(
                key for key, value in ema_state.items() if value.is_floating_point()
            )
            torch.testing.assert_close(
                direct.agent.state_dict()[first_float_key], ema_state[first_float_key]
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
