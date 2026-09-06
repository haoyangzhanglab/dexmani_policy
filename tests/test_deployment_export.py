from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from dexmani_policy.deployment import export, qualify
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
)


class DeploymentExportTest(unittest.TestCase):
    @staticmethod
    def _real_zarr_attrs() -> dict[str, object]:
        return {
            "schema_name": "dexmani-real-policy-zarr",
            "schema_version": 7,
            "domain": "real",
            "profile": "joint",
            "task_name": "task",
            "dt": 0.1,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_reference": "grid_anchor_monotonic_ns",
            "state_alignment": "control_grid_state",
            "action_semantics": "teleop_published_joint_target",
            "fingertip_points_frame": "xarm_base",
            "fingertip_points_unit": "m",
            "fingertip_points_derivation": "fk_from_processed_joint_state",
            "fingertip_points_policy_id": "arm_hand_fk_from_joint_state_v1",
            "fingertip_points_geometry_sha256": "b" * 64,
        }

    @staticmethod
    def _write_required_arrays(
        root: object, *, point_cloud: bool = False, fingertip: bool = False
    ) -> None:
        data = root.create_group("data")
        data.create_dataset("joint_state", data=np.zeros((1, 19), np.float32))
        data.create_dataset("action", data=np.zeros((1, 19), np.float32))
        data.create_dataset("action_ee", data=np.zeros((1, 21), np.float32))
        if point_cloud:
            data.create_dataset("point_cloud", data=np.zeros((1, 1024, 6), np.float32))
        if fingertip:
            data.create_dataset(
                "fingertip_points", data=np.zeros((1, 5, 3), np.float32)
            )

    def test_real_zarr_v7_contract_keeps_teleop_action_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = export.zarr.open_group(str(path), mode="w")
            root.attrs.update(self._real_zarr_attrs())
            self._write_required_arrays(root)

            contract = export._build_observation_contract(
                path,
                {"task_name": "task", "action_key": "action", "dt": 0.1},
                ["joint_state"],
            )

        self.assertEqual(contract["action_semantics"], "teleop_published_joint_target")
        self.assertNotIn("deployment_equivalent", contract)

    def test_real_zarr_v6_contract_is_rejected(self) -> None:
        attrs = self._real_zarr_attrs()
        attrs["schema_version"] = 6
        with self.assertRaises(export.InvalidZarrError):
            export._validate_core_zarr_attrs(attrs, {"task_name": "task"}, [])

    def test_point_cloud_contract_copies_validated_zarr_identity(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = export.zarr.open_group(str(path), mode="w")
            attrs = self._real_zarr_attrs()
            attrs.update(export._POINT_SEMANTICS)
            attrs.update(
                {
                    "profile": "pointcloud",
                    "observation_reference": "camera_source_monotonic_ns",
                    "state_alignment": "camera_source_aligned_state",
                    "point_cloud_config_sha256": "a" * 64,
                    "point_cloud_table_plane_abcd_json": "null",
                }
            )
            root.attrs.update(attrs)
            self._write_required_arrays(root, point_cloud=True)

            contract = export._build_observation_contract(
                path,
                {
                    "task_name": "task",
                    "action_key": "action",
                    "dt": 0.1,
                    "agent": {"num_points": 1024, "pc_dim": 6},
                },
                ["joint_state", "point_cloud"],
            )

        semantics = contract["observation_fields"]["point_cloud"]["semantics"]
        self.assertEqual(
            semantics,
            {
                "representation": "xyzrgb",
                "frame": "xarm_base",
                "position_units": "m",
                "color_order": "rgb",
                "color_source": export._POINT_SEMANTICS["point_cloud_color_source"],
                "policy_id": export._POINT_SEMANTICS["point_cloud_policy_id"],
                "config_sha256": "a" * 64,
                "table_plane_abcd_json": "null",
                "sampling": export._POINT_SEMANTICS["point_cloud_sampling"],
                "transform": export._POINT_SEMANTICS["point_cloud_transform"],
            },
        )

    def test_fingertip_contract_copies_frozen_identity(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = export.zarr.open_group(str(path), mode="w")
            root.attrs.update(self._real_zarr_attrs())
            self._write_required_arrays(root, fingertip=True)

            contract = export._build_observation_contract(
                path,
                {"task_name": "task", "action_key": "action", "dt": 0.1},
                ["joint_state", "fingertip_points"],
            )

        self.assertEqual(
            contract["observation_fields"]["fingertip_points"]["semantics"],
            {
                "representation": "point_xyz",
                "frame": "xarm_base",
                "units": "m",
                "finger_order": "thumb_index_mid_ring_pinky",
                "derivation": "fk_from_processed_joint_state",
                "policy_id": "arm_hand_fk_from_joint_state_v1",
                "geometry_sha256": "b" * 64,
            },
        )

    def test_fingertip_contract_rejects_each_missing_identity_attr(self) -> None:
        for key in (
            "fingertip_points_derivation",
            "fingertip_points_policy_id",
            "fingertip_points_geometry_sha256",
        ):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                path = Path(directory) / "task.zarr"
                root = export.zarr.open_group(str(path), mode="w")
                attrs = self._real_zarr_attrs()
                attrs.pop(key)
                root.attrs.update(attrs)
                self._write_required_arrays(root, fingertip=True)
                with self.assertRaisesRegex(export.InvalidZarrError, key):
                    export._build_observation_contract(
                        path,
                        {"task_name": "task", "action_key": "action", "dt": 0.1},
                        ["joint_state", "fingertip_points"],
                    )

    def test_payload_has_one_format_and_one_observation_contract(self) -> None:
        inference = {
            "task_name": "task",
            "action_key": "action",
            "action_dim": 19,
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "use_aux_ee": False,
            "agent": {},
            "eval": {
                "denoise_steps": 10,
                "temporal_ensemble_coeff": None,
            },
        }
        data = {
            "dt": 0.1,
            "requires_hand": True,
            "observation_fields": {
                "joint_state": {"shape": [19], "dtype": "float32"},
                "point_cloud": {"shape": [4, 6], "dtype": "float32"},
            },
        }
        payload = {
            "_format": DEPLOYMENT_FORMAT,
            "contract": {
                "schema_version": DEPLOYMENT_SCHEMA_VERSION,
                "inference_config": inference,
                "data_contract": data,
                "producer": {},
            },
            "weights": {"weight": torch.ones(1)},
        }

        export._validate_payload(payload)
        self.assertNotIn("sensor_modalities", data)
        self.assertNotIn("normalizer_keys", data)
        self.assertEqual(set(payload), {"_format", "contract", "weights"})
        self.assertEqual(payload["contract"]["inference_config"], inference)
        self.assertEqual(payload["weights"], {"weight": torch.ones(1)})

    def test_selected_inference_resolver_uses_best_record_or_config(self) -> None:
        config = {
            "eval": {
                "use_ema": False,
                "denoise_steps": 3,
                "denoise_timesteps_list": None,
            },
            "env_runner": {"temporal_ensemble_coeff": 0.4},
        }
        with TemporaryDirectory() as directory:
            experiment = Path(directory)
            checkpoint = experiment / "checkpoints" / "epoch-10.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            record = {
                "record_version": 2,
                "ckpt_relpath": "checkpoints/epoch-10.pt",
                "pct": 100,
                "global_step": 10,
                "success_rate": 1.0,
                "avg_steps": 4.0,
                "n_episodes": 2,
                "inference": {
                    "use_ema": True,
                    "denoise_steps": 7,
                    "temporal_ensemble_coeff": 0.2,
                    "policy_seed_mode": "episode_seed",
                },
                "selection": {
                    "shuffle_seed": 0,
                    "seeds": [1, 2],
                    "initial_episodes": 2,
                    "tie_break_used": False,
                },
            }
            (experiment / "best_ckpt.json").write_text(json.dumps(record))

            best = export._resolve_selected_inference_settings(
                experiment, "best", config
            )
            latest = export._resolve_selected_inference_settings(
                experiment, "latest", config
            )

        self.assertEqual((best.use_ema, best.denoise_steps), (True, 7))
        self.assertEqual(best.temporal_ensemble_coeff, 0.2)
        self.assertEqual((latest.use_ema, latest.denoise_steps), (False, 3))
        self.assertEqual(latest.temporal_ensemble_coeff, 0.4)

    def test_export_and_direct_restore_call_the_shared_resolver(self) -> None:
        marker = RuntimeError("shared resolver called")
        config = {"agent": {}}
        with TemporaryDirectory() as directory:
            experiment = Path(directory)
            checkpoint = experiment / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            with (
                mock.patch.object(
                    export, "_producer_provenance", return_value="0" * 40
                ),
                mock.patch.object(
                    export, "_resolve_checkpoint", return_value=checkpoint
                ),
                mock.patch.object(export, "_load_config", return_value=config),
                mock.patch.object(
                    export,
                    "_resolve_selected_inference_settings",
                    side_effect=marker,
                ) as resolver,
            ):
                with self.assertRaisesRegex(RuntimeError, "shared resolver called"):
                    export.export_deployment_artifact(experiment, "latest")
                resolver.assert_called_once_with(experiment, "latest", config)

            with (
                mock.patch.object(
                    export, "_resolve_checkpoint", return_value=checkpoint
                ),
                mock.patch.object(export, "_load_config", return_value=config),
                mock.patch.object(
                    export,
                    "_resolve_selected_inference_settings",
                    side_effect=marker,
                ) as resolver,
            ):
                with self.assertRaisesRegex(RuntimeError, "shared resolver called"):
                    qualify.restore_direct_policy(
                        experiment, checkpoint_selector="latest"
                    )
                resolver.assert_called_once_with(experiment, "latest", config)

    def test_dp_rgb_preprocessing_records_validation_and_processor_stages(self) -> None:
        metadata = export._rgb_preprocessing(
            {
                "rgb_backbone_name": "dino",
                "rgb_backbone_config": {"image_size": [224, 224]},
            },
            {
                "_target_": "dexmani_policy.datasets.rgb_dataset.RGBDataset",
                "rgb_preprocess_size": [240, 240],
                "rgb_random_crop_size": [224, 224],
                "rgb_color_aug": {"_target_": "training.only"},
            },
        )

        self.assertEqual(metadata["resize_hw"], [240, 240])
        self.assertEqual(metadata["center_crop_hw"], [224, 224])
        self.assertTrue(metadata["antialias"])
        self.assertEqual(metadata["output_dtype"], "float32")
        self.assertEqual(metadata["output_value_range"], [0, 1])
        self.assertEqual(metadata["processor_image_size_hw"], [224, 224])
        self.assertEqual(metadata["processor_interpolation"], "bilinear")

    def test_rgb_export_rejects_incomplete_or_ineffective_dataset_transform(
        self,
    ) -> None:
        agent = {
            "rgb_backbone_name": "dino",
            "rgb_backbone_config": {"image_size": [224, 224]},
        }
        with self.assertRaises(export.InvalidExperimentError):
            export._rgb_preprocessing(
                agent,
                {
                    "_target_": "dexmani_policy.datasets.rgb_dataset.RGBDataset",
                    "rgb_random_crop_size": [224, 224],
                },
            )
        with self.assertRaises(export.InvalidExperimentError):
            export._rgb_preprocessing(
                agent,
                {
                    "_target_": "dexmani_policy.datasets.rgb_dataset.RGBDataset",
                    "rgb_preprocess_size": None,
                    "rgb_random_crop_size": [224, 224],
                },
            )


if __name__ == "__main__":
    unittest.main()
