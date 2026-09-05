from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dexmani_policy.deployment import export
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
)


class DeploymentExportTest(unittest.TestCase):
    @staticmethod
    def _real_zarr_attrs() -> dict[str, object]:
        return {
            "schema_name": "dexmani-real-policy-zarr",
            "schema_version": 6,
            "domain": "real",
            "profile": "joint",
            "task_name": "task",
            "dt": 0.1,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_reference": "grid_anchor_monotonic_ns",
            "state_alignment": "control_grid_state",
            "action_semantics": "teleop_published_joint_target",
        }

    def test_real_zarr_v6_contract_keeps_teleop_action_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = export.zarr.open_group(str(path), mode="w")
            root.attrs.update(self._real_zarr_attrs())
            data = root.create_group("data")
            data.create_dataset("joint_state", data=np.zeros((1, 19), np.float32))
            data.create_dataset("action", data=np.zeros((1, 19), np.float32))
            data.create_dataset("action_ee", data=np.zeros((1, 21), np.float32))

            contract = export._build_observation_contract(
                path,
                {"task_name": "task", "action_key": "action", "dt": 0.1},
                ["joint_state"],
            )

        self.assertEqual(
            contract["action_semantics"], "teleop_published_joint_target"
        )
        self.assertNotIn("deployment_equivalent", contract)

    def test_real_zarr_v5_contract_is_rejected(self) -> None:
        attrs = self._real_zarr_attrs()
        attrs["schema_version"] = 5
        attrs["action_semantics"] = "deployment_grid_rate_limited_target"
        with self.assertRaises(export.InvalidZarrError):
            export._validate_core_zarr_attrs(attrs, {"task_name": "task"}, [])

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
            "eval": {"denoise_steps": 10},
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
