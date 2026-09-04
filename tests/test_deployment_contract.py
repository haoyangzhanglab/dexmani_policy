from __future__ import annotations

import copy
import unittest

import torch

from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
    DeploymentContractError,
    parse_deployment_contract,
)
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    deterministic_observation,
    prepare_deployment_observation,
)


def _payload() -> dict:
    return {
        "_format": DEPLOYMENT_FORMAT,
        "contract": {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "inference_config": {
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "eval": {"denoise_steps": 10},
                "rgb_preprocessing": {
                    "input_color_order": "rgb",
                    "input_value_range": [0, 255],
                    "resize_hw": [224, 224],
                    "center_crop_hw": None,
                    "interpolation": "bicubic",
                    "antialias": False,
                    "output_layout": "CHW",
                    "output_dtype": "float32",
                    "scale": 1.0 / 255.0,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
            },
            "data_contract": {
                "dt": 0.1,
                "requires_hand": True,
                "observation_fields": {
                    "joint_state": {"shape": [19], "dtype": "float32"},
                    "rgb": {
                        "shape": [480, 640, 3],
                        "dtype": "uint8",
                        "semantics": {
                            "layout": "HWC",
                            "color_order": "rgb",
                            "value_range": [0, 255],
                        },
                    },
                },
            },
            "producer": {},
        },
        "weights": {"weight": torch.ones(1)},
    }


class DeploymentContractTest(unittest.TestCase):
    def test_single_observation_field_mapping_drives_spec(self) -> None:
        spec = parse_deployment_contract(_payload())
        self.assertEqual(
            tuple(field.name for field in spec.observation_fields),
            ("joint_state", "rgb"),
        )
        self.assertEqual(spec.observation_fields[1].shape, (480, 640, 3))
        self.assertEqual(spec.control_dt_s, 0.1)
        self.assertTrue(spec.requires_hand)
        self.assertEqual(spec.rgb_preprocessing.resize_hw, (224, 224))

    def test_legacy_format_is_not_accepted(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["_format"] = "dexmani.deployment"
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

    def test_shape_dtype_and_rgb_presence_are_boundary_checks(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["contract"]["data_contract"]["observation_fields"]["rgb"]["shape"] = []
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

        payload = copy.deepcopy(_payload())
        del payload["contract"]["data_contract"]["observation_fields"]["rgb"]
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

    def test_semantics_are_preserved_without_modality_whitelist_validation(
        self,
    ) -> None:
        payload = copy.deepcopy(_payload())
        payload["contract"]["data_contract"]["observation_fields"] = {
            "custom_signal": {
                "shape": [7],
                "dtype": "float32",
                "semantics": {"units": "domain_owned"},
            }
        }
        payload["contract"]["inference_config"].pop("rgb_preprocessing")
        spec = parse_deployment_contract(payload)
        self.assertEqual(spec.observation_fields[0].name, "custom_signal")
        self.assertEqual(spec.observation_fields[0].semantics["units"], "domain_owned")

    def test_observation_validation_is_driven_by_field_specs(self) -> None:
        spec = parse_deployment_contract(_payload())
        observation = deterministic_observation(spec, batch_size=2)
        self.assertEqual(observation["joint_state"].shape, (2, 2, 19))
        self.assertEqual(observation["rgb"].shape, (2, 2, 480, 640, 3))
        self.assertEqual(observation["rgb"].dtype, torch.uint8)
        prepare_deployment_observation(observation, spec)

        observation["joint_state"] = observation["joint_state"].double()
        with self.assertRaises(DeploymentRestoreError):
            prepare_deployment_observation(observation, spec)


if __name__ == "__main__":
    unittest.main()
