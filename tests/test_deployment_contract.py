from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.datasets.base_dataset import BaseDataset, preprocess_validation_rgb
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
    DeploymentContractError,
    parse_deployment_contract,
)
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    RestoredDeployment,
    _validate_rgb_processor,
    deterministic_observation,
    prediction_snapshot,
    prepare_deployment_observation,
)
from dexmani_policy.deployment.qualify import (
    DirectRestoredPolicy,
    direct_prediction_snapshot,
)


class _CapturingAgent:
    def predict_action(self, observation, denoise_timesteps):
        self.observation = observation
        prediction = torch.zeros((1, 16, 19), dtype=torch.float32)
        return {
            "pred_action": prediction,
            "control_action": prediction[:, 1:9],
        }


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
                "eval": {
                    "denoise_steps": 10,
                    "temporal_ensemble_coeff": 0.01,
                },
                "rgb_preprocessing": {
                    "input_layout": "HWC",
                    "input_dtype": "uint8",
                    "input_color_order": "rgb",
                    "input_value_range": [0, 255],
                    "execution_device": "cpu",
                    "resize_hw": [240, 240],
                    "center_crop_hw": [224, 224],
                    "interpolation": "bilinear",
                    "antialias": True,
                    "output_layout": "CHW",
                    "output_dtype": "float32",
                    "scale": 1.0 / 255.0,
                    "output_value_range": [0, 1],
                    "processor_image_size_hw": [224, 224],
                    "processor_interpolation": "bilinear",
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
        self.assertEqual(spec.temporal_ensemble_coeff, 0.01)
        self.assertTrue(spec.requires_hand)
        self.assertEqual(spec.rgb_preprocessing.resize_hw, (240, 240))

    def test_temporal_ensemble_coefficient_is_required_and_validated(self) -> None:
        payload = copy.deepcopy(_payload())
        del payload["contract"]["inference_config"]["eval"]["temporal_ensemble_coeff"]
        with self.assertRaisesRegex(DeploymentContractError, "is required"):
            parse_deployment_contract(payload)

        for invalid in (True, float("inf"), -0.01):
            payload = copy.deepcopy(_payload())
            payload["contract"]["inference_config"]["eval"][
                "temporal_ensemble_coeff"
            ] = invalid
            with self.assertRaises(DeploymentContractError):
                parse_deployment_contract(payload)

        payload = copy.deepcopy(_payload())
        payload["contract"]["inference_config"]["eval"][
            "temporal_ensemble_coeff"
        ] = None
        self.assertIsNone(parse_deployment_contract(payload).temporal_ensemble_coeff)

    def test_legacy_format_is_not_accepted(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["_format"] = "dexmani.deployment.v2"
        payload["contract"]["schema_version"] = 2
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

    def test_dp_validation_and_deployment_rgb_are_numerically_identical(self) -> None:
        raw = (
            torch.arange(2 * 480 * 640 * 3, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(2, 480, 640, 3)
        )
        dataset = object.__new__(BaseDataset)
        dataset._is_val = True
        dataset.rgb_preprocess_size = (240, 240)
        dataset.rgb_random_crop_size = (224, 224)
        dataset.rgb_keep_uint8 = False
        dataset.rgb_color_aug = {"training_only": True}

        validation_rgb = dataset._preprocess_rgb_cpu(raw.numpy())
        deployment_rgb = prepare_deployment_observation(
            {
                "joint_state": torch.zeros((1, 2, 19), dtype=torch.float32),
                "rgb": raw.unsqueeze(0),
            },
            parse_deployment_contract(_payload()),
        )["rgb"]

        self.assertEqual(deployment_rgb.shape, (1, 2, 3, 224, 224))
        self.assertEqual(deployment_rgb.dtype, torch.float32)
        torch.testing.assert_close(deployment_rgb[0], validation_rgb, rtol=0, atol=0)

    def test_training_rgb_random_crop_and_color_augmentation_remain_active(
        self,
    ) -> None:
        calls = []

        def color_aug(value):
            calls.append(tuple(value.shape))
            return value

        dataset = object.__new__(BaseDataset)
        dataset._is_val = False
        dataset.rgb_preprocess_size = (24, 24)
        dataset.rgb_random_crop_size = (16, 16)
        dataset.rgb_keep_uint8 = False
        dataset.rgb_color_aug = color_aug
        raw = torch.zeros((2, 30, 40, 3), dtype=torch.uint8).numpy()

        result = dataset._preprocess_rgb_cpu(raw)

        self.assertEqual(result.shape, (2, 3, 16, 16))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(calls, [(2, 3, 16, 16)])

    def test_no_resize_preserves_raw_hwc_uint8(self) -> None:
        payload = copy.deepcopy(_payload())
        metadata = payload["contract"]["inference_config"]["rgb_preprocessing"]
        metadata.update(
            {
                "resize_hw": None,
                "center_crop_hw": None,
                "output_layout": "HWC",
                "output_dtype": "uint8",
                "scale": 1.0,
                "output_value_range": [0, 255],
            }
        )
        spec = parse_deployment_contract(payload)
        raw = deterministic_observation(spec)
        prepared = prepare_deployment_observation(raw, spec)

        self.assertEqual(prepared["rgb"].shape, (1, 2, 480, 640, 3))
        self.assertEqual(prepared["rgb"].dtype, torch.uint8)
        torch.testing.assert_close(prepared["rgb"], raw["rgb"], rtol=0, atol=0)

    def test_image_processor_accepts_each_validation_output_semantics(self) -> None:
        raw = torch.arange(2 * 30 * 40 * 3, dtype=torch.int64)
        raw = raw.remainder(256).to(torch.uint8).reshape(2, 30, 40, 3)
        processor = ImageProcessor.from_preset("dino")

        no_resize = preprocess_validation_rgb(
            raw, resize_hw=None, center_crop_hw=None, keep_uint8=False
        )
        resized_float = preprocess_validation_rgb(
            raw, resize_hw=(24, 24), center_crop_hw=(16, 16), keep_uint8=False
        )
        resized_uint8 = preprocess_validation_rgb(
            raw, resize_hw=(24, 24), center_crop_hw=(16, 16), keep_uint8=True
        )

        self.assertEqual(no_resize.shape, (2, 30, 40, 3))
        self.assertEqual(no_resize.dtype, torch.uint8)
        self.assertEqual(resized_float.shape, (2, 3, 16, 16))
        self.assertEqual(resized_float.dtype, torch.float32)
        self.assertEqual(resized_uint8.shape, (2, 3, 16, 16))
        self.assertEqual(resized_uint8.dtype, torch.uint8)
        for value in (no_resize, resized_float, resized_uint8):
            processed = processor.process_images(value)["image"]
            self.assertEqual(processed.shape, (2, 3, 224, 224))
            self.assertEqual(processed.dtype, torch.float32)

    def test_direct_and_deployment_predictions_receive_validation_rgb(self) -> None:
        spec = parse_deployment_contract(_payload())
        observation = deterministic_observation(spec)
        dataset = object.__new__(BaseDataset)
        dataset._is_val = True
        dataset.rgb_preprocess_size = (240, 240)
        dataset.rgb_random_crop_size = (224, 224)
        dataset.rgb_keep_uint8 = False
        dataset.rgb_color_aug = {"training_only": True}
        expected = dataset._preprocess_rgb_cpu(observation["rgb"][0].numpy())

        direct_agent = _CapturingAgent()
        direct = DirectRestoredPolicy(
            agent=direct_agent,
            spec=spec,
            experiment_dir=Path("/experiment"),
            checkpoint_path=Path("/experiment/selected.pt"),
            checkpoint_selector="best",
            use_ema=False,
            selected_weights="model",
        )
        deployment_agent = _CapturingAgent()
        deployment = RestoredDeployment(agent=deployment_agent, spec=spec)

        direct_prediction_snapshot(direct, observation=observation)
        prediction_snapshot(deployment, observation=observation)

        torch.testing.assert_close(
            direct_agent.observation["rgb"][0], expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            deployment_agent.observation["rgb"][0], expected, rtol=0, atol=0
        )

    def test_rgb_contract_rejects_missing_or_conflicting_chain_metadata(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["contract"]["inference_config"]["rgb_preprocessing"].pop(
            "processor_image_size_hw"
        )
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

        payload = copy.deepcopy(_payload())
        payload["contract"]["inference_config"]["rgb_preprocessing"]["scale"] = 1.0
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

        payload = copy.deepcopy(_payload())
        payload["contract"]["data_contract"]["observation_fields"]["rgb"]["semantics"][
            "color_order"
        ] = "bgr"
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(payload)

    def test_restored_image_processor_must_match_recorded_second_stage(self) -> None:
        processor = SimpleNamespace(
            image_size=(224, 224),
            interpolation="bilinear",
            image_mean=torch.tensor([0.485, 0.456, 0.406]),
            image_std=torch.tensor([0.229, 0.224, 0.225]),
        )
        agent = SimpleNamespace(obs_encoder=SimpleNamespace(image_processor=processor))
        spec = parse_deployment_contract(_payload())
        _validate_rgb_processor(agent, spec)

        processor.interpolation = "bicubic"
        with self.assertRaises(DeploymentRestoreError):
            _validate_rgb_processor(agent, spec)


if __name__ == "__main__":
    unittest.main()
