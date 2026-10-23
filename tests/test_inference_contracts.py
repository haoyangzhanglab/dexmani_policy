import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.checkpoint import inference_recipe, load_training_checkpoint
from dexmani_policy.common.config import (
    normalization_dims_from_config,
    validate_depth_recipe,
    validate_source_recipe,
)
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.training.builder import validate_hand_pretraining


class InferenceContractTest(unittest.TestCase):
    def test_runner_requires_explicit_modalities_and_pads_history(self):
        with self.assertRaises(TypeError):
            BaseRunner(n_obs_steps=2)
        modalities = ["joint_state"]
        runner = BaseRunner(n_obs_steps=2, sensor_modalities=modalities)
        modalities.append("rgb")
        runner.update_obs({"joint_state": np.array([3.0], dtype=np.float32)})
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"], [[3], [3]])
        runner.update_obs({"joint_state": np.array([4.0], dtype=np.float32)})
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"], [[3], [4]])

    def test_policy_requires_recorded_action_width(self):
        policy = BasePolicy(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 2, 19)
        spec = {"action": "limits"}
        with self.assertRaises(TypeError):
            policy.set_normalization_spec(spec)
        with self.assertRaises(KeyError):
            policy.set_normalization_spec(spec, {})
        with self.assertRaisesRegex(ValueError, "action width"):
            policy.set_normalization_spec(spec, {"action": 21})
        policy.set_normalization_spec(spec, {"action": 19})
        self.assertEqual(policy.normalization_dims, {"action": 19})

    def test_multitask_normalization_uses_child_point_channels(self):
        cfg = {
            "agent": {"action_dim": 19},
            "dataset": {"datasets": [{"pc_dim": 3}, {"pc_dim": 3}]},
            "normalization": {"point_cloud": "limits", "action": "limits"},
            "normalization_dims": {"point_cloud": 6, "action": 19},
        }
        with self.assertRaisesRegex(ValueError, "point_cloud feature width"):
            normalization_dims_from_config(cfg)
        cfg["normalization_dims"]["point_cloud"] = 3
        self.assertEqual(normalization_dims_from_config(cfg)["point_cloud"], 3)

    def test_metadata_requires_saved_modalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "checkpoints").mkdir()
            (directory / "checkpoints/latest.pt").touch()
            cfg = OmegaConf.create({
                "policy_name": "dp3",
                "task_name": "test",
                "action_key": "action",
                "agent": {
                    "_target_": "dexmani_policy.agents.policies.dp3.DP3Policy",
                    "horizon": 4, "n_obs_steps": 2, "n_action_steps": 2, "action_dim": 19,
                },
                "dataset": {"sensor_modalities": ["joint_state", "point_cloud"]},
            })
            OmegaConf.save(cfg, directory / "config.yaml")
            info = inspect_policy(directory, checkpoint="latest", weights="raw", inference_steps=1)
            self.assertEqual(info.observation_fields, ("joint_state", "point_cloud"))
            del cfg.dataset.sensor_modalities
            OmegaConf.save(cfg, directory / "config.yaml")
            with self.assertRaises(KeyError):
                inspect_policy(directory, checkpoint="latest", weights="raw", inference_steps=1)

    def test_training_checkpoint_does_not_migrate_legacy_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            torch.save({"state_dict": {}, "epoch": 0}, path)
            with self.assertRaises(KeyError):
                load_training_checkpoint(path)

    def test_inference_control_period_comes_only_from_real_runtime(self):
        cfg = {
            "agent": {"action_dim": 19},
            "dataset": {"sensor_modalities": ["joint_state"]},
            "normalization": {"joint_state": "limits", "action": "limits"},
            "normalization_dims": {"joint_state": 19, "action": 19},
            "action_key": "action",
            "hand_data": {"dt": 0.04},
            "real_runtime": {"dt": 0.02},
        }
        expected = inference_recipe(cfg)
        cfg["hand_data"]["dt"] = 0.05
        self.assertEqual(inference_recipe(cfg), expected)
        cfg["real_runtime"]["dt"] = 0.03
        self.assertNotEqual(inference_recipe(cfg), expected)

    def test_hand_data_is_checked_at_fresh_and_resume_boundaries(self):
        saved = {"hand_data": {"dt": 0.02}}
        current = {"dt": 0.04}
        model = SimpleNamespace(requires_hand_pretraining=True, hand_data=saved["hand_data"])
        cfg = OmegaConf.create(saved)
        with patch("dexmani_policy.training.builder.describe_hand_data", return_value=current):
            with self.assertRaisesRegex(ValueError, "Hand pretraining data mismatch"):
                validate_hand_pretraining(cfg, model, object())
            validate_hand_pretraining(cfg, model, object(), checkpoint=object())
        with self.assertRaisesRegex(ValueError, "hand_data.dt"):
            validate_source_recipe(saved, cfg)

    def test_depth_recipe_requires_numerical_fields_without_restricting_metadata(self):
        recipe = {
            "shape": [2, 3], "dtype": "uint16", "unit": "raw_counts",
            "depth_scale_m": 0.001, "sensor_name": "depth_camera",
        }
        self.assertEqual(validate_depth_recipe(recipe), recipe)
        del recipe["depth_scale_m"]
        with self.assertRaises(KeyError):
            validate_depth_recipe(recipe)


if __name__ == "__main__":
    unittest.main()
