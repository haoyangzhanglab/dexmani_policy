import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.checkpoint import load_training_checkpoint
from dexmani_policy.common.config import normalization_dims_from_config
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.env_runner.base_runner import BaseRunner


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


if __name__ == "__main__":
    unittest.main()
