from __future__ import annotations

import unittest

import torch

from dexmani_policy.deployment import export
from dexmani_policy.deployment.contract import DEPLOYMENT_FORMAT


class DeploymentExportTest(unittest.TestCase):
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
            "eval": {"use_ema": False, "denoise_steps": 10},
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
            "state": {
                "epoch": 1,
                "global_step": 2,
                "train_params": {},
                "inference_config": inference,
                "data_contract": data,
                "producer": {},
            },
            "weights": {"model": {"weight": torch.ones(1)}, "ema_model": None},
        }

        export._validate_payload(payload)
        self.assertNotIn("sensor_modalities", data)
        self.assertNotIn("normalizer_keys", data)
        self.assertNotIn("deployment_contract", payload["state"])

        allocation = export._build_allocation(inference, data)
        self.assertEqual(
            allocation["observation_fields"], ["joint_state", "point_cloud"]
        )
        self.assertEqual(allocation["observation_specs"], data["observation_fields"])


if __name__ == "__main__":
    unittest.main()
