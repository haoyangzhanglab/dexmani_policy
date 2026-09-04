from __future__ import annotations

import unittest

import torch

from dexmani_policy.deployment import export
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
)


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


if __name__ == "__main__":
    unittest.main()
