from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    PredictionParityError,
    PredictionSnapshot,
    assert_prediction_parity,
    deployment_spec,
    deterministic_observation,
    prediction_snapshot,
    reset_inference_seed,
    restore_deployment_agent,
)


def _payload(*, use_ema: bool = False) -> dict:
    return {
        "_format": "dexmani.deployment.v2",
        "state": {
            "inference_config": {
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "agent": {"_target_": "test.Agent"},
                "eval": {"use_ema": use_ema, "denoise_steps": 2},
            },
            "data_contract": {
                "point_cloud_num_points": 8,
                "point_cloud_feature_dim": 6,
            },
        },
        "weights": {
            "model": {"selected": torch.tensor([1.0])},
            "ema_model": {"selected": torch.tensor([2.0])} if use_ema else None,
        },
    }


class _Normalizer:
    def __init__(self, *, complete: bool = True) -> None:
        self.params_dict = {
            "action": {"scale": torch.ones(19), "offset": torch.zeros(19)},
            "joint_state": {"scale": torch.ones(19), "offset": torch.zeros(19)},
        }
        if complete:
            self.params_dict["point_cloud"] = {
                "scale": torch.ones(6),
                "offset": torch.zeros(6),
            }

    def is_fitted(self, required_keys):
        return all(key in self.params_dict for key in required_keys)


class _Agent:
    action_dim = 19
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    control_action_dim = 19

    def __init__(self, *, normalizer: _Normalizer | None = None) -> None:
        self.normalizer = normalizer or _Normalizer()
        self.loaded = None
        self.strict = False

    def load_state_dict(self, state, strict):
        self.loaded = state
        self.strict = strict
        if strict is not True:
            raise RuntimeError("restore must be strict")
        if set(state) != {"selected"}:
            raise RuntimeError("unexpected selected state")

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_action(self, observation, denoise_timesteps):
        value = self.loaded["selected"].item()
        pred = torch.full((1, 16, 19), value, dtype=torch.float32)
        return {"pred_action": pred, "control_action": pred[:, 1:9, :]}


class RestoreCoreTest(unittest.TestCase):
    def test_deterministic_observation_is_nonzero_bounded_and_seed_is_global(
        self,
    ) -> None:
        spec = deployment_spec(_payload())
        first = deterministic_observation(spec)
        second = deterministic_observation(spec)
        for key in ("joint_state", "point_cloud"):
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertTrue(torch.any(first[key] != 0))
            self.assertLess(float(torch.max(torch.abs(first[key]))), 1.0)

        reset_inference_seed(31)
        sample_a = (random.random(), np.random.rand(), torch.rand(1).item())
        reset_inference_seed(31)
        sample_b = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(sample_a, sample_b)

    def test_deployment_spec_requires_exact_format_marker(self) -> None:
        payload = _payload()
        del payload["_format"]
        with self.assertRaisesRegex(DeploymentRestoreError, "format"):
            deployment_spec(payload)

    def test_restore_uses_the_exact_selected_ema_weights_and_snapshots_prediction(
        self,
    ) -> None:
        agent = _Agent()
        with patch("hydra.utils.instantiate", return_value=agent):
            restored = restore_deployment_agent(_payload(use_ema=True))
        self.assertTrue(agent.strict)
        self.assertTrue(torch.equal(agent.loaded["selected"], torch.tensor([2.0])))

        snapshot = prediction_snapshot(restored, seed=4)
        self.assertEqual(tuple(snapshot.pred_action.shape), (1, 16, 19))
        self.assertEqual(tuple(snapshot.control_action.shape), (1, 8, 19))
        self.assertTrue(
            torch.equal(snapshot.control_action, snapshot.pred_action[:, 1:9])
        )

    def test_strict_selected_weight_failure_is_not_downgraded(self) -> None:
        class RejectingAgent(_Agent):
            def load_state_dict(self, state, strict):
                super().load_state_dict(state, strict)
                raise RuntimeError("missing selected key")

        with patch("hydra.utils.instantiate", return_value=RejectingAgent()):
            with self.assertRaisesRegex(DeploymentRestoreError, "strict restore"):
                restore_deployment_agent(_payload())

    def test_missing_normalizer_state_is_rejected_after_strict_restore(self) -> None:
        agent = _Agent(normalizer=_Normalizer(complete=False))
        with patch("hydra.utils.instantiate", return_value=agent):
            with self.assertRaisesRegex(DeploymentRestoreError, "normalizer"):
                restore_deployment_agent(_payload())
        self.assertTrue(agent.strict)

    def test_exact_parity_rejects_output_mismatch(self) -> None:
        reference = PredictionSnapshot(
            pred_action=torch.zeros((1, 16, 19)),
            control_action=torch.zeros((1, 8, 19)),
        )
        candidate = PredictionSnapshot(
            pred_action=torch.ones((1, 16, 19)),
            control_action=torch.ones((1, 8, 19)),
        )
        with self.assertRaisesRegex(
            PredictionParityError, "pred_action parity mismatch"
        ):
            assert_prediction_parity(reference, candidate)

    def test_parity_rejects_incompatible_dtype(self) -> None:
        reference = PredictionSnapshot(
            pred_action=torch.zeros((1, 16, 19), dtype=torch.float32),
            control_action=torch.zeros((1, 8, 19), dtype=torch.float32),
        )
        candidate = PredictionSnapshot(
            pred_action=torch.zeros((1, 16, 19), dtype=torch.float64),
            control_action=torch.zeros((1, 8, 19), dtype=torch.float64),
        )
        with self.assertRaisesRegex(
            PredictionParityError, "pred_action dtype mismatch"
        ):
            assert_prediction_parity(reference, candidate)


if __name__ == "__main__":
    unittest.main()
