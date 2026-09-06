"""Offline, CPU-only contract checks for the canonical deployment cleanup.

These tests verify the single current deployment schema without a trained
checkpoint. They use fake agents, synthetic metadata, and tiny dummy tensors.
Nothing here trains, restores real weights, or touches a simulator.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.deployment import contract as contract_module
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DeploymentContractError,
    DeploymentSpec,
    ObservationFieldSpec,
    deployment_contract,
    parse_deployment_contract,
)
from dexmani_policy.deployment.restore import RestoredDeployment
from dexmani_policy.deployment.runtime import (
    ExperimentInfo,
    LoadedPolicy,
    PolicySpec,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "dexmani_policy" / "configs"
_CONFIG_NAMES = (
    "action_flow",
    "dp3",
    "dp",
    "dqrise",
    "maniflow",
    "multitask_dit",
    "r3d",
    "sat",
)
_ENV_RUNNER_DIR = _REPO_ROOT / "dexmani_policy" / "env_runner"


def _joint_state_field() -> ObservationFieldSpec:
    return ObservationFieldSpec(
        name="joint_state", shape=(6,), dtype="float32", semantics={}
    )


def _deployment_spec() -> DeploymentSpec:
    return DeploymentSpec(
        action_key="action",
        action_dim=19,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        denoise_steps=10,
        observation_fields=(_joint_state_field(),),
        control_dt_s=0.02,
        requires_hand=True,
        rgb_preprocessing=None,
    )


def _best_ckpt_record() -> dict:
    """A best_ckpt.json record matching the current strict schema."""
    return {
        "ckpt_relpath": "checkpoints/step-100000.pt",
        "pct": 100,
        "global_step": 100000,
        "success_rate": 0.8,
        "avg_steps": 150.5,
        "n_episodes": 100,
        "inference": {
            "use_ema": True,
            "denoise_steps": 10,
            "policy_seed_mode": "episode_seed",
        },
        "selection": {
            "shuffle_seed": 0,
            "seeds": [1, 2, 3],
            "initial_episodes": 3,
            "tie_break_used": False,
        },
    }


class _FakeAgent:
    """Deterministic agent whose control_action is the canonical pred slice."""

    def predict_action(self, tensors, denoise_timesteps=None):
        del tensors, denoise_timesteps
        pred = torch.arange(16 * 19, dtype=torch.float32).reshape(1, 16, 19)
        control = pred[:, 1:9, :19].clone()
        return {"pred_action": pred, "control_action": control}


class DeploymentContractTest(unittest.TestCase):
    def test_canonical_runtime_output(self):
        """LoadedPolicy.predict returns the validated control_action[0]."""
        deployment_spec = _deployment_spec()
        policy_spec = PolicySpec(
            action_key="action",
            action_dim=19,
            control_action_dim=19,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            observation_fields=(_joint_state_field(),),
            control_dt_s=0.02,
            requires_hand=True,
            rgb_preprocessing=None,
        )
        info = ExperimentInfo(
            selector="p/t/e",
            experiment_dir=Path("."),
            policy_name="p",
            task_name="t",
            checkpoint_path=Path("ckpt"),
            checkpoint_name="ckpt",
            spec=policy_spec,
        )
        restored = RestoredDeployment(agent=_FakeAgent(), spec=deployment_spec)
        policy = LoadedPolicy(info, restored, device="cpu", seed=0)

        observation = {"joint_state": np.zeros((2, 6), dtype=np.float32)}
        result = policy.predict(observation)

        expected = torch.arange(16 * 19, dtype=torch.float32).reshape(1, 16, 19)
        expected_control = expected[0, 1:9, :19].to(dtype=torch.float64).numpy()
        self.assertEqual(result.shape, (8, 19))
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, expected_control)

    def test_policy_spec_has_no_temporal_field(self):
        spec = PolicySpec(
            action_key="action",
            action_dim=19,
            control_action_dim=19,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            observation_fields=(_joint_state_field(),),
            control_dt_s=0.02,
            requires_hand=True,
            rgb_preprocessing=None,
        )
        self.assertFalse(hasattr(spec, "temporal_ensemble_coeff"))
        # Unrelated fields remain intact.
        self.assertEqual(spec.action_dim, 19)
        self.assertEqual(spec.horizon, 16)
        self.assertIs(spec.requires_hand, True)
        self.assertIsNone(spec.rgb_preprocessing)

    def test_deployment_spec_has_no_temporal_field(self):
        spec = _deployment_spec()
        self.assertFalse(hasattr(spec, "temporal_ensemble_coeff"))
        self.assertEqual(spec.control_action_dim, 19)

    def test_current_deployment_contract_parses(self):
        payload = {
            "_format": DEPLOYMENT_FORMAT,
            "contract": {
                "inference_config": {
                    "task_name": "t",
                    "action_key": "action",
                    "action_dim": 19,
                    "horizon": 16,
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "eval": {"denoise_steps": 10},
                    "agent": {},
                },
                "data_contract": {
                    "dt": 0.02,
                    "requires_hand": True,
                    "observation_fields": {
                        "joint_state": {
                            "shape": [6],
                            "dtype": "float32",
                            "semantics": {},
                        }
                    },
                },
                "producer": {},
            },
            "weights": {"w": torch.ones(2)},
        }
        self.assertEqual(DEPLOYMENT_FORMAT, "dexmani.deployment")

        contract = deployment_contract(payload)
        self.assertNotIn("schema_version", contract)

        spec = parse_deployment_contract(payload)
        self.assertFalse(hasattr(spec, "temporal_ensemble_coeff"))
        self.assertEqual(spec.denoise_steps, 10)
        self.assertEqual(spec.control_action_dim, 19)

        # The deployment schema-version constant is gone; there is no dispatch.
        self.assertFalse(hasattr(contract_module, "DEPLOYMENT_SCHEMA_VERSION"))

    def test_current_deployment_contract_enforces_required(self):
        base = {
            "_format": DEPLOYMENT_FORMAT,
            "contract": {
                "inference_config": {
                    "task_name": "t",
                    "action_key": "action",
                    "action_dim": 19,
                    "horizon": 16,
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "eval": {"denoise_steps": 10},
                    "agent": {},
                },
                "data_contract": {
                    "dt": 0.02,
                    "requires_hand": True,
                    "observation_fields": {
                        "joint_state": {
                            "shape": [6],
                            "dtype": "float32",
                            "semantics": {},
                        }
                    },
                },
                "producer": {},
            },
            "weights": {"w": torch.ones(2)},
        }
        missing_denoise = copy.deepcopy(base)
        del missing_denoise["contract"]["inference_config"]["eval"]["denoise_steps"]
        with self.assertRaises(DeploymentContractError):
            parse_deployment_contract(missing_denoise)

        missing_weights = copy.deepcopy(base)
        missing_weights["weights"] = {}
        with self.assertRaises(DeploymentContractError):
            deployment_contract(missing_weights)

    def test_best_ckpt_current_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            ckpt = exp_dir / "checkpoints" / "step-100000.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.touch()
            record = {
                "ckpt_relpath": "checkpoints/step-100000.pt",
                "pct": 100,
                "global_step": 100000,
                "success_rate": 0.8,
                "avg_steps": 150.5,
                "n_episodes": 100,
                "inference": {
                    "use_ema": True,
                    "denoise_steps": 10,
                    "policy_seed_mode": "episode_seed",
                },
                "selection": {
                    "shuffle_seed": 0,
                    "seeds": [1, 2, 3],
                    "initial_episodes": 3,
                    "tie_break_used": False,
                },
            }
            (exp_dir / "best_ckpt.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

            from dexmani_policy.training.eval_utils import read_best_ckpt_json

            parsed = read_best_ckpt_json(exp_dir)
            # Current schema has no version field and no temporal coefficient.
            self.assertNotIn("record_version", parsed)
            self.assertNotIn("temporal_ensemble_coeff", parsed["inference"])
            self.assertEqual(parsed["inference"]["use_ema"], True)

    def test_best_ckpt_missing_required_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            ckpt = exp_dir / "checkpoints" / "step-100000.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.touch()
            record = {
                "ckpt_relpath": "checkpoints/step-100000.pt",
                "pct": 100,
                "global_step": 100000,
                "success_rate": 0.8,
                "avg_steps": 150.5,
                "n_episodes": 100,
                "inference": {
                    "use_ema": True,
                    "policy_seed_mode": "episode_seed",
                },
                "selection": {
                    "shuffle_seed": 0,
                    "seeds": [1, 2, 3],
                    "initial_episodes": 3,
                    "tie_break_used": False,
                },
            }
            (exp_dir / "best_ckpt.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

            from dexmani_policy.training.eval_utils import read_best_ckpt_json

            with self.assertRaises(ValueError):
                read_best_ckpt_json(exp_dir)

    def test_best_ckpt_extra_record_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            ckpt = exp_dir / "checkpoints" / "step-100000.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.touch()
            record = _best_ckpt_record()
            record["record_version"] = 1
            (exp_dir / "best_ckpt.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

            from dexmani_policy.training.eval_utils import read_best_ckpt_json

            with self.assertRaisesRegex(
                ValueError, "does not match current schema"
            ):
                read_best_ckpt_json(exp_dir)

    def test_best_ckpt_inference_temporal_ensemble_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            ckpt = exp_dir / "checkpoints" / "step-100000.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.touch()
            record = _best_ckpt_record()
            record["inference"]["temporal_ensemble_coeff"] = 1.0
            (exp_dir / "best_ckpt.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

            from dexmani_policy.training.eval_utils import read_best_ckpt_json

            with self.assertRaisesRegex(
                ValueError, "does not match current schema"
            ):
                read_best_ckpt_json(exp_dir)

    def test_configs_have_no_temporal_key(self):
        for name in _CONFIG_NAMES:
            path = _CONFIG_DIR / f"{name}.yaml"
            self.assertTrue(path.is_file(), f"missing config {path}")
            self.assertNotIn(
                "temporal_ensemble_coeff",
                path.read_text(encoding="utf-8"),
                f"{name}.yaml still references temporal_ensemble_coeff",
            )

    def test_env_runner_sources_have_no_temporal_field(self):
        for filename in (
            "base_runner.py",
            "sim_runner.py",
            "multi_task_sim_runner.py",
        ):
            path = _ENV_RUNNER_DIR / filename
            self.assertNotIn(
                "temporal_ensemble_coeff",
                path.read_text(encoding="utf-8"),
                f"{filename} still references temporal_ensemble_coeff",
            )

    def test_base_runner_signature_has_no_temporal_param(self):
        import inspect

        try:
            from dexmani_policy.env_runner.base_runner import BaseRunner
        except Exception as exc:  # pragma: no cover - environment-dependent
            self.skipTest(f"BaseRunner not importable here: {exc}")
        params = inspect.signature(BaseRunner.__init__).parameters
        self.assertNotIn("temporal_ensemble_coeff", params)


if __name__ == "__main__":
    unittest.main()
