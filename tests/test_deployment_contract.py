"""Offline, CPU-only contract checks for the canonical deployment cleanup.

These tests verify the single current deployment schema without a trained
checkpoint. They use fake agents, synthetic metadata, and tiny dummy tensors.
Nothing here trains, restores real weights, or touches a simulator.
"""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from dataclasses import asdict, fields, replace
from pathlib import Path
from unittest.mock import patch

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
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    RestoredDeployment,
    validate_prediction,
)
from dexmani_policy.deployment.runtime import (
    ExperimentInfo,
    LoadedPolicy,
    PolicySpec,
    load_experiment,
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


def _runtime(spec, agent, *, seed=0):
    policy_spec = PolicySpec(
        **{key: value for key, value in vars(spec).items() if key != "denoise_steps"},
        control_action_dim=spec.control_action_dim,
        default_inference_steps=spec.denoise_steps,
    )
    info = ExperimentInfo(
        selector="p/t/e", experiment_dir=Path("."), policy_name="p", task_name="t",
        checkpoint_path=Path("ckpt"), checkpoint_name="ckpt", spec=policy_spec,
    )
    return LoadedPolicy(info, RestoredDeployment(agent, spec), device="cpu", seed=seed)


class _SyntheticAgent:
    def __init__(self, spec, *, stochastic=False, dtype=torch.float32):
        self.spec = spec
        self.stochastic = stochastic
        self.calls = 0
        self.pred = torch.arange(spec.horizon * spec.action_dim, dtype=dtype).reshape(
            1, spec.horizon, spec.action_dim
        )

    def predict_action(self, tensors, denoise_timesteps=None):
        self.calls += 1
        self.last_tensors = tensors
        self.last_denoise_steps = denoise_timesteps
        pred = self.pred
        if self.stochastic:
            pred = pred + torch.rand_like(pred) + random.random() + np.random.random()
        start = self.spec.n_obs_steps - 1
        return {
            "pred_action": pred,
            "control_action": pred[
                :, start : start + self.spec.n_action_steps, : self.spec.control_action_dim
            ],
        }


class DeploymentContractTest(unittest.TestCase):
    def test_full_chunk_contract(self):
        for action_key, action_dim in (("action", 19), ("action", 25), ("action_ee", 24)):
            for n_action_steps in (8, 15):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(key=action_key, steps=n_action_steps, dtype=dtype):
                        spec = replace(
                            _deployment_spec(), action_key=action_key,
                            action_dim=action_dim, n_action_steps=n_action_steps,
                        )
                        agent = _SyntheticAgent(spec, dtype=dtype)
                        policy = _runtime(spec, agent)
                        obs = {"joint_state": np.zeros((2, 6), dtype=np.float32)}
                        with patch(
                            "dexmani_policy.deployment.restore.validate_prediction",
                            wraps=validate_prediction,
                        ) as validate:
                            chunk = policy.predict_action_chunk(obs)
                            self.assertEqual(validate.call_count, 1)
                        self.assertEqual(agent.calls, 1)
                        self.assertEqual(agent.last_denoise_steps, spec.denoise_steps)
                        self.assertEqual(agent.last_tensors["joint_state"].shape, (1, 2, 6))
                        self.assertEqual(policy.spec.chunk_size, 15)
                        self.assertEqual(spec.chunk_size, 15)
                        self.assertNotIn("chunk_size", asdict(policy.spec))
                        self.assertNotIn("chunk_size", asdict(spec))
                        self.assertEqual(chunk.shape, (15, spec.control_action_dim))
                        self.assertEqual(chunk.dtype, np.float64)
                        self.assertTrue(np.isfinite(chunk).all())
                        self.assertTrue(chunk.flags.owndata)
                        expected = agent.pred[0, 1:, :spec.control_action_dim].numpy()
                        np.testing.assert_array_equal(chunk, expected)
                        control = policy.predict(obs)  # Deterministic fake: same sample.
                        self.assertEqual(agent.calls, 2)
                        self.assertEqual(control.shape, (n_action_steps, spec.control_action_dim))
                        self.assertEqual(control.dtype, np.float64)
                        self.assertTrue(np.isfinite(control).all())
                        np.testing.assert_array_equal(chunk[:n_action_steps], control)
                        chunk[0, 0] = -999
                        self.assertNotEqual(expected[0, 0], -999)
                        self.assertNotEqual(control[0, 0], -999)

    def test_chunk_reuses_prediction_rejections(self):
        spec = _deployment_spec()
        for failure in ("shape", "finite", "canonical"):
            with self.subTest(failure=failure):
                agent = _SyntheticAgent(spec)
                result = agent.predict_action({})
                if failure == "shape":
                    result["pred_action"] = result["pred_action"][:, :-1]
                elif failure == "finite":
                    result["pred_action"][0, -1, 0] = float("nan")
                else:
                    result["control_action"] = result["control_action"] + 1
                with patch.object(agent, "predict_action", return_value=result):
                    with self.assertRaises(DeploymentRestoreError):
                        _runtime(spec, agent).predict_action_chunk(
                            {"joint_state": np.zeros((2, 6), dtype=np.float32)}
                        )

    def test_episode_seed_and_warmup_preserve_stream(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            spec = _deployment_spec()
            policy = _runtime(spec, _SyntheticAgent(spec, stochastic=True), seed=42)
            obs = {"joint_state": np.zeros((2, 6), dtype=np.float32)}
            policy.reset_episode()
            expected = policy.predict_action_chunk(obs)
            policy.reset_episode()
            np.testing.assert_array_equal(expected, policy.predict_action_chunk(obs))
            policy.reset_episode()
            durations = policy.warmup(samples=2)
            self.assertEqual(len(durations), 2)
            np.testing.assert_array_equal(expected, policy.predict_action_chunk(obs))
            other = _runtime(spec, _SyntheticAgent(spec, stochastic=True), seed=43)
            other.reset_episode()
            self.assertFalse(np.array_equal(expected, other.predict_action_chunk(obs)))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def test_negative_runtime_seed_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "seed must be a non-negative int"):
            load_experiment("unused", device="cpu", seed=-1)

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
            default_inference_steps=10,
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
            default_inference_steps=10,
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
        self.assertEqual(spec.chunk_size, 15)
        self.assertNotIn("chunk_size", contract["inference_config"])
        payload["contract"]["inference_config"]["n_action_steps"] = 15
        self.assertEqual(parse_deployment_contract(payload).n_action_steps, 15)
        payload["contract"]["inference_config"]["n_action_steps"] = 16
        with self.assertRaisesRegex(DeploymentContractError, "window exceeds horizon"):
            parse_deployment_contract(payload)
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


class ExportGitProvenanceRemovalTest(unittest.TestCase):
    """Export correctness must not depend on the code directory's git state.

    Scientific/model checks (strict restore, weights selection, normalizer,
    dimensions, observation contract, prediction smoke test, parity) are
    covered elsewhere and are intentionally NOT part of this removal.
    """

    def test_export_source_has_no_git_machinery(self):
        from dexmani_policy.deployment import export as export_module

        source = Path(export_module.__file__).read_text(encoding="utf-8")
        for token in (
            "subprocess",
            "urlparse",
            "_run_git",
            "_producer_provenance",
            "_is_expected_repository_remote",
            "_GIT_COMMIT_RE",
            "_SCP_REMOTE_RE",
            "producer_commit",
            "rev-parse",
            "haoyangzhanglab",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_export_receipt_keeps_non_git_fields_only(self):
        from dexmani_policy.deployment.export import ExportReceipt

        self.assertEqual(
            {field.name for field in fields(ExportReceipt)},
            {
                "checkpoint_path",
                "selector_path",
                "metadata_provenance",
                "checkpoint_selector",
            },
        )


if __name__ == "__main__":
    unittest.main()
