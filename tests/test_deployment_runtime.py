"""Offline checks for the public deployment runtime resolution API.

Covers explicit deployment-artifact selection and inference-step override
semantics without a trained checkpoint: synthetic payloads, fake agents, and
temporary experiment directories only.  Nothing here touches a simulator,
restores real weights, or connects to hardware.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.deployment.contract import (
    DeploymentSpec,
    ObservationFieldSpec,
)
from dexmani_policy.deployment.restore import RestoredDeployment
from dexmani_policy.deployment.runtime import (
    ExperimentInfo,
    LoadedPolicy,
    PolicySpec,
    _resolve_deployment_checkpoint,
    inspect_experiment,
    load_experiment,
)

_PAYLOAD = {
    "_format": "dexmani.deployment",
    "contract": {
        "inference_config": {
            "task_name": "pick_cube",
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
                "joint_state": {"shape": [6], "dtype": "float32", "semantics": {}}
            },
        },
        "producer": {},
    },
    "weights": {"w": torch.ones(2)},
}


def _deployment_spec() -> DeploymentSpec:
    return DeploymentSpec(
        action_key="action",
        action_dim=19,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        denoise_steps=10,
        observation_fields=(
            ObservationFieldSpec(
                name="joint_state", shape=(6,), dtype="float32", semantics={}
            ),
        ),
        control_dt_s=0.02,
        requires_hand=True,
        rgb_preprocessing=None,
    )


class _RecordingAgent:
    """Deterministic agent recording every denoise_timesteps value it receives."""

    def __init__(self, spec: DeploymentSpec):
        self.spec = spec
        self.denoise_calls: list[int | None] = []
        self.pred = torch.arange(
            spec.horizon * spec.action_dim, dtype=torch.float32
        ).reshape(1, spec.horizon, spec.action_dim)

    def predict_action(self, tensors, denoise_timesteps=None):
        del tensors
        self.denoise_calls.append(denoise_timesteps)
        start = self.spec.n_obs_steps - 1
        return {
            "pred_action": self.pred,
            "control_action": self.pred[
                :, start : start + self.spec.n_action_steps,
                : self.spec.control_action_dim
            ],
        }


def _loaded_policy(
    agent: _RecordingAgent,
    *,
    seed: int = 0,
    inference_steps: int | None = None,
) -> LoadedPolicy:
    spec = agent.spec
    policy_spec = PolicySpec(
        action_key=spec.action_key,
        action_dim=spec.action_dim,
        control_action_dim=spec.control_action_dim,
        horizon=spec.horizon,
        n_obs_steps=spec.n_obs_steps,
        n_action_steps=spec.n_action_steps,
        observation_fields=spec.observation_fields,
        control_dt_s=spec.control_dt_s,
        requires_hand=spec.requires_hand,
        default_inference_steps=spec.denoise_steps,
        rgb_preprocessing=spec.rgb_preprocessing,
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
    return LoadedPolicy(
        info,
        RestoredDeployment(agent, spec),
        device="cpu",
        seed=seed,
        inference_steps=inference_steps,
    )


def _observation() -> dict[str, np.ndarray]:
    return {"joint_state": np.zeros((2, 6), dtype=np.float32)}


def _write_experiment(root: Path) -> Path:
    """Create a temporary experiment with two artifacts and a latest selector."""
    experiment = root / "exp"
    checkpoints = experiment / "checkpoints"
    checkpoints.mkdir(parents=True)
    (experiment / "config.yaml").write_text(
        "policy_name: action_flow\ntask_name: pick_cube\n"
    )
    newest = checkpoints / "epoch_500-deployment.pt"
    torch.save(_PAYLOAD, newest)
    torch.save(_PAYLOAD, checkpoints / "epoch_400-deployment.pt")
    (checkpoints / "deployment_latest.pt").symlink_to(newest)
    return experiment


class ArtifactResolutionTest(unittest.TestCase):
    def test_none_resolves_deployment_latest_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = _write_experiment(Path(tmp))
            resolved = _resolve_deployment_checkpoint(experiment)
            self.assertEqual(resolved.name, "epoch_500-deployment.pt")
            self.assertTrue(resolved.is_file())

    def test_explicit_artifact_resolves_exact_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = _write_experiment(Path(tmp))
            resolved = _resolve_deployment_checkpoint(
                experiment, artifact="epoch_400-deployment.pt"
            )
            self.assertEqual(
                resolved, (experiment / "checkpoints" / "epoch_400-deployment.pt").resolve()
            )

    def test_artifact_cannot_escape_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = _write_experiment(Path(tmp))
            torch.save(_PAYLOAD, Path(tmp) / "outside.pt")
            for artifact in ("../outside.pt", str(Path(tmp) / "outside.pt"), "sub/x.pt"):
                with self.subTest(artifact=artifact):
                    with self.assertRaises(ValueError):
                        _resolve_deployment_checkpoint(experiment, artifact=artifact)
            for artifact in ("", 123):
                with self.subTest(artifact=artifact):
                    with self.assertRaises(ValueError):
                        _resolve_deployment_checkpoint(experiment, artifact=artifact)
            with self.assertRaises(FileNotFoundError):
                _resolve_deployment_checkpoint(experiment, artifact="missing.pt")

    def test_artifact_symlink_escape_and_non_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = _write_experiment(Path(tmp))
            torch.save(_PAYLOAD, Path(tmp) / "outside.pt")
            (experiment / "checkpoints" / "evil.pt").symlink_to(Path(tmp) / "outside.pt")
            (experiment / "checkpoints" / "dir.pt").mkdir()
            with self.assertRaises(FileNotFoundError):
                _resolve_deployment_checkpoint(experiment, artifact="evil.pt")
            with self.assertRaises(FileNotFoundError):
                _resolve_deployment_checkpoint(experiment, artifact="dir.pt")

    def test_inspect_experiment_pins_resolved_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = _write_experiment(Path(tmp))
            latest = inspect_experiment(experiment)
            self.assertEqual(latest.checkpoint_name, "epoch_500-deployment.pt")
            self.assertTrue(latest.checkpoint_path.is_file())
            pinned = inspect_experiment(
                experiment, artifact="epoch_400-deployment.pt"
            )
            self.assertEqual(pinned.checkpoint_name, "epoch_400-deployment.pt")
            self.assertEqual(pinned.spec.action_dim, 19)
            self.assertEqual(pinned.spec.default_inference_steps, 10)
            with self.assertRaises(ValueError):
                inspect_experiment(experiment, artifact="../outside.pt")


class InferenceStepsOverrideTest(unittest.TestCase):
    def test_none_uses_artifact_default(self):
        agent = _RecordingAgent(_deployment_spec())
        policy = _loaded_policy(agent)
        policy.predict(_observation())
        self.assertEqual(agent.denoise_calls, [10])

    def test_override_reaches_warmup_and_predict(self):
        agent = _RecordingAgent(_deployment_spec())
        policy = _loaded_policy(agent, inference_steps=3)
        policy.warmup(samples=2)
        policy.predict(_observation())
        policy.predict_action_chunk(_observation())
        self.assertEqual(agent.denoise_calls, [3, 3, 3, 3])

    def test_non_positive_override_rejected(self):
        for bad in (0, -1, True, 1.0, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _loaded_policy(
                        _RecordingAgent(_deployment_spec()), inference_steps=bad
                    )

    def test_load_experiment_rejects_before_loading(self):
        with self.assertRaisesRegex(
            ValueError, "inference_steps must be a positive int"
        ):
            load_experiment("unused", device="cpu", inference_steps=0)


if __name__ == "__main__":
    unittest.main()
