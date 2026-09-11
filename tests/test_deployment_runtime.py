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

import torch

from dexmani_policy.deployment.runtime import (
    _resolve_deployment_checkpoint,
    inspect_experiment,
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
            with self.assertRaises(ValueError):
                inspect_experiment(experiment, artifact="../outside.pt")


if __name__ == "__main__":
    unittest.main()
