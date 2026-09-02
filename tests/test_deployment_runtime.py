from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dexmani_policy.deployment import runtime
from dexmani_policy.deployment.restore import DeploymentSpec, RestoredDeployment


def _payload() -> dict:
    return {
        "_format": "dexmani.deployment.v2",
        "state": {
            "inference_config": {
                "task_name": "task",
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "eval": {"denoise_steps": 10},
            },
            "data_contract": {
                "sensor_modalities": ["joint_state", "point_cloud"],
                "point_cloud_num_points": 4,
                "point_cloud_feature_dim": 6,
                "dt": 0.1,
            },
        },
        "weights": {},
    }


def _experiment(root: Path) -> Path:
    experiment = root / "policy" / "task" / "run"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (experiment / "config.yaml").write_text(
        "policy_name: policy\ntask_name: task\n", encoding="utf-8"
    )
    checkpoint = checkpoint_dir / "artifact.pt"
    checkpoint.touch()
    (checkpoint_dir / "deployment_latest.pt").symlink_to(checkpoint.name)
    return experiment


class _Agent:
    def __init__(self) -> None:
        self.reset_count = 0
        self.devices: list[str] = []

    def predict_action(self, observation, denoise_timesteps):
        self.last_observation = observation
        self.last_denoise_timesteps = denoise_timesteps
        pred = torch.arange(16 * 19, dtype=torch.float32).reshape(1, 16, 19)
        return {
            "pred_action": pred,
            "control_action": pred[:, 1:9, :19],
        }

    def reset_episode(self) -> None:
        self.reset_count += 1

    def to(self, device: str):
        self.devices.append(device)
        return self


class DeploymentRuntimeTest(unittest.TestCase):
    def test_resolve_and_list_use_only_explicit_deployments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiments"
            experiment = _experiment(root)
            incomplete = root / "policy" / "task" / "incomplete"
            incomplete.mkdir()
            with mock.patch.object(runtime, "_EXPERIMENTS_ROOT", root):
                self.assertEqual(
                    runtime.resolve_experiment("policy/task/run"),
                    experiment.resolve(),
                )
                self.assertEqual(
                    runtime.resolve_experiment(experiment), experiment.resolve()
                )
                self.assertEqual(runtime.list_experiments(), ("policy/task/run",))
                self.assertEqual(
                    runtime.list_experiments(filter="TASK/RUN"),
                    ("policy/task/run",),
                )
                self.assertEqual(runtime.list_experiments(filter="missing"), ())
                with self.assertRaises(ValueError):
                    runtime.resolve_experiment("latest")

    def test_inspect_reads_metadata_without_restoring_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiments"
            experiment = _experiment(root)
            with (
                mock.patch.object(runtime, "_EXPERIMENTS_ROOT", root),
                mock.patch.object(
                    runtime, "_read_deployment_payload", return_value=_payload()
                ) as read_payload,
                mock.patch(
                    "dexmani_policy.deployment.restore.restore_deployment_agent"
                ) as restore,
            ):
                info = runtime.inspect_experiment("policy/task/run")

            restore.assert_not_called()
            read_payload.assert_called_once_with(
                experiment / "checkpoints" / "artifact.pt", map_location="meta"
            )
            self.assertEqual(info.selector, "policy/task/run")
            self.assertEqual(info.checkpoint_name, "artifact.pt")
            self.assertEqual(
                info.spec.sensor_modalities, ("joint_state", "point_cloud")
            )
            self.assertEqual(info.spec.point_cloud_num_points, 4)
            self.assertIsNone(info.spec.rgb_shape)
            self.assertEqual(info.spec.control_dt_s, 0.1)
            self.assertTrue(info.spec.requires_hand)

    def test_loaded_policy_has_numpy_only_prediction_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiments"
            _experiment(root)
            agent = _Agent()
            restored = RestoredDeployment(
                agent=agent,
                spec=DeploymentSpec(
                    action_key="action",
                    action_dim=19,
                    horizon=16,
                    n_obs_steps=2,
                    n_action_steps=8,
                    denoise_steps=10,
                    point_cloud_num_points=4,
                    point_cloud_feature_dim=6,
                ),
            )
            with (
                mock.patch.object(runtime, "_EXPERIMENTS_ROOT", root),
                mock.patch.object(
                    runtime, "_read_deployment_payload", return_value=_payload()
                ),
                mock.patch(
                    "dexmani_policy.deployment.restore.restore_deployment_agent",
                    return_value=restored,
                ),
            ):
                loaded = runtime.load_experiment(
                    "policy/task/run", device="cpu", seed=7
                )

            observation = {
                "joint_state": np.zeros((2, 19), dtype=np.float64),
                "point_cloud": np.zeros((2, 4, 6), dtype=np.float32),
            }
            action = loaded.predict(observation)
            self.assertEqual(action.shape, (8, 19))
            self.assertEqual(action.dtype, np.float64)
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(agent.last_denoise_timesteps, 10)
            self.assertEqual(agent.last_observation["joint_state"].shape, (1, 2, 19))
            self.assertEqual(len(loaded.warmup(samples=2)), 2)
            with self.assertRaises(ValueError):
                loaded.warmup(samples=0)
            self.assertGreaterEqual(agent.reset_count, 1)

            loaded.close()
            loaded.close()
            self.assertEqual(agent.devices, ["cpu"])
            with self.assertRaises(RuntimeError):
                loaded.predict(observation)

    def test_predict_rejects_invalid_observation(self) -> None:
        agent = _Agent()
        restored = RestoredDeployment(
            agent=agent,
            spec=DeploymentSpec("action", 19, 16, 2, 8, 10, 4, 6),
        )
        info = runtime.ExperimentInfo(
            selector="policy/task/run",
            experiment_dir=Path("/experiment"),
            policy_name="policy",
            task_name="task",
            checkpoint_path=Path("/experiment/checkpoints/artifact.pt"),
            checkpoint_name="artifact.pt",
            spec=runtime.PolicySpec(
                "action",
                19,
                19,
                16,
                2,
                8,
                ("joint_state", "point_cloud"),
                4,
                6,
                None,
                None,
                None,
                0.1,
                True,
            ),
        )
        loaded = runtime.LoadedPolicy(info, restored, device="cpu", seed=0)
        with self.assertRaises(ValueError):
            loaded.predict(
                {
                    "joint_state": np.zeros((1, 19)),
                    "point_cloud": np.zeros((2, 4, 6)),
                }
            )

    def test_policy_spec_rejects_inconsistent_modality_contract(self) -> None:
        with self.assertRaises(ValueError):
            runtime.PolicySpec(
                "action",
                19,
                19,
                16,
                2,
                8,
                ("joint_state", "point_cloud"),
                None,
                None,
                None,
                None,
                None,
                0.1,
                True,
            )


if __name__ == "__main__":
    unittest.main()
