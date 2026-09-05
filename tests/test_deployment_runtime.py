from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dexmani_policy.common.temporal_ensembler import ChunkOverlapBlender
from dexmani_policy.deployment import runtime
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
    ObservationFieldSpec,
)
from dexmani_policy.deployment.restore import DeploymentSpec, RestoredDeployment


def _fields() -> tuple[ObservationFieldSpec, ...]:
    return (
        ObservationFieldSpec("joint_state", (19,), "float32", {}),
        ObservationFieldSpec("point_cloud", (4, 6), "float32", {}),
    )


def _payload() -> dict:
    return {
        "_format": DEPLOYMENT_FORMAT,
        "contract": {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "inference_config": {
                "task_name": "task",
                "action_key": "action",
                "action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "eval": {
                    "denoise_steps": 10,
                    "temporal_ensemble_coeff": None,
                },
            },
            "data_contract": {
                "observation_fields": {
                    "joint_state": {"shape": [19], "dtype": "float32"},
                    "point_cloud": {"shape": [4, 6], "dtype": "float32"},
                },
                "dt": 0.1,
                "requires_hand": True,
            },
            "producer": {},
        },
        "weights": {"weight": torch.ones(1)},
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


class _ChunkAgent(_Agent):
    def __init__(self, chunks: list[torch.Tensor]) -> None:
        super().__init__()
        self._chunks = chunks
        self._index = 0

    def predict_action(self, observation, denoise_timesteps):
        prediction = self._chunks[self._index]
        self._index += 1
        return {
            "pred_action": prediction,
            "control_action": prediction[:, 1:9, :19],
        }


def _observation() -> dict[str, np.ndarray]:
    return {
        "joint_state": np.zeros((2, 19), dtype=np.float32),
        "point_cloud": np.zeros((2, 4, 6), dtype=np.float32),
    }


def _loaded_policy(
    agent: _Agent,
    *,
    action_dim: int = 19,
    coefficient: float | None = None,
) -> runtime.LoadedPolicy:
    deployment_spec = DeploymentSpec(
        action_key="action",
        action_dim=action_dim,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        denoise_steps=10,
        temporal_ensemble_coeff=coefficient,
        observation_fields=_fields(),
        control_dt_s=0.1,
        requires_hand=True,
        rgb_preprocessing=None,
    )
    info = runtime.ExperimentInfo(
        selector="policy/task/run",
        experiment_dir=Path("/experiment"),
        policy_name="policy",
        task_name="task",
        checkpoint_path=Path("/experiment/checkpoints/artifact.pt"),
        checkpoint_name="artifact.pt",
        spec=runtime.PolicySpec(
            action_key="action",
            action_dim=action_dim,
            control_action_dim=19,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            temporal_ensemble_coeff=coefficient,
            observation_fields=_fields(),
            control_dt_s=0.1,
            requires_hand=True,
        ),
    )
    return runtime.LoadedPolicy(
        info,
        RestoredDeployment(agent=agent, spec=deployment_spec),
        device="cpu",
        seed=0,
    )


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
                tuple(field.name for field in info.spec.observation_fields),
                ("joint_state", "point_cloud"),
            )
            self.assertEqual(info.spec.observation_fields[1].shape, (4, 6))
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
                    temporal_ensemble_coeff=None,
                    observation_fields=_fields(),
                    control_dt_s=0.1,
                    requires_hand=True,
                    rgb_preprocessing=None,
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
                "joint_state": np.zeros((2, 19), dtype=np.float32),
                "point_cloud": np.zeros((2, 4, 6), dtype=np.float32),
            }
            observation["point_cloud"].setflags(write=False)
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
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
            spec=DeploymentSpec(
                "action", 19, 16, 2, 8, 10, None, _fields(), 0.1, True, None
            ),
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
                None,
                _fields(),
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

    def test_policy_spec_rejects_empty_observation_fields(self) -> None:
        with self.assertRaises(ValueError):
            runtime.PolicySpec(
                "action",
                19,
                19,
                16,
                2,
                8,
                None,
                (),
                0.1,
                True,
            )

    def test_temporal_blender_matches_reference_and_slices_aux_dimensions(self) -> None:
        chunks = [
            torch.arange(16 * 28, dtype=torch.float32).reshape(1, 16, 28),
            torch.arange(16 * 28, dtype=torch.float32).reshape(1, 16, 28) + 1000,
        ]
        loaded = _loaded_policy(_ChunkAgent(chunks), action_dim=28, coefficient=0.2)
        reference = ChunkOverlapBlender(0.2, n_obs_steps=2)

        first = loaded.predict(_observation())
        second = loaded.predict(_observation())
        expected_first = reference.update(chunks[0][..., :19], 8)[0].double().numpy()
        expected_second = reference.update(chunks[1][..., :19], 8)[0].double().numpy()

        self.assertEqual(first.shape, (8, 19))
        self.assertEqual(second.shape, (8, 19))
        np.testing.assert_array_equal(first, expected_first)
        np.testing.assert_array_equal(second, expected_second)

    def test_reset_episode_clears_temporal_overlap(self) -> None:
        chunks = [
            torch.zeros((1, 16, 19)),
            torch.full((1, 16, 19), 10.0),
        ]
        loaded = _loaded_policy(_ChunkAgent(chunks), coefficient=0.1)
        loaded.predict(_observation())
        loaded.reset_episode()

        after_reset = loaded.predict(_observation())

        np.testing.assert_array_equal(
            after_reset, chunks[1][:, 1:9, :][0].double().numpy()
        )

    def test_warmup_preserves_existing_blender_object_state_and_rng(self) -> None:
        chunks = [
            torch.zeros((1, 16, 19)),
            torch.full((1, 16, 19), 100.0),
            torch.full((1, 16, 19), 200.0),
            torch.full((1, 16, 19), 10.0),
        ]
        loaded = _loaded_policy(_ChunkAgent(chunks), coefficient=0.1)
        reference = ChunkOverlapBlender(0.1, n_obs_steps=2)
        loaded.predict(_observation())
        reference.update(chunks[0], 8)
        original_blender = loaded._blender

        np.random.seed(41)
        torch.manual_seed(43)
        expected_numpy = np.random.RandomState(41).random_sample()
        expected_torch = torch.rand((), generator=torch.Generator().manual_seed(43))
        loaded.warmup(samples=2)

        self.assertIs(loaded._blender, original_blender)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(()), expected_torch)
        after_warmup = loaded.predict(_observation())
        expected = reference.update(chunks[3], 8)[0].double().numpy()
        np.testing.assert_array_equal(after_warmup, expected)


if __name__ == "__main__":
    unittest.main()
