"""Focused self-contained-asset checks for deployment-only construction.

These tests deliberately exercise the runtime leaf modules rather than a full
diffusion policy.  The full policies are expensive and add unrelated CUDA / point
cloud dependencies; strict state restoration of the actual codebook manager and
Uni3D encoder is the relevant asset boundary here.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

import dexmani_policy.deployment.export as exporter
from dexmani_policy.agents.core.dqrise import DQRISEAgent
from dexmani_policy.agents.core.r3d import R3DAgent
from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder
from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.deployment.restore import (
    assert_prediction_parity,
    deployment_spec,
    reset_inference_seed,
    restore_deployment_agent,
    validate_prediction,
)

try:
    from .network_guard import network_forbidden
except ImportError:
    from network_guard import network_forbidden

_HAND_DIM = 12
_CODEBOOK_SIZE = 4
_NUM_GROUPS = 2
_NUM_CODES = _CODEBOOK_SIZE**_NUM_GROUPS
_HAND_MIN = -37.5
_HAND_MAX = 83.25


def _write_codebook(path: Path) -> None:
    """Write a small but complete external DQ-RISE codebook fixture."""
    normalized_poses = np.linspace(
        -1.0, 1.0, _NUM_CODES * _HAND_DIM, dtype=np.float32
    ).reshape(_NUM_CODES, _HAND_DIM)
    raw_poses = (normalized_poses + 1.0) * 0.5 * (_HAND_MAX - _HAND_MIN) + _HAND_MIN
    np.savez(
        path,
        format_version=3,
        pose_space="affine_raw",
        sorted_hand_poses=raw_poses,
        pca_permutation=np.arange(_NUM_CODES, dtype=np.int64),
        layer_weights=np.array([0.25, 0.75], dtype=np.float32),
        hand_dim=_HAND_DIM,
        num_groups=_NUM_GROUPS,
        codebook_size=_CODEBOOK_SIZE,
        hand_min=_HAND_MIN,
        hand_max=_HAND_MAX,
        hand_normalizer_scale=np.linspace(0.1, 1.2, _HAND_DIM, dtype=np.float32),
        hand_normalizer_offset=np.linspace(-0.4, 0.7, _HAND_DIM, dtype=np.float32),
        metadata_json="{}",
    )


def _dq_train_params() -> dict[str, object]:
    return {
        "action_key": "action_ee",
        "action_dim": 21,
        "tcp_dim": 9,
        "hand_dim": _HAND_DIM,
        "control_action_dim": 21,
    }


def _dq_agent_config(path: str | None) -> dict[str, object]:
    return {
        "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
        "tcp_dim": 9,
        "codebook_path": path,
        "codebook_num_groups": _NUM_GROUPS,
        "codebook_size": _CODEBOOK_SIZE,
    }


def _prefixed_codebook_state(manager: CodebookManager) -> dict[str, torch.Tensor]:
    return {
        f"codebook_manager.{key}": value.clone()
        for key, value in manager.state_dict().items()
    }


def _normalizer_and_observation(
    *, action_dim: int, num_points: int
) -> tuple[LinearNormalizer, dict[str, torch.Tensor]]:
    """Build complete deployment normalizer state and a bounded observation."""
    action = torch.linspace(-0.9, 0.9, 2 * 4 * action_dim, dtype=torch.float32).reshape(
        2, 4, action_dim
    )
    joint_state = torch.linspace(-0.8, 0.8, 2 * 2 * 19, dtype=torch.float32).reshape(
        2, 2, 19
    )
    point_cloud = torch.linspace(
        -0.7, 0.7, 2 * 2 * num_points * 6, dtype=torch.float32
    ).reshape(2, 2, num_points, 6)
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": action,
            "joint_state": joint_state,
            "point_cloud": point_cloud,
        }
    )
    return normalizer, {
        "joint_state": joint_state[:1].clone(),
        "point_cloud": point_cloud[:1].clone(),
    }


def _deployment_payload(
    *,
    agent_config: dict[str, object],
    model_state: dict[str, torch.Tensor],
    action_key: str,
    action_dim: int,
    num_points: int,
) -> dict[str, object]:
    return {
        "_format": "dexmani.deployment.v2",
        "state": {
            "inference_config": {
                "action_key": action_key,
                "action_dim": action_dim,
                "horizon": 4,
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "agent": agent_config,
                "eval": {"use_ema": False, "denoise_steps": 1},
            },
            "data_contract": {
                "point_cloud_num_points": num_points,
                "point_cloud_feature_dim": 6,
            },
        },
        "weights": {"model": model_state, "ema_model": None},
    }


def _seeded_prediction(
    agent: nn.Module, observation: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    reset_inference_seed(17)
    agent.eval()
    with torch.inference_mode():
        return agent.predict_action(observation, denoise_timesteps=1)


class CodebookDeploymentAssetTest(unittest.TestCase):
    def _load_external_codebook(self, root: Path) -> tuple[Path, CodebookManager]:
        path = root / "training-only-codebook.npz"
        _write_codebook(path)
        manager = CodebookManager(
            _HAND_DIM, num_groups=_NUM_GROUPS, codebook_size=_CODEBOOK_SIZE
        )
        manager.load(path)
        return path, manager

    def test_state_dict_derives_all_runtime_affine_and_codebook_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manager = self._load_external_codebook(Path(directory))
            state = manager.state_dict()

        self.assertEqual(
            set(state),
            {
                "hand_min",
                "hand_max",
                "sorted_hand_poses",
                "pca_permutation",
                "layer_weights",
                "hand_normalizer_scale",
                "hand_normalizer_offset",
            },
        )
        self.assertEqual(state["hand_min"].ndim, 0)
        self.assertEqual(state["hand_max"].ndim, 0)
        torch.testing.assert_close(state["hand_min"], torch.tensor(_HAND_MIN))
        torch.testing.assert_close(state["hand_max"], torch.tensor(_HAND_MAX))
        self.assertEqual(
            tuple(state["sorted_hand_poses"].shape), (_NUM_CODES, _HAND_DIM)
        )
        self.assertEqual(tuple(state["hand_normalizer_scale"].shape), (_HAND_DIM,))
        self.assertEqual(tuple(state["hand_normalizer_offset"].shape), (_HAND_DIM,))

    def test_null_path_restore_is_complete_or_fails_closed_and_legacy_external_restore_survives(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, source = self._load_external_codebook(Path(directory))
            state = source.state_dict()

            # A legacy training checkpoint without manager buffers remains
            # directly restorable only when the caller supplied the old
            # external asset before strict state restore.
            legacy_direct = CodebookManager(
                _HAND_DIM, num_groups=_NUM_GROUPS, codebook_size=_CODEBOOK_SIZE
            )
            legacy_direct.load(path)
            legacy_direct.load_state_dict({}, strict=True)
            torch.testing.assert_close(
                legacy_direct.sorted_hand_poses, source.sorted_hand_poses
            )

            # Deployment construction has codebook_path=null.  It must rely
            # solely on checkpoint tensors after the training-only file is gone.
            path.unlink()
            self.assertFalse(path.exists())
            deployed = CodebookManager(
                _HAND_DIM, num_groups=_NUM_GROUPS, codebook_size=_CODEBOOK_SIZE
            )
            deployed.load_state_dict(state, strict=True)
            torch.testing.assert_close(deployed.hand_min, source.hand_min)
            torch.testing.assert_close(deployed.hand_max, source.hand_max)
            torch.testing.assert_close(
                deployed.continuous_index_to_hand_pose(torch.tensor([-1.0, 1.0]))[0],
                source.continuous_index_to_hand_pose(torch.tensor([-1.0, 1.0]))[0],
            )

            # A fresh null-path manager must not accept partial checkpoints.
            for missing_key in state:
                with self.subTest(missing_key=missing_key):
                    incomplete = {
                        key: value for key, value in state.items() if key != missing_key
                    }
                    fresh = CodebookManager(
                        _HAND_DIM, num_groups=_NUM_GROUPS, codebook_size=_CODEBOOK_SIZE
                    )
                    with self.assertRaisesRegex(RuntimeError, "Missing key"):
                        fresh.load_state_dict(incomplete, strict=True)

    def test_export_validator_requires_complete_null_path_codebook_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, manager = self._load_external_codebook(Path(directory))
            complete = _prefixed_codebook_state(manager)
            complete["normalizer.params_dict.action.scale"] = torch.cat(
                [torch.ones(9), manager.hand_normalizer_scale.clone()]
            )
            complete["normalizer.params_dict.action.offset"] = torch.cat(
                [torch.zeros(9), manager.hand_normalizer_offset.clone()]
            )

            sanitized = exporter._sanitize_agent_config(
                _dq_agent_config(str(path)), complete, _dq_train_params()
            )
            self.assertIsNone(sanitized["codebook_path"])

            for missing_key in manager.state_dict():
                missing_key = f"codebook_manager.{missing_key}"
                with self.subTest(missing_key=missing_key):
                    incomplete = {
                        key: value
                        for key, value in complete.items()
                        if key != missing_key
                    }
                    with self.assertRaisesRegex(
                        exporter.UnsupportedPolicyError,
                        "persistent runtime codebook state",
                    ):
                        exporter._sanitize_agent_config(
                            _dq_agent_config(str(path)), incomplete, _dq_train_params()
                        )

            mismatched = dict(complete)
            mismatched["normalizer.params_dict.action.offset"] = mismatched[
                "normalizer.params_dict.action.offset"
            ].clone()
            mismatched["normalizer.params_dict.action.offset"][-1].add_(0.1)
            with self.assertRaisesRegex(
                exporter.InvalidCheckpointError, "hand normalizer conflicts"
            ):
                exporter._sanitize_agent_config(
                    _dq_agent_config(str(path)), mismatched, _dq_train_params()
                )

    def test_dqrise_hand_normalizer_metadata_matches_policy_normalizer(self) -> None:
        manager = CodebookManager(
            _HAND_DIM, num_groups=_NUM_GROUPS, codebook_size=_CODEBOOK_SIZE
        )
        manager.sorted_hand_poses = torch.zeros(_NUM_CODES, _HAND_DIM)
        action = torch.linspace(-3.0, 5.0, 8 * 21, dtype=torch.float32).reshape(8, 21)
        normalizer = LinearNormalizer()
        normalizer.fit({"action": action})
        action_params = normalizer["action"].params_dict
        manager.set_hand_normalizer(
            action_params["scale"][-_HAND_DIM:], action_params["offset"][-_HAND_DIM:]
        )

        # This is a deliberately tiny DQRISE shell: use the real validation
        # method without constructing its unrelated point-cloud/UNet stack.
        agent = DQRISEAgent.__new__(DQRISEAgent)
        nn.Module.__init__(agent)
        agent.hand_dim = _HAND_DIM
        agent.codebook_manager = manager
        agent.normalizer = normalizer
        agent._normalizer_checked = False
        agent._missing_codebook_normalizer_warned = False
        agent._validate_codebook_normalizer()
        self.assertTrue(agent._normalizer_checked)

        manager.hand_normalizer_offset[0].add_(0.01)
        with self.assertRaisesRegex(ValueError, "do not match"):
            agent._validate_codebook_normalizer()


class R3DPretrainedAssetTest(unittest.TestCase):
    class _TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 8
            self.pos_drop = nn.Identity()
            self.blocks = nn.ModuleList([nn.Linear(8, 8)])
            self.norm = nn.LayerNorm(8)
            self.fc_norm = nn.Identity()

    @staticmethod
    def _r3d_pc_encoder_config() -> dict[str, object]:
        """Derive the test fixture from the real R3D config's encoder subtree."""
        config_path = (
            Path(__file__).parents[2] / "dexmani_policy" / "configs" / "r3d.yaml"
        )
        resolved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        assert isinstance(resolved, dict)
        agent = resolved["agent"]
        assert isinstance(agent, dict)
        pc_config = copy.deepcopy(agent["pc_encoder_config"])
        assert isinstance(pc_config, dict)
        pc_config.update(
            {
                "pc_model": "test-tiny-transformer",
                "embed_dim": 8,
                "num_group": 2,
                "group_size": 2,
                "drop_path_rate": 0.0,
            }
        )
        return pc_config

    def test_sanitized_r3d_config_strictly_restores_without_pretrained_or_network_loader(
        self,
    ) -> None:
        source_config = self._r3d_pc_encoder_config()
        self.assertIs(source_config["use_pretrained_weights"], True)

        # The source-side constructor represents training-time initialization;
        # mock it so the test never touches a local file or the network.
        with (
            patch("timm.create_model", return_value=self._TinyTransformer()),
            patch.object(
                Uni3DPointcloudEncoder, "_load_pretrained_weights"
            ) as source_loader,
        ):
            source = Uni3DPointcloudEncoder(**source_config)
        source_loader.assert_called_once()
        source_state = source.state_dict()

        sanitized_agent = exporter._sanitize_agent_config(
            {
                "_target_": "dexmani_policy.agents.core.r3d.R3DAgent",
                "pc_encoder_config": source_config,
            },
            {"weight": torch.ones(1)},
            _dq_train_params(),
        )
        sanitized_pc_config = sanitized_agent["pc_encoder_config"]
        self.assertIs(sanitized_pc_config["use_pretrained_weights"], False)

        # If the deployment constructor regresses to the training-time path,
        # this guard fails before any Hugging Face/network request can occur.
        with (
            patch("timm.create_model", return_value=self._TinyTransformer()),
            patch.object(
                Uni3DPointcloudEncoder,
                "_load_pretrained_weights",
                side_effect=AssertionError(
                    "deployment restore attempted pretrained/network loading"
                ),
            ) as deployment_loader,
        ):
            deployed = Uni3DPointcloudEncoder(**sanitized_pc_config)
            deployed.load_state_dict(source_state, strict=True)
        deployment_loader.assert_not_called()
        self.assertEqual(set(deployed.state_dict()), set(source_state))


class ActualAgentSelfContainedRestoreTest(unittest.TestCase):
    """Tiny genuine-agent parity checks for the DQ-RISE and R3D asset paths."""

    @staticmethod
    def _dqrise_config(codebook_path: str | None) -> dict[str, object]:
        return {
            "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
            "horizon": 4,
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "action_dim": 21,
            "tcp_dim": 9,
            "codebook_path": codebook_path,
            "codebook_num_groups": _NUM_GROUPS,
            "codebook_size": _CODEBOOK_SIZE,
            "encoder_type": "dp3",
            "pc_dim": 6,
            "pc_out_dim": 8,
            "state_dim": 19,
            "num_points": 8,
            "state_out_dim": 8,
            "diffusion_step_embed_dim": 8,
            "down_dims": [8],
            "kernel_size": 3,
            "n_groups": 4,
            "num_training_steps": 2,
            "num_inference_steps": 1,
            "prediction_type": "sample",
        }

    @staticmethod
    def _r3d_config(*, use_aux_ee: bool) -> dict[str, object]:
        action_dim = 28 if use_aux_ee else 19
        return {
            "_target_": "dexmani_policy.agents.core.r3d.R3DAgent",
            "horizon": 4,
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "action_dim": action_dim,
            "state_dim": 19,
            "state_out_dim": 4,
            "pc_encoder_config": {
                "pc_model": "test-tiny-transformer",
                "embed_dim": 8,
                "num_group": 2,
                "group_size": 2,
                "pc_in_channels": 6,
                "patch_dropout": 0.0,
                "drop_path_rate": 0.0,
                "feature_mode": "pointsam",
                "use_pretrained_weights": True,
                "pretrained_weights_path": "training-only-pretrained",
            },
            "timestep_embed_dim": 8,
            "embedding_dim": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_dim": 16,
            "attention_downsample_rate": 1,
            "num_training_steps": 2,
            "num_inference_steps": 1,
            "prediction_type": "sample",
            "use_aux_ee": use_aux_ee,
            "joint_dim": 19,
            "ee_dim": 9,
        }

    def test_dqrise_actual_agent_restores_without_external_npz_and_preserves_diagnostics(
        self,
    ) -> None:
        normalizer, observation = _normalizer_and_observation(
            action_dim=21, num_points=8
        )
        action_params = normalizer["action"].params_dict

        with tempfile.TemporaryDirectory() as directory:
            codebook_path = Path(directory) / "training-only-codebook.npz"
            _write_codebook(codebook_path)
            # The VQ extraction metadata has to describe the policy's actual
            # action normalizer.  This is normally set by extract_codebook.py.
            with np.load(codebook_path, allow_pickle=False) as saved:
                payload = {key: saved[key] for key in saved.files}
            payload["hand_normalizer_scale"] = (
                action_params["scale"][-_HAND_DIM:].detach().cpu().numpy()
            )
            payload["hand_normalizer_offset"] = (
                action_params["offset"][-_HAND_DIM:].detach().cpu().numpy()
            )
            np.savez(codebook_path, **payload)

            source_config = self._dqrise_config(str(codebook_path))
            source = DQRISEAgent(
                **{
                    key: value
                    for key, value in source_config.items()
                    if key != "_target_"
                }
            )
            source.load_normalizer_from_dataset(normalizer)
            source_state = dict(source.state_dict())

            sanitized = exporter._sanitize_agent_config(
                source_config, source_state, _dq_train_params()
            )
            self.assertIsNone(sanitized["codebook_path"])
            codebook_path.unlink()

            deployment = _deployment_payload(
                agent_config=sanitized,
                model_state=source_state,
                action_key="action_ee",
                action_dim=21,
                num_points=8,
            )
            with (
                network_forbidden(),
                patch.object(
                    CodebookManager,
                    "load",
                    side_effect=AssertionError(
                        "fresh deployment attempted external codebook loading"
                    ),
                ) as external_loader,
            ):
                restored = restore_deployment_agent(deployment)
            external_loader.assert_not_called()

        direct_result = _seeded_prediction(source, observation)
        restored_result = _seeded_prediction(restored.agent, observation)
        spec = deployment_spec(deployment)
        direct_snapshot = validate_prediction(direct_result, spec, batch_size=1)
        restored_snapshot = validate_prediction(restored_result, spec, batch_size=1)
        assert_prediction_parity(direct_snapshot, restored_snapshot)
        self.assertTrue(
            torch.equal(
                direct_result["pred_code_index"], restored_result["pred_code_index"]
            )
        )
        self.assertTrue(
            torch.equal(
                direct_result["pred_code_continuous"],
                restored_result["pred_code_continuous"],
            )
        )

    def test_r3d_actual_agent_restores_without_pretrained_loader_and_preserves_parity(
        self,
    ) -> None:
        for use_aux_ee in (False, True):
            with (
                self.subTest(use_aux_ee=use_aux_ee),
                tempfile.TemporaryDirectory() as directory,
            ):
                action_dim = 28 if use_aux_ee else 19
                normalizer, observation = _normalizer_and_observation(
                    action_dim=action_dim, num_points=4
                )
                source_config = self._r3d_config(use_aux_ee=use_aux_ee)
                pretrained_path = Path(directory) / "training-only-pretrained"
                pretrained_path.mkdir()
                (pretrained_path / "model.safetensors").write_bytes(b"fixture")
                source_config["pc_encoder_config"]["pretrained_weights_path"] = str(
                    pretrained_path
                )
                with (
                    patch(
                        "timm.create_model",
                        side_effect=lambda *args, **kwargs: R3DPretrainedAssetTest._TinyTransformer(),
                    ),
                    patch.object(
                        Uni3DPointcloudEncoder, "_load_pretrained_weights"
                    ) as source_loader,
                ):
                    source = R3DAgent(
                        **{
                            key: value
                            for key, value in source_config.items()
                            if key != "_target_"
                        }
                    )
                source_loader.assert_called_once()
                source.load_normalizer_from_dataset(normalizer)
                source_state = dict(source.state_dict())

                sanitized = exporter._sanitize_agent_config(
                    source_config, source_state, _dq_train_params()
                )
                self.assertIs(
                    sanitized["pc_encoder_config"]["use_pretrained_weights"], False
                )
                removed_path = pretrained_path.with_name("pretrained.removed")
                pretrained_path.rename(removed_path)
                self.assertFalse(pretrained_path.exists())
                deployment = _deployment_payload(
                    agent_config=sanitized,
                    model_state=source_state,
                    action_key="action",
                    action_dim=action_dim,
                    num_points=4,
                )
                with (
                    network_forbidden(),
                    patch(
                        "timm.create_model",
                        side_effect=lambda *args, **kwargs: R3DPretrainedAssetTest._TinyTransformer(),
                    ),
                    patch.object(
                        Uni3DPointcloudEncoder,
                        "_load_pretrained_weights",
                        side_effect=AssertionError(
                            "fresh deployment attempted pretrained/network loading"
                        ),
                    ) as deployment_loader,
                ):
                    restored = restore_deployment_agent(deployment)
                deployment_loader.assert_not_called()

                direct_snapshot = validate_prediction(
                    _seeded_prediction(source, observation),
                    deployment_spec(deployment),
                    batch_size=1,
                )
                restored_snapshot = validate_prediction(
                    _seeded_prediction(restored.agent, observation),
                    deployment_spec(deployment),
                    batch_size=1,
                )
                assert_prediction_parity(direct_snapshot, restored_snapshot)


if __name__ == "__main__":
    unittest.main()
