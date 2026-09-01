from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

import dexmani_policy.deployment.export as exporter
from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint

COMMIT = "c" * 40


def _attrs() -> dict:
    return {
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 5,
        "domain": "real",
        "profile": "pointcloud",
        "task_name": "pick_test",
        "dt": 0.0625,
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_reference": "camera_source_monotonic_ns",
        "state_alignment": "camera_source_aligned_state",
        "max_observation_skew_s": 0.1,
        "action_semantics": "deployment_grid_rate_limited_target",
        "arm_max_delta_rad_per_tick": 0.1,
        "hand_max_delta_rad_per_tick": 0.3,
        "endpoint_delta_tolerance_rad": 1e-12,
        "deployment_equivalent": True,
        "point_cloud_frame": "xarm_base",
        "point_cloud_color_source": "mean_rgb_of_aligned_depth_pixels_per_voxel",
        "point_cloud_policy_id": "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9",
        "point_cloud_config_sha256": "a" * 64,
        "point_cloud_table_plane_abcd_json": "null",
        "point_cloud_sampling": "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad",
        "point_cloud_transform": (
            "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
            "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
            "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
            "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
            "coarse_voxel_stratified_hash_or_cyclic_pad"
        ),
    }


def _config(*, use_ema: bool = False) -> dict:
    return {
        "policy_name": "dp3",
        "task_name": "pick_test",
        "zarr_path": "unused-by-override.zarr",
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_key": "action",
        "action_dim": 19,
        "use_aux_ee": False,
        "dataset": {
            "_target_": "dexmani_policy.datasets.pc_dataset.PCDataset",
            "sensor_modalities": ["joint_state", "point_cloud"],
        },
        "agent": {
            "_target_": "dexmani_policy.agents.core.dp3.DP3Agent",
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "action_dim": 19,
            "pc_dim": 6,
            "num_points": 1024,
        },
        "eval": {
            "use_ema": use_ema,
            "denoise_steps": 2,
            "denoise_timesteps_list": None,
        },
    }


def _train_params(*, include_aux: bool = True) -> dict:
    result = {
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_dim": 19,
        "horizon": 16,
        "action_key": "action",
        "tcp_dim": None,
        "hand_dim": None,
        "control_action_dim": 19,
        "num_training_steps": 100,
    }
    if include_aux:
        result["use_aux_ee"] = False
    return result


def _checkpoint(*, include_aux: bool = True, ema: bool = False) -> TrainCheckpoint:
    return TrainCheckpoint(
        epoch=3,
        global_step=42,
        model_state={"weight": torch.tensor([1.0])},
        ema_model_state={"weight": torch.tensor([2.0])} if ema else None,
        optimizer_state={"secret": 1},
        scheduler_state={"secret": 2},
        monitor={},
        train_params=_train_params(include_aux=include_aux),
    )


def _write_zarr(path: Path, attrs: dict | None = None) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(attrs or _attrs())
    data = root.create_group("data")
    data.create_dataset(
        "point_cloud",
        data=np.zeros((2, 1024, 6), dtype=np.float32),
        shape=(2, 1024, 6),
    )
    data.create_dataset(
        "joint_state", data=np.zeros((2, 19), dtype=np.float32), shape=(2, 19)
    )
    data.create_dataset(
        "action", data=np.zeros((2, 19), dtype=np.float32), shape=(2, 19)
    )
    data.create_dataset(
        "action_ee", data=np.zeros((2, 21), dtype=np.float32), shape=(2, 21)
    )


def _write_experiment(root: Path, *, use_ema: bool = False) -> tuple[Path, Path]:
    experiment = root / "experiment"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(_config(use_ema=use_ema)), experiment / "config.yaml"
    )
    zarr_path = root / "policy.zarr"
    _write_zarr(zarr_path)
    checkpoint_path = CheckpointStore(checkpoint_dir).save(
        "epoch=0003-step=00000042-100pct.pt",
        _checkpoint(ema=use_ema),
    )
    (checkpoint_dir / "latest.pt").symlink_to(checkpoint_path.name)
    return experiment, zarr_path


class CheckpointSelectionTest(unittest.TestCase):
    def test_best_index_then_best_symlink_and_never_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, _ = _write_experiment(Path(directory))
            checkpoint_dir = experiment / "checkpoints"
            selected = checkpoint_dir / "epoch=0003-step=00000042-100pct.pt"
            (experiment / "best_ckpt.json").write_text(
                json.dumps({"ckpt_relpath": f"checkpoints/{selected.name}"}),
                encoding="utf-8",
            )
            self.assertEqual(exporter._resolve_checkpoint(experiment, "best"), selected)

            (experiment / "best_ckpt.json").unlink()
            (checkpoint_dir / "best.pt").symlink_to(selected.name)
            self.assertEqual(exporter._resolve_checkpoint(experiment, "best"), selected)

            (checkpoint_dir / "best.pt").unlink()
            with self.assertRaises(exporter.InvalidCheckpointError):
                exporter._resolve_checkpoint(experiment, "best")
            self.assertEqual(
                exporter._resolve_checkpoint(experiment, "latest"), selected
            )

    def test_missing_best_errors_even_when_latest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, _ = _write_experiment(Path(directory))
            with self.assertRaises(exporter.InvalidCheckpointError):
                exporter._resolve_checkpoint(experiment, "best")

    def test_invalid_best_index_never_falls_back_to_best_symlink(self) -> None:
        for record in (
            "{malformed",
            json.dumps({}),
            json.dumps({"ckpt_relpath": "missing.pt"}),
        ):
            with (
                self.subTest(record=record),
                tempfile.TemporaryDirectory() as directory,
            ):
                experiment, _ = _write_experiment(Path(directory))
                checkpoint_dir = experiment / "checkpoints"
                selected = checkpoint_dir / "epoch=0003-step=00000042-100pct.pt"
                (checkpoint_dir / "best.pt").symlink_to(selected.name)
                (experiment / "best_ckpt.json").write_text(record, encoding="utf-8")
                with self.assertRaises(exporter.InvalidCheckpointError):
                    exporter._resolve_checkpoint(experiment, "best")

    def test_best_index_rejects_relative_and_absolute_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, _ = _write_experiment(root)
            external = root / "external.pt"
            external.write_bytes(b"external")
            for target in ("../external.pt", str(external.resolve())):
                with self.subTest(target=target):
                    (experiment / "best_ckpt.json").write_text(
                        json.dumps({"ckpt_relpath": target}), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        exporter.InvalidCheckpointError, "outside"
                    ):
                        exporter._resolve_checkpoint(experiment, "best")

    def test_best_and_latest_symlink_escape_are_rejected(self) -> None:
        for selector in ("best", "latest"):
            with (
                self.subTest(selector=selector),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                experiment, _ = _write_experiment(root)
                checkpoint_dir = experiment / "checkpoints"
                external = root / "external.pt"
                external.write_bytes(b"external")
                link = checkpoint_dir / f"{selector}.pt"
                link.unlink(missing_ok=True)
                link.symlink_to(external)
                with self.assertRaisesRegex(exporter.InvalidCheckpointError, "outside"):
                    exporter._resolve_checkpoint(experiment, selector)

    def test_explicit_and_milestone_selectors_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, _ = _write_experiment(root)
            checkpoint_dir = experiment / "checkpoints"
            external = root / "external.pt"
            external.write_bytes(b"external")
            for selector in ("../../external.pt", str(external.resolve())):
                with self.subTest(selector=selector):
                    with self.assertRaisesRegex(
                        exporter.InvalidCheckpointError, "outside"
                    ):
                        exporter._resolve_checkpoint(experiment, selector)

            milestone = checkpoint_dir / "epoch=0001-step=00000001-milestone=20pct.pt"
            milestone.symlink_to(external)
            with self.assertRaisesRegex(exporter.InvalidCheckpointError, "outside"):
                exporter._resolve_checkpoint(experiment, "20pct")


class ResolvedConfigContractTest(unittest.TestCase):
    def test_agent_deployment_fields_are_required_and_consistent(self) -> None:
        for field, value in (
            ("action_dim", 21),
            ("horizon", 15),
            ("n_obs_steps", 3),
            ("n_action_steps", 7),
        ):
            with self.subTest(field=field):
                config = _config()
                config["agent"][field] = value
                with self.assertRaisesRegex(
                    exporter.InvalidExperimentError, f"agent.{field}"
                ):
                    exporter._validate_resolved_config_contract(config, _train_params())

        config = _config()
        del config["agent"]["action_dim"]
        with self.assertRaisesRegex(exporter.InvalidExperimentError, "missing"):
            exporter._validate_resolved_config_contract(config, _train_params())

    def test_dataset_present_fields_must_be_consistent(self) -> None:
        mismatches = {
            "action_key": "action_ee",
            "horizon": 15,
            "obs_horizon": 3,
            "pad_before": 0,
            "pad_after": 8,
            "use_aux_ee": True,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                config = _config()
                config["dataset"][field] = value
                with self.assertRaisesRegex(
                    exporter.InvalidExperimentError, f"dataset.{field}"
                ):
                    exporter._validate_resolved_config_contract(config, _train_params())

    def test_actionflow_style_state_dim_does_not_equal_action_dim(self) -> None:
        config = _config()
        config.update({"action_key": "action_ee", "action_dim": 21})
        config["agent"].update({"action_dim": 21, "state_dim": 19})
        train = _train_params()
        train.update(
            {
                "action_key": "action_ee",
                "action_dim": 21,
                "control_action_dim": 21,
            }
        )
        exporter._validate_resolved_config_contract(config, train)

    def test_structural_mismatch_rejects_export_when_verify_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            config = _config()
            config["agent"]["action_dim"] = 21
            OmegaConf.save(OmegaConf.create(config), experiment / "config.yaml")
            with patch.object(exporter, "_producer_provenance", return_value=COMMIT):
                with self.assertRaisesRegex(
                    exporter.InvalidExperimentError, "agent.action_dim"
                ):
                    exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=False,
                        zarr_path=zarr_path,
                    )
            self.assertFalse(
                (experiment / "checkpoints" / "deployment_latest.pt").exists()
            )


class SourceAndPathResolutionTest(unittest.TestCase):
    def _git_result(self, root: Path, remote: str, *, status: str = ""):
        def result(_root, *args):
            values = {
                ("rev-parse", "--show-toplevel"): str(root),
                ("rev-parse", "HEAD"): COMMIT,
                ("status", "--porcelain", "--untracked-files=all"): status,
                ("remote", "get-url", "origin"): remote,
            }
            return values[args]

        return result

    def test_clean_frozen_repository_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            remote = "https://github.com/haoyangzhanglab/dexmani_policy.git"
            clean_git = self._git_result(root, remote)
            with patch.object(exporter, "_run_git", side_effect=clean_git):
                self.assertEqual(exporter._producer_provenance(root), COMMIT)
            dirty_git = self._git_result(
                root, remote, status=" M dexmani_policy/dirty.py"
            )
            with patch.object(exporter, "_run_git", side_effect=dirty_git):
                with self.assertRaisesRegex(
                    exporter.InvalidExperimentError, "working tree must be clean"
                ):
                    exporter._producer_provenance(root)

    def test_exact_canonical_repository_remotes_are_accepted(self) -> None:
        remotes = (
            "https://github.com/haoyangzhanglab/dexmani_policy",
            "https://github.com/haoyangzhanglab/dexmani_policy.git",
            "git@github.com:haoyangzhanglab/dexmani_policy",
            "git@github.com:haoyangzhanglab/dexmani_policy.git",
            "ssh://git@github.com/haoyangzhanglab/dexmani_policy.git",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for remote in remotes:
                with self.subTest(remote=remote):
                    git_result = self._git_result(root, remote)
                    with patch.object(exporter, "_run_git", side_effect=git_result):
                        self.assertEqual(exporter._producer_provenance(root), COMMIT)

    def test_wrong_or_tricky_repository_remotes_are_rejected(self) -> None:
        remotes = (
            "https://evilgithub.com/haoyangzhanglab/dexmani_policy.git",
            "https://github.com/other/dexmani_policy.git",
            "https://github.com/haoyangzhanglab/other.git",
            "git@evilgithub.com:haoyangzhanglab/dexmani_policy.git",
            "git@github.com:other/dexmani_policy.git",
            "git@github.com:haoyangzhanglab/other.git",
            "https://github.com/haoyangzhanglab/other/../dexmani_policy.git",
            "https://github.com/haoyangzhanglab/dexmani_policy.git/extra",
            "https://github.com:443/haoyangzhanglab/dexmani_policy.git",
            "https://[github.com]/haoyangzhanglab/dexmani_policy.git",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for remote in remotes:
                with self.subTest(remote=remote):
                    git_result = self._git_result(root, remote)
                    with patch.object(exporter, "_run_git", side_effect=git_result):
                        with self.assertRaises(exporter.InvalidExperimentError):
                            exporter._producer_provenance(root)

    def test_config_relative_zarr_is_repository_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "robot_data" / "task.zarr"
            expected.mkdir(parents=True)
            resolved = exporter._resolve_zarr_path(
                {"zarr_path": "robot_data/task.zarr"}, root, None
            )
            self.assertEqual(resolved, expected.resolve())


class ZarrContractTest(unittest.TestCase):
    def _validate(self, attrs: dict | None = None, config: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zarr"
            _write_zarr(path, attrs)
            return exporter._validate_zarr_contract(
                path,
                config or _config(),
                ["joint_state", "point_cloud"],
            )

    def test_valid_real_v5_is_accepted(self) -> None:
        contract = self._validate()
        self.assertEqual(contract["point_cloud_num_points"], 1024)
        self.assertEqual(contract["point_cloud_feature_dim"], 6)
        self.assertEqual(contract["sensor_modalities"], ["joint_state", "point_cloud"])

    def test_sim_and_non_equivalent_are_rejected(self) -> None:
        for key, value in (("domain", "sim"), ("deployment_equivalent", False)):
            with self.subTest(key=key):
                attrs = _attrs()
                attrs[key] = value
                with self.assertRaises(exporter.InvalidZarrError):
                    self._validate(attrs)

    def test_task_and_dt_mismatch_are_rejected(self) -> None:
        attrs = _attrs()
        attrs["task_name"] = "other"
        with self.assertRaisesRegex(exporter.InvalidZarrError, "task_name"):
            self._validate(attrs)
        config = _config()
        config["dt"] = 0.1
        with self.assertRaisesRegex(exporter.InvalidZarrError, "conflicts"):
            self._validate(config=config)

    def test_pointcloud_semantic_and_shape_mismatch_are_rejected(self) -> None:
        attrs = _attrs()
        attrs["point_cloud_config_sha256"] = "invalid"
        with self.assertRaisesRegex(exporter.InvalidZarrError, "SHA-256"):
            self._validate(attrs)
        attrs = _attrs()
        attrs["point_cloud_sampling"] = "random"
        with self.assertRaisesRegex(exporter.InvalidZarrError, "point-cloud semantics"):
            self._validate(attrs)
        config = _config()
        config["agent"]["num_points"] = 2048
        with self.assertRaisesRegex(exporter.InvalidZarrError, "point count"):
            self._validate(config=config)


class MetadataAndWeightTest(unittest.TestCase):
    def test_native_and_retrofitted_metadata(self) -> None:
        native, provenance, fields = exporter._reconcile_train_params(
            _checkpoint(), _config()
        )
        self.assertEqual(provenance, "native")
        self.assertEqual(fields, [])
        self.assertIs(native["use_aux_ee"], False)

        retrofitted, provenance, fields = exporter._reconcile_train_params(
            _checkpoint(include_aux=False), _config()
        )
        self.assertEqual(provenance, "retrofitted")
        self.assertEqual(fields, ["use_aux_ee"])
        self.assertIs(retrofitted["use_aux_ee"], False)

    def test_metadata_conflict_is_rejected(self) -> None:
        checkpoint = _checkpoint()
        checkpoint.train_params["horizon"] = 17
        with self.assertRaisesRegex(exporter.InvalidCheckpointError, "conflicts"):
            exporter._reconcile_train_params(checkpoint, _config())

    def test_model_is_always_present_and_keys_are_canonical(self) -> None:
        state = exporter._canonicalize_state_dict(
            {
                "module.layer._orig_mod.weight": torch.tensor([1.0]),
                "module.bias": torch.tensor([2.0]),
            },
            "model",
        )
        self.assertEqual(set(state), {"layer.weight", "bias"})
        payload = {
            "_format": "dexmani.deployment.v2",
            "state": {
                "epoch": 0,
                "global_step": 0,
                "train_params": {},
                "inference_config": {},
                "data_contract": {},
                "producer": {},
                "deployment_contract": {},
            },
            "weights": {"model": state, "ema_model": None},
        }
        exporter._validate_payload(payload)
        self.assertNotIn("optimizer", payload["weights"])
        self.assertNotIn("scheduler", payload["weights"])

    def test_ema_is_required_only_when_selected(self) -> None:
        inference = exporter._build_inference_config(
            _config(), _config()["agent"], _train_params()
        )
        self.assertFalse(inference["eval"]["use_ema"])
        config = _config(use_ema=True)
        inference = exporter._build_inference_config(
            config, config["agent"], _train_params()
        )
        self.assertTrue(inference["eval"]["use_ema"])


class ConstructorSanitizationTest(unittest.TestCase):
    def test_dqrise_requires_persistent_codebook_and_nulls_path(self) -> None:
        agent = {
            "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
            "tcp_dim": 9,
            "codebook_path": "/training/codebook.npz",
            "codebook_num_groups": 2,
            "codebook_size": 4,
        }
        train = _train_params()
        train.update(
            {
                "action_key": "action_ee",
                "action_dim": 21,
                "tcp_dim": 9,
                "hand_dim": 12,
                "control_action_dim": 21,
            }
        )
        with self.assertRaisesRegex(exporter.UnsupportedPolicyError, "persistent"):
            exporter._sanitize_agent_config(agent, {"weight": torch.ones(1)}, train)
        state = {
            "codebook_manager.sorted_hand_poses": torch.ones(16, 12),
            "codebook_manager.pca_permutation": torch.arange(16),
            "codebook_manager.layer_weights": torch.ones(2),
        }
        sanitized = exporter._sanitize_agent_config(agent, state, train)
        self.assertIsNone(sanitized["codebook_path"])
        self.assertEqual(sanitized["tcp_dim"], 9)
        self.assertEqual(sanitized["codebook_num_groups"], 2)
        self.assertEqual(sanitized["codebook_size"], 4)

    def test_r3d_pretrained_flag_is_sanitized_without_topology_change(self) -> None:
        agent = {
            "_target_": "dexmani_policy.agents.core.r3d.R3DAgent",
            "pc_encoder_config": {"use_pretrained_weights": True},
        }
        sanitized = exporter._sanitize_agent_config(
            agent, {"weight": torch.ones(1)}, _train_params()
        )
        self.assertIs(sanitized["pc_encoder_config"]["use_pretrained_weights"], False)

        from torch import nn

        from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import (
            Uni3DPointcloudEncoder,
        )

        class TinyTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 8
                self.blocks = nn.ModuleList([nn.Linear(8, 8)])
                self.norm = nn.LayerNorm(8)

        def make_encoder(use_pretrained: bool) -> Uni3DPointcloudEncoder:
            with (
                patch("timm.create_model", return_value=TinyTransformer()),
                patch.object(Uni3DPointcloudEncoder, "_load_pretrained_weights"),
            ):
                return Uni3DPointcloudEncoder(
                    pc_model="tiny",
                    embed_dim=8,
                    num_group=2,
                    group_size=2,
                    pc_in_channels=6,
                    use_pretrained_weights=use_pretrained,
                    pretrained_weights_path="unused",
                )

        self.assertEqual(
            set(make_encoder(True).state_dict()),
            set(make_encoder(False).state_dict()),
        )


class UnsupportedPolicyTest(unittest.TestCase):
    def test_rgb_and_multitask_are_rejected(self) -> None:
        for modalities in (["joint_state", "rgb"], None):
            config = _config()
            if modalities is None:
                config["dataset"] = {"datasets": [], "task_texts": ["do task"]}
            else:
                config["dataset"]["sensor_modalities"] = modalities
            with self.assertRaises(exporter.UnsupportedPolicyError):
                exporter._dataset_modalities(config)

    def test_custom_timestep_list_is_rejected(self) -> None:
        config = _config()
        config["eval"]["denoise_timesteps_list"] = [2, 4]
        with self.assertRaisesRegex(exporter.UnsupportedPolicyError, "timesteps_list"):
            exporter._build_inference_config(config, config["agent"], _train_params())

    def test_rgb_and_text_agent_fields_are_rejected_even_if_modalities_are_edited(
        self,
    ) -> None:
        for key, value in (
            ("rgb_backbone_name", "dino"),
            ("text_encoder_model", "clip"),
        ):
            config = _config()
            config["agent"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaises(exporter.UnsupportedPolicyError),
            ):
                exporter._dataset_modalities(config)


class ExportAndAtomicityTest(unittest.TestCase):
    def _export(self, root: Path, **kwargs):
        experiment, zarr_path = _write_experiment(
            root, use_ema=kwargs.pop("use_ema", False)
        )
        with patch.object(exporter, "_producer_provenance", return_value=COMMIT):
            receipt = exporter.export_deployment_artifact(
                experiment,
                checkpoint_selector="latest",
                zarr_path=zarr_path,
                verify=False,
                **kwargs,
            )
        return experiment, receipt

    def test_exact_checkpoint_sidecar_and_relative_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment, receipt = self._export(Path(directory))
            payload = torch.load(
                receipt.checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(payload["_format"], "dexmani.deployment.v2")
            self.assertEqual(set(payload), {"_format", "state", "weights"})
            self.assertEqual(set(payload["weights"]), {"model", "ema_model"})
            self.assertIn("model", payload["weights"])
            self.assertNotIn("optimizer", payload["weights"])
            self.assertEqual(
                os.readlink(receipt.selector_path), receipt.checkpoint_path.name
            )
            sidecar = json.loads(receipt.sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["schema_version"], 2)
            self.assertEqual(sidecar["allocation"]["required_action_steps"], 15)
            self.assertEqual(
                set(sidecar["producer"]),
                {"repository", "commit", "metadata_provenance"},
            )
            self.assertEqual(
                experiment / "checkpoints" / "deployment_latest.pt",
                receipt.selector_path,
            )

    def test_model_and_ema_are_kept_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, receipt = self._export(Path(directory), use_ema=True)
            payload = torch.load(
                receipt.checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertTrue(
                torch.equal(payload["weights"]["model"]["weight"], torch.tensor([1.0]))
            )
            self.assertTrue(
                torch.equal(
                    payload["weights"]["ema_model"]["weight"], torch.tensor([2.0])
                )
            )

    def test_missing_selected_ema_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            config = _config(use_ema=True)
            OmegaConf.save(OmegaConf.create(config), experiment / "config.yaml")
            with patch.object(exporter, "_producer_provenance", return_value=COMMIT):
                with self.assertRaisesRegex(exporter.InvalidCheckpointError, "EMA"):
                    exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=False,
                        zarr_path=zarr_path,
                    )
            self.assertFalse(
                (experiment / "checkpoints" / "deployment_latest.pt").exists()
            )

    def test_failures_never_change_existing_selector(self) -> None:
        failure_points = (
            ("_write_checkpoint_temp", OSError("checkpoint write")),
            ("_write_sidecar_temp", OSError("sidecar write")),
            (
                "_roundtrip_verify_published",
                exporter.ArtifactVerificationError("verify"),
            ),
        )
        for function_name, failure in failure_points:
            with (
                self.subTest(function=function_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                experiment, zarr_path = _write_experiment(root)
                checkpoint_dir = experiment / "checkpoints"
                old = checkpoint_dir / "old-deployment-v2.pt"
                old.write_bytes(b"old")
                selector = checkpoint_dir / "deployment_latest.pt"
                selector.symlink_to(old.name)
                with (
                    patch.object(exporter, "_producer_provenance", return_value=COMMIT),
                    patch.object(exporter, function_name, side_effect=failure),
                ):
                    with self.assertRaises(exporter.DeploymentExportError):
                        exporter.export_deployment_artifact(
                            experiment,
                            "latest",
                            verify=False,
                            zarr_path=zarr_path,
                        )
                self.assertTrue(selector.is_symlink())
                self.assertEqual(os.readlink(selector), old.name)

    def test_publication_and_selector_rollback_fsync_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            checkpoint_dir = experiment / "checkpoints"
            old = checkpoint_dir / "old-deployment-v2.pt"
            old.write_bytes(b"old")
            selector = checkpoint_dir / "deployment_latest.pt"
            selector.symlink_to(old.name)
            with (
                patch.object(exporter, "_producer_provenance", return_value=COMMIT),
                patch.object(exporter, "_fsync_directory") as fsync_directory,
                patch.object(
                    exporter,
                    "_roundtrip_verify_published",
                    side_effect=exporter.ArtifactVerificationError("verify"),
                ),
            ):
                with self.assertRaises(exporter.ArtifactVerificationError):
                    exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=False,
                        zarr_path=zarr_path,
                    )
            self.assertEqual(fsync_directory.call_count, 4)
            fsync_directory.assert_has_calls([call(checkpoint_dir)] * 4)
            self.assertEqual(os.readlink(selector), old.name)

    def test_selector_removal_rollback_fsyncs_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            selector = checkpoint_dir / "deployment_latest.pt"
            selector.symlink_to("orphan-deployment-v2.pt")
            with patch.object(exporter, "_fsync_directory") as fsync_directory:
                exporter._rollback_selector(selector, (False, None))
            self.assertFalse(selector.is_symlink())
            fsync_directory.assert_called_once_with(checkpoint_dir)

    def test_selector_fsync_failure_durably_restores_old_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            checkpoint_dir = experiment / "checkpoints"
            old = checkpoint_dir / "old-deployment-v2.pt"
            old.write_bytes(b"old")
            selector = checkpoint_dir / "deployment_latest.pt"
            selector.symlink_to(old.name)
            barriers = [None, None, OSError("selector fsync"), None]
            with (
                patch.object(exporter, "_producer_provenance", return_value=COMMIT),
                patch.object(
                    exporter, "_fsync_directory", side_effect=barriers
                ) as fsync_directory,
            ):
                with self.assertRaises(exporter.ArtifactPublicationError):
                    exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=False,
                        zarr_path=zarr_path,
                    )
            self.assertEqual(fsync_directory.call_count, 4)
            self.assertTrue(selector.is_symlink())
            self.assertEqual(os.readlink(selector), old.name)

    def test_verification_uses_reloaded_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            reloaded = object()
            with (
                patch.object(exporter, "_producer_provenance", return_value=COMMIT),
                patch.object(
                    exporter, "_load_deployment_payload", return_value=reloaded
                ),
                patch.object(exporter, "_verify_exported_model") as verify_model,
            ):
                exporter.export_deployment_artifact(
                    experiment,
                    "latest",
                    verify=True,
                    zarr_path=zarr_path,
                )
            verify_model.assert_called_once_with(reloaded)

    def test_serialized_verification_failure_preserves_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, zarr_path = _write_experiment(root)
            checkpoint_dir = experiment / "checkpoints"
            old = checkpoint_dir / "old-deployment-v2.pt"
            old.write_bytes(b"old")
            selector = checkpoint_dir / "deployment_latest.pt"
            selector.symlink_to(old.name)
            with (
                patch.object(exporter, "_producer_provenance", return_value=COMMIT),
                patch.object(
                    exporter,
                    "_verify_exported_model",
                    side_effect=exporter.ArtifactVerificationError("model verify"),
                ),
            ):
                with self.assertRaises(exporter.ArtifactVerificationError):
                    exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=True,
                        zarr_path=zarr_path,
                    )
            self.assertTrue(selector.is_symlink())
            self.assertEqual(os.readlink(selector), old.name)


class VerificationSemanticsTest(unittest.TestCase):
    def test_strict_restore_finite_shapes_and_exact_control_slice(self) -> None:
        class Normalizer:
            params_dict = {
                "action": {
                    "scale": torch.ones(19),
                    "offset": torch.zeros(19),
                },
                "joint_state": {
                    "scale": torch.ones(19),
                    "offset": torch.zeros(19),
                },
                "point_cloud": {
                    "scale": torch.ones(6),
                    "offset": torch.zeros(6),
                },
            }

            def is_fitted(self, required_keys):
                return required_keys == ["action", "joint_state", "point_cloud"]

        class Agent:
            normalizer = Normalizer()

            def load_state_dict(self, state, strict):
                self.state = state
                self.strict = strict

            def to(self, device):
                return self

            def eval(self):
                return self

            def predict_action(self, obs, denoise_timesteps):
                pred = torch.arange(16 * 19, dtype=torch.float32).reshape(1, 16, 19)
                return {"pred_action": pred, "control_action": pred[:, 1:9, :]}

        payload = {
            "state": {
                "inference_config": {
                    "agent": {"_target_": "dexmani_policy.agents.core.dp3.DP3Agent"},
                    "action_key": "action",
                    "action_dim": 19,
                    "horizon": 16,
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "eval": {"use_ema": False, "denoise_steps": 2},
                },
                "data_contract": {
                    "point_cloud_num_points": 1024,
                    "point_cloud_feature_dim": 6,
                },
            },
            "weights": {"model": {"weight": torch.ones(1)}, "ema_model": None},
        }
        agent = Agent()
        with patch("hydra.utils.instantiate", return_value=agent):
            exporter._verify_exported_model(payload)
        self.assertTrue(agent.strict)


@unittest.skipUnless(
    (Path(__file__).resolve().parents[3] / "dexmani_real").is_dir(),
    "sibling dexmani_real repository is unavailable",
)
class CrossRepositoryCompatibilityTest(unittest.TestCase):
    def test_current_real_checkpoint_and_sidecar_parsers_accept_export(self) -> None:
        sibling_root = Path(__file__).resolve().parents[3] / "dexmani_real"
        sys.path.insert(0, str(sibling_root))
        try:
            from dexmani_real.deployment.artifact import resolve_policy_artifact
            from dexmani_real.deployment.policy_checkpoint import (
                load_deployment_checkpoint_stream,
            )
            from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                experiment, zarr_path = _write_experiment(root)
                with patch.object(
                    exporter, "_producer_provenance", return_value=COMMIT
                ):
                    receipt = exporter.export_deployment_artifact(
                        experiment,
                        "latest",
                        verify=False,
                        zarr_path=zarr_path,
                    )
                artifact = resolve_policy_artifact(experiment)
                self.assertEqual(artifact.checkpoint_path, receipt.checkpoint_path)
                with receipt.checkpoint_path.open("rb") as stream:
                    loaded = load_deployment_checkpoint_stream(stream)
                self.assertEqual(loaded.producer["commit"], COMMIT)
                self.assertTrue(loaded.model_state)
                runtime = DexManiPolicyRuntime(SimpleNamespace(artifact=artifact))
                runtime._validate_artifact_receipt(loaded)
        finally:
            sys.path.remove(str(sibling_root))


if __name__ == "__main__":
    unittest.main()
