"""Export a resolved Policy experiment to the frozen Real deployment-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import zarr  # type: ignore[import-untyped]
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.config import register_resolvers

_REPOSITORY = "haoyangzhanglab/dexmani_policy"
_DEPLOYMENT_FORMAT = "dexmani.deployment.v2"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_POINT_COUNTS = frozenset({1024, 2048, 4096, 8192})
_POINT_FEATURE_DIM = 6
_MAX_ACTION_STEPS = 32
_POINT_SEMANTICS = {
    "point_cloud_frame": "xarm_base",
    "point_cloud_color_source": "mean_rgb_of_aligned_depth_pixels_per_voxel",
    "point_cloud_policy_id": "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9",
    "point_cloud_sampling": "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad",
    "point_cloud_transform": (
        "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
        "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
        "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
        "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
        "coarse_voxel_stratified_hash_or_cyclic_pad"
    ),
}

_CORE_ZARR_KEYS = (
    "schema_name",
    "schema_version",
    "domain",
    "profile",
    "task_name",
    "dt",
    "episode_start_policy",
    "obs_alignment",
    "observation_reference",
    "state_alignment",
    "max_observation_skew_s",
    "action_semantics",
    "arm_max_delta_rad_per_tick",
    "hand_max_delta_rad_per_tick",
    "endpoint_delta_tolerance_rad",
    "deployment_equivalent",
)
_POINT_ZARR_KEYS = (
    "point_cloud_frame",
    "point_cloud_color_source",
    "point_cloud_policy_id",
    "point_cloud_config_sha256",
    "point_cloud_table_plane_abcd_json",
    "point_cloud_sampling",
    "point_cloud_transform",
)


class DeploymentExportError(RuntimeError):
    """Base error for an invalid or failed deployment export."""


class InvalidExperimentError(DeploymentExportError):
    pass


class InvalidCheckpointError(DeploymentExportError):
    pass


class UnsupportedPolicyError(DeploymentExportError):
    pass


class InvalidZarrError(DeploymentExportError):
    pass


class ArtifactPublicationError(DeploymentExportError):
    pass


class ArtifactVerificationError(DeploymentExportError):
    pass


@dataclass(frozen=True)
class ExportReceipt:
    checkpoint_path: Path
    sidecar_path: Path
    selector_path: Path
    checkpoint_sha256: str
    producer_commit: str
    metadata_provenance: str
    checkpoint_selector: str


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise InvalidExperimentError("cannot establish Policy git provenance") from exc
    return result.stdout.strip()


def _producer_provenance(repo_root: Path) -> str:
    top_level = Path(_run_git(repo_root, "rev-parse", "--show-toplevel"))
    if top_level.resolve() != repo_root.resolve():
        raise InvalidExperimentError(
            "exporter package is not rooted in the Policy repository"
        )
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise InvalidExperimentError("Policy HEAD is not a lowercase 40-hex commit")
    if _run_git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise InvalidExperimentError(
            "Policy working tree must be clean for deployment export"
        )
    remote = _run_git(repo_root, "remote", "get-url", "origin")
    normalized = remote.removesuffix(".git").rstrip("/")
    if not (
        normalized.endswith("github.com/haoyangzhanglab/dexmani_policy")
        or normalized == "git@github.com:haoyangzhanglab/dexmani_policy"
    ):
        raise InvalidExperimentError(
            f"Policy origin does not identify {_REPOSITORY}: {remote!r}"
        )
    return commit


def _require_checkpoint_under_directory(candidate: Path, checkpoint_dir: Path) -> Path:
    try:
        resolved_directory = checkpoint_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidCheckpointError(
            f"cannot resolve checkpoint path: {candidate}"
        ) from exc
    try:
        resolved_candidate.relative_to(resolved_directory)
    except ValueError as exc:
        raise InvalidCheckpointError(
            f"checkpoint resolves outside experiment/checkpoints: {candidate}"
        ) from exc
    if not resolved_candidate.is_file():
        raise InvalidCheckpointError(f"checkpoint is not a file: {candidate}")
    return resolved_candidate


def _resolve_checkpoint(experiment_dir: Path, selector: str) -> Path:
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise InvalidExperimentError(
            f"checkpoint directory not found: {checkpoint_dir}"
        )

    candidate: Path
    if selector == "best":
        index_path = experiment_dir / "best_ckpt.json"
        if index_path.exists() or index_path.is_symlink():
            try:
                record = json.loads(index_path.read_text(encoding="utf-8"))
                if type(record) is not dict:
                    raise ValueError("best checkpoint record must be an object")
                raw_path = record.get("ckpt_relpath") or record.get("ckpt_path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("missing checkpoint path")
                indexed = Path(raw_path)
                candidate = (
                    indexed if indexed.is_absolute() else experiment_dir / indexed
                )
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                raise InvalidCheckpointError(
                    f"invalid best_ckpt.json: {index_path}"
                ) from exc
        else:
            candidate = checkpoint_dir / "best.pt"
    elif selector == "latest":
        candidate = checkpoint_dir / "latest.pt"
    elif selector.endswith("pct"):
        from dexmani_policy.training.eval_utils import discover_milestone_checkpoints

        try:
            percentage = int(selector.removesuffix("pct"))
        except ValueError as exc:
            raise InvalidCheckpointError(
                f"invalid milestone selector: {selector!r}"
            ) from exc
        matches = [
            item.path
            for item in discover_milestone_checkpoints(experiment_dir)
            if item.pct == percentage
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"milestone selector {selector!r} resolved {len(matches)} checkpoints"
            )
        candidate = matches[0]
    else:
        raw = Path(selector)
        candidate = raw if raw.is_absolute() else checkpoint_dir / raw

    return _require_checkpoint_under_directory(candidate, checkpoint_dir)


def _load_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise InvalidExperimentError(
            f"resolved experiment config not found: {config_path}"
        )
    register_resolvers()
    try:
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:
        raise InvalidExperimentError("experiment config is not fully resolved") from exc
    if type(plain) is not dict or type(plain.get("agent")) is not dict:
        raise InvalidExperimentError(
            "experiment config must contain a resolved agent mapping"
        )
    return cast(dict[str, Any], plain)


def _load_training_checkpoint(path: Path) -> TrainCheckpoint:
    try:
        checkpoint = CheckpointStore(path.parent).load(path)
    except Exception as exc:
        raise InvalidCheckpointError(
            f"cannot load simple.v1 checkpoint: {path}"
        ) from exc
    if checkpoint.epoch < 0 or checkpoint.global_step < 0:
        raise InvalidCheckpointError(
            "checkpoint epoch/global_step must be non-negative"
        )
    return checkpoint


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise InvalidExperimentError(f"{label} must be a positive integer")
    return value


def _require_finite_number(value: Any, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidZarrError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise InvalidZarrError(
            f"{label} must be {'positive and ' if positive else ''}finite"
        )
    return number


def _resolve_zarr_path(
    cfg_plain: dict[str, Any], repo_root: Path, override: Path | None
) -> Path:
    if override is not None:
        candidate = override.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    else:
        raw = cfg_plain.get("zarr_path")
        if not isinstance(raw, str) or not raw:
            raise UnsupportedPolicyError(
                "experiment has no single zarr_path; dynamic/multi-task datasets are unsupported"
            )
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidZarrError(f"Real Policy Zarr not found: {candidate}") from exc


def _dataset_modalities(cfg_plain: dict[str, Any]) -> list[str]:
    agent = cfg_plain.get("agent")
    if type(agent) is not dict:
        raise InvalidExperimentError("config.agent must be a plain mapping")
    if any(
        key in agent
        for key in (
            "rgb_backbone_name",
            "rgb_backbone_config",
            "text_encoder_model",
            "task_texts",
        )
    ):
        raise UnsupportedPolicyError(
            "RGB and dynamic task-text deployment are deferred"
        )
    dataset = cfg_plain.get("dataset")
    if type(dataset) is not dict:
        raise UnsupportedPolicyError("a single resolved dataset config is required")
    modalities = dataset.get("sensor_modalities")
    if type(modalities) is not list or any(
        type(item) is not str for item in modalities
    ):
        raise UnsupportedPolicyError(
            "dataset.sensor_modalities must be an explicit string list"
        )
    if "rgb" in modalities:
        raise UnsupportedPolicyError("RGB deployment is deferred")
    if set(modalities) != {"joint_state", "point_cloud"} or len(modalities) != 2:
        raise UnsupportedPolicyError(
            "first-phase deployment requires exactly joint_state + point_cloud"
        )
    return ["joint_state", "point_cloud"]


def _validate_agent_targets(value: Any, path: str = "agent") -> None:
    if type(value) is dict:
        target = value.get("_target_")
        if target is not None and (
            type(target) is not str or not target.startswith("dexmani_policy.agents.")
        ):
            raise UnsupportedPolicyError(
                f"deployment target at {path} must be under dexmani_policy.agents"
            )
        for key, nested in value.items():
            _validate_agent_targets(nested, f"{path}.{key}")
    elif type(value) is list:
        for index, nested in enumerate(value):
            _validate_agent_targets(nested, f"{path}[{index}]")


def _point_array_shape(root: Any) -> tuple[int, int]:
    try:
        array = root["data"]["point_cloud"]
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise InvalidZarrError("Zarr data/point_cloud is missing") from exc
    if len(shape) != 3:
        raise InvalidZarrError("Zarr point_cloud must have shape [T, N, F]")
    count, feature_dim = shape[-2:]
    if count not in _POINT_COUNTS or feature_dim != _POINT_FEATURE_DIM:
        raise InvalidZarrError(
            f"unsupported point-cloud shape: N={count}, feature_dim={feature_dim}"
        )
    return count, feature_dim


def _validate_required_zarr_arrays(root: Any, cfg_plain: dict[str, Any]) -> None:
    expected_dims = {"joint_state": 19, "action": 19, "action_ee": 21}
    try:
        arrays = root["data"]
        shapes = {
            key: tuple(int(value) for value in arrays[key].shape)
            for key in expected_dims
        }
    except Exception as exc:
        raise InvalidZarrError(
            "Zarr must contain joint_state, action, and action_ee arrays"
        ) from exc
    lengths = {shape[0] for shape in shapes.values() if len(shape) == 2}
    if (
        len(lengths) != 1
        or not lengths
        or next(iter(lengths)) <= 0
        or any(
            len(shapes[key]) != 2 or shapes[key][1] != dim
            for key, dim in expected_dims.items()
        )
    ):
        raise InvalidZarrError(f"Zarr action/state dimensions are invalid: {shapes}")
    action_key = cfg_plain.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise InvalidZarrError(f"config action_key is invalid: {action_key!r}")


def _validate_json_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise InvalidZarrError(f"{label} must be a non-empty JSON string")
    try:
        parsed = json.loads(
            value, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
        )
        json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidZarrError(f"{label} must contain finite JSON") from exc
    return value


def _validate_table_plane(value: Any) -> str:
    encoded = _validate_json_string(value, "point_cloud_table_plane_abcd_json")
    plane = json.loads(encoded)
    if plane is None:
        if encoded != "null":
            raise InvalidZarrError("point-cloud table plane JSON must be canonical")
        return encoded
    if (
        type(plane) is not list
        or len(plane) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in plane
        )
        or any(not math.isfinite(float(item)) for item in plane)
    ):
        raise InvalidZarrError(
            "point-cloud table plane must be null or four finite numbers"
        )
    normal_norm = math.sqrt(sum(float(item) ** 2 for item in plane[:3]))
    if normal_norm <= 0.0 or float(plane[2]) / normal_norm <= 0.0:
        raise InvalidZarrError("point-cloud table plane normal must point upward")
    canonical = json.dumps(plane, allow_nan=False, separators=(",", ":"))
    if encoded != canonical:
        raise InvalidZarrError("point-cloud table plane JSON must be canonical")
    return encoded


def _validate_zarr_contract(
    path: Path,
    cfg_plain: dict[str, Any],
    sensor_modalities: list[str],
) -> dict[str, Any]:
    try:
        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
    except Exception as exc:
        raise InvalidZarrError(f"cannot open Real Policy Zarr: {path}") from exc
    missing = [key for key in (*_CORE_ZARR_KEYS, *_POINT_ZARR_KEYS) if key not in attrs]
    if missing:
        raise InvalidZarrError(f"Real Policy Zarr is missing semantic attrs: {missing}")
    expected_fixed = {
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 5,
        "domain": "real",
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_reference": "camera_source_monotonic_ns",
        "state_alignment": "camera_source_aligned_state",
        "action_semantics": "deployment_grid_rate_limited_target",
        "deployment_equivalent": True,
    }
    mismatches = {
        key: (attrs.get(key), value)
        for key, value in expected_fixed.items()
        if attrs.get(key) != value
    }
    if mismatches:
        raise InvalidZarrError(f"invalid Real Policy Zarr semantics: {mismatches}")
    if attrs["profile"] not in {"pointcloud", "rgb_pc"}:
        raise InvalidZarrError(
            "point-cloud policy requires pointcloud or rgb_pc profile"
        )
    task_name = cfg_plain.get("task_name")
    if type(task_name) is not str or not task_name or attrs["task_name"] != task_name:
        raise InvalidZarrError(
            f"Zarr task_name={attrs['task_name']!r} does not match config task_name={task_name!r}"
        )
    dt = _require_finite_number(attrs["dt"], "Zarr dt", positive=True)
    for config_dt in (
        cfg_plain.get("dt"),
        cfg_plain.get("control_dt_s"),
        cfg_plain.get("dataset", {}).get("dt"),
    ):
        if config_dt is not None and (
            isinstance(config_dt, bool)
            or not isinstance(config_dt, (int, float))
            or not math.isclose(float(config_dt), dt, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise InvalidZarrError(
                f"Zarr dt={dt!r} conflicts with experiment dt={config_dt!r}"
            )
    _require_finite_number(
        attrs["max_observation_skew_s"], "max_observation_skew_s", positive=True
    )
    arm_delta = attrs["arm_max_delta_rad_per_tick"]
    if arm_delta is not None:
        _require_finite_number(arm_delta, "arm_max_delta_rad_per_tick", positive=True)
    _require_finite_number(
        attrs["hand_max_delta_rad_per_tick"],
        "hand_max_delta_rad_per_tick",
        positive=True,
    )
    endpoint = _require_finite_number(
        attrs["endpoint_delta_tolerance_rad"],
        "endpoint_delta_tolerance_rad",
        positive=False,
    )
    if endpoint < 0.0:
        raise InvalidZarrError("endpoint_delta_tolerance_rad must be non-negative")
    for key in ("profile", "task_name"):
        if type(attrs[key]) is not str or not attrs[key]:
            raise InvalidZarrError(f"Zarr {key} must be a non-empty string")
    point_mismatches = {
        key: (attrs.get(key), expected)
        for key, expected in _POINT_SEMANTICS.items()
        if attrs.get(key) != expected
    }
    if point_mismatches:
        raise InvalidZarrError(
            f"invalid Real point-cloud semantics: {point_mismatches}"
        )
    if _SHA256_RE.fullmatch(str(attrs["point_cloud_config_sha256"])) is None:
        raise InvalidZarrError("point_cloud_config_sha256 must be lowercase SHA-256")
    _validate_table_plane(attrs["point_cloud_table_plane_abcd_json"])
    point_count, point_feature_dim = _point_array_shape(root)
    _validate_required_zarr_arrays(root, cfg_plain)

    agent = cfg_plain["agent"]
    configured_count = agent.get("num_points")
    if configured_count is not None and configured_count != point_count:
        raise InvalidZarrError(
            f"Zarr point count={point_count} conflicts with agent.num_points={configured_count}"
        )
    configured_feature_dims = [agent.get("pc_dim")]
    pc_encoder = agent.get("pc_encoder_config")
    if type(pc_encoder) is dict:
        configured_feature_dims.append(pc_encoder.get("pc_in_channels"))
    for configured_dim in configured_feature_dims:
        if configured_dim is not None and configured_dim != point_feature_dim:
            raise InvalidZarrError(
                f"Zarr point feature dim={point_feature_dim} conflicts with agent config={configured_dim}"
            )

    contract = {key: attrs[key] for key in _CORE_ZARR_KEYS}
    contract.update({key: attrs[key] for key in _POINT_ZARR_KEYS})
    contract.update(
        {
            "sensor_modalities": list(sensor_modalities),
            "point_cloud_num_points": point_count,
            "point_cloud_feature_dim": point_feature_dim,
        }
    )
    return _require_plain_metadata(contract, "data_contract")


def _expected_train_params(cfg_plain: dict[str, Any]) -> dict[str, Any]:
    action_key = cfg_plain.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise InvalidExperimentError(f"unsupported action_key: {action_key!r}")
    action_dim = _require_positive_int(cfg_plain.get("action_dim"), "action_dim")
    use_aux_ee = cfg_plain.get("use_aux_ee", False)
    if type(use_aux_ee) is not bool:
        raise InvalidExperimentError("use_aux_ee must be bool")
    agent = cfg_plain["agent"]
    is_codebook_agent = "codebook_path" in agent
    tcp_dim = agent.get("tcp_dim") if is_codebook_agent else None
    hand_dim = (
        action_dim - tcp_dim if is_codebook_agent and type(tcp_dim) is int else None
    )
    control_action_dim = 19 if use_aux_ee else action_dim
    expected = {
        "n_obs_steps": _require_positive_int(
            cfg_plain.get("n_obs_steps"), "n_obs_steps"
        ),
        "n_action_steps": _require_positive_int(
            cfg_plain.get("n_action_steps"), "n_action_steps"
        ),
        "action_dim": action_dim,
        "horizon": _require_positive_int(cfg_plain.get("horizon"), "horizon"),
        "action_key": action_key,
        "tcp_dim": tcp_dim,
        "hand_dim": hand_dim,
        "control_action_dim": control_action_dim,
        "use_aux_ee": use_aux_ee,
    }
    if expected["n_obs_steps"] - 1 + expected["n_action_steps"] > expected["horizon"]:
        raise InvalidExperimentError("observation/action window exceeds horizon")
    expected_control = 21 if action_key == "action_ee" else 19
    if use_aux_ee:
        if action_key != "action" or action_dim != 28 or control_action_dim != 19:
            raise InvalidExperimentError(
                "use_aux_ee requires joint19_ee9 action layout"
            )
    elif action_dim != expected_control:
        raise InvalidExperimentError("action_dim does not match action_key")
    return expected


def _reconcile_train_params(
    checkpoint: TrainCheckpoint, cfg_plain: dict[str, Any]
) -> tuple[dict[str, Any], str, list[str]]:
    if type(checkpoint.train_params) is not dict:
        raise InvalidCheckpointError("checkpoint train_params must be a plain dict")
    native = checkpoint.train_params
    expected = _expected_train_params(cfg_plain)
    retrofitted: list[str] = []
    result: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key not in native:
            if key == "use_aux_ee":
                result[key] = expected_value
                retrofitted.append(key)
                continue
            raise InvalidCheckpointError(f"checkpoint train_params is missing {key}")
        if native[key] != expected_value:
            raise InvalidCheckpointError(
                f"checkpoint train_params.{key}={native[key]!r} conflicts with config={expected_value!r}"
            )
        result[key] = native[key]
    if "num_training_steps" in native:
        result["num_training_steps"] = native["num_training_steps"]
    provenance = "retrofitted" if retrofitted else "native"
    return _require_plain_metadata(result, "train_params"), provenance, retrofitted


def _validate_resolved_config_contract(
    cfg_plain: dict[str, Any], train: dict[str, Any]
) -> None:
    agent = cfg_plain.get("agent")
    if type(agent) is not dict:
        raise InvalidExperimentError("resolved config agent must be a plain dict")
    for field in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if field not in agent:
            raise InvalidExperimentError(
                f"resolved config agent is missing deployment-critical field {field}"
            )
        agent_value = _require_positive_int(agent[field], f"agent.{field}")
        if agent_value != train[field]:
            raise InvalidExperimentError(
                f"resolved config agent.{field}={agent_value!r} conflicts with "
                f"top-level {field}={train[field]!r}"
            )

    dataset = cfg_plain.get("dataset")
    if type(dataset) is not dict:
        raise InvalidExperimentError("resolved config dataset must be a plain dict")
    expected_dataset = {
        "action_key": train["action_key"],
        "horizon": train["horizon"],
        "obs_horizon": train["n_obs_steps"],
        "pad_before": train["n_obs_steps"] - 1,
        "pad_after": train["n_action_steps"] - 1,
        "use_aux_ee": train["use_aux_ee"],
    }
    for field, expected in expected_dataset.items():
        if field in dataset and (
            type(dataset[field]) is not type(expected) or dataset[field] != expected
        ):
            raise InvalidExperimentError(
                f"resolved config dataset.{field}={dataset[field]!r} conflicts with "
                f"deployment contract={expected!r}"
            )


def _canonicalize_state_dict(value: Any, label: str) -> dict[str, torch.Tensor]:
    if type(value) is not dict or not value:
        raise InvalidCheckpointError(f"{label} must be a non-empty plain state_dict")
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if type(raw_key) is not str or not raw_key or type(tensor) is not torch.Tensor:
            raise InvalidCheckpointError(
                f"{label} must map non-empty strings to tensors"
            )
        key = raw_key.replace("_orig_mod.", "")
        key = key.removeprefix("module.")
        if not key or key.startswith("module.") or "_orig_mod." in key:
            raise InvalidCheckpointError(
                f"{label} contains a non-canonical key: {raw_key!r}"
            )
        if key in result:
            raise InvalidCheckpointError(
                f"{label} canonicalization collides at {key!r}"
            )
        result[key] = tensor.detach().cpu()
    return result


def _sanitize_agent_config(
    agent_config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    train: dict[str, Any],
) -> dict[str, Any]:
    sanitized = _require_plain_metadata(agent_config, "agent config")
    sanitized = json.loads(_canonical_json(sanitized))
    _validate_agent_targets(sanitized)
    if "codebook_path" in sanitized:
        required_suffixes = (
            "codebook_manager.sorted_hand_poses",
            "codebook_manager.pca_permutation",
            "codebook_manager.layer_weights",
        )
        found: dict[str, torch.Tensor] = {}
        for suffix in required_suffixes:
            matches = [
                tensor for key, tensor in model_state.items() if key.endswith(suffix)
            ]
            if len(matches) != 1 or matches[0].numel() == 0:
                raise UnsupportedPolicyError(
                    "DQ-RISE checkpoint has no complete persistent runtime codebook state"
                )
            found[suffix] = matches[0]
        poses = found["codebook_manager.sorted_hand_poses"]
        groups = sanitized.get("codebook_num_groups")
        size = sanitized.get("codebook_size")
        if (
            type(groups) is not int
            or type(size) is not int
            or poses.ndim != 2
            or poses.shape[0] != size**groups
            or poses.shape[1] != train["hand_dim"]
            or train["tcp_dim"] != sanitized.get("tcp_dim")
            or train["action_key"] != "action_ee"
        ):
            raise InvalidCheckpointError(
                "DQ-RISE codebook/config/action metadata conflict"
            )
        sanitized["codebook_path"] = None
    pc_encoder = sanitized.get("pc_encoder_config")
    if type(pc_encoder) is dict and "use_pretrained_weights" in pc_encoder:
        pc_encoder["use_pretrained_weights"] = False
    return sanitized


def _build_inference_config(
    cfg_plain: dict[str, Any], agent_config: dict[str, Any], train: dict[str, Any]
) -> dict[str, Any]:
    eval_config = cfg_plain.get("eval")
    if type(eval_config) is not dict:
        raise InvalidExperimentError("config.eval must be a mapping")
    if eval_config.get("denoise_timesteps_list") is not None:
        raise UnsupportedPolicyError(
            "eval.denoise_timesteps_list is unsupported for deployment-v2"
        )
    denoise_steps = _require_positive_int(
        eval_config.get("denoise_steps"), "eval.denoise_steps"
    )
    use_ema = eval_config.get("use_ema")
    if type(use_ema) is not bool:
        raise InvalidExperimentError("eval.use_ema must be bool")
    inference = {
        "task_name": cfg_plain["task_name"],
        "action_key": train["action_key"],
        "action_dim": train["action_dim"],
        "horizon": train["horizon"],
        "n_obs_steps": train["n_obs_steps"],
        "n_action_steps": train["n_action_steps"],
        "use_aux_ee": train["use_aux_ee"],
        "agent": agent_config,
        "eval": {"use_ema": use_ema, "denoise_steps": denoise_steps},
    }
    return _require_plain_metadata(inference, "inference_config")


def _augment_data_contract(
    zarr_contract: dict[str, Any], train: dict[str, Any]
) -> dict[str, Any]:
    result = dict(zarr_contract)
    result.update(
        {
            "action_key": train["action_key"],
            "model_action_dim": train["action_dim"],
            "horizon": train["horizon"],
            "n_obs_steps": train["n_obs_steps"],
            "n_action_steps": train["n_action_steps"],
            "pad_before": train["n_obs_steps"] - 1,
            "pad_after": train["n_action_steps"] - 1,
            "padding_semantics": "repeat_edge",
            "use_aux_ee": train["use_aux_ee"],
        }
    )
    return _require_plain_metadata(result, "data_contract")


def _require_plain_metadata(value: Any, label: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidCheckpointError(f"{label} contains a non-finite float")
        return value
    if type(value) is list:
        return [_require_plain_metadata(item, label) for item in value]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise InvalidCheckpointError(f"{label} contains a non-string key")
            result[key] = _require_plain_metadata(item, label)
        return result
    raise InvalidCheckpointError(
        f"{label} contains non-plain metadata: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidCheckpointError("metadata must be canonical finite JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_allocation(
    inference: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    action_key = inference["action_key"]
    control_action_dim = 21 if action_key == "action_ee" else 19
    auxiliary_layout = "joint19_ee9" if inference["use_aux_ee"] else "none"
    allocation = {
        "task_name": inference["task_name"],
        "action_key": action_key,
        "action_dim": inference["action_dim"],
        "control_action_dim": control_action_dim,
        "auxiliary_action_layout": auxiliary_layout,
        "n_obs_steps": inference["n_obs_steps"],
        "n_action_steps": inference["n_action_steps"],
        "horizon": inference["horizon"],
        "required_action_steps": inference["horizon"] - (inference["n_obs_steps"] - 1),
        "control_dt_s": data["dt"],
        "sensor_modalities": data["sensor_modalities"],
        "observation_fields": ["arm_qpos", "hand_qpos", "point_cloud"],
        "requires_hand": True,
        "point_cloud_num_points": data["point_cloud_num_points"],
        "point_cloud_feature_dim": data["point_cloud_feature_dim"],
        "rgb_shape": None,
        "rgb_color_order": None,
        "rgb_value_range": None,
    }
    required_steps = allocation["required_action_steps"]
    if required_steps <= 0 or required_steps > _MAX_ACTION_STEPS:
        raise InvalidExperimentError(
            "required_action_steps exceeds the Real IPC contract"
        )
    return allocation


def _validate_payload(payload: Any) -> None:
    if type(payload) is not dict or set(payload) != {"_format", "state", "weights"}:
        raise ArtifactVerificationError("deployment checkpoint payload schema mismatch")
    if payload["_format"] != _DEPLOYMENT_FORMAT:
        raise ArtifactVerificationError("deployment checkpoint format mismatch")
    state = payload["state"]
    weights = payload["weights"]
    if type(state) is not dict or set(state) != {
        "epoch",
        "global_step",
        "train_params",
        "inference_config",
        "data_contract",
        "producer",
        "deployment_contract",
    }:
        raise ArtifactVerificationError("deployment checkpoint state schema mismatch")
    if type(weights) is not dict or set(weights) != {"model", "ema_model"}:
        raise ArtifactVerificationError("deployment checkpoint weights schema mismatch")
    _canonicalize_state_dict(weights["model"], "weights.model")
    if weights["ema_model"] is not None:
        _canonicalize_state_dict(weights["ema_model"], "weights.ema_model")
    for name in (
        "train_params",
        "inference_config",
        "data_contract",
        "producer",
        "deployment_contract",
    ):
        _require_plain_metadata(state[name], name)


def _verify_exported_model(payload: dict[str, Any]) -> None:
    state = payload["state"]
    inference = state["inference_config"]
    selected = (
        payload["weights"]["ema_model"]
        if inference["eval"]["use_ema"]
        else payload["weights"]["model"]
    )
    if selected is None:
        raise ArtifactVerificationError("eval.use_ema requires EMA weights")
    try:
        agent = hydra.utils.instantiate(OmegaConf.create(inference["agent"]))
        agent.action_key = inference["action_key"]
        agent.load_state_dict(selected, strict=True)
        agent.to("cpu")
        agent.eval()
        required_normalizers = ["action", "joint_state", "point_cloud"]
        if not agent.normalizer.is_fitted(required_keys=required_normalizers):
            raise RuntimeError("normalizer is missing required deployment state")
        normalizer_dims = {
            "action": inference["action_dim"],
            "joint_state": 19,
            "point_cloud": state["data_contract"]["point_cloud_feature_dim"],
        }
        for key, expected_dim in normalizer_dims.items():
            params = agent.normalizer.params_dict[key]
            scale = params.get("scale")
            offset = params.get("offset")
            if (
                scale is None
                or offset is None
                or scale.numel() != expected_dim
                or offset.numel() != expected_dim
                or not bool(torch.isfinite(scale).all())
                or not bool(torch.isfinite(offset).all())
                or bool(torch.any(scale == 0))
            ):
                raise RuntimeError(f"normalizer {key!r} has invalid deployment state")
        obs = {
            "joint_state": torch.zeros(
                (1, inference["n_obs_steps"], 19), dtype=torch.float32
            ),
            "point_cloud": torch.zeros(
                (
                    1,
                    inference["n_obs_steps"],
                    state["data_contract"]["point_cloud_num_points"],
                    state["data_contract"]["point_cloud_feature_dim"],
                ),
                dtype=torch.float32,
            ),
        }
        with torch.inference_mode():
            torch.manual_seed(0)
            result = agent.predict_action(
                obs, denoise_timesteps=inference["eval"]["denoise_steps"]
            )
        pred = result["pred_action"]
        control = result["control_action"]
        control_dim = 21 if inference["action_key"] == "action_ee" else 19
        if tuple(pred.shape) != (1, inference["horizon"], inference["action_dim"]):
            raise RuntimeError(f"pred_action shape mismatch: {tuple(pred.shape)}")
        if tuple(control.shape) != (1, inference["n_action_steps"], control_dim):
            raise RuntimeError(f"control_action shape mismatch: {tuple(control.shape)}")
        if not bool(torch.isfinite(pred).all()) or not bool(
            torch.isfinite(control).all()
        ):
            raise RuntimeError("prediction contains NaN/Inf")
        start = inference["n_obs_steps"] - 1
        expected = pred[:, start : start + inference["n_action_steps"], :control_dim]
        if not torch.equal(control, expected):
            raise RuntimeError(
                "control_action is not the exact canonical pred_action slice"
            )
    except ArtifactVerificationError:
        raise
    except Exception as exc:
        raise ArtifactVerificationError(
            "deployment agent strict restore/prediction failed"
        ) from exc


def _write_checkpoint_temp(directory: Path, payload: dict[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".deployment-checkpoint-", suffix=".tmp", dir=directory
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_sidecar_temp(directory: Path, sidecar: dict[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".deployment-sidecar-", suffix=".tmp", dir=directory
    )
    path = Path(raw_path)
    try:
        payload = _canonical_json(sidecar).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_deployment_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactVerificationError(
            "cannot safely reload deployment checkpoint"
        ) from exc
    _validate_payload(payload)
    return payload


def _roundtrip_verify_files(
    checkpoint_path: Path, sidecar_path: Path, expected_sidecar: dict[str, Any]
) -> None:
    _load_deployment_payload(checkpoint_path)
    try:
        raw = sidecar_path.read_bytes()
        decoded = json.loads(
            raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError("cannot reload deployment sidecar") from exc
    if raw != _canonical_json(decoded).encode("utf-8") or decoded != expected_sidecar:
        raise ArtifactVerificationError(
            "deployment sidecar is not canonical or changed"
        )
    checkpoint = decoded["checkpoint"]
    if (
        checkpoint["filename"] != checkpoint_path.name
        or checkpoint["size_bytes"] != checkpoint_path.stat().st_size
        or checkpoint["sha256"] != _sha256_file(checkpoint_path)
    ):
        raise ArtifactVerificationError(
            "deployment sidecar checkpoint identity mismatch"
        )


def _capture_selector(selector_path: Path) -> tuple[bool, str | None]:
    if selector_path.is_symlink():
        return True, os.readlink(selector_path)
    if selector_path.exists():
        raise ArtifactPublicationError(
            "refusing to replace a non-symlink deployment_latest.pt selector"
        )
    return False, None


def _replace_relative_symlink(selector_path: Path, target: str) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".deployment-selector-", dir=selector_path.parent
    )
    os.close(descriptor)
    temp = Path(raw_path)
    temp.unlink()
    try:
        temp.symlink_to(target)
        os.replace(temp, selector_path)
        _fsync_directory(selector_path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _rollback_selector(selector_path: Path, old: tuple[bool, str | None]) -> None:
    existed, target = old
    if existed:
        assert target is not None
        _replace_relative_symlink(selector_path, target)
    elif selector_path.is_symlink() or selector_path.exists():
        selector_path.unlink()
        _fsync_directory(selector_path.parent)


def _roundtrip_verify_published(
    selector_path: Path,
    checkpoint_path: Path,
    sidecar_path: Path,
    sidecar: dict[str, Any],
) -> None:
    if (
        not selector_path.is_symlink()
        or os.readlink(selector_path) != checkpoint_path.name
    ):
        raise ArtifactVerificationError(
            "deployment selector is not the expected relative symlink"
        )
    if selector_path.resolve(strict=True) != checkpoint_path.resolve(strict=True):
        raise ArtifactVerificationError(
            "deployment selector resolves to the wrong checkpoint"
        )
    _roundtrip_verify_files(checkpoint_path, sidecar_path, sidecar)


def export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    verify: bool = True,
    zarr_path: Path | None = None,
) -> ExportReceipt:
    """Export one selected simple.v1 checkpoint and atomically publish its selector."""
    repo_root = Path(__file__).resolve().parents[2]
    producer_commit = _producer_provenance(repo_root)
    try:
        experiment = Path(experiment_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise InvalidExperimentError(
            f"experiment directory not found: {experiment_dir}"
        ) from exc
    if not experiment.is_dir():
        raise InvalidExperimentError(
            f"experiment path is not a directory: {experiment}"
        )
    selected_path = _resolve_checkpoint(experiment, checkpoint_selector)
    cfg_plain = _load_config(experiment)
    sensor_modalities = _dataset_modalities(cfg_plain)
    resolved_zarr = _resolve_zarr_path(cfg_plain, repo_root, zarr_path)
    zarr_contract = _validate_zarr_contract(resolved_zarr, cfg_plain, sensor_modalities)
    checkpoint = _load_training_checkpoint(selected_path)
    train, metadata_provenance, retrofitted = _reconcile_train_params(
        checkpoint, cfg_plain
    )
    _validate_resolved_config_contract(cfg_plain, train)
    model_state = _canonicalize_state_dict(checkpoint.model_state, "weights.model")
    ema_state = (
        None
        if checkpoint.ema_model_state is None
        else _canonicalize_state_dict(checkpoint.ema_model_state, "weights.ema_model")
    )
    agent_config = _sanitize_agent_config(cfg_plain["agent"], model_state, train)
    inference = _build_inference_config(cfg_plain, agent_config, train)
    if inference["eval"]["use_ema"] and ema_state is None:
        raise InvalidCheckpointError(
            "eval.use_ema=true requires checkpoint EMA weights"
        )
    if inference["eval"]["use_ema"] and "codebook_path" in cfg_plain["agent"]:
        assert ema_state is not None
        _sanitize_agent_config(cfg_plain["agent"], ema_state, train)
    data_contract = _augment_data_contract(zarr_contract, train)
    producer = {
        "repository": _REPOSITORY,
        "commit": producer_commit,
        "metadata_provenance": metadata_provenance,
        "retrofitted_train_params_fields": retrofitted,
    }
    deployment_contract = {
        "schema_version": 1,
        "inference_config": inference,
        "data_contract": data_contract,
        "train_params": train,
        "producer": producer,
        "retrofitted_train_params_fields": retrofitted,
    }
    payload = {
        "_format": _DEPLOYMENT_FORMAT,
        "state": {
            "epoch": int(checkpoint.epoch),
            "global_step": int(checkpoint.global_step),
            "train_params": train,
            "inference_config": inference,
            "data_contract": data_contract,
            "producer": producer,
            "deployment_contract": deployment_contract,
        },
        "weights": {"model": model_state, "ema_model": ema_state},
    }
    _validate_payload(payload)

    checkpoint_dir = experiment / "checkpoints"
    if output_path is None:
        final_path = checkpoint_dir / f"{selected_path.stem}-deployment-v2.pt"
    else:
        requested = Path(output_path)
        final_path = (
            requested if requested.is_absolute() else checkpoint_dir / requested
        )
    final_path = final_path.absolute()
    if (
        final_path.parent.resolve() != checkpoint_dir.resolve()
        or final_path.suffix != ".pt"
    ):
        raise ArtifactPublicationError(
            "output_path must be a .pt file in experiment/checkpoints"
        )
    sidecar_path = final_path.with_name(f"{final_path.name}.deployment.json")
    selector_path = checkpoint_dir / "deployment_latest.pt"
    if (
        final_path.exists()
        or final_path.is_symlink()
        or sidecar_path.exists()
        or sidecar_path.is_symlink()
    ):
        raise FileExistsError(
            f"refusing to overwrite deployment artifact: {final_path}"
        )
    old_selector = _capture_selector(selector_path)

    checkpoint_temp: Path | None = None
    sidecar_temp: Path | None = None
    selector_published = False
    try:
        checkpoint_temp = _write_checkpoint_temp(checkpoint_dir, payload)
        os.replace(checkpoint_temp, final_path)
        checkpoint_temp = None
        _fsync_directory(checkpoint_dir)
        reloaded_payload = _load_deployment_payload(final_path)
        if verify:
            _verify_exported_model(reloaded_payload)
        checkpoint_sha256 = _sha256_file(final_path)
        allocation = _build_allocation(inference, data_contract)
        embedded_hash = _sha256_bytes(
            _canonical_json(deployment_contract).encode("utf-8")
        )
        sidecar = {
            "schema_version": 2,
            "checkpoint": {
                "filename": final_path.name,
                "size_bytes": final_path.stat().st_size,
                "sha256": checkpoint_sha256,
            },
            "embedded_contract_sha256": embedded_hash,
            "allocation": allocation,
            "producer": {
                "repository": _REPOSITORY,
                "commit": producer_commit,
                "metadata_provenance": metadata_provenance,
            },
        }
        sidecar_temp = _write_sidecar_temp(checkpoint_dir, sidecar)
        os.replace(sidecar_temp, sidecar_path)
        sidecar_temp = None
        _fsync_directory(checkpoint_dir)
        _roundtrip_verify_files(final_path, sidecar_path, sidecar)
        if _producer_provenance(repo_root) != producer_commit:
            raise ArtifactPublicationError(
                "Policy producer commit changed during deployment export"
            )
        selector_published = True
        _replace_relative_symlink(selector_path, final_path.name)
        _roundtrip_verify_published(selector_path, final_path, sidecar_path, sidecar)
    except BaseException as exc:
        if selector_published:
            try:
                _rollback_selector(selector_path, old_selector)
            except Exception as rollback_exc:
                raise ArtifactPublicationError(
                    "deployment publication failed and selector rollback also failed"
                ) from rollback_exc
        if isinstance(exc, DeploymentExportError):
            raise
        raise ArtifactPublicationError(
            "deployment artifact publication failed"
        ) from exc
    finally:
        if checkpoint_temp is not None:
            checkpoint_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)

    return ExportReceipt(
        checkpoint_path=final_path,
        sidecar_path=sidecar_path,
        selector_path=selector_path,
        checkpoint_sha256=checkpoint_sha256,
        producer_commit=producer_commit,
        metadata_provenance=metadata_provenance,
        checkpoint_selector=checkpoint_selector,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--zarr-path", type=Path, default=None)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="strict-restore and run one deterministic synthetic prediction (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = export_deployment_artifact(
        args.experiment_dir,
        checkpoint_selector=args.checkpoint,
        output_path=args.output,
        verify=args.verify,
        zarr_path=args.zarr_path,
    )
    print(
        _canonical_json(
            {
                "checkpoint_path": str(receipt.checkpoint_path),
                "sidecar_path": str(receipt.sidecar_path),
                "selector_path": str(receipt.selector_path),
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "producer_commit": receipt.producer_commit,
                "metadata_provenance": receipt.metadata_provenance,
                "checkpoint_selector": receipt.checkpoint_selector,
            }
        )
    )


if __name__ == "__main__":
    main()
