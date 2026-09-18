"""Export one selected training checkpoint to an explicit Real deployment artifact.

Ownership at this boundary: the selected checkpoint owns the trained
deployment semantics (agent constructor, action/window contract,
normalization, dataset/preprocessing), the experiment ``config.yaml`` owns only
identity plus the inference recipe, and the resulting artifact owns the selected
weights and an immutable observation/action contract.

The public :func:`export_deployment_artifact` always verifies — a safe
``weights_only`` reload, a strict restore and one deterministic synthetic
prediction — before it publishes, so no researcher-facing path can publish an
unverified artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zarr  # type: ignore[import-untyped]
from omegaconf import OmegaConf

from dexmani_policy.agents.obs_encoder.rgb.image_processor import (
    IMAGE_PROCESSOR_PRESETS,
)
from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TrainCheckpoint,
    make_normalization_contract,
    parse_normalization_contract,
)
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.datasets.base_dataset import DEFAULT_RGB_KEEP_UINT8
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DeploymentContractError,
    parse_deployment_contract,
    validate_agent_targets,
)
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    verify_deployment_prediction,
)

_SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {
        "joint_state",
        "point_cloud",
        "rgb",
        "contact_force",
        "fingertip_points",
        "eef_pose",
        "tactile_force",
    }
)
_POINT_COUNTS = frozenset({1024, 2048, 4096, 8192})
_POINT_FEATURE_DIM = 6
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

_FINGERTIP_SEMANTICS = {
    "fingertip_points_frame": "xarm_base",
    "fingertip_points_unit": "m",
    "fingertip_points_derivation": "fk_from_processed_joint_state",
    "fingertip_points_policy_id": "arm_hand_fk_from_joint_state_v1",
}
# Canonical EEF observation identity; deploy-time FK must resolve to the same
# derivation and algorithm so a silent frame/FK drift cannot change the values.
_EEF_POSE_SEMANTICS = {
    "eef_pose_frame": "xarm_base",
    "eef_pose_components": "position_m(3)+rot6d(6)",
    "eef_pose_derivation": "canonical_arm_fk_from_aligned_qpos",
    "eef_pose_algorithm_id": "xarm7_custom_eef_pinocchio_fk_v1",
}
# Dense tactile ordering/axis identity; unit honesty is checked separately
# because XHand SDK native values are not proven to be Newtons.
_TACTILE_FORCE_SEMANTICS = {
    "tactile_force_representation": "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
    "tactile_force_finger_order": "thumb_index_mid_ring_pinky",
    "tactile_force_sensor_order": "xhand_sdk_sensor_data_order",
    "tactile_force_point_order": "xhand_sdk_sensor_data_raw_force_order",
    "tactile_force_axis_labels": "fx_fy_fz",
}
# Members the Real producer guarantees inside each canonical config JSON attr.
_PROCESSING_CONFIG_MEMBERS = frozenset({"pointcloud", "table_plane_abcd"})
_FINGERTIP_CONFIG_MEMBERS = frozenset(
    {
        "fingertip_link_names",
        "handbase_position_eef_m",
        "handbase_quat_eef_wxyz",
    }
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
    selector_path: Path
    checkpoint_selector: str


@dataclass(frozen=True)
class _CheckpointDeploymentSource:
    """The trained deployment semantics owned by one training checkpoint."""

    agent: dict[str, Any]
    agent_config: dict[str, Any]
    dataset: dict[str, Any]
    normalization_contract: dict[str, Any]
    observation_fields: tuple[str, ...]


@dataclass(frozen=True)
class _SelectedInferenceSettings:
    use_ema: bool
    denoise_steps: int


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
        from dexmani_policy.training.eval_utils import read_best_ckpt_json

        try:
            record = read_best_ckpt_json(experiment_dir)
        except (OSError, ValueError) as exc:
            raise InvalidCheckpointError(
                f"invalid best_ckpt.json: {experiment_dir / 'best_ckpt.json'}"
            ) from exc
        candidate = experiment_dir / record["ckpt_relpath"]
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
    """Load the resolved experiment config for identity and the inference recipe.

    Deployment reads only ``policy_name`` / ``task_name`` and ``eval`` from
    here.  The ``agent`` / ``dataset`` / ``normalization`` sections are
    deliberately *not* consulted, so editing them after training cannot change
    what an existing checkpoint deploys as — that semantics belongs to
    :func:`_parse_checkpoint_deployment_source`.
    """
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
    if type(plain) is not dict:
        raise InvalidExperimentError("experiment config must be a resolved mapping")
    return cast(dict[str, Any], plain)


def _load_training_checkpoint(path: Path) -> TrainCheckpoint:
    try:
        checkpoint = CheckpointStore(path.parent).load(path)
    except Exception as exc:
        raise InvalidCheckpointError(
            f"cannot load simple.v3 checkpoint: {path}"
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
    dataset: Mapping[str, Any], repo_root: Path, override: Path | None
) -> Path:
    """Resolve the physical dataset location for observation-contract validation.

    The default comes from the checkpoint-saved dataset config.  An explicit
    ``override`` relocates the dataset only — task identity is still checked
    against the experiment in :func:`_validate_core_zarr_attrs`.
    """
    if override is not None:
        candidate = override.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    else:
        raw = dataset.get("zarr_path")
        if not isinstance(raw, str) or not raw:
            raise UnsupportedPolicyError(
                "checkpoint dataset has no single zarr_path; dynamic/multi-task "
                "datasets are unsupported"
            )
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidZarrError(f"Real Policy Zarr not found: {candidate}") from exc


def _checkpoint_observation_fields(
    dataset: dict[str, Any], agent_config: dict[str, Any]
) -> list[str]:
    """Resolve the artifact's observation modalities from checkpoint-saved semantics."""
    if any(key in agent_config for key in ("text_encoder_model", "task_texts")):
        raise UnsupportedPolicyError("dynamic task-text deployment is unsupported")
    modalities = dataset.get("sensor_modalities")
    if type(modalities) is not list or any(
        type(item) is not str for item in modalities
    ):
        raise UnsupportedPolicyError(
            "checkpoint dataset.sensor_modalities must be an explicit string list"
        )
    if (
        not modalities
        or len(set(modalities)) != len(modalities)
        or "joint_state" not in modalities
        or set(modalities) - _SUPPORTED_OBSERVATION_FIELDS
    ):
        raise UnsupportedPolicyError(
            "deployment observation fields must be unique supported names and include "
            "joint_state"
        )
    return list(modalities)


def _validate_required_zarr_arrays(root: Any, action_key: str) -> None:
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
    if action_key not in {"action", "action_ee"}:
        raise InvalidZarrError(f"checkpoint action_key is invalid: {action_key!r}")


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


def _validate_config_json_attr(
    attrs: Mapping[str, Any], key: str, required_members: frozenset[str]
) -> str:
    """Require one producer-owned numeric config JSON and keep it verbatim.

    The Real producer owns the numeric contract; this side only proves the
    attr is a non-empty JSON object carrying the required members, then
    propagates the original string so the Real deployment runtime can
    exact-compare the full physics-changing configuration.
    """
    encoded = attrs.get(key)
    if type(encoded) is not str or not encoded:
        raise InvalidZarrError(f"Zarr {key} must be a non-empty string")
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise InvalidZarrError(f"Zarr {key} is not valid JSON") from exc
    if type(parsed) is not dict or not required_members <= set(parsed):
        raise InvalidZarrError(
            f"Zarr {key} must encode an object containing "
            f"{sorted(required_members)}"
        )
    return encoded


def _build_observation_contract(
    path: Path,
    task_name: str,
    source: _CheckpointDeploymentSource,
) -> dict[str, Any]:
    """Build the observation contract from checkpoint semantics and source arrays.

    ``task_name`` is the *experiment* identity: an explicit ``--zarr-path``
    override only relocates the physical dataset, so Zarr task identity must
    still match the experiment exactly.
    """
    observation_fields = list(source.observation_fields)
    try:
        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        arrays = root["data"]
    except Exception as exc:
        raise InvalidZarrError(f"cannot open Real Policy Zarr: {path}") from exc
    _validate_core_zarr_attrs(attrs, task_name, source.dataset)
    _validate_required_zarr_arrays(root, source.agent["action_key"])

    fields: dict[str, dict[str, Any]] = {}
    for name in observation_fields:
        array = _observation_array(arrays, name)
        if name == "joint_state":
            _validate_observation_array(array, name, (19,), np.dtype(np.float32))
            fields[name] = _observation_field(
                (19,),
                "float32",
                "joint_position",
                {
                    "frame": "robot_joint",
                    "units": "rad",
                    "joint_order": "xarm7_xhand12",
                },
            )
        elif name == "point_cloud":
            (point_count, feature_dim), point_semantics = _validate_point_cloud(
                array, attrs, source.agent_config
            )
            fields[name] = _observation_field(
                (point_count, feature_dim),
                "float32",
                "xyzrgb",
                point_semantics,
            )
        elif name == "rgb":
            tail = _validate_observation_array(array, name, None, np.dtype(np.uint8))
            if len(tail) != 3 or tail[-1] != 3:
                raise InvalidZarrError("Zarr rgb must have shape [T, H, W, 3]")
            if (
                attrs.get("camera_extrinsic_semantics")
                != "T_xarm_base_from_color;native_color_optical_to_xarm_base"
            ):
                raise InvalidZarrError("Zarr RGB camera semantics are invalid")
            fields[name] = _observation_field(
                tail,
                "uint8",
                "raw_image",
                {
                    "color_order": "rgb",
                    "value_range": [0, 255],
                    "layout": "HWC",
                },
            )
        elif name == "contact_force":
            _validate_observation_array(array, name, (5, 3), np.dtype(np.float32))
            unit = attrs.get("contact_force_unit")
            if unit != "xhand_sdk_native_unknown_si":
                raise InvalidZarrError(
                    "Zarr contact_force_unit must be 'xhand_sdk_native_unknown_si'"
                )
            if (
                attrs.get("contact_force_frame")
                != "xhand_sensor_native_axes_per_finger"
            ):
                raise InvalidZarrError("Zarr contact_force_frame is invalid")
            si_verified = attrs.get("contact_force_si_verified")
            if si_verified is not False:
                raise InvalidZarrError("Zarr contact_force_si_verified must be false")
            fields[name] = _observation_field(
                (5, 3),
                "float32",
                "per_finger_sensor_axes",
                {
                    "frame": "xhand_sensor_native_axes_per_finger",
                    "units": unit,
                    "si_verified": si_verified,
                    "finger_order": "thumb_index_mid_ring_pinky",
                },
            )
        elif name == "fingertip_points":
            _validate_observation_array(array, name, (5, 3), np.dtype(np.float32))
            fingertip_semantics = _validate_fingertip_points(attrs)
            fields[name] = _observation_field(
                (5, 3),
                "float32",
                "point_xyz",
                fingertip_semantics,
            )
        elif name == "eef_pose":
            _validate_observation_array(array, name, (9,), np.dtype(np.float32))
            fields[name] = _observation_field(
                (9,),
                "float32",
                "position_m_rot6d",
                _validate_eef_pose(attrs),
            )
        elif name == "tactile_force":
            _validate_observation_array(
                array, name, (5, 120, 3), np.dtype(np.float32)
            )
            fields[name] = _observation_field(
                (5, 120, 3),
                "float32",
                "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
                _validate_tactile_force(attrs),
            )
        else:  # _checkpoint_observation_fields already rejects unknown values.
            raise InvalidZarrError(f"unsupported observation field: {name!r}")

    contract = {
        "schema_name": attrs["schema_name"],
        "schema_version": attrs["schema_version"],
        "domain": attrs["domain"],
        "task_name": attrs["task_name"],
        "dt": attrs["dt"],
        "obs_alignment": attrs["obs_alignment"],
        "observation_alignment": attrs["observation_alignment"],
        "state_alignment": attrs["state_alignment"],
        "contact_force_source": attrs["contact_force_source"],
        "action_semantics": attrs["action_semantics"],
        "requires_hand": True,
        "observation_fields": fields,
    }
    return _require_plain_metadata(contract, "data_contract")


def _validate_core_zarr_attrs(
    attrs: Mapping[str, Any],
    task_name: str,
    dataset: Mapping[str, Any],
) -> None:
    required = {
        "schema_name",
        "schema_version",
        "domain",
        "task_name",
        "dt",
        "obs_alignment",
        "observation_alignment",
        "state_alignment",
        "contact_force_source",
        "action_semantics",
    }
    missing = sorted(required - set(attrs))
    if missing:
        raise InvalidZarrError(f"Real Policy Zarr is missing semantic attrs: {missing}")
    # schema_version is informational metadata (provenance / human tracking),
    # not a compatibility gate: a dataset with compatible keys, shapes, dtypes,
    # and semantics is accepted regardless of its exact version integer.
    if (
        attrs["schema_name"] != "dexmani-real-policy-zarr"
        or attrs["domain"] != "real"
        or attrs["obs_alignment"] != "obs[t]_before_action[t]"
        or attrs["action_semantics"] != "teleop_published_joint_target"
    ):
        raise InvalidZarrError("invalid Real Policy Zarr semantics")
    # Experiment task identity is the one config-owned fact that must still
    # match: --zarr-path relocates the dataset, it does not re-label the task.
    if type(task_name) is not str or not task_name or attrs["task_name"] != task_name:
        raise InvalidZarrError("Zarr task_name does not match the experiment task_name")
    dt = _require_finite_number(attrs["dt"], "Zarr dt", positive=True)
    dataset_dt = dataset.get("dt")
    if dataset_dt is not None and (
        isinstance(dataset_dt, bool)
        or not isinstance(dataset_dt, (int, float))
        or not math.isclose(float(dataset_dt), dt, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise InvalidZarrError("Zarr dt conflicts with the checkpoint dataset contract")
    # Every modality uses the logical control step, never camera exposure time.
    if (
        attrs["observation_alignment"] != "control_step_latest_causal"
        or attrs["state_alignment"] != "control_step"
        or attrs["contact_force_source"] != "raw_hand_contact_control_step"
    ):
        raise InvalidZarrError(
            "Zarr observation timing/source must use the control-step contract"
        )


def _observation_array(arrays: Any, name: str) -> Any:
    try:
        return arrays[name]
    except Exception as exc:
        raise InvalidZarrError(f"Zarr data/{name} is missing") from exc


def _validate_observation_array(
    array: Any,
    name: str,
    expected_tail: tuple[int, ...] | None,
    expected_dtype: np.dtype[Any],
) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in array.shape)
        dtype = np.dtype(array.dtype)
    except Exception as exc:
        raise InvalidZarrError(f"Zarr data/{name} is not a valid array") from exc
    if (
        len(shape) < 2
        or shape[0] < 1
        or dtype != expected_dtype
        or (expected_tail is not None and shape[1:] != expected_tail)
    ):
        raise InvalidZarrError(
            f"Zarr data/{name} shape/dtype does not match the observation contract"
        )
    return shape[1:]


def _validate_point_cloud(
    array: Any, attrs: Mapping[str, Any], agent_config: Mapping[str, Any]
) -> tuple[tuple[int, int], dict[str, str]]:
    tail = _validate_observation_array(array, "point_cloud", None, np.dtype(np.float32))
    if len(tail) != 2 or tail[0] not in _POINT_COUNTS or tail[1] != _POINT_FEATURE_DIM:
        raise InvalidZarrError("unsupported point-cloud shape")
    if any(attrs.get(key) != value for key, value in _POINT_SEMANTICS.items()):
        raise InvalidZarrError("invalid Real point-cloud semantics")
    table_plane_abcd_json = _validate_table_plane(
        attrs.get("point_cloud_table_plane_abcd_json")
    )
    processing_config_json = _validate_config_json_attr(
        attrs, "processing_config_json", _PROCESSING_CONFIG_MEMBERS
    )
    configured_count = agent_config.get("num_points")
    if configured_count is not None and configured_count != tail[0]:
        raise InvalidZarrError("Zarr point count conflicts with agent.num_points")
    configured_dims = [agent_config.get("pc_dim")]
    pc_encoder = agent_config.get("pc_encoder_config")
    if type(pc_encoder) is dict:
        configured_dims.append(pc_encoder.get("pc_in_channels"))
    if any(value is not None and value != tail[1] for value in configured_dims):
        raise InvalidZarrError("Zarr point feature dim conflicts with agent config")
    return tail, {
        "frame": str(attrs["point_cloud_frame"]),
        "position_units": "m",
        "color_order": "rgb",
        "color_source": str(attrs["point_cloud_color_source"]),
        "policy_id": str(attrs["point_cloud_policy_id"]),
        "table_plane_abcd_json": table_plane_abcd_json,
        "processing_config_json": processing_config_json,
        "sampling": str(attrs["point_cloud_sampling"]),
        "transform": str(attrs["point_cloud_transform"]),
    }


def _validate_fingertip_points(attrs: Mapping[str, Any]) -> dict[str, str]:
    for key, expected in _FINGERTIP_SEMANTICS.items():
        if attrs.get(key) != expected:
            raise InvalidZarrError(f"Zarr {key} is invalid")
    fingertip_config_json = _validate_config_json_attr(
        attrs, "fingertip_config_json", _FINGERTIP_CONFIG_MEMBERS
    )
    return {
        "frame": str(attrs["fingertip_points_frame"]),
        "units": str(attrs["fingertip_points_unit"]),
        "finger_order": "thumb_index_mid_ring_pinky",
        "derivation": str(attrs["fingertip_points_derivation"]),
        "policy_id": str(attrs["fingertip_points_policy_id"]),
        "fingertip_config_json": fingertip_config_json,
    }


def _validate_eef_pose(attrs: Mapping[str, Any]) -> dict[str, str]:
    for key, expected in _EEF_POSE_SEMANTICS.items():
        if attrs.get(key) != expected:
            raise InvalidZarrError(f"Zarr {key} is invalid")
    return {
        "frame": str(attrs["eef_pose_frame"]),
        "position_units": "m",
        "rotation_representation": "rot6d",
        "derivation": str(attrs["eef_pose_derivation"]),
        "algorithm_id": str(attrs["eef_pose_algorithm_id"]),
    }


def _validate_tactile_force(attrs: Mapping[str, Any]) -> dict[str, Any]:
    for key, expected in _TACTILE_FORCE_SEMANTICS.items():
        if attrs.get(key) != expected:
            raise InvalidZarrError(f"Zarr {key} is invalid")
    unit = attrs.get("tactile_force_unit")
    if unit != "xhand_sdk_native_unknown_si":
        raise InvalidZarrError(
            "Zarr tactile_force_unit must be 'xhand_sdk_native_unknown_si'"
        )
    si_verified = attrs.get("tactile_force_si_verified")
    if si_verified is not False:
        raise InvalidZarrError("Zarr tactile_force_si_verified must be false")
    spatial_verified = attrs.get("tactile_force_spatial_geometry_verified")
    if spatial_verified is not False:
        raise InvalidZarrError(
            "Zarr tactile_force_spatial_geometry_verified must be false"
        )
    return {
        "finger_order": str(attrs["tactile_force_finger_order"]),
        "sensor_order": str(attrs["tactile_force_sensor_order"]),
        "point_order": str(attrs["tactile_force_point_order"]),
        "axis_labels": str(attrs["tactile_force_axis_labels"]),
        "unit": unit,
        "si_verified": si_verified,
        "spatial_geometry_verified": spatial_verified,
    }


def _observation_field(
    shape: tuple[int, ...],
    dtype: str,
    numeric_representation: str,
    semantics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "shape": list(shape),
        "dtype": dtype,
        "semantics": {
            "representation": numeric_representation,
            **semantics,
        },
    }


def _parse_checkpoint_deployment_source(
    checkpoint: TrainCheckpoint,
) -> _CheckpointDeploymentSource:
    """Parse one training checkpoint's own trained deployment semantics.

    This is the single source of the model and data semantics an artifact is
    built from: architecture/constructor (``resume_contract.agent_config``),
    action/window contract (``resume_contract.agent``), normalization
    (``resume_contract.agent.normalization``) and dataset/preprocessing
    (``resume_contract.dataset``).  The current experiment ``config.yaml``
    contributes only experiment identity and the inference recipe, so editing
    it after training can no longer change what an old checkpoint deploys as.

    Deleting reconciliation is not deleting validation: everything below is a
    strict *internal* consistency parse of the checkpoint itself, which is why
    a checkpoint whose own action/window/normalization contract is malformed
    still fails fast instead of being blindly trusted.
    """
    resume = checkpoint.resume_contract
    if type(resume) is not dict:
        raise InvalidCheckpointError("checkpoint resume_contract must be a plain dict")
    agent = resume.get("agent")
    agent_config = resume.get("agent_config")
    dataset = resume.get("dataset")
    for label, value in (
        ("agent", agent),
        ("agent_config", agent_config),
        ("dataset", dataset),
    ):
        if type(value) is not dict or not value:
            raise InvalidCheckpointError(
                f"checkpoint resume_contract.{label} must be a non-empty plain dict"
            )

    train = _parse_checkpoint_action_contract(agent)
    observation_fields = _checkpoint_observation_fields(dataset, agent_config)
    normalization_contract = _parse_checkpoint_normalization_contract(
        agent, observation_fields
    )
    return _CheckpointDeploymentSource(
        agent=_require_plain_metadata(train, "resume_contract.agent"),
        agent_config=_require_plain_metadata(
            agent_config, "resume_contract.agent_config"
        ),
        dataset=_require_plain_metadata(dataset, "resume_contract.dataset"),
        normalization_contract=normalization_contract,
        observation_fields=tuple(observation_fields),
    )


def _parse_checkpoint_action_contract(agent: dict[str, Any]) -> dict[str, Any]:
    """Strictly parse the checkpoint's own action/window contract for self-consistency."""
    required = (
        "n_obs_steps",
        "n_action_steps",
        "action_dim",
        "horizon",
        "action_key",
        "tcp_dim",
        "hand_dim",
        "control_action_dim",
        "use_aux_ee",
    )
    missing = sorted(name for name in required if name not in agent)
    if missing:
        raise InvalidCheckpointError(
            f"checkpoint resume_contract.agent is missing {missing}"
        )
    action_key = agent["action_key"]
    if action_key not in {"action", "action_ee"}:
        raise InvalidCheckpointError(
            f"checkpoint action_key is unsupported: {action_key!r}"
        )
    use_aux_ee = agent["use_aux_ee"]
    if type(use_aux_ee) is not bool:
        raise InvalidCheckpointError("checkpoint use_aux_ee must be bool")
    train = {
        name: _require_checkpoint_positive_int(agent[name], f"checkpoint {name}")
        for name in ("n_obs_steps", "n_action_steps", "action_dim", "horizon")
    }
    control_action_dim = _require_checkpoint_positive_int(
        agent["control_action_dim"], "checkpoint control_action_dim"
    )
    if train["n_obs_steps"] - 1 + train["n_action_steps"] > train["horizon"]:
        raise InvalidCheckpointError(
            "checkpoint observation/action window exceeds horizon"
        )

    expected_control = 21 if action_key == "action_ee" else 19
    if use_aux_ee:
        # The only supported auxiliary layout is joint19 + ee9 predicted
        # together while control stays joint-only.
        if action_key != "action" or train["action_dim"] != 28 or control_action_dim != 19:
            raise InvalidCheckpointError(
                "checkpoint use_aux_ee requires the joint19_ee9 action layout"
            )
    else:
        if train["action_dim"] != expected_control:
            raise InvalidCheckpointError(
                "checkpoint action_dim does not match action_key"
            )
        if control_action_dim != train["action_dim"]:
            raise InvalidCheckpointError(
                "checkpoint control_action_dim must equal action_dim without use_aux_ee"
            )

    tcp_dim = agent["tcp_dim"]
    hand_dim = agent["hand_dim"]
    if tcp_dim is None or hand_dim is None:
        if tcp_dim is not None or hand_dim is not None:
            raise InvalidCheckpointError(
                "checkpoint tcp_dim/hand_dim must both be set or both be null"
            )
    else:
        tcp_dim = _require_checkpoint_positive_int(tcp_dim, "checkpoint tcp_dim")
        hand_dim = _require_checkpoint_positive_int(hand_dim, "checkpoint hand_dim")
        if tcp_dim + hand_dim != train["action_dim"]:
            raise InvalidCheckpointError(
                "checkpoint tcp_dim + hand_dim must equal action_dim"
            )
    return {
        **train,
        "action_key": action_key,
        "tcp_dim": tcp_dim,
        "hand_dim": hand_dim,
        "control_action_dim": control_action_dim,
        "use_aux_ee": use_aux_ee,
    }


def _parse_checkpoint_normalization_contract(
    agent: dict[str, Any], observation_fields: list[str]
) -> dict[str, Any]:
    """Parse the checkpoint's versioned normalization contract, nothing else.

    The fitted ``scale``/``offset`` in this checkpoint were produced under this
    saved spec, so it is the only correct normalization metadata for the
    resulting artifact.  Exact field coverage is enforced against the
    checkpoint-derived observation fields.
    """
    saved_contract = agent.get("normalization")
    if type(saved_contract) is not dict:
        raise InvalidCheckpointError(
            "checkpoint resume_contract.agent.normalization must be a plain dict"
        )
    try:
        saved_spec = parse_normalization_contract(
            saved_contract, observation_fields=observation_fields
        )
    except ValueError as exc:
        raise InvalidCheckpointError(
            f"checkpoint normalization contract is invalid: {exc}"
        ) from exc
    return make_normalization_contract(saved_spec)


def _require_checkpoint_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise InvalidCheckpointError(f"{label} must be a positive integer")
    return value


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
    try:
        validate_agent_targets(sanitized)
    except DeploymentContractError as exc:
        raise UnsupportedPolicyError(str(exc)) from exc
    if "codebook_path" in sanitized:
        required_suffixes = (
            "codebook_manager.sorted_hand_poses",
            "codebook_manager.pca_permutation",
            "codebook_manager.layer_weights",
            "codebook_manager.hand_normalizer_scale",
            "codebook_manager.hand_normalizer_offset",
            "codebook_manager.hand_min",
            "codebook_manager.hand_max",
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
        permutation = found["codebook_manager.pca_permutation"]
        layer_weights = found["codebook_manager.layer_weights"]
        hand_normalizer_scale = found["codebook_manager.hand_normalizer_scale"]
        hand_normalizer_offset = found["codebook_manager.hand_normalizer_offset"]
        hand_min = found["codebook_manager.hand_min"]
        hand_max = found["codebook_manager.hand_max"]
        normalizer_suffixes = (
            "normalizer.params_dict.action.scale",
            "normalizer.params_dict.action.offset",
        )
        normalizer_state: dict[str, torch.Tensor] = {}
        for suffix in normalizer_suffixes:
            matches = [
                tensor for key, tensor in model_state.items() if key.endswith(suffix)
            ]
            if len(matches) != 1 or matches[0].numel() == 0:
                raise UnsupportedPolicyError(
                    "DQ-RISE checkpoint has no complete policy action normalizer state"
                )
            normalizer_state[suffix] = matches[0]
        action_scale = normalizer_state["normalizer.params_dict.action.scale"]
        action_offset = normalizer_state["normalizer.params_dict.action.offset"]
        groups = sanitized.get("codebook_num_groups")
        size = sanitized.get("codebook_size")
        hand_dim = train["hand_dim"]
        if (
            type(groups) is not int
            or type(size) is not int
            or type(hand_dim) is not int
            or poses.ndim != 2
            or poses.shape[0] != size**groups
            or poses.shape[1] != hand_dim
            or permutation.ndim != 1
            or permutation.numel() != poses.shape[0]
            or permutation.dtype
            not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
            or not torch.equal(
                torch.sort(permutation.detach().cpu().to(torch.int64)).values,
                torch.arange(poses.shape[0], dtype=torch.int64),
            )
            or layer_weights.ndim != 1
            or layer_weights.numel() != groups
            or hand_normalizer_scale.ndim != 1
            or hand_normalizer_scale.numel() != hand_dim
            or hand_normalizer_offset.ndim != 1
            or hand_normalizer_offset.numel() != hand_dim
            or hand_min.numel() != 1
            or hand_max.numel() != 1
            or action_scale.ndim != 1
            or action_scale.numel() != train["action_dim"]
            or action_offset.ndim != 1
            or action_offset.numel() != train["action_dim"]
            or not bool(torch.isfinite(poses).all())
            or not bool(torch.isfinite(permutation).all())
            or not bool(torch.isfinite(layer_weights).all())
            or not bool(torch.isfinite(hand_normalizer_scale).all())
            or not bool(torch.isfinite(hand_normalizer_offset).all())
            or bool(torch.any(hand_normalizer_scale == 0))
            or not bool(torch.isfinite(hand_min).all())
            or not bool(torch.isfinite(hand_max).all())
            or not bool(torch.isfinite(action_scale).all())
            or not bool(torch.isfinite(action_offset).all())
            or bool(torch.any(action_scale == 0))
            or not bool(hand_max.item() > hand_min.item())
            or train["tcp_dim"] != sanitized.get("tcp_dim")
            or train["action_key"] != "action_ee"
        ):
            raise InvalidCheckpointError(
                "DQ-RISE codebook/config/action metadata conflict"
            )
        try:
            torch.testing.assert_close(
                action_scale[-hand_dim:].detach().cpu(),
                hand_normalizer_scale.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                action_offset[-hand_dim:].detach().cpu(),
                hand_normalizer_offset.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            )
        except AssertionError as exc:
            raise InvalidCheckpointError(
                "DQ-RISE codebook hand normalizer conflicts with policy action normalizer"
            ) from exc
        sanitized["codebook_path"] = None
    pc_encoder = sanitized.get("pc_encoder_config")
    if type(pc_encoder) is dict and "use_pretrained_weights" in pc_encoder:
        pc_encoder["use_pretrained_weights"] = False
    return sanitized


def _build_inference_config(
    task_name: str,
    agent_config: dict[str, Any],
    source: _CheckpointDeploymentSource,
    selected: _SelectedInferenceSettings,
) -> dict[str, Any]:
    """Assemble the artifact inference config from checkpoint-owned semantics.

    Only ``task_name`` and the selected inference recipe come from outside the
    checkpoint; everything model-facing is ``source``.
    """
    train = source.agent
    inference = {
        "task_name": task_name,
        "action_key": train["action_key"],
        "action_dim": train["action_dim"],
        "horizon": train["horizon"],
        "n_obs_steps": train["n_obs_steps"],
        "n_action_steps": train["n_action_steps"],
        "use_aux_ee": train["use_aux_ee"],
        "normalization": source.normalization_contract,
        "agent": agent_config,
        "eval": {
            "use_ema": selected.use_ema,
            "denoise_steps": selected.denoise_steps,
        },
    }
    return _require_plain_metadata(inference, "inference_config")


def _resolve_selected_inference_settings(
    experiment_dir: Path,
    checkpoint_selector: str,
    cfg_plain: Mapping[str, Any],
) -> _SelectedInferenceSettings:
    """Resolve the selected checkpoint's inference recipe.

    The recipe is the one piece of model-facing configuration the experiment
    config still owns: ``best_ckpt.json["inference"]`` for the ``best``
    selector, otherwise the current ``config.eval`` (the live inference recipe
    a researcher is tuning).  Everything else deployment needs comes from the
    checkpoint.  ``denoise_steps`` is an ablation knob, not architecture, so
    overriding it per run stays legitimate.
    """
    if checkpoint_selector == "best":
        from dexmani_policy.training.eval_utils import read_best_ckpt_json

        try:
            inference = read_best_ckpt_json(experiment_dir)["inference"]
        except (OSError, ValueError) as exc:
            raise InvalidCheckpointError(
                f"invalid best_ckpt.json: {experiment_dir / 'best_ckpt.json'}"
            ) from exc
        prefix = "best_ckpt.json inference"
    else:
        eval_config = cfg_plain.get("eval")
        if type(eval_config) is not dict:
            raise InvalidExperimentError("config.eval must be a mapping")
        if eval_config.get("denoise_timesteps_list") is not None:
            raise UnsupportedPolicyError(
                "eval.denoise_timesteps_list is unsupported for deployment"
            )
        inference = {
            "use_ema": eval_config.get("use_ema"),
            "denoise_steps": eval_config.get("denoise_steps"),
        }
        prefix = "config"

    use_ema = inference["use_ema"]
    if type(use_ema) is not bool:
        raise InvalidExperimentError(f"{prefix}.use_ema must be bool")
    denoise_steps = _require_positive_int(
        inference["denoise_steps"], f"{prefix}.denoise_steps"
    )
    return _SelectedInferenceSettings(use_ema, denoise_steps)


def _rgb_preprocessing(source: _CheckpointDeploymentSource) -> dict[str, Any]:
    """Record the exact validation-spatial and ImageProcessor RGB chain.

    Both stages are read from checkpoint-saved semantics, so later edits to the
    experiment's dataset/agent RGB settings cannot change an old artifact.
    """
    agent = source.agent_config
    dataset = source.dataset
    if dataset.get("_target_") != "dexmani_policy.datasets.rgb_dataset.RGBDataset":
        raise UnsupportedPolicyError(
            "RGB deployment requires the current RGBDataset validation contract"
        )
    if "rgb_preprocess_size" not in dataset or "rgb_random_crop_size" not in dataset:
        raise InvalidCheckpointError(
            "RGB export requires explicit checkpoint validation preprocessing"
        )
    resize_hw = _optional_hw(dataset["rgb_preprocess_size"], "rgb_preprocess_size")
    center_crop_hw = _optional_hw(
        dataset["rgb_random_crop_size"], "rgb_random_crop_size"
    )
    if resize_hw is None and center_crop_hw is not None:
        raise InvalidCheckpointError(
            "dataset.rgb_random_crop_size requires rgb_preprocess_size"
        )
    keep_uint8 = dataset.get("rgb_keep_uint8", DEFAULT_RGB_KEEP_UINT8)
    if type(keep_uint8) is not bool:
        raise InvalidCheckpointError("dataset.rgb_keep_uint8 must be bool")
    validation_keeps_uint8 = (
        resize_hw is not None and keep_uint8 and dataset.get("rgb_color_aug") is None
    )

    backbone_name = agent.get("rgb_backbone_name")
    backbone_config = agent.get("rgb_backbone_config")
    if type(backbone_name) is not str or backbone_name not in IMAGE_PROCESSOR_PRESETS:
        raise UnsupportedPolicyError("RGB deployment requires a supported backbone")
    if backbone_config is None:
        backbone_config = {}
    if type(backbone_config) is not dict:
        raise InvalidCheckpointError("agent.rgb_backbone_config must be a mapping")
    if any(
        key in backbone_config
        for key in ("center_crop_size", "image_mean", "image_std")
    ):
        raise UnsupportedPolicyError(
            "RGB deployment does not support unowned ImageProcessor overrides"
        )

    preset = IMAGE_PROCESSOR_PRESETS[backbone_name]
    image_size = _processor_hw(
        (
            backbone_config["image_size"]
            if backbone_config.get("image_size") is not None
            else preset.get("image_size")
        ),
        "image_size",
    )
    interpolation = (
        backbone_config["interpolation"]
        if backbone_config.get("interpolation") is not None
        else preset.get("interpolation")
    )
    if type(interpolation) is str:
        interpolation = interpolation.lower()
    if interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise InvalidCheckpointError("agent RGB interpolation is unsupported")
    mean = _processor_rgb_vector(preset.get("image_mean"), "image_mean")
    std = _processor_rgb_vector(preset.get("image_std"), "image_std")

    if resize_hw is None:
        output_layout = "HWC"
        output_dtype = "uint8"
        output_value_range = [0, 255]
        scale = 1.0
    elif validation_keeps_uint8:
        output_layout = "CHW"
        output_dtype = "uint8"
        output_value_range = [0, 255]
        scale = 1.0
    else:
        output_layout = "CHW"
        output_dtype = "float32"
        output_value_range = [0, 1]
        scale = 1.0 / 255.0

    return {
        "input_layout": "HWC",
        "input_dtype": "uint8",
        "input_color_order": "rgb",
        "input_value_range": [0, 255],
        "execution_device": "cpu",
        "resize_hw": None if resize_hw is None else list(resize_hw),
        "center_crop_hw": (None if center_crop_hw is None else list(center_crop_hw)),
        "interpolation": "bilinear",
        "antialias": True,
        "output_layout": output_layout,
        "output_dtype": output_dtype,
        "scale": scale,
        "output_value_range": output_value_range,
        "processor_image_size_hw": (None if image_size is None else list(image_size)),
        "processor_interpolation": interpolation,
        "normalize_mean": list(mean),
        "normalize_std": list(std),
    }


def _processor_hw(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is int and value > 0:
        return value, value
    if (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value)
    ):
        return value
    if (
        type(value) is list
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value)
    ):
        return value[0], value[1]
    raise InvalidCheckpointError(f"agent RGB {label} must be positive [H, W] or null")


def _processor_rgb_vector(value: Any, label: str) -> tuple[float, float, float]:
    if (
        type(value) not in {tuple, list}
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise InvalidCheckpointError(f"agent RGB {label} must be three finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise InvalidCheckpointError(f"agent RGB {label} must be three finite numbers")
    return result  # type: ignore[return-value]


def _optional_hw(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise InvalidCheckpointError(f"dataset.{label} must be [H, W] or null")
    return value[0], value[1]


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


def _validate_payload(payload: Any) -> None:
    if type(payload) is not dict or set(payload) != {"_format", "contract", "weights"}:
        raise ArtifactVerificationError("deployment checkpoint payload schema mismatch")
    if payload["_format"] != DEPLOYMENT_FORMAT:
        raise ArtifactVerificationError("deployment checkpoint format mismatch")
    contract = payload["contract"]
    weights = payload["weights"]
    if type(contract) is not dict or set(contract) != {
        "inference_config",
        "data_contract",
        "producer",
    }:
        raise ArtifactVerificationError("deployment contract schema mismatch")
    _canonicalize_state_dict(weights, "weights")
    for name in ("inference_config", "data_contract", "producer"):
        _require_plain_metadata(contract[name], f"contract.{name}")
    try:
        parse_deployment_contract(payload)
    except DeploymentContractError as exc:
        raise ArtifactVerificationError("invalid deployment metadata") from exc


def _verify_exported_model(payload: dict[str, Any]) -> None:
    try:
        verify_deployment_prediction(payload, seed=0)
    except DeploymentRestoreError as exc:
        raise ArtifactVerificationError(
            "deployment agent strict restore/prediction failed"
        ) from exc
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


def _verify_published_selector(selector_path: Path, checkpoint_path: Path) -> None:
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
    _load_deployment_payload(checkpoint_path)


def _selector_points_at(selector_path: Path, checkpoint_path: Path) -> bool:
    """Whether the live selector already names this artifact."""
    return (
        selector_path.is_symlink()
        and os.readlink(selector_path) == checkpoint_path.name
    )


def publish_deployment_selector(
    selector_path: Path,
    checkpoint_path: Path,
) -> None:
    """Atomically publish one already-qualified canonical artifact.

    On any failure the previous selector is restored, so the caller can treat a
    raised publish as "nothing changed" and safely delete the candidate it built.
    Whether a rollback is needed is decided from the *observable* selector state,
    never from a flag set when a helper returns: the symlink swap is live the
    moment ``os.replace`` succeeds, and the durability fsync that follows can
    still raise.  Using a return-time flag there would skip the rollback and let
    the caller delete the very artifact the selector had just been pointed at.
    """
    old_selector = _capture_selector(selector_path)
    try:
        _replace_relative_symlink(selector_path, checkpoint_path.name)
        _verify_published_selector(selector_path, checkpoint_path)
    except BaseException:
        if _selector_points_at(selector_path, checkpoint_path):
            _rollback_selector(selector_path, old_selector)
        raise


def cleanup_candidate_artifact(checkpoint_path: Path) -> None:
    """Remove one unpublished candidate artifact so a same-name retry succeeds.

    The removal is durable (the directory entry is fsynced) and never silent: a
    cleanup that itself fails raises ``ArtifactPublicationError`` because the
    leftover candidate would otherwise block the obvious retry with a confusing
    ``FileExistsError``, and the operator needs to know the artifact directory
    may require manual inspection.
    """
    try:
        if checkpoint_path.is_symlink() or checkpoint_path.exists():
            checkpoint_path.unlink()
            _fsync_directory(checkpoint_path.parent)
    except OSError as exc:
        raise ArtifactPublicationError(
            "failed to remove the unpublished deployment candidate "
            f"{checkpoint_path}; inspect the checkpoint directory manually before "
            "retrying"
        ) from exc


def _build_deployment_payload(
    experiment: Path,
    checkpoint_selector: str,
    zarr_path: Path | None,
) -> tuple[dict[str, Any], Path]:
    """Build one complete, validated deployment payload from a selected checkpoint.

    Ownership is explicit here: ``source`` (the checkpoint) owns architecture,
    action/window, normalization and dataset/preprocessing semantics, while the
    experiment ``config.yaml`` contributes only identity (``task_name``) and the
    inference recipe (``eval.use_ema`` / ``eval.denoise_steps``, or
    ``best_ckpt.json`` for the ``best`` selector).
    """
    repo_root = Path(__file__).resolve().parents[2]
    selected_path = _resolve_checkpoint(experiment, checkpoint_selector)
    cfg_plain = _load_config(experiment)
    task_name = cfg_plain.get("task_name")
    if type(task_name) is not str or not task_name:
        raise InvalidExperimentError("experiment config task_name must be a non-empty string")
    selected_inference = _resolve_selected_inference_settings(
        experiment, checkpoint_selector, cfg_plain
    )
    checkpoint = _load_training_checkpoint(selected_path)
    source = _parse_checkpoint_deployment_source(checkpoint)

    # Selected weights first: decide raw vs EMA, then canonicalize, validate and
    # sanitize exactly one state.  The unselected state is never processed.
    selected_weights = "ema_model" if selected_inference.use_ema else "model"
    selected_raw = (
        checkpoint.ema_model_state
        if selected_inference.use_ema
        else checkpoint.model_state
    )
    if selected_raw is None:
        raise InvalidCheckpointError(
            f"eval.use_ema={selected_inference.use_ema!r} requires checkpoint "
            f"{selected_weights} weights"
        )
    selected_state = _canonicalize_state_dict(
        selected_raw, f"weights.{selected_weights}"
    )

    resolved_zarr = _resolve_zarr_path(source.dataset, repo_root, zarr_path)
    agent_config = _sanitize_agent_config(
        source.agent_config, selected_state, source.agent
    )
    inference = _build_inference_config(
        task_name, agent_config, source, selected_inference
    )
    data_contract = _build_observation_contract(resolved_zarr, task_name, source)
    if "rgb" in source.observation_fields:
        inference["rgb_preprocessing"] = _rgb_preprocessing(source)
    producer = {
        "source_checkpoint": selected_path.name,
        "selected_weights": selected_weights,
    }
    source_commit = _source_commit(repo_root)
    if source_commit is not None:
        producer["source_commit"] = source_commit
    deployment_inference = {
        **inference,
        "eval": {"denoise_steps": inference["eval"]["denoise_steps"]},
    }
    payload = {
        "_format": DEPLOYMENT_FORMAT,
        "contract": {
            "inference_config": deployment_inference,
            "data_contract": data_contract,
            "producer": producer,
        },
        "weights": selected_state,
    }
    _validate_payload(payload)
    return payload, selected_path


def _source_commit(repo_root: Path) -> str | None:
    """Best-effort git HEAD for paper-experiment traceability only.

    This is never a runtime compatibility gate; an unavailable revision simply
    omits the field rather than blocking an export.

    ``git -C <dir> rev-parse HEAD`` resolves the *nearest enclosing* repository,
    so a copied or installed package tree could otherwise record an unrelated
    outer repository's commit — indistinguishable from a correct sha, and worse
    than no provenance at all.  The revision is therefore accepted only when the
    package root is itself the repository top level.
    """
    import subprocess

    def _git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    if _git("rev-parse", "--show-toplevel") != str(repo_root):
        return None
    commit = _git("rev-parse", "HEAD")
    if commit is None or len(commit) != 40:
        return None
    if any(char not in "0123456789abcdef" for char in commit):
        return None
    return commit


def _resolve_artifact_paths(
    experiment: Path, selected_path: Path, output_path: Path | None
) -> tuple[Path, Path, Path]:
    checkpoint_dir = experiment / "checkpoints"
    if output_path is None:
        final_path = checkpoint_dir / f"{selected_path.stem}-deployment.pt"
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
    return checkpoint_dir, final_path, checkpoint_dir / "deployment_latest.pt"


def _export_candidate(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    zarr_path: Path | None = None,
) -> ExportReceipt:
    """Write one unpublished, unverified candidate artifact.

    Private on purpose.  The only caller that legitimately wants a candidate
    without the public verify+publish sequence is ``qualify_policy_parity``,
    which runs its own restore/parity and publishes last.  Keeping this private
    means no researcher-facing path can publish an unverified artifact.
    """
    experiment = _require_experiment_directory(experiment_dir)
    payload, selected_path = _build_deployment_payload(
        experiment, checkpoint_selector, zarr_path
    )
    checkpoint_dir, final_path, selector_path = _resolve_artifact_paths(
        experiment, selected_path, output_path
    )
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite deployment artifact: {final_path}"
        )
    checkpoint_temp: Path | None = None
    renamed = False
    try:
        checkpoint_temp = _write_checkpoint_temp(checkpoint_dir, payload)
        os.replace(checkpoint_temp, final_path)
        checkpoint_temp = None
        renamed = True
        _fsync_directory(checkpoint_dir)
    except BaseException as exc:
        if checkpoint_temp is not None:
            checkpoint_temp.unlink(missing_ok=True)
        if renamed:
            # The candidate already carries its final name, so the temp path is
            # gone and only final_path can be removed.  Without this the failed
            # candidate survives and blocks the identical retry.
            try:
                cleanup_candidate_artifact(final_path)
            except ArtifactPublicationError as cleanup_exc:
                raise cleanup_exc from exc
        if isinstance(exc, DeploymentExportError):
            raise
        raise ArtifactPublicationError(
            "deployment candidate artifact write failed"
        ) from exc
    return ExportReceipt(
        checkpoint_path=final_path,
        selector_path=selector_path,
        checkpoint_selector=checkpoint_selector,
    )


def _require_experiment_directory(experiment_dir: Path) -> Path:
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
    return experiment


def export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    zarr_path: Path | None = None,
) -> ExportReceipt:
    """Export one selected checkpoint as a verified, published deployment artifact.

    There is deliberately no way to publish an unverified artifact.  Every
    successful call has completed, in order::

        build payload -> validate payload -> write candidate
            -> safe weights_only reload -> strict restore
            -> deterministic synthetic prediction -> publish selector

    Any failure before the selector swap removes the candidate this call
    created, durably fsyncs the checkpoint directory, leaves the previous
    ``deployment_latest.pt`` untouched, and therefore leaves the identical
    command directly retryable.  ``zarr_path`` overrides only the physical
    dataset location; the Zarr's task identity must still match the experiment.
    """
    receipt = _export_candidate(
        experiment_dir,
        checkpoint_selector=checkpoint_selector,
        output_path=output_path,
        zarr_path=zarr_path,
    )
    try:
        _verify_exported_model(_load_deployment_payload(receipt.checkpoint_path))
        publish_deployment_selector(receipt.selector_path, receipt.checkpoint_path)
    except BaseException:
        cleanup_candidate_artifact(receipt.checkpoint_path)
        raise
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="override only the physical dataset location (task identity still checked)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = export_deployment_artifact(
        args.experiment_dir,
        checkpoint_selector=args.checkpoint,
        output_path=args.output,
        zarr_path=args.zarr_path,
    )
    print(
        _canonical_json(
            {
                "checkpoint_path": str(receipt.checkpoint_path),
                "selector_path": str(receipt.selector_path),
                "checkpoint_selector": receipt.checkpoint_selector,
            }
        )
    )


if __name__ == "__main__":
    main()
