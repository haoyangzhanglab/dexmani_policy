"""Real Policy Zarr data semantics shared by training checkpoints and export.

This module is the single extractor of the deployment-relevant semantics of one
Real Policy Zarr.  Training freezes its output into
``resume_contract.deployment_data_semantics``; deployment export re-runs the
same extractor on the selected (or relocated) Zarr and requires exact equality
with the checkpoint snapshot, so no model-facing physical semantic — ``dt``,
observation/state alignment, point-cloud processing, table plane, action EE
frame/components, fingertip/EEF/tactile identity, raw observation shape/dtype —
can silently drift between training and export.

The extractor reads metadata only (root attrs plus array shapes/dtypes), never
array data, so it is cheap and deterministic across DDP ranks.  It validates
the Real producer contract but owns no deployment runtime concern; errors are
plain :class:`RealPolicyContractError` (a ``ValueError``) that callers map to
their own boundary error types.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]

SUPPORTED_OBSERVATION_FIELDS = frozenset(
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

# Keys captured for provenance but excluded from semantic equality: a dataset
# with identical physics-facing semantics is the same dataset regardless of the
# producer's version integer.
INFORMATIONAL_SEMANTIC_KEYS = frozenset({"schema_version"})

_SCHEMA_NAME = "dexmani-real-policy-zarr"
_DOMAIN = "real"
_ACTION_EE_FRAME = "xarm_base"
_ACTION_EE_COMPONENTS = "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)"

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


class RealPolicyContractError(ValueError):
    """One Real Policy Zarr violates the shared data semantic contract."""


def is_real_policy_zarr(zarr_path: str | Path) -> bool:
    """Whether one physical Zarr declares the Real Policy producer schema.

    Only a clean attrs mismatch reports ``False`` (e.g. a sim dataset); an
    unreadable store raises instead, so a corrupt Real Zarr is never silently
    re-labelled as unsupported data.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root.attrs)
    return attrs.get("schema_name") == _SCHEMA_NAME and attrs.get("domain") == _DOMAIN


def validate_observation_field_list(observation_fields: Sequence[str]) -> list[str]:
    """Validate one deployment observation-field selection; return it as a list."""
    fields = list(observation_fields)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or "joint_state" not in fields
        or set(fields) - SUPPORTED_OBSERVATION_FIELDS
    ):
        raise RealPolicyContractError(
            "deployment observation fields must be unique supported names and include "
            "joint_state"
        )
    return fields


def build_real_policy_data_semantics(
    zarr_path: str | Path,
    *,
    task_name: str,
    observation_fields: Sequence[str],
    agent_config: Mapping[str, Any],
    action_key: str,
) -> dict[str, Any]:
    """Extract the deployment-relevant semantics of one Real Policy Zarr.

    The result is a plain JSON-safe dict: every value the Real producer's
    physics/preprocessing contract exposes (core timing/alignment attrs, action
    EE identity, and per-observation-field raw shape/dtype/semantics), frozen so
    training checkpoints and deployment export can be compared for exact
    equality.  ``schema_version`` is captured but informational (see
    :data:`INFORMATIONAL_SEMANTIC_KEYS`).

    ``task_name`` is the identity the Zarr must carry; it is checkpoint-owned
    at export time, so a relocated Zarr can never re-label the task.
    """
    fields = validate_observation_field_list(observation_fields)
    path = Path(zarr_path)
    try:
        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)
        arrays = root["data"]
    except Exception as exc:
        raise RealPolicyContractError(
            f"cannot open Real Policy Zarr: {path}"
        ) from exc
    _validate_core_zarr_attrs(attrs, task_name)
    time_length = _validate_required_zarr_arrays(root, action_key)

    captured: dict[str, dict[str, Any]] = {}
    for name in fields:
        array = _observation_array(arrays, name)
        if _time_length(array, name) != time_length:
            raise RealPolicyContractError(
                f"Zarr data/{name} time length does not match the action/state arrays"
            )
        if name == "joint_state":
            _validate_observation_array(array, name, (19,), np.dtype(np.float32))
            captured[name] = _observation_field(
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
                array, attrs, agent_config
            )
            captured[name] = _observation_field(
                (point_count, feature_dim),
                "float32",
                "xyzrgb",
                point_semantics,
            )
        elif name == "rgb":
            tail = _validate_observation_array(array, name, None, np.dtype(np.uint8))
            if len(tail) != 3 or tail[-1] != 3:
                raise RealPolicyContractError("Zarr rgb must have shape [T, H, W, 3]")
            if (
                attrs.get("camera_extrinsic_semantics")
                != "T_xarm_base_from_color;native_color_optical_to_xarm_base"
            ):
                raise RealPolicyContractError("Zarr RGB camera semantics are invalid")
            captured[name] = _observation_field(
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
                raise RealPolicyContractError(
                    "Zarr contact_force_unit must be 'xhand_sdk_native_unknown_si'"
                )
            if (
                attrs.get("contact_force_frame")
                != "xhand_sensor_native_axes_per_finger"
            ):
                raise RealPolicyContractError("Zarr contact_force_frame is invalid")
            si_verified = attrs.get("contact_force_si_verified")
            if si_verified is not False:
                raise RealPolicyContractError(
                    "Zarr contact_force_si_verified must be false"
                )
            captured[name] = _observation_field(
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
            captured[name] = _observation_field(
                (5, 3),
                "float32",
                "point_xyz",
                fingertip_semantics,
            )
        elif name == "eef_pose":
            _validate_observation_array(array, name, (9,), np.dtype(np.float32))
            captured[name] = _observation_field(
                (9,),
                "float32",
                "position_m_rot6d",
                _validate_eef_pose(attrs),
            )
        elif name == "tactile_force":
            _validate_observation_array(
                array, name, (5, 120, 3), np.dtype(np.float32)
            )
            captured[name] = _observation_field(
                (5, 120, 3),
                "float32",
                "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
                _validate_tactile_force(attrs),
            )
        else:  # validate_observation_field_list already rejects unknown values.
            raise RealPolicyContractError(f"unsupported observation field: {name!r}")

    return {
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
        "action_ee_frame": attrs["action_ee_frame"],
        "action_ee_components": attrs["action_ee_components"],
        "observation_fields": captured,
    }


def semantics_mismatch(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    """Return the differing semantic key paths; ``[]`` means equivalent.

    Strict by design: any model-facing difference — including a key present on
    one side only, or ``None`` versus a declared value — is drift.  Only
    :data:`INFORMATIONAL_SEMANTIC_KEYS` are ignored.  This is a targeted
    two-level comparison (top level plus observation fields), not a general
    diff framework.
    """
    differing: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if key in INFORMATIONAL_SEMANTIC_KEYS:
            continue
        if key not in expected or key not in actual:
            differing.append(key)
        elif key == "observation_fields":
            differing.extend(
                _observation_fields_mismatch(expected[key], actual[key])
            )
        elif expected[key] != actual[key]:
            differing.append(key)
    return differing


def _observation_fields_mismatch(expected: Any, actual: Any) -> list[str]:
    if type(expected) is not dict or type(actual) is not dict:
        return ["observation_fields"]
    differing: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            differing.append(f"observation_fields.{name}")
            continue
        expected_field, actual_field = expected[name], actual[name]
        if type(expected_field) is not dict or type(actual_field) is not dict:
            differing.append(f"observation_fields.{name}")
            continue
        for sub in sorted(set(expected_field) | set(actual_field)):
            expected_value, actual_value = expected_field.get(sub), actual_field.get(sub)
            if expected_value == actual_value:
                continue
            if sub == "semantics" and (
                type(expected_value) is dict and type(actual_value) is dict
            ):
                # One more level, so the error names the drifted semantic
                # (e.g. ...semantics.processing_config_json), not just the field.
                for key in sorted(set(expected_value) | set(actual_value)):
                    if expected_value.get(key) != actual_value.get(key):
                        differing.append(
                            f"observation_fields.{name}.semantics.{key}"
                        )
            else:
                differing.append(f"observation_fields.{name}.{sub}")
    return differing


def _require_finite_number(value: Any, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealPolicyContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise RealPolicyContractError(
            f"{label} must be {'positive and ' if positive else ''}finite"
        )
    return number


def _validate_core_zarr_attrs(attrs: Mapping[str, Any], task_name: str) -> None:
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
        "action_ee_frame",
        "action_ee_components",
    }
    missing = sorted(required - set(attrs))
    if missing:
        raise RealPolicyContractError(
            f"Real Policy Zarr is missing semantic attrs: {missing}"
        )
    # schema_version is informational metadata (provenance / human tracking),
    # not a compatibility gate: a dataset with compatible keys, shapes, dtypes,
    # and semantics is accepted regardless of its exact version integer.
    if (
        attrs["schema_name"] != _SCHEMA_NAME
        or attrs["domain"] != _DOMAIN
        or attrs["obs_alignment"] != "obs[t]_before_action[t]"
        or attrs["action_semantics"] != "teleop_published_joint_target"
        or attrs["action_ee_frame"] != _ACTION_EE_FRAME
    ):
        raise RealPolicyContractError("invalid Real Policy Zarr semantics")
    if attrs["action_ee_components"] != _ACTION_EE_COMPONENTS:
        raise RealPolicyContractError("Zarr action_ee_components is invalid")
    # Task identity is checkpoint-owned: a relocated Zarr proves equivalence,
    # it never re-labels the task.
    if type(task_name) is not str or not task_name or attrs["task_name"] != task_name:
        raise RealPolicyContractError(
            "Zarr task_name does not match the experiment task_name"
        )
    _require_finite_number(attrs["dt"], "Zarr dt", positive=True)
    # Every modality uses the logical control step, never camera exposure time.
    if (
        attrs["observation_alignment"] != "control_step_latest_causal"
        or attrs["state_alignment"] != "control_step"
        or attrs["contact_force_source"] != "raw_hand_contact_control_step"
    ):
        raise RealPolicyContractError(
            "Zarr observation timing/source must use the control-step contract"
        )


def _validate_required_zarr_arrays(root: Any, action_key: str) -> int:
    """Validate the action/state arrays and return their shared time length."""
    expected_dims = {"joint_state": 19, "action": 19, "action_ee": 21}
    try:
        arrays = root["data"]
        shapes = {
            key: tuple(int(value) for value in arrays[key].shape)
            for key in expected_dims
        }
    except Exception as exc:
        raise RealPolicyContractError(
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
        raise RealPolicyContractError(
            f"Zarr action/state dimensions are invalid: {shapes}"
        )
    if action_key not in {"action", "action_ee"}:
        raise RealPolicyContractError(f"checkpoint action_key is invalid: {action_key!r}")
    return next(iter(lengths))


def _observation_array(arrays: Any, name: str) -> Any:
    try:
        return arrays[name]
    except Exception as exc:
        raise RealPolicyContractError(f"Zarr data/{name} is missing") from exc


def _time_length(array: Any, name: str) -> int:
    try:
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise RealPolicyContractError(
            f"Zarr data/{name} is not a valid array"
        ) from exc
    if len(shape) < 2 or shape[0] < 1:
        raise RealPolicyContractError(
            f"Zarr data/{name} shape/dtype does not match the observation contract"
        )
    return shape[0]


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
        raise RealPolicyContractError(
            f"Zarr data/{name} is not a valid array"
        ) from exc
    if (
        len(shape) < 2
        or shape[0] < 1
        or dtype != expected_dtype
        or (expected_tail is not None and shape[1:] != expected_tail)
    ):
        raise RealPolicyContractError(
            f"Zarr data/{name} shape/dtype does not match the observation contract"
        )
    return shape[1:]


def _validate_json_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise RealPolicyContractError(f"{label} must be a non-empty JSON string")
    try:
        parsed = json.loads(
            value, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
        )
        json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RealPolicyContractError(f"{label} must contain finite JSON") from exc
    return value


def _validate_table_plane(value: Any) -> str:
    encoded = _validate_json_string(value, "point_cloud_table_plane_abcd_json")
    plane = json.loads(encoded)
    if plane is None:
        if encoded != "null":
            raise RealPolicyContractError(
                "point-cloud table plane JSON must be canonical"
            )
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
        raise RealPolicyContractError(
            "point-cloud table plane must be null or four finite numbers"
        )
    normal_norm = math.sqrt(sum(float(item) ** 2 for item in plane[:3]))
    if normal_norm <= 0.0 or float(plane[2]) / normal_norm <= 0.0:
        raise RealPolicyContractError(
            "point-cloud table plane normal must point upward"
        )
    canonical = json.dumps(plane, allow_nan=False, separators=(",", ":"))
    if encoded != canonical:
        raise RealPolicyContractError(
            "point-cloud table plane JSON must be canonical"
        )
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
        raise RealPolicyContractError(f"Zarr {key} must be a non-empty string")
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RealPolicyContractError(f"Zarr {key} is not valid JSON") from exc
    if type(parsed) is not dict or not required_members <= set(parsed):
        raise RealPolicyContractError(
            f"Zarr {key} must encode an object containing "
            f"{sorted(required_members)}"
        )
    return encoded


def _validate_point_cloud(
    array: Any, attrs: Mapping[str, Any], agent_config: Mapping[str, Any]
) -> tuple[tuple[int, int], dict[str, str]]:
    tail = _validate_observation_array(array, "point_cloud", None, np.dtype(np.float32))
    if len(tail) != 2 or tail[0] not in _POINT_COUNTS or tail[1] != _POINT_FEATURE_DIM:
        raise RealPolicyContractError("unsupported point-cloud shape")
    if any(attrs.get(key) != value for key, value in _POINT_SEMANTICS.items()):
        raise RealPolicyContractError("invalid Real point-cloud semantics")
    table_plane_abcd_json = _validate_table_plane(
        attrs.get("point_cloud_table_plane_abcd_json")
    )
    processing_config_json = _validate_config_json_attr(
        attrs, "processing_config_json", _PROCESSING_CONFIG_MEMBERS
    )
    # Optional by design: not every point-cloud agent declares num_points or
    # pc_dim at the top level (r3d carries only pc_encoder_config.pc_in_channels).
    configured_count = agent_config.get("num_points")
    if configured_count is not None and configured_count != tail[0]:
        raise RealPolicyContractError("Zarr point count conflicts with agent.num_points")
    configured_dims = [agent_config.get("pc_dim")]
    pc_encoder = agent_config.get("pc_encoder_config")
    if type(pc_encoder) is dict:
        configured_dims.append(pc_encoder.get("pc_in_channels"))
    if any(value is not None and value != tail[1] for value in configured_dims):
        raise RealPolicyContractError("Zarr point feature dim conflicts with agent config")
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
            raise RealPolicyContractError(f"Zarr {key} is invalid")
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
            raise RealPolicyContractError(f"Zarr {key} is invalid")
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
            raise RealPolicyContractError(f"Zarr {key} is invalid")
    unit = attrs.get("tactile_force_unit")
    if unit != "xhand_sdk_native_unknown_si":
        raise RealPolicyContractError(
            "Zarr tactile_force_unit must be 'xhand_sdk_native_unknown_si'"
        )
    si_verified = attrs.get("tactile_force_si_verified")
    if si_verified is not False:
        raise RealPolicyContractError("Zarr tactile_force_si_verified must be false")
    spatial_verified = attrs.get("tactile_force_spatial_geometry_verified")
    if spatial_verified is not False:
        raise RealPolicyContractError(
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
