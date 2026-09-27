"""Requested Real capabilities, checked before normalization or model construction.

No dependency on the Real hardware package. These are consumer expectations for
stable representations; unrequested modalities and additive attrs are ignored.
"""

import json
import math

import numpy as np
import zarr

_CANONICAL_FORMAT = "dexmani.real.canonical"
_SHAPES = {
    "joint_state": (19,),
    "arm_qvel": (7,),
    "arm_effort": (7,),
    "hand_current": (12,),
    "eef_pose": (9,),
    "fingertip_points": (5, 3),
    "contact_force": (5, 3),
    "tactile_force": (5, 120, 3),
    "action": (19,),
    "action_ee": (21,),
}
_ARM_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_HAND_ORDER = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
_SEMANTICS = {
    "joint_state": {
        "semantic_id": "dexmani.joint_state",
        "unit": "rad",
        "joint_order": _ARM_ORDER + _HAND_ORDER,
        "alignment": "control_step_latest_causal_before_action",
    },
    "arm_qvel": {
        "semantic_id": "dexmani.xarm.joint_velocity",
        "unit": "rad/s",
        "joint_order": _ARM_ORDER,
        "missing": "nan",
        "alignment": "control_step_latest_causal_before_action",
    },
    "arm_effort": {
        "semantic_id": "dexmani.xarm.effort.native",
        "unit": "sdk_native_unverified",
        "joint_order": _ARM_ORDER,
        "missing": "nan",
        "alignment": "control_step_latest_causal_before_action",
    },
    "hand_current": {
        "semantic_id": "dexmani.xhand.current",
        "unit": "mA",
        "joint_order": _HAND_ORDER,
        "missing": "nan",
        "alignment": "control_step_latest_causal_before_action",
    },
    "eef_pose": {
        "semantic_id": "dexmani.eef_pose.xarm_base",
        "frame": "xarm_base",
        "components": "position_m(3)+rot6d(6)",
        "rotation": "first_two_columns_column_major",
        "recipe": {"kinematic_model": "xarm7", "eef_link": "custom_eef_link"},
        "alignment": "control_step_latest_causal_before_action",
    },
    "fingertip_points": {
        "semantic_id": "dexmani.fingertip_points.xarm_base",
        "frame": "xarm_base",
        "unit": "m",
        "finger_order": _FINGERS,
        "alignment": "control_step_latest_causal_before_action",
    },
    "contact_force": {
        "semantic_id": "dexmani.xhand.tactile_aggregate.native",
        "representation": "xhand_sdk_calc_force_fx_fy_fz_bias_corrected",
        "unit": "xhand_sdk_native_unknown_si",
        "frame": "xhand_sensor_native_axes_per_finger",
        "finger_order": _FINGERS,
        "sensor_order": [2, 5, 7, 9, 11],
        "axis_order": ["fx", "fy", "fz"],
        "missing": "nan",
        "alignment": "control_step_latest_causal_before_action",
    },
    "tactile_force": {
        "semantic_id": "dexmani.xhand.tactile_dense.native",
        "representation": "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
        "unit": "xhand_sdk_native_unknown_si",
        "frame": "xhand_sensor_native_axes_per_finger",
        "finger_order": _FINGERS,
        "sensor_order": [2, 5, 7, 9, 11],
        "point_order": "xhand_sdk_sensor_data_raw_force_order",
        "axis_order": ["fx", "fy", "fz"],
        "missing": "nan",
        "alignment": "control_step_latest_causal_before_action",
    },
    "rgb": {
        "semantic_id": "dexmani.rgb.aligned_color",
        "channels": ["r", "g", "b"],
        "range": [0, 255],
        "alignment": "control_step_latest_causal_before_action",
    },
    "depth": {
        "semantic_id": "dexmani.depth.aligned_z16",
        "unit": "camera_depth_unit",
        "invalid_value": 0,
        "grid": "aligned_color",
        "alignment": "control_step_latest_causal_before_action",
    },
    "point_cloud": {
        "semantic_id": "dexmani.point_cloud.xyzrgb.xarm_base",
        "frame": "xarm_base",
        "features": ["x", "y", "z", "r", "g", "b"],
        "xyz_unit": "m",
        "rgb_range": [0, 1],
        "alignment": "control_step_latest_causal_before_action",
    },
    "action": {
        "semantic_id": "dexmani.action.joint_absolute_published",
        "unit": "rad",
        "joint_order": _ARM_ORDER + _HAND_ORDER,
        "alignment": "published_target_for_current_control_step",
    },
    "action_ee": {
        "semantic_id": "dexmani.action.ee_target",
        "frame": "xarm_base",
        "components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
        "rotation": "first_two_columns_column_major",
        "hand_joint_order": _HAND_ORDER,
        "recipe": {
            "kinematic_model": "xarm7",
            "eef_link": "custom_eef_link",
            "source": "final_published_joint_target",
        },
        "alignment": "published_target_for_current_control_step",
    },
}


def validate_modality_contract(name, attrs):
    """Validate a selected representation and retain only its runtime contract."""
    if name not in _SEMANTICS:
        raise ValueError(
            f"Required modality {name!r} has no supported Real representation"
        )
    if not isinstance(attrs, dict):
        raise ValueError(f"Required modality {name!r} needs a semantic contract")
    contract = {}
    for key, expected in _SEMANTICS[name].items():
        if attrs.get(key) != expected:
            raise ValueError(
                f"Required modality {name!r} has incompatible {key}: expected {expected!r}, got {attrs.get(key)!r}"
            )
        contract[key] = expected
    if name in {"point_cloud", "fingertip_points"}:
        recipe = attrs.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError(
                f"Required modality {name!r} is missing its representation recipe"
            )
        if name == "point_cloud":
            if (
                type(recipe.get("num_points")) is not int
                or recipe["num_points"] <= 0
                or type(recipe.get("remove_table")) is not bool
            ):
                raise ValueError(
                    "Required modality 'point_cloud' has an invalid numerical recipe"
                )
            derivation = attrs.get("derivation")
            if not isinstance(derivation, dict) or any(
                not isinstance(derivation.get(key), str) or not derivation[key]
                for key in ("transform", "color_source", "sampling")
            ):
                raise ValueError(
                    "Required modality 'point_cloud' is missing derivation semantics"
                )
            contract["derivation"] = derivation
        else:
            links = recipe.get("fingertip_link_names")
            model = recipe.get("kinematic_model")
            if (
                not isinstance(model, str)
                or not model
                or "/" in model
                or "\\" in model
                or recipe.get("mount_source") != "raw_episode"
                or not isinstance(links, list)
                or len(links) != 5
                or any(not isinstance(link, str) or not link for link in links)
                or len(set(links)) != 5
            ):
                raise ValueError(
                    "Required modality 'fingertip_points' has an invalid FK recipe"
                )
        # Preserve the actual training representation; never the historical table plane.
        json.dumps(recipe, allow_nan=False)
        contract["recipe"] = recipe
    if name == "depth":
        scale = attrs.get("scale_m_per_unit")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (float, int))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(
                "Required modality 'depth' needs one finite positive scale_m_per_unit"
            )
        contract["scale_m_per_unit"] = scale
    return contract


def read_real_contract(path, requested):
    """Inspect only requested arrays before ReplayBuffer loads their data."""
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("format") != _CANONICAL_FORMAT:
        has_real_semantics = any(
            f"data/{name}" in root
            and str(root[f"data/{name}"].attrs.get("semantic_id", "")).startswith(
                "dexmani."
            )
            for name in requested
        )
        if (
            has_real_semantics
            or attrs.get("domain") == "real"
            or str(attrs.get("format", "")).startswith("dexmani.real.")
        ):
            raise ValueError(f"Real training requires format={_CANONICAL_FORMAT!r}")
        return None  # Simulation datasets retain their existing contract.
    task = attrs.get("task_name")
    if (
        not isinstance(task, str)
        or not task.strip()
        or task != task.strip()
        or task == "unknown"
        or any(ord(char) < 32 or ord(char) == 127 for char in task)
    ):
        raise ValueError(
            "Real canonical task_name must be a nonempty, trimmed task identity"
        )
    dt = attrs.get("dt")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Real canonical dt must be finite and positive")
    if "meta/episode_ends" not in root:
        raise ValueError("Real canonical requires meta/episode_ends")
    ends = root["meta/episode_ends"][:]
    if (
        ends.ndim != 1
        or ends.dtype != np.int64
        or len(ends) == 0
        or ends[0] <= 0
        or np.any(np.diff(ends) <= 0)
    ):
        raise ValueError(
            "Real canonical episode_ends must be strictly increasing int64 endpoints"
        )
    contracts = {}
    for name in dict.fromkeys(requested):
        if f"data/{name}" not in root:
            raise ValueError(f"Required modality {name!r} is missing")
        array = root[f"data/{name}"]
        if not isinstance(array, zarr.Array):
            raise ValueError(f"Required modality {name!r} must be an array")
        contract = validate_modality_contract(name, dict(array.attrs))
        tail = array.shape[1:]
        dtype = (
            np.uint8 if name == "rgb" else np.uint16 if name == "depth" else np.float32
        )
        if name == "point_cloud":
            expected = (contract["recipe"]["num_points"], 6)
        elif name == "rgb":
            expected = (
                tail if len(tail) == 3 and min(tail[:2]) > 0 and tail[-1] == 3 else None
            )
        elif name == "depth":
            expected = tail if len(tail) == 2 and min(tail) > 0 else None
        else:
            expected = _SHAPES[name]
        if (
            not array.shape
            or array.shape[0] != ends[-1]
            or tail != expected
            or array.dtype != dtype
        ):
            raise ValueError(
                f"Required modality {name!r} has incompatible shape/dtype: {array.shape}, {array.dtype}"
            )
        contracts[name] = contract
    return {
        "task_name": task,
        "control_dt_s": float(dt),
        "modality_contracts": contracts,
    }


def validate_real_finiteness(replay_buffer):
    # The actual loaded subset is the normalization source. All current policy
    # paths lack a missing-data mask; optional telemetry is usable only if finite.
    for name, values in replay_buffer.items():
        if np.issubdtype(values.dtype, np.floating):
            for start in range(0, len(values), 256):
                if not np.isfinite(values[start : start + 256]).all():
                    raise ValueError(
                        f"Required modality {name!r} contains non-finite values; this policy has no missing-data support"
                    )


def extract_real_runtime(dataset, cfg):
    contract = getattr(dataset, "real_contract", None)
    if contract is None:
        return None
    if contract["task_name"] != cfg.task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    cloud = contract["modality_contracts"].get("point_cloud")
    if cloud is not None:
        encoder = cfg.agent.get("pc_encoder_config", {})
        for count in (cfg.agent.get("num_points"), encoder.get("num_points")):
            if count is not None and count != cloud["recipe"]["num_points"]:
                raise ValueError(
                    "Agent point count disagrees with the actual training cloud"
                )
    return {
        "control_dt_s": contract["control_dt_s"],
        "modality_contracts": contract["modality_contracts"],
    }
