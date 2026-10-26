import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from dexmani_policy.common.config import positive_int, validate_depth_recipe
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS


def describe_hand_data(dataset):
    buffer = dataset.replay_buffer
    attrs = buffer.attrs
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand representation data requires finite positive dt")
    order = ["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"]
    if (
        attrs.get("hand_joint_order", order) != order
        or attrs.get("hand_units", "radians") != "radians"
    ):
        raise ValueError("Hand data must use XHand thumb/index/middle/ring/pinky order in radians")

    validation = dataset.get_validation_dataset()
    return {
        "episode_ids": attrs.get("episode_ids"),
        "episode_ends": buffer.episode_ends.tolist(),
        "train_episodes": np.flatnonzero(dataset.train_mask).tolist(),
        "val_episodes": np.flatnonzero(dataset.val_mask).tolist(),
        "action_key": dataset.action_key,
        "dt": float(dt),
        "horizon": dataset.horizon,
        "obs_horizon": dataset.obs_horizon,
        "pad_before": dataset.pad_before,
        "pad_after": dataset.pad_after,
        "history_steps": getattr(dataset, "history_steps", None),
        "train_windows": len(dataset),
        "val_windows": 0 if validation is None else len(validation),
        "train_action_rows": len(dataset.sampler.action_source_rows),
        "val_action_rows": 0 if validation is None else len(validation.sampler.action_source_rows),
    }


def describe_real_runtime(dataset, task_name) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    buffer = dataset.replay_buffer
    attrs = buffer.attrs
    if attrs.get("format") != "dexmani.real.canonical":
        if attrs.get("domain") == "real" or str(attrs.get("format", "")).startswith(
            "dexmani.real."
        ):
            raise ValueError("Real training requires format='dexmani.real.canonical'")
        return None
    if attrs.get("task_name") != task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Real canonical dt must be finite and positive")
    runtime: dict[str, Any] = {"dt": float(dt)}
    if "point_cloud" in dataset.sensor_modalities:
        cloud = attrs.get("pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real point_cloud requires root pointcloud_config")
        count = positive_int(cloud["num_points"], "pointcloud_config.num_points")
        shape = buffer["point_cloud"].shape
        if shape[1] != count:
            raise ValueError("Stored point count disagrees with pointcloud_config")
        # Preserve the stored point count; encoder FPS selects its own subset.
        runtime["pointcloud"] = dict(cloud)
    if "hand_pointcloud" in dataset.sensor_modalities:
        cloud = attrs.get("hand_pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real hand_pointcloud requires root hand_pointcloud_config")
        count = positive_int(cloud["num_points"], "hand_pointcloud_config.num_points")
        data = buffer["hand_pointcloud"]
        if data.shape[1:] != (count, 6) or str(data.dtype) != "float32":
            raise ValueError("Stored hand_pointcloud must match the saved XYZRGB template")
        runtime["hand_pointcloud"] = dict(cloud)
    if "depth" in dataset.sensor_modalities:
        data = buffer["depth"]
        runtime["depth"] = validate_depth_recipe(
            {
                "shape": list(data.shape[1:]),
                "dtype": str(data.dtype),
                "unit": "raw_counts",
                "depth_scale_m": attrs["depth_scale_m_per_unit"],
            }
        )
    if "fingertip_xyz" in dataset.sensor_modalities:
        links = attrs.get("fingertip_link_names")
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Real fingertip_xyz requires five distinct non-empty link names")
        runtime["fingertip_link_names"] = list(links)
    return runtime


def describe_hand_chunks(dataset, data_semantics, source_id):
    """Record stored semantics separately from explicit configuration declarations."""
    if data_semantics != HAND_ACTION_SEMANTICS:
        raise ValueError("Hand chunks require the declared arm7/XHand12 absolute-command semantics")
    semantics = HAND_ACTION_SEMANTICS
    buffer = dataset.replay_buffer
    for key in ("action", "joint_state"):
        if buffer[key].shape[1:] != (19,):
            raise ValueError(f"{key} must have shape [T,19]")
    attrs = buffer.attrs
    evidence = {}
    for key, expected in semantics.items():
        if key in attrs and attrs[key] != expected:
            raise ValueError(f"Stored {key} conflicts with declared hand action semantics")
        evidence[key] = "zarr" if key in attrs else "config_declaration"
    if attrs.get("hand_joint_order", semantics["joint_order"][1:]) != semantics["joint_order"][1:]:
        raise ValueError("Stored hand_joint_order differs from XHand order")
    if attrs.get("hand_units", "radians") != "radians":
        raise ValueError("Stored hand_units must be radians")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand chunks require finite positive recorded dt")
    stored_source = attrs.get("source_id")
    if source_id is not None and (not isinstance(source_id, str) or not source_id):
        raise ValueError("source_id must be a nonempty string")
    if source_id is not None and stored_source is not None and source_id != stored_source:
        raise ValueError("Declared source_id differs from Zarr source_id")
    source_id = source_id or stored_source or dataset.zarr_path
    episode_ids = attrs.get("episode_ids")
    return {
        "contract": dict(semantics, horizon=dataset.horizon, dt=float(dt)),
        "provenance": {
            "source_id": source_id,
            "zarr_path": dataset.zarr_path,
            "semantics_evidence": evidence,
            "episode_ids": episode_ids,
            "episode_ends": buffer.episode_ends.tolist(),
            "train_episodes": np.flatnonzero(dataset.train_mask).tolist(),
            "val_episodes": np.flatnonzero(dataset.val_mask).tolist(),
        },
    }

