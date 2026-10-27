import math

import numpy as np

from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS


def describe_hand_data(buffer, *, action_key, horizon, obs_horizon, pad_before, pad_after, history_steps=None):
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

    return {
        "action_key": action_key, "dt": float(dt),
        "horizon": horizon, "obs_horizon": obs_horizon,
        "pad_before": pad_before, "pad_after": pad_after,
        "history_steps": history_steps,
    }


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
    source_id = source_id or stored_source
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
