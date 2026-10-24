"""Data settings shared by hand representation pretraining and policy training."""

import math

import numpy as np
from omegaconf import OmegaConf


def shuffled_batches(indices, batch_size, rng, *, drop_last=False):
    """Repeat shuffled passes, optionally dropping each pass's final partial batch."""
    count = int(indices) if isinstance(indices, (int, np.integer)) else len(indices)
    if count == 0:
        raise ValueError("Cannot generate shuffled batches from empty indices")
    if drop_last and count < batch_size:
        raise ValueError("drop_last=True requires at least batch_size indices")
    stop = count - count % batch_size if drop_last else count
    while True:
        order = rng.permutation(indices)
        for start in range(0, stop, batch_size):
            yield order[start : start + batch_size]


def describe_hand_data(dataset):
    """Read dataset metadata and sampler counts without scanning action/state arrays."""
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


def validate_kl_schedule(steps, beta, eligible_step):
    if not math.isfinite(beta) or beta <= 0 or steps < eligible_step:
        raise ValueError(
            f"Hand pretraining requires positive beta and steps >= {eligible_step} "
            "to complete KL warmup/ramp"
        )


HAND_ACTION_SEMANTICS = {
    "action_space": "joint",
    "action_coordinates": "absolute_joint_command",
    "joint_order": ["arm:7", "thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
    "units": "radians",
    "obs_alignment": "obs[t]_before_action[t]",
    "reference": "measured_query_state",
    "window": "query_future_full_horizon",
}


def describe_hand_chunks(dataset, data_semantics, source_id):
    """Record stored semantics separately from explicit configuration declarations."""
    semantics = (
        OmegaConf.to_container(data_semantics, resolve=True)
        if OmegaConf.is_config(data_semantics) else dict(data_semantics)
    )
    if semantics != HAND_ACTION_SEMANTICS:
        raise ValueError("Hand chunks require the declared arm7/XHand12 absolute-command semantics")
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
    for value in (source_id, stored_source):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("source_id must be a nonempty string")
    if source_id is not None and stored_source is not None and source_id != stored_source:
        raise ValueError("Declared source_id differs from Zarr source_id")
    source_id = source_id or stored_source or dataset.zarr_path
    episode_ids = attrs.get("episode_ids")
    if episode_ids is not None and (
        not isinstance(episode_ids, list) or len(episode_ids) != buffer.n_episodes
        or any(not isinstance(value, str) or not value for value in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError("episode_ids must be distinct nonempty strings matching episodes")
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


def validate_hand_chunk_compatibility(codec_data, policy_data):
    """Fresh reuse checks coordinates and split leakage, not corpus size or eligibility."""
    left, right = codec_data["contract"], policy_data["contract"]
    if left.keys() != right.keys() or any(left[key] != right[key] for key in left if key != "dt"):
        raise ValueError("Codec/policy hand action contract mismatch")
    if not math.isclose(left["dt"], right["dt"], rel_tol=1e-6, abs_tol=0):
        raise ValueError("Codec/policy dt mismatch; no automatic resampling")
    source, target = codec_data["provenance"], policy_data["provenance"]
    if source["source_id"] != target["source_id"]:
        return
    source_ids, target_ids = source["episode_ids"], target["episode_ids"]
    if source_ids is None or target_ids is None:
        if source["episode_ends"] != target["episode_ends"] or source_ids != target_ids:
            raise ValueError("Same-source subsets require stable episode_ids")
        source_ids = target_ids = list(range(len(source["episode_ends"])))
    source_train = {source_ids[i] for i in source["train_episodes"]}
    source_val = {source_ids[i] for i in source["val_episodes"]}
    target_train = {target_ids[i] for i in target["train_episodes"]}
    target_val = {target_ids[i] for i in target["val_episodes"]}
    shared = set(source_ids) & set(target_ids)
    if source_train & target_val or source_val & target_train or (source_val & shared) != (target_val & shared):
        raise ValueError("Same-source codec/policy split mismatch or validation leakage")
