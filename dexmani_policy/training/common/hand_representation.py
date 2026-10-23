"""Data settings shared by hand representation pretraining and policy training."""

import math

import numpy as np


def shuffled_batches(indices, batch_size, rng):
    """Repeat shuffled passes, yielding the final partial batch before reshuffling."""
    while True:
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
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
