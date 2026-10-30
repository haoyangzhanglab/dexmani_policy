import math

import numpy as np

from dexmani_policy.common.hand_semantics import HAND_ACTION_SEMANTICS, validate_hand_coordinates
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite
from dexmani_policy.datasets.split import split_episodes


def load_hand_source(cfg, keys, *, split_manifest: dict | None):
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare hand artifacts from a fresh policy config, not resume_from")
    if cfg.dataset.action_key != "action" or cfg.agent.action_dim != 19:
        raise ValueError("Hand pretraining requires action=arm7+hand12 joint commands")
    buffer = EpisodeStore(cfg.dataset.zarr_path, keys=keys)
    if buffer["action"].shape[1:] != (19,):
        raise ValueError("Hand pretraining requires action shape [T,19]")
    train, val, split = split_episodes(
        buffer, manifest=split_manifest, seed=cfg.dataset.seed,
        val_ratio=cfg.dataset.val_ratio, max_train_episodes=cfg.dataset.max_train_episodes,
    )
    return buffer, train, val, split


def episode_row_offsets(episode_ends, episode_mask):
    starts = np.r_[0, episode_ends[:-1]]
    lengths = (episode_ends - starts) * episode_mask
    return np.r_[0, lengths.cumsum()[:-1]] - starts


def load_hand_array(buffer, key, episode_mask):
    count = np.diff(np.r_[0, buffer.episode_ends])[episode_mask].sum()
    values = np.empty((count, 12), dtype=np.float32)
    cursor = 0
    for start, chunk in buffer.iter_chunks(key, columns=slice(7, 19), episode_mask=episode_mask):
        check_finite(buffer, key, start, chunk, columns=slice(7, 19))
        values[cursor:cursor + len(chunk)] = chunk
        cursor += len(chunk)
    return values


def recorded_dt(attrs):
    dt = attrs.get("dt")
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand motion data requires finite positive recorded dt")
    return float(dt)


def describe_hand_chunks(dataset, data_semantics):
    if data_semantics != HAND_ACTION_SEMANTICS:
        raise ValueError("Hand chunks require the declared arm7/XHand12 absolute-command semantics")
    buffer = dataset.episode_store
    for key in ("action", "joint_state"):
        if buffer[key].shape[1:] != (19,):
            raise ValueError(f"{key} must have shape [T,19]")

    attrs = buffer.attrs
    evidence = {}
    for key, expected in HAND_ACTION_SEMANTICS.items():
        if key in attrs and attrs[key] != expected:
            raise ValueError(f"Stored {key} conflicts with declared hand action semantics")
        evidence[key] = "zarr" if key in attrs else "config_declaration"
    validate_hand_coordinates(attrs)

    return {
        "contract": dict(HAND_ACTION_SEMANTICS, horizon=dataset.horizon, dt=recorded_dt(attrs)),
        "semantics_evidence": evidence,
    }
