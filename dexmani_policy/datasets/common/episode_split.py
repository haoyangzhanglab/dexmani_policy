"""Seeded episode splits and explicit trial manifests with strict restoration."""

import math
from numbers import Integral, Real

import numpy as np
from omegaconf import OmegaConf


def validate_split_manifest(manifest, attrs, episode_count):
    if OmegaConf.is_config(manifest):
        manifest = OmegaConf.to_container(manifest, resolve=True)
    ids = attrs.get("episode_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != episode_count
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
        or manifest.get("episode_ids") != ids
    ):
        raise ValueError("split manifest episode_ids must match canonical order exactly")
    canonical_ids = set(ids)
    trials = manifest.get("trial_ids")
    if (
        not isinstance(trials, dict)
        or set(trials) != canonical_ids
        or any(not isinstance(t, str) or not t.strip() for t in trials.values())
    ):
        raise ValueError("every canonical episode requires a confirmed trial_id")
    groups = []
    for key in ("train_ids", "val_ids", "exclusions"):
        values = manifest.get(key)
        if (
            not isinstance(values, list)
            or any(not isinstance(i, str) for i in values)
            or len(set(values)) != len(values)
            or not set(values) <= canonical_ids
        ):
            raise ValueError(f"invalid {key} in split manifest")
        groups.append(set(values))
    train, val, excluded = groups
    if train & val or train & excluded or val & excluded or train | val | excluded != canonical_ids:
        raise ValueError("train/val/exclusions must partition every canonical episode")
    if not train:
        raise ValueError("split manifest requires training episodes")
    if {trials[i] for i in train} & {trials[i] for i in val}:
        raise ValueError("a trial cannot cross train/val")
    normalized = dict(
        manifest,
        train_ids=[i for i in ids if i in train],
        val_ids=[i for i in ids if i in val],
        exclusions=[i for i in ids if i in excluded],
    )
    return (
        np.array([i in train for i in ids]),
        np.array([i in val for i in ids]),
        normalized,
    )


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    if max_n is None or mask.sum() <= max_n:
        return mask
    rng = np.random.default_rng(seed=seed)
    selected = rng.choice(np.flatnonzero(mask), size=max_n, replace=False)
    result = np.zeros_like(mask)
    result[selected] = True
    return result


def validate_split_options(val_ratio, max_train_episodes, *, explicit=False):
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, Real)
        or not math.isfinite(val_ratio)
        or not 0 <= val_ratio < 1
    ):
        raise ValueError(f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio!r}")
    if max_train_episodes is not None and (
        isinstance(max_train_episodes, bool)
        or not isinstance(max_train_episodes, Integral)
        or max_train_episodes < 1
    ):
        raise ValueError("max_train_episodes must be None or a positive integer")
    if explicit and (max_train_episodes is not None or val_ratio != 0):
        raise ValueError(
            "Explicit splits define final IDs: set max_train_episodes=null and val_ratio=0"
        )
