from collections import Counter

import numpy as np

from dexmani_policy.common.config import positive_int


def validate_episode_metadata(metadata, count):
    source = metadata.get("source_id")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError("source_id must be a nonempty string")
    ids, trials = metadata.get("episode_ids"), metadata.get("trial_ids")
    if ids is not None and (
        not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count
        or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise ValueError("episode_ids must uniquely identify every episode")
    if trials is not None and (
        ids is None or set(trials) != set(ids)
        or any(not isinstance(value, str) or not value.strip() for value in trials.values())
    ):
        raise ValueError("Trial grouping requires a trial_id for every episode")
    return ids, trials


def _restore_split(manifest, store):
    identity = {
        "episode_ends": store.episode_ends.tolist(),
        **{key: store.attrs.get(key) for key in ("episode_ids", "source_id", "trial_ids")},
    }
    for key, expected in identity.items():
        if manifest[key] != expected:
            raise ValueError(f"Split manifest {key} differs from the dataset")
    ids, trials = identity["episode_ids"], identity["trial_ids"]

    keys = ("train_indices", "val_indices")
    canonical = set(range(store.n_episodes))
    groups = []
    for key in keys:
        values = manifest[key]
        selected = set(values)
        if (
            any(type(value) is not int for value in values)
            or len(selected) != len(values) or not selected <= canonical
        ):
            raise ValueError(f"Invalid {key} in split manifest")
        groups.append(selected)
    train, val = groups
    if not train or train & val:
        raise ValueError("train/val must be disjoint, with nonempty train")
    if trials is not None and {trials[ids[i]] for i in train} & {trials[ids[i]] for i in val}:
        raise ValueError("a trial cannot cross train/val")
    manifest = {
        **identity,
        **{key: sorted(group) for key, group in zip(keys, groups)},
    }
    return (
        np.isin(np.arange(store.n_episodes), list(train)),
        np.isin(np.arange(store.n_episodes), list(val)),
        manifest,
    )


def get_val_mask(n_episodes, val_ratio, seed):
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio == 0:
        return mask
    if n_episodes < 2:
        raise ValueError("Holdout requires at least two episodes; use val_ratio=0 for smoke")
    count = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    mask[np.random.default_rng(seed).choice(n_episodes, count, replace=False)] = True
    return mask


def split_episodes(buffer, *, manifest: dict | None = None, val_ratio=0.1, seed=42, max_train_episodes=None):
    if max_train_episodes is not None:
        positive_int(max_train_episodes, "max_train_episodes")
    ids, trials = validate_episode_metadata(buffer.attrs, buffer.n_episodes)
    if manifest is not None:
        train, val, manifest = _restore_split(manifest, buffer)
        if max_train_episodes is not None and train.sum() > max_train_episodes:
            raise ValueError(
                f"Split manifest has {int(train.sum())} training episodes, "
                f"exceeding max_train_episodes={max_train_episodes}"
            )
        return train, val, manifest

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if trials is not None:
        sizes = Counter(trials[i] for i in ids)
        groups = list(sizes)
        val_groups = get_val_mask(len(groups), val_ratio, seed)
        validation = {group for group, selected in zip(groups, val_groups) if selected}
        eligible = [group for group in groups if group not in validation]
        train_groups, count = set(), 0
        for group in np.random.default_rng(seed).permutation(eligible):
            if max_train_episodes is None or count + sizes[group] <= max_train_episodes:
                train_groups.add(group)
                count += sizes[group]
        train = np.array([trials[i] in train_groups for i in ids])
        val = np.array([trials[i] in validation for i in ids])
    else:
        val = get_val_mask(buffer.n_episodes, val_ratio, seed)
        train = ~val
        if max_train_episodes is not None and train.sum() > max_train_episodes:
            selected = np.random.default_rng(seed).choice(
                np.flatnonzero(train), max_train_episodes, replace=False
            )
            train = np.isin(np.arange(buffer.n_episodes), selected)
    if not train.any():
        raise ValueError("No training episodes fit the requested split")
    manifest = {
        "episode_ends": buffer.episode_ends.tolist(),
        "episode_ids": ids,
        "source_id": buffer.attrs.get("source_id"),
        "trial_ids": trials,
        "train_indices": np.flatnonzero(train).tolist(),
        "val_indices": np.flatnonzero(val).tolist(),
    }
    return train, val, manifest
