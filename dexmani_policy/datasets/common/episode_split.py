"""Seeded episode partitions with explicit identity and optional trial grouping."""

import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def validate_split_manifest(manifest, attrs, episode_count, *, episode_ends, zarr_path):
    if OmegaConf.is_config(manifest):
        manifest = OmegaConf.to_container(manifest, resolve=True)
    source = manifest["source_id"]
    if source is not None and (
        not isinstance(source, str) or not source or source != attrs.get("source_id")
    ):
        raise ValueError("Split manifest source identity mismatch")
    if (
        (source is None and manifest["zarr_path"] != str(Path(zarr_path).expanduser().resolve()))
        or manifest["episode_ends"] != np.asarray(episode_ends).tolist()
        or manifest["episode_ids"] != attrs.get("episode_ids")
    ):
        raise ValueError("Split manifest data path/episode identity mismatch")
    ids, trials = manifest["episode_ids"], manifest["trial_ids"]
    if ids is not None and (
        not isinstance(ids, list)
        or len(ids) != episode_count
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("episode_ids must uniquely identify every episode in canonical order")
    if attrs.get("trial_ids") is not None and trials != attrs["trial_ids"]:
        raise ValueError("Split manifest trial_ids differ from the dataset")
    if trials is not None and (
        ids is None
        or not isinstance(trials, dict)
        or set(trials) != set(ids)
        or any(not isinstance(value, str) or not value.strip() for value in trials.values())
    ):
        raise ValueError("Trial grouping requires a confirmed trial_id for every episode")
    keys = ("train_indices", "val_indices", "exclusions")
    canonical = set(range(episode_count))
    groups = []
    for key in keys:
        values = manifest[key]
        if (
            not isinstance(values, list)
            or any(type(value) is not int for value in values)
            or len(set(values)) != len(values)
            or not set(values) <= canonical
        ):
            raise ValueError(f"Invalid {key} in split manifest")
        groups.append(set(values))
    train, val, excluded = groups
    if (
        not train
        or train & val
        or train & excluded
        or val & excluded
        or train | val | excluded != canonical
    ):
        raise ValueError("train/val/exclusions must partition every episode, with nonempty train")
    if trials is not None and {trials[ids[i]] for i in train} & {trials[ids[i]] for i in val}:
        raise ValueError("a trial cannot cross train/val")
    normalized = dict(manifest)
    normalized.update({key: sorted(group) for key, group in zip(keys, groups)})
    return (
        np.array([i in train for i in range(episode_count)]),
        np.array([i in val for i in range(episode_count)]),
        normalized,
    )


def describe_split(zarr_path, episode_ends, attrs, train, val):
    return {
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "episode_ends": np.asarray(episode_ends).tolist(),
        "episode_ids": attrs.get("episode_ids"),
        "source_id": attrs.get("source_id"),
        "trial_ids": attrs.get("trial_ids"),
        "train_indices": np.flatnonzero(train).tolist(),
        "val_indices": np.flatnonzero(val).tolist(),
        "exclusions": np.flatnonzero(~(train | val)).tolist(),
    }


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    if n_episodes < 2:
        raise ValueError("Holdout requires at least two episodes; use val_ratio=0 for smoke")

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


def build_split_manifest(
    zarr_path, episode_ends, attrs, *, val_ratio=0.1, seed=42, max_train_episodes=None
):
    """Return train/validation masks and their validated manifest."""
    validate_split_options(val_ratio, max_train_episodes)
    episode_ends = np.asarray(episode_ends)
    if (
        episode_ends.ndim != 1
        or not len(episode_ends)
        or episode_ends.dtype.kind not in "iu"
        or episode_ends[0] <= 0
        or np.any(episode_ends[1:] <= episode_ends[:-1])
    ):
        raise ValueError("episode_ends must be nonempty, positive, strictly increasing integers")
    if "trial_ids" in attrs:
        ids, trials = attrs.get("episode_ids"), attrs["trial_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) != len(episode_ends)
            or not isinstance(trials, dict)
            or set(trials) != set(ids)
            or any(not isinstance(value, str) or not value for value in trials.values())
        ):
            raise ValueError("Trial grouping requires confirmed episode_ids and trial_ids")
        groups = list(dict.fromkeys(trials[i] for i in ids))
        val_groups = get_val_mask(len(groups), val_ratio, seed)
        validation = {group for group, selected in zip(groups, val_groups) if selected}
        eligible = [group for group in groups if group not in validation]
        train_groups, count = set(), 0
        for group in np.random.default_rng(seed).permutation(eligible):
            size = sum(trials[i] == group for i in ids)
            if max_train_episodes is None or count + size <= max_train_episodes:
                train_groups.add(group)
                count += size
        train = np.array([trials[i] in train_groups for i in ids])
        val = np.array([trials[i] in validation for i in ids])
    else:
        val = get_val_mask(len(episode_ends), val_ratio, seed)
        train = downsample_mask(~val, max_train_episodes, seed)
    manifest = describe_split(zarr_path, episode_ends, attrs, train, val)
    return validate_split_manifest(
        manifest, attrs, len(episode_ends), episode_ends=episode_ends, zarr_path=zarr_path
    )


def main():
    import argparse
    import json

    import zarr

    from dexmani_policy.common.runtime import atomic_path

    parser = argparse.ArgumentParser(
        description="Create a shared episode split, keeping known trials together"
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-episodes", type=int)
    args = parser.parse_args()
    root = zarr.open_group(str(Path(args.zarr_path).expanduser()), mode="r")
    _, _, manifest = build_split_manifest(
        args.zarr_path,
        root["meta/episode_ends"][:],
        dict(root.attrs),
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=False) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Split: train={len(manifest['train_indices'])}, validation={len(manifest['val_indices'])}; {output}"
    )


if __name__ == "__main__":
    main()
