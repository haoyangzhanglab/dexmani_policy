"""Seeded episode splits and explicit index/trial manifests with strict restoration."""

import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def validate_split_manifest(manifest, attrs, episode_count, *, episode_ends, zarr_path):
    if OmegaConf.is_config(manifest):
        manifest = OmegaConf.to_container(manifest, resolve=True)
    manifest_format = manifest.get("format")
    if manifest_format == "dexmani.episode_indices.v1":
        source = manifest.get("source_id")
        if source is not None and (not isinstance(source, str) or not source or source != attrs.get("source_id")):
            raise ValueError("Indexed split manifest source identity mismatch")
        if (
            (manifest["zarr_path"] != str(Path(zarr_path).expanduser().resolve())
             and not (manifest.get("source_id") is not None and manifest["source_id"] == attrs.get("source_id")))
            or manifest["episode_ends"] != np.asarray(episode_ends).tolist()
            or manifest["episode_ids"] != attrs.get("episode_ids")
        ):
            raise ValueError("Indexed split manifest data path/episode identity mismatch")
        ids = list(range(episode_count))
        keys = ("train_indices", "val_indices", "exclusions")
        value_type = int
    elif manifest_format == "dexmani.episode_trials.v1":
        ids = attrs.get("episode_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != episode_count
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
            or manifest["episode_ids"] != ids
        ):
            raise ValueError("split manifest episode_ids must match canonical order exactly")
        trials = manifest["trial_ids"]
        if (
            not isinstance(trials, dict)
            or set(trials) != set(ids)
            or any(not isinstance(t, str) or not t.strip() for t in trials.values())
        ):
            raise ValueError("every canonical episode requires a confirmed trial_id")
        keys = ("train_ids", "val_ids", "exclusions")
        value_type = str
    else:
        raise ValueError("Unsupported split manifest format; provide an explicit current format")

    canonical_ids = set(ids)
    groups = []
    for key in keys:
        values = manifest[key]
        if (
            not isinstance(values, list)
            or any(type(value) is not value_type for value in values)
            or len(set(values)) != len(values)
            or not set(values) <= canonical_ids
        ):
            raise ValueError(f"Invalid {key} in split manifest")
        groups.append(set(values))
    train, val, excluded = groups
    if (
        not train or train & val or train & excluded or val & excluded
        or train | val | excluded != canonical_ids
    ):
        raise ValueError("train/val/exclusions must partition every episode, with nonempty train")
    if manifest_format == "dexmani.episode_trials.v1":
        if {trials[i] for i in train} & {trials[i] for i in val}:
            raise ValueError("a trial cannot cross train/val")
    normalized = dict(manifest)
    normalized.update({key: [i for i in ids if i in group] for key, group in zip(keys, groups)})
    return (
        np.array([i in train for i in ids]),
        np.array([i in val for i in ids]),
        normalized,
    )


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


def build_index_manifest(
    zarr_path, episode_ends, attrs, *, val_ratio=0.1, seed=42, max_train_episodes=None
):
    validate_split_options(val_ratio, max_train_episodes)
    episode_ends = np.asarray(episode_ends)
    if (
        episode_ends.ndim != 1 or not len(episode_ends)
        or episode_ends.dtype.kind not in "iu"
        or episode_ends[0] <= 0 or np.any(episode_ends[1:] <= episode_ends[:-1])
    ):
        raise ValueError("episode_ends must be nonempty, positive, strictly increasing integers")
    if "trial_ids" in attrs:
        ids, trials = attrs.get("episode_ids"), attrs["trial_ids"]
        if (not isinstance(ids, list) or len(ids) != len(episode_ends)
                or not isinstance(trials, dict) or set(trials) != set(ids)
                or any(not isinstance(value, str) or not value for value in trials.values())):
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
        manifest = {
            "format": "dexmani.episode_trials.v1", "episode_ids": ids, "trial_ids": trials,
            "train_ids": [i for i in ids if trials[i] in train_groups],
            "val_ids": [i for i in ids if trials[i] in validation],
            "exclusions": [i for i in ids if trials[i] not in train_groups | validation],
        }
        return validate_split_manifest(manifest, attrs, len(episode_ends), episode_ends=episode_ends, zarr_path=zarr_path)[2]
    val = get_val_mask(len(episode_ends), val_ratio, seed)
    train = downsample_mask(~val, max_train_episodes, seed)
    manifest = dict(
        format="dexmani.episode_indices.v1",
        zarr_path=str(Path(zarr_path).expanduser().resolve()),
        episode_ends=episode_ends.tolist(),
        episode_ids=attrs.get("episode_ids"),
        train_indices=np.flatnonzero(train).tolist(),
        val_indices=np.flatnonzero(val).tolist(),
        exclusions=np.flatnonzero(~(train | val)).tolist(),
    )
    if attrs.get("source_id") is not None:
        manifest["source_id"] = attrs["source_id"]
    return validate_split_manifest(
        manifest, attrs, len(episode_ends), episode_ends=episode_ends, zarr_path=zarr_path
    )[2]


def main():
    import argparse
    import json

    import zarr

    from dexmani_policy.common.runtime import atomic_path

    parser = argparse.ArgumentParser(description="Create one episode-index split for pretraining and BC")
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-episodes", type=int)
    args = parser.parse_args()
    root = zarr.open_group(str(Path(args.zarr_path).expanduser()), mode="r")
    manifest = build_index_manifest(
        args.zarr_path, root["meta/episode_ends"][:], dict(root.attrs),
        val_ratio=args.val_ratio, seed=args.seed, max_train_episodes=args.max_train_episodes,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=False) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Split: train={len(manifest['train_indices'])}, validation={len(manifest['val_indices'])}; {output}")


if __name__ == "__main__":
    main()
