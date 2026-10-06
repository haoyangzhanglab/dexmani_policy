"""Explicit episode/trial split for one immutable canonical revision."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_split_manifest(path, attrs, episode_count):
    # Read once: masks, digest and saved recipe all describe these exact contents.
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise TypeError("split manifest must be an object")
    revision = attrs.get("data_revision")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().lower() == "unknown"
    ):
        raise ValueError("split manifest requires a known canonical data_revision")
    if manifest.get("data_revision") != revision:
        raise ValueError("split manifest data_revision mismatch")
    ids = attrs.get("episode_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != episode_count
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
        or manifest.get("episode_ids") != ids
    ):
        raise ValueError(
            "split manifest episode_ids must match canonical order exactly"
        )
    trials = manifest.get("trial_ids")
    if (
        not isinstance(trials, dict)
        or set(trials) != set(ids)
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
            or not set(values) <= set(ids)
        ):
            raise ValueError(f"invalid {key} in split manifest")
        groups.append(set(values))
    train, val, excluded = groups
    if (
        train & val
        or train & excluded
        or val & excluded
        or train | val | excluded != set(ids)
    ):
        raise ValueError("train/val/exclusions must partition every canonical episode")
    if not train:
        raise ValueError("split manifest requires training episodes")
    if {trials[i] for i in train} & {trials[i] for i in val}:
        raise ValueError("a trial cannot cross train/val")
    if (
        type(manifest.get("seed")) is not int
        or not isinstance(manifest.get("group_unit"), str)
        or not manifest["group_unit"].strip()
    ):
        raise ValueError("split manifest requires seed and group_unit")
    normalized = dict(
        manifest,
        train_ids=[i for i in ids if i in train],
        val_ids=[i for i in ids if i in val],
        exclusions=[i for i in ids if i in excluded],
    )
    digest = hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return (
        np.array([i in train for i in ids]),
        np.array([i in val for i in ids]),
        normalized,
        digest,
    )
