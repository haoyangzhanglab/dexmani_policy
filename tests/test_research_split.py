"""Synthetic split audit and resume contracts; no training or existing assets."""

import hashlib
import json

import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.split import load_split_manifest
from dexmani_policy.training.resume import validate_resume_contract


def fixture(tmp_path):
    path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(path), "w")
    root.create_dataset("meta/episode_ends", data=np.array([5, 10, 15, 20, 25]))
    values = np.arange(25, dtype=np.float32)[:, None]
    root.create_dataset("data/joint_state", data=values)
    root.create_dataset("data/action", data=values)
    root.attrs.update(
        data_revision="known-revision", episode_ids=["a", "b", "c", "d", "e"]
    )
    manifest = {
        "data_revision": "known-revision",
        "episode_ids": ["a", "b", "c", "d", "e"],
        "trial_ids": {
            "a": "first",
            "b": "first",
            "c": "second",
            "d": "second",
            "e": "third",
        },
        "train_ids": ["a", "b"],
        "val_ids": ["c", "d"],
        "exclusions": ["e"],
        "seed": 42,
        "group_unit": "physical_reset",
    }
    mp = tmp_path / "split.json"
    mp.write_text(json.dumps(manifest))
    return path, root, mp, manifest


def test_manifest_masks_content_and_windows_are_one_snapshot(tmp_path):
    path, _root, mp, manifest = fixture(tmp_path)
    ds = BaseDataset(
        str(path),
        horizon=3,
        obs_horizon=2,
        split_manifest=str(mp),
        val_ratio=0.8,
        max_train_episodes=1,
    )
    saved = ds.data_recipe["split_manifest"]
    assert ds.train_mask.sum() == 1 and ds.val_mask.tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert not ds.train_mask[-1] and not ds.val_mask[-1]
    assert set(ds.sampler.source_rows().ravel() // 5) <= {0, 1}
    assert set(ds.get_validation_dataset().sampler.source_rows().ravel() // 5) == {2, 3}
    digest = hashlib.sha256(
        json.dumps(
            saved["content"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    assert digest == saved["sha256"] and saved["train_mask"] == ds.train_mask.tolist()
    assert saved["train_windows"] == len(ds) and saved["val_windows"] == len(
        ds.get_validation_dataset()
    )
    mp.write_text("{}")
    assert saved["content"]["trial_ids"] == manifest["trial_ids"]
    # Full content is serializable through the existing resolved config owner.
    assert (
        OmegaConf.to_container(
            OmegaConf.create({"data_recipe": [ds.data_recipe]}), resolve=True
        )["data_recipe"][0]
        == ds.data_recipe
    )


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "missing_revision",
        "order",
        "duplicate",
        "unknown_id",
        "omitted",
        "overlap",
        "trial_leak",
    ],
)
def test_bad_manifests_rejected(tmp_path, change):
    _path, root, mp, m = fixture(tmp_path)
    if change == "unknown":
        root.attrs["data_revision"] = "unknown"
        m["data_revision"] = "unknown"
    elif change == "missing_revision":
        del root.attrs["data_revision"]
        m["data_revision"] = None
    elif change == "order":
        m["episode_ids"] = list(reversed(m["episode_ids"]))
    elif change == "duplicate":
        m["train_ids"] = ["a", "a", "b"]
    elif change == "unknown_id":
        m["exclusions"] = ["x"]
    elif change == "omitted":
        m["exclusions"] = []
    elif change == "overlap":
        m["train_ids"].append("c")
    elif change == "trial_leak":
        m["trial_ids"]["c"] = "first"
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_split_manifest(mp, dict(root.attrs), 5)


def test_legacy_contract_unchanged_and_manifest_change_rejects_resume(tmp_path):
    path, root, mp, m = fixture(tmp_path)
    del root.attrs["data_revision"]
    cfg = {
        "_target_": "dexmani_policy.datasets.base_dataset.BaseDataset",
        "zarr_path": str(path),
        "horizon": 3,
    }
    old_recipe = {
        "window_validity": "role_finite_v1",
        "dispatch": "not_applicable",
        "normalization": "unique_train_source_rows",
    }
    ds = BaseDataset(str(path), horizon=3)
    assert ds.data_recipe == old_recipe and "split_manifest" not in cfg
    with pytest.warns(UserWarning):
        validate_resume_contract(
            {"dataset": cfg, "data_recipe": [old_recipe]},
            {"dataset": dict(cfg), "data_recipe": [ds.data_recipe]},
        )
    root.attrs["data_revision"] = "known-revision"
    first = BaseDataset(str(path), horizon=3, split_manifest=str(mp))
    saved = {
        "dataset": dict(cfg, split_manifest=str(mp)),
        "data_recipe": [first.data_recipe],
        "data_identity": {"revision": "known-revision"},
    }
    validate_resume_contract(saved, saved)
    m["trial_ids"]["a"] = m["trial_ids"]["b"] = "renamed-trial"
    mp.write_text(json.dumps(m))
    changed = BaseDataset(str(path), horizon=3, split_manifest=str(mp))
    with pytest.raises(ValueError, match="data_recipe"):
        validate_resume_contract(saved, dict(saved, data_recipe=[changed.data_recipe]))


def test_no_holdout_is_explicit(tmp_path):
    path, _root, mp, m = fixture(tmp_path)
    m["train_ids"] += m["val_ids"]
    m["val_ids"] = []
    mp.write_text(json.dumps(m))
    ds = BaseDataset(str(path), horizon=3, split_manifest=str(mp))
    assert ds.get_validation_dataset() is None
    assert ds.data_recipe["split_manifest"]["holdout"] is False
