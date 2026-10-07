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
    )
    saved = ds.data_recipe["split_manifest"]
    assert ds.train_mask.sum() == 2 and ds.val_mask.tolist() == [
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
            {"facts_format": 1, "data_recipe": [old_recipe]},
            {"facts_format": 1, "data_recipe": [ds.data_recipe]},
        )
    root.attrs["data_revision"] = "known-revision"
    first = BaseDataset(str(path), horizon=3, split_manifest=str(mp))
    saved = {
        "facts_format": 1,
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


@pytest.mark.parametrize("extra", [{"val_ratio": 0.8}, {"max_train_episodes": 1}])
def test_new_manifest_rejects_second_selection(tmp_path, extra, monkeypatch):
    from dexmani_policy.datasets.replay_buffer import ReplayBuffer
    path, _, manifest, _ = fixture(tmp_path)
    monkeypatch.setattr(ReplayBuffer, 'open', lambda *a, **kw: pytest.fail('split conflict reached I/O'))
    with pytest.raises(ValueError, match="defines final IDs"):
        BaseDataset(str(path), split_manifest=str(manifest), **extra)


def test_legacy_actual_ids_survive_missing_external_manifest(tmp_path):
    path, root, mp, manifest = fixture(tmp_path)
    _, _, normalized, digest = load_split_manifest(mp, dict(root.attrs), 5)
    # Independent historical manifest+cap evidence: only b was selected.
    saved = {"content": normalized, "sha256": digest, "actual_train_ids": ["b"],
             "train_mask": [False, True, False, False, False],
             "val_mask": [False, False, True, True, False]}
    mp.unlink()
    ds = BaseDataset(str(path), horizon=3, obs_horizon=2, split_manifest=str(mp),
                     max_train_episodes=1, val_ratio=.8, saved_split=saved)
    assert ds.train_mask.tolist() == saved["train_mask"]
    assert ds.val_mask.tolist() == saved["val_mask"]
    assert set(ds.sampler.source_rows().ravel() // 5) == {1}
    assert ds.data_recipe["split_manifest"]["actual_train_ids"] == ["b"]
    for change in ({"actual_train_ids": ["c"]}, {"sha256": "bad"},
                   {"train_mask": [True, True, False, False, False]}):
        with pytest.raises(ValueError):
            BaseDataset(str(path), saved_split=dict(saved, **change))


def test_training_builder_restores_saved_actual_split_without_source_file(tmp_path):
    from types import SimpleNamespace
    from dexmani_policy.training.build_utils import build_dataset_and_normalizer
    from dexmani_policy.agents.normalization import LinearNormalizer
    path, root, mp, _ = fixture(tmp_path)
    _, _, content, digest = load_split_manifest(mp, dict(root.attrs), 5)
    saved = {'content': content, 'sha256': digest, 'actual_train_ids': ['b'],
             'train_mask': [False, True, False, False, False],
             'val_mask': [False, False, True, True, False]}
    normalizer = LinearNormalizer()
    normalizer.fit_field('action', np.array([[-2.], [7.]], dtype='float32'), mode='limits')
    checkpoint = SimpleNamespace(resume_contract={'data_recipe': [{'split_manifest': saved}]},
        model_state={'normalizer.' + k: v for k, v in normalizer.state_dict().items()})
    cfg = OmegaConf.create({'dataset': {'_target_': 'dexmani_policy.datasets.base_dataset.BaseDataset',
        'zarr_path': str(path), 'split_manifest': str(mp), 'max_train_episodes': 1,
        'val_ratio': .8, 'sensor_modalities': ['joint_state'], 'horizon': 3, 'obs_horizon': 2},
        'resume_from': str(tmp_path / 'checkpoint.pt'), 'action_key': 'action',
        'normalization': {'action': 'limits', 'joint_state': 'identity'}})
    mp.unlink()
    ds, restored = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    assert ds.train_mask.tolist() == saved['train_mask']
    assert cfg.data_recipe[0].split_manifest.actual_train_ids == ['b']
    assert 'saved_split' not in cfg.dataset
    for key, tensor in normalizer.state_dict().items():
        np.testing.assert_array_equal(restored.state_dict()[key], tensor)
