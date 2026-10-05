import numpy as np
import pytest
import zarr

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.training.build_utils import build_normalizer


def dataset(tmp_path, *, real=False, status=True, obs_bad=(), action_bad=(), aux_bad=(), **kwargs):
    root = zarr.open(str(tmp_path / "data.zarr"), "w")
    root.create_dataset("meta/episode_ends", data=np.array([6, 12, 18, 24]))
    if real:
        root.attrs["format"] = "dexmani.real.canonical"
    obs = np.arange(24, dtype=np.float32)[:, None]
    actions = obs.copy()
    aux = np.repeat(obs, 21, axis=1)
    obs[list(obs_bad)] = np.nan
    actions[list(action_bad)] = np.nan
    for row, dim in aux_bad:
        aux[row, dim] = np.nan
    root.create_dataset("data/joint_state", data=obs)
    root.create_dataset("data/action", data=actions)
    root.create_dataset("data/action_ee", data=aux)
    root.create_dataset("data/unused", data=np.full((24, 1), np.nan, np.float32))
    if status:
        dispatch = np.ones((24, 2), dtype=np.uint8)
        if isinstance(status, tuple):
            dispatch[status[0], status[1]] = status[2]
        root.create_dataset("row_info/dispatch_status", data=dispatch)
    return BaseDataset(
        str(tmp_path / "data.zarr"), horizon=4, obs_horizon=2, pad_before=1, pad_after=2, **kwargs
    )


def test_roles_and_padding(tmp_path):
    ds = dataset(tmp_path, obs_bad=[3], action_bad=[5])
    rows = ds.sampler.source_rows()
    assert any(np.array_equal(r, [0, 1, 2, 3]) for r in rows)
    assert not (rows == 5).any()
    for i, r in enumerate(rows):
        sample = ds.sampler.sample_sequence(i)
        for key, value in sample.items():
            np.testing.assert_array_equal(value, ds.replay_buffer[key][r])
        assert len(set(r // 6)) == 1
    assert np.array_equal(rows[0], [0, 0, 1, 2])


def test_aux_only_supervised_dimensions(tmp_path):
    ds = dataset(tmp_path, use_aux_ee=True, aux_bad=[(3, 10)])
    assert len(ds) == 24
    ds = dataset(tmp_path, use_aux_ee=True, aux_bad=[(3, 8)])
    assert not (ds.sampler.source_rows() == 3).any()


@pytest.mark.parametrize("device", [0, 1])
@pytest.mark.parametrize("status", [0, 2, 3, 4])
def test_real_dispatch(tmp_path, device, status):
    ds = dataset(tmp_path, real=True, status=(2, device, status))
    assert not (ds.sampler.source_rows() == 2).any()
    assert "dispatch_status" not in ds[0]["obs"]


def test_missing_dispatch_and_sim(tmp_path):
    with pytest.raises(ValueError, match="dispatch_status"):
        dataset(tmp_path, real=True, status=False)
    assert len(dataset(tmp_path, status=False)) == 24


def test_train_unique_role_statistics(tmp_path):
    ds = dataset(tmp_path, val_ratio=0.25, max_train_episodes=1, obs_bad=[3])
    r = ds.sampler.source_rows()
    obs_rows = np.unique(r[:, :2])
    action_rows = np.unique(r)
    np.testing.assert_array_equal(ds.sampler.observation_source_rows, obs_rows)
    np.testing.assert_array_equal(ds.sampler.action_source_rows, action_rows)
    norm = build_normalizer(ds, {"joint_state": "limits", "action": "limits"}, "action")
    assert set(action_rows // 6).issubset(set(np.flatnonzero(ds.train_mask)))
    values = next(ds.iter_normalization_data("joint_state"))
    np.testing.assert_array_equal(values, ds.replay_buffer["joint_state"][obs_rows])
    np.testing.assert_allclose(norm["joint_state"].input_stats["mean"], values.mean(0))
    val = ds.get_validation_dataset()
    with pytest.raises(ValueError, match="Validation"):
        list(val.iter_normalization_data("action"))


def test_zero_valid(tmp_path):
    with pytest.raises(ValueError, match="Zero valid"):
        dataset(tmp_path, action_bad=list(range(24)))


def test_resume_uses_saved_statistics_without_refitting(tmp_path, monkeypatch):
    import torch
    from omegaconf import OmegaConf

    from dexmani_policy.agents.normalization import LinearNormalizer
    from dexmani_policy.training.build_utils import build_dataset_and_normalizer
    from dexmani_policy.training.checkpoint import CheckpointStore, TrainCheckpoint

    ds = dataset(tmp_path)
    saved = LinearNormalizer()
    saved.fit(
        {
            "action": np.array([[100.0], [300.0]], np.float32),
            "joint_state": np.array([[200.0], [500.0]], np.float32),
        },
        mode="limits",
    )
    checkpoint = TrainCheckpoint(
        0,
        0,
        0,
        {"normalizer." + k: v for k, v in saved.state_dict().items()},
        None,
        {},
        {},
        {},
        None,
        None,
        [{}],
    )
    path = CheckpointStore(tmp_path / "checkpoint").save("latest.pt", checkpoint)
    cfg = OmegaConf.create(
        {
            "dataset": {
                "_target_": "dexmani_policy.datasets.base_dataset.BaseDataset",
                "zarr_path": ds.zarr_path,
                "horizon": 4,
                "obs_horizon": 2,
                "pad_before": 1,
                "pad_after": 2,
            },
            "action_key": "action",
            "normalization": {"action": "limits", "joint_state": "limits"},
            "resume_from": str(path),
        }
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("resume attempted to fit new statistics")

    monkeypatch.setattr("dexmani_policy.training.build_utils.build_normalizer", forbidden)
    _, actual = build_dataset_and_normalizer(cfg)
    for key, value in saved.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
