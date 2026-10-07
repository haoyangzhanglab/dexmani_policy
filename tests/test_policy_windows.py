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
        root.attrs["dt"] = 0.1
        root.create_dataset("row_info/observation_timestamp_ns", data=(np.arange(24, dtype="i8") + 1) * 100_000_000)
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
        {"data_recipe": [ds.data_recipe]},
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


@pytest.mark.parametrize("stamps", [
    [100, 200, 300, 400, 100, 200, 300, 400],
    [100, 200, 800, 900, 100, 200, 300, 400],
    [100, 200, 150, 250, 100, 200, 300, 400],
    [100, 0, 300, 400, 100, 200, 300, 400],
])
@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("dtype", ["i8", "u8"])
def test_time_rule_real_source_rows_and_padding(tmp_path, stamps, horizon, dtype):
    path = tmp_path / "time.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(format="dexmani.real.canonical", dt=1e-7)
    root.create_dataset("meta/episode_ends", data=np.array([4, 8]))
    for key in ("action", "joint_state"):
        root.create_dataset("data/" + key, data=np.arange(8, dtype="f4")[:, None])
    root.create_dataset("row_info/dispatch_status", data=np.ones((8, 2), "u1"))
    root.create_dataset("row_info/observation_timestamp_ns", data=np.array(stamps, dtype=dtype))
    kwargs = {'horizon':horizon, 'obs_horizon':1, 'pad_before':horizon-1, 'pad_after':horizon-1}
    unfiltered = BaseDataset(str(path), max_time_gap_ratio=None, **kwargs)
    actual = BaseDataset(str(path), **kwargs)
    expected = []
    for row in unfiltered.sampler.source_rows():
        unique = list(dict.fromkeys(row.tolist()))
        if all(stamps[i] > 0 for i in unique) and all(0 < int(stamps[unique[k]])-int(stamps[unique[k-1]]) <= 150 for k in range(1,len(unique))):
            expected.append(row)
    np.testing.assert_array_equal(actual.sampler.source_rows(), expected)
    assert all(len(set(row // 4)) == 1 for row in expected)
    np.testing.assert_array_equal(actual.sampler.action_source_rows, np.unique(expected))
    np.testing.assert_array_equal(actual.sampler.observation_source_rows, np.unique(np.array(expected)[:, :1]))
    assert actual.sampler.validity_summary['time'] == len(unfiltered)-len(actual)
    assert actual.data_recipe['time_filter']['max_time_gap_ratio'] == 1.5
    assert unfiltered.data_recipe['time_filter']['status'] == 'unfiltered'


@pytest.mark.parametrize('ratio', [True, False, 0, -1, float('nan'), float('inf')])
def test_time_ratio_rejected(tmp_path, ratio):
    with pytest.raises(ValueError, match='max_time_gap_ratio'):
        dataset(tmp_path, max_time_gap_ratio=ratio)


def test_time_metadata_is_required_only_for_enabled_real_rule(tmp_path):
    ds = dataset(tmp_path, real=True)
    root = zarr.open_group(ds.zarr_path, mode='a')
    del root['row_info/observation_timestamp_ns']
    with pytest.raises(ValueError, match='observation_timestamp_ns'):
        BaseDataset(ds.zarr_path)
    assert len(BaseDataset(ds.zarr_path, max_time_gap_ratio=None)) == 24
    root.attrs['format'] = 'simulation'
    sim = BaseDataset(ds.zarr_path)
    assert 'time_filter' not in sim.data_recipe and 'time' not in sim.sampler.validity_summary
    root.attrs['format'] = 'dexmani.real.canonical'
    root.create_dataset('row_info/observation_timestamp_ns', data=np.zeros((24,1),dtype='f4'))
    with pytest.raises(ValueError, match='1D integer'):
        BaseDataset(ds.zarr_path)


@pytest.mark.parametrize('rule', ['legacy', 'filtered', 'unfiltered'])
def test_resume_time_rule_precedes_dataset_and_preserves_contract(tmp_path, monkeypatch, rule):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from dexmani_policy.training.build_utils import build_dataset_and_normalizer
    from dexmani_policy.training.resume import validate_resume_contract
    ds = dataset(tmp_path, real=True)
    root = zarr.open_group(ds.zarr_path, mode='a')
    root.attrs.update(task_name='test', dt=.1)
    stamps = root['row_info/observation_timestamp_ns'][:]
    stamps[3:] += 2_000_000_000
    root['row_info/observation_timestamp_ns'][:] = stamps
    cfg = OmegaConf.create({'task_name':'test', 'action_key':'action',
        'normalization':{'action':'limits','joint_state':'limits'},
        'dataset':{'_target_':'dexmani_policy.datasets.base_dataset.BaseDataset',
                   'zarr_path':ds.zarr_path,'horizon':4,'obs_horizon':2,'pad_before':1,'pad_after':2,
                   'max_time_gap_ratio':1.5 if rule=='filtered' else None}})
    initial, normalizer = build_dataset_and_normalizer(cfg)
    saved = OmegaConf.to_container(cfg.data_recipe)
    if rule == 'legacy':
        del saved[0]['time_filter']
    checkpoint = SimpleNamespace(resume_contract={'data_recipe': saved},
        model_state={'normalizer.'+k:v for k,v in normalizer.state_dict().items()})
    cfg.resume_from = str(tmp_path/'old.pt')
    cfg.dataset.max_time_gap_ratio = .1  # A current default must not alter a saved rule.
    monkeypatch.setattr('dexmani_policy.training.build_utils.build_normalizer',
                        lambda *a: pytest.fail('refit historical statistics'))
    restored, norm = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    np.testing.assert_array_equal(restored.sampler.source_rows(), initial.sampler.source_rows())
    np.testing.assert_array_equal(restored.sampler.action_source_rows, initial.sampler.action_source_rows)
    assert OmegaConf.to_container(cfg.data_recipe) == saved
    validate_resume_contract({'facts_format':1,'data_recipe':saved},
                             {'facts_format':1,'data_recipe':OmegaConf.to_container(cfg.data_recipe)})
    for key, value in normalizer.state_dict().items():
        np.testing.assert_array_equal(norm.state_dict()[key], value)
    checkpoint.resume_contract = {}
    with pytest.raises(ValueError, match='saved data_recipe'):
        build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
