"""Native Zarr/torch equivalence and bounded-consumption regressions."""

import gc
import multiprocessing as mp
import os
import pickle
import weakref

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from dexmani_policy.agents.normalization import (
    LinearNormalizer,
    build_mixed_action_normalizer_chunks,
)
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.training.build_utils import build_normalizer


def make_data(tmp_path, *, action_key="action", use_aux_ee=False):
    path = str(tmp_path / "data.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_dataset("meta/episode_ends", data=np.array([6, 12, 18, 24]))
    values = np.arange(24 * 21, dtype=np.float32).reshape(24, 21) / 100
    root.create_dataset("data/action", data=values[:, :19], chunks=(2, 19))
    root.create_dataset("data/action_ee", data=values, chunks=(2, 21))
    root.create_dataset("data/joint_state", data=values[:, :19], chunks=(2, 19))
    root.create_dataset(
        "data/point_cloud",
        data=np.repeat(values[:, None, :6], 8, axis=1),
        chunks=(2, 8, 6),
    )
    root.create_dataset(
        "data/rgb",
        data=np.arange(24 * 4 * 4 * 3, dtype=np.uint8).reshape(24, 4, 4, 3),
        chunks=(2, 4, 4, 3),
    )
    return BaseDataset(
        path,
        sensor_modalities=["joint_state", "point_cloud", "rgb"],
        horizon=3,
        obs_horizon=1,
        pad_before=1,
        pad_after=1,
        val_ratio=0.25,
        action_key=action_key,
        use_aux_ee=use_aux_ee,
    )


@pytest.mark.parametrize("mode", ["limits", "gaussian"])
@pytest.mark.parametrize("dims", [0, 1, 2])
def test_streaming_matches_one_shot(mode, dims):
    torch.set_num_threads(1)
    data = np.random.default_rng(2).normal(size=(53, 3, 4)).astype(np.float32)
    data[..., 0] = 0.3
    expected = LinearNormalizer()
    expected.fit_field("x", data, last_n_dims=dims, mode=mode)
    for size in [1, 7, 53]:
        actual = LinearNormalizer()
        actual.fit_field_chunks(
            "x",
            (data[i : i + size] for i in range(0, len(data), size)),
            last_n_dims=dims,
            mode=mode,
        )
        for key in ["scale", "offset"]:
            torch.testing.assert_close(
                actual["x"].params_dict[key],
                expected["x"].params_dict[key],
                atol=2e-6,
                rtol=2e-5,
            )
        for key in ["min", "max", "mean", "std"]:
            torch.testing.assert_close(
                actual["x"].input_stats[key],
                expected["x"].input_stats[key],
                atol=2e-6,
                rtol=2e-5,
            )


def test_empty_chunks_and_gaussian_count():
    n = LinearNormalizer()
    for chunks in [iter(()), iter((np.empty((0, 3)),))]:
        with pytest.raises(ValueError):
            n.fit_field_chunks("x", chunks)
    with pytest.raises(ValueError):
        n.fit_field_chunks("x", iter((np.ones((1, 3)),)), mode="gaussian")
    n.fit_field_chunks("x", iter((np.empty((0, 3)), np.ones((2, 3)))), mode="limits")
    assert torch.isfinite(n["x"].params_dict["scale"]).all()


def test_consumer_does_not_retain_chunks():
    refs = []

    class Once:
        consumed = False

        def __iter__(self):
            assert not self.consumed
            self.consumed = True
            for i in range(12):
                gc.collect()
                assert sum(ref() is not None for ref in refs) <= 1
                value = np.full((64, 21), i, dtype=np.float32)
                refs.append(weakref.ref(value))
                yield value
                del value

    class Dataset:
        def iter_normalization_data(self, key):
            return iter(Once())

    build_normalizer(Dataset(), {"action": "auto"}, "action_ee")
    gc.collect()
    assert not any(ref() is not None for ref in refs)


def test_mixed_statistics_and_aux(tmp_path):
    ds = make_data(tmp_path, action_key="action_ee", use_aux_ee=True)
    chunks = list(ds.iter_normalization_data("action"))
    assert max(len(chunk) for chunk in chunks) <= 2
    data = np.concatenate(chunks)
    reference = LinearNormalizer()
    reference.fit_field("x", data, mode="limits")
    actual = build_mixed_action_normalizer_chunks(iter(chunks))
    for key in ["scale", "offset"]:
        idx = [0, 1, 2] + list(range(9, data.shape[1]))
        torch.testing.assert_close(
            actual.params_dict[key][idx], reference["x"].params_dict[key][idx]
        )
    torch.testing.assert_close(actual.params_dict["scale"][3:9], torch.ones(6))
    torch.testing.assert_close(actual.input_stats["std"][3:9], torch.ones(6))
    assert data.shape[1] == 30


def test_role_reads_are_short_and_owned(tmp_path, monkeypatch):
    ds = make_data(tmp_path)
    calls = []
    read = ds.replay_buffer.read

    def record(key, rows, **kwargs):
        calls.append((key, rows.stop - rows.start))
        return read(key, rows, **kwargs)

    monkeypatch.setattr(ds.replay_buffer, "read", record)
    sample = ds[2]
    assert dict(calls) == {"joint_state": 1, "point_cloud": 1, "rgb": 1, "action": 3}
    assert sample["obs"]["rgb"].dtype == torch.uint8
    for index in [0, len(ds) - 1]:
        rows = ds.sampler.source_rows(ds.sampler.indices[index : index + 1])[0]
        result = ds[index]
        np.testing.assert_array_equal(
            result["action"], ds.replay_buffer["action"][rows]
        )
        np.testing.assert_array_equal(
            result["obs"]["joint_state"], ds.replay_buffer["joint_state"][rows[:1]]
        )
    sample["obs"]["joint_state"].fill_(99)
    assert not torch.equal(sample["obs"]["joint_state"], ds[2]["obs"]["joint_state"])


def test_open_does_not_materialize_payload(tmp_path, monkeypatch):
    ds = make_data(tmp_path)
    original = zarr.Array.__getitem__

    def read(arr, key):
        assert not arr.path.startswith("data/")
        return original(arr, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", read)
    buffer = ReplayBuffer.open(ds.zarr_path, keys=["action"])
    assert buffer.keys() == ("action",)
    assert buffer._group is None and buffer._data is None


def test_conversion_overflow_and_unique_roles(tmp_path):
    root = zarr.open_group(str(tmp_path / "overflow.zarr"), mode="w")
    root.create_dataset("meta/episode_ends", data=np.array([5]))
    values = np.zeros((5, 1), np.float64)
    values[3] = 1e300
    root.create_dataset("data/joint_state", data=values, chunks=(2, 1))
    root.create_dataset(
        "data/action", data=np.arange(5, dtype=np.float32)[:, None], chunks=(2, 1)
    )
    with np.errstate(over="ignore"):
        ds = BaseDataset(str(tmp_path / "overflow.zarr"), horizon=3, obs_horizon=1)
    assert len(ds) == 3
    np.testing.assert_array_equal(ds.sampler.observation_source_rows, [0, 1, 2])
    np.testing.assert_array_equal(ds.sampler.action_source_rows, [0, 1, 2, 3, 4])


@pytest.mark.parametrize(
    "context", ["spawn"] + (["fork"] if "fork" in mp.get_all_start_methods() else [])
)
def test_process_handles_and_persistent_workers(tmp_path, context):
    ds = make_data(tmp_path)
    ds[0]  # populate parent handle before fork/pickle
    assert ds.replay_buffer._pid == os.getpid()
    restored = pickle.loads(pickle.dumps(ds))
    assert restored.replay_buffer._group is None
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        multiprocessing_context=context,
        persistent_workers=True,
    )
    try:
        for _ in range(2):
            for index, sample in enumerate(loader):
                torch.testing.assert_close(sample["action"], ds[index]["action"])
        assert ds.get_validation_dataset() is not None
        val = ds.get_validation_dataset()
        with pytest.raises(ValueError):
            next(val.iter_normalization_data("action"))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_deterministic_never_creates_manager(tmp_path, monkeypatch):
    ds = make_data(tmp_path)

    def forbidden():
        raise AssertionError("deterministic Manager")

    monkeypatch.setattr(mp, "Manager", forbidden)
    multi = MultiTaskDataset([ds], ["one"], deterministic=True)
    before = [multi[i]["action"] for i in range(len(multi))]
    multi.set_epoch(9)
    restored = pickle.loads(pickle.dumps(multi))
    for i, value in enumerate(before):
        torch.testing.assert_close(restored[i]["action"], value)
    norm = build_normalizer(multi, {"action": "limits"}, "action")
    reference = build_normalizer(ds, {"action": "limits"}, "action")
    torch.testing.assert_close(
        norm["action"].params_dict["scale"], reference["action"].params_dict["scale"]
    )
    multi.close()
