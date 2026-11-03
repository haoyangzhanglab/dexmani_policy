"""Lightweight CPU source-frame dataset and normalization checks; no policy construction."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.io_util import atomic_path, write_json
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler, window_indices
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.datasets.variants.tactile import AdapTacDataset, InterActDexDataset, read_future_force
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import fit_normalizer


def check_splits():
    for n, maximum, val, expected in (
        (150, 100, 20, (100, 20, 30)), (150, 100, 0, (100, 0, 50)),
        (150, None, 20, (130, 20, 0)), (150, None, 0, (150, 0, 0)),
        (80, 100, 0, (80, 0, 0)), (1, None, 0, (1, 0, 0)),
        (80, 100, 10, (70, 10, 0)), (110, 100, 20, (90, 20, 0)),
        (100, 100, 20, (80, 20, 0)), (100, None, 10, (90, 10, 0)),
    ):
        first = split_episodes(n, seed=42, max_train_episodes=maximum, val_episodes=val)
        second = split_episodes(n, seed=42, max_train_episodes=maximum, val_episodes=val)
        train, validation = first
        assert train.dtype == validation.dtype == bool
        assert not (train & validation).any()
        assert (train.sum(), validation.sum(), (~(train | validation)).sum()) == expected
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        order = np.random.default_rng(42).permutation(n)
        count = expected[0]
        np.testing.assert_array_equal(np.flatnonzero(train), np.sort(order[:count]))
        np.testing.assert_array_equal(np.flatnonzero(validation), np.sort(order[count:count + val]))
    invalid_cases: tuple[tuple[Any, Any, Any], ...] = (
        (1, None, 1), (0, None, 0), (10, None, 11),
        (10, 0, 0), (10, -1, 0), (10, None, -1), (True, None, 0),
        (10, True, 0), (10, 1.5, 0), (10, None, True), (10, None, 1.5),
    )
    for n, maximum, val in invalid_cases:
        try:
            split_episodes(n, max_train_episodes=maximum, val_episodes=val)
        except ValueError:
            pass
        else:
            raise AssertionError((n, maximum, val))
    np.random.seed(17)
    expected_random = np.random.random(4)
    np.random.seed(17)
    split_episodes(150, max_train_episodes=100, val_episodes=20)
    np.testing.assert_array_equal(np.random.random(4), expected_random)


def make_source(path, ends):
    source = zarr.open_group(str(path), mode="w")
    source.create_dataset("meta/episode_ends", data=np.asarray(ends))
    rows = int(ends[-1])
    rng = np.random.default_rng(11)
    for key, shape in (
        ("action", (rows, 19)), ("action_ee", (rows, 21)), ("joint_state", (rows, 19)),
        ("point_cloud", (rows, 8, 6)), ("hand_pointcloud", (rows, 8, 6)),
        ("object_pointcloud", (rows, 16, 6)), ("finger_tactile_force", (rows, 5, 3)),
    ):
        source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype(np.float32))
    return source


def check_source_statistics(root):
    path = root / "statistics.zarr"
    ends = np.arange(1, 151) * 3
    source = make_source(path, ends)
    train, val = split_episodes(150, max_train_episodes=100, val_episodes=20)
    selected_rows = np.repeat(train, 3)
    data = source["data/action"][:]
    data[np.repeat(val, 3)] = 10000
    data[np.repeat(~(train | val), 3)] = -10000
    source["data/action"][:] = data
    # Identity labels are allowed in raw data but play no role in splitting.
    source.attrs.update(source_id="", episode_ids=["duplicate"], trial_ids={"bad": "group"})
    spec = {"action": "limits", "point_cloud": "limits"}
    dims = {"action": 19, "point_cloud": 3}
    states = []
    for horizon, before, after in ((1, 0, 0), (5, 2, 3)):
        dataset = DemonstrationDataset(
            str(path), sensor_modalities=["point_cloud"], obs_horizon=1, horizon=horizon,
            pad_before=before, pad_after=after, max_train_episodes=100, val_episodes=20, pc_dim=3,
            augmentation_cfg={"pc": {"coord_noise": {"noise_std": 100., "prob": 1.}}},
        )
        before_sample = source["data/point_cloud"][:]
        dataset[0]
        np.testing.assert_array_equal(source["data/point_cloud"][:], before_sample)
        normalizer = fit_normalizer(dataset, spec, feature_dims=dims)
        states.append(normalizer.state_dict())
        actual = normalizer["action"].normalize(data[selected_rows])
        torch.testing.assert_close(actual.amin(0), -torch.ones(19))
        torch.testing.assert_close(actual.amax(0), torch.ones(19))
        assert normalizer["point_cloud"].scale.shape == (3,)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    masks = []
    for seed in (7, 99):
        cfg = load_policy_config("dp3", [f"training.seed={seed}", f"zarr_path={path}",
                                         "horizon=4", "n_action_steps=2",
                                         "dataset.val_episodes=20", "dataset.augmentation_cfg=null"])
        masks.append(hydra.utils.instantiate(cfg.dataset, _convert_="all").train_mask)
    np.testing.assert_array_equal(*masks)


def check_affine_math():
    values = np.random.default_rng(7).normal(size=(7, 5, 3)).astype(np.float32)
    for mode in ("limits", "gaussian", "rms", "max_abs"):
        normalizer = LinearNormalizer()
        normalizer.fit_field("x", [values[:2], values[2:]], mode=mode, last_n_dims=2)
        field = normalizer["x"]
        reference = torch.from_numpy(values).double()
        if mode == "limits":
            scale = 2 / (reference.amax(0) - reference.amin(0))
            offset = -1 - scale * reference.amin(0)
        elif mode == "gaussian":
            scale = reference.std(0).reciprocal()
            offset = -reference.mean(0) * scale
        else:
            magnitude = reference.square().mean().sqrt() if mode == "rms" else reference.abs().max()
            scale = magnitude.reciprocal().expand(5, 3)
            offset = torch.zeros_like(scale)
        torch.testing.assert_close(field.scale, scale.float())
        torch.testing.assert_close(field.offset, offset.float())
        torch.testing.assert_close(field.unnormalize(field.normalize(values)), torch.from_numpy(values))
        for constant in (0., 3.):
            normalizer.fit_field("x", [np.full((4, 5, 3), constant, dtype=np.float32)], mode=mode, last_n_dims=2)
            field = normalizer["x"]
            assert field.scale.shape == field.offset.shape == (5, 3)
            assert torch.isfinite(field.scale).all() and (field.scale != 0).all()
            x = torch.full((4, 5, 3), constant)
            torch.testing.assert_close(field.unnormalize(field.normalize(x)), x)

    class IdentitySource:
        def iter_normalization_data(self, key):
            raise AssertionError("Identity scanned data")

    identity = fit_normalizer(IdentitySource(), {"action": "identity", "joint_state": "identity"},
                              feature_dims={"action": 19, "joint_state": 19})
    assert set(identity.fields) == {"action"}
    imported = {"fields.action.scale": torch.full((19,), 2.), "fields.action.offset": torch.ones(19)}
    inherited = fit_normalizer(IdentitySource(), {"action": "identity"}, feature_dims={"action": 19},
                               initial_state=imported)
    torch.testing.assert_close(inherited.state_dict(), imported, rtol=0, atol=0)

    class TaxelSource:
        def iter_normalization_data(self, key):
            yield np.arange(2 * 5 * 120 * 3, dtype=np.float32).reshape(2, 5, 120, 3)

    taxel = fit_normalizer(TaxelSource(), {"taxel_tactile_force": "rms"},
                           feature_dims={"taxel_tactile_force": 1800})["taxel_tactile_force"]
    assert taxel.scale.shape == (5, 120, 3) and taxel.scale.unique().numel() == 1
    assert (taxel.offset == 0).all()
    state = {"fields.joint_state.scale": torch.ones(19)}
    try:
        fit_normalizer(IdentitySource(), {"joint_state": "limits"}, feature_dims={"joint_state": 19},
                       initial_state=state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Partial imported affine accepted")


def check_method_targets(root):
    path = root / "targets.zarr"
    source = make_source(path, [2, 7, 13])
    force_data = np.broadcast_to(np.arange(13, dtype=np.float32)[:, None, None], (13, 5, 3))
    source["data/finger_tactile_force"][:] = force_data
    kwargs = dict(sensor_modalities=["joint_state", "finger_tactile_force"],
                  horizon=5, obs_horizon=2, pad_before=1, pad_after=3)
    for cls, extra in ((AdapTacDataset, {}), (InterActDexDataset, {"interaction_horizon": 3})):
        dataset = cls(str(path), **kwargs, **extra)
        episode_ends = dataset.episode_store.episode_ends
        for index in range(len(dataset)):
            rows = dataset.sampler.sample_source_rows(index)
            query = rows[dataset.action_start_index]
            end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
            sample_offsets = np.arange(1, 5) if cls is AdapTacDataset else np.array([3])
            target = query + sample_offsets
            sample = dataset[index]
            np.testing.assert_array_equal(sample["future_force_valid"], target < end)
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], np.minimum(target, end - 1))
            if cls is AdapTacDataset:
                assert query + 1 < end
        if cls is InterActDexDataset:
            assert any(not dataset[index]["future_force_valid"].any() for index in range(len(dataset)))
    force_store = EpisodeStore(path, keys=["finger_tactile_force"])
    for query, offsets in (
        (2, range(1, 4)), (2, (1,)), (6, (3,)),
        (5, range(1, 5)), (6, range(1, 5)),
        (2, [1, 3]), (2, [1, 1, 3]), (2, [1, 2, 2, 4]),
        (2, [1, 3, 2, 4]), (2, range(1, 5, 2)),
        (2, [4, 1]), (2, [1, 4, 2]), (2, [3, 1, 3]),
        (3, [3, 1, 5, 2]), (6, [3, 1]),
    ):
        target = query + np.asarray(offsets, dtype=np.int64)
        end = force_store.episode_ends[np.searchsorted(force_store.episode_ends, query, side="right")]
        expected_valid = target < end
        expected_rows = target.copy()
        expected_rows[~expected_valid] = end - 1
        expected = force_data[expected_rows]
        with patch.object(force_store, "read", wraps=force_store.read) as read:
            force, valid = read_future_force(force_store, query, offsets)
        read.assert_called_once_with(
            "finger_tactile_force", slice(int(expected_rows.min()), int(expected_rows.max()) + 1),
        )
        np.testing.assert_array_equal(force, expected)
        np.testing.assert_array_equal(valid, expected_valid)
        assert force.shape == (len(offsets), 5, 3) and valid.shape == (len(offsets),)
        assert force.dtype == np.float32 and valid.dtype == bool
    source["data/finger_tactile_force"][:] = 0
    for dataset in (AdapTacDataset(str(path), **kwargs),
                    InterActDexDataset(str(path), **kwargs, interaction_horizon=1)):
        zero_force = dataset[0]
        assert zero_force["future_force_valid"].any() and not zero_force["future_force"].any()
    r3d_dataset = R3DDataset(str(path), sensor_modalities=["joint_state"], obs_horizon=2,
                             horizon=5, pad_before=1, pad_after=3)
    for index in range(len(r3d_dataset)):
        rows = r3d_dataset.sampler.sample_source_rows(index)
        expected = np.concatenate((source["data/action"][:][rows], source["data/action_ee"][:][rows, :9]), -1)
        np.testing.assert_array_equal(r3d_dataset[index]["action"], expected)
    actual = np.concatenate(list(r3d_dataset.iter_normalization_data("action")))
    np.testing.assert_array_equal(actual, np.concatenate((source["data/action"][:], source["data/action_ee"][:, :9]), -1))


def check_hand_affine():
    values = np.random.default_rng(31).normal(size=(13, 19)).astype(np.float32)
    values[:, 7:] *= 1000
    hand = {"scale": torch.linspace(1, 2, 12), "offset": torch.linspace(-0.5, 0.5, 12)}

    class Source:
        action_key = "action"

        def iter_normalization_data(self, key):
            assert key == "action" and mode != "identity", "Identity scanned action statistics"
            yield values[:5]
            yield values[5:]

    for mode in ("limits", "gaussian", "identity", "rms", "max_abs"):
        reference = fit_normalizer(Source(), {"action": mode}, feature_dims={"action": 19})
        reference["action"].scale[-12:].copy_(hand["scale"])
        reference["action"].offset[-12:].copy_(hand["offset"])
        actual = fit_normalizer(Source(), {"action": mode}, feature_dims={"action": 19}, hand_affine=hand)
        torch.testing.assert_close(actual.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(actual["action"].unnormalize(actual["action"].normalize(values)),
                                   torch.from_numpy(values), rtol=2e-5, atol=2e-5)
        if mode in ("rms", "max_abs"):
            arm_only = LinearNormalizer()
            arm_only.fit_field("action", [values[:, :7]], mode=mode)
            assert not torch.allclose(actual["action"].scale[:7], arm_only["action"].scale)


def check_dataset_reads(root):
    path = root / "selected_reads.zarr"
    source = make_source(path, [3, 8])
    source.create_dataset("data/unused", data=np.array(["not a selected numeric field"]))
    dataset = DemonstrationDataset(
        path, sensor_modalities=["joint_state", "point_cloud"], obs_horizon=2,
        horizon=4, pad_before=1, pad_after=2, pc_dim=3,
    )
    assert set(dataset.episode_store.keys()) == {"joint_state", "point_cloud", "action"}
    rows = dataset.sampler.sample_source_rows(1)
    with patch.object(dataset.episode_store, "read", wraps=dataset.episode_store.read) as read:
        sample = dataset[1]
    assert {call.args[0] for call in read.call_args_list} == set(dataset.data_keys)
    for call in read.call_args_list:
        key, selection = call.args
        assert selection.stop - selection.start <= (4 if key == "action" else 2)
        if key == "point_cloud":
            assert call.kwargs["columns"] == slice(0, 3)
    np.testing.assert_array_equal(sample["action"], source["data/action"][:][rows])
    np.testing.assert_array_equal(sample["obs"]["joint_state"], source["data/joint_state"][:][rows[:2]])
    np.testing.assert_array_equal(sample["obs"]["point_cloud"], source["data/point_cloud"][:][rows[:2], :, :3])


def check_rgb_preprocessing(root):
    import torchvision.transforms.functional as TVF
    from dexmani_policy.common.image_util import preprocess_rgb_eval, preprocess_rgb_train

    rgb = np.random.default_rng(17).integers(256, size=(1, 7, 9, 3), dtype=np.uint8).repeat(3, axis=0)
    path = root / "rgb.zarr"
    source = make_source(path, [3])
    source.create_dataset("data/rgb", data=rgb)
    raw = DemonstrationDataset(path, sensor_modalities=["rgb"], obs_horizon=2, horizon=2)[0]["obs"]["rgb"]
    assert raw.dtype == torch.uint8 and raw.is_contiguous()
    np.testing.assert_array_equal(raw, rgb[:2].transpose(0, 3, 1, 2))
    for color in (None, lambda value: value):
        dataset = DemonstrationDataset(
            path, sensor_modalities=["rgb"], obs_horizon=2, horizon=2,
            rgb_preprocess_size=(10, 12), rgb_color_aug=color,
        )
        actual = dataset[0]["obs"]["rgb"]
        expected = preprocess_rgb_eval(rgb[:2], resize_hw=(10, 12), center_crop_hw=None,
                                       float_spatial_before_uint8=color is not None)
        assert actual.dtype == torch.uint8 and actual.is_contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cropped = preprocess_rgb_train(rgb[:2], resize_hw=(10, 12), random_crop_hw=(6, 8))
    assert cropped.shape == (2, 3, 6, 8) and cropped.is_contiguous()
    torch.testing.assert_close(cropped[0], cropped[1], rtol=0, atol=0)
    color = lambda value: value.mul(0.73).add(0.011)
    actual = preprocess_rgb_train(rgb[:2], resize_hw=(10, 12), random_crop_hw=None, color_augmentation=color)
    floating = torch.from_numpy(rgb[:2]).permute(0, 3, 1, 2).float().div(255)
    expected = color(TVF.resize(floating, [10, 12], antialias=True)).clamp(0, 1).mul(255).round().to(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def check_sampler():
    ends = np.array([1, 4, 9], dtype=np.int64)
    mask = np.array([True, False, True])
    expected = np.array([
        [0, 1, 3], [0, 1, 2], [0, 1, 1], [0, 1, 0],
        [4, 5, 3], [4, 6, 2], [4, 7, 1], [4, 8, 0],
        [5, 9, 0], [6, 9, 0], [7, 9, 0], [8, 9, 0],
    ], dtype=np.int64)
    actual = window_indices(ends, 4, mask, 3, 3)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(window_indices(ends[:1], 1, mask[:1]), [[0, 1, 0]])
    np.testing.assert_array_equal(
        window_indices(ends, 4, np.ones(3, dtype=bool)), [[4, 8, 0], [5, 9, 0]],
    )
    for episode_ends, horizon, selected in (
        (ends, 4, np.zeros(3, dtype=bool)),
        (ends, 32, mask),
        (ends[:0], 1, mask[:0]),
    ):
        empty = window_indices(episode_ends, horizon, selected)
        assert empty.shape == (0, 3) and empty.dtype == np.int64

    rows = np.array([
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        [4, 4, 4, 4], [4, 4, 4, 5], [4, 4, 5, 6], [4, 5, 6, 7],
        [5, 6, 7, 8], [6, 7, 8, 8], [7, 8, 8, 8], [8, 8, 8, 8],
    ])
    values = np.arange(9 * 4, dtype=np.float32).reshape(9, 4)
    with tempfile.TemporaryDirectory(prefix="dexmani-sampler-check-") as directory:
        group = zarr.open_group(directory, mode="w")
        group.create_dataset("meta/episode_ends", data=ends)
        for key in ("action", "state"):
            group.create_dataset(f"data/{key}", data=values, chunks=(3, 4))
        sampler = SequenceSampler(
            EpisodeStore(directory), 4, 3, 3, episode_mask=mask,
            fields={"action": slice(1, 4, 2), "state": None}, field_horizons={"state": 2},
        )
        assert len(sampler) == len(rows)
        for index, expected_rows in enumerate(rows):
            np.testing.assert_array_equal(sampler.sample_source_rows(index), expected_rows)
            sample = sampler.sample_sequence(index)
            np.testing.assert_array_equal(sample["action"], values[expected_rows, 1:4:2])
            np.testing.assert_array_equal(sample["state"], values[expected_rows[:2]])


def check_atomic_io(root):
    directory = root / "atomic_io"
    directory.mkdir()
    target = directory / "state.pt"
    target.write_bytes(b"complete")
    try:
        with atomic_path(target) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted write")
    except RuntimeError:
        pass
    assert target.read_bytes() == b"complete"
    assert not list(directory.glob(".dexmani-publish-*"))
    with atomic_path(target) as temporary:
        temporary.write_bytes(b"next complete")
    assert target.read_bytes() == b"next complete"

    write_json(directory / "result.json", {"score": 0.5})
    assert json.loads((directory / "result.json").read_text()) == {"score": 0.5}
    values = np.arange(5, dtype=np.float32)
    with atomic_path(directory / "codebook.npz") as temporary:
        with temporary.open("wb") as stream:
            np.savez(stream, values=values)
    with np.load(directory / "codebook.npz") as payload:
        np.testing.assert_array_equal(payload["values"], values)
    latest = directory / "latest.pt"
    with atomic_path(latest) as temporary:
        temporary.unlink()
        temporary.symlink_to(target.name)
    assert latest.is_symlink() and latest.readlink() == Path(target.name)
    assert latest.read_bytes() == b"next complete"
    assert not list(directory.glob(".dexmani-publish-*"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(2)
    check_splits()
    check_affine_math()
    check_hand_affine()
    check_sampler()
    print("PASS split counts, RNG isolation, affine math, hand affine and sampler", flush=True)
    with tempfile.TemporaryDirectory(prefix="dexmani-dataset-checks-") as directory:
        root = Path(directory)
        check_source_statistics(root)
        check_method_targets(root)
        check_dataset_reads(root)
        check_rgb_preprocessing(root)
        check_atomic_io(root)
        print("PASS train-only source statistics, method targets, selected reads, CHW uint8 RGB and atomic IO", flush=True)


if __name__ == "__main__":
    main()
