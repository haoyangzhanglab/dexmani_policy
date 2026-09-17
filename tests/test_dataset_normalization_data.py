import os

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.common.normalizer import LinearNormalizer, build_mixed_action_normalizer
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from dexmani_policy.datasets.pc_dataset import PCDataset
from dexmani_policy.training.build_utils import build_normalizer


def _write_zarr(path, *, n=8, num_points=16, joint_dim=19, seed_offset=0):
    root = zarr.open_group(str(path), mode="w")
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([4, 8], dtype=np.int64))
    data = root.create_group("data")
    data.create_dataset(
        "joint_state",
        data=np.random.default_rng(seed_offset).standard_normal((n, joint_dim)).astype(np.float32),
    )
    data.create_dataset(
        "action",
        data=np.random.default_rng(seed_offset + 1).standard_normal((n, 19)).astype(np.float32),
    )
    data.create_dataset(
        "action_ee",
        data=np.random.default_rng(seed_offset + 2).standard_normal((n, 21)).astype(np.float32),
    )
    data.create_dataset(
        "point_cloud",
        data=np.random.default_rng(seed_offset + 3).standard_normal((n, num_points, 6)).astype(np.float32),
    )
    data.create_dataset(
        "rgb",
        data=np.random.default_rng(seed_offset + 4).integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8),
    )
    return str(path)


def _base(zarr_path, **kwargs):
    defaults = dict(seed=42, horizon=4, pad_before=1, pad_after=1, val_ratio=0.0,
                    sensor_modalities=["joint_state"], action_key="action")
    defaults.update(kwargs)
    return BaseDataset(zarr_path=zarr_path, **defaults)


def test_single_dataset_yields_full_buffer_ignoring_train_mask(tmp_path):
    path = _write_zarr(tmp_path / "toy.zarr")
    ds = _base(path, max_train_episodes=1)  # train_mask limited to 1 episode
    (js,) = list(ds.iter_normalization_data("joint_state"))
    assert js.shape[0] == 8  # full buffer, not limited by train_mask
    np.testing.assert_array_equal(js, ds.replay_buffer["joint_state"])


def test_effective_action_data_semantics(tmp_path):
    path = _write_zarr(tmp_path / "toy.zarr")

    ds_action = _base(path, action_key="action")
    action = ds_action._get_effective_action_data()
    assert action.shape[1] == 19
    np.testing.assert_array_equal(action, ds_action.replay_buffer["action"])

    ds_ee = _base(path, action_key="action_ee")
    action_ee = ds_ee._get_effective_action_data()
    assert action_ee.shape[1] == 21
    np.testing.assert_array_equal(action_ee, ds_ee.replay_buffer["action_ee"])

    ds_aux = _base(path, action_key="action", use_aux_ee=True)
    action_aux = ds_aux._get_effective_action_data()
    assert action_aux.shape[1] == 19 + 9
    np.testing.assert_array_equal(action_aux[:, :19], ds_aux.replay_buffer["action"])
    np.testing.assert_array_equal(action_aux[:, 19:], ds_aux.replay_buffer["action_ee"][:, :9])


def test_multitask_shared_normalization_matches_concatenated_fit(tmp_path):
    path1 = _write_zarr(tmp_path / "a.zarr", seed_offset=0)
    path2 = _write_zarr(tmp_path / "b.zarr", seed_offset=100)

    def pc(zarr_path):
        return PCDataset(zarr_path=zarr_path, seed=42, horizon=4, pad_before=1, pad_after=1,
                         val_ratio=0.0, sensor_modalities=["joint_state", "point_cloud"],
                         action_key="action")

    ds1, ds2 = pc(path1), pc(path2)
    mt = MultiTaskDataset(datasets=[ds1, ds2], task_names=["a", "b"], action_key="action")

    chunks = list(mt.iter_normalization_data("joint_state"))
    assert len(chunks) == 2
    assert chunks[0].shape[0] == 8 and chunks[1].shape[0] == 8

    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    norm = build_normalizer(mt, spec, "action")

    all_js = np.concatenate([ds1.replay_buffer["joint_state"], ds2.replay_buffer["joint_state"]], axis=0)
    ref = LinearNormalizer()
    ref.fit_field("joint_state", all_js, mode="limits")
    torch.testing.assert_close(
        norm.params_dict["joint_state"]["scale"], ref.params_dict["joint_state"]["scale"],
        rtol=1e-4, atol=1e-6,
    )


def test_multitask_shape_mismatch_fails(tmp_path):
    path1 = _write_zarr(tmp_path / "a.zarr", joint_dim=19)
    path2 = _write_zarr(tmp_path / "b.zarr", joint_dim=20)

    def base(zarr_path):
        return BaseDataset(zarr_path=zarr_path, seed=42, horizon=4, pad_before=1, pad_after=1,
                           val_ratio=0.0, sensor_modalities=["joint_state"], action_key="action")

    mt = MultiTaskDataset(datasets=[base(path1), base(path2)], task_names=["a", "b"],
                          action_key="action")
    with pytest.raises(ValueError):
        build_normalizer(mt, {"joint_state": "limits", "action": "auto"}, "action")


def test_real_data_parity_pick_place_toy():
    zarr_path = "robot_data/pick_place_toy.zarr"
    if not os.path.exists(zarr_path):
        pytest.skip("pick_place_toy.zarr not available")

    ds = PCDataset(zarr_path=zarr_path, seed=42, horizon=16, pad_before=1, pad_after=7,
                   val_ratio=0.0, sensor_modalities=["joint_state", "point_cloud"],
                   action_key="action")
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    norm = build_normalizer(ds, spec, "action")

    ref = LinearNormalizer()
    ref.fit(
        {
            "joint_state": ds.replay_buffer["joint_state"],
            "action": ds.replay_buffer["action"],
            "point_cloud": ds.replay_buffer["point_cloud"],
        },
        last_n_dims=1,
        mode="limits",
    )
    for key in ("joint_state", "action", "point_cloud"):
        torch.testing.assert_close(
            norm.params_dict[key]["scale"], ref.params_dict[key]["scale"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            norm.params_dict[key]["offset"], ref.params_dict[key]["offset"], rtol=0, atol=0
        )


def test_build_normalizer_action_ee_uses_mixed_normalizer(tmp_path):
    path = _write_zarr(tmp_path / "ee.zarr")
    ds = PCDataset(zarr_path=path, seed=42, horizon=4, pad_before=1, pad_after=1,
                   val_ratio=0.0, sensor_modalities=["joint_state", "point_cloud"],
                   action_key="action_ee")
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    norm = build_normalizer(ds, spec, "action_ee")

    scale = norm.params_dict["action"]["scale"]
    offset = norm.params_dict["action"]["offset"]
    assert scale.shape == (21,) and offset.shape == (21,)
    # rot6d segment (dims 3:9) must be identity
    assert (scale[3:9] == 1).all()
    assert (offset[3:9] == 0).all()

    ref = build_mixed_action_normalizer(ds.replay_buffer["action_ee"])
    torch.testing.assert_close(scale, ref.params_dict["scale"], rtol=0, atol=0)
    torch.testing.assert_close(offset, ref.params_dict["offset"], rtol=0, atol=0)


def test_build_normalizer_use_aux_ee_uses_plain_limits(tmp_path):
    path = _write_zarr(tmp_path / "aux.zarr")
    ds = PCDataset(zarr_path=path, seed=42, horizon=4, pad_before=1, pad_after=1,
                   val_ratio=0.0, sensor_modalities=["joint_state", "point_cloud"],
                   action_key="action", use_aux_ee=True)
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    norm = build_normalizer(ds, spec, "action")

    scale = norm.params_dict["action"]["scale"]
    offset = norm.params_dict["action"]["offset"]
    assert scale.shape == (28,) and offset.shape == (28,)

    # effective action == training target == action (19) + action_ee[..., :9] (9)
    eff = np.concatenate(
        [ds.replay_buffer["action"], ds.replay_buffer["action_ee"][:, :9]], axis=1
    )
    ref = LinearNormalizer()
    ref.fit_field("action", eff, mode="limits")
    torch.testing.assert_close(scale, ref.params_dict["action"]["scale"], rtol=0, atol=0)
    torch.testing.assert_close(offset, ref.params_dict["action"]["offset"], rtol=0, atol=0)
