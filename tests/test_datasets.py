import copy
import json
import pickle
import shutil

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.training.common.data import build_normalizer, close_data_loader
from dexmani_policy.training.pretrain.hand_prior.hand_vae import fit_relative_statistics


def test_padding_and_training_statistics(demonstrations):
    dataset = RobotDataset(
        str(demonstrations), sensor_modalities=["joint_state"], horizon=3,
        obs_horizon=2, pad_before=1, pad_after=1, max_train_episodes=1,
    )
    episode = np.flatnonzero(dataset.train_mask)[0]
    start, end = episode * 12, (episode + 1) * 12
    np.testing.assert_array_equal(dataset.sampler.sample_source_rows(0), [start, start, start + 1])
    np.testing.assert_array_equal(dataset.sampler.sample_source_rows(-1), [end - 2, end - 1, end - 1])
    rows = np.concatenate(list(dataset.iter_normalization_data("action")))
    expected = dataset.replay_buffer["action"][start:end]
    np.testing.assert_array_equal(rows, expected)
    normalizer = build_normalizer(dataset, {"action": "limits"}, "action")
    np.testing.assert_allclose(normalizer["action"].input_stats["min"], expected.min(0))
    np.testing.assert_allclose(normalizer["action"].input_stats["max"], expected.max(0))


def test_explicit_split_and_trial_isolation(demonstrations, tmp_path):
    root = zarr.open_group(str(demonstrations), mode="a")
    root.attrs.update(
        source_id="demo-corpus",
        episode_ids=["a", "b", "c", "d"],
        trial_ids={"a": "one", "b": "one", "c": "two", "d": "two"},
    )
    alias = tmp_path / "alias.zarr"
    alias.symlink_to(demonstrations, target_is_directory=True)
    buffer = ReplayBuffer(str(alias), keys=[])
    train, val, manifest = split_episodes(buffer, val_ratio=0.5)
    assert train[0] == train[1] and val[2] == val[3]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    dataset = RobotDataset(
        str(demonstrations), sensor_modalities=["joint_state"], split_manifest=str(path),
        val_ratio=0.9, max_train_episodes=1,
    )
    np.testing.assert_array_equal(dataset.train_mask, train)
    np.testing.assert_array_equal(dataset.val_mask, val)
    bad = dict(manifest, train_indices=[0, 2], val_indices=[1, 3])
    with pytest.raises(ValueError, match="trial cannot cross"):
        split_episodes(buffer, manifest=bad)
    bad = dict(manifest, val_indices=manifest["train_indices"])
    with pytest.raises(ValueError, match="partition"):
        split_episodes(buffer, manifest=bad)
    with pytest.raises(ValueError, match="source identity"):
        split_episodes(buffer, manifest=dict(manifest, source_id=None))
    moved = tmp_path / "moved.zarr"
    shutil.copytree(demonstrations, moved)
    moved_train, moved_val, _ = split_episodes(ReplayBuffer(moved, keys=[]), manifest=manifest)
    np.testing.assert_array_equal(moved_train, train)
    np.testing.assert_array_equal(moved_val, val)


def test_invalid_frames_are_excluded(demonstrations):
    root = zarr.open_group(str(demonstrations), mode="a")
    root["data/joint_state"][4, 0] = np.nan
    root["data/action"][7, 0] = np.inf
    dataset = RobotDataset(str(demonstrations), sensor_modalities=["joint_state"], horizon=3)
    assert 4 not in dataset.sampler.observation_source_rows
    assert 7 not in dataset.sampler.action_source_rows
    assert dataset.sampler.validity_summary["valid"] < dataset.sampler.validity_summary["candidate"]
    for i in range(len(dataset)):
        assert torch.isfinite(dataset[i]["action"]).all()
        assert torch.isfinite(dataset[i]["obs"]["joint_state"]).all()


def test_augmentation_owns_samples_and_workers_reopen(demonstrations):
    dataset = RobotDataset(
        str(demonstrations), sensor_modalities=["point_cloud"], horizon=3,
        augmentation_cfg={"pc": {"coord_noise": {"noise_std": 1.0, "prob": 1.0}}},
        val_ratio=0.25,
    )
    original = dataset.replay_buffer.read("point_cloud", slice(0, 3))
    dataset[0]["obs"]["point_cloud"].fill_(100)
    np.testing.assert_array_equal(original, dataset.replay_buffer.read("point_cloud", slice(0, 3)))
    restored = pickle.loads(pickle.dumps(dataset.replay_buffer))
    np.testing.assert_array_equal(original, restored.read("point_cloud", slice(0, 3)))
    validation = dataset.get_validation_dataset()
    torch.testing.assert_close(validation[0]["obs"]["point_cloud"], validation[0]["obs"]["point_cloud"])
    with pytest.raises(ValueError, match="training normalizer"):
        list(validation.iter_normalization_data("point_cloud"))
    loader = DataLoader(
        validation, batch_size=2, num_workers=2, persistent_workers=True,
        timeout=10,
    )
    try:
        assert next(iter(loader))["action"].shape == (2, 3, 19)
    finally:
        close_data_loader(loader)


@pytest.mark.parametrize("keep_uint8", [False, True])
def test_rgb_validation_crop_and_output_dtype(demonstrations, keep_uint8):
    dataset = RobotDataset(
        demonstrations, sensor_modalities=["rgb"], horizon=2, val_ratio=0.25,
        rgb_preprocess_size=(8, 8), rgb_random_crop_size=(4, 4), rgb_keep_uint8=keep_uint8,
    )
    training = dataset[0]["obs"]["rgb"]
    assert training.shape == (2, 3, 4, 4)
    assert training.dtype == (torch.uint8 if keep_uint8 else torch.float32)
    validation = dataset.get_validation_dataset()
    rows = validation.sampler.sample_source_rows(0)
    raw = validation.replay_buffer["rgb"][:][rows, 2:6, 2:6]
    expected = torch.from_numpy(raw).permute(0, 3, 1, 2)
    if not keep_uint8:
        expected = expected.float() / 255
    torch.testing.assert_close(validation[0]["obs"]["rgb"], expected)


def test_lamp_history_never_reads_future_or_previous_episode(demonstrations):
    dataset = LAMPDataset(
        str(demonstrations), sensor_modalities=["joint_state"], history_steps=3,
    )
    bootstrap = dataset.replay_buffer["joint_state"][12, 7:19]
    expected = np.stack([bootstrap, bootstrap, dataset.hand_actions[12]])
    np.testing.assert_array_equal(dataset.history_for_frame(13), expected)
    np.testing.assert_array_equal(dataset.history_for_frame(15), dataset.hand_actions[12:15])
    dataset.hand_actions[13] = 999
    np.testing.assert_array_equal(dataset.history_for_frame(13), expected)


def test_relative_windows_and_shared_hand_arrays(demonstrations):
    hand = HandChunkDataset(
        str(demonstrations), data_semantics=HAND_ACTION_SEMANTICS, horizon=4, val_ratio=0.25,
    )
    indices = np.array([0, 1, len(hand) - 1])
    actions, q0 = hand.raw_hand_batch(indices)
    rows = hand.sampler.source_rows(hand.sampler.indices[indices])
    np.testing.assert_array_equal(actions, hand.replay_buffer["action"][:][rows, 7:19])
    np.testing.assert_array_equal(q0, hand.replay_buffer["joint_state"][:][rows[:, 0], 7:19])
    assert hand.get_validation_dataset().hand_actions is hand.hand_actions
    assert not hand.hand_actions.flags.writeable
    all_actions, all_q0 = hand.raw_hand_batch(np.arange(len(hand)))
    relative = all_actions.astype(np.float64) - all_q0[:, None].astype(np.float64)
    mean, std, _ = fit_relative_statistics(hand, batch_size=5)
    np.testing.assert_allclose(mean, relative.mean((0, 1)), atol=1e-7)
    np.testing.assert_allclose(std, relative.std((0, 1)), atol=1e-7)

    dataset = LatentFlowDataset(
        str(demonstrations), sensor_modalities=["joint_state"],
        data_semantics=copy.deepcopy(HAND_ACTION_SEMANTICS), horizon=4, obs_horizon=2,
    )
    np.testing.assert_array_equal(dataset.sampler.sample_source_rows(0), [0, 0, 1, 2, 3])
    batch = dataset[0]
    np.testing.assert_array_equal(batch["action"], dataset.commands[:4])
    np.testing.assert_array_equal(batch["action_context"]["reference_state"], dataset.states[0])
    rows = dataset.sampler.source_rows()[:, 1:]
    expected = (dataset.commands[rows] - dataset.states[rows[:, 0], None]).reshape(-1, 19)
    np.testing.assert_array_equal(np.concatenate(list(dataset.iter_normalization_data("action"))), expected)
