import copy

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.temporal_codec import TemporalHandCodec
from dexmani_policy.datasets.lamp_dataset import OBS_ALIGNMENT
from dexmani_policy.datasets.temporal_codec_dataset import TemporalCodecDataset
from dexmani_policy.training.train_temporal_codec import fit_relative_statistics, kl_coefficient, validate_config_paths


def write_data(path, lengths=(10, 10), point_cloud=False):
    n = sum(lengths)
    root = zarr.open_group(str(path), mode="w")
    actions = np.repeat(np.arange(n, dtype=np.float32)[:, None], 19, axis=1)
    states = actions - 0.25
    for name, values in (("action", actions), ("joint_state", states)):
        root.create_dataset("data/" + name, data=values, chunks=(4, 19))
    if point_cloud:
        root.create_dataset("data/point_cloud", data=np.zeros((n, 16, 3), dtype=np.float32))
    root.create_dataset("meta/episode_ends", data=np.cumsum(lengths, dtype=np.int64))
    # No dt requirement: this codec uses discrete temporal differences.
    root.attrs.update(obs_alignment=OBS_ALIGNMENT, data_revision="temporal-codec-test.v1")
    return root


def make_dataset(path, **overrides):
    options = dict(horizon=4, obs_horizon=2, pad_before=1, pad_after=1,
                   sensor_modalities=["joint_state"], require_codec=False)
    options.update(overrides)
    return TemporalCodecDataset(str(path), **options)


def write_codec(path, dataset):
    codec = TemporalHandCodec(horizon=dataset.horizon, latent_dim=2, hidden_dim=8,
                              n_layers=1, n_heads=2)
    codec.ready.fill_(True)
    payload = {"_format": "temporal_hand_codec.v1", "model_config": codec.model_config,
               "state_dict": codec.state_dict(), "metadata": dataset.codec_metadata}
    torch.save(payload, path)
    return payload


def test_raw_reference_is_query_observation_before_augmentation(tmp_path):
    path = tmp_path / "data.zarr"
    write_data(path)
    dataset = make_dataset(path, augmentation_cfg={"state": {"noise": {"noise_std": 5.0, "prob": 1.0}}})
    actions, references = dataset.raw_hand_batch(np.arange(len(dataset)))
    rows = dataset.sampler.source_rows()
    np.testing.assert_array_equal(actions, dataset.hand_actions[rows])
    np.testing.assert_array_equal(references, dataset.hand_states[rows[:, 1]])
    assert rows[0].tolist() == [0, 0, 1, 2]
    for i in (0, 1, len(dataset) - 1):
        sample = dataset[i]
        assert sample["obs"]["joint_state"].shape == (2, 19)
        assert sample["action"].shape == (4, 19)
        torch.testing.assert_close(sample["action_context"]["hand_reference"], torch.from_numpy(references[i]))
        assert not torch.equal(sample["obs"]["joint_state"][-1, 7:], sample["action_context"]["hand_reference"])
        torch.testing.assert_close(sample["action"][:, 7:], torch.from_numpy(actions[i]))
        assert "hand_latent_target" not in sample
    for sample_rows in rows:
        assert len(set(np.searchsorted([10, 20], sample_rows, side="right"))) == 1
    torch.testing.assert_close(dataset[-1]["action_context"]["hand_reference"], torch.from_numpy(references[-1]))
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]


def test_pretraining_batches_do_not_read_point_cloud_or_apply_augmentation(tmp_path, monkeypatch):
    path = tmp_path / "data.zarr"
    root = write_data(path, point_cloud=True)
    root["data/point_cloud"][2, 0, 0] = np.nan
    dataset = make_dataset(path, sensor_modalities=["joint_state", "point_cloud"], codec_path="missing.pt")
    rows = dataset.sampler.source_rows()
    assert not (rows[:, :2] == 2).any()  # Same modality qualification as the BC dataset.

    def forbidden(*args, **kwargs):
        raise AssertionError("Pretraining batches must use only cached raw hand arrays")

    monkeypatch.setattr(dataset.replay_buffer, "read", forbidden)
    monkeypatch.setattr(dataset, "apply_augmentation", forbidden)
    actions, q0 = dataset.raw_hand_batch(np.arange(len(dataset), dtype=np.uint64))
    assert actions.shape == (len(dataset), 4, 12) and q0.shape == (len(dataset), 12)
    fit_relative_statistics(dataset, batch_size=2)


def test_statistics_use_only_train_windows_and_count_repeated_padding(tmp_path):
    path = tmp_path / "data.zarr"
    root = write_data(path)
    source = make_dataset(path, val_ratio=0.5)
    val_episode = int(np.flatnonzero(source.val_mask)[0])
    root["data/action"][val_episode * 10:(val_episode + 1) * 10, 7:] += 10000
    dataset = make_dataset(path, val_ratio=0.5)
    actions, q0 = dataset.raw_hand_batch(np.arange(len(dataset)))
    relative = actions - q0[:, None]
    mean, std, recipe = fit_relative_statistics(dataset, batch_size=3)
    torch.testing.assert_close(mean, torch.from_numpy(relative.mean(axis=(0, 1))), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(std, torch.from_numpy(relative.std(axis=(0, 1))), rtol=1e-6, atol=1e-6)
    assert mean.abs().max() < 5
    assert recipe["frames"] == len(dataset) * dataset.horizon
    assert recipe["frames"] > len(dataset.sampler.action_source_rows)


def test_alignment_requires_evidence_and_rejects_conflicts(tmp_path):
    path = tmp_path / "data.zarr"
    root = write_data(path)
    del root.attrs["obs_alignment"]
    with pytest.raises(ValueError, match="explicit obs_alignment"):
        make_dataset(path)
    source = make_dataset(path, obs_alignment="obs_before_action")
    assert source.codec_metadata["source_metadata"]["alignment_source"] == "explicit_assertion"
    root.attrs["obs_alignment"] = "obs_after_action"
    with pytest.raises(ValueError, match="Unsupported stored"):
        make_dataset(path, obs_alignment="obs_before_action")


def test_codec_artifact_binding_and_validation_copy(tmp_path):
    path, codec_path = tmp_path / "data.zarr", tmp_path / "codec.pt"
    write_data(path)
    source = make_dataset(path, val_ratio=0.5)
    write_codec(codec_path, source)
    dataset = make_dataset(path, val_ratio=0.5, codec_path=codec_path, require_codec=True)
    validation = dataset.get_validation_dataset()
    assert dataset.codec_metadata == source.codec_metadata
    assert validation.codec_metadata == dataset.codec_metadata
    assert validation.codec_sha256 == dataset.codec_sha256
    assert validation.data_recipe == dataset.data_recipe
    assert validation._is_val and not validation.augmentors
    val_sample = validation[0]
    val_rows = validation.sampler.source_rows(validation.sampler.indices[:1])[0]
    torch.testing.assert_close(val_sample["action_context"]["hand_reference"],
                               torch.from_numpy(dataset.hand_states[val_rows[1]]))


@pytest.mark.parametrize("mutation", ["hand", "state", "dt", "split", "window", "query", "padding", "pc_validity"])
def test_artifact_rejects_source_split_window_and_qualification_changes(tmp_path, mutation):
    path, codec_path = tmp_path / "data.zarr", tmp_path / "codec.pt"
    root = write_data(path, point_cloud=True)
    options = {"sensor_modalities": ["joint_state", "point_cloud"]}
    source = make_dataset(path, **options)
    write_codec(codec_path, source)
    if mutation == "hand":
        root["data/action"][2, 7] += 1
    elif mutation == "state":
        root["data/joint_state"][2, 7] += 1  # Every measured reference is bound, not only episode start.
    elif mutation == "dt":
        root.attrs["dt"] = 0.05
    elif mutation == "split":
        options["val_ratio"] = 0.5
    elif mutation == "window":
        options["horizon"] = 5
    elif mutation == "query":
        options["obs_horizon"] = 1
    elif mutation == "padding":
        options["pad_after"] = 0
    else:
        root["data/point_cloud"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="mismatch|does not match"):
        make_dataset(path, codec_path=codec_path, require_codec=True, **options)


def test_replacing_codec_changes_recipe_and_config_paths_must_match(tmp_path):
    path, codec_path = tmp_path / "data.zarr", tmp_path / "codec.pt"
    write_data(path)
    source = make_dataset(path)
    payload = write_codec(codec_path, source)
    first = make_dataset(path, codec_path=codec_path, require_codec=True)
    replacement = copy.deepcopy(payload)
    replacement["state_dict"]["action_mean"] += 1
    torch.save(replacement, codec_path)
    second = make_dataset(path, codec_path=codec_path, require_codec=True)
    assert first.data_recipe["temporal_codec"] != second.data_recipe["temporal_codec"]
    cfg = OmegaConf.create({"codec": {"path": str(codec_path)}, "agent": {"codec_path": str(codec_path)},
                           "dataset": {"codec_path": str(codec_path)}})
    assert validate_config_paths(cfg) == codec_path
    cfg.agent.codec_path = str(tmp_path / "other.pt")
    with pytest.raises(ValueError, match="must match"):
        validate_config_paths(cfg)


def test_kl_warmup_then_linear_ramp():
    assert kl_coefficient(4000, 3e-4, 4000, 4000) == 0
    assert kl_coefficient(6000, 3e-4, 4000, 4000) == 1.5e-4
    assert kl_coefficient(8000, 3e-4, 4000, 4000) == 3e-4
    assert kl_coefficient(9000, 3e-4, 4000, 4000) == 3e-4
