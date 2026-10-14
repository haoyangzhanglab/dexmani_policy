import copy
import hashlib

import numpy as np
import pytest
import torch
import zarr

from dexmani_policy.datasets.lamp_dataset import LampDataset, OBS_ALIGNMENT


def write_lamp_data(path, lengths=(12, 12)):
    n = sum(lengths)
    root = zarr.open_group(str(path), mode="w")
    actions = np.repeat(np.arange(n, dtype=np.float32)[:, None], 19, axis=1)
    states = actions.copy()
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    states[starts, 7:] = -10 - np.arange(len(lengths))[:, None]
    for name, values in (("action", actions), ("joint_state", states)):
        root.create_dataset("data/" + name, data=values, chunks=(4, 19))
    root.create_dataset("meta/episode_ends", data=np.cumsum(lengths, dtype=np.int64))
    root.attrs.update(dt=0.0625, obs_alignment=OBS_ALIGNMENT, data_revision="lamp-test.v1")
    return root


def make_dataset(path, **overrides):
    options = dict(
        history_steps=3, horizon=4, obs_horizon=2, pad_before=1, pad_after=1,
        sensor_modalities=["joint_state"], require_latents=False,
    )
    options.update(overrides)
    return LampDataset(str(path), **options)


def write_artifacts(tmp_path, dataset, latent_dim=2):
    metadata = dataset.lamp_metadata
    prior_path, latent_path = tmp_path / "prior.pt", tmp_path / "latents.pt"
    torch.save({"_format": "lamp_prior.v1", "model_config": {"latent_dim": latent_dim},
                "metadata": metadata, "model_state_dict": {}}, prior_path)
    n = len(dataset.hand_actions)
    payload = {
        "_format": "lamp_latents.v1", "metadata": copy.deepcopy(metadata),
        "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "latents": torch.arange(n * latent_dim, dtype=torch.float32).reshape(n, latent_dim),
        "valid": torch.from_numpy(dataset.history_valid & np.isfinite(dataset.hand_actions).all(-1)),
        "reconstruction_error": torch.zeros(n), "latent_scale": torch.ones(latent_dim),
    }
    torch.save(payload, latent_path)
    return prior_path, latent_path, payload


def test_history_is_past_commands_with_measured_bootstrap(tmp_path):
    path = tmp_path / "data.zarr"
    write_lamp_data(path)
    dataset = make_dataset(path)
    np.testing.assert_array_equal(dataset.history_for_frame(0), np.full((3, 12), -10))
    np.testing.assert_array_equal(dataset.history_for_frame(1), np.array([-10, -10, 0])[:, None].repeat(12, 1))
    np.testing.assert_array_equal(dataset.history_for_frame(12), np.full((3, 12), -11))
    np.testing.assert_array_equal(dataset.history_for_frame(13), np.array([-11, -11, 12])[:, None].repeat(12, 1))
    first = dataset[0]
    assert first["obs"]["joint_state"].shape == (2, 19)
    assert first["action"].shape == (4, 19)
    torch.testing.assert_close(first["action_context"]["hand_history"], torch.full((3, 12), -10.0))
    assert "hand_latent_target" not in first
    assert dataset.sampler.source_rows(dataset.sampler.indices[:1])[0].tolist() == [0, 0, 1, 2]


def test_history_qualification_precedes_normalizer_rows(tmp_path):
    from dexmani_policy.training.build_utils import build_normalizer

    path = tmp_path / "data.zarr"
    root = write_lamp_data(path, (20,))
    root["data/action"][3, 7] = np.nan
    root["data/action"][4] = 1000
    dataset = make_dataset(path)
    assert dataset.history_valid[:8].tolist() == [True, True, True, True, False, False, False, True]
    rows = dataset.sampler.action_source_rows
    assert not set(rows) & {3, 4, 5, 6}
    for sample in dataset.sampler.source_rows():
        assert dataset.history_valid[sample].all()
    normalizer = build_normalizer(dataset, {"action": "limits"}, "action")
    assert normalizer["action"].input_stats["max"].max() < 1000
    root["data/joint_state"][0, 7] = np.nan
    bootstrap_bad = make_dataset(path)
    assert not bootstrap_bad.history_valid[:3].any()
    assert bootstrap_bad.history_valid[3]


def test_alignment_requires_evidence_and_conflicts_cannot_be_overridden(tmp_path):
    path = tmp_path / "data.zarr"
    root = write_lamp_data(path)
    del root.attrs["obs_alignment"]
    with pytest.raises(ValueError, match="explicit obs_alignment"):
        make_dataset(path)
    dataset = make_dataset(path, obs_alignment="obs_before_action")
    assert dataset.lamp_source_metadata["alignment_source"] == "explicit_assertion"
    root.attrs["obs_alignment"] = "obs_after_action"
    with pytest.raises(ValueError, match="Unsupported stored"):
        make_dataset(path, obs_alignment="obs_before_action")


def test_latent_rows_padding_and_validation_copy(tmp_path):
    path = tmp_path / "data.zarr"
    write_lamp_data(path)
    source = make_dataset(path, val_ratio=0.5)
    prior_path, latent_path, payload = write_artifacts(tmp_path, source)
    dataset = make_dataset(path, val_ratio=0.5, require_latents=True,
                           prior_path=prior_path, latent_path=latent_path)
    for split in (dataset, dataset.get_validation_dataset()):
        sample = split[0]
        rows = split.sampler.source_rows(split.sampler.indices[:1])[0]
        torch.testing.assert_close(sample["hand_latent_target"], payload["latents"][rows])
        query = rows[split.obs_horizon - 1]
        np.testing.assert_array_equal(sample["action_context"]["hand_history"], split.history_for_frame(query))
        assert split.data_recipe["lamp_latents"] == dataset.data_recipe["lamp_latents"]
    assert dataset.get_validation_dataset()._is_val
    assert dataset.lamp_metadata == source.lamp_metadata


@pytest.mark.parametrize("mutation", ["hand", "bootstrap", "dt", "split", "prior", "latent_valid"])
def test_artifacts_reject_changed_source_or_incomplete_targets(tmp_path, mutation):
    path = tmp_path / "data.zarr"
    root = write_lamp_data(path)
    source = make_dataset(path)
    prior_path, latent_path, payload = write_artifacts(tmp_path, source)
    options = {}
    if mutation == "hand":
        root["data/action"][1, 7] += 1
    elif mutation == "bootstrap":
        root["data/joint_state"][0, 7] += 1
    elif mutation == "dt":
        root.attrs["dt"] = 0.1
    elif mutation == "split":
        options["val_ratio"] = 0.5
    elif mutation == "prior":
        payload["prior_sha256"] = "0" * 64
        torch.save(payload, latent_path)
    else:
        payload["valid"][source.sampler.action_source_rows[0]] = False
        torch.save(payload, latent_path)
    with pytest.raises(ValueError, match="mismatch|lacks valid"):
        make_dataset(path, require_latents=True, prior_path=prior_path, latent_path=latent_path, **options)


def test_prior_preparation_does_not_read_artifact_paths(tmp_path):
    path = tmp_path / "data.zarr"
    write_lamp_data(path)
    dataset = make_dataset(path, prior_path=tmp_path / "missing.pt", latent_path=tmp_path / "missing-z.pt")
    assert len(dataset) > 0
    assert dataset.prior_sha256 is None


def test_cache_replacement_changes_recipe_without_mutating_loaded_labels(tmp_path):
    path = tmp_path / "data.zarr"
    write_lamp_data(path)
    source = make_dataset(path)
    prior_path, latent_path, payload = write_artifacts(tmp_path, source)
    options = dict(require_latents=True, prior_path=prior_path, latent_path=latent_path)
    before = make_dataset(path, **options)
    original = before[-1]["hand_latent_target"].clone()
    payload["latents"] += 10
    torch.save(payload, latent_path)
    after = make_dataset(path, **options)
    assert before.data_recipe["lamp_latents"] != after.data_recipe["lamp_latents"]
    torch.testing.assert_close(before[-1]["hand_latent_target"], original)
    torch.testing.assert_close(after[-1]["hand_latent_target"], original + 10)
    with pytest.raises(IndexError):
        before[-len(before) - 1]
    np.testing.assert_array_equal(before.history_for_rows(np.array([1], dtype=np.uint64))[0],
                                  before.history_for_frame(1))
