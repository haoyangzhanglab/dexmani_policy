"""当前 Real 来源分组、实际处理记录和接触监督的读取边界。"""

from copy import deepcopy

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.training.common.config import load_config
from dexmani_policy.training.common.data import build_dataset, record_data_processing
from dexmani_policy.training.pretrain.hand_prior.vqvae import prepare_hand_data


@pytest.fixture
def processing_source(tmp_path):
    path = tmp_path / "source.zarr"
    source = zarr.open_group(str(path), mode="w-")
    source.attrs.update(
        format="dexmani.real.canonical", task_name="pick_apple_messy",
        dt=0.05, action_space="joint",
        source_episodes=["a/take", "b/take", "a/take", "c/take"],
        source_row_ranges=[[0, 10], [0, 10], [10, 20], [0, 10]],
        data_processing={
            "point_cloud": {"num_points": 8},
            "hand_pointcloud": {"num_points": 8},
            "object_pointcloud": {"geometry_sha256": "a" * 64},
            "joint_state": {"joint_units": "rad"},
            "rgb": {"image_space": "pinhole"},
        },
    )
    source.create_dataset("meta/episode_ends", data=np.array([10, 20, 30, 40], dtype=np.int64))
    rng = np.random.default_rng(10)
    for key, shape in (
        ("point_cloud", (40, 8, 6)), ("hand_pointcloud", (40, 8, 3)),
        ("object_pointcloud", (40, 1024, 3)), ("joint_state", (40, 19)),
        ("contact_map", (40, 1024)), ("action", (40, 19)),
    ):
        values = rng.normal(size=shape).astype(np.float32)
        if key == "action":
            values[:, 7:] = np.repeat([1, 2, 1, 3], 10)[:, None]
        source.create_dataset(f"data/{key}", data=values)
    return path


@pytest.mark.parametrize("recipe", ["pretrain/obs/afro", "pretrain/obs/cordvip", "dp3"])
def test_processing_records_selected_fields_without_admission(tmp_path, processing_source, recipe):
    overrides = [f"zarr_path={processing_source}"]
    if recipe == "pretrain/obs/afro":
        overrides += ["dataset.horizon=4", "dataset.frame_stride=1", "dataset.pad_after=0"]
    cfg = load_config(recipe, overrides)
    cfg.workspace.output_dir = str(tmp_path / "run")
    dataset = build_dataset(cfg)[0] if recipe == "dp3" else hydra.utils.instantiate(cfg.dataset, _convert_="all")
    record_data_processing(cfg, dataset)
    processing = dataset.episode_store.attrs["data_processing"]
    selected = {key: processing[key] for key in dataset.episode_store.keys() if key in processing}
    assert OmegaConf.to_container(cfg.data_processing) == selected
    path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    restored = OmegaConf.load(path)
    changed = deepcopy(processing)
    for values in changed.values():
        values[next(iter(values))] = "changed research parameter"
    dataset.episode_store.attrs["data_processing"] = changed
    record_data_processing(restored, dataset)
    assert OmegaConf.to_container(restored.data_processing) == {
        key: changed[key] for key in dataset.episode_store.keys() if key in changed
    }
    assert "rgb" not in restored.data_processing


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("record", [None, {}])
def test_processing_unavailable_is_none(processing_source, real, record):
    source = zarr.open_group(str(processing_source), mode="a")
    source.attrs["data_processing"] = record
    if not real:
        del source.attrs["format"]
    dataset = CordViPPretrainDataset(processing_source, contact_loss_weight=0)
    cfg = OmegaConf.create({"data_processing": {"point_cloud": {"num_points": 32}}})
    record_data_processing(cfg, dataset)
    assert cfg.data_processing is None
    del dataset.episode_store.attrs["data_processing"]
    record_data_processing(cfg, dataset)
    assert cfg.data_processing is None


@pytest.mark.parametrize("consumer", ["policy", "afro", "cordvip"])
def test_source_split_and_segment_windows(processing_source, consumer):
    options = dict(seed=4, val_episodes=1, max_train_episodes=1)
    if consumer == "policy":
        dataset = DemonstrationDataset(
            processing_source, sensor_modalities=["joint_state"], obs_horizon=2,
            horizon=4, pad_before=1, pad_after=2, **options,
        )
    elif consumer == "afro":
        dataset = AFROPretrainDataset(
            processing_source, frame_stride=1, horizon=4, pad_before=1, pad_after=2, **options,
        )
    else:
        dataset = CordViPPretrainDataset(
            processing_source, contact_loss_weight=1, horizon=4, obs_horizon=2, pad_after=2, **options,
        )
    store = dataset.episode_store
    assert store.count_groups() == 3  # a/take and b/take are separate recordings.
    assert store.count_groups(dataset.train_mask) == store.count_groups(dataset.val_mask) == 1
    for mask in (dataset.train_mask, dataset.val_mask):
        assert mask[0] == mask[2]
    assert not (dataset.train_mask & dataset.val_mask).any()
    for index in range(len(dataset.sampler)):
        rows = dataset.sampler.sample_source_rows(index)
        assert len(np.unique(np.searchsorted(store.episode_ends, rows, side="right"))) == 1
    assert "groups=" in dataset.training_summary() and "segments=" in dataset.training_summary()


def test_vq_hand_prior_splits_source_groups(processing_source):
    cfg = OmegaConf.create({"dataset": {
        "zarr_path": str(processing_source), "seed": 4, "val_episodes": 1, "max_train_episodes": 1,
    }})
    train, val, normalizer = prepare_hand_data(cfg)
    train_ids = np.unique(normalizer["hand"].unnormalize(train).numpy().round().astype(int))
    val_ids = np.unique(normalizer["hand"].unnormalize(val).numpy().round().astype(int))
    assert len(train_ids) == len(val_ids) == 1 and train_ids[0] != val_ids[0]
    assert len(train) == (20 if train_ids[0] == 1 else 10)
    assert len(val) == (20 if val_ids[0] == 1 else 10)


def test_simulation_keeps_independent_episode_semantics(processing_source):
    source = zarr.open_group(str(processing_source), mode="a")
    del source.attrs["format"]
    store = EpisodeStore(processing_source, keys=["action"])
    assert store.episode_groups == [0, 1, 2, 3]
    train, val = split_episodes(store.episode_groups, val_episodes=2, max_train_episodes=1)
    assert train.sum() == 1 and val.sum() == 2


@pytest.mark.parametrize("key,value", [
    ("source_episodes", None), ("source_episodes", ["one"]),
    ("source_row_ranges", None), ("source_row_ranges", [[0, 9]] * 4),
    ("source_row_ranges", [[False, True]] * 4),
])
def test_real_requires_complete_sources(processing_source, key, value):
    source = zarr.open_group(str(processing_source), mode="a")
    source.attrs[key] = value
    with pytest.raises(ValueError, match=key):
        EpisodeStore(processing_source, keys=["action"])


def test_validation_count_is_source_count(processing_source):
    store = EpisodeStore(processing_source, keys=["action"])
    with pytest.raises(ValueError, match="source groups"):
        split_episodes(store.episode_groups, val_episodes=3)


@pytest.mark.parametrize("digest", [None, "", "bad-digest"])
def test_contact_requires_shared_slot_identity(processing_source, digest):
    source = zarr.open_group(str(processing_source), mode="a")
    processing = dict(source.attrs["data_processing"])
    processing["object_pointcloud"]["geometry_sha256"] = digest
    source.attrs["data_processing"] = processing
    with pytest.raises(ValueError, match="geometry_sha256"):
        CordViPPretrainDataset(processing_source, contact_loss_weight=1)


def test_contact_disabled_ignores_labels_and_allows_other_point_counts(processing_source):
    source = zarr.open_group(str(processing_source), mode="a")
    source["data"].create_dataset("object_pointcloud", data=np.ones((40, 13, 3)), overwrite=True)
    source["data"].create_dataset("contact_map", data=np.array(["unused malformed labels"]), overwrite=True)
    source.attrs["data_processing"] = None
    dataset = CordViPPretrainDataset(processing_source, contact_loss_weight=0)
    assert "contact_map" not in dataset.episode_store.keys()
    assert "contact_map" not in dataset[0]
    assert dataset[0]["obs"]["object_pointcloud"].shape == (4, 13, 3)
    del source["data/contact_map"]
    CordViPPretrainDataset(processing_source, contact_loss_weight=0)


@pytest.mark.parametrize("key", ["contact_map", "object_pointcloud"])
def test_contact_enabled_requires_1024_matching_slots(processing_source, key):
    source = zarr.open_group(str(processing_source), mode="a")
    shape = (40, 5) if key == "contact_map" else (40, 5, 3)
    source["data"].create_dataset(key, data=np.ones(shape, dtype=np.float32), overwrite=True)
    with pytest.raises(ValueError, match="1024"):
        CordViPPretrainDataset(processing_source, contact_loss_weight=1)


@pytest.mark.parametrize("contact_weight", [0, 1])
def test_contact_pretraining_current_checkpoint(tmp_path, processing_source, contact_weight):
    from dexmani_policy.training.pretrain.obs.common import train
    from dexmani_policy.training.pretrain.obs.cordvip import CordViP

    output = tmp_path / "pretrain"
    options = [
        "--output-dir", str(output), "--zarr-path", str(processing_source),
        "--device", "cpu", "--num-workers", "0", "--max-updates", "1", "--no-wandb",
        "n_obs_steps=1", "dataloader.batch_size=1", "dataloader.pin_memory=false",
        "training.loop.total_train_steps=2", "training.lr_warmup_steps=0",
        f"model.contact_loss_weight={contact_weight}",
    ]
    source = zarr.open_group(str(processing_source), mode="a")
    if not contact_weight:
        del source["data/contact_map"]
        source["data"].create_dataset("object_pointcloud", data=np.ones((40, 11, 3)), overwrite=True)
    train("cordvip", CordViP, options)
    checkpoint_path = output / "checkpoints/latest.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    assert saved["global_step"] == 1
    assert any(name.startswith("contact_head.") for name in saved["model_state"]) == bool(contact_weight)
    del saved

    resume = ["--resume-from", str(output), "--max-updates", "1", "--no-wandb"]
    processing = dict(source.attrs["data_processing"])
    processing["object_pointcloud"]["geometry_sha256"] = "b" * 64
    source.attrs["data_processing"] = processing
    if contact_weight:
        with pytest.raises(ValueError, match="resume requires the same object geometry_sha256"):
            train("cordvip", CordViP, resume)
        processing["object_pointcloud"]["geometry_sha256"] = "a" * 64
        source.attrs["data_processing"] = processing
    train("cordvip", CordViP, resume)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    assert saved["global_step"] == 2
    assert all(torch.isfinite(value).all() for value in saved["model_state"].values())
    assert all(int(state["step"]) == 2 for state in saved["optimizer_state"]["state"].values())


@pytest.mark.parametrize("digest", [None, "a" * 64])
def test_contact_geometry_generation_preserves_sources_and_summary(tmp_path, digest):
    from dexmani_policy.datasets.preprocess.contact_map import generate

    x, y = np.meshgrid(np.linspace(-.04, .04, 5), np.linspace(-.06, .06, 7))
    objects = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=-1).astype(np.float32)
    hand = np.array([[0, 0, .02]], dtype=np.float32)
    path = tmp_path / "geometry.zarr"
    source = zarr.open_group(str(path), mode="w-")
    source.attrs.update(
        format="dexmani.real.canonical", source_episodes=["left/take", "left/take"],
        source_row_ranges=[[0, 1], [1, 2]],
        data_processing={"object_pointcloud": {"geometry_sha256": digest}},
    )
    source.create_dataset("meta/episode_ends", data=np.array([1, 2]))
    source.create_dataset("data/joint_state", data=np.zeros((2, 19), dtype=np.float32))
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    source.create_dataset("data/action", data=actions)
    source.create_dataset("data/hand_pointcloud", data=np.stack((hand, hand + [1, 2, 3])).astype(np.float32))
    source.create_dataset("data/object_pointcloud", data=np.stack((objects, objects + [1, 2, 3])).astype(np.float32))
    output = tmp_path / "contact.zarr"
    generate(path, output)
    result = zarr.open_group(str(output), mode="r")
    assert dict(result.attrs) == dict(source.attrs)
    np.testing.assert_array_equal(result["meta/episode_ends"], [1, 2])
    np.testing.assert_array_equal(result["data/action"], actions)
    # For this plane and one hand point, the normal distance is exactly 0.02 m.
    distance = np.linalg.norm(objects.astype(np.float64) - hand, axis=-1)
    weighted = distance * np.exp(1 - .02 / (distance + 1e-8))
    expected = 2 / (1 + np.exp(10 * weighted))
    np.testing.assert_allclose(result["data/contact_map"][0], expected, atol=1e-6)
    np.testing.assert_allclose(result["data/contact_map"][1], expected, atol=2e-6)
