import shutil
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.preprocess.contact_map import (
    batched_contact_maps, compute_contact_map, generate as generate_contacts,
)
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.pretrain.obs import common as obs_training
from dexmani_policy.training.pretrain.obs.cordvip import CordViP, main as pretrain

from tests.helpers import assert_loader_matches_items, check_recipe_composition


@pytest.fixture
def contact_data(make_demo):
    path = make_demo(auxiliary=False, episode_ends=(4, 8, 12))
    store = zarr.open_group(str(path), mode="a")
    points = store["data/point_cloud"][:, :, :3]
    store.create_dataset("data/hand_pointcloud", data=points)
    template = np.random.default_rng(45).normal(size=(1024, 3)).astype("<f4")
    world_points = template[None] + np.arange(12, dtype=np.float32)[:, None, None] * 0.1
    store.create_dataset("data/object_pointcloud", data=world_points)
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.float32))
    return path


@pytest.mark.parametrize("contact", [False, True])
def test_cordvip_batched_contact_labels(contact_data, monkeypatch, contact):
    dataset = CordViPPretrainDataset(contact_data, obs_horizon=4, contact_loss_weight=int(contact))
    batch = assert_loader_matches_items(dataset, monkeypatch)
    assert ("contact_map" in batch) == contact


def test_contact_requires_aligned_points(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.int64))
    with pytest.raises(ValueError, match="contact_map must be floating point"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)

    del store["data/contact_map"]
    store.create_dataset("data/contact_map", data=np.zeros((12, 63), dtype=np.float32))
    with pytest.raises(ValueError, match="Contact labels must match the object point slots"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)

    points = store["data/object_pointcloud"][:, :63]
    hand = store["data/hand_pointcloud"][:]
    points[1] = points[0, ::-1]
    hand[1] = hand[0]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=points)
    store["data/hand_pointcloud"][:] = hand
    output = contact_data.parent / "cordvip" / contact_data.name
    generate_contacts(contact_data)
    labelled = zarr.open_group(str(output), mode="r")
    labels = labelled["data/contact_map"][:]
    assert labels.shape == (12, 63)
    np.testing.assert_array_equal(labelled["data/object_pointcloud"][:], points)
    np.testing.assert_allclose(labels[1], labels[0, ::-1], rtol=1e-6)

    dataset = CordViPPretrainDataset(output, contact_loss_weight=1, obs_horizon=1)
    sample = dataset[1]
    torch.testing.assert_close(sample["contact_map"], torch.from_numpy(labels[1:2]))
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), n_obs_steps=1)
    batch = {
        "obs": {key: value.unsqueeze(0) for key, value in sample["obs"].items()},
        "contact_map": sample["contact_map"].unsqueeze(0),
    }
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_contact_matches_dense_formula():
    rng = np.random.default_rng(32)
    objects = rng.normal(size=(23, 3))
    objects[-1] = [20, 0, 0]
    normals = rng.normal(size=objects.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    hand = rng.normal(size=(19, 3))
    hand[0] = objects[0]
    differences = hand[None] - objects[:, None]
    distance = np.linalg.norm(differences, axis=-1)
    alignment = np.abs(np.einsum("ijk,ik->ij", differences, normals)) / (distance + 1e-8)
    minimum = (distance * np.exp(1 - alignment)).min(axis=1)
    expected = 2 / (1 + np.exp(10 * minimum))
    contacts = compute_contact_map(objects, normals, hand, point_chunk_size=7)
    np.testing.assert_allclose(contacts, expected, atol=1e-15)
    np.testing.assert_allclose(contacts[-1], expected[-1], rtol=1e-12, atol=0)
    batched = batched_contact_maps(
        np.stack((objects, objects[::-1])),
        np.stack((normals, normals[::-1])),
        np.stack((hand, hand)),
        "cpu",
    )
    np.testing.assert_array_equal(batched, np.stack((contacts, contacts[::-1])).astype(np.float32))


def test_contact_reads_metadata_and_points_from_same_resolved_source(contact_data, tmp_path, monkeypatch):
    source = zarr.open_group(str(contact_data), mode="a")
    source.create_dataset("meta/episode_seeds", data=[32, 33, 34])
    alternate = tmp_path / "alternate.zarr"
    shutil.copytree(contact_data, alternate)
    other = zarr.open_group(str(alternate), mode="a")
    other["meta/episode_seeds"][:] = [42, 43, 44]
    other["meta/episode_ends"][:] = [3, 8, 12]
    alias = tmp_path / "latest.zarr"
    alias.symlink_to(contact_data, target_is_directory=True)
    init = EpisodeStore.__init__

    def move_alias(store, *args, **kwargs):
        init(store, *args, **kwargs)
        alias.unlink()
        alias.symlink_to(alternate, target_is_directory=True)

    monkeypatch.setattr(EpisodeStore, "__init__", move_alias)
    output = tmp_path / "prepared.zarr"
    generate_contacts(alias, output)
    labelled = zarr.open_group(str(output), mode="r")
    assert labelled.attrs["source_zarr_path"] == str(contact_data)
    np.testing.assert_array_equal(labelled["meta/episode_seeds"][:], [32, 33, 34])
    np.testing.assert_array_equal(labelled["meta/episode_ends"][:], [4, 8, 12])
    np.testing.assert_array_equal(labelled["data/object_pointcloud"][:], source["data/object_pointcloud"][:])


def test_observation_validation_reports_nonfinite_contact_metric(contact_data):
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=1, val_episodes=1, obs_horizon=1)
    validation = dataset.get_validation_dataset()
    row = validation.sampler.source_rows(0, 0)
    zarr.open_group(str(contact_data), mode="a")["data/contact_map"][row, 0] = np.nan
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), n_obs_steps=1)

    with pytest.raises(ValueError, match="Non-finite observation pretraining validation metrics: .*val/contact"):
        obs_training.evaluate(model, validation, 2, torch.device("cpu"), stop=SimpleNamespace(requested=False))


def test_observation_artifacts(output_path, contact_data, tmp_path, monkeypatch):
    value, directory = output_path
    source = contact_data
    raw = zarr.open_group(str(source), mode="a")
    groups = ["a", "b", "c"]
    raw.attrs.update(format="dexmani.real.canonical", source_episodes=groups)
    derived = source.parent / "cordvip" / source.name
    derived.parent.mkdir()
    shutil.copytree(source, derived)
    del raw["data/contact_map"]
    store = zarr.open_group(str(derived), mode="a")
    store.attrs["source_zarr_path"] = str(source)
    policy = tmp_path / "policy.yaml"
    OmegaConf.save(OmegaConf.create(dict(
        task_name="test", zarr_path=str(source),
        dataset=dict(seed=42, val_episodes=1, max_train_episodes=1),
    )), policy)
    args = [
        "--output-dir", value, "--policy-config", str(policy),
        "--device", "cpu", "--no-wandb", "--max-updates", "1",
        "normalization.joint_state=mean_std", "n_obs_steps=1", "dataloader.num_workers=0", "dataloader.batch_size=2",
        "training.loop.total_train_steps=3", "training.loop.gradient_accumulation_steps=2",
    ]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    pretrain(args)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert torch.get_float32_matmul_precision() == "medium"
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    assert saved.global_step == 1
    assert saved.effective_config["dataset"]["zarr_path"] == str(derived)
    betas = saved.optimizer_state["param_groups"][0]["betas"]
    assert isinstance(betas, tuple) and betas == (0.95, 0.999)
    assert saved.effective_config["workspace"]["output_dir"] == str(directory)
    assert OmegaConf.load(directory / "config.yaml").workspace.output_dir == str(directory)
    with pytest.raises(FileExistsError):
        pretrain(args)
    assert any(key.startswith("contact_head.") for key in saved.model_state)
    payload = torch.load(directory / "encoder.pt", weights_only=True)
    assert payload["training_data"] == saved.effective_config["training_data"]
    assert "episode_layout" not in payload
    assert saved.effective_config["episode_layout"]["episode_groups"] == groups
    assert payload["training_data"]["source"] == str(source)
    train_group, = payload["training_data"]["train_source_episodes"]
    start = 4 * groups.index(train_group)
    states = store["data/joint_state"][start:start + 4]
    np.testing.assert_allclose(payload["normalizer_state"]["fields.joint_state.scale"],
                               1 / states.std(axis=0, dtype=np.float64), rtol=1e-6)
    for name in ("encoder", "normalizer"):
        expected = {key.removeprefix(name + "."): value for key, value in saved.model_state.items()
                    if key.startswith(name + ".")}
        torch.testing.assert_close(payload[name + "_state"], expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not use fresh fitting or the fresh output guard")

    monkeypatch.setattr(obs_training, "fit_observation_normalizer", forbidden)
    monkeypatch.setattr(obs_training, "check_fresh_output", forbidden)
    resume_args = ["--resume-from", value, "--max-updates", "1"]
    pretrain(resume_args)
    resumed = load_checkpoint(directory / "checkpoints/latest.pt")
    assert resumed.global_step == 2
    assert resumed.scheduler_state["last_epoch"] == 2
    assert resumed.effective_config["training_data"] == saved.effective_config["training_data"]
    for key in payload["normalizer_state"]:
        torch.testing.assert_close(resumed.model_state["normalizer." + key], payload["normalizer_state"][key])

    artifacts = [directory / name for name in ("config.yaml", "metrics.jsonl", "encoder.pt", "checkpoints/latest.pt")]
    before = [path.read_bytes() for path in artifacts]
    with pytest.raises(OSError):
        pretrain(["--resume-from", str(artifacts[-1]), "--max-updates", "1"])
    with pytest.raises(SystemExit) as rejected:
        pretrain([*resume_args, "--zarr-path", str(source)])
    assert rejected.value.code == 2

    val_group, = payload["training_data"]["val_source_episodes"]
    store.attrs["source_episodes"] = ["changed" if group == val_group else group for group in groups]
    with pytest.raises(ValueError, match="Resume validation source groups mismatch"):
        pretrain(resume_args)
    store.attrs["source_episodes"] = groups
    store.attrs["source_zarr_path"] = str(derived)
    with pytest.raises(ValueError, match="Resume data source mismatch"):
        pretrain(resume_args)
    assert [path.read_bytes() for path in artifacts] == before


def test_disabled_contact_allows_other_point_counts(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=store["data/hand_pointcloud"][:])
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=0)
    assert "contact_map" not in dataset.episode_store.keys()
    model = CordViP(
        dict.fromkeys(dataset.sensor_modalities, "identity"), contact_loss_weight=0,
        arm_loss_weight=0,
    )
    assert not hasattr(model, "contact_head")
    assert not hasattr(model, "arm_head")
    batch = {"obs": {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}}
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "hand"}


@pytest.mark.parametrize("mode", ["limits", "identity", "mean_std"])
def test_encoder_normalization_inherited_before_configuration(tmp_path, make_demo, mode):
    path = make_demo()
    store = zarr.open_group(str(path), mode="a")
    for key in ("hand_pointcloud", "object_pointcloud"):
        store.create_dataset("data/" + key, data=store["data/point_cloud"][:], chunks=(4, 8, 6))
    fields = ["hand_pointcloud", "object_pointcloud", "joint_state"]
    modalities = list(dict.fromkeys([*fields, "joint_state"]))
    dataset = DemonstrationDataset(path, sensor_modalities=modalities, pc_dim=3, horizon=4, obs_horizon=1)
    cfg = load_policy_config('cordvip', [
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4", "agent.down_dims=[16,32]",
    ])
    cfg.agent.action_dim = 19
    cfg.policy_name = "renamed_experiment"
    reference = hydra.utils.instantiate(cfg.agent, _convert_="all")
    encoder = reference.obs_encoder.interaction_encoder
    dims = {"action": 19, **{key: 19 if key == "joint_state" else 3 for key in modalities}}
    inherited = LinearNormalizer({key: (dims[key],) for key in fields if mode != "identity"})
    for field in inherited.fields.values():
        field.scale.fill_(2.5)
        field.offset.fill_(0.75)
    source = tmp_path / "encoder.pt"
    data_source = str(make_demo("independent")) if mode == "identity" else dataset.episode_store.source_path
    payload = {
        "encoder_state": encoder.state_dict(),
        "normalizer_state": inherited.state_dict(),
        "normalization": dict.fromkeys(fields, mode),
        "training_data": {"source": data_source, "train_source_episodes": [0]},
    }
    torch.save(payload, source)
    cfg.initialization.encoder = str(source)
    model = build_policy(cfg, dataset, dims)
    assert cfg.prior_data[0].budget_verified == (mode != "identity")
    actual_encoder = model.obs_encoder.interaction_encoder
    torch.testing.assert_close(actual_encoder.state_dict(), encoder.state_dict())
    for key in fields:
        assert cfg.normalization[key] == mode
        if mode != "identity":
            torch.testing.assert_close(model.normalizer[key].state_dict(), inherited[key].state_dict())
        else:
            assert key not in model.normalizer.fields
    expected = LinearNormalizer()
    expected.fit_field("action", dataset.iter_normalization_data("action"))
    torch.testing.assert_close(model.normalizer["action"].state_dict(), expected["action"].state_dict())
    with open_dict(cfg):
        cfg.normalization_dims = dims
    checkpoint = Checkpoint(0, model.state_dict(), {}, {}, OmegaConf.to_container(cfg, resolve=False))
    source.unlink()
    for resuming in (False, True):
        restored = build_policy(cfg, dataset, dims, checkpoint=checkpoint, resuming=resuming)
        torch.testing.assert_close(restored.state_dict(), checkpoint.model_state)


def test_recipe_composition():
    check_recipe_composition("cordvip")
