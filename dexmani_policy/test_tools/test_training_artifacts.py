from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.train import build_trainer
from dexmani_policy.training.config import load_config
from dexmani_policy.training.pretrain.hand_prior.vqvae import train as train_vq
from dexmani_policy.training.pretrain.obs import common as obs_training
from dexmani_policy.training.pretrain.obs.cordvip import CordViP
from dexmani_policy.training.resume import prepare_resume


@pytest.fixture(params=["~/run", "relative/run", "absolute"])
def output_path(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    original = Path.expanduser

    def expand(path):
        if path.parts and path.parts[0] == "~":
            return tmp_path / "mapped_home" / Path(*path.parts[1:])
        return original(path)

    monkeypatch.setattr(Path, "expanduser", expand)
    value = str(tmp_path / "absolute") if request.param == "absolute" else request.param
    yield value, Path(value).expanduser().resolve()
    assert not (tmp_path / "~").exists()


def test_policy_artifact_paths(output_path, small_dp3_config):
    value, expected = output_path
    cfg = small_dp3_config(value)
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.logger.output_dir == expected
        assert trainer.cfg.workspace.output_dir == str(expected)
        trainer.train(max_updates=1)
    saved_cfg = OmegaConf.load(expected / "config.yaml")
    saved = load_checkpoint(expected / "checkpoints/latest.pt")
    assert saved_cfg.workspace.output_dir == saved.effective_config["workspace"]["output_dir"] == str(expected)
    original_config = (expected / "config.yaml").read_bytes()
    original_metrics = (expected / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_trainer(cfg)
    assert (expected / "config.yaml").read_bytes() == original_config
    assert (expected / "metrics.jsonl").read_bytes() == original_metrics
    with closing(prepare_resume(expected)) as resumed:
        assert resumed.logger.output_dir == expected
        assert resumed.global_step == 1


def test_vq_artifact_paths(output_path, make_demo):
    value, expected = output_path
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"zarr_path={make_demo()}", f"workspace.output_dir='{value}'",
        "vq_vae.device=cpu", "vq_vae.num_epochs=1", "vq_vae.save_epochs=1",
        "vq_vae.batch_size=12", "vq_vae.warmup_steps=0", "vq_vae.hidden_dim=16",
        "vq_vae.latent_dim=8", "vq_vae.num_layers=1",
    ])
    train_vq(cfg)
    assert cfg.workspace.output_dir == str(expected)
    assert OmegaConf.load(expected / "config.yaml").workspace.output_dir == str(expected)
    for name in ("metrics.jsonl", "codebook.npz", "checkpoints/last.pt"):
        assert (expected / name).exists()
    with pytest.raises(FileExistsError):
        train_vq(cfg)


@pytest.fixture
def contact_data(make_demo):
    path = make_demo(auxiliary=False)
    store = zarr.open_group(str(path), mode="a")
    points = store["data/point_cloud"][:, :, :3]
    store.create_dataset("data/hand_pointcloud", data=points)
    store.create_dataset("data/object_pointcloud", data=np.tile(points, (1, 128, 1)))
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.float32))
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": "ordered-geometry"}}
    return path


def test_observation_artifacts(output_path, contact_data, monkeypatch):
    value, directory = output_path
    args = [
        "--output-dir", value, "--zarr-path", str(contact_data),
        "--device", "cpu", "--no-wandb", "--max-updates", "1",
        "n_obs_steps=1", "dataloader.num_workers=0", "dataloader.batch_size=2",
        "training.loop.total_train_steps=3", "training.loop.gradient_accumulation_steps=2",
    ]
    obs_training.train("cordvip", CordViP, args)
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    assert saved.global_step == 1
    assert saved.effective_config["workspace"]["output_dir"] == str(directory)
    assert OmegaConf.load(directory / "config.yaml").workspace.output_dir == str(directory)
    with pytest.raises(FileExistsError):
        obs_training.train("cordvip", CordViP, args)
    assert any(key.startswith("contact_head.") for key in saved.model_state)
    payload = torch.load(directory / "encoder.pt", weights_only=True)
    for name in ("encoder", "normalizer"):
        expected = {key.removeprefix(name + "."): value for key, value in saved.model_state.items()
                    if key.startswith(name + ".")}
        torch.testing.assert_close(payload[name + "_state"], expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not use fresh fitting or the fresh output guard")

    monkeypatch.setattr(obs_training, "fit_observation_normalizer", forbidden)
    monkeypatch.setattr(obs_training, "check_fresh_output", forbidden)
    obs_training.train("cordvip", CordViP, ["--resume-from", value, "--max-updates", "1"])
    resumed = load_checkpoint(directory / "checkpoints/latest.pt")
    assert resumed.global_step == 2
    assert resumed.scheduler_state["last_epoch"] == 2
    for key in payload["normalizer_state"]:
        torch.testing.assert_close(resumed.model_state["normalizer." + key], payload["normalizer_state"][key])

    store = zarr.open_group(str(contact_data), mode="a")
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": "different-geometry"}}
    artifacts = [directory / name for name in ("config.yaml", "metrics.jsonl", "encoder.pt", "checkpoints/latest.pt")]
    before = [path.read_bytes() for path in artifacts]
    with pytest.raises(ValueError, match="same object geometry"):
        obs_training.train("cordvip", CordViP, ["--resume-from", value, "--max-updates", "1"])
    assert [path.read_bytes() for path in artifacts] == before


def test_contact_requires_geometry_identity(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": None}}
    with pytest.raises(ValueError, match="geometry_sha256"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)


def test_disabled_contact_does_not_require_labels_or_geometry(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=store["data/hand_pointcloud"][:])
    del store.attrs["data_processing"]
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=0)
    assert "contact_map" not in dataset.episode_store.keys()
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), contact_loss_weight=0)
    assert not hasattr(model, "contact_head")
    batch = {"obs": {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}}
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "arm", "hand"}
