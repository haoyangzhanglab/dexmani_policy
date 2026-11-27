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
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.training.config import load_config, load_policy_config
from dexmani_policy.training.data import build_train_loader
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.pretrain.obs import common as obs_training
from dexmani_policy.training.pretrain.obs.afro import AFRO, main as pretrain

from tests.helpers import assert_loader_matches_items, check_recipe_composition


def test_afro_default_frames_use_only_complete_windows(segments):
    path, fields = segments
    cfg = load_config("pretrain/obs/afro", ["+policy_name=pretrain_label"])
    assert cfg.policy_name == "pretrain_label"
    assert (cfg.dataset.pad_before, cfg.dataset.pad_after, cfg.dataset.frame_stride) == (0, 0, 2)
    dataset = hydra.utils.instantiate(cfg.dataset, zarr_path=str(path), horizon=5)
    assert len(dataset) == 1
    np.testing.assert_array_equal(dataset[0]["obs"]["point_cloud"], fields["point_cloud"][[3, 5, 7], :, :3])
    with pytest.raises(ValueError, match="No AFROPretrainDataset training windows"):
        AFROPretrainDataset(path, frame_stride=2, horizon=6)
    loader_cfg = OmegaConf.create({"dataloader": {"num_workers": 0, "batch_size": 2, "drop_last": True}})
    with pytest.raises(ValueError, match="Training DataLoader is empty"):
        build_train_loader(loader_cfg, dataset)


def test_afro_batched_strided_windows(segments, monkeypatch):
    path, _ = segments
    dataset = AFROPretrainDataset(path, frame_stride=2, horizon=4, pad_before=2, pad_after=3)
    assert_loader_matches_items(dataset, monkeypatch)


def test_afro_validation_reuses_noise_and_preserves_training_rng():
    model = AFRO({"point_cloud": "identity"})
    generator = torch.Generator().manual_seed(17)
    dataset = [{"obs": {"point_cloud": torch.rand(4, 16, 3, generator=generator)}} for _ in range(4)]
    metrics = []
    for seed in (42, 112):
        state = torch.Generator().manual_seed(seed).get_state()
        torch.set_rng_state(state)
        metrics.append(obs_training.evaluate(
            model, dataset, 2, torch.device("cpu"), stop=SimpleNamespace(requested=False),
        ))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        assert model.training
    assert metrics[0] == metrics[1]


def test_afro_pretraining_still_requires_single_batch(tmp_path):
    with pytest.raises(ValueError, match="VICReg requires accumulation=1"):
        pretrain([f"workspace.output_dir={tmp_path}", "training.loop.gradient_accumulation_steps=2"])


@pytest.mark.parametrize("mode", ["limits", "identity", "mean_std"])
def test_encoder_normalization_inherited_before_configuration(tmp_path, make_demo, mode):
    path = make_demo()
    store = zarr.open_group(str(path), mode="a")
    for key in ("hand_pointcloud", "object_pointcloud"):
        store.create_dataset("data/" + key, data=store["data/point_cloud"][:], chunks=(4, 8, 6))
    fields = ["point_cloud"]
    modalities = list(dict.fromkeys([*fields, "joint_state"]))
    dataset = DemonstrationDataset(path, sensor_modalities=modalities, pc_dim=3, horizon=4, obs_horizon=1)
    cfg = load_policy_config('afro', [
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4", "agent.down_dims=[16,32]",
    ])
    cfg.agent.action_dim = 19
    cfg.policy_name = "renamed_experiment"
    reference = hydra.utils.instantiate(cfg.agent, _convert_="all")
    encoder = reference.obs_encoder.pc_encoder
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
    actual_encoder = model.obs_encoder.pc_encoder
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

    expected.fit_field("joint_state", dataset.iter_normalization_data("joint_state"))
    torch.testing.assert_close(model.normalizer["joint_state"].state_dict(), expected["joint_state"].state_dict())


def test_recipe_composition():
    check_recipe_composition("afro")


def test_afro_temporal_targets(segments):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group["meta/episode_ends"][:] = [3, 6, 9, 12]
    afro = AFROPretrainDataset(path, frame_stride=2, horizon=4, pad_before=1, pad_after=3)
    assert len(afro) == 16
    for episode in range(4):
        start = episode * 3
        for offset, rows in enumerate([[start, start + 1], [start, start + 2], [start + 1, start + 2], [start + 2] * 2]):
            np.testing.assert_array_equal(afro[episode * 4 + offset]["obs"]["point_cloud"], fields["point_cloud"][rows, :, :3])


def test_pretrain_teacher_order_and_resume(tmp_path, make_demo, monkeypatch):
    events = []
    update_teacher = AFRO.update_teacher
    optimizer_step = torch.optim.AdamW.step

    def teacher(model, epoch):
        events.append(("teacher", epoch))
        return update_teacher(model, epoch)

    def step(optimizer, *args, **kwargs):
        events.append(("optimizer", None))
        return optimizer_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(AFRO, "update_teacher", teacher)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    directory = tmp_path / "afro"
    pretrain([
        "--output-dir", str(directory), "--zarr-path", str(make_demo()),
        "--device", "cpu", "--no-wandb", "--max-updates", "1",
        "dataset.horizon=5", "dataset.frame_stride=2", "dataloader.batch_size=2",
        "dataloader.num_workers=0", "training.num_epochs=2",
    ])
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    assert saved.global_step == 1
    assert saved.effective_config["training"]["loop"]["total_train_steps"] == 8
    assert events == [("teacher", 0), ("optimizer", None)]

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not refit observation normalization")

    monkeypatch.setattr(obs_training, "fit_observation_normalizer", forbidden)
    pretrain(["--resume-from", str(directory), "--max-updates", "1"])
    resumed = load_checkpoint(directory / "checkpoints/latest.pt")
    assert resumed.global_step == resumed.scheduler_state["last_epoch"] == 2
    assert events == [("teacher", 0), ("optimizer", None)] * 2
    payload = torch.load(directory / "encoder.pt", weights_only=True)
    for key, value in payload["normalizer_state"].items():
        torch.testing.assert_close(value, saved.model_state["normalizer." + key])
