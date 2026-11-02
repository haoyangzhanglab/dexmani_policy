"""Run pretraining checks on CPU or with --device cuda:0."""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.agents.policies.afro import AFROEncoder
from dexmani_policy.agents.policies.cordvip import CordViPEncoder
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.preprocess.contact_map import generate
from dexmani_policy.training.common.config import load_policy_config, validate_config
from dexmani_policy.training.pretrain.obs import common
from dexmani_policy.training.pretrain.obs.afro import AFRO, vicreg_loss


def check_tactile_configs():
    cases = (
        ("adaptac", "force", "sim"), ("adaptac", "force", "real"),
        ("adaptac", "taxel", "real"), ("adaptac", "patch", "real"),
        ("interact_dex", "force", "sim"), ("interact_dex", "force", "real"),
    )
    for method, tactile, source in cases:
        cfg = load_policy_config(method, [f"tactile={tactile}", f"tactile_source={source}"])
        validate_config(cfg)
        assert cfg.agent.tactile_source == cfg.dataset.tactile_source == source
        if method == "interact_dex":
            assert cfg.agent.interaction_horizon == cfg.dataset.interaction_horizon == 4
            disabled = load_policy_config(method, ["agent.future_force_loss_weight=0"])
            assert "interaction_horizon" not in disabled.dataset
            assert disabled.agent.interaction_horizon == 4
        encoder = hydra.utils.instantiate(cfg.agent.tactile_encoder)
        assert encoder.input_key in cfg.dataset.sensor_modalities
        assert (encoder.num_tokens, encoder.out_dim) == (5, 64)


def check_afro_directions(device=torch.device("cpu")):
    model = AFRO({"point_cloud": "identity"}).to(device)
    with torch.no_grad():
        for block in model.fdm.blocks:
            block.adaln_modulation[-1].weight.normal_(0, 0.01)
    reference = copy.deepcopy(model)
    points = torch.randn(2, 3, 8, 3, device=device)
    torch.manual_seed(17)
    loss, _ = model({"obs": {"point_cloud": points}})
    loss.backward()
    rng = get_rng_state(device)

    torch.manual_seed(17)
    online = reference.encoder(points.flatten(0, 1)).reshape(2, 3, 64)
    with torch.no_grad():
        teacher = reference.teacher(points.flatten(0, 1)).reshape(2, 3, 64)
    losses = []
    for current, following, target in (
        (online[:, :-1], online[:, 1:], teacher[:, 1:]),
        (online[:, 1:], online[:, :-1], teacher[:, :-1]),
    ):
        target = target.reshape(-1, 64)
        latent = reference.idm(following - current).reshape(-1, 16)
        timestep = torch.randint(0, 100, (len(target),), device=device)
        noisy = reference.noise_scheduler.add_noise(target, torch.randn_like(target), timestep)
        prediction = reference.fdm(noisy, timestep, current.reshape(-1, 64), latent)
        losses.append(vicreg_loss(prediction, target)[0])
    expected = sum(losses)
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=2e-4, atol=2e-5)
    actual_rng = get_rng_state(device)
    for key in ("torch", "torch_cuda"):
        torch.testing.assert_close(rng[key], actual_rng[key], rtol=0, atol=0)
    for param, other in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, other.grad, rtol=2e-4, atol=2e-5)
    model.update_teacher(17)
    reference.update_teacher(17)
    torch.testing.assert_close(model.teacher.state_dict(), reference.teacher.state_dict(), rtol=0, atol=0)


def check_contact_data(root):
    source = zarr.open_group(str(root / "source.zarr"), mode="w")
    source.create_dataset("meta/episode_ends", data=np.array([8, 16, 24]))
    rng = np.random.default_rng(7)
    for key, shape in (
        ("point_cloud", (24, 8, 3)), ("hand_pointcloud", (24, 8, 6)),
        ("object_pointcloud", (24, 1024, 6)), ("joint_state", (24, 19)),
        ("action", (24, 19)),
    ):
        source.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype(np.float32))
    # Warm Open3D before starting workers to catch unsafe fork-based execution.
    generate(root / "source.zarr", root / "serial.zarr", num_workers=1)
    generate(root / "source.zarr", root / "parallel.zarr", num_workers=2)
    serial = zarr.open_group(str(root / "serial.zarr"), mode="r")
    parallel = zarr.open_group(str(root / "parallel.zarr"), mode="r")
    for key in serial["data"]:
        np.testing.assert_array_equal(serial["data"][key][:], parallel["data"][key][:])
        if key != "contact_map":
            np.testing.assert_array_equal(source["data"][key][:], serial["data"][key][:])
    np.testing.assert_array_equal(source["meta/episode_ends"][:], parallel["meta/episode_ends"][:])
    labels = parallel["data/contact_map"][:]
    assert labels.dtype == np.float32 and np.isfinite(labels).all()
    assert (labels >= 0).all() and (labels <= 1).all()
    try:
        generate(root / "source.zarr", root / "serial.zarr")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was overwritten")
    source["data/hand_pointcloud"][0, 0, 0] = np.nan
    try:
        generate(root / "source.zarr", root / "invalid.zarr")
    except ValueError as error:
        assert "Nonfinite data" in str(error)
    else:
        raise AssertionError("Nonfinite geometry was accepted")
    assert not (root / "invalid.zarr").exists() and not list(root.glob(".contact-*"))


def check_training(root, method, device=torch.device("cpu")):
    source = root / ("source.zarr" if method == "afro" else "parallel.zarr")
    overrides = [
        f"zarr_path={source}", "dataset.horizon=3", "dataset.pad_before=0", "dataset.pad_after=0",
        "dataset.val_episodes=1", "dataloader.batch_size=4", "dataloader.num_workers=0",
        "dataloader.pin_memory=false", f"training.device={device}", "training.lr_warmup_steps=0",
        "training.val_every_epochs=1", "training.loop.log_interval_steps=1",
    ]
    if method == "afro":
        overrides += ["dataset.frame_stride=1", "training.num_epochs=2", "training.save_every_epochs=1"]
        updates = 4
    else:
        overrides += ["n_obs_steps=2", "training.loop.total_train_steps=4",
                      "training.save_every_updates=1", "training.loop.gradient_accumulation_steps=2"]
        updates = 3
    base = load_policy_config(f"pretrain/obs/{method}", overrides)
    for name, count in (("full", updates), ("resumed", 1)):
        cfg = copy.deepcopy(base)
        cfg.workspace.output_dir = str(root / method / name)
        common.train(method, cfg, max_updates=count)
    directory = root / method / "resumed"
    checkpoint = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["effective_config"])
    with patch.object(common, "fit_normalizer", side_effect=AssertionError("Refit on resume")):
        common.train(method, cfg, checkpoint, max_updates=updates - 1)
    del checkpoint
    full = torch.load(root / method / "full/resume.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "optimizer_state", "scheduler_state", "global_step"):
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    for key in ("torch", "torch_cuda"):
        torch.testing.assert_close(full["rng_state"][key], resumed["rng_state"][key], rtol=0, atol=0)
    assert full["global_step"] == updates
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert any("val/loss" in record for record in records)
    common.export_encoder(resumed["model_state"], cfg.normalization, root / "export.pt")
    package = torch.load(root / "export.pt", weights_only=True)
    assert package["normalization"] == dict(cfg.normalization)
    saved = torch.load(directory / "encoder.pt", weights_only=True)
    assert package["normalization"] == saved["normalization"]
    for key in ("encoder_state", "normalizer_state"):
        torch.testing.assert_close(package[key], saved[key], rtol=0, atol=0)
    encoder = AFROEncoder(2).pc_encoder if method == "afro" else CordViPEncoder(2).interaction_encoder
    encoder.load_state_dict(package["encoder_state"], strict=True)

    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all").get_validation_dataset()
    from dexmani_policy.training.pretrain.obs.cordvip import CordViP

    model = AFRO(cfg.normalization) if method == "afro" else CordViP(cfg.normalization, n_obs_steps=2, **cfg.model)
    model.load_state_dict(resumed["model_state"], strict=True)
    model.to(device)
    teacher = copy.deepcopy(model.teacher.state_dict()) if method == "afro" else None
    common.evaluate(model, dataset, 4, device, stop_requested=lambda: False)
    if teacher is not None:
        torch.testing.assert_close(teacher, model.teacher.state_dict(), rtol=0, atol=0)
    state = get_rng_state(device)
    with patch.object(model, "forward", side_effect=RuntimeError("validation failure")):
        try:
            common.evaluate(model, dataset, 2, device, stop_requested=lambda: False)
        except RuntimeError as error:
            assert str(error) == "validation failure"
        else:
            raise AssertionError("Validation error was swallowed")
    assert model.training
    actual_rng = get_rng_state(device)
    for key in ("torch", "torch_cuda"):
        torch.testing.assert_close(state[key], actual_rng[key], rtol=0, atol=0)
    stop = [False]
    def interrupted(batch):
        stop[0] = True
        torch.rand((), device=device)
        return torch.tensor(0.0), {"loss": torch.tensor(0.0)}
    with patch.object(model, "forward", side_effect=interrupted):
        assert common.evaluate(model, dataset, 2, device, stop_requested=lambda: stop[0]) is None
    assert model.training
    actual_rng = get_rng_state(device)
    for key in ("torch", "torch_cuda"):
        torch.testing.assert_close(state[key], actual_rng[key], rtol=0, atol=0)


def check_train_only(root, method, device=torch.device("cpu")):
    source = root / ("source.zarr" if method == "afro" else "parallel.zarr")
    overrides = [
        f"zarr_path={source}", "dataset.horizon=3", "dataset.pad_before=0", "dataset.pad_after=0",
        "dataloader.batch_size=4", "dataloader.num_workers=0", "dataloader.pin_memory=false",
        f"training.device={device}", "training.lr_warmup_steps=0", "training.val_every_epochs=0",
    ]
    if method == "afro":
        overrides += ["dataset.frame_stride=1", "training.num_epochs=1", "normalization.point_cloud=identity"]
    else:
        overrides += [
            "n_obs_steps=2", "training.loop.total_train_steps=2",
            "normalization.hand_pointcloud=identity", "normalization.object_pointcloud=identity",
            "normalization.joint_state=identity",
        ]
    cfg = load_policy_config(f"pretrain/obs/{method}", overrides)
    cfg.workspace.output_dir = str(root / method / "train_only")
    assert cfg.dataset.val_episodes == 0
    with (
        patch.object(ObservationPretrainDataset, "get_validation_dataset", side_effect=AssertionError("Built val view")),
        patch.object(ObservationPretrainDataset, "iter_normalization_data", side_effect=AssertionError("Read identity stats")),
        patch.object(common, "evaluate", side_effect=AssertionError("Evaluated without val")),
    ):
        common.train(method, cfg, max_updates=1)
    package = torch.load(Path(cfg.workspace.output_dir) / "encoder.pt", weights_only=True)
    assert package["normalizer_state"] == {}
    assert all(mode == "identity" for mode in package["normalization"].values())
    records = [json.loads(line) for line in (Path(cfg.workspace.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert all(not any(key.startswith("val/") for key in record) for record in records)


def check_pretrain_samples(root):
    path = root / "points_only.zarr"
    source = zarr.open_group(str(path), mode="w")
    source.create_dataset("meta/episode_ends", data=np.array([5, 6]))
    points = np.arange(6 * 2 * 6, dtype=np.float32).reshape(6, 2, 6)
    source.create_dataset("data/point_cloud", data=points)
    dataset = AFROPretrainDataset(path, frame_stride=2, horizon=5, pad_after=0, seed=0, val_episodes=1)
    assert dataset.episode_store.keys() == ("point_cloud",)
    with patch.object(dataset.episode_store, "read", wraps=dataset.episode_store.read) as read:
        sample = dataset[0]["obs"]["point_cloud"]
    np.testing.assert_array_equal(sample, points[[0, 2, 4], :, :3])
    np.testing.assert_array_equal(read.call_args.args[1], [0, 2, 4])
    try:
        dataset.get_validation_dataset()
    except ValueError as error:
        assert "validation windows" in str(error)
    else:
        raise AssertionError("Explicit val without windows was accepted")

    dataset = CordViPPretrainDataset(
        root / "parallel.zarr", horizon=5, obs_horizon=2, pad_before=0, pad_after=0, val_episodes=1,
    )
    sample = dataset[0]
    rows = dataset.sampler.sample_source_rows(0)[:2]
    for key in dataset.sensor_modalities:
        columns = dataset.normalization_fields[key]
        np.testing.assert_array_equal(sample["obs"][key], dataset.episode_store.read(key, rows, columns=columns))
    np.testing.assert_array_equal(sample["contact_map"], dataset.episode_store.read("contact_map", rows))
    assert "contact_map" not in sample["obs"]
    validation = dataset.get_validation_dataset()
    assert validation.episode_store is dataset.episode_store and not validation.training


def check_validation_weighting(device):
    class MeanModel(torch.nn.Module):
        def forward(self, batch):
            value = batch["obs"]["point_cloud"].mean()
            return value, {"value": value}

    model = MeanModel().to(device).train()
    dataset = [{"obs": {"point_cloud": torch.tensor([value])}} for value in (1.0, 2.0, 9.0)]
    metrics = common.evaluate(model, dataset, 2, device, stop_requested=lambda: False)
    torch.testing.assert_close(metrics["val/value"], torch.tensor(4.0, device=device, dtype=torch.float64))
    assert model.training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", type=torch.device)
    args = parser.parse_args()
    torch.set_num_threads(2)
    check_tactile_configs()
    check_afro_directions(args.device)
    check_validation_weighting(args.device)
    print("PASS tactile configs and AFRO bidirectional loss/gradients", flush=True)
    with tempfile.TemporaryDirectory(prefix="dexmani-pretrain-checks-") as directory:
        root = Path(directory)
        check_contact_data(root)
        print("PASS serial/spawn preprocessing and failure cleanup", flush=True)
        check_pretrain_samples(root)
        print("PASS AFRO stride and CordViP source-row/contact alignment", flush=True)
        for method in ("afro", "cordvip"):
            check_train_only(root, method, args.device)
            check_training(root, method, args.device)
            print(f"PASS {method} train-only, resume, validation and export", flush=True)


if __name__ == "__main__":
    main()
