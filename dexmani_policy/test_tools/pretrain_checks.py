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
from omegaconf import OmegaConf, open_dict

from dexmani_policy.agents.policies.afro import AFROEncoder
from dexmani_policy.agents.policies.cordvip import CordViPEncoder
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.preprocess.contact_map import compute_contact_map, generate
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
        assert cfg.agent.tactile_source == source
        assert "tactile_source" not in cfg.dataset
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
    source.attrs.update(dt=0.02, hand_pointcloud_config={"num_points": 8},
                        source_id="unused", format="unused", task_name="unused")
    rng = np.random.default_rng(7)
    for key, shape in (
        ("point_cloud", (24, 8, 3)), ("hand_pointcloud", (24, 8, 6)),
        ("object_pointcloud", (24, 1024, 6)), ("joint_state", (24, 19)),
        ("action", (24, 19)),
    ):
        array = source.create_dataset(f"data/{key}", data=rng.normal(size=shape))
        array.attrs["unused"] = True
    original = {key: source["data"][key][:] for key in source["data"]}
    # Warm Open3D before starting workers to catch unsafe fork-based execution.
    generate(root / "source.zarr", root / "serial.zarr", num_workers=1)
    generate(root / "source.zarr", root / "parallel.zarr", num_workers=2)
    serial = zarr.open_group(str(root / "serial.zarr"), mode="r")
    parallel = zarr.open_group(str(root / "parallel.zarr"), mode="r")
    for key, values in original.items():
        np.testing.assert_array_equal(source["data"][key][:], values)
    assert dict(serial.attrs) == {"dt": 0.02, "hand_pointcloud_config": {"num_points": 8}}
    assert set(serial["data"]) == {"hand_pointcloud", "object_pointcloud", "joint_state", "action", "contact_map"}
    for key in serial["data"]:
        actual = serial["data"][key][:]
        assert actual.dtype == np.float32 and not dict(serial["data"][key].attrs)
        np.testing.assert_allclose(actual, parallel["data"][key][:], rtol=1e-6, atol=1e-7)
        if key != "contact_map":
            values = source["data"][key][:]
            if "pointcloud" in key:
                assert actual.shape[-1] == 3 and values.shape[-1] == 6
                values = values[..., :3]
            np.testing.assert_array_equal(values.astype(np.float32), actual)
    np.testing.assert_array_equal(source["meta/episode_ends"][:], parallel["meta/episode_ends"][:])
    labels = parallel["data/contact_map"][:]
    assert labels.dtype == np.float32 and np.isfinite(labels).all()
    assert (labels >= 0).all() and (labels <= 1).all()
    policy_data = DemonstrationDataset(
        root / "serial.zarr", sensor_modalities=list(CordViPPretrainDataset.sensor_modalities),
        horizon=12, obs_horizon=4, pad_before=3, pad_after=5, pc_dim=3,
    )
    rows = policy_data.sampler.sample_source_rows(0)
    np.testing.assert_array_equal(policy_data[0]["action"], original["action"][rows].astype(np.float32))
    original_action = source["data/action"][:]
    del source["data/action"]
    generate(root / "source.zarr", root / "without_action.zarr")
    assert "action" not in zarr.open_group(str(root / "without_action.zarr"), mode="r")["data"]
    assert len(CordViPPretrainDataset(root / "without_action.zarr")) > 0
    try:
        DemonstrationDataset(root / "without_action.zarr", sensor_modalities=["joint_state"], obs_horizon=1)
    except KeyError:
        pass
    else:
        raise AssertionError("Policy dataset accepted missing action")
    source.create_dataset("data/action", data=original_action)
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


def check_contact_formula():
    rng = np.random.default_rng(23)
    objects, hand = rng.normal(size=(17, 3)), rng.normal(size=(9, 3))
    normals = rng.normal(size=(17, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    expected = []
    for point, normal in zip(objects, normals):
        difference = hand - point
        distance = np.linalg.norm(difference, axis=-1)
        alignment = np.abs(difference @ normal) / (distance + 1e-8)
        minimum: np.float64 = np.min(distance * np.exp(1 - alignment))
        expected.append(1 - 2 * (1 / (1 + np.exp(-10 * minimum)) - 0.5))
    for block in (1, 7, 128):
        for sign in (1, -1):
            actual = compute_contact_map(objects, sign * normals, hand, point_chunk_size=block)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


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
        common.train(method, *common.prepare_run(method, cfg, max_updates=count))
    directory = root / method / "resumed"
    checkpoint = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["effective_config"])
    save_config = common.ExperimentLogger.save_config

    def initialize_logging(logger, config):
        save_config(logger, config)
        torch.rand(3, device=device)

    with patch.object(common, "fit_normalizer", side_effect=AssertionError("Refit on resume")), \
         patch.object(common.ExperimentLogger, "save_config", initialize_logging):
        prepared = common.prepare_run(method, cfg, checkpoint, max_updates=updates - 1)
        del checkpoint
        common.train(method, *prepared)
    full = torch.load(root / method / "full/resume.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "scheduler_state", "global_step"):
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    torch.testing.assert_close(full["optimizer_state"]["state"], resumed["optimizer_state"]["state"], rtol=0, atol=0)
    assert full["optimizer_state"]["param_groups"] == resumed["optimizer_state"]["param_groups"]
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
    names, parameters = zip(*((name, p) for name, p in model.named_parameters() if p.requires_grad))
    reordered = torch.optim.AdamW(
        [dict(params=parameters[::-1], param_names=list(names[::-1]))], **cfg.optimizer,
    )
    try:
        common.restore_run(model, reordered, None, resumed)
    except ValueError as error:
        assert "Optimizer parameter order" in str(error)
    else:
        raise AssertionError("Pretraining restored optimizer state into reordered parameters")
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

    common.train(method, *common.prepare_run(
        method, cfg, torch.load(directory / "resume.pt", map_location="cpu", weights_only=False),
    ))
    completed = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    assert completed["global_step"] == cfg.training.loop.total_train_steps
    artifacts = {path: path.stat().st_mtime_ns for path in directory.iterdir()}
    assert common.prepare_run(method, OmegaConf.create(completed["effective_config"]), completed) is None
    assert {path: path.stat().st_mtime_ns for path in directory.iterdir()} == artifacts

    failed_cfg = copy.deepcopy(base)
    failed_cfg.workspace.output_dir = str(root / method / "nonfinite")
    model_class = AFRO if method == "afro" else CordViP
    forward = model_class.forward
    failed_model = None
    initial_parameters = {}

    def nonfinite_first_batch(model, batch):
        nonlocal failed_model
        first_batch = failed_model is None
        if first_batch:
            failed_model = model
            initial_parameters.update({name: p.detach().clone() for name, p in model.named_parameters()})
        loss, metrics = forward(model, batch)
        return loss * (float("nan") if first_batch else 1.0), metrics

    with patch.object(model_class, "forward", nonfinite_first_batch):
        try:
            common.train(method, *common.prepare_run(method, failed_cfg, max_updates=1))
        except RuntimeError as error:
            assert "Nonfinite" in str(error)
        else:
            raise AssertionError("Nonfinite accumulation group was updated")
    assert failed_model is not None
    for name, parameter in failed_model.named_parameters():
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0, atol=0)
    assert not (Path(failed_cfg.workspace.output_dir) / "resume.pt").exists()
    assert not (Path(failed_cfg.workspace.output_dir) / "encoder.pt").exists()


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
    incomplete = copy.deepcopy(cfg)
    with open_dict(incomplete.normalization):
        del incomplete.normalization[next(iter(incomplete.normalization))]
    try:
        common.prepare_run(method, incomplete, max_updates=1)
    except ValueError as error:
        assert "Normalization must declare" in str(error)
    else:
        raise AssertionError("Fresh pretraining accepted missing normalization modes")
    with (
        patch.object(common, "DataLoader", side_effect=AssertionError("Built val loader")),
        patch.object(ObservationPretrainDataset, "iter_normalization_data", side_effect=AssertionError("Read identity stats")),
        patch.object(common, "evaluate", side_effect=AssertionError("Evaluated without val")),
    ):
        common.train(method, *common.prepare_run(method, cfg, max_updates=1))
    package = torch.load(Path(cfg.workspace.output_dir) / "encoder.pt", weights_only=True)
    assert package["normalizer_state"] == {}
    assert all(mode == "identity" for mode in package["normalization"].values())
    records = [json.loads(line) for line in (Path(cfg.workspace.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert all(not any(key.startswith("val/") for key in record) for record in records)
    cfg.dataset.zarr_path = str(root / "missing.zarr")
    try:
        common.prepare_run(method, cfg, max_updates=1)
    except FileExistsError as error:
        assert "Pretraining output already exists" in str(error)
    else:
        raise AssertionError("Fresh pretraining reused an existing output")


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

    cordvip_dataset = CordViPPretrainDataset(
        root / "parallel.zarr", horizon=5, obs_horizon=2, pad_before=0, pad_after=0, val_episodes=1,
    )
    sample = cordvip_dataset[0]
    rows = cordvip_dataset.sampler.sample_source_rows(0)[:2]
    for key in cordvip_dataset.sensor_modalities:
        columns = cordvip_dataset.normalization_fields[key]
        np.testing.assert_array_equal(sample["obs"][key], cordvip_dataset.episode_store.read(key, rows, columns=columns))
    np.testing.assert_array_equal(sample["contact_map"], cordvip_dataset.episode_store.read("contact_map", rows))
    assert "contact_map" not in sample["obs"]
    validation = cordvip_dataset.get_validation_dataset()
    assert validation.episode_store is cordvip_dataset.episode_store


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
    check_contact_formula()
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
