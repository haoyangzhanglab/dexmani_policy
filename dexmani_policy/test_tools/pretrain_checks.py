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
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.pretrain.obs import common
from dexmani_policy.training.pretrain.obs.afro import AFRO, vicreg_loss


def check_run_config(root):
    for value in (None, True, False):
        for disabled in (False, True):
            overrides = [] if value is None else [f"+workspace.wandb_enabled={str(value).lower()}"]
            flags = ["--no-wandb"] if disabled else []
            cfg, _, _ = common.load_run("afro", [*flags, *overrides])
            expected = False if disabled else value
            assert cfg.workspace.get("wandb_enabled") == expected
            path = root / "config_resume.pt"
            base, _, _ = common.load_run("afro", overrides)
            torch.save({"effective_config": OmegaConf.to_container(base, resolve=True)}, path)
            restored, _, _ = common.load_run("afro", ["--resume-from", str(path), *flags])
            assert restored.workspace.get("wandb_enabled") == expected


def check_cordvip_branches(device):
    from dexmani_policy.training.pretrain.obs.cordvip import CordViP
    modes = dict(hand_pointcloud="identity", object_pointcloud="identity", joint_state="identity")
    model = CordViP(modes, n_obs_steps=2).to(device)
    batch = {"obs": {"hand_pointcloud": torch.randn(2, 2, 8, 3, device=device),
                     "object_pointcloud": torch.randn(2, 2, 1024, 3, device=device),
                     "joint_state": torch.randn(2, 2, 19, device=device)},
             "contact_map": torch.rand(2, 2, 1024, device=device)}
    loss, metrics = model(batch)
    loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    robot, objects, hand, arm = model.encoder(batch["obs"])
    contact_loss = torch.nn.functional.mse_loss(model.contact_head(torch.cat((robot, objects), -1)), batch["contact_map"])
    arm_loss = torch.nn.functional.mse_loss(
        model.arm_head(torch.cat((robot.flatten(1), objects.flatten(1), hand.flatten(1)), 1)),
        batch["obs"]["joint_state"][:, 1, :7],
    )
    hand_loss = torch.nn.functional.mse_loss(
        model.hand_head(torch.cat((robot.flatten(1), objects.flatten(1), arm.flatten(1)), 1)),
        batch["obs"]["joint_state"][:, 1, 7:],
    )
    expected = arm_loss + hand_loss + contact_loss
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name])
    del gradients, loss, expected, robot, objects, hand, arm, contact_loss, arm_loss, hand_loss
    for branch in ("contact", "arm", "hand"):
        setattr(model, f"{branch}_loss_weight", 0.0)
        head = getattr(model, f"{branch}_head")
        saved = {name: p.detach().clone() for name, p in head.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        optimizer.zero_grad(set_to_none=True)
        inputs = {"obs": batch["obs"]} if branch == "contact" else batch
        with patch.object(head, "forward", side_effect=AssertionError("Disabled head executed")):
            loss, metrics = model(inputs)
            if branch == "contact":
                bad = {**inputs, "contact_map": torch.full_like(batch["contact_map"], float("nan"))}
                torch.testing.assert_close(model(bad)[0], loss)
        assert branch not in metrics and torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        assert all(p.grad is None for p in head.parameters())
        optimizer.step()
        for name, p in head.named_parameters():
            torch.testing.assert_close(p, saved[name], rtol=0, atol=0)
        setattr(model, f"{branch}_loss_weight", 1.0)
    for weights in ((0, 0, 0), (-1, 1, 1), (1, float("nan"), 1), (1, 1, float("inf"))):
        try:
            CordViP(modes, contact_loss_weight=weights[0], arm_loss_weight=weights[1], hand_loss_weight=weights[2])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid weights accepted")


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
    assert all(p.grad is None for p in model.teacher.parameters())
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


def check_training(root, method, device=torch.device("cpu")):
    source = root / ("source.zarr" if method == "afro" else "cordvip.zarr")
    overrides = [
        f"zarr_path={source}", "dataset.horizon=3", "dataset.pad_before=0", "dataset.pad_after=0",
        "dataset.val_episodes=1", "dataloader.batch_size=4", "dataloader.num_workers=0",
        "dataloader.pin_memory=false", f"training.device={device}", "training.lr_warmup_steps=2",
        "training.val_every_epochs=1", "training.loop.log_interval_steps=1",
    ]
    if method == "afro":
        overrides += ["dataset.frame_stride=1", "training.num_epochs=2", "training.save_every_epochs=1"]
        updates = 4
    else:
        overrides += ["n_obs_steps=2", "training.loop.total_train_steps=4",
                      "training.save_every_updates=1", "training.loop.gradient_accumulation_steps=2"]
        updates = 3
    used_lrs = []
    original_step = torch.optim.AdamW.step
    def recording_step(optimizer, *args, **kwargs):
        used_lrs.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)
    for name, count in (("full", updates), ("resumed", 1)):
        with patch.object(torch.optim.AdamW, "step", recording_step):
            common.train(method, argv=["--output-dir", str(root / method / name),
                                       "--max-updates", str(count), "+workspace.wandb_enabled=false", *overrides])
    full_records = [json.loads(line) for line in (root / method / "full/metrics.jsonl").read_text().splitlines()]
    logged_lrs = [row["train/lr"] for row in full_records if "train/lr" in row]
    assert logged_lrs == used_lrs[:updates] and len(set(logged_lrs)) > 1
    directory = root / method / "resumed"
    cfg = OmegaConf.create(torch.load(
        directory / "resume.pt", map_location="cpu", weights_only=False,
    )["effective_config"])
    save_config = common.ExperimentLogger.save_config

    def initialize_logging(logger, config):
        save_config(logger, config)
        torch.rand(3, device=device)

    with patch.object(common, "fit_normalizer", side_effect=AssertionError("Refit on resume")), \
         patch.object(common.ExperimentLogger, "save_config", initialize_logging):
        common.train(method, argv=["--resume-from", str(directory),
                                   "--max-updates", str(updates - 1)])
    full = torch.load(root / method / "full/resume.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "scheduler_state", "global_step"):
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    torch.testing.assert_close(full["optimizer_state"]["state"], resumed["optimizer_state"]["state"], rtol=0, atol=0)
    assert full["optimizer_state"]["param_groups"] == resumed["optimizer_state"]["param_groups"]
    for key in ("torch", "torch_cuda"):
        torch.testing.assert_close(full["rng_state"][key], resumed["rng_state"][key], rtol=0, atol=0)
    assert full["global_step"] == updates
    assert resumed["effective_config"]["workspace"]["wandb_enabled"] is False
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert any("val/loss" in record for record in records)
    assert [row["train/lr"] for row in records if "train/lr" in row] == logged_lrs
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

    common.train(method, argv=["--resume-from", str(directory), "--no-wandb"])
    completed = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    assert completed["global_step"] == cfg.training.loop.total_train_steps
    artifacts = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()}
    with patch.object(common.hydra.utils, "instantiate", side_effect=AssertionError("Built dataset")), \
         patch.object(common, "ExperimentLogger", side_effect=AssertionError("Built logger")), \
         patch.object(AFRO, "__init__", side_effect=AssertionError("Built AFRO")), \
         patch.object(CordViP, "__init__", side_effect=AssertionError("Built CordViP")):
        assert common.train(method, argv=["--resume-from", str(directory), "--no-wandb"]) is None
    assert {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in directory.iterdir()} == artifacts

    failed_directory = root / method / "nonfinite"
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
            common.train(method, argv=["--output-dir", str(failed_directory),
                                       "--max-updates", "1", "--no-wandb", *overrides])
        except RuntimeError as error:
            assert "Nonfinite" in str(error)
        else:
            raise AssertionError("Nonfinite accumulation group was updated")
    assert failed_model is not None
    for name, parameter in failed_model.named_parameters():
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0, atol=0)
    assert not (failed_directory / "resume.pt").exists()
    assert not (failed_directory / "encoder.pt").exists()


def check_train_only(root, method, device=torch.device("cpu")):
    source = root / ("source.zarr" if method == "afro" else "cordvip.zarr")
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
    argv = ["--output-dir", cfg.workspace.output_dir, "--max-updates", "1", "--no-wandb", *overrides]
    try:
        common.train(method, argv=[*argv, f"~normalization.{next(iter(cfg.normalization))}"])
    except ValueError as error:
        assert "Normalization must declare" in str(error)
    else:
        raise AssertionError("Fresh pretraining accepted missing normalization modes")
    if method == "afro":
        with patch.object(common.hydra.utils, "instantiate", side_effect=AssertionError("Built dataset")):
            try:
                common.train(method, argv=[*argv, "training.loop.gradient_accumulation_steps=2"])
            except ValueError as error:
                assert "AFRO VICReg requires accumulation=1" in str(error)
            else:
                raise AssertionError("AFRO accepted gradient accumulation")

    failed_loggers = []
    def fail_config(logger, config):
        failed_loggers.append(logger)
        raise RuntimeError("config logging failure")

    with patch.object(common.ExperimentLogger, "save_config", fail_config):
        try:
            common.train(method, argv=["--output-dir", str(root / method / "failed_logging"),
                                       "--max-updates", "1", "--no-wandb", *overrides])
        except RuntimeError as error:
            assert str(error) == "config logging failure"
        else:
            raise AssertionError("Config logging failure was swallowed")
    assert len(failed_loggers) == 1 and failed_loggers[0].metrics_file.closed

    with (
        patch.object(common, "DataLoader", side_effect=AssertionError("Built val loader")),
        patch.object(ObservationPretrainDataset, "iter_normalization_data", side_effect=AssertionError("Read identity stats")),
        patch.object(common, "evaluate", side_effect=AssertionError("Evaluated without val")),
    ):
        common.train(method, argv=argv)
    package = torch.load(Path(cfg.workspace.output_dir) / "encoder.pt", weights_only=True)
    assert package["normalizer_state"] == {}
    assert all(mode == "identity" for mode in package["normalization"].values())
    records = [json.loads(line) for line in (Path(cfg.workspace.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert all(not any(key.startswith("val/") for key in record) for record in records)
    try:
        common.train(method, argv=[*argv, f"zarr_path={root / 'missing.zarr'}"])
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
        root / "cordvip.zarr", horizon=5, obs_horizon=2, pad_before=0, pad_after=0, val_episodes=1,
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
    torch.set_num_threads(2)
    check_afro_directions(args.device)
    check_cordvip_branches(args.device)
    check_validation_weighting(args.device)
    print("PASS AFRO loss/gradients, CordViP branches and validation weighting", flush=True)
    with tempfile.TemporaryDirectory(prefix="dexmani-pretrain-checks-") as directory:
        root = Path(directory)
        check_run_config(root)
        rng = np.random.default_rng(7)
        for name in ("source.zarr", "cordvip.zarr"):
            group = zarr.open_group(str(root / name), mode="w")
            group.create_dataset("meta/episode_ends", data=np.array([8, 16, 24]))
            for key, shape in (("point_cloud", (24, 8, 3)), ("hand_pointcloud", (24, 8, 3)),
                               ("object_pointcloud", (24, 1024, 3)), ("joint_state", (24, 19))):
                group.create_dataset(f"data/{key}", data=rng.normal(size=shape).astype(np.float32))
            labels = np.arange(24 * 1024, dtype=np.float32).reshape(24, 1024) / (24 * 1024)
            group.create_dataset("data/contact_map", data=labels)
        check_pretrain_samples(root)
        print("PASS AFRO stride and CordViP source-row/contact alignment", flush=True)
        for method in ("afro", "cordvip"):
            check_train_only(root, method, args.device)
            check_training(root, method, args.device)
            print(f"PASS {method} train-only, resume, validation and export", flush=True)


if __name__ == "__main__":
    main()
