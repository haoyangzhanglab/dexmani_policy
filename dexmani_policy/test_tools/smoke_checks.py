"""Shared config, data, training and restoration checks."""

import json
import tempfile
import traceback
from contextlib import closing, contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import load_checkpoint, policy_state_dict, restore_policy
from dexmani_policy.common.config import (
    POINT_CLOUD_FIELDS, load_policy_config, positive_int,
)
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state, set_seed
from dexmani_policy.training.common.config import validate_config


def load_smoke_config(config_name: str, overrides=()):
    cfg = load_policy_config(config_name, overrides)
    # Hydra compose does not create a runtime output directory.
    cfg.workspace.output_dir = "/tmp/smoke_test_output"
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    return cfg


def iter_hydra_targets(node):
    if isinstance(node, dict):
        if "_target_" in node:
            yield node["_target_"]
        for value in node.values():
            yield from iter_hydra_targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_hydra_targets(value)


def validate_target_imports(cfg) -> int:
    """Resolve every Hydra target without constructing models or datasets."""
    targets = list(iter_hydra_targets(OmegaConf.to_container(cfg, resolve=True)))
    for target in targets:
        hydra.utils.get_method(target)
    return len(targets)


def validate_tensor_layout(value, shape, name, dtype=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name}: expected tensor shape {tuple(shape)}, got {getattr(value, 'shape', None)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {value.dtype}")


def iter_batch_fields(batch):
    for key, value in batch.items():
        if isinstance(value, dict):
            for field, tensor in value.items():
                yield f"{key}.{field}", tensor
        else:
            yield key, value


def observation_shape(dataset, field):
    shape = tuple(dataset.replay_buffer[field].shape[1:])
    if field == "rgb" and dataset.rgb_preprocess_size is not None:
        spatial_shape = dataset.rgb_random_crop_size or dataset.rgb_preprocess_size
        shape = (3, *spatial_shape)
    elif field in POINT_CLOUD_FIELDS:
        points = dataset.random_keep_points if field == "point_cloud" else None
        shape = (points or shape[0], dataset.pc_dim or shape[-1])
    return (dataset.obs_horizon, *shape)


def validate_data_batch(batch, dataset, cfg):
    action = batch["action"]
    batch_size = len(action)
    validate_tensor_layout(action, (batch_size, cfg.horizon, cfg.agent.action_dim), "action")

    obs = batch["obs"]
    if obs.keys() != set(dataset.sensor_modalities):
        raise ValueError("Batch observation fields differ from the dataset recipe")
    for key in dataset.sensor_modalities:
        shape = observation_shape(dataset, key)
        source_dtype = dataset.replay_buffer[key].dtype
        if key == "rgb" and dataset.rgb_preprocess_size is not None:
            dtype = torch.uint8 if dataset.rgb_keep_uint8 else torch.float32
        elif np.issubdtype(source_dtype, np.floating):
            dtype = torch.float32
        else:
            dtype = torch.from_numpy(np.empty((), dtype=source_dtype)).dtype
        validate_tensor_layout(obs[key], (batch_size, *shape), f"obs.{key}", dtype)

    context = batch.get("action_context", {})
    if not isinstance(context, dict):
        raise ValueError("Batch action_context must be a dict")
    for key, value in context.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or len(value) != batch_size:
            raise ValueError(f"action_context.{key}: expected batched tensor")
        if value.dtype != torch.float32:
            raise ValueError(f"action_context.{key}: expected float32")

    for name, value in iter_batch_fields(batch):
        if value.device.type != "cpu":
            raise ValueError(f"{name}: expected CPU data")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{name}: expected finite data")


def validate_data_loader(loader, cfg, *, split, max_batches, batch_check=validate_data_batch):
    from dexmani_policy.training.common.data import close_data_loader

    iterator = None
    checked = 0
    try:
        iterator = iter(loader)
        for checked, batch in enumerate(islice(iterator, max_batches), start=1):
            batch_check(batch, loader.dataset, cfg)
            summary = ", ".join(
                f"{key}={tuple(value.shape)} {value.dtype}"
                for key, value in iter_batch_fields(batch)
            )
            print(f"[{split} batch {checked}] {summary}")
        if checked == 0:
            raise ValueError(f"{split}: no batches to check")
    finally:
        close_resources(lambda: close_data_loader(loader, iterator))
    print(f"[{split}] windows={len(loader.dataset):,}, checked batches={checked}")


def validate_dataset(cfg, *, max_batches=2, batch_check=validate_data_batch):
    from dexmani_policy.training.common.data import build_dataset, build_train_loader

    positive_int(max_batches, "max_batches")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _ = build_dataset(cfg)
    dataset.get_normalizer(cfg.normalization)
    loader = build_train_loader(cfg, dataset)
    validate_data_loader(loader, cfg, split="train", max_batches=max_batches, batch_check=batch_check)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无可用验证窗口")
    else:
        loader = DataLoader(validation, batch_size=cfg.dataloader.batch_size, num_workers=0)
        validate_data_loader(loader, cfg, split="validation", max_batches=max_batches, batch_check=batch_check)


def validate_parameter_updates(model, before):
    changed = None
    for name, parameter in model.named_parameters():
        # Compilation inserts a wrapper into parameter names.
        name = name.replace("_orig_mod.", "")
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.isfinite(current).all():
            raise AssertionError(f"Non-finite learned parameter: {name}")
        if changed is None and not torch.equal(before[name], current):
            changed = name
    if changed is None:
        raise AssertionError("No trainable parameter changed within the smoke budget")
    return changed


def validate_checkpoint_state(cfg, trainer, checkpoint):
    assert checkpoint.global_step == trainer.global_step
    assert checkpoint.epoch == trainer.current_epoch
    assert trainer.scheduler.state_dict() == checkpoint.scheduler_state
    optimizer_state = trainer.optimizer.state_dict()
    assert optimizer_state["param_groups"] == checkpoint.optimizer_state["param_groups"]
    torch.testing.assert_close(
        optimizer_state["state"], checkpoint.optimizer_state["state"],
        rtol=0, atol=0, check_device=False,
    )
    assert (
        checkpoint.effective_config["training"]["loop"]["total_train_steps"]
        == cfg.training.loop.total_train_steps
    )
    if trainer.ema_updater is not None:
        assert checkpoint.ema_updater_step == trainer.ema_updater.optimization_step
    for model, state in (
        (trainer.model, checkpoint.model_state),
        (trainer.ema_model, checkpoint.ema_model_state),
    ):
        if model is not None:
            torch.testing.assert_close(
                policy_state_dict(model), state, rtol=0, atol=0, check_device=False
            )
            torch.testing.assert_close(
                model.normalizer.state_dict(),
                trainer.train_loader.dataset.get_normalizer(cfg.normalization).state_dict(),
                rtol=0, atol=0, check_device=False,
            )


def capture_policy_predictions(model, ema_model, batch):
    """Capture CPU inputs/predictions while preserving model modes, backbones and RNG."""
    device = next(model.parameters()).device
    obs = dict_apply(
        {key: value[:1] for key, value in batch["obs"].items()},
        lambda x: x.detach().to("cpu", copy=True),
    )
    prediction_kwargs = {}
    if "action_context" in batch:
        prediction_kwargs["action_context"] = dict_apply(
            batch["action_context"], lambda value: value[:1].detach().to("cpu", copy=True),
        )
    inputs = dict_apply(obs, lambda x: x.to(device))
    kwargs = dict_apply(prediction_kwargs, lambda x: x.to(device))
    rng_state = get_rng_state(device)
    predictions = {}
    reports: dict[str, dict[str, float]] = {}
    for use_ema, policy in ((False, model), (True, ema_model)):
        if policy is None:
            continue
        label = "ema" if use_ema else "raw"
        reports[label] = {}
        modes = [(module, module.training) for module in policy.modules()]
        backbones = [
            (parent, name, child)
            for parent in policy.modules() for name, child in parent.named_children()
            if hasattr(child, "_orig_mod")
        ]
        try:
            policy.eval()
            set_rng_state(rng_state, device)
            with torch.inference_mode():
                prediction = policy.predict_action(inputs, **kwargs)["future_actions"]
            assert prediction.shape == (1, policy.future_steps, policy.control_action_dim)
            assert torch.isfinite(prediction).all()
            if backbones:
                # Compare restoration using eager kernels for every compiled child.
                for parent, name, backbone in backbones:
                    setattr(parent, name, backbone._orig_mod)
                set_rng_state(rng_state, device)
                with torch.inference_mode():
                    eager = policy.predict_action(inputs, **kwargs)["future_actions"]
                difference = (prediction - eager).abs().max().item()
                reports[label]["compiled_eager_max_abs_diff"] = difference
                print(f"{label} compiled/eager prediction max abs diff: {difference:.8g}")
                prediction = eager
            predictions[use_ema] = prediction.to("cpu", copy=True)
        finally:
            for parent, name, backbone in backbones:
                setattr(parent, name, backbone)
            for module, training in modes:
                module.training = training
            set_rng_state(rng_state, device)
    return dict(
        obs=obs, prediction_kwargs=prediction_kwargs, rng_state=rng_state,
        predictions=predictions, reports=reports,
    )


def validate_policy_restoration(checkpoint_path, snapshot, *, device):
    """Check restored raw/EMA policies without owning or modifying a Trainer."""
    obs = dict_apply(snapshot["obs"], lambda x: x.to(device))
    kwargs = dict_apply(snapshot["prediction_kwargs"], lambda x: x.to(device))
    reports = {name: dict(values) for name, values in snapshot["reports"].items()}
    rng_state = get_rng_state(device)
    try:
        for use_ema, expected in snapshot["predictions"].items():
            restored = restore_policy(checkpoint_path, use_ema=use_ema, device=device)
            set_rng_state(snapshot["rng_state"], device)
            with torch.inference_mode():
                actual = restored.predict_action(obs, **kwargs)["future_actions"].cpu()
            torch.testing.assert_close(actual, expected)
            reports["ema" if use_ema else "raw"]["eager_restore_max_abs_diff"] = (
                actual - expected
            ).abs().max().item()
            del restored
    finally:
        set_rng_state(rng_state, device)
    return reports


@contextmanager
def smoke_output(output_dir):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as temporary:
            yield Path(temporary)
    else:
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        yield directory


def validate_training(
    cfg, *, max_updates=4, resume_updates=2, rollout=False, output_dir=None,
    batch_check=validate_data_batch, deployment_check=None,
):
    """Run the same bounded training/restoration checks for every policy."""
    from dexmani_policy.training.common.data import build_dataset
    from dexmani_policy.training.trainer import PolicyTrainer

    positive_int(max_updates, "max_updates")
    if resume_updates < 0:
        raise ValueError("resume_updates must be nonnegative")
    device = torch.device(cfg.training.device)
    set_seed(cfg.training.seed)
    with smoke_output(output_dir) as directory:
        report: dict[str, Any] = {"device": str(device), "training": []}
        try:
            checkpoint = None
            source = None
            run_cfg = OmegaConf.create(cfg)
            phases = [("train", max_updates)]
            if resume_updates:
                phases.append(("resume", resume_updates))
            for phase, budget in phases:
                if checkpoint is not None:
                    total_steps = checkpoint.effective_config["training"]["loop"]["total_train_steps"]
                    if checkpoint.global_step == total_steps:
                        report["resume"] = {"status": "not_run", "reason": "training budget exhausted"}
                        break
                    run_cfg = OmegaConf.create(checkpoint.effective_config)
                with open_dict(run_cfg):
                    run_cfg.workspace.output_dir = str(directory / phase)
                    run_cfg.workspace.wandb_cfg = None
                    if source is not None:
                        run_cfg.resume_from = str(source)
                dataset, description = build_dataset(run_cfg)
                with (
                    patch.object(LinearNormalizer, "fit_field", side_effect=AssertionError("resume must not fit"))
                    if checkpoint is not None else nullcontext()
                ), closing(PolicyTrainer(run_cfg, dataset, description, checkpoint=checkpoint)) as trainer:
                    if checkpoint is not None:
                        validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                    checkpoint = None
                    start_step = trainer.global_step
                    before = {
                        name: parameter.detach().to(device="cpu", copy=True)
                        for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
                    }
                    trainer.compile_models()
                    trainer.train(max_updates=budget)
                    assert trainer.global_step == min(start_step + budget, trainer.total_train_steps)
                    changed = validate_parameter_updates(trainer.model, before)
                    del before
                    source = (trainer.checkpoint_dir / "latest.pt").resolve()
                    checkpoint = load_checkpoint(source, mmap=True)
                    validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
                    batch_check(batch, dataset, trainer.cfg)
                    snapshot = capture_policy_predictions(trainer.model, trainer.ema_model, batch)
                    validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                del trainer
                experiment = source.parent.parent
                restored = validate_policy_restoration(
                    source, snapshot, device=device,
                )
                report["training"].append({
                    "phase": phase, "updates": checkpoint.global_step - start_step,
                    "global_step": checkpoint.global_step, "updated_parameter": changed,
                    "restoration": restored,
                })
                print(f"[{phase}] step={checkpoint.global_step}; updated {changed}", flush=True)
            if deployment_check is not None:
                report["deployment"] = deployment_check(experiment, dataset, device)
            if rollout:
                report["simulation"] = validate_simulation(experiment, device)
        except BaseException:
            report["error"] = traceback.format_exc()
            raise
        finally:
            close_resources(lambda: (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n"))
        if output_dir is not None:
            print(f"Smoke report: {directory / 'report.json'}", flush=True)
        return report


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def validate_simulation(experiment, device):
    from dexmani_policy.evaluation.config import load_eval_config
    from dexmani_policy.evaluation.rollout import evaluate_policy

    request = load_eval_config(
        experiment, "latest", OmegaConf.from_dotlist([
            "eval.episodes=1", "eval.video.enabled=false", f"training.device={device}",
        ]),
    )
    record, path = evaluate_policy(request, output_root=experiment / "eval_dexsim")
    return {"status": "passed", "statistics": record["statistics"],
            "episodes": record["episodes"], "artifacts": str(path)}
