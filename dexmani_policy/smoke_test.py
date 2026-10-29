"""Validate configs/data or exercise training and checkpoint restoration."""

import argparse
import hydra
import importlib
import importlib.util
import json
import tempfile
import time
import traceback
from contextlib import ExitStack, closing, contextmanager
from itertools import islice
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, default_collate

from dexmani_policy.common.checkpoint import (
    load_checkpoint,
    policy_state_dict,
    restore_policy,
)
from dexmani_policy.common.config import (
    POINT_CLOUD_FIELDS,
    load_experiment_config,
    load_policy_config,
    positive_int,
    set_project_root,
)
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state, set_seed
from dexmani_policy.deployment.metadata import inspect_policy
from dexmani_policy.deployment.runtime import load_policy
from dexmani_policy.training.common.config import validate_config
from dexmani_policy.training.common.data import (
    build_dataset_and_normalizer,
    build_train_loader,
    close_data_loader,
)
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.trainer import PolicyTrainer


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
    """Import training targets; only locate optional simulation modules."""
    targets = list(iter_hydra_targets(OmegaConf.to_container(cfg, resolve=True)))
    for target in targets:
        module_name, attribute = target.rsplit(".", 1)
        if target == cfg.agent._target_ or module_name.startswith(
            ("dexmani_policy.training.", "dexmani_policy.datasets.")
        ):
            getattr(importlib.import_module(module_name), attribute)
        elif importlib.util.find_spec(module_name) is None:
            raise ModuleNotFoundError(f"Hydra target module not found: {target}")
    return len(targets)


def validate_tensor_layout(value, shape, name, dtype=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name}: expected tensor shape {tuple(shape)}, got {getattr(value, 'shape', None)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {value.dtype}")


def iter_batch_fields(batch):
    yield "action", batch["action"]
    if "future_force" in batch:
        yield "future_force", batch["future_force"]
        yield "future_force_valid", batch["future_force_valid"]
    for section in ("obs", "action_context"):
        for key, value in batch.get(section, {}).items():
            yield f"{section}.{key}", value


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

    if "future_force" in batch:
        validate_tensor_layout(
            batch["future_force"],
            (batch_size, cfg.horizon - cfg.n_obs_steps + 1, *obs["finger_tactile_force"].shape[-2:]),
            "future_force", torch.float32,
        )

        validate_tensor_layout(batch["future_force_valid"], batch["future_force"].shape[:2], "future_force_valid", torch.bool)
        assert batch["future_force_valid"].any(dim=1).all()

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


def validate_data_loader(loader, cfg, *, split, max_batches):
    iterator = None
    checked = 0
    try:
        iterator = iter(loader)
        for checked, batch in enumerate(islice(iterator, max_batches), start=1):
            validate_data_batch(batch, loader.dataset, cfg)
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


def validate_dataset(cfg, *, max_batches=2):
    positive_int(max_batches, "max_batches")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _, _ = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset)
    validate_data_loader(loader, cfg, split="train", max_batches=max_batches)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无可用验证窗口")
    else:
        loader = DataLoader(validation, batch_size=cfg.dataloader.batch_size, num_workers=0)
        validate_data_loader(loader, cfg, split="validation", max_batches=max_batches)


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


def validate_policy_restoration(saved_config, checkpoint_path, snapshot, *, device):
    """Check restored raw/EMA policies without owning or modifying a Trainer."""
    obs = dict_apply(snapshot["obs"], lambda x: x.to(device))
    kwargs = dict_apply(snapshot["prediction_kwargs"], lambda x: x.to(device))
    reports = {name: dict(values) for name, values in snapshot["reports"].items()}
    rng_state = get_rng_state(device)
    try:
        for use_ema, expected in snapshot["predictions"].items():
            restored = restore_policy(saved_config, checkpoint_path, use_ema=use_ema, device=device)
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


def validate_training(cfg, *, max_updates=4, resume_updates=2, rollout=False, output_dir=None):
    """Run the same bounded training/restoration checks for every policy."""
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
                dataset, normalizer, description = build_dataset_and_normalizer(
                    run_cfg, resume_checkpoint=checkpoint,
                )
                with closing(PolicyTrainer(
                    run_cfg, dataset, normalizer, description, checkpoint=checkpoint,
                )) as trainer:
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
                    validate_data_batch(batch, dataset, trainer.cfg)
                    snapshot = capture_policy_predictions(trainer.model, trainer.ema_model, batch)
                    validate_checkpoint_state(trainer.cfg, trainer, checkpoint)
                del trainer
                experiment = source.parent.parent
                restored = validate_policy_restoration(
                    load_experiment_config(experiment), source, snapshot, device=device,
                )
                report["training"].append({
                    "phase": phase, "updates": checkpoint.global_step - start_step,
                    "global_step": checkpoint.global_step, "updated_parameter": changed,
                    "restoration": restored,
                })
                print(f"[{phase}] step={checkpoint.global_step}; updated {changed}", flush=True)
            if cfg.policy_name == "adaptac":
                report["deployment"] = validate_adaptac_deployment(experiment, dataset, device)
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


ADAPTAC_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def tactile_smoke_attrs(source, task_name="pick_apple_messy"):
    attrs = {"task_name": task_name, "dt": 0.0625}
    if source == "sim":
        attrs.update(physics_dt=0.004166666666666667, frame_skip=15, action_dim=19,
                     action_space="joint", obs_alignment="obs[t]_before_action[t]", robot_setup_json="{}")
    else:
        attrs.update(format="dexmani.real.canonical", pointcloud_config={"num_points": 32})
    return attrs


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def validate_tactile_numerics():
    from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandPatchEncoder
    from dexmani_policy.common.config import normalization_feature_shapes
    from dexmani_policy.datasets.common.validation import validate_tactile_source

    generator = torch.Generator().manual_seed(24)
    encoder = XHandPatchEncoder().eval()
    for shape in ((5, 3), (5, 120, 3)):
        x = torch.randn((7, *shape), generator=generator) * 7 - 2
        for mode in ("rms", "max_abs", "limits", "gaussian"):
            full, chunked = LinearNormalizer(), LinearNormalizer()
            full.fit_field("x", [x], mode=mode, last_n_dims=len(shape))
            chunked.fit_field("x", [x[:0], *x.split(2)], mode=mode, last_n_dims=len(shape))
            torch.testing.assert_close(full.state_dict(), chunked.state_dict())
            field = full["x"]
            assert field.scale.shape == shape and field.scale.dtype == torch.float32
            if mode == "gaussian":
                torch.testing.assert_close(field.scale, x.std(dim=0).reciprocal())
            elif mode == "limits":
                torch.testing.assert_close(field.scale, 2 / (x.amax(dim=0) - x.amin(dim=0)))
            z = field.normalize(x)
            torch.testing.assert_close(field.unnormalize(z), x, atol=3e-6, rtol=1e-5)
            if mode in {"rms", "max_abs"}:
                assert not field.offset.any()
                torch.testing.assert_close(field.normalize(2 * x), 2 * z)
                assert field.normalize(x * 100).abs().max() > 1
                rescaled = LinearNormalizer()
                rescaled.fit_field("x", [x * 100], mode=mode, last_n_dims=len(shape))
                z2 = rescaled["x"].normalize(x * 100)
                torch.testing.assert_close(z, z2)
                if mode == "rms":
                    torch.testing.assert_close(z.square().mean(), torch.tensor(1.0))
                if len(shape) == 3:
                    torch.testing.assert_close(encoder._patch_descriptors(z), encoder._patch_descriptors(z2))
        for mode in ("rms", "max_abs"):
            for value in (0.0, -3.0, 4.0):
                n = LinearNormalizer()
                x = torch.full((2, *shape), value)
                n.fit_field("x", [x], mode=mode, last_n_dims=len(shape))
                torch.testing.assert_close(n["x"].normalize(x), torch.full_like(x, 0 if value == 0 else value / abs(value)))
            for invalid in ([], [torch.full((1, *shape), float("nan"))],
                            [torch.full((1, *shape), float("inf"))], [torch.full((1, *shape), 1e-40)]):
                expect_value_error(lambda: LinearNormalizer().fit_field("x", invalid, mode=mode, last_n_dims=len(shape)))
    zero = torch.zeros(2, 5, 120, 3)
    descriptors = encoder._patch_descriptors(zero)
    assert descriptors.shape == (2, 5, 5, 8) and not descriptors.any()
    assert torch.isfinite(encoder(zero)).all()
    ones = torch.ones_like(zero)
    d = encoder._patch_descriptors(ones)
    torch.testing.assert_close(d[..., :6], torch.ones_like(d[..., :6]))
    torch.testing.assert_close(d[..., 6], torch.full_like(d[..., 6], 3 ** .5 / (3 ** .5 + 1)))
    labels = (torch.tensor(encoder.effective_config["region_ids"]) + 1) % encoder.num_regions
    custom = XHandPatchEncoder(region_ids=labels, activity_scale=2.0).eval()
    restored = hydra.utils.instantiate(custom.effective_config).eval()
    assert "patch_mask" not in custom.state_dict()
    restored.load_state_dict(custom.state_dict(), strict=True)
    torch.testing.assert_close(restored.patch_mask, custom.patch_mask)
    torch.testing.assert_close(restored(z), custom(z))
    for value in (0, -1, float("nan"), float("inf"), 1e-50, 1e40):
        expect_value_error(lambda: XHandPatchEncoder(activity_scale=value))
    for labels in (torch.zeros(5, 120, dtype=torch.long), torch.full((5, 120), 5), torch.zeros(5, 119)):
        expect_value_error(lambda: XHandPatchEncoder(region_ids=labels))
    for mode in ("identity", "limits", "gaussian"):
        expect_value_error(lambda: encoder.validate_normalization({"taxel_tactile_force": mode}))
    for source in ("sim", "real"):
        validate_tactile_source(tactile_smoke_attrs(source), source)
        expect_value_error(lambda: validate_tactile_source(tactile_smoke_attrs(source), "real" if source == "sim" else "sim"))
        expect_value_error(lambda: validate_tactile_source({}, source))
    expect_value_error(lambda: normalization_feature_shapes({"taxel_tactile_force": 3}))
    return {"status": "passed", "checks": "streaming, units, shapes, affine boundaries, patch descriptors, source"}


def validate_tactile_roundtrip(cfg):
    from dexmani_policy.common.config import FIELD_FEATURE_SHAPES
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset
    from dexmani_policy.training.common.resume import resolve_input_recipe
    from dexmani_policy.agents.policies.adaptac import AdapTacPolicy
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from types import SimpleNamespace

    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    config.agent.update(dict(num_points=32, feature_dim=32, state_out_dim=16,
                             diffusion_step_embed_dim=32,
                             down_dims=[32 * 2 ** i for i in range(len(cfg.agent.down_dims))]))
    config.agent.pc_encoder_config.update(dict(channels=48, num_patches=8, group_size=8, depth=1))
    config.agent.force_predictor_config.update(dict(n_layer=1, n_emb=32))
    config.dataset.augmentation_cfg = None
    config.dataset.max_train_episodes = None
    config.dataloader.update(dict(batch_size=2, num_workers=0, pin_memory=False))
    config.workspace.wandb_cfg = None
    config.training.use_compile = False
    config.training.use_bfloat16 = False
    frames = max(24, config.horizon + 2)
    ends = [frames, 2 * frames]
    config.dataset.split_manifest = dict(episode_ends=ends, train_indices=[0], val_indices=[1])
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory(prefix="tactile-roundtrip-") as temp:
        data_path = Path(temp) / "data.zarr"
        config.zarr_path = config.dataset.zarr_path = str(data_path)
        config.workspace.output_dir = str(Path(temp) / "experiment")
        root = zarr.open_group(str(data_path), mode="w")
        root.attrs.update(tactile_smoke_attrs(cfg.tactile_source, cfg.task_name))
        root.create_dataset("meta/episode_ends", data=np.array(ends))
        for key in [*config.dataset.sensor_modalities, "action"]:
            shape = FIELD_FEATURE_SHAPES.get(key, (32, config.agent.pc_dim) if key == "point_cloud" else (19,))
            values = rng.normal(size=(ends[-1], *shape)).astype(np.float32)
            if key == "taxel_tactile_force":
                values *= 100
            elif key == "finger_tactile_force":
                values[0, 1] = -values[0, 0]
                values[0, 2:] = 0  # Zero net force remains a valid observation.
            root.create_dataset(f"data/{key}", data=values, chunks=(7, *shape))
        # Missing aggregate and wrong dense layout fail at the data boundary.
        missing = list(config.dataset.sensor_modalities) + ["absent"]
        expect_value_error(lambda: RobotDataset(str(data_path), sensor_modalities=missing))
        if "taxel_tactile_force" in config.dataset.sensor_modalities:
            data = root["data/taxel_tactile_force"][:]
            del root["data/taxel_tactile_force"]
            root.create_dataset("data/taxel_tactile_force", data=data[:, :, :-1])
            expect_value_error(lambda: RobotDataset(str(data_path), sensor_modalities=config.dataset.sensor_modalities))
            del root["data/taxel_tactile_force"]
            root.create_dataset("data/taxel_tactile_force", data=data)
        from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset

        without_aggregate = [key for key in config.dataset.sensor_modalities if key != "finger_tactile_force"]
        expect_value_error(lambda: AdapTacDataset(str(data_path), tactile_source=cfg.tactile_source, sensor_modalities=without_aggregate))
        dataset, normalizer, description = build_dataset_and_normalizer(config)
        for key in ("finger_tactile_force", "taxel_tactile_force"):
            mode = config.normalization.get(key)
            if mode in {"rms", "max_abs"}:
                values = torch.from_numpy(root[f"data/{key}"][:frames]).double()
                magnitude = values.square().mean().sqrt() if mode == "rms" else values.abs().max()
                torch.testing.assert_close(normalizer[key].scale, magnitude.reciprocal().float().expand(FIELD_FEATURE_SHAPES[key]))
        batch = default_collate([dataset[0], dataset[len(dataset) - 1]])
        validate_data_batch(batch, dataset, config)
        with closing(PolicyTrainer(config, dataset, normalizer, description)) as trainer:
            model = trainer.model
            assert model.force_dim == 15 and model.force_fusion.force_proj.in_features == 15 * (config.horizon + 1)
            inputs = dict_apply(batch, lambda x: x.to(trainer.device))
            contract = validate_adaptac_model_contract(model, inputs)
            snapshot = capture_policy_predictions(model, trainer.ema_model, batch)
            trainer.save_checkpoint("contract")
            checkpoint_path = trainer.checkpoint_dir / "latest.pt"
            saved = OmegaConf.to_container(trainer.cfg, resolve=True)
        for override in ("tactile=sim_force", "normalization.finger_tactile_force=max_abs", "agent.force_loss_weight=0"):
            expect_value_error(lambda: resolve_input_recipe(config, overrides=[override], saved_config=saved))
        expect_value_error(lambda: AdapTacPolicy.prepare_restore_config({"force_scale": 1.0}))
        data_path.rename(Path(temp) / "unavailable.zarr")
        with patch.object(LinearNormalizer, "fit_field", side_effect=AssertionError("restore must not fit")):
            wrong_source = {**saved, "tactile_source": "sim" if cfg.tactile_source == "real" else "real"}
            expect_value_error(lambda: restore_policy(wrong_source, checkpoint_path, use_ema=False, device=cfg.training.device))
            restored = validate_policy_restoration(saved, checkpoint_path, snapshot, device=torch.device(cfg.training.device))
            info = inspect_policy(config.workspace.output_dir, checkpoint="latest", weights="raw")
            observation = {k: v[0].numpy() for k, v in batch["obs"].items()}
            with closing(load_policy(info, device=cfg.training.device, seed=42)) as runtime:
                prediction = runtime.predict(observation)
                runtime.reset_episode()
                np.testing.assert_allclose(prediction, runtime.predict(observation))
                assert prediction.shape == (config.horizon, 19)
        runner = SimRunner("unused", 1, list(config.dataset.sensor_modalities), {})
        expect_value_error(lambda: next(runner.run(SimpleNamespace(tactile_source="real"), seeds=[0], inference_steps=20, action_steps=8)))
        from dexmani_policy.evaluation.config import merge_eval_config

        expect_value_error(lambda: merge_eval_config({"tactile_source": "real"}))
    return {"status": "passed", "source": cfg.tactile_source, "model": contract, "restoration": restored}


def validate_adaptac_data_contract():
    from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset

    ends = np.array([7, 10, 13, 14])
    force = (
        100 * np.arange(ends[-1])[:, None, None]
        + 10 * np.arange(5)[None, :, None] + np.arange(3)[None, None, :]
    ).astype(np.float64)
    force[1] = force[0]  # A held sample is still a valid observation row.
    force[2] = 0
    force[10:13] += 100000  # Validation outliers must not affect the affine.
    action = np.broadcast_to(np.arange(ends[-1])[:, None], (ends[-1], 19)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="adaptac-data-") as directory:
        root = zarr.open_group(directory, mode="w")
        root.attrs.update(tactile_smoke_attrs("sim"))
        root.create_dataset("meta/episode_ends", data=ends)
        root.create_dataset("data/finger_tactile_force", data=force)
        root.create_dataset("data/joint_state", data=action)
        root.create_dataset("data/action", data=action)
        dataset = AdapTacDataset(
            directory, tactile_source="sim", sensor_modalities=["joint_state", "finger_tactile_force"],
            horizon=4, obs_horizon=1, pad_after=3,
            split_manifest={
                "episode_ends": ends.tolist(), "train_indices": [0, 1, 3], "val_indices": [2],
            },
            augmentation_cfg={"state": {"noise": {"noise_std": 1.0, "prob": 1.0}}},
        )
        for view in (dataset, dataset.get_validation_dataset()):
            queries = []
            for index in range(len(view)):
                query = view.sampler.sample_source_rows(index)[0]
                end = ends[np.searchsorted(ends, query, side="right")]
                queries.append(query)
                batch = view[index]
                expected = force[np.minimum(query + np.arange(1, 5), end - 1)].astype(np.float32)
                torch.testing.assert_close(batch["future_force"], torch.from_numpy(expected))
                assert batch["future_force"].shape == (4, 5, 3)
                assert batch["future_force"].dtype == torch.float32
                torch.testing.assert_close(batch["future_force_valid"], torch.tensor(query + np.arange(1, 5) < end))
                assert batch["future_force_valid"].any()
                torch.testing.assert_close(
                    batch["future_force"].flatten(-2), torch.from_numpy(expected.reshape(4, -1)),
                )
                torch.testing.assert_close(batch["action"][0], torch.from_numpy(action[query]))
            assert not set(queries).intersection(ends - 1)
        # At query 0, the final force row is beyond action[0:4].
        torch.testing.assert_close(dataset[0]["future_force"][-1], torch.from_numpy(force[4].astype(np.float32)))
        spec = {"joint_state": "limits", "action": "limits", "finger_tactile_force": "rms"}
        normalizer = dataset.get_normalizer(spec)
        real_train = force[np.r_[0:10, 13:14]].astype(np.float32)
        observed = np.concatenate(list(dataset.iter_normalization_data("finger_tactile_force")))
        np.testing.assert_array_equal(observed, real_train)
        reference = LinearNormalizer()
        reference.fit_field("force", [real_train], mode="rms", last_n_dims=2)
        field = normalizer["finger_tactile_force"]
        for name in ("scale", "offset"):
            torch.testing.assert_close(getattr(field, name), getattr(reference["force"], name).expand(5, 3))
        assert dataset.get_validation_dataset().get_normalizer(spec) is normalizer
    return {
        "status": "passed",
        "checks": "order, t+1, beyond-action reads, boundaries, padding, singleton exclusion, train-only shared affine",
    }


def require_gradient(module, name):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), name
    total = sum(g.detach().float().abs().sum().item() for g in grads)
    assert total > 0, name
    return total


def validate_adaptac_model_contract(model, batch):
    device = next(model.parameters()).device
    raw = batch["obs"]["finger_tactile_force"]
    assert not raw[0].sum(-2).any()
    batch_size = len(raw)
    encoder = model.obs_encoder.tactile_encoder
    tokens = encoder(model.preprocess(batch["obs"])[encoder.input_key])
    assert tokens.shape == (batch_size, 5, 64) and model.force_dim == 15
    assert set(model.obs_encoder.consumed_observation_fields) == {
        "joint_state", "point_cloud", encoder.input_key,
    }
    assert set(model.consumed_observation_fields) == set(batch["obs"])
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-6)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    assert model.force_predictor.noise_scheduler is not model.action_decoder.noise_scheduler
    assert model.force_predictor.noise_scheduler.config.prediction_type == "sample"
    assert not model.force_predictor.noise_scheduler.config.clip_sample
    assert model.action_decoder.noise_scheduler.config.prediction_type == "epsilon"
    assert ("finger_tactile_force" in model.normalizer.fields) == (
        model.normalization_spec["finger_tactile_force"] != "identity"
    )
    loss, metrics = model.compute_loss(batch, log_metrics=False)
    assert torch.isfinite(loss) and metrics == {}
    gradient_results = {}
    for use_bf16 in (False, True):
        model.train().zero_grad(set_to_none=True)
        captured = {}

        def force_output(module, args, output):
            output.retain_grad()
            captured["prediction"] = output

        def fusion_input(module, args):
            captured["guide"] = args[2]

        def tactile_input(module, args):
            key = encoder.input_key
            expected = batch["obs"][key]
            if key in model.normalizer.fields:
                expected = model.normalizer[key].normalize(expected)
            torch.testing.assert_close(args[0], expected.flatten(0, 1))

        original = model.force_predictor.compute_loss_and_prediction

        def supervised_prediction(condition, target, valid):
            torch.testing.assert_close(target, model.normalize_force(batch["future_force"]).flatten(-2))
            loss, prediction = original(condition, target, valid)
            assert target.shape == prediction.shape == (batch_size, model.future_steps, model.force_dim)
            assert loss.dtype == torch.float32
            torch.testing.assert_close(loss, torch.nn.functional.mse_loss(prediction[valid], target[valid]))
            return loss, prediction

        with ExitStack() as stack:
            stack.enter_context(patch.object(model, "force_loss_weight", 0.0))
            stack.enter_context(model.force_predictor.model.register_forward_hook(force_output))
            stack.enter_context(model.force_fusion.register_forward_pre_hook(fusion_input))
            stack.enter_context(encoder.register_forward_pre_hook(tactile_input))
            stack.enter_context(patch.object(
                model.force_predictor, "compute_loss_and_prediction", side_effect=supervised_prediction,
            ))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, metrics = model.compute_loss(batch)
            torch.testing.assert_close(loss.detach(), metrics["loss_action"])
            assert torch.isfinite(loss)
            loss.backward()
        prediction = captured["prediction"]
        torch.testing.assert_close(
            captured["guide"],
            torch.cat((model.normalize_force(raw).flatten(-2), prediction.float()), dim=1).flatten(1),
        )
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        assert prediction.grad.abs().sum() > 0
        assert prediction.grad[~batch["future_force_valid"]].abs().sum() > 0
        interaction = model.obs_encoder.interaction
        modules = {"force_predictor": model.force_predictor, "tactile_attention": interaction.tactile_attn,
                   "visual_attention": interaction.visual_attn, "visual_encoder": model.obs_encoder.pc_encoder,
                   "tactile_encoder": model.obs_encoder.tactile_encoder}
        sums = {name: require_gradient(module, name) for name, module in modules.items()}
        embedding_grad = interaction.finger_embedding.grad
        assert embedding_grad is not None and torch.isfinite(embedding_grad).all()
        assert embedding_grad.abs().sum() > 0
        sums["finger_embedding"] = embedding_grad.float().abs().sum().item()
        gradient_results["bf16" if use_bf16 else "fp32"] = sums

    model.eval()
    calls = []
    with ExitStack() as stack:
        force_sample = stack.enter_context(patch.object(
            model.force_predictor, "predict_action", wraps=model.force_predictor.predict_action,
        ))
        action_sample = stack.enter_context(patch.object(
            model.action_decoder, "predict_action", wraps=model.action_decoder.predict_action,
        ))
        for name, module in (("encode", model.obs_encoder), ("interaction", model.obs_encoder.interaction),
                             ("force", model.force_predictor.model), ("action", model.action_decoder.model)):
            stack.enter_context(module.register_forward_hook(
                lambda module, args, output, name=name: calls.append(name),
            ))
        generator = torch.Generator(device=device).manual_seed(42)
        rng = get_rng_state(device)
        output = model.predict_action(batch["obs"], generator=generator)
        assert force_sample.call_args.kwargs["generator"] is generator
        assert action_sample.call_args.kwargs["generator"] is generator
        assert list(output) == ["future_actions"]
        assert output["future_actions"].shape == (batch_size, model.future_steps, 19)
        force_steps = model.force_predictor.num_inference_steps
        action_steps = model.action_decoder.num_inference_steps
        assert calls == ["interaction", "encode"] + ["force"] * force_steps + ["action"] * action_steps
        generator_after = generator.get_state()
        calls.clear()
        set_rng_state(rng, device)
        generator.manual_seed(42)
        again = model.predict_action(batch["obs"], generator=generator)
        torch.testing.assert_close(output["future_actions"], again["future_actions"])
        assert torch.equal(generator_after, generator.get_state())
    return {
        "status": "passed", "action_only_gradient_l1": gradient_results,
        "optimizer_parameters": len(ids), "tactile_source": model.tactile_source,
        "sampling": f"one encoding/interaction, independent {force_steps}+{action_steps} DDIM, generator reproducibility",
    }


@torch.inference_mode()
def validate_adaptac_deployment(experiment, dataset, device):
    validation = dataset.get_validation_dataset()
    if validation is None:
        return {"status": "not_run", "reason": "no validation windows"}
    info = inspect_policy(experiment, checkpoint="latest")
    selected = np.sort(np.random.default_rng(42).choice(len(validation), min(32, len(validation)), replace=False))
    queries = validation.sampler.source_rows(validation.sampler.indices[selected])[:, 0]
    with closing(load_policy(info, device=str(device), seed=42)) as runtime:
        model = runtime.agent
        samples = [validation[int(index)] for index in selected]
        errors, masks, zero_errors, persistence_errors = [], [], [], []
        generator = torch.Generator(device=device).manual_seed(42)
        for start in range(0, len(samples), 4):
            chunk = samples[start : start + 4]
            obs = {name: torch.stack([item["obs"][name] for item in chunk]).to(device) for name in chunk[0]["obs"]}
            features = model.obs_encoder(model.preprocess(obs))
            predicted = model.force_predictor.predict_action(
                torch.cat(features[:2], dim=-1).unsqueeze(1),
                (len(chunk), model.future_steps, model.force_dim), generator=generator, dtype=torch.float32,
            )
            target = torch.stack([item["future_force"] for item in chunk])
            error = model.unnormalize_force(predicted.reshape(-1, model.future_steps, 5, 3)).cpu() - target
            masks.append(torch.stack([item["future_force_valid"] for item in chunk]))
            zero_errors.append(-target)
            persistence_errors.append(obs["finger_tactile_force"].cpu() - target)
            assert torch.isfinite(error).all()
            errors.append(error)
        error = torch.cat(errors)
        valid = torch.cat(masks)

        def force_metrics(error):
            def window_rmse(stop):
                mask = valid[:, :stop, None, None]
                return ((error[:, :stop].square() * mask).sum() / (15 * mask.sum())).sqrt().item()
            return {
                "full_horizon_rmse": window_rmse(info.future_steps),
                "execution_window_rmse": window_rmse(info.n_action_steps),
                "per_finger_rmse": ((error.square() * valid[..., None, None]).sum((0, 1, 3))
                                    / (3 * valid.sum())).sqrt().tolist(),
            }

        report = {
            "status": "passed", "samples": len(error), "seed": 42,
            "metric_kind": "pure_noise_forecasting", "valid_successors": valid.sum().item(),
            "weights": info.weights,
            "unit": "N, world axes" if model.tactile_source == "sim" else "SDK native unknown physical scale, sensor axes",
            "finger_order": ADAPTAC_FINGER_ORDER, "query_source_rows": queries.tolist(),
            "force_ddim_steps": model.force_predictor.num_inference_steps,
            "forecast": force_metrics(error), "zero": force_metrics(torch.cat(zero_errors)),
            "persistence": force_metrics(torch.cat(persistence_errors)),
        }
        observation = {name: value.numpy() for name, value in samples[0]["obs"].items()}
        for _ in range(5):
            runtime.predict(observation)
        durations = []
        for _ in range(20):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            query_start = time.perf_counter()
            runtime.predict(observation)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(1000 * (time.perf_counter() - query_start))
        report["latency"] = {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "batch_size": 1, "parameter_dtype": str(next(runtime.agent.parameters()).dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "autocast": False, "warmups": 5, "queries": len(durations),
            "mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "includes": (
                "host staging, transfer, normalization, encoding, interaction, "
                "force DDIM, FGAF, action DDIM, CPU action output"
            ),
        }
        return report


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


def validate_adaptac_contracts(cfg):
    return {
        "normalization_patch": validate_tactile_numerics(),
        "roundtrip": validate_tactile_roundtrip(cfg),
        "data": validate_adaptac_data_contract(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--config-only",
        action="store_true",
        help="Check config interfaces and target imports; model constraints are checked at construction/binding.",
    )
    modes.add_argument(
        "--data-only", action="store_true", help="Check CPU data batches without building a model."
    )
    modes.add_argument(
        "--contracts-only", action="store_true",
        help="Run focused AdapTac data/model contracts without real data or training.",
    )
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override.")
    parser.add_argument("--device", help="Override training.device for this smoke run.")
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile policy backbones during bounded training; requires training mode.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Keep smoke artifacts in a new directory instead of a temporary directory.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=2,
        help="Data-only batch budget per split (default: 2); does not limit statistics scans.",
    )
    parser.add_argument(
        "config_names", nargs="+", help="Hydra config names or explicit test YAML paths."
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=4,
        help="Initial training-update budget; does not change the saved recipe (default: 4).",
    )
    parser.add_argument(
        "--resume-updates", type=int, default=2,
        help="Additional updates after checkpoint resume; 0 disables this phase (default: 2).",
    )
    parser.add_argument(
        "--rollout", action="store_true", help="Also run one simulation episode using the final checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.compile and (args.config_only or args.data_only or args.contracts_only):
        parser.error("--compile requires bounded training mode")

    set_project_root()
    for config_name in args.config_names:
        print(f"\nValidating {config_name}", flush=True)
        cfg = load_smoke_config(config_name, args.override)
        if args.device is not None:
            cfg.training.device = args.device
        if args.compile:
            cfg.training.use_compile = True
        output_dir = args.output_dir
        if output_dir is not None and len(args.config_names) > 1:
            output_dir = output_dir / cfg.policy_name
        if args.config_only:
            summary = f"config and {validate_target_imports(cfg)} Hydra targets"
        elif args.data_only:
            validate_dataset(cfg, max_batches=args.max_batches)
            summary = "CPU data batches"
        elif args.contracts_only:
            if cfg.policy_name != "adaptac":
                parser.error("Focused contract checks are currently available for adaptac")
            set_seed(cfg.training.seed)
            with smoke_output(output_dir) as directory:
                report = validate_adaptac_contracts(cfg)
                (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            summary = "AdapTac contracts"
        else:
            validate_training(
                cfg, max_updates=args.max_updates, resume_updates=args.resume_updates,
                rollout=args.rollout, output_dir=output_dir,
            )
            summary = "bounded training and checkpoint checks"
        print(f"✓ {config_name}: {summary} PASSED")


if __name__ == "__main__":
    main()
