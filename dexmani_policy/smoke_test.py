"""Validate configs/data or exercise training and checkpoint restoration."""

import argparse
import importlib
import importlib.util
import json
import tempfile
import time
import traceback
from contextlib import ExitStack, closing, contextmanager
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

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


def load_smoke_config(config_name: str):
    cfg = load_policy_config(config_name)
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
    reports = {}
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
        with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as directory:
            yield Path(directory)
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
        report = {"device": str(device), "training": []}
        try:
            if cfg.policy_name == "adaptac":
                report["contracts"] = validate_adaptac_contracts(device)
                set_seed(cfg.training.seed)
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
                report["force_statistics"] = adaptac_force_statistics(dataset)
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


def validate_adaptac_data_contract():
    from dexmani_policy.datasets.aux.adaptac_dataset import AdapTacDataset

    ends = np.array([7, 10, 13, 14])
    force = (
        100 * np.arange(ends[-1])[:, None, None]
        + 10 * np.arange(5)[None, :, None] + np.arange(3)[None, None, :]
    ).astype(np.float64)
    force[10:13] += 100000  # Validation outliers must not affect the affine.
    action = np.broadcast_to(np.arange(ends[-1])[:, None], (ends[-1], 19)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="adaptac-data-") as directory:
        root = zarr.open_group(directory, mode="w")
        root.create_dataset("meta/episode_ends", data=ends)
        root.create_dataset("data/finger_tactile_force", data=force)
        root.create_dataset("data/joint_state", data=action)
        root.create_dataset("data/action", data=action)
        dataset = AdapTacDataset(
            directory, sensor_modalities=["joint_state", "finger_tactile_force"],
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
                torch.testing.assert_close(
                    batch["future_force"].flatten(-2), torch.from_numpy(expected.reshape(4, -1)),
                )
                torch.testing.assert_close(batch["action"][0], torch.from_numpy(action[query]))
            assert not set(queries).intersection(ends - 1)
        # At query 0, the final force row is beyond action[0:4].
        torch.testing.assert_close(dataset[0]["future_force"][-1], torch.from_numpy(force[4].astype(np.float32)))
        spec = {"joint_state": "limits", "action": "limits", "finger_tactile_force": "limits"}
        normalizer = dataset.get_normalizer(spec)
        real_train = force[np.r_[0:10, 13:14]].astype(np.float32)
        observed = np.concatenate(list(dataset.iter_normalization_data("finger_tactile_force")))
        np.testing.assert_array_equal(observed, real_train)
        reference = LinearNormalizer()
        reference.fit_field("force", [real_train], last_n_dims=1)
        field = normalizer["finger_tactile_force"]
        for name in ("scale", "offset"):
            torch.testing.assert_close(getattr(field, name), getattr(reference["force"], name).expand(5, 3))
        assert dataset.get_validation_dataset().get_normalizer(spec) is normalizer
    return {
        "status": "passed",
        "checks": "order, t+1, beyond-action reads, boundaries, padding, singleton exclusion, train-only shared affine",
    }


def validate_adaptac_interaction():
    from dexmani_policy.agents.obs_encoder.interaction.future_tactile.visuo_tactile_interaction import (
        VisuoTactileInteraction,
    )

    block = VisuoTactileInteraction(384, 64, 5).eval()
    visual, tactile = torch.randn(2, 13, 384), torch.randn(2, 5, 64)
    with torch.no_grad():
        v, t = block.visual_norm(visual), block.tactile_norm(tactile)
        e = block.finger_embedding
        expected_t = tactile + block.tactile_attn(t + e, v, v, need_weights=False)[0]
        expected_v = visual + block.visual_attn(v, t + e, t, need_weights=False)[0]
        actual_v, actual_t = block(visual, tactile)
        torch.testing.assert_close(actual_v, expected_v)
        torch.testing.assert_close(actual_t, expected_t)
    return {"status": "passed", "visual_tokens": 13, "finger_embedding_shape": list(e.shape)}


def require_gradient(module, name):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), name
    total = sum(g.detach().float().abs().sum().item() for g in grads)
    assert total > 0, name
    return total


def validate_adaptac_model_contract(device):
    from dexmani_policy.agents.policies.adaptac import AdapTacPolicy

    model = AdapTacPolicy(
        horizon=4, n_obs_steps=1, n_action_steps=2, action_dim=19,
        num_points=32, feature_dim=32, state_out_dim=16,
        pc_encoder_config={"channels": 48, "num_patches": 8, "group_size": 8, "depth": 1},
        diffusion_step_embed_dim=32, down_dims=(32, 64), force_scale=2.5,
        force_predictor_config={"n_layer": 1, "n_emb": 32},
    ).to(device)
    spec = {
        "joint_state": "limits", "action": "limits", "point_cloud": "identity",
        "finger_tactile_force": "limits",
    }
    model.set_normalization_spec(spec, {"joint_state": 19, "action": 19, "finger_tactile_force": model.force_dim})
    model.normalizer.to(device)
    # A deliberately nonidentity encoder affine must never reach the physical guide.
    model.normalizer["finger_tactile_force"].scale.fill_(0.3)
    model.normalizer["finger_tactile_force"].offset.fill_(0.7)
    raw = torch.zeros(2, 1, 5, 3, device=device)
    raw[:, :, 0] = torch.tensor([2.0, -3.0, 4.0], device=device)
    raw[:, :, 1] = -raw[:, :, 0]
    assert not raw.sum(-2).any()
    batch = {
        "obs": {"joint_state": torch.randn(2, 1, 19, device=device),
                "point_cloud": torch.randn(2, 1, 32, 6, device=device), "finger_tactile_force": raw},
        "action": torch.randn(2, 4, 19, device=device),
        "future_force": torch.arange(2 * 4 * model.force_dim, device=device).reshape(2, 4, 5, 3).float() / 30,
    }
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-6)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    assert model.force_predictor.noise_scheduler is not model.action_decoder.noise_scheduler
    assert model.force_predictor.noise_scheduler.config.prediction_type == "sample"
    assert not model.force_predictor.noise_scheduler.config.clip_sample
    assert model.action_decoder.noise_scheduler.config.prediction_type == "epsilon"
    gradient_results = {}
    for use_bf16 in (False, True) if device.type == "cuda" else (False,):
        model.train().zero_grad(set_to_none=True)
        model.force_loss_weight = 0.0
        captured = {}

        def force_output(module, args, output):
            output.retain_grad()
            captured["prediction"] = output

        def fusion_input(module, args):
            captured["guide"] = args[2]

        original = model.force_predictor.compute_loss_and_prediction

        def supervised_prediction(condition, target):
            torch.testing.assert_close(target, batch["future_force"].flatten(-2) / model.force_scale)
            loss, prediction = original(condition, target)
            assert target.shape == prediction.shape == (2, model.future_steps, model.force_dim)
            assert loss.dtype == torch.float32
            torch.testing.assert_close(loss, (prediction.float() - target).square().mean())
            return loss, prediction

        with ExitStack() as stack:
            stack.enter_context(model.force_predictor.model.register_forward_hook(force_output))
            stack.enter_context(model.force_fusion.register_forward_pre_hook(fusion_input))
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
            torch.cat((raw.flatten(-2) / model.force_scale, prediction), dim=1).flatten(1),
        )
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
        assert prediction.grad.abs().sum() > 0
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
        assert list(output) == ["future_actions"] and output["future_actions"].shape == (2, 4, 19)
        assert calls == ["interaction", "encode"] + ["force"] * 20 + ["action"] * 20
        generator_after = generator.get_state()
        calls.clear()
        set_rng_state(rng, device)
        generator.manual_seed(42)
        again = model.predict_action(batch["obs"], generator=generator)
        torch.testing.assert_close(output["future_actions"], again["future_actions"])
        assert torch.equal(generator_after, generator.get_state())
    return {
        "status": "passed", "action_only_gradient_l1": gradient_results,
        "optimizer_parameters": len(ids), "force_scale": model.force_scale,
        "sampling": "one encoding/interaction, independent 20+20 DDIM, generator reproducibility",
    }


def adaptac_force_statistics(dataset):
    force = np.concatenate(list(dataset.iter_normalization_data("finger_tactile_force"))).astype(np.float64)
    return {
        "train_episodes": int(dataset.train_mask.sum()), "real_train_frames": len(force),
        "finger_order": ADAPTAC_FINGER_ORDER, "unit": "raw dataset force units; consult producer contract",
        "component_rms": np.sqrt(np.mean(force ** 2, axis=(0, 2))).tolist(),
        "abs_percentiles": [50, 90, 95, 99, 100],
        "per_finger_abs_percentiles": np.percentile(np.abs(force), [50, 90, 95, 99, 100], axis=(0, 2)).T.tolist(),
    }


@torch.inference_mode()
def validate_adaptac_deployment(experiment, dataset, device):
    validation = dataset.get_validation_dataset()
    if validation is None:
        return {"status": "not_run", "reason": "no validation windows"}
    info = inspect_policy(experiment, checkpoint="latest")
    indices = validation.sampler.indices
    queries = validation.sampler.source_rows(indices)[:, 0]
    ends = validation.replay_buffer.episode_ends
    episode_ends = ends[np.searchsorted(ends, queries, side="right")]
    eligible = np.flatnonzero(queries + info.future_steps < episode_ends)
    if not len(eligible):
        return {"status": "not_run", "reason": "no validation windows with K real successor frames"}
    runtime = load_policy(info, device=str(device), seed=42)
    model = runtime.agent
    selected = np.sort(np.random.default_rng(42).choice(eligible, min(32, len(eligible)), replace=False))
    samples = [validation[int(index)] for index in selected]
    errors = []
    generator = torch.Generator(device=device).manual_seed(42)
    for start in range(0, len(samples), 4):
        chunk = samples[start : start + 4]
        obs = {name: torch.stack([item["obs"][name] for item in chunk]).to(device) for name in chunk[0]["obs"]}
        features = model.obs_encoder(model.preprocess(obs))
        predicted = model.force_predictor.predict_action(
            torch.cat(features[:2], dim=-1).unsqueeze(1),
            (len(chunk), model.future_steps, model.force_dim), generator=generator,
        )
        target = torch.stack([item["future_force"] for item in chunk])
        error = predicted.float().cpu().reshape_as(target) * model.force_scale - target
        assert torch.isfinite(error).all()
        errors.append(error)
    error = torch.cat(errors)
    report = {
        "status": "passed", "samples": len(error), "seed": 42,
        "weights": info.weights, "unit": "raw dataset force units; consult producer contract",
        "finger_order": ADAPTAC_FINGER_ORDER, "query_source_rows": queries[selected].tolist(),
        "force_ddim_steps": model.force_predictor.num_inference_steps,
        "per_finger_mae": error.abs().mean((0, 1, 3)).tolist(),
        "per_finger_rmse": error.square().mean((0, 1, 3)).sqrt().tolist(),
        "per_step_mae": error.abs().mean((0, 2, 3)).tolist(),
        "per_step_rmse": error.square().mean((0, 2, 3)).sqrt().tolist(),
        "per_step_per_finger_mae": error.abs().mean((0, 3)).tolist(),
        "per_step_per_finger_rmse": error.square().mean((0, 3)).sqrt().tolist(),
    }
    observation = {name: value.numpy() for name, value in samples[0]["obs"].items()}
    for _ in range(5):
        runtime.predict(observation)
    durations = []
    for _ in range(20):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        runtime.predict(observation)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(1000 * (time.perf_counter() - start))
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


def validate_adaptac_contracts(device):
    return {
        "data": validate_adaptac_data_contract(),
        "interaction": validate_adaptac_interaction(),
        "model": validate_adaptac_model_contract(device),
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
    parser.add_argument("--device", help="Override training.device for this smoke run.")
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile both AdapTac backbones, or the configured policy backbone.",
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

    set_project_root()
    for config_name in args.config_names:
        print(f"\nValidating {config_name}", flush=True)
        cfg = load_smoke_config(config_name)
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
                report = validate_adaptac_contracts(torch.device(cfg.training.device))
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
