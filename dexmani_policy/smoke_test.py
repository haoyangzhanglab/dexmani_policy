"""Validate configs/data or exercise training and checkpoint restoration."""

import argparse
import importlib
import importlib.util
import tempfile
from contextlib import closing
from itertools import islice

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import (
    load_training_checkpoint,
    policy_state_dict,
    restore_policy_agent,
)
from dexmani_policy.common.config import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    load_experiment_config,
    load_policy_config,
    positive_int,
    set_project_root,
)
from dexmani_policy.common.runtime import dict_apply, set_seed
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.hand_prior.temporal_codec_dataset import TemporalCodecDataset
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset, observation_layout
from dexmani_policy.training.builder import (
    build_training_components,
    build_trainer,
    compile_training_models,
)
from dexmani_policy.training.data import (
    build_dataset_and_normalizer,
    build_train_loader,
    close_data_loader,
)
from dexmani_policy.training.validation import validate_config


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
    for section in ("obs", "action_context"):
        for key, value in batch.get(section, {}).items():
            yield f"{section}.{key}", value


def validate_data_batch(batch, dataset, cfg):
    action = batch["action"]
    batch_size = len(action)
    validate_tensor_layout(action, (batch_size, cfg.horizon, cfg.agent.action_dim), "action")

    obs = batch["obs"]
    for key in NON_NUMERIC_OBSERVATION_FIELDS.intersection(obs):
        if (
            not isinstance(obs[key], (list, tuple))
            or len(obs[key]) != batch_size
            or not all(isinstance(value, str) for value in obs[key])
        ):
            raise ValueError(f"obs.{key}: expected one string per sample")
    if isinstance(dataset, MultiTaskDataset):
        if not NON_NUMERIC_OBSERVATION_FIELDS <= obs.keys():
            raise ValueError("Multitask batch requires task_text and task_name")
        datasets_by_task = dict(zip(dataset.task_names, dataset.datasets))
        task_texts = dict(zip(dataset.task_names, dataset.task_texts))
        for name, text in zip(obs["task_name"], obs["task_text"]):
            if name not in datasets_by_task or text != task_texts[name]:
                raise ValueError("Multitask batch task name/text mismatch")
        children = [datasets_by_task[name] for name in dict.fromkeys(obs["task_name"])]
    else:
        children = [dataset]

    numeric_fields = obs.keys() - NON_NUMERIC_OBSERVATION_FIELDS
    context = batch.get("action_context", {})
    for child in children:
        if numeric_fields != set(child.sensor_modalities):
            raise ValueError("Batch observation fields differ from the dataset recipe")
        for key in child.sensor_modalities:
            shape, _ = observation_layout(child, key)
            source_dtype = child.replay_buffer[key].dtype
            if key == "rgb" and child.rgb_preprocess_size is not None:
                dtype = torch.uint8 if child.rgb_keep_uint8 else torch.float32
            elif np.issubdtype(source_dtype, np.floating):
                dtype = torch.float32
            else:
                dtype = torch.from_numpy(np.empty((), dtype=source_dtype)).dtype
            validate_tensor_layout(obs[key], (batch_size, *shape), f"obs.{key}", dtype)

        context_shapes = {}
        if isinstance(child, LAMPDataset):
            context_shapes["hand_history"] = (batch_size, child.history_steps, 12)
        elif isinstance(child, TemporalCodecDataset):
            context_shapes["hand_reference"] = (batch_size, 12)
        if not isinstance(context, dict) or context.keys() != context_shapes.keys():
            raise ValueError("Batch action_context differs from the dataset recipe")
        for key, shape in context_shapes.items():
            validate_tensor_layout(context[key], shape, f"action_context.{key}", torch.float32)

    # Scan each tensor once, independent of the number of tasks in the batch.
    for name, value in iter_batch_fields(batch):
        if not isinstance(value, torch.Tensor):
            continue
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
                (
                    f"{key}={tuple(value.shape)} {value.dtype}"
                    if torch.is_tensor(value)
                    else f"{key}=list[str]({len(value)})"
                )
                for key, value in iter_batch_fields(batch)
            )
            print(f"[{split} batch {checked}] {summary}")
        if checked == 0:
            raise ValueError(f"{split}: no batches to check")
    finally:
        close_data_loader(loader, iterator)
    print(f"[{split}] windows={len(loader.dataset):,}, checked batches={checked}")


def validate_dataset(cfg, *, max_batches=2):
    positive_int(max_batches, "max_batches")
    if cfg.training.get("num_gpus", 1) != 1:
        print("DDP config: single-process data check only; DDP NOT VERIFIED")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _ = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset, rank=0, world_size=1)
    validate_data_loader(loader, cfg, split="train", max_batches=max_batches)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无 holdout")
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


def validate_checkpoint_state(cfg, components, trainer, checkpoint_path):
    checkpoint = load_training_checkpoint(checkpoint_path)
    assert checkpoint.global_step == trainer.global_step
    assert checkpoint.epoch == trainer.current_epoch
    assert checkpoint.next_micro_step == trainer.next_micro_step
    assert components.scheduler.state_dict() == checkpoint.scheduler_state
    assert (
        checkpoint.effective_config["training"]["loop"]["total_train_steps"]
        == cfg.training.loop.total_train_steps
    )
    if components.ema_updater is not None:
        assert checkpoint.ema_updater_step == components.ema_updater.optimization_step
    for model, state in (
        (components.model, checkpoint.model_state),
        (components.ema_model, checkpoint.ema_model_state),
    ):
        if model is not None:
            torch.testing.assert_close(
                policy_state_dict(model), state, rtol=0, atol=0, check_device=False
            )


def validate_policy_restoration(cfg, components, trainer, checkpoint_path):
    """Compare raw/EMA predictions with restored policies after training."""
    components.train_loader.sampler.set_epoch(trainer.current_epoch, 0)
    batch = next(iter(components.train_loader))
    obs = dict_apply(
        {key: value[:1] for key, value in batch["obs"].items()},
        lambda value: value.to(components.device),
    )
    prediction_kwargs = {}
    if "action_context" in batch:
        prediction_kwargs["action_context"] = dict_apply(
            batch["action_context"], lambda value: value[:1].to(components.device)
        )

    predictions = {}
    for use_ema, model in ((False, components.model), (True, components.ema_model)):
        if model is None:
            continue
        model.eval()
        set_seed(cfg.training.seed)
        with torch.inference_mode():
            prediction = model.predict_action(obs, **prediction_kwargs)["control_action"]
        assert prediction.shape == (1, cfg.n_action_steps, model.control_action_dim)
        assert torch.isfinite(prediction).all()

        # Eager kernels isolate checkpoint roundtrips from compile rounding differences.
        backbone = model.action_decoder.model
        if hasattr(backbone, "_orig_mod"):
            model.action_decoder.model = backbone._orig_mod
            set_seed(cfg.training.seed)
            with torch.inference_mode():
                eager = model.predict_action(obs, **prediction_kwargs)["control_action"]
            print(
                f"{'EMA' if use_ema else 'raw'} compiled/eager prediction max abs diff: "
                f"{(prediction - eager).abs().max().item():.8g}"
            )
            prediction = eager
        predictions[use_ema] = prediction.cpu()
        model.to("cpu")

    # Release training tensors before loading inference copies.
    components.optimizer.state.clear()
    saved = load_experiment_config(components.workspace.output_dir)
    for use_ema, expected in predictions.items():
        restored = restore_policy_agent(
            saved, checkpoint_path, use_ema=use_ema, device=components.device
        )
        set_seed(cfg.training.seed)
        with torch.inference_mode():
            actual = restored.predict_action(obs, **prediction_kwargs)["control_action"].cpu()
        torch.testing.assert_close(actual, expected)
        del restored


def validate_training(cfg, *, max_updates=4):
    """Exercise training and checkpoint restoration within an update budget."""
    positive_int(max_updates, "max_updates")
    if cfg.training.get("num_gpus", 1) != 1:
        raise ValueError("Single-device smoke requires num_gpus=1")
    if not torch.cuda.is_available():
        raise RuntimeError("NOT VERIFIED: original training configuration requires CUDA")

    set_seed(cfg.training.seed)
    with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as tmpdir:
        cfg.workspace.output_dir = tmpdir
        cfg.workspace.wandb_cfg = None
        dataset, normalizer = build_dataset_and_normalizer(cfg)
        components = build_training_components(
            cfg, torch.device(cfg.training.device), dataset, normalizer
        )
        with closing(components):
            trainer = build_trainer(cfg, components)
            components.workspace.save_hydra_config(cfg)
            before = {
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in components.model.named_parameters()
                if parameter.requires_grad and parameter.is_floating_point()
            }
            compile_training_models(cfg, components)
            trainer.train(max_updates=max_updates)
            changed = validate_parameter_updates(components.model, before)
            del before
            print(f"Observed parameter update: {changed} (step={trainer.global_step})")

            checkpoint_path = components.workspace.checkpoint_dir / "latest.pt"
            validate_checkpoint_state(cfg, components, trainer, checkpoint_path)
            validate_policy_restoration(cfg, components, trainer, checkpoint_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--config-only",
        action="store_true",
        help="Validate configs and target imports without data or GPU execution.",
    )
    modes.add_argument(
        "--data-only", action="store_true", help="Check CPU data batches without building a model."
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
        help="Execution-only optimizer-update budget (default: 4).",
    )
    args = parser.parse_args(argv)

    set_project_root()
    for config_name in args.config_names:
        print(f"\nValidating {config_name}", flush=True)
        cfg = load_smoke_config(config_name)
        if args.config_only:
            summary = f"config and {validate_target_imports(cfg)} Hydra targets"
        elif args.data_only:
            validate_dataset(cfg, max_batches=args.max_batches)
            summary = "CPU data batches"
        else:
            validate_training(cfg, max_updates=args.max_updates)
            summary = "training, prediction and raw/EMA restoration"
        print(f"✓ {config_name}: {summary} PASSED")


if __name__ == "__main__":
    main()
