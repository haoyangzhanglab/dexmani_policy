"""Validate configs/data or exercise training and checkpoint restoration."""

import argparse
import importlib
import importlib.util
import tempfile
import traceback

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
    build_dataset_and_normalizer,
    build_training_components,
    build_train_loader,
    build_trainer,
    compile_models,
)
from dexmani_policy.training.validation import validate_config

set_project_root()


def load_smoke_config(config_name: str):
    cfg = load_policy_config(config_name)
    # Hydra compose does not create a runtime output directory.
    cfg.workspace.output_dir = "/tmp/smoke_test_output"
    OmegaConf.resolve(cfg)
    return cfg


def iter_targets(node, path: str = "$"):
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)

    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str):
            yield path, target
        for key, value in node.items():
            yield from iter_targets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_targets(value, f"{path}[{index}]")


def validate_target_references(cfg) -> int:
    """Resolve policy, training and dataset classes; locate optional simulation modules."""
    agent_target = cfg.get("agent", {}).get("_target_")
    count = 0
    for path, target in iter_targets(cfg):
        module_name, separator, attribute = target.rpartition(".")
        if not separator:
            raise ImportError(f"Invalid Hydra target {target!r} at {path}")
        try:
            if target == agent_target or module_name.startswith(
                ("dexmani_policy.training.", "dexmani_policy.datasets.")
            ):
                module = importlib.import_module(module_name)
                getattr(module, attribute)
            elif importlib.util.find_spec(module_name) is None:
                raise ModuleNotFoundError(module_name)
        except Exception as exc:
            raise ImportError(f"Failed to resolve Hydra target {target!r} at {path}") from exc
        count += 1
    return count


def validate_config_only(config_name: str):
    print(f"\nConfig check: {config_name}")
    cfg = load_smoke_config(config_name)
    validate_config(cfg)
    target_count = validate_target_references(cfg)
    print(f"✓ {config_name}: {target_count} Hydra targets checked, config check PASSED")


def validate_data_tensor(value, shape, name, dtype=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name}: expected tensor shape {tuple(shape)}, got {getattr(value, 'shape', None)}"
        )
    if value.device.type != "cpu" or not torch.isfinite(value).all():
        raise ValueError(f"{name}: expected finite CPU data")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {value.dtype}")


def validate_data_batch(batch, dataset, cfg):
    action = batch["action"]
    batch_size = len(action)
    if batch_size == 0:
        raise ValueError("Empty data batch")
    validate_data_tensor(action, (batch_size, cfg.horizon, cfg.agent.action_dim), "action")
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
        children = dict(zip(dataset.task_names, dataset.datasets))
        texts = dict(zip(dataset.task_names, dataset.task_texts))
        for name, text in zip(obs["task_name"], obs["task_text"]):
            if name not in children or text != texts[name]:
                raise ValueError("Multitask batch task name/text mismatch")
        children = [children[name] for name in dict.fromkeys(obs["task_name"])]
    else:
        children = [dataset]
    for child in children:
        if set(obs) - NON_NUMERIC_OBSERVATION_FIELDS != set(child.sensor_modalities):
            raise ValueError("Batch observation fields differ from the dataset recipe")
        for key in child.sensor_modalities:
            shape, _ = observation_layout(child, key)
            source_dtype = child.replay_buffer[key].dtype
            dtype = torch.from_numpy(np.empty((), dtype=source_dtype)).dtype
            if np.issubdtype(source_dtype, np.floating):
                dtype = torch.float32
            if key == "rgb" and child.rgb_preprocess_size is not None:
                dtype = torch.uint8 if child.rgb_keep_uint8 else torch.float32
            validate_data_tensor(obs[key], (batch_size, *shape), f"obs.{key}", dtype)
        context_shapes = {}
        if isinstance(child, LAMPDataset):
            context_shapes["hand_history"] = (batch_size, child.history_steps, 12)
        elif isinstance(child, TemporalCodecDataset):
            context_shapes["hand_reference"] = (batch_size, 12)
        context = batch.get("action_context", {})
        if not isinstance(context, dict) or context.keys() != context_shapes.keys():
            raise ValueError("Batch action_context differs from the dataset recipe")
        for key, shape in context_shapes.items():
            validate_data_tensor(context[key], shape, f"action_context.{key}", torch.float32)


def close_data_loader(loader, iterator=None):
    # PyTorch retains persistent iterators after early exits, including exceptions.
    iterator = loader._iterator if iterator is None else iterator
    try:
        if iterator is not None and loader.num_workers > 0:
            iterator._shutdown_workers()
    finally:
        loader._iterator = None


def check_data_loader(loader, cfg, *, split, max_batches):
    iterator = None
    checked = 0
    try:
        iterator = iter(loader)
        for _ in range(min(max_batches, len(loader))):
            batch = next(iterator)
            validate_data_batch(batch, loader.dataset, cfg)
            fields = {"action": batch["action"]}
            fields.update({f"obs.{key}": value for key, value in batch["obs"].items()})
            fields.update(
                {
                    f"action_context.{key}": value
                    for key, value in batch.get("action_context", {}).items()
                }
            )
            summary = ", ".join(
                (
                    f"{key}={tuple(value.shape)} {value.dtype}"
                    if torch.is_tensor(value)
                    else f"{key}=list[str]({len(value)})"
                )
                for key, value in fields.items()
            )
            checked += 1
            print(f"[{split} batch {checked}] {summary}")
        if checked == 0:
            raise ValueError(f"{split}: no batches to check")
    finally:
        close_data_loader(loader, iterator)
    print(f"[{split}] windows={len(loader.dataset):,}, checked batches={checked}")


def validate_data_only(config_name: str, *, max_batches=2):
    positive_int(max_batches, "max_batches")
    print(f"\nData check (CPU): {config_name}")
    cfg = load_smoke_config(config_name)
    validate_config(cfg)
    if cfg.get("resume_from") is not None:
        raise ValueError("data-only checks current configs, not resume checkpoints")
    if cfg.training.get("num_gpus", 1) != 1:
        print("DDP config: single-process data check only; DDP NOT VERIFIED")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _ = build_dataset_and_normalizer(cfg)
    loader = build_train_loader(cfg, dataset, rank=0, world_size=1)
    check_data_loader(loader, cfg, split="train", max_batches=max_batches)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无 holdout")
    else:
        loader = DataLoader(validation, batch_size=cfg.dataloader.batch_size, num_workers=0)
        check_data_loader(loader, cfg, split="validation", max_batches=max_batches)
    print(f"✓ {config_name}: data check PASSED (no model/training execution)")


def validate_parameter_updates(model, before):
    changed = None
    for name, parameter in model.named_parameters():
        # Compilation inserts a wrapper into parameter names.
        name = name.replace("._orig_mod.", ".")
        if name not in before:
            continue
        current = parameter.detach().cpu()
        if not torch.isfinite(current).all():
            raise AssertionError(f"Non-finite learned parameter: {name}")
        if changed is None and not torch.equal(before[name], current):
            changed = name
    if changed is None:
        raise AssertionError("No finite learning-parameter update observed within the smoke budget")
    return changed


def validate_checkpoint(cfg, components, trainer, checkpoint_path):
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
        if model is None:
            continue
        actual = policy_state_dict(model)
        assert actual.keys() == state.keys()
        for key, tensor in actual.items():
            torch.testing.assert_close(tensor.detach().cpu(), state[key], rtol=0, atol=0)


def validate_prediction_roundtrip(cfg, components, trainer, checkpoint_path):
    """Compare raw/EMA predictions with restored policies after training has finished."""
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


def smoke_test(config_name: str, *, max_updates=4):
    """Run the production single-device Trainer with an execution-only budget."""
    positive_int(max_updates, "max_updates")
    cfg = load_smoke_config(config_name)
    validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("NOT VERIFIED: original training configuration requires CUDA")
    set_seed(cfg.training.seed)
    with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as tmpdir:
        cfg.workspace.output_dir = tmpdir
        cfg.workspace.wandb_cfg = None
        if cfg.training.get("num_gpus", 1) != 1:
            raise ValueError("Single-device smoke requires num_gpus=1")
        dataset, normalizer = build_dataset_and_normalizer(cfg)
        components = build_training_components(
            cfg, torch.device(cfg.training.device), dataset, normalizer
        )
        try:
            trainer = build_trainer(cfg, components)
            components.workspace.save_hydra_config(cfg)
            before = {
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in components.model.named_parameters()
                if parameter.requires_grad and parameter.is_floating_point()
            }
            if cfg.training.get("use_compile", False):
                compile_models(
                    components.model,
                    components.ema_model,
                    mode=cfg.training.get("compile_mode", "reduce-overhead"),
                )
            trainer.train(max_updates=max_updates)
            changed = validate_parameter_updates(components.model, before)
            del before
            print(f"Observed parameter update: {changed} (step={trainer.global_step})")

            checkpoint_path = components.workspace.checkpoint_dir / "latest.pt"
            validate_checkpoint(cfg, components, trainer, checkpoint_path)
            validate_prediction_roundtrip(cfg, components, trainer, checkpoint_path)
        finally:
            try:
                close_data_loader(components.train_loader)
            finally:
                components.workspace.close()
    print(f"✓ {config_name}: Trainer updates, prediction and raw/EMA roundtrip PASSED")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one or more DexMani Hydra policy configs."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Resolve/validate config, check Hydra target modules, and import the "
            "agent and training targets without data or GPU execution."
        ),
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

    for config_name in args.config_names:
        try:
            if args.config_only:
                mode = "config check"
                validate_config_only(config_name)
            elif args.data_only:
                mode = "data check"
                validate_data_only(config_name, max_batches=args.max_batches)
            else:
                mode = "smoke test"
                smoke_test(config_name, max_updates=args.max_updates)
        except Exception as exc:
            print(f"\n✗ {config_name} {mode} FAILED: {exc}\n")
            traceback.print_exc()
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
