"""Validate policy configs or exercise training and checkpoint restoration."""

import argparse
import importlib
import importlib.util
import tempfile
import traceback

import torch
from omegaconf import OmegaConf

from dexmani_policy.training.builder import build_train_components, build_trainer, compile_models
from dexmani_policy.training.common.checkpoint import (
    CheckpointStore,
    restore_policy_agent,
    unwrap_state_dict,
)
from dexmani_policy.training.validation import validate_config
from dexmani_policy.utils.config import (
    load_experiment_config,
    load_policy_config,
    positive_int,
    set_project_root,
)
from dexmani_policy.utils.runtime import dict_apply, set_seed

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
    checkpoint = CheckpointStore(checkpoint_path.parent).load(checkpoint_path)
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
        actual = unwrap_state_dict(model.state_dict())
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
        components = build_train_components(cfg)
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
            components.workspace.close()
    print(f"✓ {config_name}: Trainer updates, prediction and raw/EMA roundtrip PASSED")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one or more DexMani Hydra policy configs."
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Resolve/validate config, check Hydra target modules, and import the "
            "agent and training targets without data or GPU execution."
        ),
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
                validate_config_only(config_name)
            else:
                smoke_test(config_name, max_updates=args.max_updates)
        except Exception as exc:
            mode = "config check" if args.config_only else "smoke test"
            print(f"\n✗ {config_name} {mode} FAILED: {exc}\n")
            traceback.print_exc()
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
