from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import tempfile
import traceback

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dexmani_policy.training.checkpoint import (
    CheckpointStore,
    fix_state_dict,
)
from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.utils.tensor import dict_apply
from dexmani_policy.utils.random import set_seed
from dexmani_policy.training.build_utils import (
    validate_config,
)

register_resolvers()


def load_config(config_name: str):
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    config_dir = os.path.join(ROOT_DIR, "dexmani_policy", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        # compose has no Hydra runtime; provide the only runtime value used by configs.
        cfg.workspace.output_dir = "/tmp/smoke_test_output"
        OmegaConf.resolve(cfg)
    return cfg


def _iter_targets(node, path: str = "$"):
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)

    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str):
            yield path, target
        for key, value in node.items():
            yield from _iter_targets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_targets(value, f"{path}[{index}]")


def _split_target(target: str) -> tuple[str, str]:
    module_name, separator, attribute = target.rpartition(".")
    if not separator:
        raise ImportError(f"Invalid Hydra target: {target!r}")
    return module_name, attribute


def _validate_target_references(cfg) -> int:
    """Check target modules cheaply; import only the policy Agent target.

    Some runtime targets (notably env runners) intentionally import optional
    external packages such as dexmani_sim. Config-only validation should not
    require those runtime dependencies just to prove the policy config is
    structurally wired.
    """
    targets = list(_iter_targets(cfg))
    agent_target = cfg.get("agent", {}).get("_target_")

    for path, target in targets:
        module_name, attribute = _split_target(target)
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"Failed to locate Hydra target module {module_name!r} "
                f"for {target!r} at {path}"
            ) from exc
        if spec is None:
            raise ImportError(
                f"Failed to locate Hydra target module {module_name!r} "
                f"for {target!r} at {path}"
            )

        if target == agent_target:
            try:
                module = importlib.import_module(module_name)
                getattr(module, attribute)
            except Exception as exc:
                raise ImportError(
                    f"Failed to import agent target {target!r} at {path}"
                ) from exc

    return len(targets)


def validate_config_only(config_name: str):
    print(f"\n{'=' * 60}")
    print(f"Config check: {config_name}")
    print(f"{'=' * 60}")

    cfg = load_config(config_name)
    validate_config(cfg)

    target_count = _validate_target_references(cfg)

    print(f"      checked Hydra targets: {target_count}")
    print(f"\n✓ {config_name} config check PASSED\n")
    return True


def smoke_test(config_name: str, *, max_updates=4):
    """Run the production single-device Trainer with an execution-only budget."""
    from dexmani_policy.train import build_train_components, build_trainer
    from dexmani_policy.agents.loader import load_experiment_config, restore_policy_agent
    from dexmani_policy.training.resume import optimizer_to
    from dexmani_policy.utils.validation import positive_int

    positive_int(max_updates, "max_updates")
    cfg = load_config(config_name)
    validate_config(cfg)
    if cfg.training.get("num_gpus", 1) > 1:
        raise RuntimeError("NOT VERIFIED: multi-rank configs require a bounded train_ddp run")
    if not torch.cuda.is_available():
        raise RuntimeError("NOT VERIFIED: original training configuration requires CUDA")
    set_seed(cfg.training.seed)
    with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as tmpdir:
        cfg.workspace.output_dir = tmpdir
        cfg.workspace.wandb_cfg = None
        comp = build_train_components(cfg)
        trainer = build_trainer(cfg, comp)
        try:
            comp.workspace.save_hydra_config(cfg)
            before = {name: p.detach().cpu().clone()
                      for name, p in comp.model.named_parameters()
                      if p.requires_grad and p.is_floating_point()}
            trainer.train(max_updates=max_updates)
            changed = []
            for name, p in comp.model.named_parameters():
                # compile wraps submodules; checkpoint naming removes the wrapper.
                name = name.replace("._orig_mod.", ".")
                if name in before:
                    current = p.detach().cpu()
                    if not torch.isfinite(current).all():
                        raise AssertionError(f"Non-finite learned parameter: {name}")
                    if not torch.equal(before[name], current):
                        changed.append(name)
            del before
            if not changed:
                raise AssertionError("No finite learning-parameter update observed within the smoke budget")
            print(f"Observed parameter update: {changed[0]} (step={trainer.global_step})")

            path = comp.workspace.resolve_checkpoint_path("latest")
            checkpoint = CheckpointStore(path.parent).load(path)
            assert checkpoint.global_step == trainer.global_step
            assert checkpoint.epoch == trainer.current_epoch
            assert checkpoint.next_micro_step == trainer.next_micro_step
            assert comp.scheduler.state_dict() == checkpoint.scheduler_state
            assert checkpoint.resume_contract["training"]["loop"]["total_train_steps"] == cfg.training.loop.total_train_steps
            if comp.ema_updater is not None:
                assert checkpoint.ema_updater_step == comp.ema_updater.optimization_step
            for selected, state in ((comp.model, checkpoint.model_state),
                                    (comp.ema_model, checkpoint.ema_model_state)):
                if selected is not None:
                    actual = fix_state_dict(selected.state_dict(), False)
                    assert actual.keys() == state.keys()
                    for key, tensor in actual.items():
                        torch.testing.assert_close(tensor.detach().cpu(), state[key], rtol=0, atol=0)
            del checkpoint

            # One real batch, unchanged preprocessing; inference uses one observation.
            comp.train_loader.sampler.set_epoch(trainer.current_epoch, 0)
            batch = next(iter(comp.train_loader))
            obs = dict_apply({k: v[:1] for k, v in batch["obs"].items()},
                             lambda x: x.to(comp.device))
            predictions = {}
            for use_ema, selected in ((False, comp.model), (True, comp.ema_model)):
                if selected is None:
                    continue
                selected.eval()
                set_seed(cfg.training.seed)
                with torch.inference_mode():
                    pred = selected.predict_action(obs)["control_action"]
                assert pred.shape == (1, cfg.n_action_steps, selected.control_action_dim)
                assert torch.isfinite(pred).all()
                predictions[use_ema] = pred.cpu()
                selected.to("cpu")
            optimizer_to(comp.optimizer, "cpu")
            saved = load_experiment_config(tmpdir)
            for use_ema, expected in predictions.items():
                restored = restore_policy_agent(saved, path, use_ema=use_ema, device=comp.device)
                set_seed(cfg.training.seed)
                with torch.inference_mode():
                    actual = restored.predict_action(obs)["control_action"].cpu()
                torch.testing.assert_close(actual, expected)
                del restored
        finally:
            comp.workspace.close()
    print(f"✓ {config_name}: Trainer updates, prediction and raw/EMA roundtrip PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate one or more DexMani Hydra policy configs."
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Resolve/validate config, check Hydra target modules, and import the "
            "agent target without data or GPU execution."
        ),
    )
    parser.add_argument(
        "config_names", nargs="+", help="Hydra config names to validate."
    )
    parser.add_argument("--max-updates", type=int, default=4, help="Execution-only optimizer-update budget (default: 4).")
    args = parser.parse_args()

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
