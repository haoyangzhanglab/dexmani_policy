"""Check configs and data, or run bounded training and checkpoint resume."""

import argparse
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import load_policy_config, positive_int, set_project_root
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.common.config import validate_config


def load_smoke_config(config_name, overrides):
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


def validate_target_imports(cfg):
    targets = list(iter_hydra_targets(OmegaConf.to_container(cfg, resolve=True)))
    for target in targets:
        hydra.utils.get_method(target)
    return len(targets)


def iter_batch_fields(batch):
    for key, value in batch.items():
        if isinstance(value, dict):
            for field, tensor in value.items():
                yield f"{key}.{field}", tensor
        else:
            yield key, value


def validate_data_batch(batch, dataset, cfg):
    batch_size = len(batch["action"])
    assert batch["action"].shape == (batch_size, cfg.horizon, cfg.agent.action_dim), "action shape"
    assert batch["obs"].keys() == set(dataset.sensor_modalities), "observation fields"
    for key, value in batch["obs"].items():
        assert value.shape[:2] == (batch_size, dataset.obs_horizon), f"obs.{key} shape"
    for name, value in iter_batch_fields(batch):
        assert value.shape[0] == batch_size, f"{name}: batch size"
        assert value.device.type == "cpu", f"{name}: expected CPU data"
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all(), f"{name}: non-finite data"


def validate_data_loader(loader, cfg, split, max_batches):
    from dexmani_policy.training.common.data import close_data_loader

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
        assert checked, f"{split}: no batches to check"
    finally:
        close_resources(lambda: close_data_loader(loader, iterator))
    print(f"[{split}] windows={len(loader.dataset):,}, checked batches={checked}")


def validate_dataset(cfg, max_batches):
    from dexmani_policy.training.common.data import build_dataset, build_train_loader

    positive_int(max_batches, "max_batches")
    set_seed(cfg.training.seed)
    cfg.dataloader.pin_memory = False
    dataset, _ = build_dataset(cfg)
    validate_data_loader(build_train_loader(cfg, dataset), cfg, "train", max_batches)
    validation = dataset.get_validation_dataset()
    if validation is None:
        print("[validation] 无可用验证窗口")
    else:
        loader = DataLoader(validation, batch_size=cfg.dataloader.batch_size, num_workers=0)
        validate_data_loader(loader, cfg, "validation", max_batches)


def validate_parameter_updates(model, before):
    changed = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().cpu()
        assert torch.isfinite(current).all(), f"Non-finite parameter: {name}"
        if changed is None and not torch.equal(before[name], current):
            changed = name
    assert changed is not None, "No trainable parameter changed within the smoke budget"
    return changed


@contextmanager
def smoke_output(output_dir: Path | None) -> Iterator[Path]:
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as temporary:
            yield Path(temporary)
    else:
        directory = output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        yield directory


def validate_training(cfg, directory: Path, max_updates: int, resume_updates: int) -> int:
    from dexmani_policy.training.common.data import build_dataset
    from dexmani_policy.training.trainer import PolicyTrainer

    set_seed(cfg.training.seed)
    checkpoint = source = None
    phases = [("train", max_updates)]
    if resume_updates:
        phases.append(("resume", resume_updates))
    for phase, budget in phases:
        if checkpoint is not None:
            total_steps = checkpoint.effective_config["training"]["loop"]["total_train_steps"]
            if checkpoint.global_step == total_steps:
                print("[resume] 未执行：训练总预算已耗尽", flush=True)
                break
        run_cfg = OmegaConf.create(cfg if checkpoint is None else checkpoint.effective_config)
        with open_dict(run_cfg):
            run_cfg.workspace.output_dir = str(directory / phase)
            run_cfg.workspace.wandb_cfg = None
            if source is not None:
                run_cfg.resume_from = str(source)
        dataset, description = build_dataset(run_cfg)
        start_step = 0 if checkpoint is None else checkpoint.global_step
        with closing(PolicyTrainer(run_cfg, dataset, description, checkpoint=checkpoint)) as trainer:
            assert trainer.global_step == start_step, "Resume step mismatch"
            checkpoint = None
            trainer.compile_models()
            before = {
                name: parameter.detach().to(device="cpu", copy=True)
                for name, parameter in trainer.model.named_parameters() if parameter.requires_grad
            }
            trainer.train(max_updates=budget)
            assert trainer.global_step == min(start_step + budget, trainer.total_train_steps), "Training step mismatch"
            changed = validate_parameter_updates(trainer.model, before)
            del before
            step = trainer.global_step
            source = (trainer.checkpoint_dir / "latest.pt").resolve()
        del trainer
        checkpoint = load_checkpoint(source, mmap=True)
        assert checkpoint.global_step == step, "Checkpoint step mismatch"
        print(f"[{phase}] step={step}; updated {changed}; checkpoint={source}", flush=True)
    return step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--config-only", action="store_true", help="Check config interfaces and Hydra target imports.")
    modes.add_argument("--data-only", action="store_true", help="Check CPU data batches without building a model.")
    parser.add_argument("config_names", nargs="+", help="Hydra config names or explicit YAML paths.")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override.")
    parser.add_argument("--device", help="Override training.device.")
    parser.add_argument("--output-dir", type=Path, help="Keep training artifacts in a new directory.")
    parser.add_argument("--max-batches", type=int, default=2, help="Data-only batch budget per split (default: 2).")
    parser.add_argument("--max-updates", type=int, default=4, help="Initial training-update budget (default: 4).")
    parser.add_argument("--resume-updates", type=int, default=2, help="Additional updates; 0 disables resume (default: 2).")
    args = parser.parse_args(argv)
    training = not (args.config_only or args.data_only)
    if training:
        positive_int(args.max_updates, "max_updates")
        if args.resume_updates < 0:
            parser.error("--resume-updates must be nonnegative")

    set_project_root()
    if training and args.output_dir is not None and len(args.config_names) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    for config_name in args.config_names:
        print(f"\nValidating {config_name}", flush=True)
        cfg = load_smoke_config(config_name, args.override)
        if args.device is not None:
            cfg.training.device = args.device
        if args.config_only:
            summary = f"config and {validate_target_imports(cfg)} Hydra targets"
        elif args.data_only:
            validate_dataset(cfg, args.max_batches)
            summary = "CPU data batches"
        else:
            output_dir = args.output_dir
            if output_dir is not None and len(args.config_names) > 1:
                output_dir = output_dir / cfg.policy_name
            with smoke_output(output_dir) as directory:
                step = validate_training(cfg, directory, args.max_updates, args.resume_updates)
            summary = f"training/checkpoint smoke (step={step})"
        print(f"✓ {config_name}: {summary} PASSED", flush=True)


if __name__ == "__main__":
    main()
