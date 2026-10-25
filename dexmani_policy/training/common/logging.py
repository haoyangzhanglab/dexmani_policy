import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("WANDB_SILENT", "true")


class JsonlLogger:
    def __init__(self, output_dir: Path, filename: str = "metrics.jsonl"):
        self.output_dir = Path(output_dir)
        self.file = open(self.output_dir / filename, "a", buffering=1, encoding="utf-8")

    def log(self, data: dict[str, Any], step: int | None = None):
        record = {"step": int(step) if step is not None else None}
        record.update(data)
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self.file.close()


class WandbLogger:
    def __init__(
        self,
        output_dir: Path,
        project: str,
        name: str,
        group: str,
        mode: str,
    ):
        import wandb

        self.output_dir = Path(output_dir)

        # Checkpoint continuation belongs to a new logging run, including repeated sweep names.
        self.run = wandb.init(
            dir=str(self.output_dir),
            project=project,
            name=name,
            group=group,
            id=uuid.uuid4().hex,
            resume="never",
            mode=mode,
        )

    def log(self, data: dict[str, Any], step: int | None = None):
        self.run.log(data, step=step)

    def log_config(self, cfg_dict: dict[str, Any]):
        config = {**cfg_dict, "output_dir": str(self.output_dir)}
        self.run.config.update(config)

    def close(self):
        self.run.finish()


def accumulate_metrics(totals, metrics):
    """Accumulate detached scalars without synchronizing the device."""
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().reshape(()).to(dtype=torch.float64)
        else:
            value = float(value)
        totals[key] = totals.get(key, 0.0) + value


def to_log_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    scalars = {}
    tensors = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            keys, values = tensors.setdefault(value.device, ([], []))
            keys.append(key)
            values.append(value.detach().reshape(()).to(dtype=torch.float64))
        else:
            scalars[key] = float(value)
    for keys, values in tensors.values():
        scalars.update(zip(keys, torch.stack(values).cpu().tolist()))
    return scalars


def count_params(module) -> tuple[int, int]:
    total = trainable = 0
    for parameter in module.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def print_param_count(agent) -> None:
    from termcolor import cprint

    total, _ = count_params(agent)
    fields = agent.consumed_observation_fields
    if fields:
        cprint(f"[Modalities] {', '.join(fields)}", "white")
    cprint(f"[{type(agent).__name__}] Parameter Count", "cyan", attrs=["bold"])
    cprint(f"  Total: {total / 1e6:.2f} M", "white")

    for name, child in agent.named_children():
        t, tr = count_params(child)
        frozen = t - tr
        color = "green" if tr > 0 else "white"
        cprint(
            f"  {name:<20}: {t / 1e6:.2f} M  (trainable={tr / 1e6:.2f} M  frozen={frozen / 1e6:.2f} M)",
            color,
        )


def print_training_recipe(
    cfg, *, world_size: int, batches_per_epoch: int, samples_per_rank: int
) -> None:
    """Distinguish nominal batches from the windows actually processed."""
    per_device_batch = int(cfg.dataloader.batch_size)
    grad_accum = cfg.training.loop.gradient_accumulation_steps
    total_train_steps = int(cfg.training.loop.total_train_steps)
    nominal_global_batch = per_device_batch * world_size * grad_accum
    updates_per_epoch = (batches_per_epoch + grad_accum - 1) // grad_accum
    remainder = batches_per_epoch % grad_accum
    last_group_microbatches = grad_accum if remainder == 0 else remainder
    epoch_samples = min(samples_per_rank, batches_per_epoch * per_device_batch)
    last_group_global_batch = world_size * (
        epoch_samples - (updates_per_epoch - 1) * grad_accum * per_device_batch
    )
    full_epochs, partial_updates = divmod(total_train_steps, updates_per_epoch)
    processed_budget = world_size * (
        full_epochs * epoch_samples
        + min(partial_updates * grad_accum * per_device_batch, epoch_samples)
    )
    has_partial_accumulation_group = last_group_microbatches != grad_accum
    lr = cfg.optimizer.lr
    obs_lr = cfg.optimizer.get("obs_lr")
    obs_lr = lr if obs_lr is None else obs_lr

    rows = [
        (
            "temporal sampling",
            "recorded_rows (no interpolation; eligibility follows data_recipe)",
        ),
        ("per-device batch", per_device_batch),
        ("world size", world_size),
        ("gradient accumulation", grad_accum),
        ("nominal global batch", nominal_global_batch),
        ("batches / epoch", batches_per_epoch),
        ("optimizer updates / epoch", updates_per_epoch),
        ("partial accum group", "yes" if has_partial_accumulation_group else "no"),
        ("last group micro-batches", last_group_microbatches),
        ("last group global batch", last_group_global_batch),
        ("total optimizer updates", total_train_steps),
        ("nominal sample budget", nominal_global_batch * total_train_steps),
        ("processed window budget", processed_budget),
        ("window counting", "includes sampler repeats / DDP padding; not unique source rows"),
        ("drop_last", cfg.dataloader.drop_last),
        ("learning rate", f"{lr:.3e}"),
        ("obs learning rate", f"{obs_lr:.3e}"),
        ("warmup updates", cfg.training.lr_warmup_steps),
        ("scheduler", cfg.training.lr_scheduler),
        ("precision", "bf16" if cfg.training.get("use_bfloat16", False) else "fp32"),
        ("torch.compile", cfg.training.get("use_compile", False)),
        ("EMA", cfg.training.use_ema),
    ]
    if "contract" in cfg.get("hand_data", {}):
        dt = cfg.hand_data.contract.dt
        rows.extend([
            ("recorded dt (s)", dt),
            ("prediction control duration (s)", cfg.horizon * dt),
            ("execution control duration (s)", cfg.n_action_steps * dt),
            ("codec precision", "fp32; policy master parameters fp32, autocast as above"),
        ])
    print("=" * 60)
    print("Training Recipe")
    print("-" * 60)
    for label, value in rows:
        print(f"{label:<26}: {value}")
    print("=" * 60)
