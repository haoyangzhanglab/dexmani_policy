import json
import os
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dexmani_policy.common.io_util import atomic_path, close_resources


class ExperimentLogger:
    def __init__(self, output_dir: str, wandb_cfg: Mapping | None = None, *,
                 resume_step=None, wandb_enabled=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resuming = resume_step is not None
        if self.resuming:
            path = self.output_dir / "metrics.jsonl"
            if path.exists():
                lines = path.read_text().splitlines(keepends=True)
                kept = []
                for line in lines:
                    step = json.loads(line)["step"]
                    if step is None or step <= resume_step:
                        kept.append(line)
                if len(kept) != len(lines):
                    backup = path.with_name(f"metrics.{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl.bak")
                    shutil.copy2(path, backup)
                    with atomic_path(path) as temporary:
                        temporary.write_text("".join(kept))
        else:
            for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
                if (self.output_dir / name).exists():
                    raise FileExistsError(f"Training output already exists: {self.output_dir / name}")

        self.metrics_file = (self.output_dir / "metrics.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )
        self.wandb_run = None
        if wandb_enabled and wandb_cfg is not None:
            try:
                os.environ.setdefault("WANDB_SILENT", "true")
                import wandb

                # Continuation creates a new logging run, including repeated sweep names.
                self.wandb_run = wandb.init(
                    dir=str(self.output_dir), id=uuid.uuid4().hex, resume="never", **wandb_cfg
                )
            except BaseException:
                self.close()
                raise

    def save_config(self, hydra_config):
        if not self.resuming:
            with atomic_path(self.output_dir / "config.yaml", overwrite=False) as temporary:
                OmegaConf.save(hydra_config, temporary, resolve=True)
        if self.wandb_run is not None:
            cfg_dict = OmegaConf.to_container(hydra_config, resolve=True)
            self.wandb_run.config.update(cfg_dict)

    def log(self, data: dict[str, Any], step: int | None = None):
        record = {"step": int(step) if step is not None else None, **data}
        self.metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(data, step=step)

    def close(self):
        callbacks = [self.metrics_file.close]
        if self.wandb_run is not None:
            callbacks.append(self.wandb_run.finish)
        close_resources(*callbacks)


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
    tensors: dict[torch.device, tuple[list[str], list[torch.Tensor]]] = {}
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
    total, _ = count_params(agent)
    fields = agent.consumed_observation_fields
    if fields:
        print(f"[Modalities] {', '.join(fields)}")
    print(f"[{type(agent).__name__}] Parameter Count")
    print(f"  Total: {total / 1e6:.2f} M")

    for name, child in agent.named_children():
        t, tr = count_params(child)
        frozen = t - tr
        print(
            f"  {name:<20}: {t / 1e6:.2f} M  "
            f"(trainable={tr / 1e6:.2f} M  frozen={frozen / 1e6:.2f} M)"
        )


def print_training_recipe(cfg, *, batches_per_epoch, samples_per_epoch):
    batch = cfg.dataloader.batch_size
    accumulation = cfg.training.loop.gradient_accumulation_steps
    updates = (batches_per_epoch + accumulation - 1) // accumulation
    samples = min(samples_per_epoch, batches_per_epoch * batch)
    tail = samples - (updates - 1) * accumulation * batch
    print(
        f"[Training] batch={batch}, accumulation={accumulation}, "
        f"updates/epoch={updates}, last group samples={tail}, "
        f"total updates={cfg.training.loop.total_train_steps}"
    )
    print(
        f"[Optimizer] lr={cfg.optimizer.lr:g}, scheduler={cfg.training.lr_scheduler}, "
        f"warmup={cfg.training.lr_warmup_steps}, EMA={cfg.training.use_ema}, "
        f"bf16={cfg.training.get('use_bfloat16', False)}, "
        f"compile={cfg.training.get('use_compile', False)}"
    )
