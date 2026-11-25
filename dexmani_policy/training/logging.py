import json
import os
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dexmani_policy.common.io_util import atomic_path, cleanup_on_exit


class ExperimentLogger:
    def __init__(self, cfg, *, resuming: bool = False):
        workspace = cfg.workspace
        self.output_dir = Path(workspace.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            self.metrics_file = (self.output_dir / "metrics.jsonl").open("a", buffering=1, encoding="utf-8")
            resources.enter_context(cleanup_on_exit(self.metrics_file.close))
            self.wandb_run = None
            wandb_cfg = workspace.get("wandb_cfg")
            if workspace.get("wandb_enabled", True) and wandb_cfg is not None:
                os.environ.setdefault("WANDB_SILENT", "true")
                import wandb

                # Continuation creates a new logging run, including repeated sweep names.
                self.wandb_run = wandb.init(
                    dir=str(self.output_dir), id=uuid.uuid4().hex, resume="never", **wandb_cfg
                )
                resources.enter_context(cleanup_on_exit(self.wandb_run.finish))
            if not resuming:
                with atomic_path(self.output_dir / "config.yaml") as temporary:
                    OmegaConf.save(cfg, temporary, resolve=True)
            if self.wandb_run is not None:
                self.wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True))
            self._resources = resources.pop_all()

    def log(self, data: dict[str, Any], step: int):
        record = {"step": step, **data}
        self.metrics_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(data, step=step)

    def close(self):
        self._resources.close()


def accumulate_metrics(totals, metrics, *, weight: int) -> None:
    """Accumulate detached scalars without synchronizing the device."""
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().reshape(()).to(dtype=torch.float64)
        else:
            value = float(value)
        totals[key] = totals.get(key, 0.0) + value * weight


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


def print_param_count(agent) -> None:
    total = trainable = 0
    for parameter in agent.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    print(f"[{type(agent).__name__}] parameters={total / 1e6:.2f} M, trainable={trainable / 1e6:.2f} M")


def print_training_recipe(cfg, *, updates_per_epoch):
    batch = cfg.dataloader.batch_size
    accumulation = cfg.training.loop.gradient_accumulation_steps
    print(
        f"[Training] batch={batch}, accumulation={accumulation}, "
        f"updates/epoch={updates_per_epoch}, "
        f"total updates={cfg.training.loop.total_train_steps}"
    )
