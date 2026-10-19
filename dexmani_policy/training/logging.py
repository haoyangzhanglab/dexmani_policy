import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch

os.environ.setdefault("WANDB_SILENT", "true")


class JsonlLogger:
    def __init__(self, output_dir: Path, filename: str = "metrics.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_dir / filename, "a", buffering=1, encoding="utf-8")

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        record = {"step": int(step) if step is not None else None}
        record.update(data or {})
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        if self.file is None:
            return
        try:
            self.file.close()
        except OSError:
            pass
        finally:
            self.file = None


class WandbLogger:
    def __init__(
        self,
        output_dir: Path,
        project: str,
        name: str,
        group: str,
        mode: str,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

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

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        if self.run is None:
            return
        self.run.log(data, step=step, **kwargs)

    def log_config(self, cfg_dict: Dict[str, Any], output_dir: str):
        if self.run is None:
            return
        self.run.config.update(cfg_dict)
        self.run.config.update({"output_dir": str(output_dir)})

    def close(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except OSError:
            pass
        finally:
            self.run = None


def to_log_scalars(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = value.item()
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def count_params(module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts for *module*."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_count(agent) -> None:
    """Pretty-print parameter counts grouped by immediate children."""
    from termcolor import cprint

    total, _ = count_params(agent)
    fields = getattr(agent, "consumed_observation_fields", None)
    if fields is None:
        fields = getattr(getattr(agent, "obs_encoder", None), "consumed_observation_fields", ())
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
