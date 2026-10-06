import atexit
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

os.environ.setdefault("WANDB_SILENT", "true")


def is_video_key(key: Any) -> bool:
    return "video" in str(key).lower()


class JsonlLogger:
    def __init__(self, output_dir: Path, filename: str = "metrics.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_dir / filename, "a", buffering=1, encoding="utf-8")
        atexit.register(self.close)

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        record = {"step": int(step) if step is not None else None}
        for key, value in (data or {}).items():
            if not is_video_key(key):
                record[key] = value
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
        id: str,
        resume: str,
        mode: str,
        video_fps: int = 15,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            dir=str(self.output_dir),
            project=project,
            name=name,
            group=group,
            id=id,
            resume=resume,
            mode=mode,
        )
        self.video_fps = int(video_fps)

        atexit.register(self.close)

    def format_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data or {})
        for key, value in list(payload.items()):
            if not is_video_key(key):
                continue
            if not isinstance(value, np.ndarray) or value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError(f"Key '{key}' must be a NumPy array with shape (T, H, W, 3).")
            payload[key] = self._wandb.Video(
                np.transpose(value, (0, 3, 1, 2)),
                fps=self.video_fps,
                format="mp4",
            )
        return payload

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        if self.run is None:
            return
        self.run.log(self.format_payload(data), step=step, **kwargs)

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
    observation = getattr(agent, "obs_encoder", None)
    if observation is not None:
        fields = getattr(observation, "consumed_observation_fields", ())
        cprint(f"  Actual observation fields: {tuple(fields)}", "white")
        pointcloud = getattr(observation, "pc_encoder", None)
        if pointcloud is not None:
            values = {key: getattr(pointcloud, key) for key in (
                "input_channels", "pc_in_channels", "out_dim", "hidden_channels",
                "num_layers", "num_group", "num_patches", "include_global_token",
                "feature_mode",
            ) if hasattr(pointcloud, key)}
            if hasattr(pointcloud, "stages"):
                values["stages"] = len(pointcloud.stages)
            cprint(f"  Actual point-cloud module: {type(pointcloud).__name__} {values}", "white")
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


def print_storage_dtypes(model, ema_model, *, autocast):
    """Startup-only metadata; reading dtype does not synchronize tensor values."""
    def groups(module):
        result = {"parameters": set(), "lora": set(), "frozen_backbone": set()}
        if module is not None:
            for name, param in module.named_parameters():
                result["parameters"].add(str(param.dtype))
                if "lora_" in name:
                    result["lora"].add(str(param.dtype))
                elif "backbone" in name and not param.requires_grad:
                    result["frozen_backbone"].add(str(param.dtype))
        return {key: sorted(value) for key, value in result.items()}
    print(f"Storage dtype: model={groups(model)}, EMA={groups(ema_model)}, "
          f"BF16 autocast={autocast}")
