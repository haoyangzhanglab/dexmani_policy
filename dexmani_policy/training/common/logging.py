import os
import sys
import json
import torch
import atexit
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("WANDB_SILENT", "true")


def is_video_key(key: Any) -> bool:
    return "video" in str(key).lower()


def to_log_scalars(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tensor_keys = []
    tensor_vals = []

    for key, value in (metrics or {}).items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                tensor_keys.append(key)
                tensor_vals.append(value.detach())
            elif value.dim() == 1:
                if value.shape[0] <= 50:
                    cpu_vec = value.detach().float().cpu().tolist()
                    for i, val in enumerate(cpu_vec):
                        out[f"{key}/{i}"] = float(val)
                else:
                    v = value.detach().float()
                    out[f"{key}/mean"] = v.mean().item()
                    out[f"{key}/std"] = v.std().item()
            else:
                v = value.detach().float()
                out[f"{key}/mean"] = v.mean().item()
                out[f"{key}/std"] = v.std().item()
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            pass

    if tensor_vals:
        cpu_vals = torch.stack(tensor_vals).float().cpu().tolist()
        for key, value in zip(tensor_keys, cpu_vals):
            out[key] = float(value)

    return out


class Logger:
    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError



class JsonlLogger(Logger):
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
        except Exception:
            pass
        finally:
            self.file = None



class WandbLogger(Logger):
    def __init__(
        self,
        output_dir: Path,
        project: str,
        name: str,
        group: str,
        id: str,
        resume: str,
        mode: str,
        video_fps: int,
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


    def close(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass
        finally:
            self.run = None



class MultiLogger(Logger):
    def __init__(self, loggers: Iterable[Logger]):
        self.loggers = list(loggers)

    # 将同一份日志分发到多个 logger。
    def log(self, data: Dict[str, Any], step: Optional[int] = None, **kwargs):
        for logger in self.loggers:
            try:
                logger.log(data, step=step, **kwargs)
            except Exception as e:
                print(f"[MultiLogger] {logger.__class__.__name__}.log failed: {e}", file=sys.stderr)

    def close(self):
        for logger in self.loggers:
            try:
                logger.close()
            except Exception as e:
                print(f"[MultiLogger] {logger.__class__.__name__}.close failed: {e}", file=sys.stderr)