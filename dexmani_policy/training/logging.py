import os
import json
import atexit
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

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

