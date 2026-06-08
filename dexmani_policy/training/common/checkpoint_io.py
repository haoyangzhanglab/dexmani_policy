import re
import time
import copy
import torch
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

MonitorMode = Literal["max", "min"]


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_copy_to_cpu(v) for v in x]
    else:
        return copy.deepcopy(x)


@dataclass
class TrainCheckpoint:
    epoch: int
    global_step: int

    model_state: Dict[str, Any]
    ema_model_state: Optional[Dict[str, Any]]

    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]

    monitor: Dict[str, Any]
    train_params: Optional[Dict[str, Any]] = None



class CheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_thread: Optional[threading.Thread] = None

    def _save_payload(self, payload: Dict[str, Any], tmp_path: Path, final_path: Path):
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)
        torch.cuda.empty_cache()

    def save(self, filename: str, checkpoint: TrainCheckpoint,
             use_thread: bool = True) -> Path:
        path = self.checkpoint_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        # Wait for previous async save to finish before starting a new one
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        payload = {
            "state": {
                "epoch": int(checkpoint.epoch),
                "global_step": int(checkpoint.global_step),
                "monitor": checkpoint.monitor,
                "train_params": checkpoint.train_params,
            },
            "weights": {
                "model": _copy_to_cpu(checkpoint.model_state) if use_thread else checkpoint.model_state,
                "ema_model": _copy_to_cpu(checkpoint.ema_model_state) if use_thread and checkpoint.ema_model_state is not None else checkpoint.ema_model_state,
                "optimizer": _copy_to_cpu(checkpoint.optimizer_state) if use_thread else checkpoint.optimizer_state,
                "scheduler": _copy_to_cpu(checkpoint.scheduler_state) if use_thread else checkpoint.scheduler_state,
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }

        if use_thread:
            self._save_thread = threading.Thread(
                target=self._save_payload,
                args=(payload, tmp_path, path),
                daemon=True,
            )
            self._save_thread.start()
        else:
            self._save_payload(payload, tmp_path, path)

        return path

    def load(self, path: Path) -> TrainCheckpoint:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)

        fmt = payload.get("_format")
        if fmt != "simple.v1":
            raise RuntimeError(
                f"Unsupported checkpoint format: {fmt!r} (expected 'simple.v1'). "
                f"The checkpoint was saved by a different version of the training code."
            )

        state = payload["state"]
        weights = payload["weights"]

        return TrainCheckpoint(
            epoch=int(state["epoch"]),
            global_step=int(state["global_step"]),
            monitor=state.get("monitor", {}),
            train_params=state.get("train_params"),
            model_state=weights["model"],
            ema_model_state=weights.get("ema_model"),
            optimizer_state=weights["optimizer"],
            scheduler_state=weights["scheduler"],
        )

    def resolve_path(self, tag_or_path: str, best_fn=None):
        if tag_or_path == "latest":
            path = self.checkpoint_dir / "latest.pt"
        elif tag_or_path == "best":
            if best_fn is not None:
                path = best_fn()
            else:
                ckpts = list(self.checkpoint_dir.glob("epoch=*.pt"))
                if not ckpts:
                    raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")
                ckpts.sort(key=lambda p: self._parse_ckpt_score(p), reverse=True)
                path = ckpts[0]
            if path is None:
                raise FileNotFoundError(f"No best checkpoint found in {self.checkpoint_dir}.")
        else:
            path = Path(tag_or_path)
            if not path.is_absolute():
                path = self.checkpoint_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    @staticmethod
    def _parse_ckpt_score(path: Path) -> float:
        m = re.search(r'-score=([\d.eE+-]+)\.pt$', path.name)
        return float(m.group(1)) if m else float("-inf")



class TopKCheckpointTracker:
    def __init__(
        self,
        checkpoint_dir: Path,
        monitor_key: str,
        mode: MonitorMode = "max",
        k: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = int(k)

    def _list_ckpts(self):
        return list(self.checkpoint_dir.glob("epoch=*.pt"))

    def _parse_score(self, path: Path) -> float:
        m = re.search(r'-score=([\d.eE+-]+)\.pt$', path.name)
        if m:
            return float(m.group(1))
        return float("-inf" if self.mode == "max" else "inf")

    def _sorted_ckpts(self):
        reverse = (self.mode == "max")
        return sorted(self._list_ckpts(), key=self._parse_score, reverse=reverse)

    def update(self, checkpoint_path: Path, checkpoint: TrainCheckpoint) -> Optional[Path]:
        if self.k <= 0:
            return self.best_path()

        ckpts = self._sorted_ckpts()
        for p in ckpts[self.k:]:
            try:
                p.unlink()
            except OSError:
                pass
        return self.best_path()

    def best_path(self) -> Optional[Path]:
        ckpts = self._sorted_ckpts()
        return ckpts[0] if ckpts else None