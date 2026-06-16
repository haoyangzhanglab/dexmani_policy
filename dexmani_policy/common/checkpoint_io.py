import re
import time
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

MonitorMode = Literal["max", "min"]


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

    def save(self, filename: str, checkpoint: TrainCheckpoint) -> Path:
        path = self.checkpoint_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        payload = {
            "state": {
                "epoch": int(checkpoint.epoch),
                "global_step": int(checkpoint.global_step),
                "monitor": checkpoint.monitor,
                "train_params": checkpoint.train_params,
            },
            "weights": {
                "model": checkpoint.model_state,
                "ema_model": checkpoint.ema_model_state,
                "optimizer": checkpoint.optimizer_state,
                "scheduler": checkpoint.scheduler_state,
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }

        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        torch.cuda.empty_cache()
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
        # Cache: path → score populated from checkpoint.monitor during update().
        # Falls back to filename regex when entry is not in cache (e.g. during
        # eval-time "best" resolution where no checkpoint object is available).
        self._score_cache: dict[Path, float] = {}

    def _list_ckpts(self):
        return list(self.checkpoint_dir.glob("epoch=*.pt"))

    def _parse_score(self, path: Path) -> float:
        # Use cached score from checkpoint.monitor if available.
        if path in self._score_cache:
            return self._score_cache[path]
        # Fall back to filename regex for backward compatibility
        # and eval-time "best" resolution.
        m = re.search(r'-score=([\d.eE+-]+)\.pt$', path.name)
        if m:
            return float(m.group(1))
        return float("-inf" if self.mode == "max" else "inf")

    def _sorted_ckpts(self):
        reverse = (self.mode == "max")
        return sorted(self._list_ckpts(), key=self._parse_score, reverse=reverse)

    def update(self, checkpoint_path: Path, checkpoint: Optional[TrainCheckpoint] = None) -> Optional[Path]:
        # Populate score cache from the authoritative source: checkpoint.monitor.
        if checkpoint is not None and self.monitor_key in checkpoint.monitor:
            score = checkpoint.monitor[self.monitor_key]
            if score is not None:
                self._score_cache[checkpoint_path] = float(score)

        if self.k <= 0:
            return self.best_path()

        ckpts = self._sorted_ckpts()
        for p in ckpts[self.k:]:
            try:
                p.unlink()
            except OSError:
                pass
            self._score_cache.pop(p, None)  # purge deleted entries
        return self.best_path()

    def best_path(self) -> Optional[Path]:
        ckpts = self._sorted_ckpts()
        return ckpts[0] if ckpts else None