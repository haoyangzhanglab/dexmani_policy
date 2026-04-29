import json
import math
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
    ema_updater_state: Optional[Dict[str, Any]]

    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]

    monitor: Dict[str, Any]



class CheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _serialize(self, checkpoint: TrainCheckpoint) -> Dict[str, Any]:
        return {
            "state": {
                "epoch": int(checkpoint.epoch),
                "global_step": int(checkpoint.global_step),
                "monitor": checkpoint.monitor,
            },
            "weights": {
                "model": checkpoint.model_state,
                "ema_model": checkpoint.ema_model_state,
                "ema_updater": checkpoint.ema_updater_state,
                "optimizer": checkpoint.optimizer_state,
                "scheduler": checkpoint.scheduler_state,
            },
            "_format": "simple.v1",
            "_saved_at": time.time(),
        }

    def _deserialize(self, payload: Dict[str, Any]) -> TrainCheckpoint:
        state = payload["state"]
        weights = payload["weights"]

        return TrainCheckpoint(
            epoch=int(state["epoch"]),
            global_step=int(state["global_step"]),
            monitor=state.get("monitor", {}),
            model_state=weights["model"],
            ema_model_state=weights.get("ema_model"),
            ema_updater_state=weights.get("ema_updater"),
            optimizer_state=weights["optimizer"],
            scheduler_state=weights["scheduler"],
        )

    def save(self, filename: str, checkpoint: TrainCheckpoint) -> Path:
        path = self.checkpoint_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._serialize(checkpoint), tmp_path)
        tmp_path.replace(path)
        return path

    def load(self, path: Path) -> TrainCheckpoint:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return self._deserialize(payload)



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

        self.manifest_path = self.checkpoint_dir / "topk_manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"items": []}
        try:
            return json.loads(self.manifest_path.read_text("utf-8"))
        except Exception:
            return {"items": []}

    def _save_manifest(self) -> None:
        tmp_path = self.manifest_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.manifest_path)

    def _fallback_score(self) -> float:
        return float("-inf" if self.mode == "max" else "inf")

    def _normalize_score(self, value: Any) -> float:
        if value is None:
            return self._fallback_score()

        try:
            score = float(value)
        except (TypeError, ValueError):
            return self._fallback_score()

        return score if math.isfinite(score) else self._fallback_score()

    def _sort_key(self, item: Dict[str, Any]):
        score = item["score"]
        epoch = item["epoch"]

        if self.mode == "max":
            return (-score, -epoch)
        return (score, -epoch)

    def _remove_extra_checkpoints(self, items) -> None:
        for item in items:
            path = self.checkpoint_dir / item["path"]
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def update(self, checkpoint_path: Path, checkpoint: TrainCheckpoint) -> Optional[Path]:
        if self.k <= 0:
            return self.best_path()

        items = list(self._manifest.get("items", []))
        items.append(
            {
                "path": Path(checkpoint_path).name,
                "score": self._normalize_score(checkpoint.monitor.get(self.monitor_key)),
                "step": int(checkpoint.global_step),
                "epoch": int(checkpoint.epoch),
            }
        )

        items.sort(key=self._sort_key)

        if len(items) > self.k:
            self._remove_extra_checkpoints(items[self.k :])
            items = items[: self.k]

        self._manifest["items"] = items
        self._save_manifest()
        return self.best_path()


    def best_path(self) -> Optional[Path]:
        items = self._manifest.get("items", [])
        if not items:
            return None
        return self.checkpoint_dir / items[0]["path"]