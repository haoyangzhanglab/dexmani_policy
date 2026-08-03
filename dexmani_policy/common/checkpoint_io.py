"""Atomic training checkpoint I/O with persistent top-k score tracking."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

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


def build_train_params(model, num_training_steps=None) -> Dict[str, Any]:
    """Build the ``train_params`` metadata dict embedded in every checkpoint.

    This is the **single source of truth** for which agent attributes are
    serialised alongside the weights.  Both the trainer (save path) and
    the smoke test (roundtrip path) call this function so that the set of
    keys stays consistent.
    """
    params = {
        'n_obs_steps': model.n_obs_steps,
        'n_action_steps': model.n_action_steps,
        'action_dim': model.action_dim,
        'horizon': model.horizon,
        'action_key': getattr(model, 'action_key', 'action'),
        'tcp_dim': getattr(model, 'tcp_dim', None),
        'use_faas': getattr(model, 'use_faas', False),
        'hand_dim': getattr(model, 'hand_dim', None),
        'control_action_dim': model.control_action_dim,
        'num_training_steps': num_training_steps,
    }

    return params


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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return path

    def load(self, path: Path) -> TrainCheckpoint:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("_format") != "simple.v1":
            raise RuntimeError(
                f"Unsupported checkpoint format: {payload.get('_format')!r}"
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

    def resolve_path(self, tag_or_path: str, best_fn=None) -> Path:
        if tag_or_path == "latest":
            path = self.checkpoint_dir / "latest.pt"
        elif tag_or_path == "best":
            if best_fn is not None:
                path = best_fn()
            else:
                checkpoints = list(self.checkpoint_dir.glob("epoch=*.pt"))
                if not checkpoints:
                    raise FileNotFoundError(
                        f"No checkpoint found in {self.checkpoint_dir}"
                    )
                checkpoints.sort(
                    key=self._parse_ckpt_score, reverse=True
                )
                path = checkpoints[0]
            if path is None:
                raise FileNotFoundError(
                    f"No best checkpoint found in {self.checkpoint_dir}"
                )
        else:
            path = Path(tag_or_path)
            if not path.is_absolute():
                path = self.checkpoint_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    @staticmethod
    def _parse_ckpt_score(path: Path) -> float:
        match = re.search(r"-score=([\d.eE+-]+)\.pt$", path.name)
        return float(match.group(1)) if match else float("-inf")


class TopKCheckpointTracker:
    """Track top-k checkpoints across process restarts.

    Scores are persisted in ``scores.json``.  If the index is absent or stale,
    the tracker recovers the authoritative score from each checkpoint's
    ``state.monitor`` dictionary rather than trusting filesystem order.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor_key: str,
        mode: MonitorMode = "max",
        k: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = int(k)
        self.index_path = self.checkpoint_dir / "scores.json"
        self._score_cache: dict[str, float] = self._load_index()

    def _load_index(self) -> dict[str, float]:
        if not self.index_path.exists():
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if payload.get("monitor_key") != self.monitor_key:
                return {}
            return {
                str(name): float(score)
                for name, score in payload.get("scores", {}).items()
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _write_index(self) -> None:
        payload = {
            "monitor_key": self.monitor_key,
            "mode": self.mode,
            "scores": self._score_cache,
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.replace(tmp, self.index_path)

    def _list_ckpts(self) -> list[Path]:
        return list(self.checkpoint_dir.glob("epoch=*.pt"))

    def _invalid_score(self) -> float:
        return float("-inf" if self.mode == "max" else "inf")

    def _read_score_from_checkpoint(self, path: Path) -> float:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            monitor = payload.get("state", {}).get("monitor", {})
            score = monitor.get(self.monitor_key)
            if score is None:
                return self._invalid_score()
            return float(score)
        except (OSError, RuntimeError, ValueError, TypeError):
            return self._invalid_score()

    def _score(self, path: Path) -> float:
        if path.name in self._score_cache:
            return self._score_cache[path.name]

        # Backward-compatible filename fallback.
        match = re.search(r"-score=([\d.eE+-]+)\.pt$", path.name)
        if match:
            score = float(match.group(1))
        else:
            score = self._read_score_from_checkpoint(path)
        self._score_cache[path.name] = score
        return score

    def _sorted_ckpts(self) -> list[Path]:
        reverse = self.mode == "max"
        checkpoints = self._list_ckpts()
        sorted_paths = sorted(checkpoints, key=self._score, reverse=reverse)
        self._write_index()
        return sorted_paths

    def update(
        self,
        checkpoint_path: Path,
        checkpoint: Optional[TrainCheckpoint] = None,
    ) -> Optional[Path]:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint is not None:
            score = checkpoint.monitor.get(self.monitor_key)
            if score is not None:
                self._score_cache[checkpoint_path.name] = float(score)
        else:
            self._score_cache[checkpoint_path.name] = self._score(checkpoint_path)

        if self.k > 0:
            checkpoints = self._sorted_ckpts()
            for path in checkpoints[self.k :]:
                try:
                    path.unlink()
                except OSError:
                    pass
                self._score_cache.pop(path.name, None)
        self._write_index()
        return self.best_path()

    def best_path(self) -> Optional[Path]:
        checkpoints = self._sorted_ckpts()
        if not checkpoints:
            return None
        best = checkpoints[0]
        if self._score(best) == self._invalid_score():
            return None
        return best
