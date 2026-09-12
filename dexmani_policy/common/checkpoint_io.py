"""Atomic training checkpoint I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

TRAIN_CHECKPOINT_FORMAT = "simple.v3"


@dataclass
class TrainCheckpoint:
    epoch: int
    global_step: int
    next_micro_step: int
    model_state: Dict[str, Any]
    ema_model_state: Optional[Dict[str, Any]]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    monitor: Dict[str, Any]
    resume_contract: Dict[str, Any]
    ema_updater_step: Optional[int]
    ema_decay: Optional[float]
    rng_states: list[Dict[str, Any]]


def build_agent_contract(model) -> Dict[str, Any]:
    """Agent metadata shared by training, evaluation and deployment export."""
    return {
        "n_obs_steps": model.n_obs_steps,
        "n_action_steps": model.n_action_steps,
        "action_dim": model.action_dim,
        "horizon": model.horizon,
        "action_key": model.action_key,
        "tcp_dim": getattr(model, "tcp_dim", None),
        "hand_dim": getattr(model, "hand_dim", None),
        "control_action_dim": model.control_action_dim,
        "use_aux_ee": bool(getattr(model, "use_aux_ee", False)),
    }


def validate_resume_contract(saved, current) -> None:
    """Report all missing, extra and changed values, including nested keys."""
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}"
                if key not in left:
                    differences.append(f"{child}: missing in checkpoint")
                elif key not in right:
                    differences.append(f"{child}: unexpected checkpoint key")
                else:
                    compare(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f"{path}: length saved={len(left)}, current={len(right)}")
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{i}]")
        elif type(left) is not type(right) or left != right:
            differences.append(f"{path}: saved={left!r}, current={right!r}")

    compare(saved, current, "resume_contract")
    if differences:
        raise ValueError("Resume contract mismatch:\n" + "\n".join(differences))


def validate_ema_resume_state(
    checkpoint: TrainCheckpoint, *, require_ema: bool
) -> None:
    """Require the complete EMA state needed to resume EMA training."""
    if not require_ema:
        return
    if checkpoint.ema_model_state is None:
        raise RuntimeError("Resume checkpoint is missing required ema_model_state")

    step = checkpoint.ema_updater_step
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(
            "Resume checkpoint ema_updater_step must be an int (not bool) >= 0"
        )


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
                "next_micro_step": checkpoint.next_micro_step,
                "monitor": checkpoint.monitor,
                "resume_contract": checkpoint.resume_contract,
                "ema_updater_step": checkpoint.ema_updater_step,
                "ema_decay": checkpoint.ema_decay,
                "rng_states": checkpoint.rng_states,
            },
            "weights": {
                "model": checkpoint.model_state,
                "ema_model": checkpoint.ema_model_state,
                "optimizer": checkpoint.optimizer_state,
                "scheduler": checkpoint.scheduler_state,
            },
            "_format": TRAIN_CHECKPOINT_FORMAT,
            "_saved_at": time.time(),
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return path

    def load(self, path: Path) -> TrainCheckpoint:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if set(payload) != {"state", "weights", "_format", "_saved_at"}:
            raise RuntimeError("Checkpoint root does not match the training schema")
        if payload.get("_format") != TRAIN_CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"Unsupported checkpoint format: {payload.get('_format')!r}"
            )
        state = payload["state"]
        weights = payload["weights"]
        expected_state = {
            "epoch",
            "global_step",
            "next_micro_step",
            "monitor",
            "resume_contract",
            "ema_updater_step",
            "ema_decay",
            "rng_states",
        }
        expected_weights = {"model", "ema_model", "optimizer", "scheduler"}
        if set(state) != expected_state or set(weights) != expected_weights:
            raise RuntimeError(
                f"Checkpoint does not match the {TRAIN_CHECKPOINT_FORMAT} schema"
            )
        for key in ("epoch", "global_step", "next_micro_step"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Checkpoint {key} must be an int >= 0")
        if not isinstance(state["resume_contract"], dict):
            raise ValueError("Checkpoint resume_contract must be a dict")
        if not isinstance(state["rng_states"], list) or not state["rng_states"]:
            raise ValueError("Checkpoint rng_states must be a nonempty rank-ordered list")
        return TrainCheckpoint(
            epoch=int(state["epoch"]),
            global_step=int(state["global_step"]),
            next_micro_step=state["next_micro_step"],
            monitor=state["monitor"],
            resume_contract=state["resume_contract"],
            ema_updater_step=state["ema_updater_step"],
            ema_decay=state["ema_decay"],
            rng_states=state["rng_states"],
            model_state=weights["model"],
            ema_model_state=weights["ema_model"],
            optimizer_state=weights["optimizer"],
            scheduler_state=weights["scheduler"],
        )

    def resolve_path(self, tag_or_path: str) -> Path:
        if tag_or_path == "latest":
            path = self.checkpoint_dir / "latest.pt"
        else:
            path = Path(tag_or_path)
            if path.is_absolute():
                # An absolute experiment directory resolves to its resume
                # checkpoint; an absolute .pt file is used directly.  This is
                # what `resume_from=<experiment_dir|checkpoint>` relies on.
                if path.is_dir():
                    path = path / "checkpoints" / "latest.pt"
            else:
                path = self.checkpoint_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
