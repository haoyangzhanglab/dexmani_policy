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
    resume_contract: Dict[str, Any]
    ema_updater_step: Optional[int]
    ema_decay: Optional[float]
    rng_states: list[Dict[str, Any]]


class CheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, filename: str, checkpoint: TrainCheckpoint) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "state": {
                "epoch": int(checkpoint.epoch),
                "global_step": int(checkpoint.global_step),
                "next_micro_step": checkpoint.next_micro_step,
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
            raise ValueError(
                "Checkpoint rng_states must be a nonempty rank-ordered list"
            )
        return TrainCheckpoint(
            epoch=int(state["epoch"]),
            global_step=int(state["global_step"]),
            next_micro_step=state["next_micro_step"],
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


def fix_state_dict(state_dict: Dict, is_current_ddp: bool) -> Dict:
    # Strip _orig_mod. from keys wherever it appears.
    # torch.compile on a submodule produces "child._orig_mod.param";
    # torch.compile on the top-level model produces "_orig_mod.param".
    # Both can coexist with DDP: "_orig_mod.module.param".
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    first_key = next(iter(state_dict.keys()))
    is_checkpoint_ddp = first_key.startswith("module.")

    if is_checkpoint_ddp and not is_current_ddp:
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    elif not is_checkpoint_ddp and is_current_ddp:
        return {f"module.{k}": v for k, v in state_dict.items()}

    return state_dict
