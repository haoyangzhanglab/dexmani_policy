"""Atomic training checkpoint I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dexmani_policy.utils.atomic import atomic_path

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
        with atomic_path(path) as temporary:
            torch.save(payload, temporary)
        return path

    def load_payload(self, path: Path) -> dict:
        """Read one .pt container; purpose-specific readers validate required fields."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("_format") != TRAIN_CHECKPOINT_FORMAT:
            raise RuntimeError("Unsupported checkpoint format")
        if not isinstance(payload.get("weights"), dict) or not isinstance(payload.get("state"), dict):
            raise RuntimeError("Checkpoint requires state and weights mappings")
        return payload

    def load_inference(self, path: Path, *, use_ema: bool):
        payload = self.load_payload(path)
        state = payload["weights"].get("ema_model" if use_ema else "model")
        if state is None and use_ema:
            raise ValueError("Requested EMA weights are absent; select raw explicitly")
        if not isinstance(state, dict) or not state or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise RuntimeError("Inference requires a nonempty tensor state_dict")
        step = payload["state"].get("global_step")
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint global_step must be an int >= 0")
        return state, step

    def load(self, path: Path) -> TrainCheckpoint:
        payload = self.load_payload(path)
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
        if not expected_state <= state.keys() or not expected_weights <= weights.keys():
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
        for key in ("model", "optimizer", "scheduler"):
            if not isinstance(weights[key], dict):
                raise ValueError(f"Resume requires {key} state mapping")
        if weights["ema_model"] is not None and not isinstance(weights["ema_model"], dict):
            raise ValueError("ema_model must be a state mapping or None")
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
