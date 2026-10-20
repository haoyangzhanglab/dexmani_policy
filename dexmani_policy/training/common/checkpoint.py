"""Checkpoint selection, atomic persistence and shared Policy restoration."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_policy.utils.runtime import atomic_path

TRAIN_CHECKPOINT_FORMAT = "simple.v4"


@dataclass
class TrainCheckpoint:
    epoch: int
    global_step: int
    next_micro_step: int
    model_state: dict[str, Any]
    ema_model_state: dict[str, Any] | None
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    resume_contract: dict[str, Any]
    ema_updater_step: int | None
    ema_decay: float | None
    rng_states: list[dict[str, Any]]
    effective_config: dict[str, Any]


class CheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save(self, filename: str, checkpoint: TrainCheckpoint) -> Path:
        import torch

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        payload = {
            "effective_config": checkpoint.effective_config,
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
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("_format") != TRAIN_CHECKPOINT_FORMAT:
            raise RuntimeError(f"Checkpoint requires {TRAIN_CHECKPOINT_FORMAT}")
        config = payload.get("effective_config")
        if not isinstance(config, dict) or not config:
            raise ValueError("Checkpoint requires a nonempty effective_config mapping")
        return payload

    def load(self, path: Path) -> TrainCheckpoint:
        payload = self.load_payload(path)
        state = payload["state"]
        weights = payload["weights"]
        for key in ("epoch", "global_step", "next_micro_step"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Checkpoint {key} must be an int >= 0")
        return TrainCheckpoint(
            epoch=state["epoch"],
            global_step=state["global_step"],
            next_micro_step=state["next_micro_step"],
            resume_contract=state["resume_contract"],
            ema_updater_step=state["ema_updater_step"],
            ema_decay=state["ema_decay"],
            rng_states=state["rng_states"],
            model_state=weights["model"],
            ema_model_state=weights["ema_model"],
            optimizer_state=weights["optimizer"],
            scheduler_state=weights["scheduler"],
            effective_config=payload["effective_config"],
        )


def inference_weights(payload, *, use_ema):
    """Select raw or EMA tensors from a checkpoint already read by CheckpointStore."""
    state = payload["weights"]["ema_model" if use_ema else "model"]
    if state is None and use_ema:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    step = payload["state"]["global_step"]
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint global_step must be an int >= 0")
    return state, step


def unwrap_state_dict(state_dict: dict) -> dict:
    """Remove training wrappers for loading or saving the underlying model."""
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    is_checkpoint_ddp = next(iter(state_dict), "").startswith("module.")

    if is_checkpoint_ddp:
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    return state_dict


def read_checkpoint_record(path):
    info = json.loads(path.read_text())
    if not isinstance(info, dict):
        raise ValueError(f"Checkpoint record must contain an object: {path}")
    return info


def resolve_record_path(root, relative, *, checkpoint=False, immutable=False):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Record requires a nonempty relative file path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Record path must be relative to experiment")
    if immutable and (path.name == "latest.pt" or (root / path).is_symlink()):
        raise ValueError("Selection must name an immutable milestone checkpoint")
    resolved = (root / path).resolve(strict=True)
    directory = root / "checkpoints" if checkpoint else root
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError(f"Record path must be a file inside {directory}")
    return resolved


def validate_inference_settings(info):
    inference = info.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("Record requires inference settings")
    if type(inference.get("use_ema")) is not bool:
        raise ValueError("inference.use_ema must be a bool")
    steps = inference.get("inference_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("inference.inference_steps must be a positive int (not bool)")
    if inference.get("policy_seed_mode") != "episode_seed":
        raise ValueError("Record requires inference.policy_seed_mode=episode_seed")


def resolve_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    record_path = Path(record_path).expanduser().resolve(strict=True)
    if not record_path.is_relative_to(root):
        raise ValueError("Selection record must be inside the experiment")
    info = read_checkpoint_record(record_path)
    if info["status"] != "success":
        raise ValueError("Selection has not completed successfully")
    validate_inference_settings(info)
    checkpoint = resolve_record_path(root, info["ckpt_relpath"], checkpoint=True, immutable=True)
    info["selection_result"] = str(record_path.relative_to(root))
    return info, checkpoint


def resolve_best_checkpoint(experiment_dir):
    root = Path(experiment_dir).resolve()
    pointer = read_checkpoint_record(root / "best_ckpt.json")
    record = resolve_record_path(root, pointer["selection_result"])
    return resolve_selection_record(root, record)


def resolve_checkpoint(experiment_dir, selector="best"):
    root = Path(experiment_dir).resolve()
    if selector == "best":
        return resolve_best_checkpoint(root)[1]
    elif selector == "latest":
        path = root / "checkpoints/latest.pt"
    else:
        path = Path(selector)
        if not path.is_absolute():
            path = root / "checkpoints" / path
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root / "checkpoints"):
        raise ValueError("Checkpoint must be inside the selected experiment/checkpoints")
    return path


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    """Restore inference from config and tensors, independent of resume semantics."""
    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.utils.config import (
        normalization_dims_from_config,
        resolve_normalization_spec,
        validate_action_window,
    )

    store = CheckpointStore(Path(checkpoint_path).parent)
    payload = store.load_payload(checkpoint_path)
    if saved_config is not None:
        if inference_config(payload["effective_config"]) != inference_config(saved_config):
            raise ValueError("Checkpoint inference recipe mismatch with saved config")
    saved_config = payload["effective_config"]
    cfg = OmegaConf.create(saved_config)
    validate_action_window(cfg.agent.horizon, cfg.agent.n_obs_steps, cfg.agent.n_action_steps)
    if cfg.action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    state, global_step = inference_weights(payload, use_ema=use_ema)
    state = unwrap_state_dict(state)
    cfg = checkpoint_agent_config(cfg, state)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.checkpoint_step = global_step
    agent.action_key = cfg.action_key
    agent.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    agent.load_state_dict(state, strict=True)
    agent.validate_normalizer()
    agent.inference_config = saved_config
    agent.to(device)
    agent.eval()
    return agent


def inference_config(config):
    """Compare inference semantics without training, data-location or initialization settings."""
    from omegaconf import OmegaConf

    from dexmani_policy.datasets.common.rgb_preprocessing import rgb_preprocessing_kwargs

    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    agent = dict(config["agent"])
    for key in ("prior_path", "codec_path", "codebook_path"):
        agent.pop(key, None)

    def inputs(dataset):
        values = {
            key: dataset.get(key)
            for key in ("sensor_modalities", "pc_dim", "history_steps", "task_names", "task_texts")
        }
        values["rgb"] = rgb_preprocessing_kwargs(dataset)
        values["datasets"] = [inputs(child) for child in dataset.get("datasets", ())]
        return values

    return {
        "agent": agent,
        "inputs": inputs(config["dataset"]),
        "normalization": config.get("normalization"),
        "normalization_dims": config.get("normalization_dims"),
        "action_key": config["action_key"],
        "task_name": config.get("task_name"),
        "real_runtime": config.get("real_runtime"),
        "control_dt": (config.get("real_runtime") or {}).get(
            "dt", (config.get("hand_data") or {}).get("dt")
        ),
    }


def checkpoint_agent_config(cfg, state):
    """Construct from saved weights without loading external initialization assets."""
    import copy
    import torch
    from omegaconf import DictConfig, open_dict

    cfg = copy.deepcopy(cfg)
    agent = cfg.agent
    with open_dict(agent):
        if agent.get("rgb_backbone_name") is not None:
            rgb = dict(agent.get("rgb_backbone_config") or {})
            if agent.rgb_backbone_name == "resnet" and not {"tune_mode", "norm_mode"} <= rgb.keys():
                raise ValueError("Saved ResNet requires explicit tune_mode and norm_mode")
            if agent.rgb_backbone_name in {"dino", "siglip"} and not isinstance(
                rgb.get("architecture"), (dict, DictConfig)
            ):
                raise ValueError("Saved HF backbone requires an architecture mapping")
            rgb["load_pretrained"] = False
            agent.rgb_backbone_config = rgb
        if (
            agent.get("_target_") == "dexmani_policy.agents.core.multitask_policy.MultiTaskPolicy"
            and agent.get("task_texts") is not None
        ):
            table = state.get("task_emb_table")
            texts = list(dict.fromkeys(agent.task_texts))
            dim = agent.get("text_embed_dim")
            if (
                type(dim) is not int
                or dim <= 0
                or not isinstance(table, torch.Tensor)
                or table.ndim != 2
                or tuple(table.shape) != (len(texts), dim)
            ):
                raise ValueError(
                    "Closed text checkpoint requires task_emb_table matching saved task_texts and text_embed_dim"
                )
    return cfg


MILESTONE_PATTERN = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    path: Path
    pct: int
    global_step: int

    @property
    def label(self) -> str:
        return f"{self.pct}% (step={self.global_step})"


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    """Find regular milestone files inside this experiment, sorted by pct."""
    ckpt_dir = exp_dir / "checkpoints"
    found: list[MilestoneCheckpoint] = []
    for pt_file in ckpt_dir.glob("epoch=*.pt"):
        match = MILESTONE_PATTERN.fullmatch(pt_file.name)
        if not match:
            continue
        if pt_file.is_symlink():
            raise ValueError(f"Milestone checkpoint must not be a symlink: {pt_file}")
        found.append(
            MilestoneCheckpoint(
                path=resolve_checkpoint(exp_dir, pt_file.name),
                pct=int(match.group("pct")),
                global_step=int(match.group("step")),
            )
        )

    if not found:
        raise FileNotFoundError(f"No milestone checkpoints found in {ckpt_dir}")

    return sorted(found, key=lambda checkpoint: (checkpoint.pct, checkpoint.path.name))


def resolve_checkpoint_path(
    exp_dir: Path,
    ckpt_tag_or_path: str,
) -> tuple[Path, str]:
    """Resolve a checkpoint tag to an absolute path and human-readable label."""
    if ckpt_tag_or_path.endswith("pct"):
        milestones = discover_milestone_checkpoints(exp_dir)
        target_pct = int(ckpt_tag_or_path.removesuffix("pct"))
        match = [m for m in milestones if m.pct == target_pct]
        if not match:
            available = sorted(m.pct for m in milestones)
            raise FileNotFoundError(
                f"No {target_pct}% milestone checkpoint. Available: {available}"
            )
        return match[0].path, match[0].label

    path = resolve_checkpoint(exp_dir, ckpt_tag_or_path)
    return path, f"{ckpt_tag_or_path} ({path.name})"
