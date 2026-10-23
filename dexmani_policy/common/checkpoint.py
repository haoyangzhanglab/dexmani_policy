"""Atomic checkpoint persistence and policy restoration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_policy.common.runtime import atomic_path


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
    rng_states: list[dict[str, Any]]
    effective_config: dict[str, Any]


def save_training_checkpoint(path: Path, checkpoint: TrainCheckpoint) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "effective_config": checkpoint.effective_config,
        "state": {
            "epoch": int(checkpoint.epoch),
            "global_step": int(checkpoint.global_step),
            "next_micro_step": checkpoint.next_micro_step,
            "resume_contract": checkpoint.resume_contract,
            "ema_updater_step": checkpoint.ema_updater_step,
            "rng_states": checkpoint.rng_states,
        },
        "weights": {
            "model": checkpoint.model_state,
            "ema_model": checkpoint.ema_model_state,
            "optimizer": checkpoint.optimizer_state,
            "scheduler": checkpoint.scheduler_state,
        },
    }
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)
    return path


def read_checkpoint(path: Path) -> dict:
    """Read one .pt container; purpose-specific readers validate required fields."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_training_checkpoint(path: Path) -> TrainCheckpoint:
    payload = read_checkpoint(path)
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
        rng_states=state["rng_states"],
        model_state=weights["model"],
        ema_model_state=weights["ema_model"],
        optimizer_state=weights["optimizer"],
        scheduler_state=weights["scheduler"],
        effective_config=payload["effective_config"],
    )


def inference_weights(payload, *, use_ema):
    """Select raw or EMA tensors from a loaded checkpoint."""
    state = payload["weights"]["ema_model" if use_ema else "model"]
    if state is None and use_ema:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    step = payload["state"]["global_step"]
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint global_step must be an int >= 0")
    return state, step


def unwrap_state_dict(state_dict: dict) -> dict:
    """Remove DDP/compile parameter prefixes before saving model weights."""
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if next(iter(state_dict), "").startswith("module."):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    return state_dict


def load_policy_state(model, state):
    """Restore saved parameter precision before strict copying, including FP32 EMA."""
    for name, parameter in model.named_parameters():
        saved = state.get(name)
        if (
            saved is not None
            and parameter.is_floating_point()
            and saved.is_floating_point()
            and parameter.dtype != saved.dtype
        ):
            parameter.data = parameter.data.to(dtype=saved.dtype)
    model.load_state_dict(state, strict=True)


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    """Restore inference from config and tensors, independent of resume semantics."""
    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.common.config import (
        normalization_dims_from_config,
        resolve_normalization_spec,
        validate_action_window,
    )

    payload = read_checkpoint(checkpoint_path)
    if saved_config is not None:
        if inference_config(payload["effective_config"]) != inference_config(saved_config):
            raise ValueError("Checkpoint inference recipe mismatch with saved config")
    saved_config = payload["effective_config"]
    cfg = OmegaConf.create(saved_config)
    validate_action_window(cfg.agent.horizon, cfg.agent.n_obs_steps, cfg.agent.n_action_steps)
    if cfg.action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    state, global_step = inference_weights(payload, use_ema=use_ema)
    cfg = checkpoint_agent_config(cfg, state)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.checkpoint_step = global_step
    agent.action_key = cfg.action_key
    agent.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    load_policy_state(agent, state)
    agent.validate_normalizer()
    agent.inference_config = saved_config
    agent.to(device)
    agent.eval()
    return agent


def inference_config(config):
    """Compare inference semantics without training, data-location or initialization settings."""
    from omegaconf import OmegaConf

    from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs

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
            agent.get("_target_")
            == "dexmani_policy.agents.policies.multitask_dit.MultiTaskDiTPolicy"
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
