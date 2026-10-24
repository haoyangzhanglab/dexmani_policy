"""Atomic checkpoint persistence and policy restoration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_policy.common.runtime import atomic_path


@dataclass
class TrainingCheckpoint:
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


def save_training_checkpoint(path: Path, checkpoint: TrainingCheckpoint) -> Path:
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


def load_training_checkpoint(path: Path) -> TrainingCheckpoint:
    payload = read_checkpoint(path)
    state = payload["state"]
    weights = payload["weights"]
    for key in ("epoch", "global_step", "next_micro_step"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f"Checkpoint {key} must be an int >= 0")
    return TrainingCheckpoint(
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


def select_inference_weights(payload, *, use_ema):
    state = payload["weights"]["ema_model" if use_ema else "model"]
    if state is None and use_ema:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    step = payload["state"]["global_step"]
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint global_step must be an int >= 0")
    return state, step


def policy_state_dict(model) -> dict:
    """Save an unwrapped policy with eager parameter names, including compiled backbones."""
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


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
        resolve_action_start,
        resolve_normalization_spec,
        validate_action_window,
    )

    payload = read_checkpoint(checkpoint_path)
    if inference_recipe(payload["effective_config"]) != inference_recipe(saved_config):
        raise ValueError("Checkpoint inference recipe mismatch with saved config")
    saved_config = payload["effective_config"]
    cfg = OmegaConf.create(saved_config)
    validate_action_window(
        cfg.agent.horizon, cfg.agent.n_obs_steps, cfg.agent.n_action_steps,
        action_start_index=resolve_action_start(cfg.agent),
    )
    if cfg.action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    state, global_step = select_inference_weights(payload, use_ema=use_ema)
    cfg = prepare_restore_config(cfg)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.checkpoint_step = global_step
    agent.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    load_policy_state(agent, state)
    agent.validate_normalizer()
    agent.to(device)
    agent.eval()
    return agent


def inference_recipe(config):
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
        "normalization": config["normalization"],
        "normalization_dims": config["normalization_dims"],
        "action_key": config["action_key"],
        "task_name": config.get("task_name"),
        "real_runtime": config.get("real_runtime"),
    }


def prepare_restore_config(cfg):
    """Construct from saved weights without loading external initialization assets."""
    import copy
    from omegaconf import DictConfig, open_dict

    from dexmani_policy.common.config import positive_int

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
            positive_int(agent.text_embed_dim, "saved text_embed_dim")
    return cfg
