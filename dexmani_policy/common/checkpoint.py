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
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
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


def read_checkpoint(path: Path, *, mmap=False) -> dict:
    """Read a trusted project .pt container; this is not an untrusted upload format."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)


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


def policy_state_dict(model) -> dict:
    """Save an unwrapped policy with eager parameter names, including compiled backbones."""
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


def load_policy_state(model, state):
    """Restore saved parameter precision before strict copying, including FP32 EMA."""
    from dexmani_policy.common.normalization import validate_normalizer_state

    for name, parameter in model.named_parameters():
        saved = state[name]
        if (
            parameter.dtype != saved.dtype
            and parameter.is_floating_point()
            and saved.is_floating_point()
        ):
            parameter.data = parameter.data.to(dtype=saved.dtype)
    model.load_state_dict(state, strict=True)
    validate_normalizer_state(model.normalizer, model.normalization_spec, model.normalization_dims)


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    """Restore inference from config and tensors, independent of resume semantics."""
    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.common.config import (
        normalization_dims_from_config,
        resolve_normalization_spec,
    )
    from dexmani_policy.common.runtime import get_rng_state, set_rng_state

    payload = read_checkpoint(checkpoint_path, mmap=True)
    if inference_recipe(payload["effective_config"]) != inference_recipe(saved_config):
        raise ValueError("Checkpoint inference recipe mismatch with saved config")
    cfg = OmegaConf.create(payload["effective_config"])
    state = payload["weights"]["ema_model" if use_ema else "model"]
    if state is None and use_ema:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    global_step = payload["state"]["global_step"]
    if type(global_step) is not int or global_step < 0:
        raise ValueError("Checkpoint global_step must be an int >= 0")
    # Release unselected weights and optimizer state before model construction.
    del payload

    if cfg.action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    agent_cfg = prepare_restore_config(cfg.agent)
    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(agent_cfg)
    finally:
        set_rng_state(rng_state)

    if agent.inference_metadata != OmegaConf.to_container(cfg.policy_metadata, resolve=True):
        raise ValueError("Restored policy metadata differs from saved construction")
    agent.checkpoint_step = global_step
    agent.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    load_policy_state(agent, state)
    del state
    agent.to(device)
    agent.eval()
    return agent


def inference_recipe(config):
    """Compare inference semantics without training, data-location or initialization settings."""
    from omegaconf import OmegaConf

    from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    agent = dict(config["agent"])
    for key in ("prior_path", "codec_path", "codebook_path"):
        agent.pop(key, None)

    dataset = config["dataset"]
    inputs = {
        key: dataset.get(key) for key in ("sensor_modalities", "pc_dim", "history_steps")
    }
    inputs["rgb"] = rgb_preprocessing_kwargs(dataset)

    return {
        "agent": agent,
        "policy_metadata": config["policy_metadata"],
        "inputs": inputs,
        "normalization": config["normalization"],
        "normalization_dims": config["normalization_dims"],
        "action_key": config["action_key"],
        "task_name": config.get("task_name"),
        "real_runtime": config.get("real_runtime"),
    }


def prepare_restore_config(agent_cfg):
    """Copy only policy constructor settings; keep the saved recipe unchanged."""
    import copy
    import hydra
    from omegaconf import open_dict

    agent_cfg = copy.deepcopy(agent_cfg)
    with open_dict(agent_cfg):
        hydra.utils.get_class(agent_cfg._target_).prepare_restore_config(agent_cfg)
    return agent_cfg
