from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_policy.common.io_util import atomic_path


@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    model_state: dict[str, Any]
    ema_model_state: dict[str, Any] | None
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    ema_updater_step: int | None
    rng_state: dict[str, Any]
    effective_config: dict[str, Any]


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(vars(checkpoint), temporary)
    return path


def load_checkpoint(path: Path, *, mmap=False) -> Checkpoint:
    # Project checkpoints are trusted pickle containers.
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    return Checkpoint(**payload)


def policy_state_dict(model) -> dict:
    """Save an unwrapped policy with eager parameter names, including compiled backbones."""
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


def load_policy_state(model, state):
    """Restore saved parameter precision before strict copying, including FP32 EMA."""
    from dexmani_policy.common.normalizer import validate_normalizer_state

    for name, parameter in model.named_parameters():
        saved = state[name]
        if (
            parameter.dtype != saved.dtype
            and parameter.is_floating_point()
            and saved.is_floating_point()
        ):
            parameter.data = parameter.data.to(dtype=saved.dtype)
    model.load_state_dict(state, strict=True)
    validate_normalizer_state(model.normalizer)


def prepare_restore_config(agent_cfg):
    """Copy only policy constructor settings; keep the saved recipe unchanged."""
    import copy
    import hydra
    from omegaconf import open_dict

    from dexmani_policy.agents.policies.base import BasePolicy

    agent_cfg = copy.deepcopy(agent_cfg)
    agent_class: type[BasePolicy] = hydra.utils.get_class(agent_cfg._target_)
    with open_dict(agent_cfg):
        agent_class.prepare_restore_config(agent_cfg)
    return agent_cfg


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    import hydra

    from dexmani_policy.common.config import (
        normalization_dims_from_config,
        resolve_normalization_spec,
    )
    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    cfg = checkpoint.effective_config
    recipe = inference_recipe(cfg)
    if recipe != inference_recipe(saved_config):
        raise ValueError("Checkpoint inference recipe mismatch with saved config")
    state = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    if state is None and use_ema:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    global_step = checkpoint.global_step
    # Release unselected weights and optimizer state before model construction.
    del checkpoint

    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(recipe["agent"])
    finally:
        set_rng_state(rng_state)

    if agent.inference_metadata != cfg["policy_metadata"]:
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

    from dexmani_policy.common.image_util import rgb_preprocessing_kwargs

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    agent = OmegaConf.to_container(
        prepare_restore_config(OmegaConf.create(config["agent"])), resolve=True
    )

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
