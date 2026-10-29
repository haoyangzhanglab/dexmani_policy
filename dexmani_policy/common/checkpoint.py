from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dexmani_policy.common.io_util import atomic_path

if TYPE_CHECKING:
    from dexmani_policy.agents.policies.base import BasePolicy


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


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(vars(checkpoint), temporary)
    return path


def load_checkpoint(path: str | Path, *, mmap: bool = False) -> Checkpoint:
    import torch

    # Project checkpoints contain trusted training state, not just tensors.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    checkpoint = Checkpoint(**payload)
    raw = checkpoint.model_state
    ema = checkpoint.ema_model_state
    affine_keys = {key for key in raw if key.startswith("normalizer.")}
    if ema is not None and affine_keys != {key for key in ema if key.startswith("normalizer.")}:
        raise ValueError("Raw/EMA normalizer fields differ")
    for key in affine_keys:
        if raw[key].dtype != torch.float32:
            raise ValueError(f"Normalizer buffer must be FP32: {key}")
        if ema is not None and (ema[key].dtype != torch.float32 or not torch.equal(raw[key], ema[key])):
            raise ValueError(f"Raw/EMA normalizer mismatch: {key}")
    return checkpoint


def policy_state_dict(model) -> dict:
    # Compiled backbones add _orig_mod to parameter names.
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


def load_policy_state(model, state):
    from dexmani_policy.common.normalizer import validate_normalizer_state

    # Preserve checkpoint precision, including FP32 EMA parameters.
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
    import hydra
    from omegaconf import OmegaConf

    if OmegaConf.is_config(agent_cfg):
        agent_cfg = OmegaConf.to_container(agent_cfg, resolve=True)
    agent_cfg = OmegaConf.create(agent_cfg)
    agent_class: type[BasePolicy] = hydra.utils.get_class(agent_cfg._target_)
    agent_class.prepare_restore_config(agent_cfg)
    return agent_cfg


def restore_policy(saved_config, checkpoint_path, *, use_ema, device):
    import hydra

    from dexmani_policy.common.config import validate_observation_fields
    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    cfg = checkpoint.effective_config
    recipe = _inference_config(cfg)
    if recipe != _inference_config(saved_config):
        raise ValueError("Checkpoint inference recipe mismatch with saved config")
    state = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    if state is None:
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
    if hasattr(agent, "tactile_source") and agent.tactile_source != cfg.get("tactile_source"):
        raise ValueError("Restored AdapTac tactile_source differs from saved top-level source")
    validate_observation_fields(agent, cfg["dataset"]["sensor_modalities"])
    agent.checkpoint_step = global_step
    agent.set_normalization_spec(cfg["normalization"], cfg["normalization_dims"])
    load_policy_state(agent, state)
    del state
    agent.to(device)
    agent.eval()
    return agent


def _inference_config(config):
    from dexmani_policy.common.image_util import rgb_preprocessing_kwargs

    dataset = config["dataset"]
    inputs = {
        key: dataset.get(key) for key in ("sensor_modalities", "pc_dim", "history_steps")
    }
    inputs["rgb"] = rgb_preprocessing_kwargs(dataset)

    return {
        "agent": prepare_restore_config(config["agent"]),
        "policy_metadata": config["policy_metadata"],
        "inputs": inputs,
        "normalization": config["normalization"],
        "normalization_dims": config["normalization_dims"],
        "action_key": config["action_key"],
        "task_name": config.get("task_name"),
        "real_runtime": config.get("real_runtime"),
        "tactile_source": config.get("tactile_source"),
    }
