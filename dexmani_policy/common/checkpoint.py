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
