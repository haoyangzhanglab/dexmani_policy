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
    return Checkpoint(**payload)


def policy_state_dict(model) -> dict:
    # Compiled backbones add _orig_mod to parameter names.
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


def restore_policy(checkpoint_path, *, use_ema, device):
    import hydra

    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    cfg = checkpoint.effective_config
    state = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    if state is None:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    global_step = checkpoint.global_step
    # Release unselected weights and optimizer state before model construction.
    del checkpoint

    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(cfg["agent"])
    finally:
        set_rng_state(rng_state)

    agent.checkpoint_step = global_step
    agent.configure_normalizer(cfg["normalization_dims"], cfg["normalizer_shapes"])
    agent.load_state_dict(state, strict=True, assign=True)
    del state
    agent.to(device)
    agent.eval()
    return agent
