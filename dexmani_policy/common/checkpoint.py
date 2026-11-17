from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexmani_policy.common.io_util import atomic_path


@dataclass
class Checkpoint:
    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    effective_config: dict[str, Any]
    ema_model_state: dict[str, Any] | None = None


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(vars(checkpoint), temporary)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    import torch

    # Project checkpoints contain trusted training state, not just tensors.
    return Checkpoint(**torch.load(path, map_location="cpu", weights_only=False, mmap=True))


def policy_state_dict(model) -> dict:
    # Compiled backbones add _orig_mod to parameter names.
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}
