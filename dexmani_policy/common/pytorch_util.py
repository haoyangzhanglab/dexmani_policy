import random
from collections.abc import Callable
from typing import Any

import numpy as np


def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_rng_state(device="cpu") -> dict[str, Any]:
    """Capture process RNG; worker augmentation and prefetch state are not included."""
    import torch

    device = torch.device(device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def set_rng_state(state: dict[str, Any], device="cpu") -> None:
    import torch

    device = torch.device(device)
    cuda_state = state["torch_cuda"]
    if (cuda_state is not None) != (device.type == "cuda"):
        raise ValueError("Checkpoint CUDA RNG state must match the target device type")
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def dict_apply(x: dict, func: Callable) -> dict:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
