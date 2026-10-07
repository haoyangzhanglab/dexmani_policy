"""Process and DataLoader worker random state helpers."""

import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_rng_state(device="cpu") -> Dict[str, Any]:
    """Capture main-process Python/NumPy/Torch/CUDA RNG for checkpoint resume.

    Together with the saved sampler epoch/cursor, this restores process RNG
    and sample ordering. DataLoader worker augmentation RNG and prefetch state
    are not captured, so worker augmentation is not guaranteed to be bit-exact
    across resume, including with persistent workers.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(torch.device(device))
            if torch.device(device).type == "cuda" else None
        ),
    }


def set_rng_state(state: Dict[str, Any], device="cpu") -> None:
    """Restore this rank's stream to its actual current training device."""
    device = torch.device(device)
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("Checkpoint CUDA RNG must be a per-rank tensor; restore older formats with their original code")
        if device.type != "cuda":
            raise ValueError("CUDA training RNG requires a CUDA target device for exact resume")
        if cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1 or not cuda_state.numel():
            raise ValueError("Invalid CUDA RNG tensor")
    elif device.type == "cuda":
        raise ValueError("Checkpoint has no CUDA RNG for exact CUDA resume")
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
