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


def get_rng_state() -> Dict[str, Any]:
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
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore the main-process RNG state captured by :func:`get_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
