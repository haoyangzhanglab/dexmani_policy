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


def _cuda_state_for_rank(cuda_state, *, source_config, rank, world_size):
    if cuda_state is None or isinstance(cuda_state, torch.Tensor):
        return cuda_state
    if not isinstance(cuda_state, list) or not cuda_state or any(not isinstance(x, torch.Tensor) for x in cuda_state):
        raise ValueError("Invalid checkpoint CUDA RNG state")
    training = (source_config or {}).get("training", {})
    slot = None
    if world_size > 1:
        ids = training.get("gpu_ids")
        if training.get("num_gpus") == world_size:
            if ids is None:
                slot = rank
            elif (len(ids) == world_size and all(type(i) is int and i >= 0 for i in ids)
                  and len(set(ids)) == world_size):
                slot = ids[rank]
    elif "device" in training:
        original = torch.device(training["device"])
        if original.type == "cuda":
            slot = original.index if original.index is not None else 0
    if slot is None and len(cuda_state) == 1 and not any(k in training for k in ("device", "gpu_ids", "num_gpus")):
        slot = 0
    if slot is None or not 0 <= slot < len(cuda_state):
        raise ValueError("Legacy CUDA RNG list has no unambiguous source device in saved config; exact resume refused")
    return cuda_state[slot]


def set_rng_state(state: Dict[str, Any], device="cpu", *, source_config=None, rank=0, world_size=1) -> None:
    """Restore this rank's stream to its actual current training device."""
    device = torch.device(device)
    cuda_state = _cuda_state_for_rank(state.get("torch_cuda"), source_config=source_config,
                                     rank=rank, world_size=world_size)
    if cuda_state is not None:
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
