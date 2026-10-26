"""Atomic persistence, random state and tensor containers."""

from __future__ import annotations

import os
import random
import sys
import tempfile
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


@contextmanager
def atomic_path(path, *, overwrite=True, suffix=".tmp"):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".dexmani-publish-", suffix=suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_symlink():
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A hard link publishes atomically without replacing an existing target.
            os.link(temporary, path)
    finally:
        close_resources(lambda: temporary.unlink(missing_ok=True))


def close_resources(*callbacks):
    """Close owned resources; report secondary errors and propagate the first failure."""
    primary = sys.exc_info()[1]
    failure = None
    for close in callbacks:
        try:
            close()
        except BaseException as error:
            if primary is None and failure is None:
                failure = error
            else:
                traceback.print_exc()
    if failure is not None:
        raise failure


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
