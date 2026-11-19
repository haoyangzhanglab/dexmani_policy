import random
from collections.abc import Callable
from typing import Any

import numpy as np


def rebase_eef_rotvec(actions):
    """Change the rotation reference by Rx(pi), its own inverse in SO(3)."""
    import torch

    arms = actions.unflatten(-1, (actions.shape[-1] // 22, 22))
    rotvec = arms[..., 3:6].float()
    half_angle = 0.5 * rotvec.norm(dim=-1, keepdim=True)
    x, y, z = (0.5 * torch.sinc(half_angle / torch.pi) * rotvec).split(1, dim=-1)
    # Left multiplication by the reference sends (w, x, y, z) to (-x, w, -z, y).
    vector = torch.cat((half_angle.cos(), -z, y), dim=-1)
    half_angle = torch.atan2(vector.norm(dim=-1, keepdim=True), x.abs())
    gain = 2 / torch.sinc(half_angle / torch.pi)
    rotvec = torch.where(x > 0, -gain, gain) * vector
    return torch.cat((arms[..., :3], rotvec, arms[..., 6:]), dim=-1).flatten(-2)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state(device="cpu") -> dict[str, Any]:
    """Capture process RNG, excluding DataLoader worker state."""
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
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def dict_apply(x: dict, func: Callable[[Any], Any]) -> dict:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
