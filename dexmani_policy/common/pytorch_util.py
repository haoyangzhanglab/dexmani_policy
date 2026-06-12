import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, Callable, List, Optional, Union


def ensure_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert numpy array to tensor; pass through torch.Tensor unchanged."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def dict_apply(
    x: Dict[str, torch.Tensor],
    func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
        
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            result[key] = [func(item) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            result[key] = func(value)
    return result


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device | str) -> torch.optim.Optimizer:
    """Move all tensor state in an optimizer to the given device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def fix_state_dict(state_dict: Dict, is_current_ddp: bool) -> Dict:
    first_key = next(iter(state_dict.keys()))

    # Strip _orig_mod. prefix (from torch.compile / OptimizedModule).
    # This can coexist with module. (DDP), producing _orig_mod.module.xxx.
    if first_key.startswith('_orig_mod.'):
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        first_key = next(iter(state_dict.keys()))

    is_checkpoint_ddp = first_key.startswith('module.')

    if is_checkpoint_ddp and not is_current_ddp:
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    elif not is_checkpoint_ddp and is_current_ddp:
        return {f'module.{k}': v for k, v in state_dict.items()}

    return state_dict


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def create_mlp(
    in_channels: int,
    hidden_channels: List[int],
    out_channels: Optional[int] = None,
    activation: type = nn.ReLU,
    use_norm: bool = False,
):
    layers = []
    prev = in_channels
    for h in hidden_channels:
        layers.extend([nn.Linear(prev, h), activation(inplace=True)])
        if use_norm:
            layers.append(nn.LayerNorm(h))
        prev = h
    if out_channels is not None:
        layers.append(nn.Linear(prev, out_channels))
    return nn.Sequential(*layers)