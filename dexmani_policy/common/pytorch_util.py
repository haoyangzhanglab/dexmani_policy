import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, Callable, List, Optional


def dict_apply(
    x: Dict[str, torch.Tensor], 
    func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
        
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            result[key] = [func(item) if hasattr(item, 'to') else item for item in value]
        else:
            result[key] = func(value)
    return result


def optimizer_to(optimizer, device):
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
    # 检查 checkpoint 是否来自 DDP
    first_key = next(iter(state_dict.keys()))
    is_checkpoint_ddp = first_key.startswith('module.')

    # checkpoint 是 DDP，当前不是 DDP：移除 'module.' 前缀
    if is_checkpoint_ddp and not is_current_ddp:
        return {k.removeprefix("module."): v for k, v in state_dict.items()}

    # checkpoint 不是 DDP，当前是 DDP：添加 'module.' 前缀
    elif not is_checkpoint_ddp and is_current_ddp:
        return {f'module.{k}': v for k, v in state_dict.items()}

    return state_dict


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


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