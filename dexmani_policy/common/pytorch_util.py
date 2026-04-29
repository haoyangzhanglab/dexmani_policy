import torch
import random
import numpy as np
import collections
import torch.nn as nn
from typing import Dict, Callable, List


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


def fix_state_dict(state_dict: Dict, is_current_ddp: bool) -> Dict:
    """
    自动处理 DDP state_dict 的 'module.' 前缀。

    Args:
        state_dict: 要处理的 state_dict
        is_current_ddp: 当前模型是否是 DDP 包装的

    Returns:
        处理后的 state_dict
    """
    if not state_dict:
        return state_dict

    # 检查 checkpoint 是否来自 DDP
    first_key = next(iter(state_dict.keys()))
    is_checkpoint_ddp = first_key.startswith('module.')

    # checkpoint 是 DDP，当前不是 DDP：移除 'module.' 前缀
    if is_checkpoint_ddp and not is_current_ddp:
        return {k[7:]: v for k, v in state_dict.items()}

    # checkpoint 不是 DDP，当前是 DDP：添加 'module.' 前缀
    elif not is_checkpoint_ddp and is_current_ddp:
        return {f'module.{k}': v for k, v in state_dict.items()}

    # 其他情况：不需要处理
    return state_dict