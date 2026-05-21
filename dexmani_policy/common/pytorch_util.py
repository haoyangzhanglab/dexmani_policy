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
            result[key] = [func(item) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            result[key] = func(value)
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    target_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in target_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify all targets were replaced
    remaining = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(remaining) == 0
    return root_module


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def fix_state_dict(state_dict: Dict, is_current_ddp: bool) -> Dict:
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