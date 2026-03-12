import torch.nn as nn
from typing import List, Optional, Tuple


def get_default_optim_group(
    module: nn.Module, 
    weight_decay: float
):
    module_params ={
            'params': [p for p in module.parameters() if p.requires_grad],
            'weight_decay': weight_decay
    }
    return [module_params]


def get_optim_group_with_no_decay(
    module: nn.Module,
    weight_decay: float,
    no_decay_names: Optional[List[str]] = None,
    decay_names: Optional[List[str]] = None,
    extra_whitelist: Optional[Tuple[nn.Module]] = None,
    extra_blacklist: Optional[Tuple[nn.Module]] = None,
):
    whitelist = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.MultiheadAttention,
    )

    blacklist = (
        nn.LayerNorm,
        nn.Embedding,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    if extra_whitelist is not None:
        whitelist = whitelist + tuple(extra_whitelist)
    if extra_blacklist is not None:
        blacklist = blacklist + tuple(extra_blacklist)

    no_decay_names = set(no_decay_names or [])
    decay_names = set(decay_names or [])

    overlap = no_decay_names & decay_names
    if overlap:
        raise ValueError(f"Parameters appear in both decay_names and no_decay_names: {sorted(overlap)}")

    param_dict = {name: param for name, param in module.named_parameters() if param.requires_grad}

    unknown_no_decay = no_decay_names - set(param_dict.keys())
    if unknown_no_decay:
        raise ValueError(f"Unknown parameter names in no_decay_names: {sorted(unknown_no_decay)}")

    unknown_decay = decay_names - set(param_dict.keys())
    if unknown_decay:
        raise ValueError(f"Unknown parameter names in decay_names: {sorted(unknown_decay)}")

    decay = set(decay_names)
    no_decay = set(no_decay_names)

    for mn, m in module.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn

            if fpn in decay_names or fpn in no_decay_names:
                continue

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist):
                decay.add(fpn)

    inter = decay & no_decay
    if inter:
        raise ValueError(f"Parameters in both decay and no_decay: {sorted(inter)}")

    union = decay | no_decay
    missing = set(param_dict.keys()) - union
    if missing:
        missing_info = {k: tuple(param_dict[k].shape) for k in sorted(missing)}
        raise ValueError(
            "Some trainable parameters were not assigned to decay/no_decay. "
            f"Please add them manually: {missing_info}"
        )

    optim_group = [
        {
            "params": [param_dict[name] for name in sorted(decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[name] for name in sorted(no_decay)],
            "weight_decay": 0.0,
        },
    ]

    all_params = [p for g in optim_group for p in g["params"]]
    assert len(all_params) == len(set(map(id, all_params))), "Some parameters appear in more than one group"

    return optim_group