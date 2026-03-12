import torch
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