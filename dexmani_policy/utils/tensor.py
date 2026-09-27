"""Generic tensor conversion and container mapping."""

from typing import Callable, Dict, Union

import numpy as np
import torch


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:

    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            result[key] = [
                func(item) if isinstance(item, torch.Tensor) else item for item in value
            ]
        else:
            result[key] = func(value)
    return result


def ensure_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert numpy array to tensor; pass through torch.Tensor unchanged."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)
