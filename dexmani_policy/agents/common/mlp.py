import torch.nn as nn
from typing import List


def create_mlp(
    in_channels: int,
    layer_channels: List[int],
    activation: type[nn.Module] = nn.ReLU,
):
    layers = []
    prev = in_channels
    for h in layer_channels:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        prev = h
    return nn.Sequential(*layers)
