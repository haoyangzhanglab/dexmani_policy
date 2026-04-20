import torch
import torch.nn as nn
from typing import List, Optional, Type


def create_mlp(
    in_channels: int,
    hidden_channels: List[int],
    out_channels: int,
    activation: Type[nn.Module] = nn.ReLU,
):
    layers = []
    prev = in_channels

    for h in hidden_channels:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h

    layers.append(nn.Linear(prev, out_channels))
    return nn.Sequential(*layers)


class StateMLP(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: Optional[List[int]] = None,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64]

        self._out_dim = output_channels
        self.mlp = create_mlp(
            in_channels=input_channels,
            hidden_channels=hidden_channels,
            out_channels=output_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @property
    def out_dim(self) -> int:
        return self._out_dim