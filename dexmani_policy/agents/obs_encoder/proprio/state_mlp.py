import torch
import torch.nn as nn
from typing import List, Optional

from dexmani_policy.common.pytorch_util import create_mlp


class StateMLP(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: Optional[List[int]] = None,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64]

        self.in_dim = input_channels
        self.out_dim = output_channels
        self.mlp = create_mlp(
            in_channels=input_channels,
            hidden_channels=hidden_channels,
            out_channels=output_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)