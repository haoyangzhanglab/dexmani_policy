"""Proprioceptive state encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        out_dim: int = 64,
        hidden_dims: Sequence[int] = (64,),
    ):
        super().__init__()
        self.out_dim = out_dim
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
