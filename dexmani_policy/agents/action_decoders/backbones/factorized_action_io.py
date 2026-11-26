"""Arm/hand temporal tokens for DeCAL-style factorized flow matching."""

import torch
from torch import nn

from .ditx import DiTXOutput
from .modules import WEIGHT_INIT_STD, initialize_linear


class FactorizedActionIO(nn.Module):
    def __init__(self, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        self.arm_embedder = nn.Linear(7, hidden_dim)
        self.hand_embedder = nn.Linear(12, hidden_dim)
        self.time_pos_embed = nn.Parameter(torch.empty(1, horizon, hidden_dim))
        self.arm_head = DiTXOutput(hidden_dim, 7)
        self.hand_head = DiTXOutput(hidden_dim, 12)
        self.apply(initialize_linear)
        nn.init.normal_(self.arm_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.hand_embedder.weight, std=WEIGHT_INIT_STD)
        nn.init.normal_(self.time_pos_embed, std=WEIGHT_INIT_STD)
        nn.init.zeros_(self.arm_head.ffn_final.fc2.weight)
        nn.init.zeros_(self.hand_head.ffn_final.fc2.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        arm, hand = x.split((7, 12), dim=-1)
        arm, hand = self.arm_embedder(arm), self.hand_embedder(hand)
        position = self.time_pos_embed[:, :x.shape[1]].to(arm.dtype)
        return torch.cat((arm + position, hand + position), dim=1)

    def decode_velocity(self, tokens: torch.Tensor) -> torch.Tensor:
        arm, hand = tokens.chunk(2, dim=1)
        return torch.cat((self.arm_head(arm), self.hand_head(hand)), dim=-1)
