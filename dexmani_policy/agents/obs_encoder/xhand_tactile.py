"""Per-frame identity tactile encoders; XHand Real inputs use SDK / 255."""

import torch
from torch import nn

__all__ = ["XHandForceEncoder", "XHandTaxelEncoder"]


class XHandForceEncoder(nn.Module):
    """Identity force [BT,5,3] (XHand Real: SDK / 255) -> [BT,5,64] with one shared 3-layer MLP."""

    num_tokens = 5
    out_dim = 64
    input_key = "finger_tactile_force"

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=self.net[0].weight.dtype))


class XHandTaxelEncoder(nn.Module):
    """Identity taxel force [BT,5,120,3] (XHand Real: SDK / 255) -> [BT,5,64] with KineDex 1D CNN."""

    num_tokens = 5
    out_dim = 64
    input_key = "taxel_tactile_force"

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, self.out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.net[0].weight.dtype)
        fingers = x.flatten(0, 1).transpose(1, 2)
        return self.net(fingers).squeeze(-1).reshape(x.shape[0], self.num_tokens, self.out_dim)
