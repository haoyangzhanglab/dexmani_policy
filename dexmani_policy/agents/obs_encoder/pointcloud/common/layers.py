import torch
import torch.nn as nn


class PointMLP(nn.Module):
    """Point cloud MLP: Linear + LayerNorm + GELU.

    Used across pointnext, tokenizer, and patch_tokenizer modules.
    """

    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)
