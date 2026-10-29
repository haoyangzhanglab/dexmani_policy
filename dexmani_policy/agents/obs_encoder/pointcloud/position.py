"""Shared continuous metric position encodings for 3D attention."""

import math

import torch
from torch import nn


class RotaryPositionEmbedding3D(nn.Module):
    """Continuous metric XYZ RoPE; relative-position encoding, not SE(3) equivariance."""

    def __init__(self, head_dim: int, scale: float = 0.05, base: float = 100.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 6:
            raise ValueError("head_dim must be positive and divisible by 6")
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(base) or base <= 1:
            raise ValueError("RoPE requires finite scale > 0 and base > 1")
        axis_dim = head_dim // 3
        self.scale = scale
        self.register_buffer(
            "inv_freq",
            base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim),
            persistent=False,
        )

    def forward(self, xyz: torch.Tensor):
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            # A shared origin improves phase precision without changing pairwise offsets.
            positions = (xyz.float() - xyz.float().mean(dim=1, keepdim=True)) / self.scale
            angles = positions.unsqueeze(-1) * self.inv_freq.float()
            angles = angles.repeat_interleave(2, dim=-1).flatten(-2)
            return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        pairs = x.float().unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x.float() * cos + rotated * sin).to(x.dtype)
