"""Positional encoding modules shared across backbones and point-cloud encoders."""

import math
import torch
import torch.nn as nn

POS_ENCODING_BASE = 10000.0
"""Standard base frequency for sinusoidal positional encoding."""


# ---------------------------------------------------------------------------
# 1D sinusoidal positional encoding (timestep / sequence position)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(POS_ENCODING_BASE) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# 3D sinusoidal / relative positional encodings (point-cloud coordinates)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb3D(nn.Module):
    """Standard sinusoidal positional encoding for continuous 3D coordinates."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim must be divisible by 6, but got {dim}")
        self.dim = dim

        half_axis_dim = dim // 6
        exponent = math.log(POS_ENCODING_BASE) / max(half_axis_dim - 1, 1)
        inv_freq = torch.exp(-exponent * torch.arange(half_axis_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim < 2 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must have shape [..., 3], but got {tuple(xyz.shape)}")

        angles = xyz.unsqueeze(-1) * self.inv_freq
        pos_emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return pos_emb.flatten(start_dim=-2)


class RelativePositionalEncoding3D(nn.Module):
    """Learned relative positional encoding for local point neighborhoods."""

    def __init__(self, out_dim: int, hidden_dim: int | None = None, use_distance: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_distance = use_distance

        in_dim = 4 if use_distance else 3
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, relative_xyz: torch.Tensor) -> torch.Tensor:
        if relative_xyz.ndim < 2 or relative_xyz.size(-1) != 3:
            raise ValueError(
                f"relative_xyz must have shape [..., 3], but got {tuple(relative_xyz.shape)}"
            )

        if self.use_distance:
            distance = torch.linalg.norm(relative_xyz, dim=-1, keepdim=True)
            relative_input = torch.cat((relative_xyz, distance), dim=-1)
        else:
            relative_input = relative_xyz

        return self.mlp(relative_input)
