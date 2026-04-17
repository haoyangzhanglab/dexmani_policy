import math
import torch
import torch.nn as nn


class SinusoidalPosEmb3D(nn.Module):
    """Standard sinusoidal positional encoding for continuous 3D coordinates."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim must be divisible by 6, but got {dim}")
        self.dim = dim

        half_axis_dim = dim // 6
        exponent = math.log(10000.0) / max(half_axis_dim - 1, 1)
        inv_freq = torch.exp(-exponent * torch.arange(half_axis_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim < 2 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must have shape [..., 3], but got {tuple(xyz.shape)}")

        angles = xyz.unsqueeze(-1) * self.inv_freq
        pos_emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return pos_emb.flatten(start_dim=-2)


class NeRFSinusoidalPosEmb(nn.Module):
    """NeRF-style positional encoding for continuous 3D coordinates."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim must be divisible by 6, but got {dim}")
        self.dim = dim

        half_axis_dim = dim // 6
        freq_bands = 2.0 ** torch.arange(half_axis_dim, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim < 2 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must have shape [..., 3], but got {tuple(xyz.shape)}")

        angles = xyz.unsqueeze(-1) * self.freq_bands
        pos_emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return pos_emb.flatten(start_dim=-2)


class RotaryPositionEncoding(nn.Module):
    """Rotary positional encoding for 1D positions."""

    def __init__(self, feature_dim: int):
        super().__init__()
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim must be even, but got {feature_dim}")
        self.feature_dim = feature_dim

        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, feature_dim, 2, dtype=torch.float32) / feature_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.shape != cos.shape or x.shape != sin.shape:
            raise ValueError(
                f"x, cos, sin must have the same shape, but got "
                f"{tuple(x.shape)}, {tuple(cos.shape)}, {tuple(sin.shape)}"
            )

        x_half_turn = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)
        return x * cos + x_half_turn * sin

    def forward(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = position.unsqueeze(-1) * self.inv_freq
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
        return cos, sin


class RotaryPositionEncoding3D(nn.Module):
    """Rotary positional encoding for 3D coordinates."""

    def __init__(self, feature_dim: int):
        super().__init__()
        if feature_dim % 6 != 0:
            raise ValueError(f"feature_dim must be divisible by 6, but got {feature_dim}")
        self.feature_dim = feature_dim

        axis_dim = feature_dim // 3
        if axis_dim % 2 != 0:
            raise ValueError(
                f"feature_dim // 3 must be even for 3D rotary encoding, but got axis_dim={axis_dim}"
            )

        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.shape != cos.shape or x.shape != sin.shape:
            raise ValueError(
                f"x, cos, sin must have the same shape, but got "
                f"{tuple(x.shape)}, {tuple(cos.shape)}, {tuple(sin.shape)}"
            )

        x_half_turn = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)
        return x * cos + x_half_turn * sin

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if xyz.ndim < 2 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must have shape [..., 3], but got {tuple(xyz.shape)}")

        angle_x = xyz[..., 0:1].unsqueeze(-1) * self.inv_freq
        angle_y = xyz[..., 1:2].unsqueeze(-1) * self.inv_freq
        angle_z = xyz[..., 2:3].unsqueeze(-1) * self.inv_freq

        sin_x = torch.repeat_interleave(torch.sin(angle_x).squeeze(-2), 2, dim=-1)
        cos_x = torch.repeat_interleave(torch.cos(angle_x).squeeze(-2), 2, dim=-1)
        sin_y = torch.repeat_interleave(torch.sin(angle_y).squeeze(-2), 2, dim=-1)
        cos_y = torch.repeat_interleave(torch.cos(angle_y).squeeze(-2), 2, dim=-1)
        sin_z = torch.repeat_interleave(torch.sin(angle_z).squeeze(-2), 2, dim=-1)
        cos_z = torch.repeat_interleave(torch.cos(angle_z).squeeze(-2), 2, dim=-1)

        cos = torch.cat((cos_x, cos_y, cos_z), dim=-1)
        sin = torch.cat((sin_x, sin_y, sin_z), dim=-1)
        return cos, sin


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