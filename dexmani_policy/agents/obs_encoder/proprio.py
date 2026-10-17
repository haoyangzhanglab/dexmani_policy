"""Proprioceptive state and wrist–fingertip spatial encoders."""

from __future__ import annotations

import math
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
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FourierPositionEmbedding3D(nn.Module):
    """Continuous XYZ Fourier features, axis-major sin/cos at frequencies 2**k."""

    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        if type(num_frequencies) is not int or num_frequencies <= 0:
            raise ValueError("num_frequencies must be a positive integer")
        self.out_dim = 6 * num_frequencies
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim < 1 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must have shape [..., 3], got {tuple(xyz.shape)}")
        angles = xyz.unsqueeze(-1) * self.frequencies.to(dtype=xyz.dtype)
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)


class HandSpatialEncoder(nn.Module):
    """Encode finite world-frame XYZ [BT,6,3] as wrist–fingertip tokens [BT,6,D]."""

    KEYPOINTS = ("wrist", "thumb", "index", "middle", "ring", "pinky")

    def __init__(
        self,
        token_dim: int = 192,
        hidden_dim: int = 128,
        position_scale: float = 0.25,
        hand_scale: float = 0.10,
        num_frequencies: int = 2,
        use_wrist_offset: bool = True,
    ) -> None:
        super().__init__()
        if token_dim < 2 or hidden_dim < 1:
            raise ValueError(
                "token_dim >= 2 and hidden_dim >= 1 prevent degenerate features"
            )
        for name, scale in (
            ("position_scale", position_scale),
            ("hand_scale", hand_scale),
        ):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.token_dim = token_dim
        self.position_scale = float(position_scale)
        self.hand_scale = float(hand_scale)
        self.position_pe = (
            FourierPositionEmbedding3D(num_frequencies) if num_frequencies else None
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3 + 6 * num_frequencies, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.relative_proj = (
            nn.Linear(3, token_dim, bias=False) if use_wrist_offset else None
        )
        if self.relative_proj is not None:
            # Start as the absolute-only model; the linear weights still learn.
            nn.init.zeros_(self.relative_proj.weight)
        self.keypoint_embedding = nn.Parameter(torch.empty(1, 6, token_dim))
        nn.init.normal_(self.keypoint_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[1:] != (6, 3):
            raise ValueError("xyz must have shape [BT,6,3]")
        if not xyz.is_floating_point():
            raise TypeError("xyz must be floating point, preferably FP32")

        with torch.autocast(device_type=xyz.device.type, enabled=False):
            points = xyz.float()
            scaled = points / self.position_scale
            features = scaled
            if self.position_pe is not None:
                features = torch.cat((scaled, self.position_pe(math.pi * scaled)), -1)
            if self.relative_proj is not None:
                offsets = (points - points[:, :1]) / self.hand_scale

        dtype = self.position_mlp[0].weight.dtype
        tokens = self.position_mlp(features.to(dtype)).float()
        if self.relative_proj is not None:
            tokens = tokens + self.relative_proj(offsets.to(dtype)).float()
        # FP32 additions preserve small identity/residual updates under BF16 autocast.
        tokens = tokens + self.keypoint_embedding.float()
        return self.output_norm(tokens.to(self.output_norm.weight.dtype))

    @property
    def out_dim(self) -> int:
        return self.token_dim

    @property
    def out_shape(self) -> tuple[int, int]:
        return (6, self.token_dim)
