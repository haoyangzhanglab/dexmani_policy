"""World-frame wrist/fingertip positions -> six semantic spatial tokens.

This encoder supplies hand queries, not contact estimates or complete poses.
Hand-scene geometry and tactile fusion belong to the interaction modules.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dexmani_policy.agents.position_encodings import NeRFSinusoidalPosEmb3D


class HandSpatialEncoder(nn.Module):
    """[BT,6,3] metric XYZ -> [BT,6,D], without FK or frame rotations.

    Raw world positions and low-frequency Fourier features describe location.
    An optional, zero-initialized linear residual exposes each finger's offset
    from the wrist along the SAME world axes. It is a layout cue, not a full
    hand pose; the whole encoder is neither translation invariant nor SE(3)
    equivariant. Keep geometry inputs in FP32 when using mixed precision.
    """

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
        if type(token_dim) is not int or token_dim < 2:
            raise ValueError("token_dim must be an integer >= 2 for LayerNorm")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        for name, scale in (("position_scale", position_scale), ("hand_scale", hand_scale)):
            if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(num_frequencies) is not int or not 0 <= num_frequencies <= 16:
            raise ValueError("num_frequencies must be an integer in [0,16]")
        if type(use_wrist_offset) is not bool:
            raise ValueError("use_wrist_offset must be bool")

        self.token_dim = token_dim
        self.position_scale = float(position_scale)
        self.hand_scale = float(hand_scale)
        self.position_pe = (
            NeRFSinusoidalPosEmb3D(num_frequencies) if num_frequencies else None
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
        if xyz.ndim != 3 or xyz.shape[1:] != (6, 3) or xyz.shape[0] == 0:
            raise ValueError("xyz must have shape [BT,6,3] with BT > 0")
        if not xyz.is_floating_point():
            raise TypeError("xyz must be floating point, preferably FP32")
        if not torch.isfinite(xyz).all():
            raise ValueError("hand positions must be finite")

        with torch.autocast(device_type=xyz.device.type, enabled=False):
            points = xyz.float()
            scaled = points / self.position_scale
            features = scaled
            if self.position_pe is not None:
                features = torch.cat((scaled, self.position_pe(math.pi * scaled)), -1)
            # A world-axis displacement vector, not a wrist-frame position.
            offsets = (points - points[:, :1]) / self.hand_scale

        dtype = self.position_mlp[0].weight.dtype
        tokens = self.position_mlp(features.to(dtype)).float()
        if self.relative_proj is not None:
            tokens = tokens + self.relative_proj(offsets.to(dtype)).float()
        # Preserve small identity/residual updates under BF16 autocast.
        # Standard AMP retains FP32 parameters and returns FP32 spatial tokens.
        tokens = tokens + self.keypoint_embedding.float()
        return self.output_norm(tokens.to(self.output_norm.weight.dtype))

    @property
    def out_dim(self) -> int:
        return self.token_dim

    @property
    def out_shape(self) -> tuple[int, int]:
        return (6, self.token_dim)
