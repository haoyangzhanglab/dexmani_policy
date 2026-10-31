"""Proprioceptive state and measured wrist–fingertip state encoders."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


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


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Decode consecutive rotation columns; caller supplies FP32 outside autocast."""
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class HandStateEncoder(nn.Module):
    """Encode aligned joints and measured world geometry; the wrist uses the EEF frame."""

    KEYPOINTS = ("wrist", "thumb", "index", "middle", "ring", "pinky")

    def __init__(
        self,
        token_dim: int = 192,
        hidden_dim: int = 128,
        position_scale: float = 0.25,
        hand_scale: float = 0.10,
        num_frequencies: int = 2,
    ) -> None:
        super().__init__()
        if token_dim < 2 or hidden_dim < 1:
            raise ValueError("token_dim >= 2 and hidden_dim >= 1 prevent degenerate features")
        if type(num_frequencies) is not int or num_frequencies < 0:
            raise ValueError("num_frequencies must be a nonnegative integer")
        for name, scale in (
            ("position_scale", position_scale),
            ("hand_scale", hand_scale),
        ):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.token_dim = token_dim
        self.position_scale = float(position_scale)
        self.hand_scale = float(hand_scale)
        self.position_pe = FourierPositionEmbedding3D(num_frequencies) if num_frequencies else None
        position_dim = 3 + (self.position_pe.out_dim if self.position_pe is not None else 0)
        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.arm_mlp = nn.Sequential(
            nn.Linear(7 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.finger_mlp = nn.Sequential(
            nn.Linear(3 + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.keypoint_embedding = nn.Parameter(torch.empty(1, 6, token_dim))
        nn.init.normal_(self.keypoint_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(token_dim)

    def _position_features(self, xyz: torch.Tensor, scale: float) -> torch.Tensor:
        scaled = xyz / scale
        if self.position_pe is None:
            return scaled
        return torch.cat((scaled, self.position_pe(math.pi * scaled)), -1)

    def forward(
        self,
        joint_state: torch.Tensor,
        eef_pose: torch.Tensor,
        fingertip_xyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Consume flattened frames with raw meter positions and consecutive rotation columns."""
        if joint_state.ndim != 2 or joint_state.shape[1] != 19:
            raise ValueError("joint_state must have shape [N,19]")
        batch = joint_state.shape[0]
        if eef_pose.shape != (batch, 9) or fingertip_xyz.shape != (batch, 5, 3):
            raise ValueError("eef_pose/fingertip_xyz must have shape [N,9] / [N,5,3]")
        if not all(value.is_floating_point() for value in (joint_state, eef_pose, fingertip_xyz)):
            raise TypeError("joint_state, eef_pose and fingertip_xyz must be floating point")

        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            joints = joint_state.float()
            anchors = torch.cat((eef_pose[:, None, :3].float(), fingertip_xyz.float()), 1)
            rotation = _rotation_6d_to_matrix(eef_pose[:, 3:].float())
            rot6d = torch.cat((rotation[..., :, 0], rotation[..., :, 1]), -1)
            local_xyz = (anchors[:, 1:] - anchors[:, :1]) @ rotation
            position_features = self._position_features(anchors, self.position_scale)
            local_features = self._position_features(local_xyz, self.hand_scale)
            finger_joints = torch.stack(
                (joints[:, 7:10], joints[:, 10:13], F.pad(joints[:, 13:15], (0, 1)),
                 F.pad(joints[:, 15:17], (0, 1)), F.pad(joints[:, 17:19], (0, 1))),
                1,
            )
            arm_features = torch.cat((joints[:, :7], rot6d), -1)
            finger_features = torch.cat((finger_joints, local_features), -1)

        dtype = self.position_mlp[0].weight.dtype
        position = self.position_mlp(position_features.to(dtype)).float()
        shared_state = self.arm_mlp(arm_features.to(dtype)).float()[:, None]
        finger_state = self.finger_mlp(finger_features.to(dtype)).float()
        tokens = torch.cat((position[:, :1], position[:, 1:] + finger_state), 1)
        # FP32 additions preserve small identity/residual updates under BF16 autocast.
        tokens = tokens + shared_state + self.keypoint_embedding.float()
        return {
            "hand_tokens": self.output_norm(tokens.to(self.output_norm.weight.dtype)),
            "anchor_xyz": anchors,
            "rotation_world_from_eef": rotation,
        }

    @property
    def out_dim(self) -> int:
        return self.token_dim
