"""Proprioceptive state and measured wrist–fingertip state encoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.common.config import positive_float


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


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Decode consecutive nondegenerate columns; caller supplies FP32 outside autocast."""
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class HandGeometryEncoder(nn.Module):
    """Encode metric Base-frame wrist/fingertips into six wrist-first tokens."""

    def __init__(self, wrist_scale: float = 0.25, finger_scale: float = 0.10) -> None:
        super().__init__()
        self.wrist_scale = positive_float(wrist_scale, "wrist_scale")
        self.finger_scale = positive_float(finger_scale, "finger_scale")
        self.wrist_mlp = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 192))
        self.finger_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 192))
        self.body_identity = nn.Parameter(torch.empty(1, 6, 192))
        nn.init.normal_(self.body_identity, std=0.02)
        self.norm = nn.LayerNorm(192)

    def forward(
        self, eef_pose: torch.Tensor, fingertip_xyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eef_pose.ndim < 1 or eef_pose.shape[-1] != 9:
            raise ValueError(f"eef_pose requires [...,9], got {tuple(eef_pose.shape)}")
        if fingertip_xyz.ndim < 2 or fingertip_xyz.shape[-2:] != (5, 3):
            raise ValueError(f"fingertip_xyz requires [...,5,3], got {tuple(fingertip_xyz.shape)}")
        if eef_pose.shape[:-1] != fingertip_xyz.shape[:-2]:
            raise ValueError(
                f"Leading dimensions must match: eef_pose {tuple(eef_pose.shape)}, "
                f"fingertip_xyz {tuple(fingertip_xyz.shape)}"
            )
        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            pose = eef_pose.float()
            position = pose[..., :3]
            fingertips = fingertip_xyz.float()
            rotation = _rotation_6d_to_matrix(pose[..., 3:])
            local = (fingertips - position.unsqueeze(-2)) @ rotation
            fingers = local / self.finger_scale
            wrist = torch.cat(
                (position / self.wrist_scale, rotation[..., :, 0], rotation[..., :, 1]), dim=-1,
            )
            anchors = torch.cat((position.unsqueeze(-2), fingertips), dim=-2)
        tokens = torch.cat((self.wrist_mlp(wrist).unsqueeze(-2), self.finger_mlp(fingers)), dim=-2)
        return {"tokens": self.norm(tokens + self.body_identity[0]), "anchors_base": anchors}
