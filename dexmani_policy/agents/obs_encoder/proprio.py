"""Proprioceptive state and measured wrist–fingertip state encoders."""

from __future__ import annotations

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


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Decode consecutive rotation columns; caller supplies FP32 outside autocast."""
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class HandGeometryEncoder(nn.Module):
    """Base-frame wrist and anchors, wrist-local finger geometry; no joint inputs."""

    def __init__(self):
        super().__init__()
        self.wrist_mlp = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 192))
        self.finger_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 192))
        self.body_identity = nn.Parameter(torch.empty(1, 6, 192))
        nn.init.normal_(self.body_identity, std=0.02)
        self.norm = nn.LayerNorm(192)

    def forward(self, eef_pose, fingertip_xyz):
        with torch.autocast(device_type=eef_pose.device.type, enabled=False):
            pose = eef_pose.float()
            rotation = _rotation_6d_to_matrix(pose[:, 3:])
            local = (fingertip_xyz.float() - pose[:, None, :3]) @ rotation
            wrist = torch.cat((pose[:, :3] / 0.25, rotation[:, :, 0], rotation[:, :, 1]), -1)
            anchors = torch.cat((pose[:, None, :3], fingertip_xyz.float()), 1)
        tokens = torch.cat((self.wrist_mlp(wrist)[:, None], self.finger_mlp(local / 0.10)), 1)
        return {"tokens": self.norm(tokens.float() + self.body_identity), "anchors_base": anchors}
