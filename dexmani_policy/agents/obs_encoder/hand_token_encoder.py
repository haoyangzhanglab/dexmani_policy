import torch
from torch import nn


class HandTokenEncoder(nn.Module):
    """Encode the wrist and five fingertips in anatomical order."""

    def __init__(self) -> None:
        super().__init__()
        self.wrist_mlp = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 192))
        self.finger_mlp = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 192))
        self.slot_embeddings = nn.Parameter(torch.empty(1, 6, 192))
        nn.init.normal_(self.slot_embeddings, std=0.02)
        self.norm = nn.LayerNorm(192)

    def forward(
        self, eef_pose: torch.Tensor, fingertip_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokens and metric Base-frame landmarks, both wrist first."""
        if fingertip_xyz.shape != (eef_pose.shape[0], 5, 3):
            raise ValueError("fingertip_xyz must have shape [B, 5, 3] with the same batch as eef_pose")
        eef_pose = eef_pose.float()
        fingertips_base = fingertip_xyz.float()
        wrist_xyz_base = eef_pose[:, :3]
        relative_xyz_base = fingertips_base - wrist_xyz_base.unsqueeze(1)
        wrist_input = torch.cat((wrist_xyz_base / 0.25, eef_pose[:, 3:]), dim=-1)
        finger_inputs = torch.cat((fingertips_base / 0.25, relative_xyz_base / 0.10), dim=-1)

        wrist_token = self.wrist_mlp(wrist_input).unsqueeze(1)
        finger_tokens = self.finger_mlp(finger_inputs)
        hand_tokens = torch.cat((wrist_token, finger_tokens), dim=1)
        hand_landmarks_base = torch.cat((wrist_xyz_base.unsqueeze(1), fingertips_base), dim=1)
        return self.norm(hand_tokens + self.slot_embeddings), hand_landmarks_base
