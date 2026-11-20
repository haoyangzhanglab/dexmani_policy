import torch
import torch.nn.functional as F
from torch import nn


class HandKinematicEncoder(nn.Module):
    """Encode the Base-frame wrist pose and Wrist-frame fingertip positions."""

    wrist_position_scale = 0.25
    finger_position_wrist_scale = 0.10

    def __init__(self) -> None:
        super().__init__()
        self.wrist_mlp = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 192))
        self.finger_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 192))
        self.slot_embeddings = nn.Parameter(torch.empty(1, 6, 192))
        nn.init.normal_(self.slot_embeddings, std=0.02)
        self.norm = nn.LayerNorm(192)

    def forward(
        self, eef_pose: torch.Tensor, fingertip_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokens and metric Base-frame landmarks, both wrist first."""
        if fingertip_xyz.shape != (eef_pose.shape[0], 5, 3):
            raise ValueError("fingertip_xyz must have shape [B, 5, 3] with the same batch as eef_pose")
        wrist_position_base, rotation_6d = eef_pose.float().split((3, 6), dim=-1)
        fingertip_positions_base = fingertip_xyz.float()
        wrist_input = torch.cat((wrist_position_base / self.wrist_position_scale, rotation_6d), dim=-1)

        with torch.autocast(device_type=wrist_position_base.device.type, enabled=False):
            a1 = rotation_6d[:, :3]
            a2 = rotation_6d[:, 3:6]
            b1 = F.normalize(a1, dim=-1)
            b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)
            rotation_base_from_wrist = torch.stack((b1, b2, b3), dim=-1)
            relative_xyz_base = fingertip_positions_base - wrist_position_base.unsqueeze(1)
            # Right-multiplying row vectors applies the inverse rotation.
            fingertip_positions_wrist = relative_xyz_base @ rotation_base_from_wrist
            finger_inputs = fingertip_positions_wrist / self.finger_position_wrist_scale

        wrist_token = self.wrist_mlp(wrist_input).unsqueeze(1)
        finger_tokens = self.finger_mlp(finger_inputs)
        hand_tokens = torch.cat((wrist_token, finger_tokens), dim=1)
        hand_landmarks_base = torch.cat((wrist_position_base.unsqueeze(1), fingertip_positions_base), dim=1)
        return self.norm(hand_tokens + self.slot_embeddings), hand_landmarks_base
