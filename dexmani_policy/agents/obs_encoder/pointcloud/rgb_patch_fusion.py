"""Sample RGB semantics at projected point-patch centers."""

import torch
import torch.nn.functional as F
from torch import nn

from .camera_projection import ImageProjection


class RGBPatchFusion(nn.Module):
    def __init__(self, image_channels: int, token_channels: int = 192):
        super().__init__()
        if min(image_channels, token_channels) <= 0:
            raise ValueError("channel counts must be positive")
        self.rgb_norm = nn.LayerNorm(image_channels)
        self.semantic_projection = nn.Linear(image_channels, token_channels)
        self.geometry_norm = nn.LayerNorm(token_channels)

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        rgb_grid: torch.Tensor,
        projection: ImageProjection,
    ) -> torch.Tensor:
        with torch.autocast(device_type=rgb_grid.device.type, enabled=False):
            sampled = F.grid_sample(
                rgb_grid.permute(0, 3, 1, 2).float(),
                projection.grid_xy.float().unsqueeze(2),
                mode="bilinear", padding_mode="border", align_corners=False,
            ).squeeze(-1).transpose(1, 2)
        semantic = self.semantic_projection(
            self.rgb_norm(sampled.to(self.rgb_norm.weight.dtype)),
        )
        # Mask after the projection so its bias cannot fill unsupported patches.
        semantic = torch.where(projection.valid.unsqueeze(-1), semantic, 0.0)
        geometry = self.geometry_norm(geometry_tokens)
        return torch.cat((geometry, semantic.to(geometry.dtype)), dim=-1)
