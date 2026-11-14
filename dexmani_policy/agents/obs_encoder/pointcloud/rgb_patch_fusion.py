"""Average valid RGB samples within each point patch, then fuse with geometry."""

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
        grid, valid = projection
        if (
            geometry_tokens.ndim != 3
            or geometry_tokens.shape[-1:] != self.geometry_norm.normalized_shape
        ):
            raise ValueError("geometry_tokens must have shape [BT,G,geometry_channels]")
        if (
            rgb_grid.ndim != 4
            or rgb_grid.shape[0] != geometry_tokens.shape[0]
            or rgb_grid.shape[-1:] != self.rgb_norm.normalized_shape
        ):
            raise ValueError("rgb_grid must have shape [BT,H,W,image_channels]")
        if (
            grid.ndim != 4
            or grid.shape[:2] != geometry_tokens.shape[:2]
            or grid.shape[-1] != 2
            or grid.shape[2] < 1
            or valid.shape != grid.shape[:-1]
        ):
            raise ValueError("projection requires grid [BT,G,K,2] and valid [BT,G,K]")
        with torch.autocast(device_type=rgb_grid.device.type, enabled=False):
            sampled = F.grid_sample(
                rgb_grid.permute(0, 3, 1, 2).float(),
                grid,
                mode="bilinear", padding_mode="border", align_corners=False,
            ).permute(0, 2, 3, 1)
            if valid.shape[2] == 1:
                # Avoid reductions in the default center-sampling path.
                sampled = sampled.squeeze(2)
                group_valid = valid.squeeze(2)
            else:
                count = valid.sum(dim=2, keepdim=True)
                sampled = torch.where(valid.unsqueeze(-1), sampled, 0.0).sum(dim=2)
                sampled = sampled / count.clamp_min(1)
                group_valid = count.squeeze(-1) > 0
        semantic = self.semantic_projection(
            self.rgb_norm(sampled.to(self.rgb_norm.weight.dtype)),
        )
        # Mask after the projection so its bias cannot fill unsupported patches.
        semantic = torch.where(group_valid.unsqueeze(-1), semantic, 0.0)
        geometry = self.geometry_norm(geometry_tokens)
        return torch.cat((geometry, semantic.to(geometry.dtype)), dim=-1)
