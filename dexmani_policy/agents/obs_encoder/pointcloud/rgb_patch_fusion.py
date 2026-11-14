"""Average valid RGB samples within each point patch, then fuse with geometry."""

import torch
import torch.nn.functional as F
from torch import nn

from .camera_projection import ImageProjection


class RGBPatchFusion(nn.Module):
    def __init__(self, image_channels: int, token_channels: int = 192):
        super().__init__()
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
        if rgb_grid.shape[0] != geometry_tokens.shape[0]:
            raise ValueError("RGB and geometry must have the same BT")
        if (
            grid.ndim != 4
            or grid.shape[:2] != geometry_tokens.shape[:2]
            or grid.shape[-1] != 2
            or grid.shape[2] < 1
            or valid.shape != grid.shape[:-1]
        ):
            raise ValueError("projection requires grid [BT,G,K,2] and valid [BT,G,K]")
        with torch.autocast(device_type=rgb_grid.device.type, enabled=False):
            rgb_input = rgb_grid.permute(0, 3, 1, 2).float()
            if valid.shape[2] == 1:
                sampled = F.grid_sample(
                    rgb_input, grid,
                    mode="bilinear", padding_mode="border", align_corners=False,
                ).squeeze(-1).transpose(1, 2)
                group_valid = valid.squeeze(2)
            else:
                counts = valid.sum(dim=2, keepdim=True)
                group_valid = counts.squeeze(-1) > 0
                counts = counts.clamp_min(1)
                sampled_groups = []
                # Bound temporary FP32 samples while retaining every KNN neighbor.
                for start in range(0, grid.shape[1], 8):
                    neighbor_valid = valid[:, start : start + 8]
                    sampled = F.grid_sample(
                        rgb_input, grid[:, start : start + 8],
                        mode="bilinear", padding_mode="border", align_corners=False,
                    ).permute(0, 2, 3, 1)
                    sampled = torch.where(neighbor_valid.unsqueeze(-1), sampled, 0.0).sum(dim=2)
                    sampled_groups.append(sampled / counts[:, start : start + 8])
                sampled = torch.cat(sampled_groups, dim=1)
        semantic = self.semantic_projection(
            self.rgb_norm(sampled.to(self.rgb_norm.weight.dtype)),
        )
        # Mask after the projection so its bias cannot fill unsupported patches.
        semantic = torch.where(group_valid.unsqueeze(-1), semantic, 0.0)
        geometry = self.geometry_norm(geometry_tokens)
        return torch.cat((geometry, semantic.to(geometry.dtype)), dim=-1)
