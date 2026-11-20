"""Project RGB features onto point patches and fuse them with geometry."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .camera_projection import ImageProjection, project_to_image
from .ops import index_points


def project_rgb_patches(
    obs: dict[str, torch.Tensor], points: torch.Tensor, patches: dict[str, torch.Tensor],
    *, image_processor, rgb_encoder, sampling: Literal["center", "group_mean"],
) -> tuple[torch.Tensor, ImageProjection]:
    """Build the RGB feature grid and its projection for point-patch sampling."""
    rgb, image_transform = image_processor.process_images(obs["rgb"], return_transform=True)
    image_hw = (rgb.shape[-2], rgb.shape[-1])
    # Sampling coordinates assume the patch grid covers the whole processed image.
    if any(size % rgb_encoder.patch_size for size in image_hw):
        raise ValueError("RGB input height and width must be divisible by the encoder patch_size")
    patch_grid = tuple(size // rgb_encoder.patch_size for size in image_hw)
    sample_xyz = patches["xyz"].unsqueeze(2)
    if sampling == "group_mean":
        sample_xyz = index_points(points[..., :3], patches["neighbor_idx"])
    intrinsics = obs["camera_intrinsic"]
    extrinsics = obs["camera_extrinsic"]
    projection = project_to_image(
        points=sample_xyz.flatten(1, 2),
        intrinsics=intrinsics.unflatten(-1, (3, 3)) if intrinsics.ndim == 2 else intrinsics,
        extrinsics=extrinsics.unflatten(-1, (3, 4)) if extrinsics.ndim == 2 else extrinsics,
        image_transform=image_transform,
        image_hw=image_hw,
    )
    image_grid = rgb_encoder(rgb)["patch_tokens"].unflatten(1, patch_grid)
    group_shape = sample_xyz.shape[1:3]
    return image_grid, ImageProjection(
        projection.grid_xy.unflatten(1, group_shape),
        projection.valid.unflatten(1, group_shape),
    )


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
