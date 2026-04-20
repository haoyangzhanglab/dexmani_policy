import torch
import torch.nn as nn
from typing import Dict, Tuple

from dexmani_policy.agents.obs_encoder.pointcloud.common.position_encodings import (
    RelativePositionalEncoding3D,
    SinusoidalPosEmb3D,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    farthest_point_sample,
    index_points,
    query_ball_point,
)


def normalize_relative_xyz(relative_xyz: torch.Tensor, radius: float, eps: float = 1e-6) -> torch.Tensor:
    return relative_xyz / max(radius, eps)


class PointMLP(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)
        self.norm = nn.LayerNorm(output_channels)
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class LocalPatchEncoder(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        token_channels: int,
        radius: float,
        num_neighbors: int,
        position_encoding_channels: int = 24,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(position_encoding_channels)
        self.point_mlp = nn.Sequential(
            PointMLP(stem_channels + 3 + position_encoding_channels, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor, patch_center: torch.Tensor) -> torch.Tensor:
        neighbor_idx = query_ball_point(self.radius, self.num_neighbors, xyz, patch_center)
        neighbor_xyz = index_points(xyz, neighbor_idx)
        neighbor_feature = index_points(point_feature, neighbor_idx)
        relative_xyz = neighbor_xyz - patch_center.unsqueeze(2)
        normalized_relative_xyz = normalize_relative_xyz(relative_xyz, self.radius)
        relative_pos_feature = self.relative_position_encoding(normalized_relative_xyz)
        group_input = torch.cat((neighbor_feature, normalized_relative_xyz, relative_pos_feature), dim=-1)
        return self.point_mlp(group_input).max(dim=2).values


class MultiScalePatchTokenizer(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        token_channels: int,
        num_patches: int,
        patch_radii: Tuple[float, ...],
        patch_neighbors: Tuple[int, ...],
    ):
        super().__init__()
        if len(patch_radii) != len(patch_neighbors):
            raise ValueError("patch_radii and patch_neighbors must have the same length")

        self.num_patches = num_patches
        self.scale_encoders = nn.ModuleList(
            [
                LocalPatchEncoder(stem_channels, token_channels, radius, neighbors)
                for radius, neighbors in zip(patch_radii, patch_neighbors)
            ]
        )
        self.patch_center_position_encoding = SinusoidalPosEmb3D(96)
        self.token_projection = nn.Sequential(
            PointMLP(stem_channels + len(self.scale_encoders) * token_channels + 96, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        patch_center, patch_center_idx = farthest_point_sample(xyz, self.num_patches)
        patch_center_feature = index_points(point_feature, patch_center_idx)

        multi_scale_patch_feature_list = [
            scale_encoder(xyz, point_feature, patch_center) for scale_encoder in self.scale_encoders
        ]
        patch_center_pos_feature = self.patch_center_position_encoding(patch_center)
        patch_token = self.token_projection(
            torch.cat((patch_center_feature, *multi_scale_patch_feature_list, patch_center_pos_feature), dim=-1)
        )
        return patch_token, patch_center


class PointNextPatchTokenizer(nn.Module):
    supports_global_token = True
    supports_intermediate_outputs = True
    requires_fixed_num_points = False

    def __init__(
        self,
        input_channels: int = 6,
        stem_channels: int = 64,
        token_channels: int = 128,
        num_patches: int = 64,
        patch_radii: Tuple[float, ...] = (0.04, 0.08),
        patch_neighbors: Tuple[int, ...] = (16, 32),
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        self.input_channels = input_channels
        self.num_patches = num_patches
        self.token_channels = token_channels

        self.geometry_stem = nn.Sequential(
            PointMLP(input_channels, stem_channels),
            PointMLP(stem_channels, stem_channels),
        )
        self.local_patch_tokenizer = MultiScalePatchTokenizer(
            stem_channels=stem_channels,
            token_channels=token_channels,
            num_patches=num_patches,
            patch_radii=patch_radii,
            patch_neighbors=patch_neighbors,
        )
        self.global_position_embedding = SinusoidalPosEmb3D(96)
        self.global_position_projection = nn.Sequential(
            nn.Linear(96, token_channels),
            nn.LayerNorm(token_channels),
            nn.GELU(),
        )
        self.global_token_projection = nn.Sequential(
            nn.Linear(token_channels, token_channels),
            nn.LayerNorm(token_channels),
        )

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_global_token: bool = False,
        return_intermediate: bool = False,
    ):
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        xyz = pointcloud[..., :3]
        input_point_feature = pointcloud[..., : self.input_channels]
        stem_point_feature = self.geometry_stem(input_point_feature)
        patch_token, patch_center = self.local_patch_tokenizer(xyz, stem_point_feature)

        outputs = [patch_token, patch_center]
        if return_global_token:
            outputs.append(self.get_global_token(patch_token, patch_center))
        if return_intermediate:
            intermediate_outputs: Dict[str, torch.Tensor] = {
                "stem_point_feature": stem_point_feature,
                "patch_center": patch_center,
                "patch_token": patch_token,
            }
            outputs.append(intermediate_outputs)
        return tuple(outputs)

    def get_global_token(self, patch_token: torch.Tensor, patch_center: torch.Tensor) -> torch.Tensor:
        pooled_patch_token = patch_token.max(dim=1).values
        pooled_patch_center = patch_center.mean(dim=1)
        global_token_feature = pooled_patch_token + self.global_position_projection(
            self.global_position_embedding(pooled_patch_center)
        )
        return self.global_token_projection(global_token_feature).unsqueeze(1)

    @property
    def out_dim(self) -> int:
        return self.token_channels

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.num_patches, self.token_channels)


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    pointnext_tokenizer = PointNextPatchTokenizer(
        input_channels=6,
        stem_channels=64,
        token_channels=128,
        num_patches=96,
        patch_radii=(0.04, 0.08),
        patch_neighbors=(16, 32),
    )
    with torch.no_grad():
        patch_token, patch_center, global_token, intermediate_outputs = pointnext_tokenizer(
            pointcloud,
            return_global_token=True,
            return_intermediate=True,
        )

    print("=== PointNextPatchTokenizer Example ===")
    print("input:", tuple(pointcloud.shape))
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))
    for name, value in intermediate_outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("out_dim:", pointnext_tokenizer.out_dim)
    print("out_shape:", pointnext_tokenizer.out_shape)


if __name__ == "__main__":
    example()