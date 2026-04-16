import torch
import torch.nn as nn
from typing import Tuple

from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    farthest_point_sample,
    index_points,
    query_ball_point,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.position_encodings import (
    RelativePositionalEncoding3D,
    SinusoidalPosEmb3D,
)
from dexmani_policy.agents.obs_encoder.pointcloud.common.layers import PointMLP


class LocalPatchEncoder(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        patch_channels: int,
        radius: float,
        num_neighbors: int,
        position_encoding_dim: int = 24,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(position_encoding_dim)
        self.point_mlp = nn.Sequential(
            PointMLP(stem_channels + 3 + position_encoding_dim, patch_channels),
            PointMLP(patch_channels, patch_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor, patch_center: torch.Tensor):
        neighbor_idx = query_ball_point(self.radius, self.num_neighbors, xyz, patch_center)
        grouped_xyz = index_points(xyz, neighbor_idx)
        grouped_feature = index_points(point_feature, neighbor_idx)
        relative_xyz = grouped_xyz - patch_center.unsqueeze(2)
        relative_position = self.relative_position_encoding(relative_xyz)
        grouped_input = torch.cat((grouped_feature, relative_xyz, relative_position), dim=-1)
        patch_feature = self.point_mlp(grouped_input).max(dim=2).values
        return patch_feature, neighbor_idx


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
        self.center_position_encoding = SinusoidalPosEmb3D(96)
        self.token_projection = nn.Sequential(
            PointMLP(stem_channels + len(self.scale_encoders) * token_channels + 96, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        patch_center, center_idx = farthest_point_sample(xyz, self.num_patches)
        center_feature = index_points(point_feature, center_idx)

        scale_features = []
        for scale_encoder in self.scale_encoders:
            patch_feature, neighbor_idx = scale_encoder(xyz, point_feature, patch_center)
            scale_features.append(patch_feature)

        center_position = self.center_position_encoding(patch_center)
        patch_token = self.token_projection(torch.cat((center_feature, *scale_features, center_position), dim=-1))

        return patch_token, patch_center


class GlobalSceneTokenizer(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        token_channels: int,
        radius: float = 0.16,
        num_neighbors: int = 32,
    ):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.relative_position_encoding = RelativePositionalEncoding3D(24)
        self.absolute_position_encoding = SinusoidalPosEmb3D(96)
        self.scene_point_mlp = nn.Sequential(
            PointMLP(stem_channels + 3 + 24 + 96, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )
        self.scene_projection = nn.Sequential(
            PointMLP(token_channels, token_channels),
            PointMLP(token_channels, token_channels, use_activation=False),
        )

    def forward(self, xyz: torch.Tensor, point_feature: torch.Tensor):
        neighbor_idx = query_ball_point(self.radius, self.num_neighbors, xyz, xyz)
        grouped_xyz = index_points(xyz, neighbor_idx)
        grouped_feature = index_points(point_feature, neighbor_idx)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)
        relative_position = self.relative_position_encoding(relative_xyz)
        absolute_position = self.absolute_position_encoding(xyz)
        absolute_position = absolute_position.unsqueeze(2).expand(-1, -1, grouped_feature.size(2), -1)

        grouped_input = torch.cat((grouped_feature, relative_xyz, relative_position, absolute_position), dim=-1)
        scene_point_feature = self.scene_point_mlp(grouped_input).max(dim=2).values
        global_scene_token = self.scene_projection(scene_point_feature.max(dim=1, keepdim=True).values)

        return global_scene_token


class PointNextPatchTokenizer(nn.Module):
    """多尺度 Point Patch Token 提取器。"""

    def __init__(
        self,
        input_channels: int = 6,
        stem_channels: int = 64,
        token_channels: int = 128,
        num_patches: int = 64,
        patch_radii: Tuple[float, ...] = (0.04, 0.08),
        patch_neighbors: Tuple[int, ...] = (16, 32),
        global_radius: float = 0.16,
        global_neighbors: int = 32,
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3 because xyz is required")

        self.input_channels = input_channels
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
        self.global_scene_tokenizer = GlobalSceneTokenizer(
            stem_channels=stem_channels,
            token_channels=token_channels,
            radius=global_radius,
            num_neighbors=global_neighbors,
        )

    def forward(self, pointcloud: torch.Tensor):
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(-1) < self.input_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but input_channels={self.input_channels}"
            )

        xyz = pointcloud[..., :3]
        input_feature = pointcloud[..., : self.input_channels]
        point_feature = self.geometry_stem(input_feature)

        patch_token, patch_center = self.local_patch_tokenizer(xyz, point_feature)
        # 缓存用于 get_global_token
        self._last_xyz = xyz
        self._last_point_feature = point_feature

        return patch_token, patch_center

    def get_global_token(self, patch_token: torch.Tensor) -> torch.Tensor:
        """从 patch_token 衍生全局向量，委托 GlobalSceneTokenizer 做复杂聚合。"""
        return self.global_scene_tokenizer(self._last_xyz, self._last_point_feature)

    @property
    def out_shape(self) -> int:
        return self.token_channels


def example() -> None:
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5
    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    print("=== PointNextPatchTokenizer Example ===")
    model = PointNextPatchTokenizer(
        input_channels=6,
        stem_channels=64,
        token_channels=128,
        num_patches=96,
        patch_radii=(0.04, 0.08),
        patch_neighbors=(16, 32),
        global_radius=0.16,
        global_neighbors=32,
    )
    with torch.no_grad():
        patch_token, patch_center = model(pointcloud)
        global_token = model.get_global_token(patch_token)

    print("input:", tuple(pointcloud.shape))
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))
    print("out_shape:", model.out_shape)


if __name__ == "__main__":
    example()