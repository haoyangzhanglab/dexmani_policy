import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.utils import (
    farthest_point_sample,
    group,
    index_points,
    query_ball_point,
    sample_and_group,
    sample_and_group_all,
)
from dexmani_policy.agents.obs_encoder.pointcloud.position_encodings import (
    RelativePositionalEncoding3D, 
    SinusoidalPosEmb3D,
)


class PointMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class GeometryStem(nn.Module):
    def __init__(self, input_channels: int, stem_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            PointMLP(input_channels, stem_channels),
            PointMLP(stem_channels, stem_channels),
        )

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        return self.stem(pointcloud)


class LocalPatchEncoder(nn.Module):
    def __init__(self, stem_channels: int, patch_channels: int, radius: float, num_neighbors: int, position_encoding_dim: int = 24):
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
        patch_radii: tuple[float, float],
        patch_neighbors: tuple[int, int],
    ):
        super().__init__()
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
        scale_neighbor_indices = []
        for scale_encoder in self.scale_encoders:
            patch_feature, neighbor_idx = scale_encoder(xyz, point_feature, patch_center)
            scale_features.append(patch_feature)
            scale_neighbor_indices.append(neighbor_idx)

        center_position = self.center_position_encoding(patch_center)
        patch_token = self.token_projection(torch.cat((center_feature, *scale_features, center_position), dim=-1))

        traces = {
            "patch_center": patch_center,
            "center_feature": center_feature,
            "scale_neighbor_indices": scale_neighbor_indices,
            "scale_features": scale_features,
            "patch_token": patch_token,
        }
        return patch_token, traces


class GlobalSceneTokenizer(nn.Module):
    def __init__(self, stem_channels: int, token_channels: int, radius: float = 0.20, num_neighbors: int = 32):
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

        traces = {
            "scene_neighbor_indices": neighbor_idx,
            "scene_point_feature": scene_point_feature,
            "global_scene_token": global_scene_token,
        }
        return global_scene_token, traces


class PointNextPatchTokenizer(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        stem_channels: int = 64,
        token_channels: int = 96,
        num_patches: int = 128,
        patch_radii: tuple[float, float] = (0.05, 0.10),
        patch_neighbors: tuple[int, int] = (16, 32),
    ):
        super().__init__()
        self.geometry_stem = GeometryStem(input_channels, stem_channels)
        self.local_patch_tokenizer = MultiScalePatchTokenizer(
            stem_channels=stem_channels,
            token_channels=token_channels,
            num_patches=num_patches,
            patch_radii=patch_radii,
            patch_neighbors=patch_neighbors,
        )
        self.global_scene_tokenizer = GlobalSceneTokenizer(stem_channels, token_channels)

    def forward(self, pointcloud: torch.Tensor, return_intermediate: bool = False):
        xyz = pointcloud[..., :3]
        point_feature = self.geometry_stem(pointcloud)
        patch_token, patch_traces = self.local_patch_tokenizer(xyz, point_feature)
        global_scene_token, global_traces = self.global_scene_tokenizer(xyz, point_feature)

        traces = {
            "input_xyz": xyz,
            "stem_feature": point_feature,
            **patch_traces,
            **global_traces,
        }

        if return_intermediate:
            return patch_token, global_scene_token, traces
        return patch_token, global_scene_token


def example() -> None:
    pointcloud = torch.randn(1, 1024, 3)
    model = PointNextPatchTokenizer(input_channels=3, stem_channels=64, token_channels=96, num_patches=16)
    with torch.no_grad():
        patch_token, global_scene_token, traces = model(pointcloud, return_intermediate=True)

    print("=== PointNextPatchTokenizer V2 Example ===")
    print("input:", tuple(pointcloud.shape))
    print("stem_feature:", tuple(traces["stem_feature"].shape))
    print("patch_center:", tuple(traces["patch_center"].shape))
    for scale_index, neighbor_idx in enumerate(traces["scale_neighbor_indices"]):
        print(f"scale_{scale_index}_neighbor_idx:", tuple(neighbor_idx.shape))
        print(f"scale_{scale_index}_feature:", tuple(traces['scale_features'][scale_index].shape))
    print("patch_token:", tuple(patch_token.shape))
    print("scene_point_feature:", tuple(traces["scene_point_feature"].shape))
    print("global_scene_token:", tuple(global_scene_token.shape))


if __name__ == "__main__":
    example()