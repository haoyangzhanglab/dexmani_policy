import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.common.utils import (
    farthest_point_sample,
    index_points,
    knn_point,
)


class PointGroupNorm(nn.Module):
    def __init__(self, num_channels: int, max_groups: int = 8):
        super().__init__()
        # Pick the largest groups where channels are divisible and >= 8 per group
        for groups in (max_groups, 4, 2, 1):
            if num_channels % groups == 0 and num_channels // groups >= 8:
                break
        self.norm = nn.GroupNorm(groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RawPointEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, max_groups: int = 8):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.norm = PointGroupNorm(embed_dim, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(point_features)))


class ResidualPointMLP2d(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5, max_groups: int = 8):
        super().__init__()
        hidden_channels = max(int(channels * expansion), 32)
        self.net1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = PointGroupNorm(hidden_channels, max_groups=max_groups)
        self.net2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.norm2 = PointGroupNorm(channels, max_groups=max_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.norm1(self.net1(x)))
        x = self.norm2(self.net2(x))
        return self.act(x + identity)


class PositionalEncodingGeometry(nn.Module):
    def __init__(self, out_dim: int, alpha: float, beta: float):
        super().__init__()
        self.out_dim = out_dim
        self.beta = beta
        self.in_dim = 3

        feat_dim = max((self.out_dim + self.in_dim * 2 - 1) // (self.in_dim * 2), 1)
        feat_range = torch.arange(feat_dim, dtype=torch.float32)
        dim_embed = torch.pow(torch.tensor(alpha, dtype=torch.float32), feat_range / feat_dim)
        self.register_buffer("dim_embed", dim_embed, persistent=False)

    def forward(self, knn_xyz: torch.Tensor, knn_x: torch.Tensor) -> torch.Tensor:
        batch_size, _, group_num, neighbor_num = knn_xyz.shape
        dim_embed = self.dim_embed.to(device=knn_xyz.device, dtype=knn_xyz.dtype)
        div_embed = self.beta * knn_xyz.unsqueeze(-1) / dim_embed
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], dim=-1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(batch_size, -1, group_num, neighbor_num)
        position_embed = position_embed[:, : self.out_dim]
        return knn_x + position_embed


class FPSkNN(nn.Module):
    def __init__(self, group_num: int, k_neighbors: int):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, point_coordinates: torch.Tensor, point_features: torch.Tensor):
        _, fps_idx = farthest_point_sample(point_coordinates, self.group_num)
        local_coordinates = index_points(point_coordinates, fps_idx)
        local_features = index_points(point_features, fps_idx)

        knn_idx = knn_point(self.k_neighbors, point_coordinates, local_coordinates)
        knn_coordinates = index_points(point_coordinates, knn_idx)
        knn_features = index_points(point_features, knn_idx)
        return local_coordinates, local_features, knn_coordinates, knn_features


class LocalGeometryAggregation(nn.Module):
    def __init__(
        self,
        out_dim: int,
        alpha: float,
        beta: float,
        block_num: int,
        point_cloud_type: str,
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        self.point_cloud_type = point_cloud_type
        self.positional_encoding = PositionalEncodingGeometry(out_dim, alpha, beta)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualPointMLP2d(
                    out_dim,
                    expansion=residual_expansion,
                    max_groups=max_groups,
                )
                for _ in range(block_num)
            ]
        )

    def normalize_knn_coordinates(
        self,
        local_coordinates: torch.Tensor,
        knn_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        center = local_coordinates.unsqueeze(2)
        relative_xyz = knn_coordinates - center

        if self.point_cloud_type == "scan":
            scale = relative_xyz.abs().amax(dim=2, keepdim=True).clamp_min(1e-5)
            return relative_xyz / scale

        if self.point_cloud_type == "mn40":
            scale = torch.std(relative_xyz).clamp_min(1e-5)
            return relative_xyz / scale

        return relative_xyz

    def forward(
        self,
        local_coordinates: torch.Tensor,
        local_features: torch.Tensor,
        knn_coordinates: torch.Tensor,
        knn_features: torch.Tensor,
    ) -> torch.Tensor:
        knn_coordinates = self.normalize_knn_coordinates(local_coordinates, knn_coordinates)
        center_features = local_features.unsqueeze(2).expand(-1, -1, knn_features.size(2), -1)
        knn_features = torch.cat([knn_features, center_features], dim=-1)

        knn_coordinates = knn_coordinates.permute(0, 3, 1, 2).contiguous()
        knn_features = knn_features.permute(0, 3, 1, 2).contiguous()

        knn_features = self.positional_encoding(knn_coordinates, knn_features)
        return self.residual_blocks(knn_features)


class ParametricEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_points: int,
        num_stages: int,
        embed_dim: int,
        k_neighbors,
        alpha: float,
        beta: float,
        lga_blocks,
        dim_expansion,
        point_cloud_type: str,
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        self.raw_point_embedding = RawPointEmbedding(in_channels, embed_dim, max_groups=max_groups)
        self.fps_knn_stages = nn.ModuleList()
        self.local_geometry_aggregation_stages = nn.ModuleList()

        k_neighbors = PointPNTokenizer.resolve_stage_values(k_neighbors, num_stages, "k_neighbors")
        lga_blocks = PointPNTokenizer.resolve_stage_values(lga_blocks, num_stages, "lga_blocks")
        dim_expansion = PointPNTokenizer.resolve_stage_values(dim_expansion, num_stages, "dim_expansion")

        out_dim = embed_dim
        group_num = input_points
        for stage_index in range(num_stages):
            out_dim = out_dim * dim_expansion[stage_index]
            group_num = max(group_num // 2, 1)

            self.fps_knn_stages.append(
                FPSkNN(
                    group_num=group_num,
                    k_neighbors=k_neighbors[stage_index],
                )
            )
            self.local_geometry_aggregation_stages.append(
                LocalGeometryAggregation(
                    out_dim=out_dim,
                    alpha=alpha,
                    beta=beta,
                    block_num=lga_blocks[stage_index],
                    point_cloud_type=point_cloud_type,
                    residual_expansion=residual_expansion,
                    max_groups=max_groups,
                )
            )

    def forward(self, point_coordinates: torch.Tensor, point_features: torch.Tensor):
        point_features = self.raw_point_embedding(point_features)
        intermediate_outputs = {
            "raw_point_feature": point_features,
        }

        for stage_index, (fps_knn, lga) in enumerate(
            zip(self.fps_knn_stages, self.local_geometry_aggregation_stages)
        ):
            local_coordinates, local_features, knn_coordinates, knn_features = fps_knn(
                point_coordinates,
                point_features.transpose(1, 2).contiguous(),
            )
            knn_features = lga(local_coordinates, local_features, knn_coordinates, knn_features)
            point_coordinates = local_coordinates
            point_features = knn_features.max(dim=-1).values  # inline Pooling

            intermediate_outputs[f"stage_{stage_index}_coordinates"] = point_coordinates
            intermediate_outputs[f"stage_{stage_index}_feature"] = point_features

        intermediate_outputs["global_feature"] = point_features.max(dim=-1).values
        return point_coordinates, point_features, intermediate_outputs


class PointPNTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        input_points: int = 1024,
        num_stages: int = 3,
        embed_dim: int = 64,
        beta: float = 16.0,
        alpha: float = 100.0,
        lga_blocks=(2, 2, 1),
        dim_expansion=(2, 2, 2),
        point_cloud_type: str = "scan",
        k_neighbors=(24, 24, 16),
        residual_expansion: float = 0.5,
        max_groups: int = 8,
    ):
        super().__init__()
        if in_channels < 3:
            raise ValueError("in_channels must be at least 3 because xyz is required")

        self.in_channels = in_channels
        self.input_points = input_points
        self.encoder = ParametricEncoder(
            in_channels=in_channels,
            input_points=input_points,
            num_stages=num_stages,
            embed_dim=embed_dim,
            k_neighbors=k_neighbors,
            alpha=alpha,
            beta=beta,
            lga_blocks=lga_blocks,
            dim_expansion=dim_expansion,
            point_cloud_type=point_cloud_type,
            residual_expansion=residual_expansion,
            max_groups=max_groups,
        )
        self.out_channels = embed_dim
        for expansion in self.resolve_stage_values(dim_expansion, num_stages, "dim_expansion"):
            self.out_channels *= expansion

    def forward(
        self,
        pointcloud: torch.Tensor,
        return_intermediate: bool = False,
    ):
        if pointcloud.ndim != 3:
            raise ValueError(f"pointcloud must be [B, N, C], but got shape {tuple(pointcloud.shape)}")
        if pointcloud.size(1) != self.input_points:
            raise ValueError(
                f"pointcloud has {pointcloud.size(1)} points, but input_points={self.input_points}"
            )
        if pointcloud.size(-1) < self.in_channels:
            raise ValueError(
                f"pointcloud has {pointcloud.size(-1)} channels, but in_channels={self.in_channels}"
            )

        point_coordinates = pointcloud[..., :3]
        point_features = pointcloud[..., : self.in_channels].transpose(1, 2).contiguous()
        final_coordinates, final_features, intermediate_outputs = self.encoder(point_coordinates, point_features)

        patch_token = final_features.permute(0, 2, 1).contiguous()  # (B, seq_len, C)
        patch_center = final_coordinates  # (B, seq_len, 3)

        if return_intermediate:
            return patch_token, patch_center, intermediate_outputs
        return patch_token, patch_center

    def get_global_token(self, patch_token: torch.Tensor) -> torch.Tensor:
        return patch_token.max(dim=1, keepdim=True).values  # (B, 1, C)

    @staticmethod
    def resolve_stage_values(value, num_stages: int, name: str):
        if len(value) != num_stages:
            raise ValueError(f"{name} must have length {num_stages}, but got {len(value)}")
        return tuple(value)


def example():
    batch_size, num_points = 2, 1024

    xyz = torch.empty(batch_size, num_points, 3)
    xyz[..., 0] = torch.rand(batch_size, num_points) * 0.6 - 0.3
    xyz[..., 1] = torch.rand(batch_size, num_points) * 0.8 - 0.4
    xyz[..., 2] = torch.rand(batch_size, num_points) * 0.5

    rgb = torch.rand(batch_size, num_points, 3)
    pointcloud = torch.cat([xyz, rgb], dim=-1)

    model = PointPNTokenizer(
        in_channels=6,
        input_points=num_points,
        num_stages=3,
        embed_dim=64,
        beta=16.0,
        alpha=100.0,
        lga_blocks=(2, 2, 1),
        dim_expansion=(2, 2, 2),
        point_cloud_type="scan",
        k_neighbors=(24, 24, 16),
    )

    patch_token, patch_center, intermediate_outputs = model(
        pointcloud,
        return_intermediate=True,
    )
    global_token = model.get_global_token(patch_token)

    print("input_pointcloud:", tuple(pointcloud.shape))
    for name, value in intermediate_outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("patch_token:", tuple(patch_token.shape))
    print("patch_center:", tuple(patch_center.shape))
    print("global_token:", tuple(global_token.shape))


if __name__ == "__main__":
    example()