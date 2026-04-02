import torch
import torch.nn as nn
from dexmani_policy.agents.obs_encoder.pointcloud.utils import (
    farthest_point_sample,
    index_points,
    knn_point,
)


class PointLayerNorm1d(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class PointLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class RawPointEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.norm = PointLayerNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(point_features)))


class ResidualPointMLP2d(nn.Module):
    def __init__(self, channels: int, stage_index: int = 0):
        super().__init__()
        hidden_channels = 32 if stage_index == 2 else max(channels // 2, 1)
        self.net1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.norm1 = PointLayerNorm2d(hidden_channels)
        self.net2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.norm2 = PointLayerNorm2d(channels)
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
        self.alpha = alpha
        self.beta = beta
        self.in_dim = 3

    def forward(self, knn_xyz: torch.Tensor, knn_x: torch.Tensor) -> torch.Tensor:
        batch_size, _, group_num, neighbor_num = knn_xyz.shape
        feat_dim = (self.out_dim + self.in_dim * 2 - 1) // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim, device=knn_xyz.device, dtype=knn_xyz.dtype)
        alpha = torch.tensor(self.alpha, device=knn_xyz.device, dtype=knn_xyz.dtype)
        dim_embed = torch.pow(alpha, feat_range / max(feat_dim, 1))
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
        stage_index: int,
        point_cloud_type: str,
    ):
        super().__init__()
        self.point_cloud_type = point_cloud_type
        self.positional_encoding = PositionalEncodingGeometry(out_dim, alpha, beta)
        self.residual_blocks = nn.Sequential(
            *[ResidualPointMLP2d(out_dim, stage_index=stage_index) for _ in range(block_num)]
        )

    def normalize_knn_coordinates(
        self,
        local_coordinates: torch.Tensor,
        knn_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if self.point_cloud_type == "mn40":
            center = local_coordinates.unsqueeze(2)
            scale = torch.std(knn_coordinates - center).clamp_min(1e-5)
            return (knn_coordinates - center) / scale

        if self.point_cloud_type == "scan":
            relative_xyz = knn_coordinates - local_coordinates.unsqueeze(2)
            scale = relative_xyz.abs().amax(dim=2, keepdim=True).clamp_min(1e-5)
            return relative_xyz / scale

        return knn_coordinates - local_coordinates.unsqueeze(2)

    def forward(
        self,
        local_coordinates: torch.Tensor,
        local_features: torch.Tensor,
        knn_coordinates: torch.Tensor,
        knn_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, group_num, neighbor_num, _ = knn_features.shape

        knn_coordinates = self.normalize_knn_coordinates(local_coordinates, knn_coordinates)
        center_features = local_features.unsqueeze(2).expand(-1, -1, neighbor_num, -1)
        knn_features = torch.cat([knn_features, center_features], dim=-1)

        knn_coordinates = knn_coordinates.permute(0, 3, 1, 2)
        knn_features = knn_features.permute(0, 3, 1, 2).reshape(batch_size, -1, group_num, neighbor_num)

        knn_features = self.positional_encoding(knn_coordinates, knn_features)
        return self.residual_blocks(knn_features)


class Pooling(nn.Module):
    def forward(self, knn_features: torch.Tensor) -> torch.Tensor:
        return knn_features.max(dim=-1).values


class ParametricEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_points: int,
        num_stages: int,
        embed_dim: int,
        k_neighbors: int,
        alpha: float,
        beta: float,
        lga_blocks,
        dim_expansion,
        point_cloud_type: str,
    ):
        super().__init__()
        self.raw_point_embedding = RawPointEmbedding(in_channels, embed_dim)
        self.fps_knn_stages = nn.ModuleList()
        self.local_geometry_aggregation_stages = nn.ModuleList()
        self.pooling_stages = nn.ModuleList()

        out_dim = embed_dim
        group_num = input_points
        for stage_index in range(num_stages):
            out_dim = out_dim * dim_expansion[stage_index]
            group_num = max(group_num // 2, 1)
            self.fps_knn_stages.append(FPSkNN(group_num=group_num, k_neighbors=k_neighbors))
            self.local_geometry_aggregation_stages.append(
                LocalGeometryAggregation(
                    out_dim=out_dim,
                    alpha=alpha,
                    beta=beta,
                    block_num=lga_blocks[stage_index],
                    stage_index=stage_index,
                    point_cloud_type=point_cloud_type,
                )
            )
            self.pooling_stages.append(Pooling())

    def forward(self, point_coordinates: torch.Tensor, point_features: torch.Tensor):
        point_features = self.raw_point_embedding(point_features)
        intermediate_outputs = {"raw_point_feature": point_features.transpose(1, 2)}

        for stage_index, (fps_knn, lga, pooling) in enumerate(
            zip(self.fps_knn_stages, self.local_geometry_aggregation_stages, self.pooling_stages)
        ):
            local_coordinates, local_features, knn_coordinates, knn_features = fps_knn(
                point_coordinates,
                point_features.transpose(1, 2),
            )
            knn_features = lga(local_coordinates, local_features, knn_coordinates, knn_features)
            point_coordinates = local_coordinates
            point_features = pooling(knn_features)

            intermediate_outputs[f"stage_{stage_index}_coordinates"] = point_coordinates
            intermediate_outputs[f"stage_{stage_index}_feature"] = point_features.transpose(1, 2)

        return point_coordinates, point_features, intermediate_outputs


class PointPNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_points: int = 1024,
        num_stages: int = 3,
        embed_dim: int = 96,
        beta: float = 100.0,
        alpha: float = 1000.0,
        lga_blocks=(2, 1, 1),
        dim_expansion=(2, 2, 2),
        point_cloud_type: str = "mn40",
        k_neighbors: int = 81,
    ):
        super().__init__()
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
        )
        self.out_channels = embed_dim
        for expansion in dim_expansion[:num_stages]:
            self.out_channels *= expansion

    def forward(self, point_features: torch.Tensor, point_coordinates: torch.Tensor):
        return self.encoder(point_coordinates, point_features)


Point_PN_scan = PointPNEncoder


def example():
    batch_size, num_points = 2, 64
    point_coordinates = torch.randn(batch_size, num_points, 3)
    point_features = point_coordinates.transpose(1, 2).contiguous()

    model = PointPNEncoder(
        in_channels=3,
        input_points=num_points,
        num_stages=3,
        embed_dim=32,
        lga_blocks=(2, 1, 1),
        dim_expansion=(2, 2, 2),
        point_cloud_type="mn40",
        k_neighbors=16,
    )

    final_coordinates, final_features, intermediate_outputs = model(point_features, point_coordinates)

    print("input_coordinates:", tuple(point_coordinates.shape))
    print("input_features:", tuple(point_features.shape))
    for name, value in intermediate_outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    print("final_coordinates:", tuple(final_coordinates.shape))
    print("final_features:", tuple(final_features.shape))


if __name__ == "__main__":
    example()