import torch
import pytorch3d.ops as torch3d_ops


def farthest_point_sample(pointcloud: torch.Tensor, num_samples: int = 1024):
    if pointcloud.ndim != 3 or pointcloud.size(-1) < 3:
        raise ValueError(f"pointcloud must have shape [B, N, C>=3], but got {tuple(pointcloud.shape)}")

    num_samples = min(num_samples, pointcloud.size(1))
    sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
        points=pointcloud[..., :3],
        K=num_samples,
    )
    sampled_points = sampled_xyz if pointcloud.size(-1) == 3 else index_points(pointcloud, sample_idx)

    return sampled_points.contiguous(), sample_idx


def square_distance(source_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    if source_xyz.ndim != 3 or target_xyz.ndim != 3:
        raise ValueError(
            f"source_xyz and target_xyz must have shape [B, N, 3] / [B, M, 3], "
            f"but got {tuple(source_xyz.shape)} and {tuple(target_xyz.shape)}"
        )
    if source_xyz.size(-1) != 3 or target_xyz.size(-1) != 3:
        raise ValueError(
            f"source_xyz and target_xyz last dim must be 3, "
            f"but got {source_xyz.size(-1)} and {target_xyz.size(-1)}"
        )

    distance = (
        source_xyz.square().sum(dim=-1, keepdim=True)
        - 2 * source_xyz @ target_xyz.transpose(1, 2)
        + target_xyz.square().sum(dim=-1).unsqueeze(1)
    )
    return distance.clamp_min(0.0)


def index_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(points.size(0), device=points.device, dtype=torch.long)
    view_shape = (points.size(0),) + (1,) * (index.ndim - 1)
    return points[batch_index.view(*view_shape), index]


def knn_point(num_neighbors: int, support_xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    result = torch3d_ops.knn_points(
        p1=query_xyz,
        p2=support_xyz,
        K=min(num_neighbors, support_xyz.size(1)),
        return_sorted=False,
    )
    return result.idx if hasattr(result, "idx") else result[1]


def query_ball_point(
    radius: float,
    num_neighbors: int,
    support_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
):
    result = torch3d_ops.ball_query(
        p1=query_xyz,
        p2=support_xyz,
        K=min(num_neighbors, support_xyz.size(1)),
        radius=radius,
        return_nn=False,
    )
    neighbor_idx = result.idx if hasattr(result, "idx") else result[1]

    if (neighbor_idx < 0).any():
        nearest_idx = knn_point(1, support_xyz, query_xyz)

        valid_mask = neighbor_idx >= 0
        has_valid = valid_mask.any(dim=-1, keepdim=True)

        first_valid_pos = valid_mask.to(torch.int64).argmax(dim=-1, keepdim=True)
        first_valid_idx = torch.gather(neighbor_idx, dim=-1, index=first_valid_pos)

        fallback_idx = torch.where(has_valid, first_valid_idx, nearest_idx)
        neighbor_idx = torch.where(valid_mask, neighbor_idx, fallback_idx.expand_as(neighbor_idx))

    return neighbor_idx


def group(
    radius: float,
    num_neighbors: int,
    xyz: torch.Tensor,
    features: torch.Tensor | None,
):
    neighbor_idx = query_ball_point(radius, num_neighbors, xyz, xyz)
    grouped_xyz = index_points(xyz, neighbor_idx)
    relative_xyz = grouped_xyz - xyz.unsqueeze(2)

    if features is None:
        return relative_xyz

    grouped_features = index_points(features, neighbor_idx)
    return torch.cat((relative_xyz, grouped_features), dim=-1)


def sample_and_group(
    sample_ratio: float,
    radius: float,
    num_neighbors: int,
    xyz: torch.Tensor,
    features: torch.Tensor | None,
    returnfps: bool = False,
):
    num_centers = max(1, int(sample_ratio * xyz.size(1)))
    center_xyz, fps_idx = farthest_point_sample(xyz, num_centers)
    center_features = None if features is None else index_points(features, fps_idx)

    neighbor_idx = query_ball_point(radius, num_neighbors, xyz, center_xyz)
    grouped_xyz = index_points(xyz, neighbor_idx)
    relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)

    if features is None:
        grouped_features = relative_xyz
    else:
        neighborhood_features = index_points(features, neighbor_idx)
        grouped_features = torch.cat((relative_xyz, neighborhood_features), dim=-1)

    if returnfps:
        return center_xyz, grouped_features, grouped_xyz, fps_idx
    return center_xyz, grouped_features, center_features


def sample_and_group_all(xyz: torch.Tensor, features: torch.Tensor | None):
    center_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.unsqueeze(1)
    relative_xyz = grouped_xyz - center_xyz.unsqueeze(2)

    if features is None:
        grouped_features = relative_xyz
    else:
        grouped_features = torch.cat((relative_xyz, features.unsqueeze(1)), dim=-1)

    return center_xyz, grouped_features


def resolve_stage_values(value, num_stages: int, name: str):
    if len(value) != num_stages:
        raise ValueError(f"{name} must have length {num_stages}, but got {len(value)}")
    return tuple(value)