import torch
import pytorch3d.ops as torch3d_ops


def farthest_point_sample(pointcloud, num_samples=1024):
    """Farthest point sampling on a batched point cloud."""
    num_samples = min(num_samples, pointcloud.size(1))
    sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
        points=pointcloud[..., :3],
        K=num_samples,
    )
    sampled_points = sampled_xyz if pointcloud.size(-1) == 3 else index_points(pointcloud, sample_idx)
    return sampled_points.contiguous(), sample_idx


def square_distance(source_xyz, target_xyz):
    """Pairwise squared Euclidean distance."""
    return (
        source_xyz.square().sum(dim=-1, keepdim=True)
        - 2 * source_xyz @ target_xyz.transpose(1, 2)
        + target_xyz.square().sum(dim=-1).unsqueeze(1)
    )


def index_points(points, index):
    """Batch-aware gather for point features."""
    batch_index = torch.arange(points.size(0), device=points.device, dtype=torch.long)
    view_shape = (points.size(0),) + (1,) * (index.ndim - 1)
    return points[batch_index.view(*view_shape), index]


def knn_point(num_neighbors, support_xyz, query_xyz):
    """KNN indices from support points to query points."""
    result = torch3d_ops.knn_points(
        p1=query_xyz,
        p2=support_xyz,
        K=min(num_neighbors, support_xyz.size(1)),
        return_sorted=False,
    )
    return result.idx if hasattr(result, "idx") else result[1]


def query_ball_point(radius, num_neighbors, support_xyz, query_xyz):
    """Ball-query indices with safe padding replacement."""
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
        first_valid_idx = torch.where(neighbor_idx[..., :1] >= 0, neighbor_idx[..., :1], nearest_idx)
        neighbor_idx = torch.where(neighbor_idx >= 0, neighbor_idx, first_valid_idx.expand_as(neighbor_idx))

    return neighbor_idx


def group(radius, num_neighbors, xyz, features):
    """Group local neighborhoods around each point."""
    neighbor_idx = query_ball_point(radius, num_neighbors, xyz, xyz)
    grouped_xyz = index_points(xyz, neighbor_idx)
    relative_xyz = grouped_xyz - xyz.unsqueeze(2)

    if features is None:
        return relative_xyz

    grouped_features = index_points(features, neighbor_idx)
    return torch.cat((relative_xyz, grouped_features), dim=-1)


def sample_and_group(sample_ratio, radius, num_neighbors, xyz, features, returnfps=False):
    """FPS + local grouping used in PointNet++ style set abstraction."""
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


def sample_and_group_all(xyz, features):
    """Treat the whole point cloud as one global neighborhood."""
    center_xyz = xyz.mean(dim=1, keepdim=True)
    grouped_xyz = xyz.unsqueeze(1)

    if features is None:
        grouped_features = grouped_xyz
    else:
        grouped_features = torch.cat((grouped_xyz, features.unsqueeze(1)), dim=-1)

    return center_xyz, grouped_features

