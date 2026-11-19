from __future__ import annotations

import torch


def farthest_point_sample(
    pointcloud: torch.Tensor,
    num_samples: int = 1024,
    use_random: bool = True,
    use_random_start: bool = True,
    use_shuffle_output: bool = True,
    *,
    training: bool = True,
) -> torch.Tensor:
    """Sample original points (K <= N) by XYZ distance, preserving all channels."""
    from pytorch3d.ops import sample_farthest_points

    if pointcloud.size(-1) < 3:
        raise ValueError("pointcloud must have at least three XYZ channels")

    xyz = pointcloud[..., :3]
    use_random = training and use_random
    sampled_xyz, sample_idx = sample_farthest_points(
        points=xyz,
        K=num_samples,
        random_start_point=use_random and use_random_start,
    )
    if use_random and use_shuffle_output:
        shuffle_perm = torch.rand(
            xyz.shape[0], num_samples, device=pointcloud.device,
        ).argsort(dim=1)
        if pointcloud.size(-1) == 3:
            sampled_xyz = index_points(sampled_xyz, shuffle_perm)
        else:
            sample_idx = sample_idx.gather(1, shuffle_perm)

    # Reuse native output for XYZ-only input.
    if pointcloud.size(-1) == 3:
        return sampled_xyz.contiguous()
    return index_points(pointcloud, sample_idx).contiguous()


def preprocess_point_cloud(
    pc: torch.Tensor,
    num_points: int,
    fps_random_config: dict | None = None,
    training: bool = True,
) -> torch.Tensor:
    """Sample FP32 points, with deterministic FPS at evaluation."""
    pc = pc.float()
    if pc.shape[1] > num_points:
        pc = farthest_point_sample(pc, num_points, training=training, **(fps_random_config or {}))
    return pc


def index_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = points.shape[:2]
    flat = points.reshape(batch_size, num_points, -1)
    expanded = index.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, expanded).reshape(*index.shape, *points.shape[2:])


@torch.no_grad()
def sample_point_patches(
    xyz: torch.Tensor,
    num_patches: int,
    group_size: int,
    fps_random_config: dict | None = None,
    training: bool = True,
    *,
    torch_knn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FPS centers and kNN member indices."""
    if xyz.shape[1] < max(num_patches, group_size):
        raise ValueError("N must be at least max(num_patches, group_size)")
    centers = farthest_point_sample(
        xyz, num_patches, training=training, **(fps_random_config or {}),
    )
    if torch_knn:
        distances = torch.cdist(centers, xyz)
        if group_size == 1:
            neighbors = distances.min(dim=2, keepdim=True).indices
        else:
            neighbors = distances.topk(group_size, dim=2, largest=False, sorted=False).indices
    else:
        from pytorch3d.ops import knn_points

        neighbors = knn_points(p1=centers, p2=xyz, K=group_size, return_sorted=False).idx
    return centers, neighbors
