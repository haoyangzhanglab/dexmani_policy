from __future__ import annotations

import torch


def resolve_fps_random_config(
    fps_random_config: dict | None,
    training: bool,
) -> dict:
    """Return FPS settings with deterministic sampling enforced for evaluation."""
    config = dict(fps_random_config or {})
    if not training:
        config["use_random"] = False
    return config


def farthest_point_sample(
    pointcloud: torch.Tensor,
    num_samples: int = 1024,
    use_random: bool = True,
    use_random_start: bool = True,
    use_shuffle_output: bool = True,
) -> torch.Tensor:
    """Sample original points (K <= N) by XYZ distance, preserving all channels."""
    from pytorch3d.ops import sample_farthest_points

    if pointcloud.size(-1) < 3:
        raise ValueError("pointcloud must have at least three XYZ channels")

    xyz = pointcloud[..., :3]
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
    if pc.dtype != torch.float32:
        pc = pc.float()
    if pc.shape[1] > num_points:
        fps_config = resolve_fps_random_config(fps_random_config, training)
        pc = farthest_point_sample(pc, num_points, **fps_config)
    return pc


def index_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = points.shape[:2]
    flat = points.reshape(batch_size, num_points, -1)
    expanded = index.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, expanded).reshape(*index.shape, *points.shape[2:])


def knn_indices(
    num_neighbors: int,
    support_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    *,
    backend: str = "pytorch3d",
) -> torch.Tensor:
    """Return unsorted kNN indices (K <= N), preserving backend distance/tie rules."""
    if backend == "torch":
        distances = torch.cdist(query_xyz, support_xyz)
        if num_neighbors == 1:
            return distances.min(dim=2, keepdim=True).indices
        return distances.topk(num_neighbors, dim=2, largest=False, sorted=False).indices
    if backend != "pytorch3d":
        raise ValueError(f"Unknown kNN backend: {backend}")
    from pytorch3d.ops import knn_points

    result = knn_points(
        p1=query_xyz,
        p2=support_xyz,
        K=num_neighbors,
        return_sorted=False,
    )
    return result.idx


@torch.no_grad()
def sample_point_patches(
    xyz: torch.Tensor,
    num_patches: int,
    group_size: int,
    fps_random_config: dict | None = None,
    training: bool = True,
    *,
    knn_backend: str = "pytorch3d",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FPS centers and kNN member indices."""
    if xyz.shape[1] < max(num_patches, group_size):
        raise ValueError("N must be at least max(num_patches, group_size)")
    centers = farthest_point_sample(
        xyz, num_patches, **resolve_fps_random_config(fps_random_config, training)
    )
    neighbors = knn_indices(group_size, xyz, centers, backend=knn_backend)
    return centers, neighbors
