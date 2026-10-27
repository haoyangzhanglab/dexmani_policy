from __future__ import annotations

import torch


def load_pointcloud_ops():
    try:
        import pytorch3d.ops
    except (ImportError, OSError) as error:
        raise ImportError(
            "This FPS/kNN path requires PyTorch3D built for the installed Torch/CUDA. "
            "Install the matching native backend as described in README; "
            "DP3 without point downsampling does not require it."
        ) from error
    return pytorch3d.ops


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
    random_noise_scale: float = 0.0,
):
    """Sample original points (K <= N); XYZ noise affects only FPS selection."""
    torch3d_ops = load_pointcloud_ops()

    if pointcloud.size(-1) < 3:
        raise ValueError("pointcloud must have at least three XYZ channels")

    xyz = pointcloud[..., :3]
    B, N, _ = xyz.shape

    if not use_random:
        sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
            points=xyz,
            K=num_samples,
        )
        sampled_points = (
            sampled_xyz if pointcloud.size(-1) == 3 else index_points(pointcloud, sample_idx)
        )
        return sampled_points.contiguous(), sample_idx

    start_indices = None
    if use_random_start:
        start_indices = torch.randint(0, N, (B,), device=pointcloud.device)
        modified_xyz = xyz.clone()
        arange_b = torch.arange(B, device=pointcloud.device)
        val_0 = modified_xyz[:, 0].clone()
        modified_xyz[:, 0] = modified_xyz[arange_b, start_indices]
        modified_xyz[arange_b, start_indices] = val_0
    else:
        modified_xyz = xyz

    if random_noise_scale > 0:
        noisy_xyz = modified_xyz + torch.randn_like(modified_xyz) * random_noise_scale
    else:
        noisy_xyz = modified_xyz

    sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
        points=noisy_xyz,
        K=num_samples,
    )

    # Undo the initial swap to recover indices into the original cloud.
    if use_random_start:
        start = start_indices.unsqueeze(1)
        is_zero = sample_idx == 0
        is_start = sample_idx == start
        sample_idx = torch.where(is_start, 0, torch.where(is_zero, start, sample_idx))

    shuffle_perm = None
    if use_shuffle_output:
        shuffle_perm = torch.rand(B, num_samples, device=pointcloud.device).argsort(dim=1)
        sample_idx = torch.gather(sample_idx, 1, shuffle_perm)

    # Reuse native XYZ output when sampling did not perturb coordinates.
    if random_noise_scale == 0.0 and pointcloud.size(-1) == 3:
        if shuffle_perm is not None:
            sampled_xyz = sampled_xyz.gather(
                1,
                shuffle_perm.unsqueeze(-1).expand(-1, -1, 3),
            )
        return sampled_xyz.contiguous(), sample_idx

    sampled_points = index_points(pointcloud, sample_idx)
    return sampled_points.contiguous(), sample_idx


def preprocess_point_cloud(
    pc: torch.Tensor,
    num_points: int,
    use_coord_only: bool,
    fps_random_config: dict | None = None,
    training: bool = True,
) -> torch.Tensor:
    """Select channels and sample FP32 points, with deterministic FPS at evaluation."""
    if use_coord_only:
        pc = pc[..., :3]
    if pc.dtype != torch.float32:
        pc = pc.float()
    if pc.shape[1] > num_points:
        fps_config = resolve_fps_random_config(fps_random_config, training)
        pc, _ = farthest_point_sample(pc, num_points, **fps_config)
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
    result = load_pointcloud_ops().knn_points(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FPS centers, center indices and kNN member indices."""
    if xyz.shape[1] < max(num_patches, group_size):
        raise ValueError("N must be at least max(num_patches, group_size)")
    centers, center_indices = farthest_point_sample(
        xyz, num_patches, **resolve_fps_random_config(fps_random_config, training)
    )
    neighbors = knn_indices(group_size, xyz, centers, backend=knn_backend)
    return centers, center_indices, neighbors
