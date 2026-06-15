import torch
import torch.nn as nn
import pytorch3d.ops as torch3d_ops


def farthest_point_sample(
    pointcloud: torch.Tensor,
    num_samples: int = 1024,
    use_random: bool = True,
    random_start: bool = True,
    shuffle_output: bool = True,
    random_noise_scale: float = 0.0,
):
    """Farthest point sampling with optional randomness for data augmentation.

    Standard FPS is deterministic: given the same point cloud, it always selects
    the same subset in the same order.  This can cause the policy to overfit to
    the deterministic sampling pattern.  Randomizing the start point and output
    order acts as a cheap, effective regulariser (R3D-Policy, 2026).

    Args:
        pointcloud:          (B, N, C>=3) – xyz must be the first three channels.
        num_samples:         number of points to keep.
        use_random:          master switch – when ``False``, FPS is fully
                             deterministic regardless of the flags below.
        random_start:        pick a random start point instead of the default
                             (farthest point from the centroid).
        shuffle_output:      randomly permute the output order after sampling.
        random_noise_scale:  std of Gaussian noise added to xyz *during FPS only*
                             (0 = no noise).  The output points are unaffected.
    Returns:
        sampled_points: (B, num_samples, C) — selected points.
        sample_idx:     (B, num_samples)    — indices into the input.
    """
    if pointcloud.ndim != 3 or pointcloud.size(-1) < 3:
        raise ValueError(f"pointcloud must have shape [B, N, C>=3], but got {tuple(pointcloud.shape)}")

    num_samples = min(num_samples, pointcloud.size(1))
    xyz = pointcloud[..., :3]  # (B, N, 3)
    B, N, _ = xyz.shape

    if not use_random:
        # Fast path — original deterministic FPS.
        sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
            points=xyz, K=num_samples,
        )
        sampled_points = (
            sampled_xyz if pointcloud.size(-1) == 3
            else index_points(pointcloud, sample_idx)
        )
        return sampled_points.contiguous(), sample_idx

    # ── FPS with randomisation (adapted from R3D-Policy) ──
    if random_start:
        start_indices = torch.randint(0, N, (B,), device=pointcloud.device)
        modified_xyz = xyz.clone()
        # Vectorized swap: put randomly-chosen start point at position 0.
        arange_b = torch.arange(B, device=pointcloud.device)
        val_0 = modified_xyz[:, 0].clone()
        modified_xyz[:, 0] = modified_xyz[arange_b, start_indices]
        modified_xyz[arange_b, start_indices] = val_0
    else:
        modified_xyz = xyz
        start_indices = None  # type: ignore

    # Optional coordinate noise (used only for FPS distance computation).
    if random_noise_scale > 0:
        noisy_xyz = modified_xyz + torch.randn_like(modified_xyz) * random_noise_scale
    else:
        noisy_xyz = modified_xyz

    sampled_xyz, sample_idx = torch3d_ops.sample_farthest_points(
        points=noisy_xyz, K=num_samples,
    )

    # Map back indices when random-start was used.
    if random_start:
        # Vectorized: swap index 0 ↔ start_indices[b] in each row of sample_idx.
        mask_0 = sample_idx == 0                                 # (B, K)
        si_expanded = start_indices.unsqueeze(1)                  # (B, 1)
        mask_si = sample_idx == si_expanded                       # (B, K)
        sample_idx[mask_0] = si_expanded.expand(B, num_samples)[mask_0]
        sample_idx[mask_si] = 0

    # ── Shuffle ──
    shuffle_perm = None
    if shuffle_output:
        # Vectorized: generate a random permutation per batch item.
        shuffle_perm = torch.rand(B, num_samples, device=pointcloud.device).argsort(dim=1)
        sample_idx = torch.gather(sample_idx, 1, shuffle_perm)

    # ── Build output ──
    # Fast path: pc_dim=3 + no coordinate noise → pytorch3d already returned
    # valid XYZ.  Only need to match the shuffle (if any), no full gather.
    if random_noise_scale == 0.0 and pointcloud.size(-1) == 3:
        if shuffle_perm is not None:
            sampled_xyz = sampled_xyz.gather(
                1, shuffle_perm.unsqueeze(-1).expand(-1, -1, 3),
            )
        return sampled_xyz.contiguous(), sample_idx

    # Gather full channels using the (possibly remapped & shuffled) indices.
    sampled_points = torch.gather(
        pointcloud, 1,
        sample_idx.unsqueeze(-1).long().expand(-1, -1, pointcloud.shape[-1]),
    )
    return sampled_points.contiguous(), sample_idx


def preprocess_point_cloud(
    pc: torch.Tensor,
    num_points: int,
    use_coord_only: bool,
    fps_random_config: dict | None = None,
) -> torch.Tensor:
    """Slice channels, cast to fp32, and apply farthest-point sampling.

    Shared by ``DP3ObsEncoder``, ``MoEObsEncoder``, ``ManiflowObsEncoder``.
    """
    if use_coord_only:
        pc = pc[..., :3]
    if pc.dtype != torch.float32:
        pc = pc.float()
    if pc.shape[1] > num_points:
        pc, _ = farthest_point_sample(pc, num_points, **(fps_random_config or {}))
    return pc


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
    fps_random_config: dict | None = None,
):
    fps_kwargs = fps_random_config or {}
    num_centers = max(1, int(sample_ratio * xyz.size(1)))
    center_xyz, fps_idx = farthest_point_sample(xyz, num_centers, **fps_kwargs)
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


def normalize_relative_xyz(
    relative_xyz: torch.Tensor,
    radius: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize relative coordinates by ball query radius."""
    return relative_xyz / max(radius, eps)


class PointMLP(nn.Module):
    """Linear + LayerNorm + GELU block used in point cloud encoders."""

    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)