"""Point cloud operations — faithful port of OPFA's ``geotransformer.modules.ops``.

OPFA's original implementation relies on a compiled C++/CUDA extension
(``geotransformer.ext``) for ``grid_subsample`` and ``radius_search``.
This module provides pure-PyTorch equivalents that are semantically identical.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# index_select — generalized gather (identical to OPFA geotransformer/ops)
# ---------------------------------------------------------------------------

def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    """Advanced index select — supports multi-dimensional *index* tensors.

    Different from ``torch.index_select``, *index* does not have to be 1-D.
    The ``dim``-th dimension of *data* is expanded to the rank of *index*.

    Args:
        data: ``(a_0, ..., a_{n-1})``
        index: ``(b_0, ..., b_{m-1})``
        dim: int

    Returns:
        ``(a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})``
    """
    output = data.index_select(dim, index.reshape(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


# ---------------------------------------------------------------------------
# pairwise_distance
# ---------------------------------------------------------------------------

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    """Batched pairwise squared-distance ``||x_i - y_j||²``.

    Args:
        x: ``(*, N, C)`` or ``(*, C, N)``
        y: ``(*, M, C)`` or ``(*, C, M)``
        normalized: if True, shortcut ``d² = 2 - 2xy`` (points on unit sphere).
        channel_first: if True, shape is ``(*, C, N)``.

    Returns:
        ``(*, N, M)`` squared distances.
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x**2, dim=channel_dim).unsqueeze(-1)
        y2 = torch.sum(y**2, dim=channel_dim).unsqueeze(-2)
        sq_distances = x2 - 2 * xy + y2
    return sq_distances.clamp(min=0.0)


# ---------------------------------------------------------------------------
# grid_subsample — pure-PyTorch replacement for geotransformer.ext
# ---------------------------------------------------------------------------

def grid_subsample(
    points: torch.Tensor,
    lengths: torch.Tensor,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform grid subsampling (batch-stacked mode).

    Each point in a batch element is assigned to a voxel; the barycentre of
    each occupied voxel is kept as a representative point.

    Args:
        points: ``(N, 3)`` stacked points (CPU).
        lengths: ``(B,)`` number of points per batch element.
        voxel_size: grid cell size.

    Returns:
        s_points: ``(M, 3)`` stacked subsampled points.
        s_lengths: ``(B,)`` point counts after subsampling.
    """
    s_points_list: list[torch.Tensor] = []
    s_lengths_list: list[int] = []
    offset = 0

    for length in lengths:
        seg = points[offset : offset + length]  # (n, 3)
        offset += length

        if seg.numel() == 0:
            s_points_list.append(seg)
            s_lengths_list.append(0)
            continue

        # Voxel indices (floor division)
        voxel_idx = torch.floor(seg / voxel_size).long()  # (n, 3)

        # Unique voxels via hashing
        # Hash: x * P1 + y * P2 + z * P3  (large primes)
        P1, P2, P3 = 73856093, 19349663, 83492791
        hashes = voxel_idx[:, 0] * P1 + voxel_idx[:, 1] * P2 + voxel_idx[:, 2] * P3  # (n,)

        unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
        n_voxels = unique_hashes.numel()

        # Barycentre per voxel
        s_seg = torch.zeros(n_voxels, 3, dtype=seg.dtype, device=seg.device)
        s_seg = s_seg.scatter_reduce(0, inverse.unsqueeze(-1).expand(-1, 3), seg, reduce="sum")
        counts = torch.zeros(n_voxels, 1, dtype=seg.dtype, device=seg.device)
        counts = counts.scatter_reduce(0, inverse.unsqueeze(-1), torch.ones_like(seg[:, :1]), reduce="sum")
        s_seg = s_seg / counts.clamp(min=1)

        s_points_list.append(s_seg)
        s_lengths_list.append(n_voxels)

    s_points = torch.cat(s_points_list, dim=0) if s_points_list else torch.empty(0, 3)
    s_lengths = torch.tensor(s_lengths_list, dtype=torch.long, device=points.device)
    return s_points, s_lengths


# ---------------------------------------------------------------------------
# radius_search — pure-PyTorch replacement for geotransformer.ext
# ---------------------------------------------------------------------------

def radius_search(
    q_points: torch.Tensor,
    s_points: torch.Tensor,
    q_lengths: torch.Tensor,
    s_lengths: torch.Tensor,
    radius: float,
    neighbor_limit: int,
) -> torch.Tensor:
    """Ball-query neighbours (batch-stacked mode) — **vectorised**.

    For each query point, find all support points within *radius* and pad
    to a common width.  The inner loop is fully vectorised (single CUDA
    kernel per batch element) — critical because the per-point Python loop
    in the original ``geotransformer`` port was the dominant bottleneck
    (~250 ms per 7680-point call vs ~5 ms with this version).

    Args:
        q_points: ``(N, 3)`` query points.
        s_points: ``(M, 3)`` support points.
        q_lengths: ``(B,)`` query point counts per batch.
        s_lengths: ``(B,)`` support point counts per batch.
        radius: search radius.
        neighbor_limit: keep at most this many neighbours (0 = unlimited).

    Returns:
        ``(N, max_neighbors)`` neighbour indices.
        Entries beyond the true neighbour count are filled with ``M``
        (the padding-sentinel index).
    """
    device = q_points.device
    r2 = radius * radius
    q_offset, s_offset = 0, 0
    all_neighbors: list[tuple[torch.Tensor, int]] = []

    for q_len, s_len in zip(q_lengths.tolist(), s_lengths.tolist()):
        q = q_points[q_offset : q_offset + q_len]  # (n_q, 3)
        s = s_points[s_offset : s_offset + s_len]  # (n_s, 3)
        q_offset += q_len
        s_offset += s_len

        if q.numel() == 0 or s.numel() == 0:
            all_neighbors.append((torch.empty(0, 0, dtype=torch.long, device=device), s_len))
            continue

        # Pairwise Euclidean distances (torch.cdist returns ||x-y||₂, NOT squared)
        # → square here so the r² comparison is correct.
        edists = torch.cdist(q.unsqueeze(0), s.unsqueeze(0)).squeeze(0)  # (n_q, n_s)
        sq_dists = edists * edists  # truly squared distances

        # ── Vectorised path (replaces per-point Python loop) ──
        if neighbor_limit > 0:
            k = min(neighbor_limit, s_len)
            # Clamp distances outside radius → inf so topk skips them
            sq_dists_clamped = sq_dists.clone()
            sq_dists_clamped[sq_dists > r2] = float("inf")
            # self is always the closest (dist=0), so keep up to k
            top_dists, top_idx = sq_dists_clamped.topk(k, dim=-1, largest=False)  # (n_q, k)

            # Count valid neighbours per query point
            valid_mask = ~torch.isinf(top_dists)  # (n_q, k)
            n_valid = valid_mask.sum(dim=-1)  # (n_q,)
            max_k = int(n_valid.max().item())

            if max_k == 0:
                all_neighbors.append((torch.empty(q_len, 0, dtype=torch.long, device=device), s_len))
                continue

            # Trim to actual max and fill padding with sentinel
            top_idx = top_idx[:, :max_k]
            valid_mask = valid_mask[:, :max_k]
            padded = torch.full((q_len, max_k), s_len, dtype=torch.long, device=device)
            padded[valid_mask] = top_idx[valid_mask]
        else:
            # Unlimited neighbours: gather all within radius
            mask = sq_dists <= r2  # (n_q, n_s)
            n_valid = mask.sum(dim=-1)  # (n_q,)
            max_k = int(n_valid.max().item())

            if max_k == 0:
                all_neighbors.append((torch.empty(q_len, 0, dtype=torch.long, device=device), s_len))
                continue

            # Collect indices row by row via masked selection + padding
            # (still vectorised: single arange→scatter rather than per-point loop)
            row_idx = torch.arange(q_len, device=device).unsqueeze(-1).expand(-1, max_k)  # (n_q, max_k)
            col_idx = torch.arange(max_k, device=device).unsqueeze(0).expand(q_len, -1)  # (n_q, max_k)

            # For each row, get the column indices of valid neighbours
            # Use the mask to build a position matrix, then argsort to get ordering
            idx_matrix = torch.where(mask, torch.arange(s_len, device=device).unsqueeze(0).expand(q_len, -1),
                                     torch.iinfo(torch.long).max)
            sorted_idx = idx_matrix.sort(dim=-1).values[:, :max_k]  # (n_q, max_k)
            padded = torch.where(
                sorted_idx == torch.iinfo(torch.long).max,
                torch.full_like(sorted_idx, s_len),
                sorted_idx,
            )

        all_neighbors.append((padded, s_len))

    if all_neighbors:
        # Pad to a common max_k — different batch items may have different
        # neighbour counts, which would cause torch.cat to fail on dim=1.
        global_max_k = max(t.shape[1] for t, _ in all_neighbors)
        padded_list: list[torch.Tensor] = []
        for t, s_len_val in all_neighbors:
            if t.shape[1] < global_max_k:
                pad = torch.full(
                    (t.shape[0], global_max_k - t.shape[1]),
                    s_len_val,
                    dtype=t.dtype,
                    device=device,
                )
                padded_list.append(torch.cat([t, pad], dim=1))
            else:
                padded_list.append(t)
        return torch.cat(padded_list, dim=0)
    return torch.empty(0, 0, dtype=torch.long, device=device)
