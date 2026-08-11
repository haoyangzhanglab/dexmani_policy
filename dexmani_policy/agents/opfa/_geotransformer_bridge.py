"""Bridge to official OPFA ``geotransformer`` C++ extension.

The official OPFA geotransformer provides ``grid_subsample`` and
``radius_search`` backed by a compiled C++ extension (nanoflann KD-tree).
This module handles the required torch-library preloading (to resolve
libc10 ABI conflicts between conda and PyTorch) and re-exports those
ops with automatic ``.contiguous()`` + ``.cpu()`` wrapping (the official
extension is CPU-only and requires contiguous tensors).

Usage::

    from dexmani_policy.agents.opfa._geotransformer_bridge import (
        grid_subsample, radius_search, _ensure_geotransformer,
    )
    _ensure_geotransformer()
    # now grid_subsample(...) and radius_search(...) delegate to the
    # official C++ implementations.
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch

_OFFICIAL_DIR = os.environ.get("OPFA_OFFICIAL_DIR")

_INITIALISED = False


def _ensure_geotransformer() -> None:
    """Preload correct torch libs and register the official package path.

    Must be called **before** any ``geotransformer.*`` import.  Idempotent
    (subsequent calls are no-ops).
    """
    global _INITIALISED
    if _INITIALISED:
        return

    # Preload the PyTorch-shipped libc10 / libtorch_cpu / libtorch_python
    # into the global symbol namespace so that the geotransformer extension
    # resolves symbols against the *correct* ABI (conda environments often
    # ship an older libc10.so in ``$CONDA_PREFIX/lib`` that conflicts).
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    for lib in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so"):
        ctypes.CDLL(os.path.join(torch_lib, lib), mode=ctypes.RTLD_GLOBAL)

    # The official geotransformer package lives inside the autoencoder
    # directory of the One-Policy-Fits-All repo.
    if _OFFICIAL_DIR is None:
        raise ImportError(
            "OPFA_OFFICIAL_DIR environment variable is not set. "
            "Set it to the autoencoder/ directory of the One-Policy-Fits-All repo, "
            "or use the pure-PyTorch fallback (already loaded)."
        )
    if _OFFICIAL_DIR not in sys.path:
        sys.path.insert(0, _OFFICIAL_DIR)

    _INITIALISED = True


# ---------------------------------------------------------------------------
# Wrapped ops — transparent CPU/GPU bridge
# ---------------------------------------------------------------------------


def grid_subsample(
    points: torch.Tensor,
    lengths: torch.Tensor,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official C++ grid subsampling (voxel barycentre).

    Args:
        points: ``(N, 3)`` stacked points (any device).
        lengths: ``(B,)`` point counts per batch element.
        voxel_size: grid cell size.

    Returns:
        ``(s_points, s_lengths)`` — both on the **original** device.
    """
    _ensure_geotransformer()
    from geotransformer.modules.ops import grid_subsample as _official_gs

    src_device = points.device
    s_points, s_lengths = _official_gs(
        points.contiguous().cpu(), lengths.contiguous().cpu(), voxel_size,
    )
    return s_points.contiguous().to(src_device), s_lengths.contiguous().to(src_device)


def radius_search(
    q_points: torch.Tensor,
    s_points: torch.Tensor,
    q_lengths: torch.Tensor,
    s_lengths: torch.Tensor,
    radius: float,
    neighbor_limit: int,
) -> torch.Tensor:
    """Official nanoflann KD-tree radius search (CPU, sorted neighbours).

    Args:
        q_points: ``(N, 3)`` query points (any device).
        s_points: ``(M, 3)`` support points (any device).
        q_lengths: ``(B,)`` query point counts per batch element.
        s_lengths: ``(B,)`` support point counts per batch element.
        radius: search radius (Euclidean).
        neighbor_limit: max neighbours per query (0 = unlimited).

    Returns:
        ``(N, max_neighbors)`` neighbour indices on the **original** device.
        Padding entries are filled with ``M`` (sentinel).
    """
    _ensure_geotransformer()
    from geotransformer.modules.ops import radius_search as _official_rs

    src_device = q_points.device
    neighbors = _official_rs(
        q_points.contiguous().cpu(), s_points.contiguous().cpu(),
        q_lengths.contiguous().cpu(), s_lengths.contiguous().cpu(),
        radius, neighbor_limit,
    )
    return neighbors.contiguous().to(src_device)
