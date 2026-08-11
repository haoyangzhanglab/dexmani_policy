"""Minimal kernel point generation — used ONLY as a fallback when the pre-computed
``.ply`` file is missing.  Requires ``numpy`` and ``open3d`` (optional install).

Imports ``matplotlib`` lazily inside the function — it is only needed for verbose
mode (never exercised in normal operation).
"""

from __future__ import annotations

import numpy as np


def spherical_Lloyd(
    radius: float,
    num_cells: int,
    dimension: int = 3,
    fixed: str = "center",
    approximation: str = "monte-carlo",
    approx_n: int = 5000,
    max_iter: int = 500,
    momentum: float = 0.9,
    verbose: int = 0,
) -> np.ndarray:
    """Lloyd relaxation on a sphere — generates uniformly distributed kernel points.

    This is a direct port of the OPFA ``kernel_points.py`` function, simplified
    by removing verbose matplotlib plotting.  Only the algorithmic core is kept.

    Returns:
        ``(num_cells, dimension)`` float64 array scaled by *radius*.
    """
    radius0 = 1.0

    # Kernel initialisation — uniform random on a shell
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[
            np.logical_and(d2 < radius0**2, (0.9 * radius0) ** 2 < d2), :
        ]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    if fixed == "center":
        kernel_points[0, :] *= 0
    if fixed == "verticals":
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    # Monte-Carlo approximation points
    X = np.zeros((0, dimension))
    max_moves = np.zeros((0,))

    for iteration in range(max_iter):
        if approximation == "monte-carlo":
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]

        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)

        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = cell_inds == c
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                centers.append(kernel_points[c])

        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        if fixed == "center":
            kernel_points[0, :] *= 0
        if fixed == "verticals":
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

    return kernel_points * radius
