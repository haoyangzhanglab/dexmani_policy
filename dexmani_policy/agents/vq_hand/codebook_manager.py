"""
CodebookManager — VQ-VAE codebook extraction, PCA re-indexing, and lookup.

Three-phase workflow:
  1. extract_from_vqvae(vqvae)   — grab codebook tensors from trained model
  2. reindex_by_pca(vqvae)       — enumerate all combinations, decode to hand
                                    poses, denormalise to raw servo space,
                                    sort by PCA principal component
  3. Runtime lookup:
       - continuous_index_to_hand_pose(idx)   — inference: diffusion output → hand
       - hand_pose_to_continuous_index(pose)  — training: GT hand → diffusion label

Key design decisions (faithful to DQ-RISE official code):
  - sorted_hand_poses stored in RAW servo space [hand_min, hand_max]
    (default [0, 65535]), exactly matching DQ-RISE eval_vqvae.py.
  - PCA fitted on raw-space hand poses (matching official).
  - L2-distance nearest-neighbour in raw space (matching official
    train_dqrise.py:161-162).
  - Normalised ↔ raw conversion at the API boundary — callers always
    pass/receive normalised [-1, 1] values.
  - PCA re-indexing uses hardcoded 0.5 group weights (matching
    DQ-RISE eval_vqvae.py:96), NOT learned layer_weights.
"""

from __future__ import annotations

import torch

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA


class CodebookManager:
    """Manages a sorted codebook for VQ-VAE hand-pose discretisation.

    Codebook entries are stored in **raw servo space** ``[hand_min, hand_max]``
    (default [0, 65535]), exactly matching DQ-RISE official.  Conversion to/from
    normalised [-1, 1] space happens at the API boundary so callers always work
    in normalised space.

    Parameters
    ----------
    hand_dim : int
        Dimensionality of the hand joint state.
    num_groups : int
        Number of residual VQ groups (default 2).
    codebook_size : int
        Number of codes per group (default 4).
    hand_min : float
        Minimum raw servo value (default 0, matching DQ-RISE).
    hand_max : float
        Maximum raw servo value (default 65535, matching DQ-RISE).
    """

    def __init__(
        self,
        hand_dim: int,
        num_groups: int = 2,
        codebook_size: int = 4,
        hand_min: float = 0.0,
        hand_max: float = 65535.0,
    ):
        self.hand_dim = hand_dim
        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.hand_min = hand_min
        self.hand_max = hand_max
        assert hand_max > hand_min, f'hand_max ({hand_max}) must be > hand_min ({hand_min})'

        # Total discrete combinations  (4² = 16 by default)
        self.total_combinations = codebook_size ** num_groups

        # ---- state ----
        # sorted_hand_poses:  (total_combinations, hand_dim)  in RAW space [hand_min, hand_max]
        self.sorted_hand_poses: torch.Tensor | None = None

        # codebooks:  (num_groups, codebook_size, latent_dim)  raw VQ embeddings
        self.codebooks: torch.Tensor | None = None

        # layer_weights:  (num_groups,)  trained softmax-normalised weights from ResidualVQ
        self.layer_weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Normalised ↔ raw conversion
    # ------------------------------------------------------------------

    def _to_raw(self, normalized: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [hand_min, hand_max]"""
        return (normalized + 1.0) / 2.0 * (self.hand_max - self.hand_min) + self.hand_min

    def _from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """[hand_min, hand_max] → [-1, 1]"""
        return (raw - self.hand_min) / (self.hand_max - self.hand_min) * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_vqvae(vqvae) -> 'CodebookManager':
        """Create a CodebookManager from a trained VqVaeHand.

        Reads codebook embeddings and trained layer_weights.  Does NOT run
        PCA — call ``reindex_by_pca`` afterwards.
        """
        mgr = CodebookManager(
            hand_dim=vqvae.hand_dim,
            num_groups=vqvae.num_groups,
            codebook_size=vqvae.codebook_size,
        )
        # (num_groups, codebook_size, latent_dim)
        mgr.codebooks = vqvae.codebooks.clone().detach().cpu()
        # Save trained softmax-normalised layer_weights for inference consistency
        mgr.layer_weights = torch.softmax(vqvae.vq_layer.layer_weights.data, dim=0).clone().detach().cpu()
        return mgr

    # ------------------------------------------------------------------
    # PCA re-indexing
    # ------------------------------------------------------------------

    def reindex_by_pca(self, vqvae) -> np.ndarray:
        """Enumerate all code combinations, decode, sort by PCA.

        Algorithm (faithful to DQ-RISE eval_vqvae.py)
        ----------------------------------------------
        1. For each of the ``codebook_size ** num_groups`` combinations:
           - weighted sum: Σ_g 0.5 · codebook[g, idx_g]  →  latent
           - vqvae.decode_from_latent(latent)             →  hand_pose (normalised [-1,1])
           - _to_raw(hand_pose)                           →  raw [hand_min, hand_max]
        2. PCA (n_components=1) on the raw-space hand pose matrix.
        3. Sort hand poses by the 1-D principal component projection.
        4. Store sorted hand poses in **raw space** in ``self.sorted_hand_poses``.

        Parameters
        ----------
        vqvae : VqVaeHand
            Trained VQ-VAE (in eval mode, on the target device).

        Returns
        -------
        sorted_hand_poses : np.ndarray  (total_combinations, hand_dim)
            Hand poses in raw space, ordered by PCA principal component.
        """
        if self.codebooks is None:
            raise RuntimeError(
                'No codebooks loaded. Call extract_from_vqvae() first.'
            )

        K = self.codebook_size
        G = self.num_groups
        device = next(vqvae.parameters()).device

        # Collect per-combination hand poses
        hand_poses: list[torch.Tensor] = []

        # Enumerate all K^G combinations
        indices_grid = torch.cartesian_prod(*[torch.arange(K)] * G)  # (K^G, G)

        for combo in indices_grid:
            latent = torch.zeros(vqvae.latent_dim, device=device)
            for g in range(G):
                weight = self.layer_weights[g].item() if self.layer_weights is not None else 0.5
                latent += weight * self.codebooks[g, combo[g]].to(device)
            latent = latent.unsqueeze(0)                     # (1, latent_dim)
            hp_norm = vqvae.decode_from_latent(latent).cpu()  # (1, hand_dim) in [-1,1]
            # Denormalise to raw servo space (matching DQ-RISE eval_vqvae.py:101)
            hp_raw = self._to_raw(hp_norm)                    # (1, hand_dim) in [0, 65535]
            hand_poses.append(hp_raw)

        all_poses = torch.cat(hand_poses, dim=0)              # (K^G, hand_dim) — raw

        # PCA on raw-space hand poses → 1-D projection → sort
        pca = PCA(n_components=1)
        proj_1d = pca.fit_transform(all_poses.numpy())        # (K^G, 1)
        sorted_idx = np.argsort(proj_1d[:, 0])

        self.sorted_hand_poses = all_poses[sorted_idx]        # (K^G, hand_dim) — raw

        return self.sorted_hand_poses.numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save sorted hand poses and metadata to a .npz file.

        Values are in raw servo space [hand_min, hand_max] — compatible with
        DQ-RISE official ``eval_vqvae.py`` output modulo the .npz wrapper.
        """
        if self.sorted_hand_poses is None:
            raise RuntimeError('No sorted hand poses to save. Run reindex_by_pca() first.')

        save_data = dict(
            format_version=2,  # raw-space codebook (compatible with DQ-RISE official)
            sorted_hand_poses=self.sorted_hand_poses.numpy(),
            hand_dim=self.hand_dim,
            num_groups=self.num_groups,
            codebook_size=self.codebook_size,
            hand_min=self.hand_min,
            hand_max=self.hand_max,
        )
        if self.layer_weights is not None:
            save_data['layer_weights'] = self.layer_weights.numpy()

        # Save per-group sorted poses (for multi-index prediction)
        if hasattr(self, '_group_sorted_poses'):
            for g, poses_g in enumerate(self._group_sorted_poses):
                save_data[f'_group_sorted_poses_g{g}'] = poses_g.numpy()

        np.savez(str(path), **save_data)

    def load(self, path: str | Path) -> None:
        """Load sorted hand poses from a .npz or .npy file.

        Supports three formats:
        - ``.npz`` with ``format_version >= 2`` (DexMani raw-space, preferred)
        - ``.npz`` with ``format_version == 1`` (DexMani legacy normalised-space —
          automatically converted on load)
        - ``.npy`` (official DQ-RISE or DexMani legacy, raw-space, no metadata)
        """
        path = Path(path)
        if path.suffix == '.npz':
            data = np.load(str(path))
            version = int(data.get('format_version', 0))
            self.sorted_hand_poses = torch.from_numpy(data['sorted_hand_poses']).float()

            # Restore hand range from metadata
            if 'hand_min' in data:
                self.hand_min = float(data['hand_min'])
            if 'hand_max' in data:
                self.hand_max = float(data['hand_max'])

            # Restore trained layer_weights (v3+)
            if 'layer_weights' in data:
                self.layer_weights = torch.from_numpy(data['layer_weights']).float()

            # Restore per-group sorted poses (multi-index prediction)
            group_poses = []
            for g in range(self.num_groups):
                key = f'_group_sorted_poses_g{g}'
                if key in data:
                    group_poses.append(torch.from_numpy(data[key]).float())
            if len(group_poses) == self.num_groups:
                self._group_sorted_poses = group_poses

            # Legacy v1: stored in normalised [-1, 1] space → convert to raw
            if version == 1:
                self.sorted_hand_poses = self._to_raw(self.sorted_hand_poses)

            # No version: official DQ-RISE codebook — values already in raw space
            if version == 0 and 'hand_min' not in data:
                # Assume official DQ-RISE defaults [0, 65535]
                pass

            # Validate file metadata against construction parameters before overwriting.
            # Mismatched metadata means the codebook was trained for a different robot
            # or VQ-VAE configuration, and using it would produce silent wrong results.
            loaded_dim = self.sorted_hand_poses.shape[-1]
            mismatches = []
            if 'hand_dim' in data and int(data['hand_dim']) != self.hand_dim:
                mismatches.append(
                    f'hand_dim: expected {self.hand_dim}, '
                    f'got {int(data["hand_dim"])}'
                )
            if 'num_groups' in data and int(data['num_groups']) != self.num_groups:
                mismatches.append(
                    f'num_groups: expected {self.num_groups}, '
                    f'got {int(data["num_groups"])}'
                )
            if 'codebook_size' in data and int(data['codebook_size']) != self.codebook_size:
                mismatches.append(
                    f'codebook_size: expected {self.codebook_size}, '
                    f'got {int(data["codebook_size"])}'
                )
            # Also check actual tensor shape against declared metadata
            if loaded_dim != self.hand_dim:
                mismatches.append(
                    f'sorted_hand_poses.shape[-1]={loaded_dim} '
                    f'!= hand_dim={self.hand_dim}'
                )
            if mismatches:
                raise ValueError(
                    f'Codebook at {path} has mismatched parameters:\n'
                    + '\n'.join(f'  - {m}' for m in mismatches)
                    + f'\nThe codebook was trained for a different hand '
                    f'or VQ-VAE configuration than the current DQRISEAgent expects.'
                )

            # Restore metadata (now safe — validated above)
            if 'hand_dim' in data:
                self.hand_dim = int(data['hand_dim'])
            if 'num_groups' in data:
                self.num_groups = int(data['num_groups'])
            if 'codebook_size' in data:
                self.codebook_size = int(data['codebook_size'])
            self.total_combinations = self.codebook_size ** self.num_groups
        elif path.suffix == '.npy':
            arr = np.load(str(path))
            self.sorted_hand_poses = torch.from_numpy(arr).float()
            self.total_combinations = len(arr)
        else:
            # Try .npy first
            arr = np.load(str(path))
            self.sorted_hand_poses = torch.from_numpy(arr).float()
            self.total_combinations = len(arr)

    def get_num_codes(self) -> int:
        """Return total discrete code combinations (e.g. 16)."""
        if self.sorted_hand_poses is not None:
            return len(self.sorted_hand_poses)
        return self.total_combinations

    # ------------------------------------------------------------------
    # Per-group codebook lookup (preserves ResidualVQ factor structure)
    # ------------------------------------------------------------------

    def build_per_group_codebooks(self, vqvae) -> None:
        """Build per-group sorted hand poses from a trained VQ-VAE.

        For each group g, enumerate its ``codebook_size`` codes, decode to
        hand poses via the VQ-VAE decoder (using trained layer_weights),
        and sort by PCA principal component.  Stores results in
        ``_group_sorted_poses`` list of length ``num_groups``, each of
        shape ``(codebook_size, hand_dim)`` in raw space.
        """
        self._group_sorted_poses: list[torch.Tensor] = []
        device = next(vqvae.parameters()).device
        for g in range(self.num_groups):
            poses_g = []
            for c in range(self.codebook_size):
                latent = torch.zeros(vqvae.latent_dim, device=device)
                w = self.layer_weights[g].item() if self.layer_weights is not None else 0.5
                latent += w * self.codebooks[g, c].to(device)
                hp_norm = vqvae.decode_from_latent(latent.unsqueeze(0)).cpu()
                hp_raw = self._to_raw(hp_norm)
                poses_g.append(hp_raw)
            all_g = torch.cat(poses_g, dim=0)                     # (K, hand_dim)
            pca = PCA(n_components=1)
            proj = pca.fit_transform(all_g.numpy())
            sorted_idx = np.argsort(proj[:, 0])
            self._group_sorted_poses.append(all_g[sorted_idx])     # (K, hand_dim)

    def group_continuous_index_to_hand_pose(
        self, continuous_index: torch.Tensor, group: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map continuous VQ index → hand pose for a single group.

        Parameters
        ----------
        continuous_index : Tensor  (*,)  float in [-1, 1]
        group : int  group index (0..num_groups-1)

        Returns
        -------
        hand_pose : Tensor      (*, hand_dim)  normalised [-1, 1]
        discrete_index : Tensor  (*,)           long in [0, K-1]
        """
        if not hasattr(self, '_group_sorted_poses'):
            raise RuntimeError('Per-group codebooks not built. Call build_per_group_codebooks() first.')
        poses_g = self._group_sorted_poses[group]               # (K, hand_dim)
        K = len(poses_g)
        device = continuous_index.device
        scaled = (continuous_index + 1.0) / 2.0 * (K - 1)
        discrete_idx = scaled.round().long().clamp(0, K - 1)
        hand_pose_raw = poses_g.to(device)[discrete_idx]        # (*, hand_dim) raw
        hand_pose = self._from_raw(hand_pose_raw)                # (*, hand_dim) [-1,1]
        return hand_pose, discrete_idx

    def hand_pose_to_group_continuous_index(
        self, hand_pose: torch.Tensor, group: int,
    ) -> torch.Tensor:
        """Map hand pose → continuous VQ index for a single group.

        Parameters
        ----------
        hand_pose : Tensor  (*, hand_dim)  normalised [-1, 1]
        group : int  group index (0..num_groups-1)

        Returns
        -------
        continuous_index : Tensor  (*, 1)  float in [-1, 1]
        """
        if not hasattr(self, '_group_sorted_poses'):
            raise RuntimeError('Per-group codebooks not built. Call build_per_group_codebooks() first.')
        poses_g = self._group_sorted_poses[group]               # (K, hand_dim)
        K = len(poses_g)
        device = hand_pose.device
        lead_shape = hand_pose.shape[:-1]
        hp_flat = hand_pose.reshape(-1, self.hand_dim).float()
        hp_raw = self._to_raw(hp_flat)                           # (N, hand_dim) raw
        sorted_raw = poses_g.to(device)                           # (K, hand_dim)
        diff = hp_raw.unsqueeze(1) - sorted_raw.unsqueeze(0)     # (N, K, hand_dim)
        dist2 = (diff ** 2).sum(dim=-1)                           # (N, K)
        discrete_idx = dist2.argmin(dim=-1).float()               # (N,)
        continuous = discrete_idx / max(K - 1, 1) * 2.0 - 1.0    # (N,)
        return continuous.reshape(*lead_shape, 1)                  # (*, 1)

    # ------------------------------------------------------------------
    # Runtime lookup (legacy — PCA-sorted combined codebook)
    # ------------------------------------------------------------------

    def continuous_index_to_hand_pose(
        self, continuous_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map continuous VQ index → hand pose  (inference path).

        Performs the "continuous relaxation" step:
          continuous_index ∈ [-1, 1]
            → [0, num_codes-1]  via linear rescale
            → round → clamp   (nearest discrete code)
            → lookup sorted_hand_poses (raw) → _from_raw → normalised [-1, 1]

        Parameters
        ----------
        continuous_index : Tensor  (*,)  float in [-1, 1]

        Returns
        -------
        hand_pose : Tensor      (*, hand_dim)  in normalised space [-1, 1]
        discrete_index : Tensor  (*,)           long in [0, num_codes-1]
        """
        if self.sorted_hand_poses is None:
            raise RuntimeError('No sorted hand poses loaded. Call load() first.')

        num_codes = self.get_num_codes()
        device = continuous_index.device

        # [-1, 1] → [0, num_codes-1]
        scaled = (continuous_index + 1.0) / 2.0 * (num_codes - 1)
        discrete_idx = scaled.round().long().clamp(0, num_codes - 1)

        # Lookup raw codebook, then normalise to [-1, 1] for the caller
        poses_raw = self.sorted_hand_poses.to(device)    # (C, hand_dim) — raw
        hand_pose_raw = poses_raw[discrete_idx]           # (*, hand_dim) — raw
        hand_pose = self._from_raw(hand_pose_raw)         # (*, hand_dim) — [-1, 1]
        return hand_pose, discrete_idx

    def hand_pose_to_continuous_index(
        self, hand_pose: torch.Tensor
    ) -> torch.Tensor:
        """Map hand pose → continuous VQ index  (training label generation).

        Algorithm (faithful to DQ-RISE train_dqrise.py:160-162):
          1. Denormalise hand_pose from [-1, 1] → raw [hand_min, hand_max]
          2. L2 nearest-neighbour against raw-space sorted_hand_poses
          3. Normalise discrete index to [-1, 1]

        This matches DQ-RISE exactly:
          distances = cdist(handpose, code_book_actions)
          indices = argmin(distances) / (TCP_DIM+HAND_DIM) * 2 - 1

        (Our denominator ``num_codes - 1`` equals their ``TCP_DIM+HAND_DIM=15``
        for the default 16-code configuration.)

        Parameters
        ----------
        hand_pose : Tensor  (*, hand_dim)  in normalised space [-1, 1]

        Returns
        -------
        continuous_index : Tensor  (*, 1)  float in [-1, 1]
        """
        if self.sorted_hand_poses is None:
            raise RuntimeError('No sorted hand poses loaded. Call load() first.')

        num_codes = self.get_num_codes()
        device = hand_pose.device
        sorted_raw = self.sorted_hand_poses.to(device)    # (C, hand_dim) — raw

        # Flatten leading dims → (N, hand_dim)
        lead_shape = hand_pose.shape[:-1]
        hp_flat = hand_pose.reshape(-1, self.hand_dim).float()  # (N, hand_dim) — force float32

        # Denormalise to raw space for distance computation (matching official)
        hp_raw = self._to_raw(hp_flat)                     # (N, hand_dim) — raw

        # L2 distances in raw space:  d[n, c] = ||hp_raw[n] - sorted_raw[c]||
        diff = hp_raw.unsqueeze(1) - sorted_raw.unsqueeze(0)   # (N, C, hand_dim)
        dist2 = (diff ** 2).sum(dim=-1)                          # (N, C)
        discrete_idx = dist2.argmin(dim=-1).float()              # (N,)

        # Normalise: [0, num_codes-1] → [-1, 1]
        continuous = discrete_idx / max(num_codes - 1, 1) * 2.0 - 1.0  # (N,)

        return continuous.reshape(*lead_shape, 1)                      # (*, 1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self.sorted_hand_poses is not None
        return (
            f'CodebookManager(hand_dim={self.hand_dim}, '
            f'groups={self.num_groups}, codebook_size={self.codebook_size}, '
            f'total={self.total_combinations}, loaded={loaded}, '
            f'hand_range=[{self.hand_min}, {self.hand_max}])'
        )
