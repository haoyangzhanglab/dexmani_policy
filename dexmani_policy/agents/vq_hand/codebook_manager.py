"""Codebook management for DQ-RISE hand-state quantisation.

This module owns the PCA-ordered hand prototypes used by DQ-RISE.  The
manager is an ``nn.Module`` so the runtime codebook is stored inside the
policy ``state_dict``.  A policy checkpoint is therefore self-contained and
cannot silently change behaviour when an external ``.npz`` file is replaced.

Runtime convention
------------------
* Callers pass and receive hand poses in the policy-normalised space.
* ``sorted_hand_poses`` is stored in the historical affine ``raw`` space
  ``[hand_min, hand_max]`` for compatibility with existing DQ-RISE artifacts.
  For XHand this is only an affine representation of normalised coordinates;
  it is not a physical servo-unit definition.
* Continuous code indices are represented in ``[-1, 1]`` and decoded with
  nearest-integer (half-up) rounding followed by clamping.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn


class CodebookManager(nn.Module):
    """Manage extraction, persistence, ordering, and lookup of hand prototypes."""

    FORMAT_VERSION = 3

    def __init__(
        self,
        hand_dim: int,
        num_groups: int = 2,
        codebook_size: int = 4,
        hand_min: float = 0.0,
        hand_max: float = 65535.0,
    ) -> None:
        super().__init__()
        if hand_dim <= 0:
            raise ValueError(f"hand_dim must be positive, got {hand_dim}")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if hand_max <= hand_min:
            raise ValueError(f"hand_max ({hand_max}) must be larger than hand_min ({hand_min})")

        self.hand_dim = int(hand_dim)
        self.num_groups = int(num_groups)
        self.codebook_size = int(codebook_size)
        self.hand_min = float(hand_min)
        self.hand_max = float(hand_max)
        self.total_combinations = self.codebook_size**self.num_groups

        # Persistent runtime state.  These buffers are included in the policy
        # checkpoint and moved automatically by ``model.to(device)``.
        self.register_buffer(
            "sorted_hand_poses",
            torch.empty((0, self.hand_dim), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("pca_permutation", torch.empty((0,), dtype=torch.long), persistent=True)
        self.register_buffer("layer_weights", torch.empty((0,), dtype=torch.float32), persistent=True)
        self.register_buffer(
            "hand_normalizer_scale",
            torch.empty((0,), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "hand_normalizer_offset",
            torch.empty((0,), dtype=torch.float32),
            persistent=True,
        )

        # Extraction-only state; the latent codebooks are unnecessary at runtime.
        self.register_buffer("codebooks", torch.empty((0,), dtype=torch.float32), persistent=False)

        self.artifact_metadata: dict[str, object] = {}
        self.last_export_diagnostics: dict[str, float] = {}
        self._group_sorted_poses: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # State-dict compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Buffers are dynamically sized.  Resize them to incoming checkpoint
        # tensors before delegating to nn.Module's copy implementation.
        persistent_names = (
            "sorted_hand_poses",
            "pca_permutation",
            "layer_weights",
            "hand_normalizer_scale",
            "hand_normalizer_offset",
        )
        for name in persistent_names:
            key = prefix + name
            if key not in state_dict:
                continue
            incoming = state_dict[key]
            current = getattr(self, name)
            if tuple(current.shape) != tuple(incoming.shape):
                setattr(
                    self,
                    name,
                    torch.empty(
                        incoming.shape,
                        dtype=incoming.dtype,
                        device=current.device,
                    ),
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Keep plain-Python metadata in sync with the loaded buffers (mirrors
        # what the .npz load() path does at lines 455-456).  Without this,
        # self.num_groups / self.codebook_size / self.hand_min / self.hand_max
        # would stay at their constructor defaults even after a checkpoint
        # written by a different configuration is loaded.
        if self.is_loaded:
            n_poses = self.sorted_hand_poses.shape[0]
            self.hand_dim = self.sorted_hand_poses.shape[1]
            # For the DQ-RISE 2-group coding, the only supported decomposition
            # is codebook_size ** num_groups = n_poses with num_groups = 2.
            self.num_groups = 2
            self.codebook_size = int(round(n_poses ** (1.0 / self.num_groups)))
            self.total_combinations = n_poses

        # Backward compatibility: old policy checkpoints did not include the
        # codebook.  Loading is still safe when an external codebook was already
        # supplied during agent construction.
        if self.is_loaded:
            for name in persistent_names:
                key = prefix + name
                if key not in state_dict and key in missing_keys:
                    missing_keys.remove(key)

    # ------------------------------------------------------------------
    # Normalised <-> affine raw conversion
    # ------------------------------------------------------------------

    def _to_raw(self, normalized: torch.Tensor) -> torch.Tensor:
        return (normalized + 1.0) * 0.5 * (self.hand_max - self.hand_min) + self.hand_min

    def _from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.hand_min) / (self.hand_max - self.hand_min) * 2.0 - 1.0

    @property
    def is_loaded(self) -> bool:
        return self.sorted_hand_poses.numel() > 0

    @property
    def has_hand_normalizer(self) -> bool:
        return (
            self.hand_normalizer_scale.numel() == self.hand_dim
            and self.hand_normalizer_offset.numel() == self.hand_dim
        )

    def set_hand_normalizer(
        self, scale: torch.Tensor | np.ndarray, offset: torch.Tensor | np.ndarray
    ) -> None:
        scale_t = torch.as_tensor(scale, dtype=torch.float32).flatten().cpu()
        offset_t = torch.as_tensor(offset, dtype=torch.float32).flatten().cpu()
        if scale_t.numel() != self.hand_dim or offset_t.numel() != self.hand_dim:
            raise ValueError(
                "Hand normalizer shape mismatch: "
                f"expected {self.hand_dim}, got scale={scale_t.numel()}, "
                f"offset={offset_t.numel()}"
            )
        self.hand_normalizer_scale = scale_t
        self.hand_normalizer_offset = offset_t

    # ------------------------------------------------------------------
    # Factory and extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_vqvae(vqvae) -> "CodebookManager":
        mgr = CodebookManager(
            hand_dim=vqvae.hand_dim,
            num_groups=vqvae.num_groups,
            codebook_size=vqvae.codebook_size,
        )
        mgr.codebooks = vqvae.codebooks.detach().float().cpu().clone()
        mgr.layer_weights = torch.softmax(vqvae.vq_layer.layer_weights.detach(), dim=0).float().cpu().clone()
        mgr.artifact_metadata.update(
            {
                "hand_dim": int(vqvae.hand_dim),
                "latent_dim": int(vqvae.latent_dim),
                "hidden_dim": int(vqvae.hidden_dim),
                "num_layers": int(vqvae.num_layers),
                "num_groups": int(vqvae.num_groups),
                "codebook_size": int(vqvae.codebook_size),
                "act_scale": float(vqvae.act_scale.detach().cpu()),
            }
        )
        return mgr

    @staticmethod
    def _decode_valid_pose(vqvae, latent: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Decode and clamp one latent while returning range diagnostics."""
        with torch.no_grad():
            hp = vqvae.decode_from_latent(latent).detach().float().cpu()
        outside = hp.abs() > 1.0
        diag = {
            "decoder_min": float(hp.min()),
            "decoder_max": float(hp.max()),
            "outside_count": int(outside.sum()),
            "element_count": int(hp.numel()),
            "max_violation": float((hp.abs() - 1.0).clamp_min(0).max()),
        }
        return hp.clamp(-1.0, 1.0), diag

    def reindex_by_pca(self, vqvae) -> np.ndarray:
        """Enumerate all residual-code combinations and sort decoded poses by PC1.

        This follows the repository implementation: PCA is fitted on the decoded
        prototype matrix, not on all demonstration frames.  Learned residual
        layer weights are used, preserving train/export consistency.
        """
        if self.codebooks.numel() == 0:
            raise RuntimeError("No latent codebooks loaded; call extract_from_vqvae().")
        if self.layer_weights.numel() != self.num_groups:
            raise RuntimeError(f"Expected {self.num_groups} layer weights, got {self.layer_weights.numel()}")

        device = next(vqvae.parameters()).device
        was_training = vqvae.training
        vqvae.eval()

        poses: list[torch.Tensor] = []
        combos = list(itertools.product(range(self.codebook_size), repeat=self.num_groups))
        diag_acc = {
            "decoder_min": float("inf"),
            "decoder_max": float("-inf"),
            "outside_count": 0,
            "element_count": 0,
            "max_violation": 0.0,
        }

        try:
            for combo in combos:
                latent = torch.zeros(vqvae.latent_dim, device=device)
                for group, code_idx in enumerate(combo):
                    latent = latent + (
                        self.layer_weights[group].to(device) * self.codebooks[group, code_idx].to(device)
                    )
                hp_norm, diag = self._decode_valid_pose(vqvae, latent.unsqueeze(0))
                hp_raw = self._to_raw(hp_norm)
                poses.append(hp_raw)

                diag_acc["decoder_min"] = min(diag_acc["decoder_min"], diag["decoder_min"])
                diag_acc["decoder_max"] = max(diag_acc["decoder_max"], diag["decoder_max"])
                diag_acc["outside_count"] += diag["outside_count"]
                diag_acc["element_count"] += diag["element_count"]
                diag_acc["max_violation"] = max(diag_acc["max_violation"], diag["max_violation"])
        finally:
            vqvae.train(was_training)

        all_poses = torch.cat(poses, dim=0).float()
        if all_poses.shape != (self.total_combinations, self.hand_dim):
            raise RuntimeError(
                "Unexpected decoded prototype shape: "
                f"expected {(self.total_combinations, self.hand_dim)}, "
                f"got {tuple(all_poses.shape)}"
            )

        pca = PCA(n_components=1)
        projection = pca.fit_transform(all_poses.numpy())[:, 0]
        permutation = np.argsort(projection)

        self.pca_permutation = torch.as_tensor(permutation, dtype=torch.long)
        self.sorted_hand_poses = all_poses[self.pca_permutation]

        outside_fraction = diag_acc["outside_count"] / max(diag_acc["element_count"], 1)
        self.last_export_diagnostics = {
            "decoder_min_before_clamp": float(diag_acc["decoder_min"]),
            "decoder_max_before_clamp": float(diag_acc["decoder_max"]),
            "outside_fraction": float(outside_fraction),
            "max_violation": float(diag_acc["max_violation"]),
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        }
        self.artifact_metadata["export_diagnostics"] = dict(self.last_export_diagnostics)

        if pca.explained_variance_ratio_[0] < 0.3:
            warnings.warn(
                f"PCA explained_variance_ratio is low "
                f"({pca.explained_variance_ratio_[0]:.4f}). "
                "Hand pose prototypes may lack a clear 1-D ordering. "
                "Consider improving VQ-VAE training quality — low PC1 ratio "
                "often correlates with low vq_idx_used and downstream "
                "success-rate collapse.",
                RuntimeWarning,
            )

        if outside_fraction > 0:
            warnings.warn(
                "VQ decoder produced values outside [-1, 1]; prototypes were "
                f"clamped. outside_fraction={outside_fraction:.6f}, "
                f"max_violation={diag_acc['max_violation']:.6f}",
                RuntimeWarning,
            )

        return self.sorted_hand_poses.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if not self.is_loaded:
            raise RuntimeError("No sorted hand poses to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "format_version": np.asarray(self.FORMAT_VERSION, dtype=np.int64),
            "pose_space": np.asarray("affine_raw"),
            "sorted_hand_poses": self.sorted_hand_poses.detach().cpu().numpy(),
            "pca_permutation": self.pca_permutation.detach().cpu().numpy(),
            "hand_dim": np.asarray(self.hand_dim, dtype=np.int64),
            "num_groups": np.asarray(self.num_groups, dtype=np.int64),
            "codebook_size": np.asarray(self.codebook_size, dtype=np.int64),
            "hand_min": np.asarray(self.hand_min, dtype=np.float64),
            "hand_max": np.asarray(self.hand_max, dtype=np.float64),
            "metadata_json": np.asarray(json.dumps(self.artifact_metadata, sort_keys=True)),
        }
        if self.layer_weights.numel() > 0:
            payload["layer_weights"] = self.layer_weights.detach().cpu().numpy()
        if self.has_hand_normalizer:
            payload["hand_normalizer_scale"] = self.hand_normalizer_scale.detach().cpu().numpy()
            payload["hand_normalizer_offset"] = self.hand_normalizer_offset.detach().cpu().numpy()
        if self._group_sorted_poses is not None:
            for group, poses in enumerate(self._group_sorted_poses):
                payload[f"_group_sorted_poses_g{group}"] = poses.detach().cpu().numpy()

        np.savez(str(path), **payload)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Codebook not found: {path}")

        if path.suffix == ".npz":
            with np.load(str(path), allow_pickle=False) as data:
                keys = set(data.files)
                version = int(data["format_version"]) if "format_version" in keys else 0
                loaded_hand_min = float(data["hand_min"]) if "hand_min" in keys else 0.0
                loaded_hand_max = float(data["hand_max"]) if "hand_max" in keys else 65535.0
                poses = torch.from_numpy(np.asarray(data["sorted_hand_poses"], dtype=np.float32))

                # Legacy v1 stored normalised poses.
                if version == 1:
                    old_min, old_max = self.hand_min, self.hand_max
                    self.hand_min, self.hand_max = loaded_hand_min, loaded_hand_max
                    poses = self._to_raw(poses)
                    self.hand_min, self.hand_max = old_min, old_max

                declared_hand_dim = int(data["hand_dim"]) if "hand_dim" in keys else poses.shape[-1]
                declared_groups = int(data["num_groups"]) if "num_groups" in keys else self.num_groups
                declared_size = int(data["codebook_size"]) if "codebook_size" in keys else self.codebook_size

                mismatches: list[str] = []
                if declared_hand_dim != self.hand_dim:
                    mismatches.append(f"hand_dim expected {self.hand_dim}, got {declared_hand_dim}")
                if declared_groups != self.num_groups:
                    mismatches.append(f"num_groups expected {self.num_groups}, got {declared_groups}")
                if declared_size != self.codebook_size:
                    mismatches.append(f"codebook_size expected {self.codebook_size}, got {declared_size}")
                if poses.ndim != 2 or poses.shape[-1] != self.hand_dim:
                    mismatches.append(
                        f"sorted_hand_poses shape expected (*, {self.hand_dim}), got {tuple(poses.shape)}"
                    )
                expected_codes = self.codebook_size**self.num_groups
                if poses.shape[0] != expected_codes:
                    mismatches.append(f"number of poses expected {expected_codes}, got {poses.shape[0]}")
                if mismatches:
                    raise ValueError(f"Codebook {path} is incompatible:\n  - " + "\n  - ".join(mismatches))

                self.hand_min = loaded_hand_min
                self.hand_max = loaded_hand_max
                self.sorted_hand_poses = poses
                self.pca_permutation = (
                    torch.from_numpy(data["pca_permutation"]).long()
                    if "pca_permutation" in keys
                    else torch.arange(expected_codes, dtype=torch.long)
                )
                self.layer_weights = (
                    torch.from_numpy(data["layer_weights"]).float()
                    if "layer_weights" in keys
                    else torch.empty((0,), dtype=torch.float32)
                )
                if "hand_normalizer_scale" in keys:
                    self.set_hand_normalizer(
                        data["hand_normalizer_scale"],
                        data["hand_normalizer_offset"],
                    )
                else:
                    self.hand_normalizer_scale = torch.empty((0,), dtype=torch.float32)
                    self.hand_normalizer_offset = torch.empty((0,), dtype=torch.float32)

                if "metadata_json" in keys:
                    try:
                        self.artifact_metadata = json.loads(str(data["metadata_json"].item()))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        self.artifact_metadata = {}

                group_poses: list[torch.Tensor] = []
                for group in range(self.num_groups):
                    key = f"_group_sorted_poses_g{group}"
                    if key in keys:
                        group_poses.append(torch.from_numpy(data[key]).float())
                self._group_sorted_poses = group_poses if len(group_poses) == self.num_groups else None
            return

        # Legacy official .npy format.
        poses_np = np.load(str(path), allow_pickle=False)
        poses = torch.from_numpy(np.asarray(poses_np, dtype=np.float32))
        expected_codes = self.codebook_size**self.num_groups
        if poses.shape != (expected_codes, self.hand_dim):
            raise ValueError(
                f"Legacy codebook shape expected {(expected_codes, self.hand_dim)}, got {tuple(poses.shape)}"
            )
        self.sorted_hand_poses = poses
        self.pca_permutation = torch.arange(expected_codes, dtype=torch.long)
        self.artifact_metadata = {"source_format": "legacy_npy"}

    # ------------------------------------------------------------------
    # Per-group experimental codebooks
    # ------------------------------------------------------------------

    def build_per_group_codebooks(self, vqvae) -> None:
        if self.codebooks.numel() == 0:
            raise RuntimeError("No latent codebooks loaded.")
        device = next(vqvae.parameters()).device
        was_training = vqvae.training
        vqvae.eval()
        group_results: list[torch.Tensor] = []
        try:
            for group in range(self.num_groups):
                poses: list[torch.Tensor] = []
                for code_idx in range(self.codebook_size):
                    latent = torch.zeros(vqvae.latent_dim, device=device)
                    latent = latent + (
                        self.layer_weights[group].to(device) * self.codebooks[group, code_idx].to(device)
                    )
                    hp_norm, _ = self._decode_valid_pose(vqvae, latent.unsqueeze(0))
                    poses.append(self._to_raw(hp_norm))
                all_group = torch.cat(poses, dim=0)
                projection = PCA(n_components=1).fit_transform(all_group.numpy())[:, 0]
                order = np.argsort(projection)
                group_results.append(all_group[order])
        finally:
            vqvae.train(was_training)
        self._group_sorted_poses = group_results

    # ------------------------------------------------------------------
    # Runtime mappings
    # ------------------------------------------------------------------

    @property
    def num_codes(self) -> int:
        return int(self.sorted_hand_poses.shape[0]) if self.is_loaded else self.total_combinations

    @staticmethod
    def _nearest_integer_half_up(value: torch.Tensor) -> torch.Tensor:
        return torch.floor(value + 0.5)

    def continuous_index_to_hand_pose(
        self, continuous_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_loaded:
            raise RuntimeError("No runtime codebook loaded.")
        num_codes = self.num_codes
        clipped = continuous_index.clamp(-1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * (num_codes - 1)
        discrete_idx = self._nearest_integer_half_up(scaled).long().clamp(0, num_codes - 1)
        poses_raw = self.sorted_hand_poses.to(continuous_index.device)
        pose_norm = self._from_raw(poses_raw[discrete_idx])
        return pose_norm, discrete_idx

    def hand_pose_to_continuous_index(self, hand_pose: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("No runtime codebook loaded.")
        if hand_pose.shape[-1] != self.hand_dim:
            raise ValueError(f"Expected hand pose last dimension {self.hand_dim}, got {hand_pose.shape[-1]}")
        lead_shape = hand_pose.shape[:-1]
        flat = hand_pose.reshape(-1, self.hand_dim).float()
        flat_raw = self._to_raw(flat)
        prototypes = self.sorted_hand_poses.to(flat.device)
        dist2 = ((flat_raw[:, None, :] - prototypes[None, :, :]) ** 2).sum(-1)
        discrete = dist2.argmin(dim=-1).float()
        num_codes = self.num_codes
        continuous = discrete / max(num_codes - 1, 1) * 2.0 - 1.0
        return continuous.reshape(*lead_shape, 1)

    def group_continuous_index_to_hand_pose(
        self, continuous_index: torch.Tensor, group: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._group_sorted_poses is None:
            raise RuntimeError("Per-group codebooks have not been built or loaded.")
        if not 0 <= group < self.num_groups:
            raise IndexError(group)
        poses = self._group_sorted_poses[group].to(continuous_index.device)
        count = poses.shape[0]
        scaled = (continuous_index.clamp(-1.0, 1.0) + 1.0) * 0.5 * (count - 1)
        idx = self._nearest_integer_half_up(scaled).long().clamp(0, count - 1)
        return self._from_raw(poses[idx]), idx

    def hand_pose_to_group_continuous_index(self, hand_pose: torch.Tensor, group: int) -> torch.Tensor:
        if self._group_sorted_poses is None:
            raise RuntimeError("Per-group codebooks have not been built or loaded.")
        if not 0 <= group < self.num_groups:
            raise IndexError(group)
        lead_shape = hand_pose.shape[:-1]
        flat_raw = self._to_raw(hand_pose.reshape(-1, self.hand_dim).float())
        poses = self._group_sorted_poses[group].to(hand_pose.device)
        dist2 = ((flat_raw[:, None, :] - poses[None, :, :]) ** 2).sum(-1)
        idx = dist2.argmin(-1).float()
        continuous = idx / max(poses.shape[0] - 1, 1) * 2.0 - 1.0
        return continuous.reshape(*lead_shape, 1)

    def __repr__(self) -> str:
        return (
            f"CodebookManager(hand_dim={self.hand_dim}, groups={self.num_groups}, "
            f"codebook_size={self.codebook_size}, loaded={self.is_loaded}, "
            f"num_codes={self.num_codes}, "
            f"has_normalizer={self.has_hand_normalizer})"
        )
