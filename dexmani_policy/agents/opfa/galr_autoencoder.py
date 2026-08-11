"""GaLR Autoencoder — faithful port of OPFA's ``GeoTransformer`` model class.

Architecture (identical to OPFA ``model.py:72-122``)::

    finger/link PE  ──→  KPConvFPN (4-stage)  ──→  superpoint PE
    (Linear 2→1024)      64→128→256→512→1024        (from coarse lengths)

     ──→  GeometricTransformer  ──→  L2-norm  ──→  mean-pool  ──→  latent
           3× PETrLayer(h=256,n=4)                                 (1024,)

     ──→  Linear(1024, 26) → index_select  ──→  XHand joint angles

Key simplifications vs OPFA (noted, justified in plan):

1. Loads official OPFA pretrained weights (epoch-129.pth.tar from Hugging Face)
   via ``latent_to_unified_angles`` (26-d) + XHand index selection → 12 joints.

2. Batch dimension: OPFA operates on single samples (batch_size=1 in autoencoder
   training).  Our forward() also processes one sample at a time; batching is
   done externally by the caller (e.g. ``encode_one_frame`` in preprocess.py).

Dependencies (all local):
  - ``dexmani_policy.agents.opfa.kpconv.KPConvFPN``
  - ``dexmani_policy.agents.opfa.transformer.GeometricTransformer``
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.opfa.hand_fk import HandFKGenerator
from dexmani_policy.agents.opfa.kpconv import KPConvFPN
from dexmani_policy.agents.opfa.transformer import GeometricTransformer

# =============================================================================
# Angle → latent LRU cache
# =============================================================================


class LatentCache:
    """Angle → latent LRU cache with float-rounding keys.

    Trajectory data contains many repeated joint configurations
    (holding poses, approach postures).  Caching avoids redundant FK+GaLR
    encoding for identical (within tolerance) hand poses.

    Args:
        max_size: max cache entries.
        tolerance: rounding tolerance in radians (default 0.001 ≈ 0.057°).
    """

    def __init__(self, max_size: int = 10000, tolerance: float = 0.001):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ndigits = -int(np.log10(tolerance))
        self._hits = 0
        self._misses = 0

    def _key(self, angles: np.ndarray) -> tuple[float, ...]:
        return tuple(round(float(a), self._ndigits) for a in angles)

    def get(self, angles: np.ndarray) -> torch.Tensor | None:
        key = self._key(angles)
        if key in self._cache:
            # Move to end (LRU)
            val = self._cache.pop(key)
            self._cache[key] = val
            self._hits += 1
            return val
        self._misses += 1
        return None

    def put(self, angles: np.ndarray, latent: torch.Tensor):
        key = self._key(angles)
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # LRU eviction (pop oldest)
        self._cache[key] = latent

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._cache), "hit_rate": hit_rate}


class GaLRAutoencoder(nn.Module):
    """OPFA GaLR autoencoder — faithful architecture reproduction.

    Encoder: joint angles → FK → hand PC → finger/link PE → KPConvFPN
             → superpoint PE → GeometricTransformer → L2-norm+mean-pool → latent

    Decoder: Linear(1024, 26) → index_select → 12 XHand joint angles (VAE order).

    Args:
        hand_type: always ``"xhand"``.
        backbone_input_dim: input dim for KPConvFPN (= embedding dim, 1024).
        backbone_output_dim: KPConvFPN output dim (= 256, unused but configurable).
        backbone_init_dim: initial KPConv channel count (= 64).
        kernel_size: KPConv kernel point count (= 15).
        init_radius: initial KPConv radius (= 0.025).
        init_sigma: KPConv sigma (= 0.02).
        group_norm: GroupNorm groups per stage (= 32).
        transformer_input_dim: GeometricTransformer input/output dim (= 1024).
        transformer_hidden_dim: attention hidden dim (= 256).
        transformer_num_heads: attention heads (= 4).
        transformer_blocks: PETransformerLayer list (default ``['self','self','self']``).
        unified_joint_dim: dimension of unified joint space (= 26 in OPFA; we use 12 for XHand).
    """

    # -----------------------------------------------------------------
    # Default config (matches OPFA backbone.py + geotransformer config)
    # -----------------------------------------------------------------

    DEFAULT_CONFIG = {
        "backbone_input_dim": 1024,
        "backbone_output_dim": 256,
        "backbone_init_dim": 64,
        "kernel_size": 15,
        "init_radius": 0.025,
        "init_sigma": 0.02,
        "group_norm": 32,
        "transformer_input_dim": 1024,
        "transformer_output_dim": 1024,
        "transformer_hidden_dim": 256,
        "transformer_num_heads": 4,
        "transformer_blocks": ["self", "self", "self"],
        "unified_joint_dim": 26,  # OPFA unified joint space (all hands)
    }

    # Unified → per-hand joint index mapping (from OPFA geotransformer/hands.py)
    _HAND_INDICES = {
        "xhand": [0, 2, 3, 4, 6, 7, 11, 12, 16, 17, 21, 22],
        "inspire": [0, 2, 6, 11, 16, 21],
        "inspire_force": [0, 2, 6, 11, 16, 21],
        "leap": [17, 16, 14, 18, 1, 0, 2, 3, 6, 4, 7, 8, 11, 9, 12, 13],
        "allegro": [5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 0, 1, 2, 3],
        "umi": [24],
        "robotiq_gripper": [24],
        "ability": [0, 2, 6, 11, 16, 21],
        "svh": [0, 2, 16, 21, 6, 7, 11, 12],
        "shadow": [4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 1, 0, 2, 3],
        "robotiq": [2, 3, 25, 4, 6, 7, 8, 19, 21, 22, 23],
    }

    def __init__(
        self,
        hand_type: str = "xhand",
        **kwargs,
    ):
        super().__init__()
        cfg = {**self.DEFAULT_CONFIG, **kwargs}
        self.hand_type = hand_type

        # Resolve joint indices for this hand type
        self._hand_joint_indices = self._HAND_INDICES.get(hand_type, list(range(cfg["unified_joint_dim"])))
        self._native_joint_dim = len(self._hand_joint_indices)

        # Finger/link positional encoding: Linear(2, 1024)
        self.embedding = nn.Linear(2, cfg["backbone_input_dim"], bias=True)

        # KPConvFPN: 4-stage encoder 64→128→256→512→1024
        self.backbone = KPConvFPN(
            input_dim=cfg["backbone_input_dim"],
            output_dim=cfg["backbone_output_dim"],
            init_dim=cfg["backbone_init_dim"],
            kernel_size=cfg["kernel_size"],
            init_radius=cfg["init_radius"],
            init_sigma=cfg["init_sigma"],
            group_norm=cfg["group_norm"],
        )

        # GeometricTransformer: 3× PETransformerLayer
        self.transformer = GeometricTransformer(
            input_dim=cfg["transformer_input_dim"],
            output_dim=cfg["transformer_output_dim"],
            hidden_dim=cfg["transformer_hidden_dim"],
            num_heads=cfg["transformer_num_heads"],
            blocks=cfg["transformer_blocks"],
        )

        # Decoder: latent → unified joint angles (26-d) → index_select → native joints
        # Key name matches OPFA checkpoint: "latent_to_unified_angles"
        self.latent_to_unified_angles = nn.Linear(cfg["backbone_input_dim"], cfg["unified_joint_dim"], bias=True)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, data_dict: dict, return_latent_only: bool = False) -> dict:
        """GaLR encoder-decoder forward — identical to OPFA ``model.py:72-122``.

        Args:
            data_dict: from ``HandFKGenerator.forward()``, with keys:
                - ``features``:  ``(N, 2)`` — per-point (finger_id, link_id).
                - ``points``:    ``list[(N_i, 3)]`` — 4 scales, fine→coarse.
                - ``lengths``:   ``list[(L,)]`` — per-link point counts at each scale.
                - ``neighbors``: ``list[(N_i, K)]`` — ball-query neighbours.
                - ``subsampling``: ``list[(N_{i+1}, K)]`` — grid sub-sampling indices.
                - ``hand_type``: ``str`` (ignored — this model is xhand-specific).
            return_latent_only: if True, skip joint angle decoding.

        Returns:
            dict with:
              - ``latents``: ``(1024,)`` latent (mean-pooled from L2-normalized features).
              - ``angles``:  ``(12,)`` predicted XHand joint angles (VAE order).
              - ``feats_c_norm``: ``(N_s, 1024)`` per-superpoint normalized features.
        """
        output_dict = {}

        feats = data_dict["features"]  # (N, 2) — may be numpy or tensor
        points_c = data_dict["points"][-1].detach()  # (N_s, 3) coarsest scale
        lengths = data_dict["lengths"]  # list of per-link tensors

        # Normalise: numpy → torch
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        # ---- Step 1: Embed (finger_id, link_id) → 1024-d PE ----
        input_feats = self.embedding(feats.float())  # (N, 1024)

        # ---- Step 2: KPConvFPN encoder ----
        feats_c = self.backbone(input_feats, data_dict)  # (N_s, 1024)

        # ---- Step 3: Build superpoint PE from coarse-scale per-link counts ----
        lengths_c = lengths[-1]  # (L,) per-link points at coarsest scale
        device = lengths_c.device if torch.is_tensor(lengths_c) else feats_c.device

        pos_emb_chunks = []
        for idx, link_name in enumerate(HandFKGenerator._LINK_NAMES):
            fid, lid = HandFKGenerator._FINGER_LINK_INDICES[link_name]
            num_pts = int(lengths_c[idx]) if torch.is_tensor(lengths_c) else lengths_c[idx]
            if num_pts == 0:
                continue
            pos_emb_chunks.append(
                torch.tensor([fid, lid], device=device, dtype=torch.float32).repeat(num_pts, 1)
            )

        pos_emb_c = torch.cat(pos_emb_chunks, dim=0)  # (N_s, 2)
        pos_emb_c = self.embedding(pos_emb_c)  # (N_s, 1024)

        # ---- Step 4: Geometric Transformer ----
        feats_c = self.transformer(
            points_c.unsqueeze(0),  # (1, N_s, 3)
            feats_c.unsqueeze(0),  # (1, N_s, 1024)
            pos_emb_c.unsqueeze(0),  # (1, N_s, 1024)
        )
        feats_c_norm = F.normalize(feats_c.squeeze(0), p=2, dim=1)  # (N_s, 1024)

        # ---- Step 5: Mean pool → latent ----
        latents = feats_c_norm.mean(dim=0)  # (1024,)
        output_dict["latents"] = latents
        output_dict["feats_c_norm"] = feats_c_norm

        # ---- Step 6: Decode → unified joint angles → index_select ----
        if not return_latent_only:
            pred_unified = self.latent_to_unified_angles(latents)  # (26,)
            pred_native = pred_unified[self._hand_joint_indices]  # (native_dim,)
            output_dict["angles"] = pred_native
            output_dict["angles_unified"] = pred_unified

        return output_dict

    # -----------------------------------------------------------------
    # Convenience: encode only (for offline preprocessing)
    # -----------------------------------------------------------------

    def encode(self, data_dict: dict) -> torch.Tensor:
        """Encode hand point cloud → 1024-d latent (no decoding)."""
        output = self.forward(data_dict, return_latent_only=True)
        return output["latents"]  # (1024,)

    # -----------------------------------------------------------------
    # Convenience: decode only (for policy inference)
    # -----------------------------------------------------------------

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 1024-d latent → native joint angles via unified 26-d space.

        Args:
            latents: ``(*, 1024)`` latent vector(s).

        Returns:
            ``(*, native_dim)`` joint angles (12 for XHand) in VAE order.
        """
        unified = self.latent_to_unified_angles(latents)  # (*, 26)
        return unified[..., self._hand_joint_indices]  # (*, native_dim)

    def __repr__(self) -> str:
        return (
            f"GaLRAutoencoder(hand={self.hand_type}, "
            f"kpconv_dim=64→1024, kernel_size=15, "
            f"transformer=3×PE(d=256,h=4), "
            f"decoder=1024→26→{self._native_joint_dim})"
        )


# =============================================================================
# Shared utility: load frozen GaLR encoder from checkpoint
# =============================================================================


def load_galr_encoder(ckpt_path: str, device: str = "cpu") -> GaLRAutoencoder:
    """Load frozen GaLR autoencoder from checkpoint.

    Handles multiple checkpoint formats:
      - Official OPFA (``epoch-129.pth.tar``): ``{"model": state_dict}``
      - Custom GaLR training output: ``{"model": state_dict}`` or ``{"state_dict": state_dict}``
      - Bare ``state_dict``

    Args:
        ckpt_path: path to checkpoint file.
        device: torch device to load the model onto.

    Returns:
        Frozen GaLRAutoencoder in eval mode.
    """
    galr = GaLRAutoencoder()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Handle various checkpoint formats
    if "model" in state:
        state = state["model"]
    if "state_dict" in state:
        state = state["state_dict"]

    galr.load_state_dict(state, strict=True)
    galr.to(device)
    galr.eval()
    for p in galr.parameters():
        p.requires_grad_(False)
    return galr
