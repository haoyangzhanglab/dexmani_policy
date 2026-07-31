"""
DexLatent autoencoder for cross-hand latent space in DexMani_Policy.

Provides a frozen VAE-style MLP that maps between XHand native joint space
(12D) and a shared cross-hand latent space (32D).  Weights are extracted
from the pretrained DexLatent checkpoint (Phase 0).

This module is the direct analogue of ``FAASHandMapper`` — same interface,
same I/O boundary pattern — but uses a learned encoder/decoder instead of
a fixed scatter/gather mapping.

Reference: XL-VLA (CVPR 2026 Highlight), arXiv:2603.10158
           DexLatent repo: https://github.com/EmptyBlueBox/DexLatent
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class DexLatentHandVAE(nn.Module):
    """Frozen XHand autoencoder mapping 12D native ↔ 32D cross-hand latent.

    Architecture (matching DexLatent ``HandAutoencoder`` exactly)::

        Encoder:  Linear(12,64)→LN→ReLU→Linear(64,128)→LN→ReLU
                  →Linear(128,64)→LN→ReLU→Linear(64,32)

        Decoder:  Linear(32,64)→LN→ReLU→Linear(64,128)→LN→ReLU
                  →Linear(128,64)→LN→ReLU→Linear(64,12)→Tanh

    All parameters are frozen by default (``requires_grad=False``).  This is
    a pure I/O conversion module — no URDF, FK, or Pinocchio dependencies.

    Parameters
    ----------
    hand_dim : int
        Native XHand joint count (=12).
    latent_dim : int
        Cross-hand latent dimension (=32).
    hidden_dims : tuple of int
        MLP hidden layer widths (=(64, 128, 64)).
    """

    # ── Class-level constants (matching DexLatent pretrained model) ──
    DEFAULT_HAND_DIM: int = 12
    DEFAULT_LATENT_DIM: int = 32
    DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (64, 128, 64)
    # State dict key prefixes as saved by Phase 0 extract_dexlatent_weights.py
    KEY_ENC_BACKBONE: str = "encoder_backbone"
    KEY_ENC_HEAD: str = "encoder_head"
    KEY_DECODER: str = "decoder"

    def __init__(
        self,
        hand_dim: int = DEFAULT_HAND_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.hand_dim = int(hand_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(hidden_dims)

        # ── Encoder backbone ──
        enc_layers: list[nn.Module] = []
        in_ch = self.hand_dim
        for width in self.hidden_dims:
            enc_layers.append(nn.Linear(in_ch, width))
            enc_layers.append(nn.LayerNorm(width))
            enc_layers.append(nn.ReLU())
            in_ch = width
        self.encoder_backbone = nn.Sequential(*enc_layers)
        self.encoder_head = nn.Linear(in_ch, self.latent_dim)

        # ── Decoder ──
        dec_layers: list[nn.Module] = []
        in_ch = self.latent_dim
        for width in self.hidden_dims:
            dec_layers.append(nn.Linear(in_ch, width))
            dec_layers.append(nn.LayerNorm(width))
            dec_layers.append(nn.ReLU())
            in_ch = width
        dec_layers.append(nn.Linear(in_ch, self.hand_dim))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    # ── Core hand-only conversions ────────────────────────────────────

    def encode(self, hand_qpos: torch.Tensor) -> torch.Tensor:
        """Encode native XHand joints → cross-hand latent vector.

        Uses the deterministic mean path (no VAE sampling).

        Args:
            hand_qpos: ``(..., hand_dim)`` normalized in ``[-1, 1]``.

        Returns:
            ``(..., latent_dim)`` latent vector (~N(0,1) distributed).
        """
        features = self.encoder_backbone(hand_qpos)
        return self.encoder_head(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode cross-hand latent → native XHand joints.

        Args:
            latent: ``(..., latent_dim)`` latent vector.

        Returns:
            ``(..., hand_dim)`` normalized joint angles in ``[-1, 1]``
            (Tanh-bounded).
        """
        return self.decoder(latent)

    def forward(
        self, hand_qpos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode→decode roundtrip (for validation / smoke testing).

        Args:
            hand_qpos: ``(..., hand_dim)`` normalized in ``[-1, 1]``.

        Returns:
            ``(reconstructed_hand, latent)`` both with same batch dims.
        """
        latent = self.encode(hand_qpos)
        recon = self.decode(latent)
        return recon, latent

    # ── Full-action convenience helpers (mirror FAASHandMapper) ───────

    def transform_action(
        self, action: torch.Tensor, arm_dim: int = 7
    ) -> torch.Tensor:
        """Convert full action: native [arm | hand] → latent [arm | latent].

        The arm portion passes through unchanged (identity).

        Args:
            action: ``(..., arm_dim + hand_dim)`` native action.
            arm_dim: number of leading arm dimensions (7 for joint, 9 for EE).

        Returns:
            ``(..., arm_dim + latent_dim)`` in latent space.
        """
        arm = action[..., :arm_dim]
        hand = action[..., arm_dim:]
        return torch.cat([arm, self.encode(hand)], dim=-1)

    def inverse_transform_action(
        self, action: torch.Tensor, arm_dim: int = 7
    ) -> torch.Tensor:
        """Convert full action: latent [arm | latent] → native [arm | hand].

        Args:
            action: ``(..., arm_dim + latent_dim)`` latent action.
            arm_dim: number of leading arm dimensions.

        Returns:
            ``(..., arm_dim + hand_dim)`` native action.
        """
        arm = action[..., :arm_dim]
        latent = action[..., arm_dim:]
        return torch.cat([arm, self.decode(latent)], dim=-1)

    def transform_joint_state(
        self, joint_state: torch.Tensor
    ) -> torch.Tensor:
        """Convert joint_state: native 19D → latent 39D.

        joint_state arm is ALWAYS 7D (proprioceptive), regardless of
        ``action_key``.

        Args:
            joint_state: ``(..., 19)`` = [arm_joints(7) | hand(12)].

        Returns:
            ``(..., 39)`` = [arm_joints(7) | latent(32)].
        """
        arm = joint_state[..., :7]
        hand = joint_state[..., 7:]
        return torch.cat([arm, self.encode(hand)], dim=-1)

    # ── Persistence ───────────────────────────────────────────────────

    @classmethod
    def load_pretrained(
        cls,
        path: str,
        hand_name: str = "xarm7_xhand_right",
    ) -> "DexLatentHandVAE":
        """Load pretrained weights from a Phase 0 checkpoint.

        The checkpoint must contain ``autoencoders[hand_name]`` with keys
        matching ``encoder_backbone.*`` / ``encoder_head.*`` / ``decoder.*``.

        Args:
            path: path to ``dexlatent_autoencoders.pt``.
            hand_name: which hand's weights to load.

        Returns:
            DexLatentHandVAE with frozen pretrained weights in eval mode.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            KeyError: if ``hand_name`` is not in the checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"DexLatent checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=True)
        all_weights = payload["autoencoders"]
        if hand_name not in all_weights:
            available = list(all_weights.keys())
            raise KeyError(
                f"Hand '{hand_name}' not in checkpoint. Available: {available}"
            )

        raw_sd = all_weights[hand_name]

        # Auto-detect hand_dim from the encoder's first Linear weight
        # Key: "encoder_backbone.0.weight" shape = (hidden[0], hand_dim)
        first_weight_key = f"{cls.KEY_ENC_BACKBONE}.0.weight"
        hand_dim = raw_sd[first_weight_key].shape[1]
        latent_dim = payload.get("latent_dim_hand", cls.DEFAULT_LATENT_DIM)
        hidden_dims = payload.get("hand_hidden_dims", cls.DEFAULT_HIDDEN_DIMS)

        model = cls(
            hand_dim=hand_dim,
            latent_dim=latent_dim,
            hidden_dims=tuple(hidden_dims),
        )

        # Map flat keys → nested module state dict
        model_sd = model.state_dict()
        mapped_sd: Dict[str, torch.Tensor] = {}

        for our_key in model_sd.keys():
            if our_key.startswith("encoder_backbone."):
                # our: "encoder_backbone.0.weight" → ckpt: same
                mapped_sd[our_key] = raw_sd[our_key]
            elif our_key.startswith("encoder_head."):
                mapped_sd[our_key] = raw_sd[our_key]
            elif our_key.startswith("decoder."):
                mapped_sd[our_key] = raw_sd[our_key]
            else:
                raise KeyError(f"Unexpected state dict key: {our_key}")

        model.load_state_dict(mapped_sd)

        # Freeze and set eval mode
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        return model

    # ── Introspection ─────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"hand_dim={self.hand_dim}, latent_dim={self.latent_dim}, "
            f"hidden_dims={self.hidden_dims}"
        )
