"""
FAAS (Function-Actuator-Aligned Space) mapper for DexMani_Policy.

Maps between native XHand joint space (12D) and UniDex FAAS unified hand
joint space (32D). The mapping is derived from UniDex's hand_utils.json.

Reference: UniDex (CVPR 2026), arXiv:2603.22264
           docs/UniDex-知识体系.md §5 FAAS
           docs/FAAS-迁移-最佳方案.md (v5)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FAASHandMapper(nn.Module):
    """XHand native joint space <-> FAAS unified hand space.

    FAAS is a 32-dim functional alignment space where each index represents
    a specific joint role (e.g. index [1] = thumb CMC flexion across ALL
    hand types).  XHand uses only 12 of 32 indices; the remaining 20 are
    zero-padded and the model learns to ignore them.

    This is an nn.Module so it can be included in checkpoint state, but
    all its parameters are buffers (non-trainable).

    Mapping formula (matching UniDex ``_apply_scale_shift``)::

        faas_value = native_value * scale + offset        (forward)
        native_value = (faas_value - offset) / scale      (inverse)

    For XHand all offsets are 0.0 and all scales are ±1, so the
    general inverse formula is numerically identical to the
    simplified ``native = faas * scale``, but the general form is
    used for robustness to future non-±1 scale values.
    """

    # ── FAAS space constants ──
    MAPPED_JOINT_DIM: int = 32   # single-hand FAAS dimension
    JOINT_DIM_IN_USE: int = 27   # actively used slots (across all 8 hands)
    NATIVE_HAND_DIM: int = 12    # XHand native hand joint count

    # ── XHand -> FAAS mapping tables (from UniDex hand_utils.json) ──

    # native XHand index -> FAAS index (ordered per DexMani joint convention)
    _NATIVE_TO_FAAS_INDICES: tuple = (
        1,   # thumb_bend_joint       -> FAAS Thumb CMC Flexion
        2,   # thumb_rota_joint1      -> FAAS Thumb MCP Pitch
        3,   # thumb_rota_joint2      -> FAAS Thumb Intermediate
        6,   # index_bend_joint       -> FAAS Index Spread
        7,   # index_joint1           -> FAAS Index Proximal (MCP)
        8,   # index_joint2           -> FAAS Index Intermediate (PIP)
        12,  # mid_joint1             -> FAAS Middle Proximal (MCP)
        13,  # mid_joint2             -> FAAS Middle Intermediate (PIP)
        17,  # ring_joint1            -> FAAS Ring Proximal (MCP)
        18,  # ring_joint2            -> FAAS Ring Intermediate (PIP)
        22,  # pinky_joint1           -> FAAS Pinky Proximal (MCP)
        23,  # pinky_joint2           -> FAAS Pinky Intermediate (PIP)
    )

    # Sign corrections: index_bend_joint rotates opposite direction in FAAS
    # (DexMani URDF axis = (-1, 0, 0); FAAS convention is opposite).
    _NATIVE_TO_FAAS_SCALES: tuple = (
        1.0, 1.0, 1.0,   # thumb
        -1.0, 1.0, 1.0,  # index (bend=-1)
        1.0, 1.0,         # middle
        1.0, 1.0,         # ring
        1.0, 1.0,         # pinky
    )

    # Per-joint offsets (all zero for XHand; non-zero for hands like Inspire).
    _NATIVE_TO_FAAS_OFFSETS: tuple = (0.0,) * 12

    # ─────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        idx = torch.tensor(self._NATIVE_TO_FAAS_INDICES, dtype=torch.long)
        scales = torch.tensor(self._NATIVE_TO_FAAS_SCALES, dtype=torch.float32)
        offsets = torch.tensor(self._NATIVE_TO_FAAS_OFFSETS, dtype=torch.float32)
        self.register_buffer('_faas_indices', idx, persistent=True)
        self.register_buffer('_scales', scales, persistent=True)
        self.register_buffer('_offsets', offsets, persistent=True)

    # ── Core hand-only conversions ──────────────────────────────────

    def native_to_faas(self, native_hand: torch.Tensor) -> torch.Tensor:
        """12D native XHand → 32D FAAS (zero-padded on unmapped indices).

        Transformation (matching UniDex ``_apply_scale_shift``)::

            faas_value = native_value * scale + offset

        For XHand, all offsets are 0.0 (hand_utils.json verified).

        Args:
            native_hand: ``(..., 12)`` in DexMani XHand joint order.

        Returns:
            ``(..., 32)`` in FAAS order.
        """
        assert native_hand.shape[-1] == self.NATIVE_HAND_DIM, (
            f"Expected last dim {self.NATIVE_HAND_DIM}, "
            f"got {native_hand.shape[-1]}"
        )
        shape = native_hand.shape[:-1]
        transformed = native_hand * self._scales + self._offsets
        faas = native_hand.new_zeros(*shape, self.MAPPED_JOINT_DIM)
        faas[..., self._faas_indices] = transformed
        return faas

    def faas_to_native(self, faas_hand: torch.Tensor) -> torch.Tensor:
        """32D FAAS → 12D native XHand.

        Inverse transformation::

            native = (faas_value - offset) / scale

        Uses the general division formula for robustness to future
        non-±1 scale values.  For XHand (all scales ∈ {1, -1}) this
        is numerically identical to the simplified form
        ``native = faas * scale`` (since 1/s = s).

        Args:
            faas_hand: ``(..., 32)`` in FAAS order.

        Returns:
            ``(..., 12)`` in DexMani XHand joint order.
        """
        assert faas_hand.shape[-1] == self.MAPPED_JOINT_DIM, (
            f"Expected last dim {self.MAPPED_JOINT_DIM}, "
            f"got {faas_hand.shape[-1]}"
        )
        native = faas_hand[..., self._faas_indices]
        return (native - self._offsets) / self._scales

    # ── Full-action convenience helpers ─────────────────────────────

    def transform_action(
        self, action: torch.Tensor, arm_dim: int,
    ) -> torch.Tensor:
        """Convert full action from native to FAAS (hand portion only).

        Args:
            action: ``(..., A)`` where A = arm_dim + 12 (native).
            arm_dim: number of arm dimensions (7 for joint, 9 for action_ee).

        Returns:
            ``(..., arm_dim + 32)`` in FAAS space.
        """
        arm = action[..., :arm_dim]
        hand = action[..., arm_dim:]
        return torch.cat([arm, self.native_to_faas(hand)], dim=-1)

    def inverse_transform_action(
        self, action: torch.Tensor, arm_dim: int,
    ) -> torch.Tensor:
        """Convert full action from FAAS to native (hand portion only).

        Args:
            action: ``(..., F)`` where F = arm_dim + 32 (FAAS).
            arm_dim: number of arm dimensions (7 for joint, 9 for action_ee).

        Returns:
            ``(..., arm_dim + 12)`` in native space.
        """
        arm = action[..., :arm_dim]
        hand = action[..., arm_dim:]
        return torch.cat([arm, self.faas_to_native(hand)], dim=-1)

    def transform_joint_state(
        self, joint_state: torch.Tensor,
    ) -> torch.Tensor:
        """Convert joint_state from native (19D) to FAAS (39D).

        joint_state arm is ALWAYS 7D arm joint angles, regardless of
        ``action_key``.  This is a fixed constant because joint_state
        comes from the robot's proprioceptive sensors (not the action
        control mode).

        Args:
            joint_state: ``(..., 19)`` = [arm_joints(7) | hand(12)].

        Returns:
            ``(..., 39)`` = [arm_joints(7) | FAAS_hand(32)].
        """
        arm = joint_state[..., :7]
        hand = joint_state[..., 7:]
        return torch.cat([arm, self.native_to_faas(hand)], dim=-1)

    # ── Introspection ───────────────────────────────────────────────

    def get_active_mask(self) -> torch.Tensor:
        """Return bool mask of shape ``(32,)`` — True where XHand has a joint."""
        mask = torch.zeros(self.MAPPED_JOINT_DIM, dtype=torch.bool)
        mask[self._faas_indices] = True
        return mask

    def get_active_count(self) -> int:
        """Return number of active FAAS dimensions for XHand (=12)."""
        return self.NATIVE_HAND_DIM
