"""
DexLatent-wrapped dataset: converts native actions/joint_state to latent space.

Wraps an existing PCDataset (or any BaseDataset subclass) and applies
DexLatentHandVAE encoding in ``__getitem__``, exactly mirroring the FAAS
``_apply_faas_mapping`` pattern.

This is a standalone Dataset class — zero modifications to BaseDataset.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dexmani_policy.common.dexlatent_autoencoder import DexLatentHandVAE
from dexmani_policy.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    build_mixed_action_normalizer,
)
from dexmani_policy.datasets.base_dataset import BaseDataset


# ═══════════════════════════════════════════════════════════════════════════════
# URDF joint limits for xarm7_xhand_right (the hand the VAE was trained on)
# ═══════════════════════════════════════════════════════════════════════════════
# Extracted from DexLatent Assets/xarm7_xhand/xarm7_xhand_right_hand.urdf.
# These MUST match angles_to_normalized() / _normalized_to_all_joint_angles()
# in HandLatent.kinematics, otherwise the VAE receives out-of-distribution inputs.
#
# Arm (joint1–joint7) + Hand (thumb_bend … pinky2) = 19 independent revolute DoFs.

_ARM_LOWER = np.array(
    [-6.2832, -2.0590, -6.2832, -0.1920, -6.2832, -1.6930, -6.2832],
    dtype=np.float32,
)
_ARM_UPPER = np.array(
    [6.2832, 2.0944, 6.2832, 3.9270, 6.2832, 3.1416, 6.2832],
    dtype=np.float32,
)
_HAND_LOWER = np.array(
    [0.0, -0.698, 0.0, -0.174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_HAND_UPPER = np.array(
    [1.832, 1.745, 1.745, 0.174, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919],
    dtype=np.float32,
)

# Full 19D (7 arm + 12 hand)
_FULL19_LOWER = np.concatenate([_ARM_LOWER, _HAND_LOWER])
_FULL19_UPPER = np.concatenate([_ARM_UPPER, _HAND_UPPER])


def _urdf_scale_offset(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scale/offset that map ``[lower, upper] → [-1, 1]``.

    Matches the formula in ``HandKinematicsModel.angles_to_normalized()``::

        normalised = 2 * (raw - lower) / (upper - lower) - 1
                   = raw * scale + offset      (our LinearNormalizer convention)

    where  ``scale = 2 / span``,  ``offset = -1 - lower * scale``.
    """
    span = upper - lower
    safe_span = np.where(np.abs(span) < 1e-8, np.ones_like(span), span)
    scale = 2.0 / safe_span
    offset = -1.0 - lower * scale
    return scale.astype(np.float32), offset.astype(np.float32)


def _build_urdf_normalizer_for_joint_state() -> SingleFieldLinearNormalizer:
    """Return a ``SingleFieldLinearNormalizer`` keyed by URDF joint limits
    for the native 19D ``joint_state`` (7 arm + 12 hand)."""
    scale, offset = _urdf_scale_offset(_FULL19_LOWER, _FULL19_UPPER)
    norm = SingleFieldLinearNormalizer.create_manual(
        scale=torch.from_numpy(scale),
        offset=torch.from_numpy(offset),
    )
    for p in norm.parameters():
        p.requires_grad_(False)
    return norm


def _build_urdf_normalizer_for_action(
    action_key: str,
    replay_buffer: dict,
) -> SingleFieldLinearNormalizer:
    """Return a ``SingleFieldLinearNormalizer`` for the native action space.

    - *joint* mode (19D = 7 arm + 12 hand): pure URDF limits.
    - *action_ee* mode (21D = 3 xyz + 6 rot6d + 12 hand): xyz gets data-driven
      limits (Cartesian, not joint angles); rot6d gets identity; hand gets URDF.
    """
    if action_key == "action":
        scale, offset = _urdf_scale_offset(_FULL19_LOWER, _FULL19_UPPER)
        norm = SingleFieldLinearNormalizer.create_manual(
            scale=torch.from_numpy(scale),
            offset=torch.from_numpy(offset),
        )
        for p in norm.parameters():
            p.requires_grad_(False)
        return norm

    # action_ee: 21D = [xyz(3), rot6d(6), hand(12)]
    assert action_key == "action_ee"
    hand_scale, hand_offset = _urdf_scale_offset(_HAND_LOWER, _HAND_UPPER)

    # Fit xyz(0:3) from replay buffer data
    xyz_data = replay_buffer["action_ee"][..., :3]
    xyz_min = np.min(xyz_data, axis=tuple(range(xyz_data.ndim - 1)))
    xyz_max = np.max(xyz_data, axis=tuple(range(xyz_data.ndim - 1)))
    xyz_span = xyz_max - xyz_min
    xyz_safe = np.where(np.abs(xyz_span) < 1e-4, np.ones_like(xyz_span), xyz_span)
    xyz_scale = 2.0 / xyz_safe
    xyz_offset = -1.0 - xyz_min * xyz_scale

    scale = np.concatenate([
        xyz_scale,
        np.ones(6, dtype=np.float32),  # rot6d → identity
        hand_scale,
    ])
    offset = np.concatenate([
        xyz_offset,
        np.zeros(6, dtype=np.float32),  # rot6d → identity
        hand_offset,
    ])
    norm = SingleFieldLinearNormalizer.create_manual(
        scale=torch.from_numpy(scale.astype(np.float32)),
        offset=torch.from_numpy(offset.astype(np.float32)),
    )
    for p in norm.parameters():
        p.requires_grad_(False)
    return norm


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level encode helpers (shared by __getitem__ and get_normalizer)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_action(
    action: torch.Tensor,
    vae: DexLatentHandVAE,
    tcp_dim: int,
) -> torch.Tensor:
    """Convert native action → latent action (hand portion only).

    Args:
        action: ``(..., tcp_dim + hand_dim)`` native action tensor.
        vae: frozen DexLatentHandVAE instance.
        tcp_dim: arm dimension (7 for joint, 9 for action_ee).

    Returns:
        ``(..., tcp_dim + latent_dim)`` latent action tensor.
    """
    arm = action[..., :tcp_dim]
    hand = action[..., tcp_dim:]
    return torch.cat([arm, vae.encode(hand)], dim=-1)


def encode_joint_state(
    joint_state: torch.Tensor,
    vae: DexLatentHandVAE,
) -> torch.Tensor:
    """Convert native joint_state → latent joint_state.

    joint_state arm is ALWAYS 7D (proprioceptive arm joint angles),
    independent of ``action_key``.

    Args:
        joint_state: ``(..., 19)`` = [arm_joints(7) | hand(12)].
        vae: frozen DexLatentHandVAE instance.

    Returns:
        ``(..., 39)`` = [arm_joints(7) | latent(32)].
    """
    arm = joint_state[..., :7]
    hand = joint_state[..., 7:]
    return torch.cat([arm, vae.encode(hand)], dim=-1)


def decode_action(
    latent_action: torch.Tensor,
    vae: DexLatentHandVAE,
    tcp_dim: int,
) -> torch.Tensor:
    """Convert latent action → native action (hand portion only).

    Args:
        latent_action: ``(..., tcp_dim + latent_dim)`` latent action tensor.
        vae: frozen DexLatentHandVAE instance.
        tcp_dim: arm dimension.

    Returns:
        ``(..., tcp_dim + hand_dim)`` native action tensor.
    """
    arm = latent_action[..., :tcp_dim]
    latent = latent_action[..., tcp_dim:]
    return torch.cat([arm, vae.decode(latent)], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class DexLatentPCDataset(torch.utils.data.Dataset):
    """Wraps a BaseDataset subclass, converting actions to DexLatent latent space.

    The wrapper intercepts ``__getitem__`` after the inner dataset has done
    all augmentation and tensor conversion, then encodes the hand portion
    of both ``action`` and ``obs['joint_state']`` through the frozen VAE.

    Parameters
    ----------
    source_dataset : BaseDataset
        Underlying dataset (PCDataset, RGBPCDataset, etc.) configured
        **without** FAAS (``use_faas=False``).
    vae : DexLatentHandVAE
        Frozen pretrained autoencoder.
    tcp_dim : int
        Arm dimension: 7 for ``action_key='action'``, 9 for ``action_ee``.
    """

    def __init__(
        self,
        source_dataset: BaseDataset,
        vae: DexLatentHandVAE,
        tcp_dim: int = 7,
    ) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.vae = vae
        self.tcp_dim = int(tcp_dim)

        # Expose attributes that downstream code inspects
        self.action_key = source_dataset.action_key
        self.use_aux_ee = source_dataset.use_aux_ee
        self.sensor_modalities = source_dataset.sensor_modalities
        self.horizon = source_dataset.horizon

        # ── Fit native (19D) normalizer for pre-VAE normalization ──
        # The VAE was trained on normalized [-1, 1] hand poses.  We must
        # normalise raw radian values BEFORE feeding them to the encoder.
        self._native_normalizer = self._fit_native_normalizer()

    # ── Native normalizer (pre-VAE [-1,1] mapping) ──────────────────

    def _fit_native_normalizer(self) -> LinearNormalizer:
        """Build a native normalizer using **URDF joint limits** (NOT data-driven).

        The DexLatent VAE was trained on data normalised by
        ``HandKinematicsModel.angles_to_normalized()``, which maps
        ``[joint_lower, joint_upper] → [-1, 1]`` using URDF ``<limit>`` tags.

        If we used data-driven ``LinearNormalizer(mode='limits')`` instead,
        the VAE would receive inputs in a *different* parameterisation than
        its training distribution — the root cause of the 0 % success rate.

        .. note::
            For ``action_ee`` mode the arm portion is TCP pose (xyz + rot6d),
            not joint angles.  xyz gets data-driven limits; rot6d is identity.
        """
        normalizer = LinearNormalizer()
        replay_buffer = self.source_dataset.replay_buffer

        # joint_state is always 19D (7 arm joints + 12 hand joints)
        normalizer["joint_state"] = _build_urdf_normalizer_for_joint_state()

        # action: joint mode (19D) → URDF; action_ee mode (21D) → mixed
        if self.use_aux_ee:
            # Aux-EE: action_key='action' but extra EE info appended → 28D.
            # Fall back to data-driven for the full concatenated action
            # (this config is mutually exclusive with DexLatent in practice
            #  per _validate_dexlatent_config, but kept for safety).
            parts = [replay_buffer["action"]]
            parts.append(replay_buffer["action_ee"][..., :9])
            action_raw = np.concatenate(parts, axis=-1)
            normalizer.fit(
                data={"action": action_raw},
                last_n_dims=1,
                mode="limits",
            )
        else:
            normalizer["action"] = _build_urdf_normalizer_for_action(
                self.action_key, replay_buffer
            )

        return normalizer

    @property
    def native_normalizer(self) -> LinearNormalizer:
        """The native 19D normalizer used for pre-VAE [-1,1] mapping."""
        return self._native_normalizer

    # ── Core Dataset protocol ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample with actions and joint_state in latent space.

        The inner dataset returns native-space tensors (raw radian values).
        We first normalise the hand portion to [-1, 1] so the VAE encoder
        operates within its training distribution, then encode to latent.
        """
        data = self.source_dataset[idx]

        # Step 1: Normalise native 19D → [-1, 1] (VAE training distribution)
        data["action"] = self._native_normalizer["action"].normalize(
            data["action"]
        )
        if "joint_state" in data.get("obs", {}):
            data["obs"]["joint_state"] = (
                self._native_normalizer["joint_state"].normalize(
                    data["obs"]["joint_state"]
                )
            )

        # Step 2: VAE-encode hand → latent (arm passes through unchanged)
        data["action"] = encode_action(data["action"], self.vae, self.tcp_dim)
        if "joint_state" in data.get("obs", {}):
            data["obs"]["joint_state"] = encode_joint_state(
                data["obs"]["joint_state"], self.vae
            )

        return data

    # ── Normalizer ───────────────────────────────────────────────────

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """Fit normalizer on latent-space data.

        **Two-stage fitting** (matches the ``__getitem__`` flow exactly):

        1.  Normalise raw replay-buffer data to [-1, 1] via the native
            normalizer (same instance used in ``__getitem__``).
        2.  VAE-encode the normalised data → 39D latent.
        3.  Fit the *latent-space* normalizer that the diffusion model uses.

        The native normalizer is attached as ``normalizer._native`` so the
        agent can access it at inference time for pre-VAE normalisation and
        post-VAE denormalisation.
        """
        normalizer = LinearNormalizer()
        replay_buffer = self.source_dataset.replay_buffer

        # ── Step 1: gather raw native data ──
        if self.use_aux_ee:
            parts = [replay_buffer["action"]]
            parts.append(replay_buffer["action_ee"][..., :9])
            action_native = np.concatenate(parts, axis=-1)
        else:
            action_native = replay_buffer[self.action_key]
        js_native = replay_buffer["joint_state"]

        # ── Step 2: normalise raw → [-1, 1] (VAE training distribution) ──
        js_norm = self._native_normalizer["joint_state"].normalize(js_native)
        action_norm = self._native_normalizer["action"].normalize(action_native)

        # ── Step 3: VAE-encode hand portion ──
        js_t = torch.from_numpy(np.asarray(js_norm)).float()
        action_t = torch.from_numpy(np.asarray(action_norm)).float()

        joint_state_latent = encode_joint_state(js_t, self.vae).numpy()
        action_latent = encode_action(action_t, self.vae, self.tcp_dim).numpy()

        # ── Step 4: fit latent-space (39D) normalizer ──
        fit_dict: dict = {"joint_state": joint_state_latent}

        if "point_cloud" in replay_buffer:
            fit_dict["point_cloud"] = replay_buffer["point_cloud"]

        if self.action_key == "action_ee" and not self.use_aux_ee:
            # EE mode: mixed normalizer (rot6d identity, rest limits)
            normalizer.fit(data=fit_dict, last_n_dims=1, mode=mode)
            normalizer["action"] = build_mixed_action_normalizer(action_latent)
        else:
            fit_dict["action"] = action_latent
            normalizer.fit(data=fit_dict, last_n_dims=1, mode=mode)

        # ── Attach native normalizer for agent access at inference ──
        normalizer._native = self._native_normalizer

        return normalizer

    # ── Validation split ─────────────────────────────────────────────

    def get_validation_dataset(self) -> Optional["DexLatentPCDataset"]:
        """Create a validation split of this dataset.

        Returns a new ``DexLatentPCDataset`` wrapping the inner dataset's
        validation split (no augmentation, val-only sampler).
        """
        inner_val = self.source_dataset.get_validation_dataset()
        if inner_val is None:
            return None
        return DexLatentPCDataset(
            source_dataset=inner_val,
            vae=self.vae,
            tcp_dim=self.tcp_dim,
        )
