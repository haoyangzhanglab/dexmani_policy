"""
DexLatent agent: UNetDiffusionAgent variant that operates in cross-hand latent space.

Predicts 39D latent actions (7 arm pass-through + 32 hand latent) and decodes
back to native 19D at the I/O boundary — identical in structure to the FAAS
pipeline in ``BaseAgent``.

Reference: XL-VLA (CVPR 2026 Highlight), arXiv:2603.10158
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from dexmani_policy.agents.core.dp3 import DP3Agent
from dexmani_policy.common.normalizer import LinearNormalizer


class DexLatentAgent(DP3Agent):
    """DP3+Diffusion agent that predicts in DexLatent cross-hand latent space.

    Inherits DP3Agent (and therefore UNetDiffusionAgent), accepting all
    standard dp3 config fields (encoder_type, pc_dim, down_dims, etc.).

    Training:  39D latent actions from DexLatentPCDataset → diffusion loss
    Inference: predict 39D latent → decode hand(32D)→hand(12D) → native 19D

    **Normalisation pipeline** (two-stage)::

        Native 19D (radians)
          → native_normalizer → [-1, 1]          (pre-VAE)
          → VAE.encode(hand)  → 39D latent
          → latent_normalizer → [-1, 1]          (pre-diffusion)
          → Diffusion → [-1, 1]
          → latent_normalizer.unnormalize         (post-diffusion)
          → VAE.decode(hand) → [-1, 1]
          → native_normalizer.unnormalize         (post-VAE → radians)
          → env.step()
    """

    # ── Native normalizer (lazy, not nn.Module) ──────────────────────

    @property
    def native_normalizer(self) -> LinearNormalizer | None:
        """Native 19D normalizer, lazily rebuilt from stored state dict.

        Stored as a plain dict (``_native_norm_state``) rather than an
        ``nn.Module`` so it does NOT appear in ``state_dict()`` — this
        avoids ``strict=True`` checkpoint-loading errors at eval time.
        """
        state = getattr(self, "_native_norm_state", None)
        if state is None:
            return None
        cache = getattr(self, "_native_norm_cache", None)
        if cache is not None:
            return cache
        norm = LinearNormalizer()
        norm.load_state_dict(state)
        self._native_norm_cache = norm
        return norm

    # ── predict_action ───────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        """Normalise native obs → VAE-encode hand → predict in latent space.

        Idempotent: only converts when input is native 19D.  If already
        39D (called from compute_loss / training), passes through unchanged.
        """
        self._validate_obs_dict(obs_dict)
        if getattr(self, "use_dexlatent", False):
            # Stage 1: native 19D (radians) → [-1, 1] via native normalizer
            obs_dict = self._preprocess_native(obs_dict)
            # Stage 2: VAE-encode hand portion → 39D latent
            obs_dict = self._convert_obs_to_latent(obs_dict)
        cond, _ = self._build_cond(obs_dict)
        return self.predict_action_from_cond(cond, denoise_timesteps)

    def _preprocess_native(self, obs_dict: Dict) -> Dict:
        """Normalise only the joint_state key using the native normalizer.

        Idempotent: if ``joint_state`` is already 39D (latent), skip.
        Otherwise normalise the 19D native joint_state to [-1, 1] so the
        VAE encoder receives in-distribution inputs.
        """
        native_norm = getattr(self, "native_normalizer", None)
        if native_norm is None or "joint_state" not in obs_dict:
            return obs_dict
        js = obs_dict["joint_state"]
        if js.shape[-1] != 7 + self.hand_dim:  # 19 → already latent
            return obs_dict
        obs_dict = dict(obs_dict)  # shallow copy
        obs_dict["joint_state"] = native_norm["joint_state"].normalize(js)
        return obs_dict

    def _convert_obs_to_latent(self, obs_dict: Dict) -> Dict:
        """Convert native joint_state (19D, [-1,1]) → latent (39D).

        **Idempotent**: only converts when input is native-dim
        (19D = 7 arm + 12 hand).  If already latent (39D), returns unchanged.

        The caller must ensure the hand portion is already in [-1, 1]
        (via ``_preprocess_native``) before calling this method.
        """
        if "joint_state" not in obs_dict:
            return obs_dict
        js = obs_dict["joint_state"]
        if js.shape[-1] != 7 + self.hand_dim:  # 19
            return obs_dict  # already latent-dim
        arm_state = js[..., :7]
        hand_state = js[..., 7:]
        latent_hand = self.dexlatent_vae.encode(hand_state)
        return {**obs_dict, "joint_state": torch.cat([arm_state, latent_hand], dim=-1)}

    # ── predict_action_from_cond ──────────────────────────────────────

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None) -> Dict:
        """Predict latent action, then decode to native 19D before returning.

        Decoding pipeline::

            Diffusion (39D norm) → unnormalize (39D latent range)
            → VAE.decode(hand) → 19D [-1, 1]
            → native_normalizer.unnormalize → 19D radians → env.step()
        """
        # --- Diffusion denoise in 39D latent space ---
        template = torch.zeros(
            cond.shape[0], self.horizon, self.action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
        pred = self.normalizer["action"].unnormalize(pred)

        # --- Decode latent → native [-1, 1] ---
        if getattr(self, "use_dexlatent", False):
            if self.dexlatent_vae is None:
                raise RuntimeError(
                    "DexLatent autoencoder not loaded. "
                    "Call inject_dexlatent_into_agent first."
                )
            pred = self.dexlatent_vae.inverse_transform_action(pred, self.tcp_dim)

            # --- Denormalise native [-1, 1] → radians ---
            native_norm = getattr(self, "native_normalizer", None)
            if native_norm is not None:
                pred = native_norm["action"].unnormalize(pred)

        # --- Slice control_action + tail ---
        start = self.n_obs_steps - 1
        control_action = pred[:, start : start + self.n_action_steps]
        tail = pred[:, start + self.n_action_steps :]
        if self.control_action_dim != self.action_dim:
            control_action = control_action[..., : self.control_action_dim]
            tail = tail[..., : self.control_action_dim]
        return {
            "pred_action": pred,
            "control_action": control_action,
            "tail": tail,
        }

    # ── control_action_dim ────────────────────────────────────────────

    @property
    def control_action_dim(self) -> int:
        """Native action dimension for env.step (19D)."""
        if getattr(self, "use_dexlatent", False):
            return self.tcp_dim + self.hand_dim
        return self.action_dim

    # ── compute_action_mse ────────────────────────────────────────────

    @torch.no_grad()
    def compute_action_mse(self, batch: Dict[str, Any]) -> float:
        """MSE in native joint space (radians).

        GT from dataset is 39D latent; we decode it through the full
        pipeline (VAE decode + native denormalise) before comparing.
        """
        obs = batch["obs"]
        gt_action = batch["action"]
        if getattr(self, "use_dexlatent", False):
            gt_action = self.dexlatent_vae.inverse_transform_action(
                gt_action, self.tcp_dim
            )
            native_norm = getattr(self, "native_normalizer", None)
            if native_norm is not None:
                gt_action = native_norm["action"].unnormalize(gt_action)
        pred_action = self.predict_action(obs)["pred_action"]
        return F.mse_loss(pred_action, gt_action).item()
