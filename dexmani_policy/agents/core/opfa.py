"""OPFAAgent — DP3-based diffusion policy in OPFA GaLR latent space.

Predicts arm action (7-d) + hand action latent (1024-d) = 1031-d jointly,
then decodes the hand latent back to 12-d XHand joint angles via a frozen
GaLR autoencoder.

=== Key design ===

**Observation** (faithful to OPFA paper):
  - Scene point cloud: xyz only (pc_dim=3), default PointNeXT encoder → 64-d global token.
  - Hand state latent: 1024-d GaLR latent → MLP(1024→64).
    *During training*: pre-computed by ``dexmani_policy.agents.opfa.preprocess`` and loaded from .pt file.
    *During inference*: computed on-the-fly via FK + frozen GaLR encoder from joint_state.
  - Concat → 128-d per timestep → flatten across n_obs_steps.
  - NO arm joint state (unlike DP3). OPFA paper uses only scene PC + hand latent.

**Action space** (1031-d, matching official OPFA):
  - arm: 7 raw joint angles (xArm7) — unnormalized radian values.
  - hand_latent: 1024-d GaLR action latent (mean-pooled from L2-normalized
    superpoint features, per-dim ~0.03).
  - Scale balancing: hand_latent × sqrt(1024) ≈ ×32, then per-dim normalizer
    fitted on the full ``[arm_raw(7), action_latent_scaled(1024)]`` = 1031-d.
  - The per-dim ``(scale, offset)`` normalizer automatically balances arm
    (radian-scale, few dims) and hand-latent (many dims) in the denoising loss.

**Training** (compute_loss):
  1. Build 1031-d raw target: ``[arm_raw(7), action_latent × sqrt(1024)]``
  2. Normalize through the 1031-d per-dim normalizer
  3. UNet denoise → MSE loss

**Inference** (predict_action_from_cond):
  1. UNet denoise → pred_1031 (normalized space)
  2. Unnormalize full 1031-d via per-dim normalizer → raw space
  3. Split arm_raw(7) + hand_latent_scaled(1024)
  4. Unscale hand_latent → GaLR-decode → hand_joints(12)
  5. Assemble → 19-d native action

=== Dependencies ===

  - ``dexmani_policy.agents.opfa.GaLRAutoencoder`` — frozen autoencoder (encode for obs, decode for action).
  - ``dexmani_policy.agents.opfa.HandFKGenerator`` — FK + KPConv data for on-the-fly encoding.
  - ``dexmani_policy.agents.obs_encoder.pointcloud.registry`` — scene PC encoder.
  - ``dexmani_policy.agents.core.base.UNetDiffusionAgent`` — base class.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from dexmani_policy.agents.core.base import UNetDiffusionAgent
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.opfa.galr_autoencoder import GaLRAutoencoder, LatentCache, load_galr_encoder
from dexmani_policy.agents.opfa.hand_fk import HandFKGenerator

# Scale factor to bring L2-normalized 1024-d latent (~0.03 per dim) to
# the same ballpark as arm joint angles ([-1, 1], std ~0.5).
# sqrt(1024) ≈ 32 makes each dim roughly unit-scale.
_HAND_LATENT_SCALE = math.sqrt(1024)  # ≈ 32.0


# =============================================================================
# OPFA Observation Encoder
# =============================================================================


class OPFAObsEncoder(nn.Module):
    """OPFA observation encoder: scene PC (xyz) + hand latent (1024-d).

    Faithful to OPFA paper: NO arm joint state in observation.
    The OPFA paper refers to this as ``agent_pos``; we call it ``hand_latent``
    (1024-d GaLR-encoded hand state).
    """

    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        hand_latent_dim: int,
        num_points: int,
        n_obs_steps: int,
        hand_out_dim: int = 64,
        norm_before_activation: bool = False,
        fps_random_config: dict | None = None,
    ):
        super().__init__()
        self.pc_encoder = build_pc_global_encoder(
            encoder_type,
            pc_dim,
            config={
                "output_channels": pc_out_dim,
                "norm_before_activation": norm_before_activation,
                "fps_random_config": fps_random_config,
            },
        )
        self.hand_mlp = create_state_mlp(hand_latent_dim, hand_out_dim)
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}
        self.out_dim = self.pc_encoder.out_dim + self.hand_mlp.out_dim

    def forward(self, obs: dict) -> tuple[torch.Tensor, dict]:
        """OPFA observation encoding.

        Args:
            obs: dict with:
              - ``"point_cloud"``: ``(B*T, N, 3)`` scene PC (xyz only).
              - ``"hand_latent"``:  ``(B*T, 1024)`` pre-computed hand state latent.

        Returns:
            ``(cond, {})`` where cond is ``(B, out_dim * n_obs_steps)``.
        """
        pc = preprocess_point_cloud(
            obs["point_cloud"], self.num_points, self.use_coord_only, self.fps_random_config
        )
        pc_feat = self.pc_encoder(pc)["global_token"]  # (B*T, pc_out_dim)

        hand_feat = self.hand_mlp(obs["hand_latent"])  # (B*T, hand_out_dim)

        feat = torch.cat([pc_feat, hand_feat], dim=-1)  # (B*T, out_dim)
        B = feat.shape[0] // self.n_obs_steps
        return feat.reshape(B, -1), {}


# =============================================================================
# OPFAAgent
# =============================================================================


class OPFAAgent(UNetDiffusionAgent):
    """OPFA policy agent — diffusion in GaLR latent space.

    Predicts arm(7) + hand_latent(1024) = 1031-d jointly.
    I/O boundary transparently converts between 1031-d (policy space) and
    19-d (native joint space) via frozen GaLR decoder.
    """

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,  # 1031 = 7 + 1024
        encoder_type: str,
        pc_dim: int,
        pc_out_dim: int,
        hand_latent_dim: int,
        num_points: int,
        hand_out_dim: int = 64,
        fps_random_config: dict | None = None,
        galr_ckpt_path: str | None = None,
        arm_dim: int = 7,
        norm_before_activation: bool = False,
        **kwargs,
    ):
        # Build observation encoder
        obs_encoder = OPFAObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            hand_latent_dim=hand_latent_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            hand_out_dim=hand_out_dim,
            norm_before_activation=norm_before_activation,
            fps_random_config=fps_random_config,
        )

        super().__init__(
            obs_encoder=obs_encoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            **kwargs,
        )

        self.arm_dim = arm_dim
        self.hand_latent_dim = hand_latent_dim
        self.hand_latent_scale = _HAND_LATENT_SCALE
        self.galr_ckpt_path = galr_ckpt_path

        # GaLR autoencoder (frozen) — used for BOTH:
        #   encode: joint_state → hand latent (inference-time observation)
        #   decode: hand latent → joint angles (action I/O boundary)
        self._galr_model: GaLRAutoencoder | None = None

        # Linear correction for GaLR decoder (latent → native joint angles).
        # Optional: a linear least-squares fit on (latent, joint_angle) pairs can
        # improve decoder accuracy from ~0.4 rad (GaLR decoder) to ~0.04 rad on
        # in-distribution data.  Not loaded by default; call load_galr_correction()
        # with a separately-trained correction file to enable.
        self._galr_correction_weight: torch.Tensor | None = None  # (1024, 12)
        self._galr_correction_bias: torch.Tensor | None = None    # (12,)

        # HandFKGenerator — generates KPConv data for on-the-fly encoding.
        # Created lazily on first use (FK is lightweight, but URDF/STL loading
        # is slow).
        self._fk_gen: HandFKGenerator | None = None

        # LatentCache — angle→latent LRU cache for _compute_hand_latent.
        # Created lazily alongside _fk_gen on first inference call.
        self._latent_cache: LatentCache | None = None

        if galr_ckpt_path is not None and Path(galr_ckpt_path).exists():
            self._load_galr_model(galr_ckpt_path)

        # Auto-load linear correction if available (trained on trajectory data).
        # Reduces GaLR decoder error from ~0.45 rad → ~0.004 rad.
        correction_path = str(Path(galr_ckpt_path).parent / "galr_correction.pt") if galr_ckpt_path else None
        if correction_path and Path(correction_path).exists():
            self.load_galr_correction(correction_path)

    # -----------------------------------------------------------------
    # GaLR model management (encoder + decoder, both frozen)
    # -----------------------------------------------------------------

    def _load_galr_model(self, ckpt_path: str):
        """Load frozen GaLR autoencoder via shared loader."""
        self._galr_model = load_galr_encoder(ckpt_path)

    def set_galr_model(self, galr: GaLRAutoencoder):
        """Externally set and freeze the GaLR autoencoder."""
        galr.requires_grad_(False)
        galr.eval()
        self._galr_model = galr

    def load_galr_correction(self, correction_path: str):
        """Load linear correction weights for latent → joint angle decoding.

        The GaLR decoder has inherent reconstruction error (~0.4 rad MAE) due to
        the autoencoder bottleneck (1024→12 lossy compression).  A linear
        least-squares fit on (latent, joint_angle) pairs from ACTUAL trajectory
        data can reduce hand action error to ~0.04 rad *in distribution*.

        Note: correction trained on random-angle latents overfits badly
        (train ~0.05 rad, test ~0.5+ rad).  Train only on trajectory data.

        Expected ``.pt`` format: ``{"weight": (1024, 12), "bias": (12,)}``.
        Usage: ``corrected_angles = latent @ weight + bias``.
        """
        data = torch.load(correction_path, map_location="cpu", weights_only=True)
        self._galr_correction_weight = data["weight"]  # (1024, 12)
        self._galr_correction_bias = data["bias"]      # (12,)

    def load_state_dict(self, state_dict, strict=True):
        """Override to clean up lazily-created inference state before loading.

        ``_fk_gen`` (HandFKGenerator) and its 6 registered buffers are only
        created during ``_compute_hand_latent()`` (inference path), never
        during training.  When reusing the same agent across multiple
        checkpoints (e.g. ``select_best_ckpt.py`` adaptive elimination),
        stale ``_fk_gen`` buffers would cause ``strict=True`` failures
        because training checkpoints don't contain ``_fk_gen.*`` keys.
        """
        if self._fk_gen is not None:
            self._fk_gen = None
        super().load_state_dict(state_dict, strict=strict)

    @property
    def galr_ready(self) -> bool:
        return self._galr_model is not None

    # -----------------------------------------------------------------
    # Inference: observation preprocessing
    # -----------------------------------------------------------------

    def predict_action(self, obs_dict, denoise_timesteps=None):
        """Override to compute hand latent on-the-fly during inference.

        Training path: OPFADataset provides pre-computed ``hand_latent``.
        Inference path: the simulator provides ``joint_state`` (19-d native);
        we extract the 12-d hand joint angles and run FK + frozen GaLR encoder
        to produce the 1024-d hand state latent required by OPFAObsEncoder.
        """
        if "hand_latent" not in obs_dict:
            obs_dict = self._compute_hand_latent(obs_dict)
        return super().predict_action(obs_dict, denoise_timesteps)

    def _compute_hand_latent(self, obs_dict: dict) -> dict:
        """Compute hand state latent from observed joint state.

        Extracts 12-d hand joint angles from ``obs_dict["joint_state"]``,
        runs HandFKGenerator (GPU) + GaLR encoder to produce a 1024-d hand
        latent.  Uses a float-rounding LRU cache to avoid redundant FK+GaLR
        work for repeated joint configurations (common during continuous control).

        Args:
            obs_dict: must contain ``"joint_state"`` with shape
              ``(B, n_obs_steps, 19)``.

        Returns:
            New dict with ``"hand_latent"`` key added,
            shape ``(B, n_obs_steps, 1024)``.
        """
        joint_state = obs_dict["joint_state"]  # (B, n_obs_steps, 19)
        hand_joints = joint_state[..., self.arm_dim:self.arm_dim + 12]  # (B, n_obs_steps, 12)

        B, T, D = hand_joints.shape
        hand_joints_flat = hand_joints.reshape(-1, D)  # (B*T, 12)

        # Lazy-init FK generator + cache on first use
        device = next(self.parameters()).device
        if self._fk_gen is None:
            self._fk_gen = HandFKGenerator(cache_size=0).to(device)
            self._fk_gen.eval()
            for p in self._fk_gen.parameters():
                p.requires_grad_(False)
            self._latent_cache = LatentCache(max_size=2000, tolerance=0.001)

        latents = []
        for i in range(hand_joints_flat.shape[0]):
            angles = hand_joints_flat[i]  # (12,) on device
            angles_np = angles.cpu().numpy()

            cached = self._latent_cache.get(angles_np)
            if cached is not None:
                latents.append(cached.to(device))
                continue

            with torch.no_grad():
                data_dict = self._fk_gen(angles, hand_type="xhand")
                latent = self._galr_model.encode(data_dict)  # (1024,)

            self._latent_cache.put(angles_np, latent.cpu())
            latents.append(latent.to(device))

        # Build new dict to avoid mutating caller's obs_dict
        obs_dict = dict(obs_dict)
        obs_dict["hand_latent"] = torch.stack(latents).reshape(B, T, self.hand_latent_dim)
        return obs_dict

    # -----------------------------------------------------------------
    # Training: compute_loss
    # -----------------------------------------------------------------

    def _validate_batch(self, batch):
        """Relaxed validation — OPFA uses 19-d native actions internally.

        The base ``_validate_batch`` checks ``action_dim==1031``, but our
        dataset provides standard 19-d native actions.  compute_loss converts
        to 1031-d on-the-fly.
        """
        native_dim = self.arm_dim + 12  # 19
        if batch["action"].ndim != 3 or batch["action"].shape[1] != self.horizon:
            raise ValueError(
                f"action must be 3D (B, {self.horizon}, action_dim). "
                f"Got: {tuple(batch['action'].shape)}"
            )
        if batch["action"].shape[-1] != native_dim:
            raise ValueError(
                f"expected native action dim {native_dim} (arm={self.arm_dim}+hand=12). "
                f"Got {batch['action'].shape[-1]}. Check action_key."
            )
        self._validate_obs_dict(batch["obs"])

    def compute_loss(self, batch, **kwargs):
        """Build 1031-d target → per-dim normalise via 1031-d normalizer.

        Matches the official OPFA approach: the normalizer is fitted on
        ``[arm_raw(7), action_latent_scaled(1024)]`` so that every dimension
        receives an independent ``(scale, offset)``.  This automatically
        balances the arm (radian-scale, 7 dims) and hand-latent
        (mean-pooled from L2-normalised features, 1024 dims) in the denoising loss.
        """
        self._validate_batch(batch)
        cond, aux = self._build_cond(batch["obs"])

        # Build 1031-d raw target (matching normalizer fitting format)
        arm_raw = batch["action"][:, :, : self.arm_dim]               # (B, H, 7)  raw rad
        action_latent_scaled = batch["action_latent"] * self.hand_latent_scale  # (B, H, 1024)
        target = torch.cat([arm_raw, action_latent_scaled], dim=-1)   # (B, H, 1031)

        # Per-dim normalisation (normalizer was fitted on identically-built data)
        normed_target = self.normalizer["action"].normalize(target)
        if not torch.isfinite(normed_target).all():
            nan_count = (~torch.isfinite(normed_target)).sum().item()
            raise ValueError(
                f"NaN/Inf in normalized 1031-d target ({nan_count}/{normed_target.numel()} "
                f"elements). Check normalizer or data pipeline."
            )

        action_loss, loss_dict = self.action_decoder.compute_loss(
            cond, normed_target, dim_groups=self._get_dim_groups(), **kwargs
        )
        return self._merge_aux_loss(action_loss, loss_dict, aux)

    # -----------------------------------------------------------------
    # Inference: predict_action_from_cond
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None):
        """Denoise 1031-d → unnormalize → split → decode → 19-d native action.

        1. UNet denoise → pred_1031 = (B, horizon, 1031) in normalized space
        2. Unnormalize full 1031-d via per-dim normalizer → raw space
        3. Split → arm_raw(7) + hand_latent_scaled(1024)
        4. Unscale hand_latent → GaLR-decode → hand_joints(12)
        5. Assemble → 19-d native action
        """
        template = torch.zeros(
            cond.shape[0], self.horizon, self.action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps=denoise_timesteps)
        # pred: (B, horizon, 1031) — in normalized space

        # Unnormalize full 1031-d → raw space
        pred_raw = self.normalizer["action"].unnormalize(pred)  # (B, horizon, 1031)

        # Split
        arm_raw = pred_raw[:, :, : self.arm_dim]                # (B, horizon, 7)
        hand_latent_scaled = pred_raw[:, :, self.arm_dim:]      # (B, horizon, 1024)
        hand_latent = hand_latent_scaled / self.hand_latent_scale

        # Decode hand latent → joint angles.
        # Linear correction (if loaded) is preferred for trajectory-latent data.
        # Falls back to GaLR decoder (~0.4 rad MAE due to autoencoder bottleneck).
        B, H, D = hand_latent.shape
        if self._galr_correction_weight is not None and self._galr_correction_bias is not None:
            weight = self._galr_correction_weight.to(device=hand_latent.device, dtype=hand_latent.dtype)
            bias = self._galr_correction_bias.to(device=hand_latent.device, dtype=hand_latent.dtype)
            hl_flat = hand_latent.reshape(-1, D)  # (B*H, 1024)
            hand_joints = hl_flat @ weight + bias  # (B*H, 12)
            hand_joints = hand_joints.reshape(B, H, -1)  # (B, horizon, 12)
        elif self._galr_model is not None:
            hand_joints = self._galr_model.decode(hand_latent.reshape(-1, D))  # (B*H, 12)
            hand_joints = hand_joints.reshape(B, H, -1)  # (B, horizon, 12)
        else:
            # Fallback: no decoder available (e.g. smoke test without checkpoint)
            hand_joints = torch.zeros(B, H, 12, device=arm_raw.device, dtype=arm_raw.dtype)

        # Assemble native 19-d action
        pred_native = torch.cat([arm_raw, hand_joints], dim=-1)  # (B, horizon, 19)

        start = self.n_obs_steps - 1
        control_action = pred_native[:, start : start + self.n_action_steps]
        tail = pred_native[:, start + self.n_action_steps :]

        return {
            "pred_action": pred_native,
            "control_action": control_action,
            "tail": tail,
            "hand_latent": hand_latent,
            "arm_raw": arm_raw,
        }

    # -----------------------------------------------------------------
    # Dim groups — per-component loss logging
    # -----------------------------------------------------------------

    def _get_dim_groups(self):
        """Per-group loss for debugging arm vs hand latent balance."""
        return {"arm": (0, self.arm_dim), "hand_latent": (self.arm_dim, self.action_dim)}

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def control_action_dim(self):
        """Control action is always 19-d native (7 arm + 12 hand joints)."""
        return self.arm_dim + 12  # 19

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    def get_optim_param_groups(self, lr, obs_lr, weight_decay, obs_wd):
        """Optimizer groups: GaLR decoder is frozen and excluded."""
        action_groups = self.action_decoder.model.get_optim_groups(weight_decay)
        for g in action_groups:
            g["lr"] = lr
        obs_groups = []
        for name, param in self.obs_encoder.named_parameters():
            if param.requires_grad:
                obs_groups.append({"params": [param], "lr": obs_lr, "weight_decay": obs_wd})
        return action_groups + obs_groups

    def __repr__(self) -> str:
        galr_info = "✓" if self.galr_ready else "✗"
        return (
            f"OPFAAgent(action={self.action_dim}d=arm{self.arm_dim}+latent{self.hand_latent_dim}, "
            f"galr_model={galr_info}, horizon={self.horizon})"
        )
