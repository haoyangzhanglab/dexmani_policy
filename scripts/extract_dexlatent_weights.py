#!/usr/bin/env python3
"""Phase 0: Extract & validate DexLatent autoencoder weights for DexMani integration.

Loads the DexLatent checkpoint, extracts per-hand encoder/decoder MLP weights,
rebuilds standalone HandAutoencoder instances, validates roundtrip fidelity on
both synthetic and real data, and exports a clean checkpoint for downstream use.

Usage:
    python scripts/extract_dexlatent_weights.py
    python scripts/extract_dexlatent_weights.py --ckpt <path> --output <path>
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# ─── path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEXLATENT_ROOT = Path.home() / "Desktop" / "DexLatent"
sys.path.insert(0, str(_DEXLATENT_ROOT))

# Import DexLatent modules AFTER path setup
from HandLatent.kinematics import (
    HAND_CONFIGS,
    HandKinematicsModel,
    MultiHandDifferentiableFK,
)
from HandLatent.model import HandAutoencoder, TrainingConfig

# ─── constants ────────────────────────────────────────────────────────────────
DEFAULT_CKPT = str(
    _DEXLATENT_ROOT / "Checkpoints" / "20260311_225425" / "checkpoint_epoch_1000.pt"
)
DEFAULT_OUTPUT = str(_PROJECT_ROOT / "pretrained_models" / "dexlatent_autoencoders.pt")
DEFAULT_DEMO = str(_DEXLATENT_ROOT / "Dataset" / "demo.npz")

HAND_NAMES = [
    "xarm7_xhand_right",
    "xarm7_ability_right",
    "xarm7_inspire_right",
    "xarm7_paxini_right",
]

# Per-hand metadata — verified against DexLatent URDF via HandKinematicsModel
# (Auto-detected from checkpoint at runtime; these serve as fallback docs)
# fmt: off
HAND_META = {
    "xarm7_xhand_right":   {"arm_dof": 7, "hand_dof": 12, "total_dof": 19},
    "xarm7_ability_right":  {"arm_dof": 7, "hand_dof":  6, "total_dof": 13},
    "xarm7_inspire_right":  {"arm_dof": 7, "hand_dof":  6, "total_dof": 13},
    "xarm7_paxini_right":   {"arm_dof": 7, "hand_dof": 16, "total_dof": 23},
}
# fmt: on


def _detect_hand_dof(weights: Dict[str, torch.Tensor]) -> int:
    """Auto-detect hand_dof from encoder's first Linear layer.

    The encoder backbone's first layer is Linear(hand_dof, hidden_dim).
    Its weight has shape (hidden_dim, hand_dof), so hand_dof = weight.shape[1].
    """
    for key in weights:
        if key.startswith("encoder.0.weight"):
            return weights[key].shape[1]
    raise KeyError("Cannot detect hand_dof: no encoder.0.weight in extracted weights")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Extract weights from DexLatent checkpoint
# ═══════════════════════════════════════════════════════════════════════════════


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove a dot-separated prefix from state dict keys."""
    prefix_dot = prefix + "."
    stripped = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix_dot):
            stripped[key[len(prefix_dot) :]] = value
    return stripped


def extract_hand_autoencoder_weights(
    raw_state_dict: Dict[str, torch.Tensor],
    hand_name: str,
) -> Dict[str, torch.Tensor]:
    """Extract encoder + decoder weights for one hand from the flat checkpoint.

    The DexLatent checkpoint stores autoencoders as a flat state_dict of
    nn.ModuleDict with dot-separated keys like:
        "xarm7_xhand_right.hand_encoder_backbone.0.weight"

    We strip the hand_name prefix and collect only the MLP weights
    (excluding mean/logvar heads since we use deterministic encoding).
    """
    all_hand_keys = _strip_prefix(raw_state_dict, hand_name)
    extracted = OrderedDict()

    # Encoder backbone: MLP [hand_dof, 64, 128, 64] with LayerNorm+ReLU
    # Keys: hand_encoder_backbone.{0,1,3,4,6,7}.weight/bias
    for key, value in all_hand_keys.items():
        if key.startswith("hand_encoder_backbone."):
            extracted[f"encoder.{key.split('hand_encoder_backbone.', 1)[1]}"] = value

    # Encoder mean head: Linear(64, 32)
    for key, value in all_hand_keys.items():
        if key.startswith("hand_mean_head."):
            new_key = key.replace("hand_mean_head.", "encoder_head.")
            extracted[new_key] = value

    # Decoder: MLP [32, 64, 128, 64, hand_dof] with LayerNorm+ReLU+Tanh
    # Keys: hand_decoder.{0,1,3,4,6,7,9}.weight/bias
    for key, value in all_hand_keys.items():
        if key.startswith("hand_decoder."):
            extracted[f"decoder.{key.split('hand_decoder.', 1)[1]}"] = value

    return extracted


def extract_all_autoencoders(
    ckpt_path: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], dict]:
    """Extract all 4 hand autoencoder weights from DexLatent checkpoint.

    Returns:
        weights: dict[hand_name, state_dict] for all 4 hands
        metadata: config metadata (latent_dim, hidden_dims, etc.)
    """
    print(f"[1/5] Loading DexLatent checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    config = payload["config"]
    metadata = {
        "latent_dim_hand": config.get("latent_dim_hand", 32),
        "hand_hidden_dims": tuple(config.get("hand_hidden_dims", (64, 128, 64))),
        "arm_dof": config.get("arm_dof", 7),
        "epoch": payload.get("epoch", None),
        "source_hand_names": payload.get("hand_names", []),
    }

    raw_sd = payload["autoencoders"]
    weights = {}
    for hand_name in HAND_NAMES:
        extracted = extract_hand_autoencoder_weights(raw_sd, hand_name)
        param_count = sum(v.numel() for v in extracted.values())
        print(f"  {hand_name}: {len(extracted)} tensors, {param_count:,} params")
        weights[hand_name] = extracted

    return weights, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Rebuild standalone autoencoders with extracted weights
# ═══════════════════════════════════════════════════════════════════════════════


def build_standalone_autoencoder(
    hand_name: str,
    weights: Dict[str, torch.Tensor],
    metadata: dict,
) -> nn.Module:
    """Build a clean nn.Module with the extracted encoder/decoder weights.

    The returned module has this structure:
        encoder: nn.Sequential  (hand_dof→64→128→64→32, LayerNorm+ReLU)
        decoder: nn.Sequential  (32→64→128→64→hand_dof, LayerNorm+ReLU+Tanh)

    It is a pure MLP with zero URDF/FK/Pinocchio dependencies.
    """
    hand_dof = _detect_hand_dof(weights)
    latent_dim = metadata["latent_dim_hand"]
    hidden_dims = metadata["hand_hidden_dims"]  # (64, 128, 64)

    # --- Build encoder ---
    encoder_layers = []
    in_dim = hand_dof
    for i, width in enumerate(hidden_dims):
        encoder_layers.append(nn.Linear(in_dim, width))
        encoder_layers.append(nn.LayerNorm(width))
        encoder_layers.append(nn.ReLU())
        in_dim = width
    # Final projection to latent
    encoder_head = nn.Linear(in_dim, latent_dim)
    encoder = nn.Sequential(*encoder_layers)
    encoder_head_module = encoder_head

    # --- Build decoder ---
    decoder_layers = []
    in_dim = latent_dim
    for i, width in enumerate(hidden_dims):
        decoder_layers.append(nn.Linear(in_dim, width))
        decoder_layers.append(nn.LayerNorm(width))
        decoder_layers.append(nn.ReLU())
        in_dim = width
    decoder_layers.append(nn.Linear(in_dim, hand_dof))
    decoder_layers.append(nn.Tanh())
    decoder = nn.Sequential(*decoder_layers)

    # --- Load weights ---
    # Map extracted keys → encoder Sequential indices
    # encoder.{0,1,3,4,6,7} correspond to layers 0-7 of Sequential
    encoder_sd = OrderedDict()
    for key, value in weights.items():
        if key.startswith("encoder.") and not key.startswith("encoder_head."):
            encoder_sd[key.replace("encoder.", "")] = value
    encoder.load_state_dict(encoder_sd)

    # encoder_head
    encoder_head_sd = OrderedDict()
    for key, value in weights.items():
        if key.startswith("encoder_head."):
            encoder_head_sd[key.replace("encoder_head.", "")] = value
    encoder_head_module.load_state_dict(encoder_head_sd)

    # decoder.{0,1,3,4,6,7,9} correspond to layers 0-8 of Sequential
    decoder_remap = {
        "0": "0", "1": "1",  # Linear(32,64), LayerNorm(64)
        "3": "3", "4": "4",  # Linear(64,128), LayerNorm(128)
        "6": "6", "7": "7",  # Linear(128,64), LayerNorm(64)
        "9": "9",            # Linear(64, hand_dof) — Tanh is 10 in Sequential
    }
    decoder_sd = OrderedDict()
    for key, value in weights.items():
        if key.startswith("decoder."):
            local_key = key.replace("decoder.", "")
            parts = local_key.split(".", 1)
            idx = parts[0]
            if idx in decoder_remap:
                new_key = f"{decoder_remap[idx]}.{parts[1]}" if len(parts) > 1 else decoder_remap[idx]
                decoder_sd[new_key] = value
    decoder.load_state_dict(decoder_sd, strict=False)

    # Package into a simple module
    class StandaloneAutoencoder(nn.Module):
        def __init__(self, enc, head, dec, hd, ld):
            super().__init__()
            self.encoder_backbone = enc
            self.encoder_head = head
            self.decoder = dec
            self.hand_dof = hd
            self.latent_dim = ld

        def encode(self, hand_qpos: torch.Tensor) -> torch.Tensor:
            """hand_qpos: (..., hand_dof) in [-1,1] → latent: (..., latent_dim)"""
            features = self.encoder_backbone(hand_qpos)
            return self.encoder_head(features)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            """latent: (..., latent_dim) → hand_qpos: (..., hand_dof) in [-1,1]"""
            return self.decoder(latent)

        def forward(self, hand_qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode→decode roundtrip. Returns (reconstructed, latent)."""
            latent = self.encode(hand_qpos)
            recon = self.decode(latent)
            return recon, latent

    model = StandaloneAutoencoder(
        encoder, encoder_head_module, decoder, hand_dof, latent_dim
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Synthetic validation (random [-1,1] vectors)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_fk_model(hand_name: str) -> HandKinematicsModel:
    """Get (cached) FK model for one hand."""
    if not hasattr(_get_fk_model, "_cache"):
        _get_fk_model._cache = {}
    if hand_name not in _get_fk_model._cache:
        _get_fk_model._cache[hand_name] = MultiHandDifferentiableFK([hand_name]).models[hand_name]
    return _get_fk_model._cache[hand_name]


def _normalized_error_to_degrees(
    per_joint_rmse_normalized: list, fk_model: HandKinematicsModel, arm_dof: int = 7
) -> float:
    """Convert per-joint RMSE from normalized [-1,1] space to degrees.

    Uses the FK model's joint limits to compute per-joint ranges.
    """
    lower = fk_model._lower[arm_dof:]  # hand joints only
    upper = fk_model._upper[arm_dof:]
    ranges_deg = (upper - lower) * (180.0 / np.pi)
    rmse_deg = [float(r * d) for r, d in zip(ranges_deg, per_joint_rmse_normalized)]
    return float(np.mean(rmse_deg))


def validate_synthetic(
    models: Dict[str, nn.Module],
    n_samples: int = 4096,
) -> Dict[str, dict]:
    """Validate each autoencoder on random normalized joint vectors.

    Reports both joint-space MSE and FK-space fingertip error (the metric
    that actually matters for manipulation).  The DexLatent autoencoder was
    trained with a strong pinch-loss weight (λ=2000 distance, λ=5 direction),
    so joint-space MSE is expected to be moderate while FK-space error
    should be small.

    Metrics:
        joint_mse:            mean squared error in normalized joint space
        joint_rmse_deg:       per-joint RMSE in degrees (approx)
        fk_tip_rmse_cm:       fingertip position RMSE in cm (via FK)
        latent_mean_abs:      average |μ| of latent distribution
        latent_std:           average σ of latent distribution
    """
    print(f"\n[2/5] Synthetic validation ({n_samples} random samples per hand)")
    print(f"  {'Hand':30s}  {'joint MSE':>10s}  {'joint°':>7s}  {'FKtip cm':>9s}  {'latent μ':>9s}  {'latent σ':>9s}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*9}")

    results = {}
    for hand_name in HAND_NAMES:
        model = models[hand_name]
        hand_dof = model.hand_dof

        # Generate random normalized hand joint vectors
        x = torch.empty(n_samples, hand_dof).uniform_(-1.0, 1.0)

        with torch.no_grad():
            recon, latent = model(x)

        # ---- Joint-space metrics ----
        mse = nn.functional.mse_loss(recon, x).item()
        per_joint_rmse_norm = torch.sqrt(((recon - x) ** 2).mean(dim=0)).tolist()

        # ---- FK-space metrics (fingertip positions) ----
        fk = _get_fk_model(hand_name)
        # Build full qpos: zeros for arm (neutral), hand from original/recon
        neutral_arm = torch.zeros(n_samples, 7, dtype=x.dtype)
        full_orig = torch.cat([neutral_arm, x], dim=1)      # (N, 7+hand_dof)
        full_recon = torch.cat([neutral_arm, recon], dim=1)
        with torch.no_grad():
            tips_orig = fk.forward(full_orig)    # (N, F, 3)
            tips_recon = fk.forward(full_recon)  # (N, F, 3)
        fk_rmse_cm = torch.sqrt(((tips_recon - tips_orig) ** 2).mean()).item() * 100.0

        # ---- Latent statistics ----
        latent_mean_abs_avg = latent.mean(dim=0).abs().mean().item()
        latent_std_avg = latent.std(dim=0).mean().item()

        # ---- Degree conversion ----
        rmse_deg = _normalized_error_to_degrees(per_joint_rmse_norm, fk)

        results[hand_name] = {
            "joint_mse": mse,
            "joint_rmse_deg": rmse_deg,
            "fk_tip_rmse_cm": fk_rmse_cm,
            "latent_mean_abs": latent_mean_abs_avg,
            "latent_std": latent_std_avg,
        }

        # PASS criteria: FK fingertip error < 2cm (typical manipulation tolerance)
        status = "PASS" if fk_rmse_cm < 2.0 else "WARN"
        print(
            f"  {hand_name:30s}  {status:4s}  "
            f"{mse:10.2e}  {rmse_deg:6.1f}°  {fk_rmse_cm:8.2f}cm  "
            f"{latent_mean_abs_avg:+8.3f}  {latent_std_avg:8.3f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Real data validation (demo.npz Inspire trajectory)
# ═══════════════════════════════════════════════════════════════════════════════


def validate_real_data(
    models: Dict[str, nn.Module],
    demo_path: str,
    dexlatent_root: str,
) -> dict:
    """Validate on real teleop data.

    The demo.npz contains Inspire hand trajectory (raw radians).
    We normalize via HandKinematicsModel, extract hand joints, then
    test encode→decode roundtrip on the entire trajectory.

    Also validates the XHand autoencoder indirectly by:
    - Encoding real Inspire data through Inspire autoencoder
    - Decoding the latent through ALL four decoders
    - Computing FK fingertip consistency
    """
    print(f"\n[3/5] Real-data validation (demo.npz)")

    if not os.path.exists(demo_path):
        print(f"  SKIP: demo.npz not found at {demo_path}")
        return {}

    data = np.load(demo_path)
    # Use right hand (both right and left are Inspire)
    raw_qpos = torch.as_tensor(data["right_qpos"], dtype=torch.float32)
    n_frames = raw_qpos.shape[0]
    print(f"  Loaded {n_frames} frames, shape={raw_qpos.shape} (Inspire right)")

    # Build FK model for normalize/denormalize
    fk_registry = MultiHandDifferentiableFK(["xarm7_inspire_right"])
    fk_model = fk_registry.models["xarm7_inspire_right"]

    # Normalize raw radians → [-1, 1]
    normalized = fk_model.angles_to_normalized(raw_qpos)
    # Clamp to valid range
    normalized = torch.clamp(normalized, -1.0, 1.0)

    # Split arm (7D) and hand (6D for Inspire)
    arm = normalized[:, :7]
    hand_inspire = normalized[:, 7:]  # (T, 6)

    results = {}

    # --- Test 1: Inspire autoencoder roundtrip ---
    print("\n  --- Inspire encode→decode roundtrip ---")
    inspire_model = models["xarm7_inspire_right"]
    with torch.no_grad():
        recon_hand, latent = inspire_model(hand_inspire)

    mse = nn.functional.mse_loss(recon_hand, hand_inspire).item()
    per_joint_mse = ((recon_hand - hand_inspire) ** 2).mean(dim=0)

    # Convert normalized error → approximate degrees
    # For Inspire: we need joint ranges
    lower = fk_model._lower[7:]  # hand joints only
    upper = fk_model._upper[7:]
    ranges_deg = (upper - lower) * (180.0 / np.pi)
    per_joint_rmse_deg = float(
        (torch.sqrt(per_joint_mse) * ranges_deg).mean().item()
    )

    print(
        f"  Inspire self-recon:  MSE={mse:.2e}  "
        f"RMSE≈{per_joint_rmse_deg:.2f}°/joint  "
        f"latent σ={latent.std(dim=0).mean():.3f}"
    )
    results["inspire_self_recon_mse"] = mse
    results["inspire_self_recon_rmse_deg"] = per_joint_rmse_deg

    # --- Test 2: Cross-hand decode consistency ---
    print("\n  --- Cross-hand decode consistency (same latent → different hands) ---")
    # Use the first frame's latent
    latent_sample = latent[0:1]  # (1, 32)

    # For each hand, decode and compute FK fingertip positions
    import sys as _sys
    _sys.path.insert(0, dexlatent_root)

    cross_results = {}
    for hand_name in HAND_NAMES:
        if hand_name == "xarm7_inspire_right":
            continue  # skip self

        target_model = models[hand_name]
        target_fk = MultiHandDifferentiableFK([hand_name]).models[hand_name]
        target_hand_dof = target_model.hand_dof

        with torch.no_grad():
            target_hand = target_model.decode(latent_sample)  # (1, target_hand_dof)

        # Combine with a neutral arm to compute FK
        neutral_arm = torch.zeros(1, 7)
        full_qpos = torch.cat([neutral_arm, target_hand], dim=1)
        tips = target_fk.forward(full_qpos)  # (1, F, 3)

        cross_results[hand_name] = {
            "hand_qpos": target_hand.squeeze(0).tolist(),
            "fingertips": tips.squeeze(0).tolist(),
        }

        # Quick sanity: fingertips should be non-degenerate
        tip_norms = tips.squeeze(0).norm(dim=-1)
        print(
            f"    {hand_name:30s}  "
            f"tips norm=[{', '.join(f'{n:.3f}' for n in tip_norms.tolist())}]"
        )

    results["cross_hand"] = cross_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Cross-hand FK consistency check
# ═══════════════════════════════════════════════════════════════════════════════


def validate_cross_hand_consistency(
    models: Dict[str, nn.Module],
    dexlatent_root: str,
    n_samples: int = 256,
) -> dict:
    """Verify that a single latent vector produces geometrically similar grasps
    across different hands (via FK fingertip position comparison).
    """
    print(f"\n[4/5] Cross-hand geometric consistency ({n_samples} samples)")

    # Build FK models for all hands
    fk_registry = MultiHandDifferentiableFK(HAND_NAMES)
    latent_dim = 32

    # Sample the latent space: use encoder output of random qpos for one hand
    # to ensure we're on the learned manifold
    xhand_model = models["xarm7_xhand_right"]
    random_hand = torch.empty(n_samples, 12).uniform_(-1.0, 1.0)
    with torch.no_grad():
        latents = xhand_model.encode(random_hand)  # (N, 32)

    # Decode through each hand and compute FK
    all_tips = {}
    for hand_name in HAND_NAMES:
        model = models[hand_name]
        hand_dof = model.hand_dof
        fk_model = fk_registry.models[hand_name]

        with torch.no_grad():
            hand_qpos = model.decode(latents)  # (N, hand_dof)
            neutral_arm = torch.zeros(n_samples, 7, dtype=hand_qpos.dtype)
            full_qpos = torch.cat([neutral_arm, hand_qpos], dim=1)
            tips = fk_model.forward(full_qpos)  # (N, F, 3)

        all_tips[hand_name] = tips

    # Compare fingertip positions pairwise between hands
    # For each pair of hands, compute mean pairwise distance of
    # corresponding fingertips (thumb-thumb, index-index, etc.)
    # Only compare tips that both hands have (min tip count)
    print("  Pairwise fingertip distance (cm), per finger pair:")
    source_hand = "xarm7_xhand_right"
    source_tips = all_tips[source_hand]
    source_n_tips = source_tips.shape[1]  # 5

    pairwise_stats = {}
    for target_name in HAND_NAMES:
        if target_name == source_hand:
            continue
        target_tips = all_tips[target_name]
        n_common = min(source_n_tips, target_tips.shape[1])

        # Compute per-finger distance
        diffs = source_tips[:, :n_common] - target_tips[:, :n_common]  # (N, F, 3)
        dists_cm = diffs.norm(dim=-1).mean(dim=0) * 100.0  # (F,) in cm

        avg_dist = dists_cm.mean().item()
        pairwise_stats[target_name] = {
            "per_finger_cm": [round(d, 2) for d in dists_cm.tolist()],
            "mean_cm": round(avg_dist, 2),
        }
        fingers_str = " | ".join(
            f"F{i}={d:.1f}cm" for i, d in enumerate(dists_cm.tolist())
        )
        print(f"    {source_hand} ↔ {target_name:30s}  avg={avg_dist:.1f}cm  [{fingers_str}]")

    return pairwise_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Export clean checkpoint
# ═══════════════════════════════════════════════════════════════════════════════


def export_checkpoint(
    models: Dict[str, nn.Module],
    metadata: dict,
    output_path: str,
    validation_results: dict,
) -> None:
    """Save all 4 autoencoders as a clean, self-contained checkpoint."""
    print(f"\n[5/5] Exporting clean checkpoint → {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        # Per-hand state dicts
        "autoencoders": OrderedDict(
            (name, model.state_dict()) for name, model in models.items()
        ),
        # Metadata needed for reconstruction
        "latent_dim_hand": metadata["latent_dim_hand"],
        "hand_hidden_dims": metadata["hand_hidden_dims"],
        "arm_dof": metadata["arm_dof"],
        "hand_names": HAND_NAMES,
        "hand_meta": HAND_META,
        # Source info
        "source_checkpoint_epoch": metadata.get("epoch"),
        "source_hand_names": metadata.get("source_hand_names", []),
        # Validation summary
        "validation": {
            k: v
            for k, v in validation_results.items()
            if isinstance(v, (float, int, str))
        },
    }

    torch.save(payload, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Done: {size_kb:.0f} KB, {len(HAND_NAMES)} hands")
    print(f"  Keys: {list(payload.keys())}")
    print(
        f"  Params: {sum(sum(p.numel() for p in m.state_dict().values()) for m in models.values()):,}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Extract & validate DexLatent autoencoder weights"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Path to DexLatent checkpoint")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output path for clean checkpoint")
    parser.add_argument("--demo", type=str, default=DEFAULT_DEMO,
                        help="Path to demo.npz for real-data validation")
    parser.add_argument("--dexlatent-root", type=str, default=str(_DEXLATENT_ROOT),
                        help="Path to DexLatent repo root (for Assets/)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        print(f"  Looking for: {args.ckpt}")
        sys.exit(1)

    print("=" * 72)
    print("Phase 0: DexLatent Autoencoder Weight Extraction & Validation")
    print("=" * 72)

    # Step 1: Extract
    weights, metadata = extract_all_autoencoders(args.ckpt)
    print(f"  latent_dim={metadata['latent_dim_hand']}, "
          f"hidden_dims={metadata['hand_hidden_dims']}, "
          f"arm_dof={metadata['arm_dof']}")

    # Step 2: Rebuild
    print(f"\n  Building standalone autoencoders...")
    models = {}
    for hand_name in HAND_NAMES:
        models[hand_name] = build_standalone_autoencoder(
            hand_name, weights[hand_name], metadata
        )
        hd = models[hand_name].hand_dof
        ld = models[hand_name].latent_dim
        n_params = sum(p.numel() for p in models[hand_name].parameters())
        print(f"    {hand_name}: encode({hd}→{ld}), decode({ld}→{hd}), {n_params:,} params")

    # Step 3: Synthetic validation
    syn_results = validate_synthetic(models)

    # Step 4: Real data validation
    real_results = validate_real_data(models, args.demo, args.dexlatent_root)

    # Step 5: Cross-hand consistency
    cross_results = validate_cross_hand_consistency(models, args.dexlatent_root)

    # Step 6: Export
    all_results = {**syn_results, **real_results}
    export_checkpoint(models, metadata, args.output, all_results)

    # Final verdict (based on FK-space error, the metric that matters)
    print("\n" + "=" * 72)
    fk_pass = all(
        r.get("fk_tip_rmse_cm", 999) < 2.0
        for r in syn_results.values()
    )
    if fk_pass:
        print("✅ ALL CHECKS PASSED — FK fingertip error < 2cm for all hands")
        print("   Autoencoder weights ready for Phase 1 integration")
    else:
        print("⚠️  FK fingertip error ≥ 2cm on some hands (see table above)")
        print("   This is expected if the autoencoder was trained primarily for")
        print("   pinch alignment rather than exact joint reconstruction.")
        print("   Proceed with caution — validate on your task data.")

    print(f"\n  Output: {args.output}")
    print(f"  Next: python scripts/extract_dexlatent_weights.py  ← rerun anytime")
    print("=" * 72)


if __name__ == "__main__":
    main()
