#!/usr/bin/env python3
"""Extract codebook from a trained VQ-VAE checkpoint and save as PCA-sorted .npz.

This is step 2 of the DQ-RISE pipeline:
  1. Train VQ-VAE        → train_vq_hand.py
  2. Extract codebook     → extract_codebook.py  (this script)
  3. Train DQ-RISE policy → train.py dqrise

The output .npz file is consumed by DQRISEAgent via the ``codebook_path``
config field in dqrise.yaml.

Usage:
    python scripts/extract_codebook.py \\
        --checkpoint experiments/vq_hand/pick_apple_messy/vqvae_hand_last.pt \\
        --output robot_data/sorted_hand_poses_pick_apple_messy.npz

    # With explicit hand parameters (if checkpoint doesn't store them):
    python scripts/extract_codebook.py \\
        --checkpoint vqvae.pt --output codebook.npz \\
        --hand_dim 12 --num_groups 2 --codebook_size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dexmani_policy.agents.vq_hand import VqVaeHand, CodebookManager


def extract_codebook(
    checkpoint_path: str,
    output_path: str,
    hand_dim: int | None = None,
    num_groups: int | None = None,
    codebook_size: int | None = None,
    device: str = "cuda",
) -> CodebookManager:
    """Load VQ-VAE checkpoint, extract codebook, PCA re-index, save.

    Parameters are auto-detected from the checkpoint's saved args when
    available.  CLI overrides take precedence for flexibility.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Resolve model params: CLI > checkpoint args > defaults ───────────
    saved_args = ckpt.get("args", {})
    # Normalise to dict (checkpoints save either Namespace or dict)
    if hasattr(saved_args, "hand_dim"):
        saved_args = vars(saved_args)

    _hand_dim = hand_dim or saved_args.get("hand_dim")
    _num_groups = num_groups or saved_args.get("num_groups")
    _codebook_size = codebook_size or saved_args.get("codebook_size")
    _num_layers = saved_args.get("num_layers", 1)

    if _hand_dim is None:
        raise ValueError(
            "hand_dim not found in checkpoint args and not provided via CLI. "
            "Pass --hand_dim explicitly."
        )

    # ── Reconstruct & load ──────────────────────────────────────────────
    # loss_weight is required by VqVaeHand but unused during extraction
    # (only forward() uses it for L1 reconstruction loss).  Pass a dummy
    # all-ones vector matching hand_dim.
    _loss_weight = [1.0] * _hand_dim
    vqvae = VqVaeHand(
        hand_dim=_hand_dim,
        loss_weight=_loss_weight,
        num_groups=_num_groups,
        codebook_size=_codebook_size,
        num_layers=_num_layers,
    ).to(device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    # ── Extract, re-index, save ─────────────────────────────────────────
    mgr = CodebookManager.extract_from_vqvae(vqvae)
    poses = mgr.reindex_by_pca(vqvae)
    mgr.build_per_group_codebooks(vqvae)  # per-group sorted poses for multi-index prediction
    mgr.save(output_path)

    # Report
    all_poses = torch.from_numpy(poses).float()
    pca_ratio = (
        all_poses.std(dim=0).max().item() / all_poses.std(dim=0).min().item()
    )
    print(f"Codebook extracted: {len(poses)} hand poses, PCA ratio = {pca_ratio:.2f}x")
    print(f"Saved to: {output_path}")
    return mgr


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract PCA-sorted codebook from a trained VQ-VAE checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checkpoint vqvae_hand_last.pt --output codebook.npz
  %(prog)s --checkpoint vqvae.pt --output codebook.npz --hand_dim 12
  %(prog)s --checkpoint vqvae.pt --output codebook.npz --device cpu
        """,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to VQ-VAE checkpoint (.pt)")
    ap.add_argument("--output", required=True, help="Output path for sorted hand poses (.npz)")
    ap.add_argument("--hand_dim", type=int, default=None,
                    help="Hand joint dimensionality (auto-detected from checkpoint if omitted)")
    ap.add_argument("--num_groups", type=int, default=None,
                    help="Number of residual VQ groups")
    ap.add_argument("--codebook_size", type=int, default=None,
                    help="Codes per VQ group")
    ap.add_argument("--device", default="cuda", help="Device for model inference")
    args = ap.parse_args()

    extract_codebook(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        hand_dim=args.hand_dim,
        num_groups=args.num_groups,
        codebook_size=args.codebook_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
