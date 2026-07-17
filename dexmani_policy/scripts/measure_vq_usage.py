#!/usr/bin/env python3
"""Measure vq_idx_used kill-gate metric for a VQ-VAE checkpoint on hand data.

vq_idx_used = number of the 16 PCA-sorted codebook combos that the training
hand data actually maps to (via L2 nearest-neighbor). Kill-gate: >= 12/16.
Also reports recon_mse for the reconstruction-fidelity check.
"""
import argparse
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import numpy as np
import torch
from dexmani_policy.agents.vq_hand import VqVaeHand, CodebookManager
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.common.normalizer import LinearNormalizer


def measure(checkpoint_path, zarr_path, action_key='action_ee', tcp_dim=9):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = ckpt.get('args', {})
    if hasattr(args, 'hand_dim'):
        args = vars(args)
    hand_dim = args.get('hand_dim', 12)
    num_groups = args.get('num_groups', 2)
    codebook_size = args.get('codebook_size', 4)
    num_layers = args.get('num_layers', 2)

    vqvae = VqVaeHand(
        hand_dim=hand_dim, loss_weight=[1.0] * hand_dim,
        num_groups=num_groups, codebook_size=codebook_size, num_layers=num_layers,
    )
    vqvae.load_state_dict(ckpt['model_state_dict'])
    vqvae.eval()

    # Load + normalise hand data (same as train_vq_hand.py)
    buffer = ReplayBuffer.copy_from_path(zarr_path, keys=[action_key])
    hand = buffer[action_key][:, tcp_dim:]
    norm = LinearNormalizer()
    norm.fit(data={'hand': hand}, mode='limits', range_eps=1e-4)
    hand_n = norm['hand'].normalize(hand).numpy().astype(np.float32)

    # Build codebook (PCA-sorted 16 combos)
    mgr = CodebookManager.extract_from_vqvae(vqvae)
    mgr.reindex_by_pca(vqvae)

    # Map every hand pose → nearest of 16 sorted codes (L2 in raw space)
    hand_t = torch.from_numpy(hand_n)
    idx = mgr.hand_pose_to_continuous_index(hand_t)          # (N, 1) in [-1,1]
    num_codes = mgr.get_num_codes()
    discrete = ((idx.squeeze(-1) + 1.0) / 2.0 * (num_codes - 1)).round().long()
    counts = torch.bincount(discrete, minlength=num_codes)
    used = int((counts > 0).sum())
    used_1pct = int((counts.float() / counts.sum() > 0.01).sum())

    # recon_mse (scaled space, unweighted) over a sample
    with torch.no_grad():
        sub = hand_t[torch.randperm(len(hand_t))[:5000]]
        enc_loss, vq_loss, _, recon_mse = vqvae(sub)

    return dict(
        used=used, used_1pct=used_1pct, num_codes=num_codes,
        recon_mse=float(recon_mse), counts=counts.tolist(),
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Measure vq_idx_used kill-gate metric for a VQ-VAE checkpoint',
    )
    ap.add_argument('--checkpoint', required=True, help='Path to VQ-VAE checkpoint (.pt)')
    ap.add_argument('--zarr', default='robot_data/pour.zarr',
                    help='Path to Zarr dataset (default: robot_data/pour.zarr)')
    args = ap.parse_args()
    r = measure(args.checkpoint, args.zarr)
    print(f"vq_idx_used = {r['used']}/{r['num_codes']}  (>1%: {r['used_1pct']})")
    print(f"recon_mse   = {r['recon_mse']:.5f}")
    print(f"histogram   = {r['counts']}")
