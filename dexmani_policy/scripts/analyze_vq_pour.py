#!/usr/bin/env python3
"""Deep analysis: why DQ-RISE fails on pour but succeeds on pick_bottle."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import zarr
from dexmani_policy.agents.vq_hand import VqVaeHand, CodebookManager


def load_vqvae(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vqvae = VqVaeHand(hand_dim=10, num_groups=2, codebook_size=4).to(device)
    vqvae.load_state_dict(ckpt['model_state_dict'])
    vqvae.eval()
    return vqvae


def analyze_vq_reconstruction(vqvae, mgr, zarr_path, device='cuda', max_samples=50000):
    z = zarr.open(zarr_path, 'r')
    actions = z['data']['action'][:max_samples]
    hand_actions = actions[:, 9:19].astype(np.float32)

    # Min-max normalize to [-1, 1]
    h_min = hand_actions.min(axis=0)
    h_max = hand_actions.max(axis=0)
    h_range = h_max - h_min
    h_range[h_range < 1e-8] = 1.0
    hand_norm = (hand_actions - h_min) / h_range * 2.0 - 1.0
    hand_t = torch.from_numpy(hand_norm).to(device)

    # VQ-VAE reconstruction
    errors = []
    all_indices = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(hand_t), batch_size):
            batch = hand_t[i:i + batch_size]
            x = batch / vqvae.act_scale
            z_e = vqvae.encoder(x)
            z_q, idx, _ = vqvae.vq_layer(z_e.unsqueeze(1))
            z_q = z_q.squeeze(1)
            recon = vqvae.decoder(z_q) * vqvae.act_scale
            per_sample_l2 = (batch - recon).pow(2).sum(dim=-1).sqrt()
            errors.append(per_sample_l2.cpu())
            all_indices.append(idx.squeeze(1).cpu())  # (B, num_groups)

    errors = torch.cat(errors).numpy()
    all_indices = torch.cat(all_indices).numpy()  # (N, 2)

    # Codebook nearest-neighbor distance (training label error)
    poses = mgr.sorted_hand_poses  # (16, 10)
    code_dists = []
    for i in range(0, len(hand_norm), batch_size):
        batch = hand_norm[i:i + batch_size]
        batch_t = torch.from_numpy(batch)
        diff = batch_t.unsqueeze(1) - poses.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        code_dists.append(dist2.min(dim=-1).values.sqrt().numpy())
    code_dists = np.concatenate(code_dists)

    return {
        'vq_recon_l2': errors,
        'codebook_nn_l2': code_dists,
        'vq_indices': all_indices,
        'hand_actions_raw': hand_actions,
        'hand_actions_norm': hand_norm,
        'h_min': h_min,
        'h_max': h_max,
    }


def main():
    device = 'cuda'

    # Load VQ-VAEs
    print('Loading VQ-VAEs...')
    vqvae_pour = load_vqvae('experiments/dqrise/pour/vq_hand/vqvae_hand_last.pt', device)
    vqvae_bottle = load_vqvae('experiments/dqrise/pick_bottle/vq_hand/vqvae_hand_last.pt', device)

    # Load codebook managers (extract_from_vqvae is @staticmethod, returns new obj)
    mgr_pour = CodebookManager.extract_from_vqvae(vqvae_pour)
    mgr_pour.reindex_by_pca(vqvae_pour)

    mgr_bottle = CodebookManager.extract_from_vqvae(vqvae_bottle)
    mgr_bottle.reindex_by_pca(vqvae_bottle)

    # ============================================================
    # Section 1: Codebook quality analysis
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 1: Codebook Quality')
    print('=' * 60)

    for name, mgr in [('Pour', mgr_pour), ('Pick Bottle', mgr_bottle)]:
        poses = mgr.sorted_hand_poses
        # PCA ratio
        pca_ratio = poses.std(dim=0).max().item() / (poses.std(dim=0).min().item() + 1e-8)
        # Pairwise distances
        diffs = poses.unsqueeze(0) - poses.unsqueeze(1)
        dists = diffs.norm(dim=-1)
        mask = ~torch.eye(16, dtype=bool)
        # Coverage: std of all poses vs std of full data
        print(f'\n{name}:')
        print(f'  PCA ratio: {pca_ratio:.2f}x')
        print(f'  Pairwise L2: min={dists[mask].min():.4f}, mean={dists[mask].mean():.4f}, max={dists[mask].max():.4f}')
        # Per-dimension std
        for d in range(10):
            print(f'  Dim {d}: mean={poses[:, d].mean():+.4f}, std={poses[:, d].std():.4f}')

    # ============================================================
    # Section 2: Reconstruction error
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 2: VQ Reconstruction Error on Full Dataset')
    print('=' * 60)

    pour_r = analyze_vq_reconstruction(vqvae_pour, mgr_pour,
                                        'robot_data/pour.zarr', device)
    bottle_r = analyze_vq_reconstruction(vqvae_bottle, mgr_bottle,
                                          'robot_data/pick_bottle.zarr', device)

    for name, r in [('Pour', pour_r), ('Pick Bottle', bottle_r)]:
        e = r['vq_recon_l2']
        c = r['codebook_nn_l2']
        print(f'\n{name}:')
        print(f'  VQ-VAE recon L2:')
        print(f'    mean={e.mean():.4f}, std={e.std():.4f}')
        print(f'    p50={np.percentile(e, 50):.4f}, p90={np.percentile(e, 90):.4f}')
        print(f'    p95={np.percentile(e, 95):.4f}, p99={np.percentile(e, 99):.4f}')
        print(f'  Codebook NN L2 (quantization error):')
        print(f'    mean={c.mean():.4f}, std={c.std():.4f}')
        print(f'    p50={np.percentile(c, 50):.4f}, p90={np.percentile(c, 90):.4f}')
        print(f'    p95={np.percentile(c, 95):.4f}, p99={np.percentile(c, 99):.4f}')
        print(f'    max={c.max():.4f}')

    # ============================================================
    # Section 3: Code usage distribution
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 3: Code Usage Distribution')
    print('=' * 60)

    for name, r in [('Pour', pour_r), ('Pick Bottle', bottle_r)]:
        indices = r['vq_indices']  # (N, 2)
        # Combine group indices into single code ID: code_id = idx[0] * 4 + idx[1]
        code_ids = indices[:, 0] * 4 + indices[:, 1]
        usage = np.bincount(code_ids, minlength=16)
        usage_frac = usage / usage.sum()
        print(f'\n{name}:')
        print(f'  Code frequencies (sorted):')
        sorted_idx = np.argsort(-usage)
        for rank, cid in enumerate(sorted_idx):
            g0, g1 = cid // 4, cid % 4
            bar = '█' * int(usage_frac[cid] * 100)
            print(f'    #{rank}: code({g0},{g1}) = {usage[cid]:6d} ({usage_frac[cid]*100:5.1f}%) {bar}')
        # Entropy
        ent = -np.sum(usage_frac[usage_frac > 0] * np.log(usage_frac[usage_frac > 0]))
        max_ent = np.log(16)
        print(f'  Normalized entropy: {ent/max_ent:.4f} (1.0 = uniform)')

    # ============================================================
    # Section 4: Per-action-dimension reconstruction analysis
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 4: Per-Dimension Action Statistics')
    print('=' * 60)

    for name, r in [('Pour', pour_r), ('Pick Bottle', bottle_r)]:
        raw = r['hand_actions_raw']
        print(f'\n{name} (raw action space):')
        for d in range(10):
            print(f'  Dim {d}: mean={raw[:, d].mean():+.4f}, std={raw[:, d].std():.4f}, '
                  f'range=[{raw[:, d].min():+.4f}, {raw[:, d].max():+.4f}]')

    # ============================================================
    # Section 5: Temporal analysis - code transitions
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 5: Temporal Code Transition Analysis')
    print('=' * 60)

    # Pour has episodes; let's analyze per-episode code transitions
    z_pour = zarr.open('robot_data/pour.zarr', 'r')
    z_bottle = zarr.open('robot_data/pick_bottle.zarr', 'r')

    ep_ends_pour = z_pour['meta']['episode_ends'][:]
    ep_ends_bottle = z_bottle['meta']['episode_ends'][:]

    for name, z_data, ep_ends, vqvae, mgr in [
        ('Pour', z_pour, ep_ends_pour, vqvae_pour, mgr_pour),
        ('Pick Bottle', z_bottle, ep_ends_bottle, vqvae_bottle, mgr_bottle)
    ]:
        actions = z_data['data']['action'][:]
        hand = actions[:, 9:19].astype(np.float32)

        # Normalize
        h_min = hand.min(axis=0)
        h_max = hand.max(axis=0)
        h_range = h_max - h_min
        h_range[h_range < 1e-8] = 1.0
        hand_norm = (hand - h_min) / h_range * 2.0 - 1.0

        # Get continuous indices for each frame
        hand_t = torch.from_numpy(hand_norm).float()
        cont_idx = mgr.hand_pose_to_continuous_index(hand_t).squeeze(-1).numpy()  # (N,)

        # Per-episode analysis
        n_episodes = len(ep_ends)
        ep_start = 0
        ep_unique_codes = []
        ep_code_transitions = []  # fraction of frames where code changes
        ep_code_std = []  # std of continuous index within episode
        ep_cont_idx_range = []  # range of continuous index

        for ep_end in ep_ends:
            ep_cont = cont_idx[ep_start:ep_end]
            # Number of unique discrete codes in this episode
            discrete = ((ep_cont + 1) / 2 * 15).round().clip(0, 15).astype(int)
            ep_unique_codes.append(len(np.unique(discrete)))
            # Transition rate
            if len(discrete) > 1:
                transitions = (discrete[1:] != discrete[:-1]).sum()
                ep_code_transitions.append(transitions / (len(discrete) - 1))
            else:
                ep_code_transitions.append(0.0)
            # Std and range of continuous index
            ep_code_std.append(ep_cont.std())
            ep_cont_idx_range.append(ep_cont.max() - ep_cont.min())
            ep_start = ep_end

        ep_unique_codes = np.array(ep_unique_codes)
        ep_code_transitions = np.array(ep_code_transitions)
        ep_code_std = np.array(ep_code_std)
        ep_cont_idx_range = np.array(ep_cont_idx_range)

        print(f'\n{name} ({n_episodes} episodes):')
        print(f'  Unique codes per episode:  mean={ep_unique_codes.mean():.1f}, '
              f'min={ep_unique_codes.min()}, max={ep_unique_codes.max()}')
        print(f'  Code transition rate:      mean={ep_code_transitions.mean():.3f}, '
              f'std={ep_code_transitions.std():.3f}')
        print(f'  Cont. index std per ep:    mean={ep_code_std.mean():.4f}, '
              f'max={ep_code_std.max():.4f}')
        print(f'  Cont. index range per ep:  mean={ep_cont_idx_range.mean():.4f}, '
              f'max={ep_cont_idx_range.max():.4f}')

    # ============================================================
    # Section 6: Codebook coverage gap analysis
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 6: Codebook Coverage Gap')
    print('=' * 60)

    for name, r, mgr in [('Pour', pour_r, mgr_pour), ('Pick Bottle', bottle_r, mgr_bottle)]:
        hand_norm = r['hand_actions_norm']
        code_poses = mgr.sorted_hand_poses.numpy()

        # For each dimension, check if codebook covers the full data range
        for d in range(10):
            data_min, data_max = hand_norm[:, d].min(), hand_norm[:, d].max()
            code_min, code_max = code_poses[:, d].min(), code_poses[:, d].max()
            data_range = data_max - data_min
            coverage_min = (code_min - data_min) / (data_range + 1e-8) * 100
            coverage_max = (data_max - code_max) / (data_range + 1e-8) * 100
            gap = max(0, -coverage_min) + max(0, -coverage_max)
            if gap > 5:  # More than 5% gap
                print(f'  {name} dim {d}: data=[{data_min:+.3f}, {data_max:+.3f}], '
                      f'codebook=[{code_min:+.3f}, {code_max:+.3f}], gap={gap:.1f}%')

    # ============================================================
    # Section 7: Loss landscape analysis
    # ============================================================
    print('\n' + '=' * 60)
    print('SECTION 7: VQ Information Loss Relative to Action Magnitude')
    print('=' * 60)

    for name, r in [('Pour', pour_r), ('Pick Bottle', bottle_r)]:
        hand_raw = r['hand_actions_raw']
        hand_norm = r['hand_actions_norm']
        vq_err = r['vq_recon_l2']
        nn_err = r['codebook_nn_l2']

        # Per-sample action magnitude vs reconstruction error
        action_magnitude = np.linalg.norm(hand_norm, axis=-1)  # (N,)

        # Relative error: VQ recon error / action magnitude
        rel_vq_err = vq_err / (action_magnitude + 1e-8)
        rel_nn_err = nn_err / (action_magnitude + 1e-8)

        print(f'\n{name}:')
        print(f'  Action magnitude:        mean={action_magnitude.mean():.4f}, std={action_magnitude.std():.4f}')
        print(f'  Relative VQ recon error: mean={rel_vq_err.mean():.4f}, p95={np.percentile(rel_vq_err, 95):.4f}')
        print(f'  Relative NN quant error: mean={rel_nn_err.mean():.4f}, p95={np.percentile(rel_nn_err, 95):.4f}')

        # Correlation: does error increase with action magnitude?
        corr_vq = np.corrcoef(action_magnitude, vq_err)[0, 1]
        corr_nn = np.corrcoef(action_magnitude, nn_err)[0, 1]
        print(f'  Corr(magnitude, VQ err): {corr_vq:.4f}')
        print(f'  Corr(magnitude, NN err): {corr_nn:.4f}')


if __name__ == '__main__':
    main()
