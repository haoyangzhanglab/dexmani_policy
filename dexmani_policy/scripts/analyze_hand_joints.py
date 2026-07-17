#!/usr/bin/env python3
"""
Analyze XHand joint → fingertip mapping from Zarr data.

Infers which hand joint indices drive which fingers by computing
correlations between hand actions and fingertip positions.
Also computes per-joint statistics to recommend VQ-VAE loss weights.

Usage:
    python scripts/analyze_hand_joints.py                          # all tasks
    python scripts/analyze_hand_joints.py --task pick_apple_messy  # single task
    python scripts/analyze_hand_joints.py --action_key action      # joint space
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

def load_zarr_data(zarr_path: str) -> dict:
    """Load hand actions and fingertip points from a Zarr dataset."""
    import zarr
    root = zarr.open(zarr_path, mode='r')

    result = {}
    for key in ['action', 'action_ee', 'fingertip_points', 'joint_state']:
        if key in root['data']:
            result[key] = root['data'][key][:]
            print(f"  {key}: {result[key].shape}  "
                  f"range=[{result[key].min(axis=0).round(3)}, {result[key].max(axis=0).round(3)}]")
    return result


def compute_fingertip_displacement(fingertips: np.ndarray) -> np.ndarray:
    """Convert absolute fingertip positions to frame-to-frame displacements.

    Args:
        fingertips: (N, 15) — 5 fingertips × 3 xyz

    Returns:
        disp: (N-1, 15) — per-frame displacement of each fingertip coordinate
    """
    return np.diff(fingertips, axis=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_hand_fingertip_correlation(
    hand_actions: np.ndarray,
    fingertips: np.ndarray,
) -> np.ndarray:
    """Compute Pearson r between each hand joint and each fingertip's motion.

    Args:
        hand_actions: (N, hand_dim) — hand joint actions
        fingertips:   (N, 15)       — 5 fingertips × 3 xyz

    Returns:
        corr_matrix: (hand_dim, 5) — r(joint_i, fingertip_f)
    """
    hand_dim = hand_actions.shape[1]
    hand_disp = np.diff(hand_actions, axis=0)       # (N-1, hand_dim)
    ft_disp = compute_fingertip_displacement(fingertips)  # (N-1, 15)

    # Per-fingertip summary: L2 norm of xyz displacement per fingertip
    # (N-1, 15) → (N-1, 5)
    ft_motion = np.stack([
        np.linalg.norm(ft_disp[:, f*3:(f+1)*3], axis=1)
        for f in range(5)
    ], axis=1)  # (N-1, 5)

    corr = np.zeros((hand_dim, 5), dtype=np.float32)
    for j in range(hand_dim):
        for f in range(5):
            # Pearson r; guard against constant signals
            hj, ff = hand_disp[:, j], ft_motion[:, f]
            hj_std, ff_std = hj.std(), ff.std()
            if hj_std < 1e-10 or ff_std < 1e-10:
                corr[j, f] = 0.0
            else:
                corr[j, f] = np.corrcoef(hj, ff)[0, 1]

    return corr


def compute_joint_statistics(hand_actions: np.ndarray) -> dict:
    """Compute per-joint usage statistics.

    Returns dict with keys: std, range, mean_abs, variance, quantile_95
    """
    return {
        'std': hand_actions.std(axis=0),
        'range': hand_actions.max(axis=0) - hand_actions.min(axis=0),
        'mean_abs': np.abs(hand_actions).mean(axis=0),
        'variance': hand_actions.var(axis=0),
        'mean': hand_actions.mean(axis=0),
    }


def suggest_loss_weights(stats: dict, corr_matrix: np.ndarray) -> dict:
    """Generate loss-weight suggestions from data.

    Returns a dict with multiple candidate weight vectors.
    """
    hand_dim = len(stats['std'])

    # --- Candidate 1: variance-proportional (data-driven) ---
    var = stats['variance']
    w_var = var / var.mean()              # normalize → mean ≈ 1
    w_var = np.clip(w_var, 0.3, 2.0)      # clamp extremes

    # --- Candidate 2: std-proportional ---
    std = stats['std']
    w_std = std / std.mean()
    w_std = np.clip(w_std, 0.3, 2.0)

    # --- Candidate 3: activity-weighted (combines std + correlation diversity) ---
    # Joints that correlate strongly with MANY fingertips are "root" joints
    # (e.g. thumb base); joints that correlate with ONE fingertip are distal.
    # Both are important — we use max correlation per joint as activity signal.
    max_corr_per_joint = np.abs(corr_matrix).max(axis=1)  # (hand_dim,)
    n_fingers_correlated = (np.abs(corr_matrix) > 0.1).sum(axis=1)  # (hand_dim,)
    w_activity = (std / std.mean()) * (0.5 + 0.5 * max_corr_per_joint)
    w_activity = np.clip(w_activity, 0.3, 2.0)

    return {
        'uniform': np.ones(hand_dim, dtype=np.float32),
        'variance': w_var.astype(np.float32),
        'std': w_std.astype(np.float32),
        'activity': w_activity.astype(np.float32),
    }


def map_joints_to_fingers(corr_matrix: np.ndarray) -> dict:
    """Infer per-joint → finger mapping from correlation matrix.

    Each joint is assigned to the finger it correlates with most strongly.
    """
    hand_dim = corr_matrix.shape[0]
    best_finger = np.argmax(np.abs(corr_matrix), axis=1)  # (hand_dim,)
    best_corr = np.array([
        corr_matrix[j, best_finger[j]] for j in range(hand_dim)
    ])

    # Group joints by finger
    groups = {f: [] for f in range(5)}
    for j in range(hand_dim):
        f = best_finger[j]
        groups[f].append((j, best_corr[j]))

    # Sort each group by correlation strength (proximal → distal heuristic)
    for f in range(5):
        groups[f].sort(key=lambda x: -abs(x[1]))

    return {
        'best_finger': best_finger,       # (hand_dim,)  finger index per joint
        'best_corr': best_corr,           # (hand_dim,)  correlation strength
        'groups': groups,                  # per-finger joint lists
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(
    corr_matrix: np.ndarray,
    stats: dict,
    weights: dict,
    mapping: dict,
    task_name: str,
    save_path: str | None = None,
):
    """Generate diagnostic plots."""
    hand_dim = corr_matrix.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'XHand Joint Analysis — {task_name}', fontsize=14, fontweight='bold')

    # --- (0,0): Correlation heatmap ---
    ax = axes[0, 0]
    im = ax.imshow(corr_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Hand Joint Index')
    ax.set_ylabel('Fingertip')
    ax.set_yticks(range(5))
    ax.set_yticklabels(FINGER_NAMES)
    ax.set_title('Joint ↔ Fingertip Correlation (r)')
    plt.colorbar(im, ax=ax)
    # Annotate strongest correlation per joint
    for j in range(hand_dim):
        best_f = np.argmax(np.abs(corr_matrix[j]))
        val = corr_matrix[j, best_f]
        if abs(val) > 0.05:
            ax.annotate(f'{val:.2f}', (j, best_f), fontsize=7,
                        ha='center', va='bottom',
                        color='black' if abs(val) < 0.3 else 'white')

    # --- (0,1): Per-joint std ---
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.arange(hand_dim) % 10)
    ax.bar(range(hand_dim), stats['std'], color=colors)
    ax.set_xlabel('Hand Joint Index')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Per-Joint Activity (std)')

    # --- (0,2): Per-joint range ---
    ax = axes[0, 2]
    ax.bar(range(hand_dim), stats['range'], color=colors)
    ax.set_xlabel('Hand Joint Index')
    ax.set_ylabel('Range (max-min)')
    ax.set_title('Per-Joint Motion Range')

    # --- (1,0): Suggested loss weights ---
    ax = axes[1, 0]
    x = np.arange(hand_dim)
    width = 0.2
    for i, (name, w) in enumerate(weights.items()):
        ax.bar(x + i * width, w, width, label=name, alpha=0.8)
    ax.set_xlabel('Hand Joint Index')
    ax.set_ylabel('Weight')
    ax.set_title('Candidate Loss Weights')
    ax.legend(fontsize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # --- (1,1): Joint → Finger mapping ---
    ax = axes[1, 1]
    best_finger = mapping['best_finger']
    finger_colors = plt.cm.Set2(np.arange(5))
    for j in range(hand_dim):
        f = best_finger[j]
        ax.barh(j, abs(mapping['best_corr'][j]), color=finger_colors[f], alpha=0.8)
        ax.text(abs(mapping['best_corr'][j]) + 0.01, j,
                FINGER_NAMES[f][:3], va='center', fontsize=8)
    ax.set_yticks(range(hand_dim))
    ax.set_ylabel('Hand Joint Index')
    ax.set_xlabel('Max |Correlation| with Best Finger')
    ax.set_title('Inferred Joint → Finger Mapping')
    ax.invert_yaxis()

    # --- (1,2): Text summary ---
    ax = axes[1, 2]
    ax.axis('off')
    lines = ["=== Per-Finger Joint Groups ==="]
    for f in range(5):
        joints = mapping['groups'][f]
        if joints:
            idx_str = ', '.join(f'J{j}(r={c:.2f})' for j, c in joints)
        else:
            idx_str = '(none)'
        lines.append(f"{FINGER_NAMES[f]:>7s}: [{idx_str}]")

    lines.append("")
    lines.append("=== Recommended Loss Weights ===")
    w_best = weights['activity']
    lines.append("activity-based:")
    lines.append('[' + ', '.join(f'{w:.2f}' for w in w_best) + ']')

    lines.append("")
    lines.append("=== Per-Joint Stats (sorted by std) ===")
    order = np.argsort(-stats['std'])
    for rank, j in enumerate(order):
        lines.append(
            f"  J{j:2d}: std={stats['std'][j]:.4f}  "
            f"range={stats['range'][j]:.4f}  "
            f"finger={FINGER_NAMES[best_finger[j]]}"
        )

    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved → {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze XHand joint → fingertip mapping from Zarr data',
    )
    parser.add_argument(
        '--task', type=str, default=None,
        help='Task name (e.g. pick_apple_messy). Default: all tasks.',
    )
    parser.add_argument(
        '--action_key', type=str, default='action',
        choices=['action', 'action_ee'],
        help='Which action key to use (default: action for 19-dim, 7 arm + 12 hand)',
    )
    parser.add_argument(
        '--tcp_dim', type=int, default=None,
        help='TCP dimension (default: 7 for action, 9 for action_ee)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='scripts/outputs/hand_analysis',
        help='Output directory for plots',
    )
    args = parser.parse_args()

    # Resolve TCP dim
    tcp_dim = args.tcp_dim
    if tcp_dim is None:
        tcp_dim = 7 if args.action_key == 'action' else 9
    print(f"Using action_key={args.action_key}, tcp_dim={tcp_dim}")

    # Find Zarr paths
    data_root = Path(__file__).parent.parent / 'robot_data'
    if args.task:
        zarr_paths = [data_root / f'{args.task}.zarr']
    else:
        zarr_paths = sorted(data_root.glob('*.zarr'))

    zarr_paths = [p for p in zarr_paths if p.exists()]
    if not zarr_paths:
        print(f"No Zarr datasets found in {data_root}")
        sys.exit(1)

    print(f"\nAnalyzing {len(zarr_paths)} task(s):")
    for p in zarr_paths:
        print(f"  - {p.name}")

    # Accumulate across tasks
    all_hand = []
    all_ft = []

    for zarr_path in zarr_paths:
        task_name = zarr_path.name.replace('.zarr', '')
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")

        data = load_zarr_data(str(zarr_path))

        # Extract hand actions
        action_key = args.action_key
        if action_key not in data:
            print(f"  SKIP: '{action_key}' not in dataset (available: {list(data.keys())})")
            continue

        actions = data[action_key]
        hand_actions = actions[:, tcp_dim:]  # (N, hand_dim)
        hand_dim = hand_actions.shape[1]

        if 'fingertip_points' not in data:
            print(f"  SKIP: no fingertip_points in dataset")
            continue

        fingertips = data['fingertip_points']

        print(f"\n  Hand dim: {hand_dim}  Fingertips: {fingertips.shape[1]}")
        print(f"  Hand action range: [{hand_actions.min():.3f}, {hand_actions.max():.3f}]")

        all_hand.append(hand_actions)
        all_ft.append(fingertips)

    if not all_hand:
        print("\nNo valid data found!")
        sys.exit(1)

    # Concatenate all tasks
    hand_all = np.concatenate(all_hand, axis=0)
    ft_all = np.concatenate(all_ft, axis=0)
    hand_dim = hand_all.shape[1]
    print(f"\n{'='*50}")
    print(f"Combined: {hand_all.shape[0]} frames, {hand_dim} hand joints")
    print(f"{'='*50}")

    # ── Analysis ──
    print("\n[1/3] Computing joint-fingertip correlations ...")
    corr_matrix = analyze_hand_fingertip_correlation(hand_all, ft_all)

    print("\n[2/3] Computing per-joint statistics ...")
    stats = compute_joint_statistics(hand_all)

    print("\n[3/3] Inferring joint → finger mapping ...")
    mapping = map_joints_to_fingers(corr_matrix)
    weights = suggest_loss_weights(stats, corr_matrix)

    # ── Print results ──
    print(f"\n{'─'*60}")
    print("Correlation Matrix (joint × fingertip)")
    print(f"{'─'*60}")
    header = "Joint  " + "  ".join(f"{n:>7s}" for n in FINGER_NAMES) + "  Best"
    print(header)
    print("-" * len(header))
    for j in range(hand_dim):
        row = "  ".join(f"{corr_matrix[j, f]:7.3f}" for f in range(5))
        best_f = mapping['best_finger'][j]
        best_r = mapping['best_corr'][j]
        print(f"  J{j:2d}  {row}  → {FINGER_NAMES[best_f]:>7s} (r={best_r:.3f})")

    print(f"\n{'─'*60}")
    print("Per-Finger Joint Groups (from correlation)")
    print(f"{'─'*60}")
    total_assigned = 0
    for f in range(5):
        joints = mapping['groups'][f]
        total_assigned += len(joints)
        if joints:
            items = ', '.join(f'J{j}(r={c:.3f})' for j, c in joints)
        else:
            items = '(none)'
        print(f"  {FINGER_NAMES[f]:>7s} ({len(joints)} joints): {items}")
    print(f"  Total assigned: {total_assigned}/{hand_dim}")

    print(f"\n{'─'*60}")
    print("Per-Joint Statistics")
    print(f"{'─'*60}")
    order = np.argsort(-stats['std'])
    print(f"  {'Idx':>4s}  {'std':>8s}  {'range':>8s}  {'mean_abs':>8s}  {'mean':>8s}  Finger")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for j in order:
        print(f"  {j:4d}  {stats['std'][j]:8.4f}  {stats['range'][j]:8.4f}  "
              f"{stats['mean_abs'][j]:8.4f}  {stats['mean'][j]:8.4f}  "
              f"{FINGER_NAMES[mapping['best_finger'][j]]}")

    print(f"\n{'─'*60}")
    print("Candidate Loss Weights (for --loss_weight)")
    print(f"{'─'*60}")
    for name, w in weights.items():
        w_str = ', '.join(f'{v:.2f}' for v in w)
        print(f"  {name:>10s}: [{w_str}]")

    print(f"\n{'─'*60}")
    print("Recommended CLI snippet:")
    print(f"{'─'*60}")
    w_best = weights['activity']
    w_str = ','.join(f'{v:.2f}' for v in w_best)
    print(f"  --loss_weight {w_str}")

    # ── Plot ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_label = args.task or 'all_tasks'
    save_path = output_dir / f'hand_joint_analysis_{task_label}.png'

    print(f"\nGenerating plot ...")
    plot_results(corr_matrix, stats, weights, mapping, task_label, str(save_path))

    # ── Save raw data ──
    npz_path = output_dir / f'hand_joint_analysis_{task_label}.npz'
    np.savez(
        str(npz_path),
        corr_matrix=corr_matrix,
        **{f'stat_{k}': v for k, v in stats.items()},
        best_finger=mapping['best_finger'],
        best_corr=mapping['best_corr'],
        **{f'weight_{k}': v for k, v in weights.items()},
    )
    print(f"  Raw data saved → {npz_path}")

    print(f"\n✓ Done. Review the plot to verify finger assignments.")


if __name__ == '__main__':
    main()
