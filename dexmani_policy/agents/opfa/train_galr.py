"""GaLR autoencoder training — OPFA Stage 1.

Lightweight helpers for pose generation, plus the full training loop and CLI.
Does NOT import HandFKGenerator or KPConv modules at module level — safe to
import ``XHAND_JOINT_LIMITS`` / ``generate_random_poses`` without URDF/STL
dependencies.

Training protocol (matches OPFA exactly):
  - batch_size: 1 (OPFA convention)
  - lr: 1e-5 (OPFA convention)
  - loss: MSE(pred, gt) × 100
  - poses: uniform random within joint_limits['xhand']

CLI Usage::

    python -m dexmani_policy.agents.opfa.train_galr                           # all defaults
    python -m dexmani_policy.agents.opfa.train_galr --hand_type xhand --epochs 130
    python -m dexmani_policy.agents.opfa.train_galr --num_poses 20000 --batch_size 4
    python -m dexmani_policy.agents.opfa.train_galr --output_dir data/pretrained/galr/

Programmatic::

    from dexmani_policy.agents.opfa.train_galr import XHAND_JOINT_LIMITS, generate_random_poses

    poses = generate_random_poses(20000, hand_type="xhand")  # (20000, 12) float32

Output:
  - ``{output_dir}/best.pt`` — best validation checkpoint.
  - ``{output_dir}/latest.pt`` — latest checkpoint.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Joint limits (matching OPFA hands.py, slightly tightened)
# =============================================================================

XHAND_JOINT_LIMITS: dict[str, list[tuple[float, float]]] = {
    "xhand": [
        (0.0, 1.6),   # thumb_bend (abduction)
        (-0.3, 1.4),  # thumb_rota1 (rotation prox)
        (0.0, 1.5),   # thumb_rota2 (rotation dist)
        (-0.05, 0.14),  # index_bend (abduction)
        (0.0, 1.7),   # index_joint1 (prox)
        (0.0, 1.7),   # index_joint2 (dist)
        (0.0, 1.7),   # mid_joint1
        (0.0, 1.7),   # mid_joint2
        (0.0, 1.7),   # ring_joint1
        (0.0, 1.7),   # ring_joint2
        (0.0, 1.7),   # pinky_joint1
        (0.0, 1.7),   # pinky_joint2
    ],
}


def generate_random_poses(
    num_poses: int,
    hand_type: str = "xhand",
    seed: int = 42,
) -> np.ndarray:
    """Generate random joint angles within limits.

    Args:
        num_poses: number of random poses to generate.
        hand_type: hand type (must be a key in ``XHAND_JOINT_LIMITS``).
        seed: random seed for reproducibility.

    Returns:
        ``(num_poses, n_joints)`` float32 array.
    """
    limits = XHAND_JOINT_LIMITS[hand_type]
    low = np.array([lo for lo, _ in limits], dtype=np.float32)
    high = np.array([hi for _, hi in limits], dtype=np.float32)
    return np.random.RandomState(seed).uniform(low, high, size=(num_poses, len(limits)))


# =============================================================================
# Training loop (heavy imports deferred — only needed for CLI training)
# =============================================================================


def train(args) -> None:
    """Run GaLR autoencoder training loop.

    Args:
        args: argparse.Namespace with hand_type, num_poses, epochs, lr,
              loss_scale, val_ratio, output_dir.
    """
    import os
    import time

    import torch
    import torch.nn.functional as F

    from dexmani_policy.agents.opfa.galr_autoencoder import GaLRAutoencoder
    from dexmani_policy.agents.opfa.hand_fk import HandFKGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build FK generator ---
    print("Building HandFKGenerator...")
    fk = HandFKGenerator()

    # --- Build model ---
    print("Building GaLRAutoencoder...")
    model = GaLRAutoencoder(hand_type=args.hand_type)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # --- Pre-generate data (batch_size=1, so pre-generate all) ---
    print(f"Generating {args.num_poses} random poses...")
    all_poses = generate_random_poses(args.num_poses, args.hand_type)

    print("Pre-computing KPConv data for all poses...")
    all_data = []
    valid_pose_list = []  # Track poses that survived FK (correct alignment)
    for i in range(args.num_poses):
        angles = torch.from_numpy(all_poses[i])
        try:
            data = fk(angles, hand_type=args.hand_type)
        except Exception as e:
            print(f"  WARNING: skip pose {i}: {e}")
            continue
        # Move to GPU
        data_gpu = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data_gpu[k] = v.to(device)
            elif isinstance(v, np.ndarray):
                data_gpu[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, list):
                data_gpu[k] = [
                    x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device)
                    for x in v
                ]
            else:
                data_gpu[k] = v
        all_data.append(data_gpu)
        valid_pose_list.append(all_poses[i])  # Only append on success
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{args.num_poses}")

    valid_poses = torch.from_numpy(np.stack(valid_pose_list))
    print(f"Valid poses: {len(all_data)}/{args.num_poses}")

    # --- Train/val split ---
    n_val = max(1, int(len(all_data) * args.val_ratio))
    train_data = all_data[:-n_val]
    val_data = all_data[-n_val:]
    train_gt = valid_poses[:-n_val]
    val_gt = valid_poses[-n_val:]

    if len(train_data) == 0:
        raise ValueError(
            f"Empty training split: {len(all_data)} valid poses with val_ratio={args.val_ratio} "
            f"produces 0 training samples. Reduce val_ratio or increase num_poses."
        )
    if len(val_data) == 0:
        raise ValueError(
            f"Empty validation split: {len(all_data)} valid poses with val_ratio={args.val_ratio} "
            f"produces 0 validation samples. Reduce val_ratio or increase num_poses."
        )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    if args.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {args.epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training ---
    best_val_loss = float("inf")
    val_loss = float("inf")  # Initialised before loop (defensive)
    loss_scale = args.loss_scale  # OPFA default: 100

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for i in range(len(train_data)):
            output = model(train_data[i])
            pred = output["angles"]  # (12,)
            gt = train_gt[i].to(device)  # (12,)

            loss = F.mse_loss(pred, gt) * loss_scale

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_data)):
                output = model(val_data[i])
                pred = output["angles"]
                gt = val_gt[i].to(device)
                val_loss += F.mse_loss(pred, gt).item()
        val_loss = val_loss / len(val_data) * loss_scale

        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )

        # --- Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                os.path.join(args.output_dir, "best.pt"),
            )
            print(f"  => best checkpoint saved (val_loss={val_loss:.4f})")

    # --- Final checkpoint ---
    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "val_loss": val_loss},
        os.path.join(args.output_dir, "latest.pt"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for GaLR autoencoder training (OPFA Stage 1)."""
    import argparse

    parser = argparse.ArgumentParser(description="Train GaLR autoencoder (OPFA Stage 1)")
    parser.add_argument("--hand_type", default="xhand", help="Hand type")
    parser.add_argument("--num_poses", type=int, default=20000, help="Training poses")
    parser.add_argument("--epochs", type=int, default=130, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (OPFA default)")
    parser.add_argument("--loss_scale", type=float, default=100.0, help="MSE loss scale")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--output_dir", default="data/pretrained/galr/", help="Output directory")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
