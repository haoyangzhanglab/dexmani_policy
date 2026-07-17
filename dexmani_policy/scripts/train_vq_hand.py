"""
VQ-VAE hand-pose pretraining script.

Loads hand joint data from a Zarr dataset, normalises to [-1, 1],
trains a VqVaeHand model, and saves checkpoints.

Usage:
    # With YAML config (recommended):
    python -m dexmani_policy.scripts.train_vq_hand \
        --config configs/dqrise.yaml \
        --zarr_path robot_data/pick_apple_messy.zarr \
        --output_dir experiments/vq_hand/pick_apple_messy

    # Override any field via CLI:
    python -m dexmani_policy.scripts.train_vq_hand \
        --config configs/dqrise.yaml \
        --zarr_path robot_data/pick_apple_messy.zarr \
        --num_epochs 2000 --lr 1e-4

    # Without config (all arguments must be explicit):
    python -m dexmani_policy.scripts.train_vq_hand \
        --zarr_path robot_data/pick_apple_messy.zarr \
        --hand_dim 12 --tcp_dim 9 \
        --loss_weight 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0, 0.5,0.5, 0.5,0.5

    # Or with the shell wrapper:
    bash scripts/train_vq_hand.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# ── project imports ──────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dexmani_policy.agents.vq_hand import VqVaeHand
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.common.normalizer import LinearNormalizer

logger = logging.getLogger(__name__)


# =========================================================================
# helper
# =========================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# checkpoint
# =========================================================================

def _save_checkpoint(
    path: str,
    epoch: int,
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: LinearNormalizer,
    args: argparse.Namespace,
    train_history: list[float],
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': vqvae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'normalizer_params': normalizer.params_dict,
        'args': vars(args),
        'train_history': train_history,
    }, path)
    logger.info('  → checkpoint saved: %s', path)


# =========================================================================
# training
# =========================================================================

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # ── 1. Load hand data from Zarr ──────────────────────────────────
    logger.info('Loading hand data from %s ...', args.zarr_path)
    buffer = ReplayBuffer.copy_from_path(args.zarr_path, keys=[args.action_key])
    all_actions = buffer[args.action_key]                      # (N, action_dim)
    hand_data = all_actions[:, args.tcp_dim:]                  # (N, hand_dim)
    logger.info('  full action: %s  →  hand subset: %s', all_actions.shape, hand_data.shape)
    logger.info('  hand range: [%.4f, %.4f]', float(hand_data.min()), float(hand_data.max()))

    # ── 2. Normalizer ──────────────────────────────────────────────
    normalizer = LinearNormalizer()
    normalizer.fit(data={'hand': hand_data}, mode='limits', range_eps=1e-4)
    hand_normed = normalizer['hand'].normalize(hand_data).numpy()  # (N, hand_dim) in [-1, 1]
    logger.info('  normalised range: [%.4f, %.4f]', float(hand_normed.min()), float(hand_normed.max()))

    # ── 3. Dataset & DataLoader (with train/val split) ─────────────
    hand_tensor = torch.from_numpy(hand_normed.astype(np.float32))
    n_total = len(hand_tensor)
    n_train = int(n_total * 0.95)
    indices = torch.randperm(n_total)
    train_ds = TensorDataset(hand_tensor[indices[:n_train]])
    val_ds = TensorDataset(hand_tensor[indices[n_train:]])
    logger.info('  train: %d  val: %d  (split=95/5)', n_train, n_total - n_train)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == 'cuda'),
    )
    logger.info('  %d train samples, %d batches/epoch (batch_size=%d)',
                len(train_ds), len(loader), args.batch_size)

    # ── 4. Build VQ-VAE ────────────────────────────────────────────
    vqvae = VqVaeHand(
        hand_dim=args.hand_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_groups=args.num_groups,
        codebook_size=args.codebook_size,
        num_layers=args.num_layers,
        act_scale=args.act_scale,
        loss_weight=args.loss_weight,
        vq_decay=args.vq_decay,
        threshold_ema_dead_code=args.threshold_ema_dead_code,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
    ).to(device)

    n_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    logger.info('  VqVaeHand: %.2fM parameters', n_params / 1e6)
    logger.info('  codebook: %d groups × %d codes = %d combinations',
                args.num_groups, args.codebook_size, args.codebook_size ** args.num_groups)

    # ── 5. Optimizer & scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        vqvae.parameters(),
        lr=args.lr,
        betas=(args.betas[0], args.betas[1]),
        weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.num_epochs

    # CosineAnnealingLR with linear warmup via SequentialLR.
    # Previously LambdaLR(warmup) was overwritten by CosineAnnealingLR
    # because both operated on the same optimizer and the last .step()
    # always won — the warmup was silently non-functional.
    if args.warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0,
            total_iters=args.warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - args.warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

    # ── 6. Output directory ────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 7. Training loop ───────────────────────────────────────────
    train_history: list[float] = []
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        epoch_enc_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_mse = 0.0
        epoch_code_usage = torch.zeros(args.num_groups, args.codebook_size, dtype=torch.long)

        vqvae.train()
        for (hand_batch,) in loader:
            hand_batch = hand_batch.to(device)                     # (B, hand_dim)

            enc_loss, vq_loss, indices, recon_mse = vqvae(hand_batch)

            # codebook usage tracking (DQ-RISE convention)
            with torch.no_grad():
                for g in range(args.num_groups):
                    idx_g = indices[:, g]
                    cnt = torch.bincount(idx_g, minlength=args.codebook_size)
                    epoch_code_usage[g] += cnt.cpu()

            # total loss  (aligned with DQ-RISE: L1_recon × 3 + VQ × 5)
            total = enc_loss * args.enc_loss_weight + vq_loss * args.vq_loss_weight

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), args.max_grad_norm)
            optimizer.step()

            scheduler.step()

            epoch_enc_loss += enc_loss.item()
            epoch_vq_loss += vq_loss.item()
            epoch_mse += recon_mse.item()
            global_step += 1

        n_batches = len(loader)
        epoch_enc_loss /= n_batches
        epoch_vq_loss /= n_batches
        epoch_mse /= n_batches
        epoch_total = epoch_enc_loss * args.enc_loss_weight + epoch_vq_loss * args.vq_loss_weight
        train_history.append(epoch_total)

        # ── validation ──────────────────────────────────────────────
        val_enc_loss = val_vq_loss = val_mse = 0.0
        if val_loader is not None:
            vqvae.eval()
            with torch.no_grad():
                for (hand_batch,) in val_loader:
                    hand_batch = hand_batch.to(device)
                    enc_l, vq_l, _, mse_l = vqvae(hand_batch)
                    val_enc_loss += enc_l.item()
                    val_vq_loss += vq_l.item()
                    val_mse += mse_l.item()
            n_val = len(val_loader)
            val_enc_loss /= n_val
            val_vq_loss /= n_val
            val_mse /= n_val

        lr_now = optimizer.param_groups[0]['lr']
        logger.info(
            'epoch %4d | lr %.2e | enc %.4f | vq %.4f | mse %.4f | total %.4f | '
            'val_enc %.4f | val_vq %.4f | val_mse %.4f',
            epoch, lr_now, epoch_enc_loss, epoch_vq_loss, epoch_mse, epoch_total,
            val_enc_loss, val_vq_loss, val_mse,
        )

        # ── periodic checkpoint ─────────────────────────────────
        if epoch % args.save_epochs == 0 or epoch == args.num_epochs:
            _save_checkpoint(
                os.path.join(args.output_dir, f'vqvae_hand_epoch={epoch:04d}.pt'),
                epoch, vqvae, optimizer, scheduler, normalizer, args, train_history,
            )

        # ── codebook usage report ───────────────────────────────
        if epoch % args.codebook_report_epochs == 0 and epoch > 0:
            for g in range(args.num_groups):
                usage = epoch_code_usage[g].cpu().numpy()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(np.arange(args.codebook_size), usage)
                ax.set_title(f'Group {g} Code Usage — Epoch {epoch}')
                ax.set_xlabel('Code Index')
                ax.set_ylabel('Usage Count')
                fig.tight_layout()
                fig_path = os.path.join(args.output_dir, f'code_usage_g{g}_epoch{epoch:04d}.png')
                fig.savefig(fig_path)
                plt.close(fig)
                logger.info('  → code usage plot: %s', fig_path)

    # ── 8. Final checkpoint ──────────────────────────────────────────
    _save_checkpoint(
        os.path.join(args.output_dir, 'vqvae_hand_last.pt'),
        args.num_epochs, vqvae, optimizer, scheduler, normalizer, args, train_history,
    )

    # ── 9. Loss curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('VQ-VAE Training Loss')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close(fig)

    logger.info('Training complete.  %d epochs, final loss=%.4f', args.num_epochs, train_history[-1])


# =========================================================================
# CLI
# =========================================================================

def _parse_loss_weight(s: str | None) -> list[float] | None:
    """Parse comma-separated float list, e.g. '1.0,1.0,1.0,0.5,0.5,1.0'."""
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    return [float(x.strip()) for x in s.split(',')]


def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file, return dict of argparse-compatible keys.

    If the config has a top-level ``vq_vae`` section (e.g. dqrise.yaml),
    only that section is used.  Otherwise the whole file is treated as a
    flat VQ-VAE config (standalone mode).
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'YAML config must be a mapping, got {type(cfg).__name__}')

    # Extract vq_vae section from policy configs (e.g. dqrise.yaml)
    if 'vq_vae' in cfg and isinstance(cfg['vq_vae'], dict):
        return dict(cfg['vq_vae'])
    return dict(cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with no hardcoded defaults (all come from YAML)."""
    ap = argparse.ArgumentParser(
        description='Train VQ-VAE for hand-pose discretisation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── config ──
    ap.add_argument('--config', type=str, default=None,
                    help='YAML config file (e.g. configs/dqrise.yaml, reads vq_vae section). '
                         'CLI arguments override config values.')

    # ── data (no defaults — either from YAML or explicit CLI) ──
    ap.add_argument('--zarr_path', default=None,
                    help='Path to Zarr dataset (e.g. robot_data/pick_apple_messy.zarr)')
    ap.add_argument('--action_key', default=None,
                    help='Key for action array in Zarr data group')
    ap.add_argument('--tcp_dim', type=int, default=None,
                    help='TCP (arm) dimension — first tcp_dim cols are arm, rest are hand')
    ap.add_argument('--hand_dim', type=int, default=None,
                    help='Hand joint dimension (auto-detected from action_dim - tcp_dim if 0)')

    # ── model ──
    ap.add_argument('--latent_dim', type=int, default=None)
    ap.add_argument('--hidden_dim', type=int, default=None)
    ap.add_argument('--num_groups', type=int, default=None,
                    help='Number of residual VQ groups → total codes = codebook_size^num_groups')
    ap.add_argument('--codebook_size', type=int, default=None,
                    help='Number of codes per group')
    ap.add_argument('--num_layers', type=int, default=None,
                    help='Hidden layers in encoder/decoder MLP trunk')
    ap.add_argument('--act_scale', type=float, default=None,
                    help='Action scale factor (1.0 = data already in [-1,1])')
    ap.add_argument('--loss_weight', type=_parse_loss_weight, default=None,
                    help='Per-dim L1 weight, comma-separated '
                         '(XHand: thumb/index/middle=1.0, ring/pinky=0.5)')
    ap.add_argument('--vq_decay', type=float, default=None,
                    help='EMA decay for codebook updates')
    ap.add_argument('--threshold_ema_dead_code', type=int, default=None,
                    help='Dead-code replacement threshold (0 = disabled)')
    ap.add_argument('--kmeans_init', type=lambda x: x.lower() in ('true', '1', 'yes'), default=None,
                    help='Enable k-means codebook initialisation (true/false)')
    ap.add_argument('--kmeans_iters', type=int, default=None,
                    help='K-means iterations (default 10)')

    # ── training ──
    ap.add_argument('--num_epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--betas', type=float, nargs=2, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--enc_loss_weight', type=float, default=None,
                    help='Multiplier on encoder L1 reconstruction loss')
    ap.add_argument('--vq_loss_weight', type=float, default=None,
                    help='Multiplier on VQ commitment loss')
    ap.add_argument('--max_grad_norm', type=float, default=None)
    ap.add_argument('--warmup_steps', type=int, default=None)

    # ── logging ──
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--save_epochs', type=int, default=None)
    ap.add_argument('--codebook_report_epochs', type=int, default=None)

    # ── misc ──
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)

    return ap


def main(argv: list[str] | None = None):
    ap = _build_arg_parser()

    # ── First pass: extract --config path ────────────────────────────
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    # ── Load YAML config as defaults ─────────────────────────────────
    if config_args.config is not None:
        yaml_path = config_args.config
        if not os.path.isabs(yaml_path):
            # Relative to project root
            yaml_path = os.path.join(_project_root, yaml_path)
        yaml_defaults = _load_yaml_config(yaml_path)
        ap.set_defaults(**yaml_defaults)
        logger.info('Loaded config: %s', config_args.config)

    # ── Second pass: parse CLI (overrides YAML defaults) ─────────────
    args = ap.parse_args(remaining_argv)

    # Validate required fields
    if args.zarr_path is None:
        ap.error('--zarr_path is required (set in config or via CLI)')
    if args.output_dir is None:
        ap.error('--output_dir is required (set via CLI, e.g. --output_dir experiments/vq_hand/my_task)')

    # auto-detect hand_dim if zero or unset
    if args.hand_dim is None or args.hand_dim <= 0:
        import zarr
        root = zarr.open(os.path.expanduser(args.zarr_path), 'r')
        full_dim = root['data'][args.action_key].shape[-1]
        args.hand_dim = full_dim - args.tcp_dim
        logger.info('Auto-detected hand_dim = %d (%d - %d)', args.hand_dim, full_dim, args.tcp_dim)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    train(args)


if __name__ == '__main__':
    main()
