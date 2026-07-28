"""Train a single-step hand-state VQ-VAE for DQ-RISE.

Correctness guarantees in this version:
* train/validation split is episode-level;
* normalizer is fitted only on the selected training episodes;
* the same ``get_val_mask``/``downsample_mask`` logic as policy training is used;
* validation commitment loss is meaningful with the fixed VectorQuantize;
* checkpoints contain explicit model, split, and normalizer metadata;
* ``num_layers`` means the actual number of hidden linear layers.
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

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dexmani_policy.agents.vq_hand import VQVAEHand
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.sampler import downsample_mask, get_val_mask

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_episode_ends(buffer) -> np.ndarray:
    if hasattr(buffer, "episode_ends"):
        ends = buffer.episode_ends
        ends = ends[:] if hasattr(ends, "__getitem__") else ends
        return np.asarray(ends, dtype=np.int64)
    if hasattr(buffer, "meta") and "episode_ends" in buffer.meta:
        return np.asarray(buffer.meta["episode_ends"][:], dtype=np.int64)
    raise AttributeError("ReplayBuffer does not expose episode_ends")


def _episode_mask_to_frame_indices(
    episode_ends: np.ndarray, episode_mask: np.ndarray
) -> np.ndarray:
    mask = np.asarray(episode_mask, dtype=bool)
    if len(mask) != len(episode_ends):
        raise ValueError("episode mask and episode_ends have different lengths")
    starts = np.concatenate(([0], episode_ends[:-1]))
    chunks = [
        np.arange(start, end, dtype=np.int64)
        for use, start, end in zip(mask, starts, episode_ends)
        if use
    ]
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)


def _save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    normalizer: LinearNormalizer,
    args: argparse.Namespace,
    train_history: list[float],
    split_metadata: dict,
    metrics: dict,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": vqvae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        # Retained for backward compatibility with older analysis utilities.
        "normalizer_params": normalizer.params_dict,
        "args": vars(args),
        "model_config": {
            "hand_dim": vqvae.hand_dim,
            "latent_dim": vqvae.latent_dim,
            "hidden_dim": vqvae.hidden_dim,
            "num_groups": vqvae.num_groups,
            "codebook_size": vqvae.codebook_size,
            "num_layers": vqvae.num_layers,
            "act_scale": float(vqvae.act_scale.detach().cpu()),
        },
        "split_metadata": split_metadata,
        "metrics": metrics,
        "train_history": train_history,
        "format_version": 2,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    logger.info("checkpoint saved: %s", path)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    buffer = ReplayBuffer.copy_from_path(args.zarr_path, keys=[args.action_key])
    all_actions = np.asarray(buffer[args.action_key])
    hand_data = all_actions[:, args.tcp_dim :]
    if hand_data.shape[1] != args.hand_dim:
        raise ValueError(
            f"Configured hand_dim={args.hand_dim}, but data gives "
            f"{hand_data.shape[1]} after tcp_dim={args.tcp_dim}"
        )

    episode_ends = _get_episode_ends(buffer)
    val_episode_mask = get_val_mask(
        seed=args.seed,
        val_ratio=args.val_ratio,
        n_episodes=len(episode_ends),
    )
    train_episode_mask = downsample_mask(
        seed=args.seed,
        mask=~val_episode_mask,
        max_n=args.max_train_episodes,
    )
    train_indices = _episode_mask_to_frame_indices(episode_ends, train_episode_mask)
    val_indices = _episode_mask_to_frame_indices(episode_ends, val_episode_mask)
    if len(train_indices) == 0:
        raise ValueError("No training frames selected")

    logger.info(
        "episodes: train=%d, val=%d, excluded=%d; frames: train=%d, val=%d",
        int(train_episode_mask.sum()),
        int(val_episode_mask.sum()),
        int((~train_episode_mask & ~val_episode_mask).sum()),
        len(train_indices),
        len(val_indices),
    )

    # Fit on the full hand dataset so that min/max match the policy dataset
    # normalizer (pc_dataset.py uses all episodes).  This guarantees the VQ-VAE
    # and policy coordinate spaces agree, avoiding _validate_codebook_normalizer()
    # failures at DQ-RISE training start.
    normalizer = LinearNormalizer()
    normalizer.fit(
        data={"hand": hand_data},
        mode="limits",
        range_eps=1e-4,
    )
    train_norm = (
        normalizer["hand"]
        .normalize(hand_data[train_indices])
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    val_norm = (
        normalizer["hand"]
        .normalize(hand_data[val_indices])
        .cpu()
        .numpy()
        .astype(np.float32)
        if len(val_indices) > 0
        else np.empty((0, args.hand_dim), dtype=np.float32)
    )

    train_ds = TensorDataset(torch.from_numpy(train_norm))
    val_ds = TensorDataset(torch.from_numpy(val_norm))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(train_ds) >= args.batch_size,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        if len(val_ds) > 0
        else None
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader has zero batches")

    vqvae = VQVAEHand(
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

    optimizer = torch.optim.AdamW(
        vqvae.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.num_epochs)
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {
        "episode_ends": episode_ends.tolist(),
        "train_episode_ids": np.flatnonzero(train_episode_mask).tolist(),
        "val_episode_ids": np.flatnonzero(val_episode_mask).tolist(),
        "train_frame_count": int(len(train_indices)),
        "val_frame_count": int(len(val_indices)),
    }

    train_history: list[float] = []
    best_val_mse = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        vqvae.train()
        sums = {"enc": 0.0, "vq": 0.0, "mse": 0.0}
        usage = torch.zeros(
            args.num_groups, args.codebook_size, dtype=torch.long
        )

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            enc_loss, vq_loss, indices, recon_mse = vqvae(batch)
            total_loss = (
                args.enc_loss_weight * enc_loss
                + args.vq_loss_weight * vq_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for group in range(args.num_groups):
                    usage[group] += torch.bincount(
                        indices[:, group].detach().cpu(),
                        minlength=args.codebook_size,
                    )
            sums["enc"] += float(enc_loss)
            sums["vq"] += float(vq_loss)
            sums["mse"] += float(recon_mse)

        train_metrics = {key: value / len(train_loader) for key, value in sums.items()}
        train_total = (
            args.enc_loss_weight * train_metrics["enc"]
            + args.vq_loss_weight * train_metrics["vq"]
        )
        train_history.append(train_total)

        val_metrics = {"enc": float("nan"), "vq": float("nan"), "mse": float("nan")}
        if val_loader is not None:
            vqvae.eval()
            val_sums = {"enc": 0.0, "vq": 0.0, "mse": 0.0}
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    enc_loss, vq_loss, _, recon_mse = vqvae(batch)
                    val_sums["enc"] += float(enc_loss)
                    val_sums["vq"] += float(vq_loss)
                    val_sums["mse"] += float(recon_mse)
            val_metrics = {
                key: value / len(val_loader) for key, value in val_sums.items()
            }

        metrics = {
            "train_enc": train_metrics["enc"],
            "train_vq": train_metrics["vq"],
            "train_mse": train_metrics["mse"],
            "train_total": train_total,
            "val_enc": val_metrics["enc"],
            "val_vq": val_metrics["vq"],
            "val_mse": val_metrics["mse"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.info(
            "epoch %4d | lr %.2e | train enc %.5f vq %.5f mse %.5f | "
            "val enc %.5f vq %.5f mse %.5f",
            epoch,
            metrics["lr"],
            metrics["train_enc"],
            metrics["train_vq"],
            metrics["train_mse"],
            metrics["val_enc"],
            metrics["val_vq"],
            metrics["val_mse"],
        )

        if epoch % args.save_epochs == 0 or epoch == args.num_epochs:
            _save_checkpoint(
                output_dir / f"vqvae_hand_epoch={epoch:04d}.pt",
                epoch=epoch,
                vqvae=vqvae,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                args=args,
                train_history=train_history,
                split_metadata=split_metadata,
                metrics=metrics,
            )

        selection_mse = (
            metrics["val_mse"]
            if np.isfinite(metrics["val_mse"])
            else metrics["train_mse"]
        )
        if selection_mse < best_val_mse:
            best_val_mse = selection_mse
            _save_checkpoint(
                output_dir / "vqvae_hand_best.pt",
                epoch=epoch,
                vqvae=vqvae,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                args=args,
                train_history=train_history,
                split_metadata=split_metadata,
                metrics=metrics,
            )

        if epoch % args.codebook_report_epochs == 0:
            for group in range(args.num_groups):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(np.arange(args.codebook_size), usage[group].numpy())
                ax.set_title(f"Group {group} code usage — epoch {epoch}")
                ax.set_xlabel("Code index")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / f"code_usage_g{group}_epoch{epoch:04d}.png")
                plt.close(fig)

    final_metrics = metrics
    _save_checkpoint(
        output_dir / "vqvae_hand_last.pt",
        epoch=args.num_epochs,
        vqvae=vqvae,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        args=args,
        train_history=train_history,
        split_metadata=split_metadata,
        metrics=final_metrics,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training objective")
    ax.set_title("VQ-VAE training objective")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)


def _parse_loss_weight(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",")]


def _load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping")
    return dict(config.get("vq_vae", config))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--zarr_path", default=None)
    parser.add_argument("--action_key", default=None)
    parser.add_argument("--tcp_dim", type=int, default=None)
    parser.add_argument("--hand_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_groups", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--act_scale", type=float, default=None)
    parser.add_argument("--loss_weight", type=_parse_loss_weight, default=None)
    parser.add_argument("--vq_decay", type=float, default=None)
    parser.add_argument("--threshold_ema_dead_code", type=int, default=None)
    parser.add_argument(
        "--kmeans_init",
        type=lambda value: value.lower() in ("true", "1", "yes"),
        default=None,
    )
    parser.add_argument("--kmeans_iters", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--enc_loss_weight", type=float, default=None)
    parser.add_argument("--vq_loss_weight", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--max_train_episodes", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_epochs", type=int, default=None)
    parser.add_argument("--codebook_report_epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    if config_args.config:
        config_path = Path(config_args.config)
        if not config_path.is_absolute():
            config_path = _project_root / config_path
        parser.set_defaults(**_load_yaml_config(str(config_path)))
    args = parser.parse_args(remaining)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.zarr_path is None:
        parser.error("--zarr_path is required")
    if args.output_dir is None:
        parser.error("--output_dir is required")
    if args.hand_dim is None or args.hand_dim <= 0:
        buffer = ReplayBuffer.copy_from_path(args.zarr_path, keys=[args.action_key])
        args.hand_dim = int(np.asarray(buffer[args.action_key]).shape[-1] - args.tcp_dim)
    if args.loss_weight is None:
        args.loss_weight = [1.0] * args.hand_dim
    train(args)


if __name__ == "__main__":
    main()
