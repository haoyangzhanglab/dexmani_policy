"""Train single-step hand-state quantization using the target DQ-RISE Policy data recipe."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.core.normalizer import AffineNormalizer, FieldNormalizer
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.training.builder import build_normalizer
from dexmani_policy.training.pretrain.hand_prior.export_codebook import (
    build_runtime_codebook,
    deployment_diagnostics,
)
from dexmani_policy.training.validation import validate_config
from dexmani_policy.training.workspace import prepare_output_dir
from dexmani_policy.utils.config import load_policy_config, positive_int, resolve_normalization_spec
from dexmani_policy.utils.hand_representation import describe_hand_data
from dexmani_policy.utils.runtime import atomic_path

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_vq(vqvae, loader, device):
    """Sample means, independent of validation batch partitioning."""
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    count = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        enc, vq, _, mse = vqvae(batch)
        n = batch.shape[0]
        count += n
        sums += torch.stack((enc, vq, mse)).double() * n
    if not count:
        raise ValueError("VQ validation loader is empty")
    return dict(zip(("enc", "vq", "mse"), (sums / count).tolist()))


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    normalizer: AffineNormalizer,
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
        "args": vars(args),
        "model_config": {
            "hand_dim": vqvae.hand_dim,
            "latent_dim": vqvae.latent_dim,
            "hidden_dim": vqvae.hidden_dim,
            "num_groups": vqvae.num_groups,
            "codebook_size": vqvae.codebook_size,
            "num_layers": vqvae.num_layers,
            "act_scale": float(vqvae.act_scale.detach().cpu()),
            "loss_weight": vqvae.loss_weight.detach().cpu().tolist(),
            "vq_decay": args.vq_decay,
            "threshold_ema_dead_code": args.threshold_ema_dead_code,
            "kmeans_iters": args.kmeans_iters,
        },
        "split_metadata": split_metadata,
        "metrics": metrics,
        "train_history": train_history,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)
    logger.info("checkpoint saved: %s", path)


def build_policy_dataset(cfg):
    """Reuse Policy's actual Dataset, including observation/window qualification."""
    validate_config(cfg)
    if cfg.agent._target_ != "dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy":
        raise ValueError("--policy-config requires a DQRISEPolicy configuration")
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare a new codebook from a fresh Policy config, not resume_from")
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not isinstance(dataset, BaseDataset):
        raise TypeError("DQ-RISE VQ preparation requires a single BaseDataset")
    if dataset.replay_buffer[dataset.action_key].shape[1:] != (cfg.agent.action_dim,):
        raise ValueError("Dataset action shape does not match the Policy layout")
    return dataset


def policy_hand_rows(dataset, tcp_dim):
    """Read each split's already-qualified unique action rows, without augmentation."""
    if dataset is None:
        return np.empty((0, 12), dtype=np.float32)
    rows = dataset.sampler.action_source_rows
    buffer = dataset.replay_buffer
    blocks = []
    step = buffer.chunk_rows(dataset.action_key)
    for start in range(0, int(buffer.episode_ends[-1]), step):
        selected = rows[np.searchsorted(rows, start) : np.searchsorted(rows, start + step)] - start
        if len(selected):
            blocks.append(
                buffer.read(
                    dataset.action_key,
                    slice(start, start + step),
                    columns=slice(tcp_dim, tcp_dim + 12),
                )[selected]
            )
    return np.concatenate(blocks) if blocks else np.empty((0, 12), dtype=np.float32)


def prepare_policy_data(cfg):
    dataset = build_policy_dataset(cfg)
    hand_data = describe_hand_data(dataset)
    spec = resolve_normalization_spec(cfg)
    action = build_normalizer(dataset, {"action": spec["action"]}, cfg.action_key)["action"]
    hand_slice = slice(int(cfg.agent.tcp_dim), int(cfg.agent.action_dim))
    normalizer = AffineNormalizer()
    normalizer["hand"] = FieldNormalizer.from_parameters(
        action.params_dict["scale"][hand_slice].clone(),
        action.params_dict["offset"][hand_slice].clone(),
        {key: value[hand_slice].clone() for key, value in action.input_stats.items()},
    )
    val = dataset.get_validation_dataset()
    train_hand = policy_hand_rows(dataset, cfg.agent.tcp_dim)
    val_hand = policy_hand_rows(val, cfg.agent.tcp_dim)
    metadata = {
        "hand_data": hand_data,
        "policy_config": OmegaConf.to_container(cfg, resolve=False),
        "resolved_dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "normalization_spec": spec,
        "train_windows": dataset.sampler.validity_summary,
        "val_windows": val.sampler.validity_summary if val is not None else None,
    }
    metadata["resolved_dataset"]["zarr_path"] = dataset.zarr_path
    return (
        normalizer["hand"].normalize(train_hand).numpy(),
        normalizer["hand"].normalize(val_hand).numpy(),
        normalizer,
        metadata,
    )


def train(args: argparse.Namespace, *, policy_cfg) -> None:
    for name in ("num_epochs", "batch_size", "save_epochs", "codebook_report_epochs"):
        positive_int(getattr(args, name), name)
    if args.output_dir is None:
        run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        args.output_dir = str(Path("experiments/vq_hand") / Path(args.zarr_path).stem / run)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir)
    for pattern in ("vqvae_hand_*.pt", "*.npz", "code_usage_*.png", "loss_curve.png"):
        if any(output_dir.glob(pattern)):
            raise FileExistsError(
                f"VQ output already exists in {output_dir}; use a new output directory"
            )
    logger.info("VQ run: %s", output_dir.resolve())
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_norm, val_norm, normalizer, split_metadata = prepare_policy_data(policy_cfg)
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
    vqvae = HandVQVAE(
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    selection_metric = "val_mse" if val_loader is not None else "train_mse"
    split_metadata["selection_metric"] = selection_metric
    train_history: list[float] = []
    best_mse = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        vqvae.train()
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        train_count = 0
        usage = torch.zeros(args.num_groups, args.codebook_size, dtype=torch.long, device=device)

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            enc_loss, vq_loss, indices, recon_mse = vqvae(batch)
            total_loss = args.enc_loss_weight * enc_loss + args.vq_loss_weight * vq_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                usage.scatter_add_(1, indices.T, torch.ones_like(indices.T))
                sums += torch.stack((enc_loss, vq_loss, recon_mse)).double() * batch.shape[0]
            train_count += batch.shape[0]

        train_metrics = dict(zip(("enc", "vq", "mse"), (sums / train_count).tolist()))
        train_total = (
            args.enc_loss_weight * train_metrics["enc"] + args.vq_loss_weight * train_metrics["vq"]
        )
        train_history.append(train_total)

        val_metrics = {"enc": float("nan"), "vq": float("nan"), "mse": float("nan")}
        if val_loader is not None:
            vqvae.eval()
            val_metrics = evaluate_vq(vqvae, val_loader, device)

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
            "epoch %4d | lr %.2e | train enc %.5f vq %.5f mse %.5f | val enc %.5f vq %.5f mse %.5f",
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
            save_checkpoint(
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

        selection_mse = metrics[selection_metric]
        if not np.isfinite(selection_mse):
            raise FloatingPointError(
                f"Non-finite {selection_metric} at epoch {epoch}; previous best preserved"
            )
        if selection_mse < best_mse:
            best_mse = selection_mse
            save_checkpoint(
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
                ax.bar(np.arange(args.codebook_size), usage[group].cpu().numpy())
                ax.set_title(f"Group {group} code usage — epoch {epoch}")
                ax.set_xlabel("Code index")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / f"code_usage_g{group}_epoch{epoch:04d}.png")
                plt.close(fig)

    save_checkpoint(
        output_dir / "vqvae_hand_last.pt",
        epoch=args.num_epochs,
        vqvae=vqvae,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        args=args,
        train_history=train_history,
        split_metadata=split_metadata,
        metrics=metrics,
    )

    # Deployment diagnostics evaluate the selected best without changing selection.
    selected_path = output_dir / "vqvae_hand_best.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    runtime_codebook = build_runtime_codebook(selected, selected_path, device="cpu")
    diagnostics = deployment_diagnostics(
        runtime_codebook, {"train": train_norm, "held_out": val_norm}
    )
    with atomic_path(output_dir / "codebook_diagnostics.json") as temporary:
        temporary.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training objective")
    ax.set_title("VQ-VAE training objective")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)


def parse_loss_weight(value):
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",")]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-config",
        required=True,
        help="Target DQ-RISE Policy YAML providing data and VQ defaults",
    )
    parser.add_argument(
        "--policy-override",
        action="append",
        default=[],
        help="Hydra override for the target Policy (repeatable)",
    )
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_groups", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--act_scale", type=float, default=None)
    parser.add_argument("--loss_weight", type=parse_loss_weight, default=None)
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
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_epochs", type=int, default=None)
    parser.add_argument("--codebook_report_epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    initial = parser.parse_args(argv)
    policy_cfg = load_policy_config(initial.policy_config, initial.policy_override)
    parser.set_defaults(**OmegaConf.to_container(policy_cfg.vq_vae, resolve=True))
    args = parser.parse_args(argv)
    args.zarr_path = str(policy_cfg.dataset.zarr_path)
    args.hand_dim = int(policy_cfg.agent.action_dim) - int(policy_cfg.agent.tcp_dim)
    if (args.num_groups, args.codebook_size) != (
        policy_cfg.agent.codebook_num_groups,
        policy_cfg.agent.codebook_size,
    ):
        parser.error("VQ num_groups/codebook_size must match the target Policy")
    if args.loss_weight is None:
        args.loss_weight = [1.0] * args.hand_dim
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args, policy_cfg=policy_cfg)


if __name__ == "__main__":
    main()
