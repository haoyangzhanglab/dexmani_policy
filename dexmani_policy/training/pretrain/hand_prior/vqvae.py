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

from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import load_policy_config, positive_int, resolve_normalization_spec
from dexmani_policy.common.normalization import AffineNormalizer, FieldNormalizer
from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.training.common.hand_representation import describe_hand_data
from dexmani_policy.training.common.scheduler import get_scheduler
from dexmani_policy.training.data import build_normalizer
from dexmani_policy.training.pretrain.hand_prior.export_codebook import (
    build_hand_prototypes,
    build_runtime_codebook,
    deployment_diagnostics,
)
from dexmani_policy.training.validation import validate_config
from dexmani_policy.training.workspace import prepare_output_dir

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_vq(vqvae, loader, device, *, reconstruction=True, chunk_size=4096):
    """Evaluate a fixed model's runtime lookup and optional VQ reconstruction."""
    positive_int(chunk_size, "chunk_size")
    training = vqvae.training
    vqvae.eval()
    try:
        prototypes, _ = build_hand_prototypes(vqvae)
        chunk_size = min(
            chunk_size, max(1, (16 * 1024 * 1024) // (prototypes.numel() * 4))
        )
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        lookup_sum = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            n = batch.shape[0]
            count += n
            if reconstruction:
                enc, vq, _, mse = vqvae(batch)
                sums += torch.stack((enc, vq, mse)).double() * n
            for values in batch.float().split(chunk_size):
                distances = (values[:, None] - prototypes).square_().sum(-1)
                lookup_sum += distances.amin(-1).double().sum()
        if not count:
            raise ValueError("VQ evaluation loader is empty")
        lookup_mse = float(lookup_sum / (count * vqvae.hand_dim))
        if not np.isfinite(lookup_mse):
            raise FloatingPointError("Non-finite lookup_mse during VQ evaluation")
        metrics = {"lookup_mse": lookup_mse}
        if reconstruction:
            metrics.update(zip(("enc", "vq", "mse"), (sums / count).tolist()))
        return metrics
    finally:
        vqvae.train(training)


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    vqvae: torch.nn.Module,
    normalizer: AffineNormalizer,
    args: argparse.Namespace,
    train_history: list[float],
    split_metadata: dict,
    metrics: dict,
) -> None:
    payload = {
        "format": HandVQVAE.artifact_format,
        "epoch": int(epoch),
        "model_state_dict": vqvae.state_dict(),
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
            "kmeans_init": args.kmeans_init,
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
    if cfg.agent._target_ != "dexmani_policy.agents.policies.dqrise.DQRISEPolicy":
        raise ValueError("--policy-config requires a DQRISEPolicy configuration")
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare a new codebook from a fresh Policy config, not resume_from")
    if cfg.dataset.split_manifest is None:
        raise ValueError("Pretraining requires an explicit shared dataset.split_manifest")
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not isinstance(dataset, DemonstrationDataset):
        raise TypeError("DQ-RISE VQ preparation requires a single DemonstrationDataset")
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
        "split_manifest": dataset.data_recipe["split_manifest"],
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
    if not 0 <= args.max_grad_norm < float("inf") or args.warmup_steps < 0:
        raise ValueError("max_grad_norm and warmup_steps must be finite/nonnegative")
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
    selection_loader = val_loader
    if selection_loader is None:
        selection_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
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
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_scheduler(
        optimizer, "cosine", num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    selection_metric = "val_lookup_mse" if val_loader is not None else "train_lookup_mse"
    split_metadata["selection_metric"] = selection_metric
    split_metadata["selection_split"] = (
        "validation" if val_loader is not None else "train_no_holdout"
    )
    logger.info("VQ selection: %s (%s)", selection_metric, split_metadata["selection_split"])
    train_history: list[float] = []
    best_lookup_mse = float("inf")

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
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Nonfinite VQ training loss")
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    vqvae.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
            elif not all(
                torch.isfinite(p.grad).all() for p in vqvae.parameters() if p.grad is not None
            ):
                raise FloatingPointError("Nonfinite VQ gradient")
            lr = optimizer.param_groups[0]["lr"]
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

        selection_metrics = evaluate_vq(
            vqvae, selection_loader, device, reconstruction=val_loader is not None
        )
        metrics = {
            "train_enc": train_metrics["enc"],
            "train_vq": train_metrics["vq"],
            "train_mse": train_metrics["mse"],
            "train_total": train_total,
            selection_metric: selection_metrics["lookup_mse"],
            "lr": lr,
        }
        logger.info(
            "epoch %4d | lr %.2e | train enc %.5f vq %.5f recon_mse %.5f",
            epoch,
            metrics["lr"],
            metrics["train_enc"],
            metrics["train_vq"],
            metrics["train_mse"],
        )
        if val_loader is not None:
            metrics.update({f"val_{key}": selection_metrics[key] for key in ("enc", "vq", "mse")})
            logger.info(
                "validation | enc %.5f vq %.5f recon_mse %.5f",
                metrics["val_enc"], metrics["val_vq"], metrics["val_mse"],
            )
        logger.info("%s %.7f", selection_metric, metrics[selection_metric])

        checkpoint_paths = []
        if epoch % args.save_epochs == 0 or epoch == args.num_epochs:
            checkpoint_paths.append(output_dir / f"vqvae_hand_epoch={epoch:04d}.pt")
        lookup_mse = metrics[selection_metric]
        if lookup_mse < best_lookup_mse:
            best_lookup_mse = lookup_mse
            checkpoint_paths.append(output_dir / "vqvae_hand_best.pt")
        if epoch == args.num_epochs:
            checkpoint_paths.append(output_dir / "vqvae_hand_last.pt")
        for checkpoint_path in checkpoint_paths:
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                vqvae=vqvae,
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

    # Deployment diagnostics evaluate the selected best without changing selection.
    selected_path = output_dir / "vqvae_hand_best.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
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
        action=argparse.BooleanOptionalAction,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args, policy_cfg=policy_cfg)


if __name__ == "__main__":
    main()
