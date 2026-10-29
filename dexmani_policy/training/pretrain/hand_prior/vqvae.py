"""Train single-step hand-state quantization using the target DQ-RISE episode split."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.agents.hand_prior.codebook import validate_dq_normalization
from dexmani_policy.common.config import load_policy_config, positive_int, validate_normalization_config
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.hand_prior.data import prepare_hand_source
from dexmani_policy.datasets.common.validation import check_finite
from dexmani_policy.common.hand_representation import validate_hand_coordinates
from dexmani_policy.training.common.optimization import get_scheduler
from dexmani_policy.training.pretrain.hand_prior.codebook import (
    build_hand_prototypes,
    build_runtime_codebook,
    deployment_diagnostics,
)

logger = logging.getLogger(__name__)


def prepare_policy_data(cfg):
    validate_normalization_config(cfg)
    spec = cfg.normalization
    validate_dq_normalization(spec)
    if cfg.agent.tcp_dim != 7:
        raise ValueError("DQ/VQ requires tcp_dim=7")
    buffer, train, val, split = prepare_hand_source(cfg, ["action"])
    validate_hand_coordinates(buffer.attrs)

    def read_hand(mask):
        chunks = []
        for start, values in buffer.iter_chunks("action", columns=slice(7, 19), episode_mask=mask):
            check_finite(buffer, "action", start, values, columns=slice(7, 19))
            chunks.append(values)
        return np.concatenate(chunks) if chunks else np.empty((0, 12), dtype=np.float32)

    train_hand, val_hand = read_hand(train), read_hand(val)
    normalizer = LinearNormalizer()
    normalizer.fit_field("hand", (train_hand[start:start + 8192] for start in range(0, len(train_hand), 8192)), mode="limits")
    metadata = dict(
        hand_data={"action_key": cfg.dataset.action_key}, split_manifest=split,
        dataset_config={key: cfg.dataset[key] for key in (
            "zarr_path", "action_key", "seed", "val_ratio", "max_train_episodes",
        )}, normalization_spec={"action": spec["action"]},
    )
    metadata["dataset_config"]["zarr_path"] = buffer.path
    train_hand = normalizer["hand"].normalize(train_hand).numpy()
    val_hand = normalizer["hand"].normalize(val_hand).numpy()
    return train_hand, val_hand, normalizer, metadata


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
    normalizer: LinearNormalizer,
    recipe,
    split_metadata: dict,
    metrics: dict,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": vqvae.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "training_config": OmegaConf.to_container(recipe, resolve=True),
        "model_config": vqvae.model_config,
        "split_metadata": split_metadata,
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)
    logger.info("checkpoint saved: %s", path)


def train(cfg):
    import matplotlib.pyplot as plt

    recipe = cfg.vq_vae
    hand_dim = cfg.agent.action_dim - cfg.agent.tcp_dim
    if (recipe.num_groups, recipe.codebook_size) != (cfg.agent.codebook_num_groups, cfg.agent.codebook_size):
        raise ValueError("VQ num_groups/codebook_size must match the target policy")
    for name in ("num_epochs", "batch_size", "save_epochs", "codebook_report_epochs"):
        positive_int(recipe[name], name)
    if not 0 <= recipe.max_grad_norm < float("inf") or recipe.warmup_steps < 0:
        raise ValueError("max_grad_norm and warmup_steps must be finite/nonnegative")
    output_dir = recipe.output_dir
    if output_dir is None:
        run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        output_dir = Path("experiments/vq_hand") / Path(cfg.dataset.zarr_path).stem / run
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info("VQ run: %s", output_dir.resolve())
    device = torch.device(recipe.device)
    set_seed(recipe.seed)
    train_norm, val_norm, normalizer, split_metadata = prepare_policy_data(cfg)
    train_ds = TensorDataset(torch.from_numpy(train_norm))
    val_ds = TensorDataset(torch.from_numpy(val_norm))
    if len(train_ds) < recipe.batch_size:
        raise ValueError("VQ training requires a full batch; reduce vq_vae.batch_size")
    train_loader = DataLoader(
        train_ds,
        batch_size=recipe.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    has_validation = len(val_ds) > 0
    selection_loader = DataLoader(
        val_ds if has_validation else train_ds,
        batch_size=recipe.batch_size,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(recipe.seed),
    )
    vqvae = HandVQVAE(
        hand_dim=hand_dim,
        latent_dim=recipe.latent_dim,
        hidden_dim=recipe.hidden_dim,
        num_groups=recipe.num_groups,
        codebook_size=recipe.codebook_size,
        num_layers=recipe.num_layers,
        act_scale=recipe.act_scale,
        loss_weight=recipe.loss_weight,
        vq_decay=recipe.vq_decay,
        threshold_ema_dead_code=recipe.threshold_ema_dead_code,
        kmeans_init=recipe.kmeans_init,
        kmeans_iters=recipe.kmeans_iters,
    ).to(device)

    optimizer = torch.optim.AdamW(
        vqvae.parameters(),
        lr=recipe.lr,
        betas=tuple(recipe.betas),
        weight_decay=recipe.weight_decay,
    )
    total_steps = len(train_loader) * recipe.num_epochs
    scheduler = get_scheduler(
        optimizer, "cosine", num_warmup_steps=recipe.warmup_steps, num_training_steps=total_steps
    )

    selection_metric = "val_lookup_mse" if has_validation else "train_lookup_mse"
    split_metadata["selection_metric"] = selection_metric
    split_metadata["selection_split"] = (
        "validation" if has_validation else "train_no_holdout"
    )
    logger.info("VQ selection: %s (%s)", selection_metric, split_metadata["selection_split"])
    train_history: list[float] = []
    best_lookup_mse = float("inf")

    for epoch in range(1, recipe.num_epochs + 1):
        vqvae.train()
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        train_count = 0
        report_usage = epoch % recipe.codebook_report_epochs == 0
        if report_usage:
            usage = torch.zeros(recipe.num_groups, recipe.codebook_size, dtype=torch.long, device=device)

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            enc_loss, vq_loss, indices, recon_mse = vqvae(batch)
            total_loss = recipe.enc_loss_weight * enc_loss + recipe.vq_loss_weight * vq_loss
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Nonfinite VQ training loss")
            total_loss.backward()
            if recipe.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    vqvae.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
                )
            elif not all(
                torch.isfinite(p.grad).all() for p in vqvae.parameters() if p.grad is not None
            ):
                raise FloatingPointError("Nonfinite VQ gradient")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if report_usage:
                    usage.scatter_add_(1, indices.T, torch.ones_like(indices.T))
                sums += torch.stack((enc_loss, vq_loss, recon_mse)).double() * batch.shape[0]
            train_count += batch.shape[0]

        train_metrics = dict(zip(("enc", "vq", "mse"), (sums / train_count).tolist()))
        train_total = (
            recipe.enc_loss_weight * train_metrics["enc"] + recipe.vq_loss_weight * train_metrics["vq"]
        )
        train_history.append(train_total)

        selection_metrics = evaluate_vq(
            vqvae, selection_loader, device, reconstruction=has_validation
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
        if has_validation:
            metrics.update({f"val_{key}": selection_metrics[key] for key in ("enc", "vq", "mse")})
            logger.info(
                "validation | enc %.5f vq %.5f recon_mse %.5f",
                metrics["val_enc"], metrics["val_vq"], metrics["val_mse"],
            )
        logger.info("%s %.7f", selection_metric, metrics[selection_metric])

        checkpoint_paths = []
        if epoch % recipe.save_epochs == 0 or epoch == recipe.num_epochs:
            checkpoint_paths.append(output_dir / f"vqvae_hand_epoch={epoch:04d}.pt")
        lookup_mse = metrics[selection_metric]
        if lookup_mse < best_lookup_mse:
            best_lookup_mse = lookup_mse
            checkpoint_paths.append(output_dir / "vqvae_hand_best.pt")
        if epoch == recipe.num_epochs:
            checkpoint_paths.append(output_dir / "vqvae_hand_last.pt")
        for checkpoint_path in checkpoint_paths:
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                vqvae=vqvae,
                normalizer=normalizer,
                recipe=recipe,
                split_metadata=split_metadata,
                metrics=metrics,
            )

        if report_usage:
            for group in range(recipe.num_groups):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(np.arange(recipe.codebook_size), usage[group].cpu().numpy())
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
    codebook_path = output_dir / "codebook.npz"
    with atomic_path(codebook_path) as temporary:
        runtime_codebook.save(temporary)
    print(f"Saved selected runtime codebook: {codebook_path}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pretrain/hand_prior/vqvae", help="Hydra config name or YAML path")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. vq_vae.device=cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(load_policy_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
