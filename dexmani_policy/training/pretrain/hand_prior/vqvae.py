"""Train single-step hand-state quantization on selected episodes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.hand_prior.codebook import (
    codebook_distances, validate_hand_coordinates,
)
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.optimization import build_scheduler
from dexmani_policy.training.pretrain.hand_prior.codebook import build_hand_prototypes, export_codebook

logger = logging.getLogger(__name__)


def codebook_chunk_size(num_codes: int) -> int:
    """Budget 16 MiB for an FP32 distance matrix, always allowing at least one row."""
    return min(4096, max(1, (16 * 1024 * 1024) // (num_codes * 4)))


def load_hand_array(buffer, episode_mask):
    count = np.diff(np.r_[0, buffer.episode_ends])[episode_mask].sum()
    values = np.empty((count, 12), dtype=np.float32)
    cursor = 0
    for start, chunk in buffer.iter_chunks("action", columns=slice(7, 19), episode_mask=episode_mask):
        check_finite(buffer, "action", start, chunk, columns=slice(7, 19))
        values[cursor:cursor + len(chunk)] = chunk
        cursor += len(chunk)
    return values


def prepare_hand_data(cfg):
    buffer = EpisodeStore(cfg.dataset.zarr_path, keys=["action"])
    if buffer["action"].shape[1:] != (19,):
        raise ValueError("VQ pretraining requires action shape [T,19]")
    train, val = split_episodes(
        buffer.n_episodes, seed=cfg.dataset.seed,
        val_episodes=cfg.dataset.val_episodes, max_train_episodes=cfg.dataset.max_train_episodes,
    )
    validate_hand_coordinates(buffer.attrs)

    train_hand = load_hand_array(buffer, train)
    val_hand = load_hand_array(buffer, val) if val.any() else None
    normalizer = LinearNormalizer()
    normalizer.fit_field("hand", (train_hand[start:start + 8192] for start in range(0, len(train_hand), 8192)), mode="limits")
    cfg.dataset.zarr_path = buffer.path
    logger.info("VQ episodes: total=%d train=%d val=%d unused=%d; train frames=%d",
                buffer.n_episodes, train.sum(), val.sum(), (~(train | val)).sum(), len(train_hand))
    logger.info("Hand normalization: fitted from train episodes only")
    train_hand = normalizer["hand"].normalize(train_hand).numpy()
    if val_hand is not None:
        val_hand = normalizer["hand"].normalize(val_hand).numpy()
    return train_hand, val_hand, normalizer


@torch.no_grad()
def evaluate_vq(vqvae, loader, device) -> float:
    """Evaluate a fixed model's runtime lookup error."""
    prototypes = build_hand_prototypes(vqvae)
    lookup_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        count += batch.shape[0]
        distances = codebook_distances(batch, prototypes)
        lookup_sum += distances.amin(-1).double().square().sum()
    if not count:
        raise ValueError("VQ evaluation loader is empty")
    lookup_mse = float(lookup_sum / (count * vqvae.hand_dim))
    if not np.isfinite(lookup_mse):
        raise FloatingPointError("Non-finite lookup_mse during VQ evaluation")
    return lookup_mse


def train(cfg):
    recipe = cfg.vq_vae
    hand_dim = 12
    for name in ("num_epochs", "batch_size", "save_epochs"):
        positive_int(recipe[name], name)
    if not 0 <= recipe.max_grad_norm < float("inf") or recipe.warmup_steps < 0:
        raise ValueError("max_grad_norm and warmup_steps must be finite/nonnegative")
    output_dir = Path(cfg.workspace.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    logger.info("VQ run: %s", output_dir.resolve())
    device = torch.device(recipe.device)
    set_seed(recipe.seed)
    train_norm, val_norm, normalizer = prepare_hand_data(cfg)
    cfg.workspace.output_dir = str(output_dir)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    train_ds = TensorDataset(torch.from_numpy(train_norm))
    train_loader = DataLoader(
        train_ds,
        batch_size=recipe.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_norm is not None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_norm)),
            batch_size=codebook_chunk_size(recipe.codebook_size ** recipe.num_groups),
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
        fused=device.type == "cuda",
    )
    total_steps = len(train_loader) * recipe.num_epochs
    scheduler = build_scheduler(
        optimizer, "cosine", num_warmup_steps=recipe.warmup_steps, num_training_steps=total_steps
    )

    logger.info("VQ export: %s", "validation lookup_mse best" if val_loader is not None else "last checkpoint")
    best_lookup_mse = float("inf")

    for epoch in range(1, recipe.num_epochs + 1):
        vqvae.train()
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        train_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_loss, commitment_loss, recon_mse = vqvae(batch)
            total_loss = recipe.enc_loss_weight * recon_loss + recipe.vq_loss_weight * commitment_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Nonfinite VQ training loss")
            total_loss.backward()
            if recipe.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    vqvae.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
                )
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                sums += torch.stack((recon_loss, commitment_loss, recon_mse)).double() * batch.shape[0]
            train_count += batch.shape[0]

        train_metrics = dict(zip(("enc", "vq", "mse"), (sums / train_count).tolist()))
        train_total = (
            recipe.enc_loss_weight * train_metrics["enc"] + recipe.vq_loss_weight * train_metrics["vq"]
        )
        metrics = {
            "train_enc": train_metrics["enc"],
            "train_vq": train_metrics["vq"],
            "train_mse": train_metrics["mse"],
            "train_total": train_total,
            "lr": lr,
        }
        if val_loader is not None:
            metrics["val_lookup_mse"] = evaluate_vq(vqvae, val_loader, device)
        logger.info(
            "epoch %4d | lr %.2e | train recon %.5f commitment %.5f recon_mse %.5f",
            epoch,
            metrics["lr"],
            metrics["train_enc"],
            metrics["train_vq"],
            metrics["train_mse"],
        )
        if val_loader is not None:
            logger.info("val_lookup_mse %.7f", metrics["val_lookup_mse"])
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": epoch, **metrics}, allow_nan=False) + "\n")

        checkpoint_paths = []
        if epoch % recipe.save_epochs == 0 and epoch != recipe.num_epochs:
            checkpoint_paths.append(checkpoint_dir / f"epoch_{epoch:04d}.pt")
        if val_loader is not None and metrics["val_lookup_mse"] < best_lookup_mse:
            best_lookup_mse = metrics["val_lookup_mse"]
            checkpoint_paths.append(checkpoint_dir / "best.pt")
        if epoch == recipe.num_epochs:
            checkpoint_paths.append(checkpoint_dir / "last.pt")
        if checkpoint_paths:
            payload = {
                "epoch": int(epoch),
                "model_state_dict": vqvae.state_dict(),
                "normalizer_state_dict": normalizer.state_dict(),
                "model_config": vqvae.model_config,
                "metrics": metrics,
            }
            for checkpoint_path in checkpoint_paths:
                with atomic_path(checkpoint_path) as temporary:
                    torch.save(payload, temporary)
                logger.info("checkpoint saved: %s", checkpoint_path)

    selected_path = checkpoint_dir / ("best.pt" if val_loader is not None else "last.pt")
    codebook_path = output_dir / "codebook.npz"
    export_codebook(selected_path, codebook_path, device="cpu")
    print(f"Saved selected runtime codebook: {codebook_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pretrain/hand_prior/vqvae", help="Config name or relative path under dexmani_policy/configs")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. vq_vae.device=cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(load_policy_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
