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

from dexmani_policy.agents.hand_prior.codebook import (
    codebook_chunk_size, codebook_distances, validate_hand_coordinates,
)
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import positive_int
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite
from dexmani_policy.datasets.split import read_split_manifest, split_episodes
from dexmani_policy.training.common.optimization import build_scheduler
from dexmani_policy.training.pretrain.hand_prior.codebook import build_hand_prototypes, export_codebook

logger = logging.getLogger(__name__)


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
    if cfg.get("resume_from") is not None:
        raise ValueError("VQ pretraining does not support resume_from")
    buffer = EpisodeStore(cfg.dataset.zarr_path, keys=["action"])
    if buffer["action"].shape[1:] != (19,):
        raise ValueError("VQ pretraining requires action shape [T,19]")
    train, val, split = split_episodes(
        buffer, manifest=read_split_manifest(cfg.dataset.split_manifest), seed=cfg.dataset.seed,
        val_ratio=cfg.dataset.val_ratio, max_train_episodes=cfg.dataset.max_train_episodes,
    )
    validate_hand_coordinates(buffer.attrs)

    train_hand = load_hand_array(buffer, train)
    val_hand = load_hand_array(buffer, val)
    normalizer = LinearNormalizer()
    normalizer.fit_field("hand", (train_hand[start:start + 8192] for start in range(0, len(train_hand), 8192)), mode="limits")
    cfg.dataset.zarr_path = buffer.path
    cfg.dataset.split_manifest = split
    metadata = dict(split_manifest=split)
    train_hand = normalizer["hand"].normalize(train_hand).numpy()
    val_hand = normalizer["hand"].normalize(val_hand).numpy()
    return train_hand, val_hand, normalizer, metadata


@torch.no_grad()
def evaluate_vq(vqvae, loader, device, *, chunk_size=4096):
    """Evaluate a fixed model's runtime lookup error."""
    chunk_size = codebook_chunk_size(vqvae.codebook_size ** vqvae.num_groups, chunk_size)
    training = vqvae.training
    vqvae.eval()
    try:
        prototypes = build_hand_prototypes(vqvae)
        lookup_sum = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            n = batch.shape[0]
            count += n
            for values in batch.float().split(chunk_size):
                distances = codebook_distances(values, prototypes)
                lookup_sum += distances.amin(-1).double().square().sum()
        if not count:
            raise ValueError("VQ evaluation loader is empty")
        lookup_mse = float(lookup_sum / (count * vqvae.hand_dim))
        if not np.isfinite(lookup_mse):
            raise FloatingPointError("Non-finite lookup_mse during VQ evaluation")
        return {"lookup_mse": lookup_mse}
    finally:
        vqvae.train(training)


def train(cfg):
    recipe = cfg.vq_vae
    hand_dim = 12
    for name in ("num_epochs", "batch_size", "save_epochs"):
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
    train_norm, val_norm, normalizer, split_metadata = prepare_hand_data(cfg)
    cfg.vq_vae.output_dir = str(output_dir)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
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
    scheduler = build_scheduler(
        optimizer, "cosine", num_warmup_steps=recipe.warmup_steps, num_training_steps=total_steps
    )

    selection_metric = "val_lookup_mse" if has_validation else "train_lookup_mse"
    split_metadata["selection_metric"] = selection_metric
    split_metadata["selection_split"] = (
        "validation" if has_validation else "train_no_holdout"
    )
    logger.info("VQ selection: %s (%s)", selection_metric, split_metadata["selection_split"])
    best_lookup_mse = float("inf")

    for epoch in range(1, recipe.num_epochs + 1):
        vqvae.train()
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        train_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_loss, commitment_loss, _, recon_mse = vqvae(batch)
            total_loss = recipe.enc_loss_weight * recon_loss + recipe.vq_loss_weight * commitment_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Nonfinite VQ training loss")
            total_loss.backward()
            if recipe.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    vqvae.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
                )
            else:
                gradients = [torch.isfinite(p.grad).all() for p in vqvae.parameters() if p.grad is not None]
                if not torch.stack(gradients).all():
                    raise FloatingPointError("Nonfinite VQ gradient")
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
        selection_metrics = evaluate_vq(vqvae, selection_loader, device)
        metrics = {
            "train_enc": train_metrics["enc"],
            "train_vq": train_metrics["vq"],
            "train_mse": train_metrics["mse"],
            "train_total": train_total,
            selection_metric: selection_metrics["lookup_mse"],
            "lr": lr,
        }
        logger.info(
            "epoch %4d | lr %.2e | train recon %.5f commitment %.5f recon_mse %.5f",
            epoch,
            metrics["lr"],
            metrics["train_enc"],
            metrics["train_vq"],
            metrics["train_mse"],
        )
        logger.info("%s %.7f", selection_metric, metrics[selection_metric])
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"epoch": epoch, **metrics}, allow_nan=False) + "\n")

        checkpoint_paths = []
        if epoch % recipe.save_epochs == 0 or epoch == recipe.num_epochs:
            checkpoint_paths.append(output_dir / f"vqvae_hand_epoch={epoch:04d}.pt")
        lookup_mse = metrics[selection_metric]
        if lookup_mse < best_lookup_mse:
            best_lookup_mse = lookup_mse
            checkpoint_paths.append(output_dir / "vqvae_hand_best.pt")
        if epoch == recipe.num_epochs:
            checkpoint_paths.append(output_dir / "vqvae_hand_last.pt")
        if checkpoint_paths:
            payload = {
                "epoch": int(epoch),
                "model_state_dict": vqvae.state_dict(),
                "normalizer_state_dict": normalizer.state_dict(),
                "model_config": vqvae.model_config,
                "split_metadata": split_metadata,
                "metrics": metrics,
            }
            for checkpoint_path in checkpoint_paths:
                with atomic_path(checkpoint_path) as temporary:
                    torch.save(payload, temporary)
                logger.info("checkpoint saved: %s", checkpoint_path)

    selected_path = output_dir / "vqvae_hand_best.pt"
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
