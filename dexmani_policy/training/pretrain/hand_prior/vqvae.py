"""Train single-step hand-state quantization on selected episodes."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.hand_prior.codebook import codebook_distances
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.hand import load_hand_array
from dexmani_policy.datasets.split import format_split_summary, split_episodes
from dexmani_policy.training.config import check_fresh_output, load_config, normalize_output_dir
from dexmani_policy.training.data import record_training_data, use_policy_split
from dexmani_policy.training.optimization import build_scheduler, clip_gradients
from dexmani_policy.training.pretrain.hand_prior.codebook import build_hand_prototypes, export_codebook

logger = logging.getLogger(__name__)


def codebook_chunk_size(num_codes: int) -> int:
    """Budget 16 MiB for an FP32 distance matrix, always allowing at least one row."""
    return min(4096, max(1, (16 * 1024 * 1024) // (num_codes * 4)))


def prepare_hand_data(cfg):
    buffer = EpisodeStore(cfg.dataset.zarr_path, keys=["action"])
    train, val = split_episodes(
        buffer.episode_groups, seed=cfg.dataset.seed,
        val_episodes=cfg.dataset.val_episodes, max_train_episodes=cfg.dataset.max_train_episodes,
    )
    record_training_data(cfg, buffer, train, val)

    train_hand = load_hand_array(buffer, train)
    val_hand = load_hand_array(buffer, val) if val.any() else None
    normalizer = LinearNormalizer()
    normalizer.fit_field(
        "hand", (train_hand[start:start + 8192] for start in range(0, len(train_hand), 8192)),
        check_values=False,
    )
    logger.info(
        "VQ %s; train frames=%d", format_split_summary(buffer, train, val), len(train_hand),
    )
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
    output_dir = normalize_output_dir(cfg)
    check_fresh_output(output_dir, "codebook.npz")
    device = torch.device(recipe.device)
    set_seed(recipe.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    train_norm, val_norm, normalizer = prepare_hand_data(cfg)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    logger.info("VQ run: %s", output_dir)
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
            total_loss.backward()
            clip_gradients(
                vqvae.parameters(), torch.isfinite(total_loss.detach()),
                recipe.max_grad_norm, step=scheduler.last_epoch,
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
        if epoch % recipe.save_epochs == 0 or epoch == recipe.num_epochs:
            checkpoint_paths.append(checkpoint_dir / "last.pt")
        if val_loader is not None and metrics["val_lookup_mse"] < best_lookup_mse:
            best_lookup_mse = metrics["val_lookup_mse"]
            checkpoint_paths.append(checkpoint_dir / "best.pt")
        if checkpoint_paths:
            payload = {
                "epoch": int(epoch),
                "model_state_dict": vqvae.state_dict(),
                "normalizer_state_dict": normalizer.state_dict(),
                "model_config": vqvae.model_config,
                "metrics": metrics,
                "training_data": OmegaConf.to_container(cfg.training_data),
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
    parser.add_argument("--policy-config", help="Read the data split from a resolved policy YAML")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. vq_vae.device=cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config, args.overrides)
    if args.policy_config is not None:
        use_policy_split(cfg, args.policy_config)
    train(cfg)


if __name__ == "__main__":
    main()
