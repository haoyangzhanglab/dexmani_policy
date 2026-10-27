"""Pretrain a hand codec from an independent, episodically split command/state corpus."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.hand_vae import HandVAE
from dexmani_policy.agents.hand_prior.losses import diagonal_gaussian_kl
from dexmani_policy.common.config import load_policy_config, positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.hand_prior.hand_chunk_dataset import HandChunkDataset


def build_codec_dataset(cfg):
    if not issubclass(hydra.utils.get_class(cfg.dataset._target_), HandChunkDataset):
        raise TypeError("HandVAE pretraining requires HandChunkDataset")
    if cfg.dataset.horizon != cfg.horizon or cfg.dataset.seed != cfg.seed:
        raise ValueError("HandVAE dataset horizon/seed must match the training configuration")
    dataset = hydra.utils.instantiate(cfg.dataset)
    if dataset.get_validation_dataset() is None:
        raise ValueError("HandVAE training requires a nonempty held-out validation split")
    return dataset


def fit_relative_statistics(dataset, batch_size=4096, std_floor=1e-6):
    """Population moments over full train windows; repeated rows retain their own q0."""
    if dataset.is_validation:
        raise ValueError("Validation must reuse training codec statistics")
    total = np.zeros(12, dtype=np.float64)
    squared = np.zeros(12, dtype=np.float64)
    count = 0
    for start in range(0, len(dataset), batch_size):
        actions, q0 = dataset.raw_hand_batch(np.arange(start, min(start + batch_size, len(dataset))))
        relative = actions.astype(np.float64) - q0[:, None].astype(np.float64)
        total += relative.sum(axis=(0, 1))
        squared += np.square(relative).sum(axis=(0, 1))
        count += relative.shape[0] * relative.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 0)).clip(min=std_floor)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("HandVAE train relative statistics must be finite")
    return (
        torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32)),
        dict(population="train_hand_windows", windows=len(dataset), frames=count, std_floor=std_floor),
    )


def codec_batch(dataset, indices, device):
    actions, q0 = dataset.raw_hand_batch(indices)
    return torch.as_tensor(actions, device=device), torch.as_tensor(q0, device=device)


def kl_coefficient(step, beta, warmup, ramp):
    return beta * min(1.0, max(0.0, (step - warmup) / max(1, ramp)))


@torch.no_grad()
def evaluate(codec, dataset, batch_size):
    was_training = codec.training
    codec.eval()
    absolute = torch.zeros(12, dtype=torch.float64, device=codec.action_mean.device)
    totals = torch.zeros(2, dtype=torch.float64, device=codec.action_mean.device)
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        actions, q0 = codec_batch(dataset, indices, codec.action_mean.device)
        mu, log_var = codec.encode(actions, q0)
        decoded = codec.decode(mu, q0)
        physical = codec.denormalize_actions(decoded, q0)
        absolute += (physical - actions).abs().double().sum(dim=(0, 1))
        reconstruction = codec.reconstruction_loss(decoded, codec.normalize_actions(actions, q0))
        kl = diagonal_gaussian_kl(mu, log_var)
        totals += torch.stack((reconstruction, kl)).double() * len(indices)
    reconstruction, kl = (totals / len(dataset)).tolist()
    joint_mae = absolute / (len(dataset) * dataset.horizon)
    codec.train(was_training)
    metrics = dict(
        physical_mae=float(joint_mae.mean()), physical_joint_mae=joint_mae.tolist(),
        units="radians", reconstruction=reconstruction, kl=kl, windows=len(dataset),
    )
    if not all(math.isfinite(metrics[key]) for key in ("physical_mae", "reconstruction", "kl")):
        raise ValueError("HandVAE evaluation became nonfinite")
    return metrics


def train(cfg, *, overwrite=False):
    recipe = cfg.training
    for key in ("steps", "batch_size", "eval_every", "print_every"):
        positive_int(recipe[key], key)
    for key in ("kl_warmup", "kl_ramp"):
        if type(recipe[key]) is not int or recipe[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not 0 <= recipe.weight_decay < float("inf"):
        raise ValueError("weight_decay must be nonnegative and finite")
    if not 0 < recipe.lr < float("inf") or not 0 < recipe.clip_grad < float("inf"):
        raise ValueError("lr and clip_grad must be positive and finite")
    kl_end_step = recipe.kl_warmup + max(1, recipe.kl_ramp)
    if not 0 < recipe.beta < float("inf") or recipe.steps < kl_end_step:
        raise ValueError(f"HandVAE requires positive finite beta and steps >= {kl_end_step}")
    output = Path(cfg.codec.path).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Codec already exists: {output}; use --overwrite to replace it")
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dataset = build_codec_dataset(cfg)
    validation = dataset.get_validation_dataset()
    if len(dataset) < recipe.batch_size:
        raise ValueError("HandVAE training requires at least one full batch; set a smaller batch explicitly")
    codec = HandVAE(
        horizon=dataset.horizon, latent_dim=cfg.codec.latent_dim, hidden_dim=cfg.codec.hidden_dim,
        n_layers=cfg.codec.n_layers, n_heads=cfg.codec.n_heads, mlp_ratio=cfg.codec.mlp_ratio,
    ).to(recipe.device)
    weights = torch.as_tensor(list(cfg.codec.joint_weights), device=recipe.device, dtype=torch.float32)
    if weights.shape != (12,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("codec.joint_weights must contain 12 positive finite values")
    codec.joint_weights.copy_(weights)
    mean, std, statistics = fit_relative_statistics(dataset, recipe.batch_size)
    codec.action_mean.copy_(mean.to(recipe.device))
    codec.action_std.copy_(std.to(recipe.device))
    optimizer = torch.optim.AdamW(codec.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.steps)
    dt = dataset.hand_data["contract"]["dt"]
    print(f"HandVAE: train={len(dataset)}, validation={len(validation)}, dt={dt:g}s, H*dt={dataset.horizon * dt:g}s")
    batches_per_pass = len(dataset) // recipe.batch_size
    best_mae = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, recipe.steps + 1):
        batch = (step - 1) % batches_per_pass
        if batch == 0:
            order = rng.permutation(len(dataset))
        start = batch * recipe.batch_size
        actions, q0 = codec_batch(dataset, order[start : start + recipe.batch_size], recipe.device)
        mu, log_var = codec.encode(actions, q0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        reconstruction = codec.reconstruction_loss(codec.decode(z, q0), codec.normalize_actions(actions, q0))
        kl = diagonal_gaussian_kl(mu, log_var)
        beta = kl_coefficient(step, recipe.beta, recipe.kl_warmup, recipe.kl_ramp)
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ValueError(f"HandVAE loss became nonfinite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), recipe.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % recipe.print_every == 0:
            print(f"Step {step}/{recipe.steps}: loss={float(loss.detach()):g}, beta={beta:g}")
        if step % recipe.eval_every == 0 or step == recipe.steps:
            metrics = evaluate(codec, validation, recipe.batch_size)
            print(f"Validation at {step}: {metrics}")
            if metrics["physical_mae"] < best_mae:
                codec.ready.fill_(True)
                codec.validate_state()
                metadata = dict(
                    model_config=codec.model_config, hand_data=dataset.hand_data,
                    relative_statistics=statistics,
                    loss_reduction={
                        "kl": "mean_over_batch_time_latent",
                        "reconstruction": "weighted_huber_mean_over_batch_valid_frames_joints",
                        "joint_weights": "multiply_numerator_only_do_not_change_denominator",
                    },
                    selection=dict(step=step, beta=beta, split="validation", metric="physical_mae", value=metrics["physical_mae"]),
                    completed_updates=step,
                )
                payload = dict(
                    model_config=codec.model_config,
                    state_dict={k: v.detach().cpu().clone() for k, v in codec.state_dict().items()},
                    metadata=metadata, config=OmegaConf.to_container(cfg, resolve=True), metrics=metrics,
                )
                with atomic_path(output, overwrite=overwrite or best_mae < float("inf")) as temporary:
                    torch.save(payload, temporary)
                best_payload, best_mae = payload, metrics["physical_mae"]
    best_payload["metadata"]["completed_updates"] = recipe.steps
    with atomic_path(output) as temporary:
        torch.save(best_payload, temporary)
    print(f"Saved codec: {output}; selection={best_payload['metadata']['selection']}")
    return best_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="hand_vae", help="Hydra config name or YAML path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. training.device=cpu")
    args = parser.parse_args()
    train(load_policy_config(args.config, args.overrides), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
