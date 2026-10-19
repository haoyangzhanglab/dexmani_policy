"""Pretrain the temporal hand VAE on exactly the policy's qualified windows.

python -m dexmani_policy.training.train_temporal_codec --config temporal_codec_flow
    zarr_path=/path/to/data.zarr dataset.obs_alignment=obs_before_action
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.temporal_vae import TemporalHandVAE
from dexmani_policy.datasets.temporal_codec_dataset import TemporalCodecDataset
from dexmani_policy.training.build_utils import validate_config
from dexmani_policy.utils.atomic import atomic_path
from dexmani_policy.utils.config import load_policy_config, resolve_normalization_spec
from dexmani_policy.utils.hand_artifact import (
    metadata_sha256,
    source_contract,
    validate_selection_schedule,
)
from dexmani_policy.utils.validation import positive_int


def build_codec_dataset(cfg):
    validate_config(cfg)
    if (
        cfg.agent._target_
        != "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy"
    ):
        raise ValueError(
            "Codec preparation requires a TemporalLatentFlowPolicy configuration"
        )
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare codec from a fresh policy config, not resume_from")
    # Keep policy eligibility; raw pretraining batches bypass observation augmentation.
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not isinstance(dataset, TemporalCodecDataset):
        raise TypeError("Temporal codec pretraining requires TemporalCodecDataset")
    return dataset


def fit_relative_statistics(dataset, batch_size=4096, std_floor=1e-6):
    """Population moments over train windows, counting repeated/padded frames."""
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
        raise ValueError("Temporal codec train relative statistics must be finite")
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32)), {
        "population": "train_policy_windows_including_repeated_source_rows_and_padding",
        "windows": len(dataset), "frames": count, "std_floor": float(std_floor),
    }


def raw_batch(dataset, indices, device):
    actions, q0 = dataset.raw_hand_batch(indices)
    return torch.as_tensor(actions, device=device), torch.as_tensor(q0, device=device)


def kl_coefficient(step, beta, warmup, ramp):
    return beta * min(1.0, max(0.0, (step - warmup) / max(1, ramp)))


@torch.no_grad()
def evaluate(codec, dataset, batch_size):
    was_training = codec.training
    codec.eval()
    absolute = torch.zeros(12, device=codec.action_mean.device)
    reconstruction = kl_sum = 0.0
    for start in range(0, len(dataset), batch_size):
        indices = np.arange(start, min(start + batch_size, len(dataset)))
        actions, q0 = raw_batch(dataset, indices, codec.action_mean.device)
        mu, log_var = codec.encode(actions, q0)
        decoded = codec.decode(mu, q0)
        physical = decoded * codec.action_std + codec.action_mean + q0[:, None]
        absolute += (physical - actions).abs().sum(dim=(0, 1))
        reconstruction += float(codec.reconstruction_loss(decoded, codec.normalize_actions(actions, q0))) * len(indices)
        kl_sum += float(-0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()) * len(indices)
    joint_mae = absolute / (len(dataset) * dataset.horizon)
    codec.train(was_training)
    metrics = {"physical_mae": float(joint_mae.mean()), "physical_joint_mae": joint_mae.tolist(),
               "reconstruction": reconstruction / len(dataset), "kl": kl_sum / len(dataset),
               "windows": len(dataset)}
    if not all(math.isfinite(metrics[key]) for key in ("physical_mae", "reconstruction", "kl")):
        raise ValueError("Temporal codec evaluation became nonfinite")
    return metrics


def save_codec(path, codec, dataset, cfg, args, step, metrics, statistics):
    codec.ready.fill_(True)
    codec.validate_state()
    contract = source_contract(dataset)
    evaluation = metrics["evaluation"]
    metadata = dict(
        implementation=dataset.hand_implementation_version,
        source_contract=contract,
        source_contract_sha256=metadata_sha256(contract),
        model_config=codec.model_config,
        normalization_spec=resolve_normalization_spec(cfg),
        selection=dict(
            selection_step=int(step),
            beta=metrics["beta"],
            target_beta=args.beta,
            eligible_step=args.kl_warmup + max(1, args.kl_ramp),
            split=metrics["selection_split"],
            metric="physical_mae",
            value=evaluation["physical_mae"],
            reconstruction=evaluation["reconstruction"],
            kl=evaluation["kl"],
            physical_joint_mae=evaluation["physical_joint_mae"],
        ),
    )
    metadata["relative_statistics"] = statistics
    metadata["loss_reduction"] = {
        "kl": "mean_over_batch_time_latent",
        "reconstruction": "weighted_huber_mean_over_batch_valid_frames_joints",
        "joint_weights": "multiply_numerator_only_do_not_change_denominator",
    }
    payload = {
        "_format": "temporal_hand_codec.v1", "model_config": dict(codec.model_config),
        "state_dict": {key: value.detach().cpu().clone() for key, value in codec.state_dict().items()},
        "metadata": metadata, "policy_config": OmegaConf.to_container(cfg, resolve=False),
        "train_args": vars(args), "step": int(step), "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def train(args):
    for key in ("steps", "batch_size", "eval_every", "print_every"):
        positive_int(getattr(args, key), key)
    for key in ("kl_warmup", "kl_ramp"):
        if getattr(args, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("beta", "weight_decay"):
        if not 0 <= getattr(args, key) < float("inf"):
            raise ValueError(f"{key} must be nonnegative and finite")
    if not 0 < args.lr < float("inf") or not 0 < args.clip_grad < float("inf"):
        raise ValueError("lr and clip_grad must be positive and finite")
    validate_selection_schedule(args.steps, args.beta, args.kl_warmup + max(1, args.kl_ramp))
    cfg = load_policy_config(args.policy_config or args.config, args.overrides)
    output = Path(cfg.codec.path).expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Codec already exists: {output}; use --overwrite to replace it")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a small local check")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = build_codec_dataset(cfg)
    source_contract(dataset)
    validation = dataset.get_validation_dataset()
    selection = "validation" if validation is not None else "final_no_holdout"
    selection_dataset = validation if validation is not None else dataset
    codec = TemporalHandVAE(
        hand_dim=12, horizon=dataset.horizon, latent_dim=int(cfg.codec.latent_dim),
        hidden_dim=int(cfg.codec.hidden_dim), n_layers=int(cfg.codec.n_layers),
        n_heads=int(cfg.codec.n_heads), mlp_ratio=float(cfg.codec.mlp_ratio),
    ).to(device)
    weights = torch.as_tensor(list(cfg.codec.joint_weights), dtype=torch.float32, device=device)
    if weights.shape != (12,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("codec.joint_weights must contain 12 positive finite values")
    codec.joint_weights.copy_(weights)
    mean, std, statistics = fit_relative_statistics(dataset, args.batch_size)
    codec.action_mean.copy_(mean.to(device))
    codec.action_std.copy_(std.to(device))
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    print(f"Codec windows: train={len(dataset)}, validation={0 if validation is None else len(validation)}; "
          f"selection={selection}, metric=physical MAE; output={output}")
    print(f"Relative statistics: {statistics}")
    best_mae = float("inf")
    order, cursor = rng.permutation(len(dataset)), 0
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order, cursor = rng.permutation(len(dataset)), 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += len(indices)
        actions, q0 = raw_batch(dataset, indices, device)
        mu, log_var = codec.encode(actions, q0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        reconstruction = codec.reconstruction_loss(codec.decode(z, q0), codec.normalize_actions(actions, q0))
        kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()
        beta = kl_coefficient(step, args.beta, args.kl_warmup, args.kl_ramp)
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ValueError(f"Temporal codec loss became nonfinite at step {step}")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        metrics = {"loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()),
                   "kl": float(kl.detach()), "beta": beta, "lr": lr}
        if step == 1 or step % args.print_every == 0:
            print(f"Step {step}/{args.steps}: {metrics}")
        eligible = step >= args.kl_warmup + max(1, args.kl_ramp)
        if eligible and ((validation is not None and step % args.eval_every == 0) or step == args.steps):
            evaluation = evaluate(codec, selection_dataset, args.batch_size)
            metrics.update(selection_split=selection, evaluation=evaluation)
            print(f"{selection} at {step}: {evaluation}")
            if validation is None or evaluation["physical_mae"] < best_mae:
                best_mae = evaluation["physical_mae"]
                save_codec(output, codec, dataset, cfg, args, step, metrics, statistics)
    print(f"Saved {selection} codec: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", default="temporal_codec_flow", help="Hydra policy config name")
    group.add_argument("--policy-config", help="Policy YAML path instead of a config name")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. zarr_path=...")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=3e-4)
    parser.add_argument("--kl-warmup", type=int, default=4000)
    parser.add_argument("--kl-ramp", type=int, default=4000)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
