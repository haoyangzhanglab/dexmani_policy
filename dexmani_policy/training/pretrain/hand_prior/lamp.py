"""Train the LAMP prior on eligible policy source rows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.training.builder import build_normalizer
from dexmani_policy.training.validation import validate_config
from dexmani_policy.utils.config import load_policy_config, positive_int, resolve_normalization_spec
from dexmani_policy.utils.hand_representation import (
    describe_hand_data,
    validate_kl_schedule,
)
from dexmani_policy.utils.runtime import atomic_path


def build_prior_dataset(cfg):
    """Bootstrap the exact policy split/history rules without opening artifacts."""
    validate_config(cfg)
    if cfg.agent._target_ != "dexmani_policy.agents.core.lamp_policy.LAMPPolicy":
        raise ValueError("Prior preparation requires a LAMPPolicy configuration")
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare prior artifacts from a fresh policy config, not resume_from")
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not isinstance(dataset, LAMPDataset):
        raise TypeError("LAMP prior preparation requires LAMPDataset")
    return dataset


def source_rows(dataset):
    train = np.asarray(dataset.sampler.action_source_rows, dtype=np.int64)
    validation = dataset.get_validation_dataset()
    val = (
        np.asarray(validation.sampler.action_source_rows, dtype=np.int64)
        if validation is not None
        else np.empty(0, dtype=np.int64)
    )
    return train, val


def raw_batch(dataset, rows, device):
    history = torch.as_tensor(dataset.history_for_rows(rows), device=device)
    target = torch.as_tensor(dataset.hand_actions[rows], device=device)
    return history, target


@torch.no_grad()
def evaluate(prior, dataset, rows, batch_size):
    if not len(rows):
        return None
    totals = torch.zeros(3, dtype=torch.float64, device=prior.hand_scale.device)
    physical_squared = torch.zeros(prior.hand_dim, device=prior.hand_scale.device)
    physical_absolute = torch.zeros_like(physical_squared)
    frame_rmse = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        history, raw_target = raw_batch(dataset, selected, prior.hand_scale.device)
        target = prior.normalize_hand(raw_target)
        mu, log_var = prior.encode_history(history)
        normalized_error = prior.decode(mu) - target
        reconstruction = normalized_error.square().mean()
        physical_error = normalized_error / prior.hand_scale
        physical_squared += physical_error.square().sum(0)
        physical_absolute += physical_error.abs().sum(0)
        frame_rmse.append(physical_error.square().mean(-1).sqrt().cpu())
        copy_error = (prior.normalize_hand(history[:, -1]) - target).square().mean()
        kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()
        totals += torch.stack((reconstruction, copy_error, kl)).double() * len(selected)
    mse, copy_mse, kl = (totals / len(rows)).tolist()
    metrics = {
        "mse": mse,
        "copy_mse": copy_mse,
        "kl": kl,
        "count": len(rows),
        "physical_joint_rmse": (physical_squared / len(rows)).sqrt().tolist(),
        "physical_joint_mae": (physical_absolute / len(rows)).tolist(),
        "physical_frame_rmse_p95": float(torch.quantile(torch.cat(frame_rmse), 0.95)),
    }
    if not all(math.isfinite(metrics[key]) for key in ("mse", "copy_mse", "kl")):
        raise ValueError("Prior evaluation became nonfinite")
    return metrics


def save_prior(path, prior, hand_data, cfg, args, step, metrics, optimizer):
    prior.ready.fill_(True)
    prior.validate_state()
    evaluation = metrics["evaluation"]
    metadata = dict(
        hand_data=hand_data,
        normalization_spec=resolve_normalization_spec(cfg),
        selection=dict(
            selection_step=int(step),
            split="validation" if hand_data["val_episodes"] else "final_no_holdout",
            metric="prior_mean_mse",
            value=evaluation["mse"],
        ),
    )
    payload = {
        "model_config": dict(prior.model_config),
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in prior.state_dict().items()
        },
        "metadata": metadata,
        "policy_config": OmegaConf.to_container(cfg, resolve=False),
        "train_args": vars(args),
        "step": int(step),
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def train(args):
    for key in ("steps", "batch_size", "eval_every", "print_every"):
        positive_int(getattr(args, key), key)
    if args.beta_warmup < 0 or args.warmup_steps < 0:
        raise ValueError("Warmup steps must be nonnegative")
    for key in ("noise_std", "weight_decay", "min_lr"):
        if not 0 <= getattr(args, key) < float("inf"):
            raise ValueError(f"{key} must be nonnegative and finite")
    if not 0 < args.lr < float("inf") or not 0 < args.clip_grad < float("inf"):
        raise ValueError("lr and clip_grad must be positive and finite")
    if args.min_lr > args.lr:
        raise ValueError("min_lr must not exceed lr")
    validate_kl_schedule(args.steps, args.beta, max(1, args.beta_warmup))
    cfg = load_policy_config(args.policy_config or args.config, args.overrides)
    output = Path(args.output or cfg.agent.prior_path).expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Prior already exists: {output}; use --overwrite to replace it")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = build_prior_dataset(cfg)
    hand_data = describe_hand_data(dataset)
    train_rows, val_rows = source_rows(dataset)
    prior = HandMotionPrior(
        hand_dim=12,
        history_steps=int(cfg.agent.history_steps),
        latent_dim=int(cfg.agent.latent_dim),
        hidden_dim=int(cfg.agent.prior_hidden_dim),
    ).to(device)
    spec = resolve_normalization_spec(cfg)
    normalizer = build_normalizer(dataset, {"action": spec["action"]}, cfg.action_key)
    params = normalizer["action"].params_dict
    prior.hand_scale.copy_(params["scale"][7:19].to(device))
    prior.hand_offset.copy_(params["offset"][7:19].to(device))
    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Prior rows: train={len(train_rows)}, validation={len(val_rows)}; output={output}")
    initial = evaluate(prior, dataset, val_rows, args.batch_size)
    if initial is not None:
        print(f"Initial validation: {initial}")
    best_mse = float("inf")
    order = rng.permutation(train_rows)
    cursor = 0
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order, cursor = rng.permutation(train_rows), 0
        rows = order[cursor : cursor + args.batch_size]
        cursor += len(rows)
        history, target_raw = raw_batch(dataset, rows, device)
        if args.noise_std:
            # Apply normalized-coordinate noise to the physical command history.
            history = history + torch.randn_like(history) * args.noise_std / prior.hand_scale
        mu, log_var = prior.encode_history(history)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        reconstruction = (prior.decode(z) - prior.normalize_hand(target_raw)).square().mean()
        kl = -0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()
        beta = args.beta * min(1.0, step / max(1, args.beta_warmup))
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ValueError(f"Prior loss became nonfinite at step {step}")
        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        report = step == 1 or step % args.print_every == 0
        select = (step % args.eval_every == 0 or step == args.steps) and step >= max(
            1, args.beta_warmup
        )
        if report or select:
            last_metrics = {
                "loss": float(loss.detach()),
                "reconstruction": float(reconstruction.detach()),
                "kl": float(kl.detach()),
                "beta": beta,
                "lr": lr,
            }
        if report:
            print(f"Step {step}/{args.steps}: {last_metrics}")
        if select:
            if len(val_rows):
                evaluation = evaluate(prior, dataset, val_rows, args.batch_size)
                last_metrics["evaluation"] = evaluation
                print(f"Validation at {step}: {evaluation}")
                if evaluation["mse"] < best_mse:
                    best_mse = evaluation["mse"]
                    save_prior(output, prior, hand_data, cfg, args, step, last_metrics, optimizer)
            elif step == args.steps:
                last_metrics["evaluation"] = evaluate(prior, dataset, train_rows, args.batch_size)
                save_prior(output, prior, hand_data, cfg, args, step, last_metrics, optimizer)
    print(f"Saved {'best validation' if len(val_rows) else 'final'} prior: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", default="lamp", help="Hydra policy config name")
    group.add_argument("--policy-config", help="Policy YAML path instead of a config name")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. zarr_path=...")
    parser.add_argument("--output", help="Defaults to agent.prior_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--beta-warmup", type=int, default=2000)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Gaussian history noise in normalized hand coordinates",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
