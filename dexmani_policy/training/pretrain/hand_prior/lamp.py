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
from dexmani_policy.common.config import load_policy_config, positive_int
from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.common.hand_representation import describe_hand_data
from dexmani_policy.training.common.hand_representation import shuffled_batches, validate_kl_schedule
from dexmani_policy.training.common.scheduler import get_scheduler
from dexmani_policy.training.validation import validate_config


def build_prior_dataset(cfg):
    """Bootstrap the exact policy split/history rules without opening artifacts."""
    validate_config(cfg)
    if cfg.agent._target_ != "dexmani_policy.agents.policies.lamp.LAMPPolicy":
        raise ValueError("Prior preparation requires a LAMPPolicy configuration")
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare prior artifacts from a fresh policy config, not resume_from")
    if cfg.dataset.split_manifest is None:
        raise ValueError("Pretraining requires an explicit shared dataset.split_manifest")
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


def prior_batch(dataset, rows, device):
    history = torch.as_tensor(dataset.history_for_rows(rows), device=device)
    target = torch.as_tensor(dataset.hand_actions[rows], device=device)
    return history, target


def perturb_history(history, noise_std):
    return history + torch.randn_like(history) * noise_std if noise_std > 0 else history


def kl_weight(update, beta, warmup_steps):
    if warmup_steps == 0 or update >= max(1, warmup_steps - 1):
        return beta
    return beta * update / max(1, warmup_steps - 1)


@torch.no_grad()
def evaluate(prior, dataset, rows, batch_size):
    if not len(rows):
        return None
    training = prior.training
    prior.eval()
    try:
        totals = torch.zeros(3, dtype=torch.float64, device=next(prior.parameters()).device)
        physical_squared = torch.zeros(prior.hand_dim, device=next(prior.parameters()).device)
        physical_absolute = torch.zeros_like(physical_squared)
        frame_rmse = []
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            history, target = prior_batch(dataset, selected, next(prior.parameters()).device)
            mu, log_var = prior.encode_history(history)
            error = prior.decode(mu) - target
            reconstruction = error.square().mean()
            physical_squared += error.square().sum(0)
            physical_absolute += error.abs().sum(0)
            frame_rmse.append(error.square().mean(-1).sqrt().cpu())
            copy_error = (history[:, -1] - target).square().mean()
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
    finally:
        prior.train(training)


def save_prior(path, prior, hand_data, cfg, args, step, metrics, split_manifest):
    prior.ready.fill_(True)
    prior.validate_state()
    evaluation = metrics["evaluation"]
    metadata = dict(
        hand_data=hand_data,
        split_manifest=split_manifest,
        hand_joint_order=prior.hand_joint_order,
        selection=dict(
            selection_step=int(step),
            split="validation" if hand_data["val_episodes"] else "final_no_holdout",
            metric="prior_mean_mse",
            value=evaluation["mse"],
        ),
    )
    payload = {
        "format": prior.artifact_format,
        "coordinate_space": prior.coordinate_space,
        "model_config": dict(prior.model_config),
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in prior.state_dict().items()
        },
        "metadata": metadata,
        "policy_config": OmegaConf.to_container(cfg, resolve=False),
        "train_args": vars(args),
        "step": int(step),
        "metrics": metrics,
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
    eligible_step = max(2, args.beta_warmup) if args.beta_warmup else 1
    validate_kl_schedule(args.steps, args.beta, eligible_step)
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
    if not len(train_rows):
        raise ValueError("LAMP prior training requires nonempty training source rows")
    drop_last = len(train_rows) >= args.batch_size
    if not drop_last:
        print(
            f"LAMP prior small-data smoke exception: actual batch size={len(train_rows)} "
            f"< configured batch size={args.batch_size}; keeping one partial batch per pass"
        )
    prior = HandMotionPrior(
        hand_dim=12,
        history_steps=int(cfg.agent.history_steps),
        latent_dim=int(cfg.agent.latent_dim),
        hidden_dim=int(cfg.agent.prior_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        prior.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(
        optimizer, "cosine_min_lr", num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps, lr_min_ratio=args.min_lr / args.lr,
    )
    print(f"Prior rows: train={len(train_rows)}, validation={len(val_rows)}; output={output}")
    initial = evaluate(prior, dataset, val_rows, args.batch_size)
    if initial is not None:
        print(f"Initial validation: {initial}")
    best_mse = float("inf")
    batches = shuffled_batches(train_rows, args.batch_size, rng, drop_last=drop_last)
    for update in range(args.steps):
        step = update + 1
        prior.train()
        history, target = prior_batch(dataset, next(batches), device)
        history = perturb_history(history, args.noise_std)
        beta = kl_weight(update, args.beta, args.beta_warmup)
        loss, reconstruction, kl = prior.compute_loss(history, target, beta)
        if not torch.isfinite(loss):
            raise ValueError(f"Prior loss became nonfinite at step {step}")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        report = step == 1 or step % args.print_every == 0
        select = (step % args.eval_every == 0 or step == args.steps) and step >= eligible_step
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
            if not len(val_rows) and step != args.steps:
                continue
            evaluation = evaluate(
                prior, dataset, val_rows if len(val_rows) else train_rows, args.batch_size
            )
            last_metrics["evaluation"] = evaluation
            split = "Validation" if len(val_rows) else "Train (no holdout)"
            print(f"{split} at {step}: {evaluation}")
            if evaluation["mse"] < best_mse:
                best_mse = evaluation["mse"]
                save_prior(
                    output, prior, hand_data, cfg, args, step, last_metrics,
                    dataset.data_recipe["split_manifest"],
                )
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
        help="Gaussian history noise in raw hand radians",
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
