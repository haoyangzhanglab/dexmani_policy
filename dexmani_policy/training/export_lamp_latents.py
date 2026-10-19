"""Cache one absolute LAMP latent per real frame, independent of policy queries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.motion_prior import load_motion_prior
from dexmani_policy.training.train_lamp_prior import (
    add_config_arguments, build_prior_dataset, load_policy_config, raw_batch,
    source_rows, validate_source_metadata,
)
from dexmani_policy.utils.atomic import atomic_path
from dexmani_policy.utils.validation import positive_int


def split_diagnostics(dataset, rows, latents, errors, initial_errors, physical_errors):
    if not len(rows):
        return None
    raw_error = physical_errors[rows]
    norms = latents[rows].norm(dim=-1)
    episodes = np.searchsorted(dataset.episode_ends, rows, side="right")
    adjacent = (np.diff(rows) == 1) & (np.diff(episodes) == 0)
    left, right = rows[:-1][adjacent], rows[1:][adjacent]
    jumps = (latents[right] - latents[left]).norm(dim=-1)
    same_target = (dataset.hand_actions[right] == dataset.hand_actions[left]).all(axis=-1)
    same_target_jumps = jumps[same_target]
    reconstruction_increase = errors[rows] - initial_errors[rows]
    return {
        "count": len(rows), "initial_mse": float(initial_errors[rows].mean()),
        "inverted_mse": float(errors[rows].mean()),
        "p95_mse": float(torch.quantile(errors[rows], 0.95)),
        "reconstruction_nonworse_initial_fraction": float((reconstruction_increase <= 0).float().mean()),
        "max_reconstruction_increase": float(reconstruction_increase.max()),
        "physical_joint_rmse": raw_error.square().mean(0).sqrt().tolist(),
        "physical_joint_mae": raw_error.abs().mean(0).tolist(),
        "physical_frame_rmse_p95": float(torch.quantile(raw_error.square().mean(-1).sqrt(), 0.95)),
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_p95": float(torch.quantile(norms, 0.95)),
        "adjacent_same_episode_pairs": int(len(jumps)),
        "latent_jump_mean": float(jumps.mean()) if len(jumps) else None,
        "latent_jump_p95": float(torch.quantile(jumps, 0.95)) if len(jumps) else None,
        "latent_jump_p99": float(torch.quantile(jumps, 0.99)) if len(jumps) else None,
        "latent_jump_max": float(jumps.max()) if len(jumps) else None,
        "same_target_adjacent_pairs": int(len(same_target_jumps)),
        "same_target_latent_jump_mean": float(same_target_jumps.mean()) if len(same_target_jumps) else None,
        "same_target_latent_jump_p95": float(torch.quantile(same_target_jumps, 0.95)) if len(same_target_jumps) else None,
        "same_target_latent_jump_p99": float(torch.quantile(same_target_jumps, 0.99)) if len(same_target_jumps) else None,
        "same_target_latent_jump_max": float(same_target_jumps.max()) if len(same_target_jumps) else None,
    }


def export(args):
    positive_int(args.batch_size, "batch_size")
    if type(args.steps) is not int or args.steps < 0:
        raise ValueError("Inversion steps must be nonnegative")
    if not 0 < args.lr < float("inf"):
        raise ValueError("Inversion lr must be positive and finite")
    if not 0 <= args.center_penalty < float("inf"):
        raise ValueError("Inversion center_penalty must be nonnegative and finite")
    cfg = load_policy_config(args)
    output = Path(args.output or cfg.lamp.latent_path).expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Latent cache exists: {output}; use --overwrite to replace it")
    prior_path = Path(args.prior or cfg.lamp.prior_path).expanduser()
    if output.resolve() == prior_path.resolve():
        raise ValueError("Latent output must not replace the prior checkpoint")
    prior, prior_payload = load_motion_prior(prior_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a small local check")
    prior.to(device)
    dataset = build_prior_dataset(cfg)
    validate_source_metadata(prior_payload["metadata"], dataset)
    expected_config = dict(hand_dim=12, history_steps=int(cfg.lamp.history_steps),
                           latent_dim=int(cfg.lamp.latent_dim), hidden_dim=int(cfg.lamp.hidden_dim))
    if prior.model_config != expected_config:
        raise ValueError("Prior architecture does not match lamp policy configuration")
    train_rows, val_rows = source_rows(dataset)
    rows = np.union1d(train_rows, val_rows)
    frame_valid = np.asarray(dataset.history_valid) & np.isfinite(dataset.hand_actions).all(axis=-1)
    if not frame_valid[rows].all():
        raise ValueError("A policy source row has no finite hand target/history")
    valid = np.zeros(len(dataset.hand_actions), dtype=bool)
    valid[rows] = True
    latents = torch.zeros((len(valid), prior.latent_dim), dtype=torch.float32)
    errors = torch.full((len(valid),), float("nan"), dtype=torch.float32)
    initial_errors = torch.full_like(errors, float("nan"))
    physical_errors = torch.full((len(valid), prior.hand_dim), float("nan"), dtype=torch.float32)
    for start in range(0, len(rows), args.batch_size):
        selected = rows[start:start + args.batch_size]
        history, raw_target = raw_batch(dataset, selected, device)
        with torch.no_grad():
            mu, _ = prior.encode_history(history)
            target = prior.normalize_hand(raw_target)
            initial_errors[selected] = (prior.decode(mu) - target).square().mean(-1).cpu()
        z = prior.invert(raw_target, history, steps=args.steps, lr=args.lr,
                         center_penalty=args.center_penalty)
        with torch.no_grad():
            latents[selected] = z.cpu()
            normalized_error = prior.decode(z) - target
            errors[selected] = normalized_error.square().mean(-1).cpu()
            physical_errors[selected] = (normalized_error / prior.hand_scale).cpu()
        if start == 0 or start + args.batch_size >= len(rows) or start // args.batch_size % 100 == 0:
            print(f"Inverted {min(start + args.batch_size, len(rows))}/{len(rows)} frames")
    latent_scale = latents[train_rows].std(dim=0, correction=0).clamp_min(1e-6)
    if not torch.isfinite(latent_scale).all():
        raise ValueError("Training latent scale became nonfinite")
    metadata = dict(dataset.lamp_metadata)
    metadata["inversion"] = {"steps": args.steps, "lr": args.lr,
                             "initialization": "history_mean", "center_penalty": args.center_penalty,
                             "objective": "mean_hand_reconstruction_plus_center_penalty_mean_latent_squared_distance",
                             "center": "source_history_mean",
                             "center_metric": "mean_squared_raw_latent_distance",
                             "selection": "per_sample_best_penalized_objective_including_initial"}
    metadata["latent_scale_fit"] = "std_absolute_latents_unique_train_action_rows_floor_1e-6"
    metadata["export_scope"] = "unique_train_and_validation_action_source_rows"
    metrics = {name: split_diagnostics(dataset, split_rows, latents, errors,
                                       initial_errors, physical_errors)
               for name, split_rows in (("train", train_rows), ("validation", val_rows))}
    payload = {"_format": "lamp_latents.v1", "metadata": metadata,
               "prior_sha256": prior_payload["_file_sha256"],
               "latents": latents, "valid": torch.from_numpy(valid),
               "reconstruction_error": errors, "latent_scale": latent_scale,
               "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output) as temporary:
        torch.save(payload, temporary)
    print(f"Saved {output}; reconstruction metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--prior", help="Defaults to lamp.prior_path")
    parser.add_argument("--output", help="Defaults to lamp.latent_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--center-penalty", type=float, default=0.0,
                        help="Optional source-history mean anchor; default 0 preserves reconstruction-only inversion")
    parser.add_argument("--overwrite", action="store_true")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
