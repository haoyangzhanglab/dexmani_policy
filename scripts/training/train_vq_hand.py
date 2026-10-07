"""Train a single-step hand-state VQ-VAE for DQ-RISE.

Use --policy-config for DQ-RISE: the target Policy Dataset decides splits,
valid windows and unique source rows; its action normalizer supplies hand stats.
Without this option the historical independent recipe (full-data stats) is kept.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.normalization import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    resolve_normalization_spec,
)
from dexmani_policy.agents.vq_hand import VQVAEHand
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.sampler import downsample_mask, get_val_mask
from dexmani_policy.training.run_identity import claim_run
from dexmani_policy.training.build_utils import build_normalizer
from dexmani_policy.utils.config import register_resolvers

_project_root = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_vq(vqvae, loader, device):
    """Sample means, independent of validation batch partitioning."""
    sums = {"enc": 0.0, "vq": 0.0, "mse": 0.0}
    count = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        enc, vq, _, mse = vqvae(batch)
        n = batch.shape[0]
        count += n
        for key, value in (("enc", enc), ("vq", vq), ("mse", mse)):
            sums[key] += float(value) * n
    if not count:
        raise ValueError("VQ validation loader is empty")
    return {key: value / count for key, value in sums.items()}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_mask_to_frame_indices(
    episode_ends: np.ndarray, episode_mask: np.ndarray
) -> np.ndarray:
    mask = np.asarray(episode_mask, dtype=bool)
    if len(mask) != len(episode_ends):
        raise ValueError("episode mask and episode_ends have different lengths")
    starts = np.concatenate(([0], episode_ends[:-1]))
    chunks = [
        np.arange(start, end, dtype=np.int64)
        for use, start, end in zip(mask, starts, episode_ends)
        if use
    ]
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)


def _save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    vqvae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    normalizer: LinearNormalizer,
    args: argparse.Namespace,
    train_history: list[float],
    split_metadata: dict,
    metrics: dict,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": vqvae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
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
        },
        "split_metadata": split_metadata,
        "metrics": metrics,
        "train_history": train_history,
        "format_version": 3,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from dexmani_policy.utils.atomic import atomic_path
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)
    logger.info("checkpoint saved: %s", path)


def prepare_standalone_data(args):
    buffer = ReplayBuffer.open(args.zarr_path, keys=[args.action_key])
    all_actions = buffer.read(args.action_key, slice(None))
    hand_data = all_actions[:, args.tcp_dim :]
    if hand_data.shape[1] != args.hand_dim:
        raise ValueError(
            f"Configured hand_dim={args.hand_dim}, but data gives "
            f"{hand_data.shape[1]} after tcp_dim={args.tcp_dim}"
        )

    episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
    val_episode_mask = get_val_mask(
        seed=args.seed,
        val_ratio=args.val_ratio,
        n_episodes=len(episode_ends),
    )
    train_episode_mask = downsample_mask(
        seed=args.seed,
        mask=~val_episode_mask,
        max_n=args.max_train_episodes,
    )
    train_indices = _episode_mask_to_frame_indices(episode_ends, train_episode_mask)
    val_indices = _episode_mask_to_frame_indices(episode_ends, val_episode_mask)
    if len(train_indices) == 0:
        raise ValueError("No training frames selected")

    logger.info(
        "episodes: train=%d, val=%d, excluded=%d; frames: train=%d, val=%d",
        int(train_episode_mask.sum()),
        int(val_episode_mask.sum()),
        int((~train_episode_mask & ~val_episode_mask).sum()),
        len(train_indices),
        len(val_indices),
    )

    # Preserve this VQ recipe: full-dataset hand statistics, including validation.
    # Policy uses valid train-window source rows and checks codebook compatibility;
    # matching dataset paths alone does not guarantee equal affine parameters.
    normalizer = LinearNormalizer()
    normalizer.fit(
        data={"hand": hand_data},
        mode="limits",
        range_eps=1e-4,
    )
    train_norm = (
        normalizer["hand"]
        .normalize(hand_data[train_indices])
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    val_norm = (
        normalizer["hand"]
        .normalize(hand_data[val_indices])
        .cpu()
        .numpy()
        .astype(np.float32)
        if len(val_indices) > 0
        else np.empty((0, args.hand_dim), dtype=np.float32)
    )

    split_metadata = {
        "episode_ends": episode_ends.tolist(),
        "train_episode_ids": np.flatnonzero(train_episode_mask).tolist(),
        "val_episode_ids": np.flatnonzero(val_episode_mask).tolist(),
        "train_frame_count": len(train_indices),
        "val_frame_count": len(val_indices),
    }
    return train_norm, val_norm, normalizer, split_metadata


def load_policy_config(path, overrides=()):
    """Compose the target config without constructing a model or runtime."""
    register_resolvers()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy config not found: {path}")
    config_root = _project_root / "dexmani_policy" / "configs"
    if path.is_relative_to(config_root):
        config_name = path.relative_to(config_root).with_suffix("").as_posix()
    else:
        # External standalone YAML keeps absolute defaults relative to its own root.
        config_root, config_name = path.parent, path.stem
    with hydra.initialize_config_dir(config_dir=str(config_root), version_base=None):
        return hydra.compose(config_name=config_name, overrides=list(overrides))


def build_policy_dataset(cfg):
    """Reuse Policy's actual Dataset, including observation/window qualification."""
    if cfg.agent._target_ != "dexmani_policy.agents.core.dqrise.DQRISEAgent":
        raise ValueError("--policy-config requires a DQRISEAgent configuration")
    if cfg.dataset.get("use_aux_ee", False):
        raise ValueError(
            "DQ-RISE VQ preparation does not support auxiliary action layouts"
        )
    layout = {"action": (7, 19), "action_ee": (9, 21)}.get(cfg.action_key)
    if layout is None or (cfg.agent.tcp_dim, cfg.agent.action_dim) != layout:
        raise ValueError("Supported DQ-RISE layouts are action=7+12 or action_ee=9+12")
    if cfg.dataset.action_key != cfg.action_key:
        raise ValueError("Policy and Dataset action_key must match")
    if cfg.get("resume_from") is not None:
        raise ValueError(
            "Prepare a new codebook from a fresh Policy config, not resume_from"
        )
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not isinstance(dataset, BaseDataset):
        raise TypeError("DQ-RISE VQ preparation requires a single BaseDataset")
    if dataset.replay_buffer[dataset.action_key].shape[1:] != (layout[1],):
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
        selected = (
            rows[np.searchsorted(rows, start) : np.searchsorted(rows, start + step)]
            - start
        )
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
    spec = resolve_normalization_spec(cfg)
    if spec.get("action", "identity") == "identity":
        raise ValueError("DQ-RISE codebooks require a fitted action normalizer")
    action = build_normalizer(dataset, {"action": spec["action"]}, cfg.action_key)[
        "action"
    ]
    hand_slice = slice(int(cfg.agent.tcp_dim), int(cfg.agent.action_dim))
    normalizer = LinearNormalizer()
    normalizer["hand"] = SingleFieldLinearNormalizer.create_manual(
        action.params_dict["scale"][hand_slice].clone(),
        action.params_dict["offset"][hand_slice].clone(),
        {key: value[hand_slice].clone() for key, value in action.input_stats.items()},
    )
    val = dataset.get_validation_dataset()
    train_hand = policy_hand_rows(dataset, cfg.agent.tcp_dim)
    val_hand = policy_hand_rows(val, cfg.agent.tcp_dim)
    metadata = {
        "normalization_scope": "unique_valid_policy_train_action_source_rows",
        "policy_config": OmegaConf.to_container(cfg, resolve=False),
        "resolved_dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "normalization_spec": spec,
        "data_recipe": dataset.data_recipe,
        "data_revision": dataset.data_revision,
        "episode_ends": dataset.replay_buffer.episode_ends.tolist(),
        "train_episode_ids": np.flatnonzero(dataset.train_mask).tolist(),
        "val_episode_ids": np.flatnonzero(dataset.val_mask).tolist(),
        "train_source_rows": dataset.sampler.action_source_rows.tolist(),
        "val_source_rows": val.sampler.action_source_rows.tolist()
        if val is not None
        else [],
        "train_frame_count": len(train_hand),
        "val_frame_count": len(val_hand),
        "train_windows": dataset.sampler.validity_summary,
        "val_windows": val.sampler.validity_summary if val is not None else None,
    }
    metadata["resolved_dataset"]["zarr_path"] = dataset.zarr_path
    # Only low-dimensional hand samples are materialized for the existing VQ trainer.
    return (
        normalizer["hand"].normalize(train_hand).numpy(),
        normalizer["hand"].normalize(val_hand).numpy(),
        normalizer,
        metadata,
    )


def train(args: argparse.Namespace, *, policy_cfg) -> None:
    from dexmani_policy.utils.validation import positive_int
    for name in ("num_epochs", "batch_size", "save_epochs", "codebook_report_epochs"):
        positive_int(getattr(args, name), name)
    if args.output_dir is None:
        run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        args.output_dir = str(Path("experiments/vq_hand") / Path(args.zarr_path).stem / run)
    output_dir = Path(args.output_dir)
    claim_run(output_dir)
    from dexmani_policy.training.source_snapshot import save_source_snapshot
    save_source_snapshot(output_dir)
    logger.info("VQ run: %s", output_dir.resolve())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if policy_cfg is not None:
        train_norm, val_norm, normalizer, split_metadata = prepare_policy_data(
            policy_cfg
        )
    else:
        train_norm, val_norm, normalizer, split_metadata = prepare_standalone_data(args)

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
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader has zero batches")

    vqvae = VQVAEHand(
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
    total_steps = max(1, len(train_loader) * args.num_epochs)
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

    selection_metric = "val_mse" if val_loader is not None else "train_mse"
    split_metadata["selection_metric"] = selection_metric
    train_history: list[float] = []
    best_mse = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        vqvae.train()
        sums = {"enc": 0.0, "vq": 0.0, "mse": 0.0}
        train_count = 0
        usage = torch.zeros(args.num_groups, args.codebook_size, dtype=torch.long)

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            enc_loss, vq_loss, indices, recon_mse = vqvae(batch)
            total_loss = args.enc_loss_weight * enc_loss + args.vq_loss_weight * vq_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for group in range(args.num_groups):
                    usage[group] += torch.bincount(
                        indices[:, group].detach().cpu(),
                        minlength=args.codebook_size,
                    )
            train_count += batch.shape[0]
            sums["enc"] += float(enc_loss) * batch.shape[0]
            sums["vq"] += float(vq_loss) * batch.shape[0]
            sums["mse"] += float(recon_mse) * batch.shape[0]

        train_metrics = {key: value / train_count for key, value in sums.items()}
        train_total = (
            args.enc_loss_weight * train_metrics["enc"]
            + args.vq_loss_weight * train_metrics["vq"]
        )
        train_history.append(train_total)

        val_metrics = {"enc": float("nan"), "vq": float("nan"), "mse": float("nan")}
        if val_loader is not None:
            vqvae.eval()
            val_metrics = evaluate_vq(vqvae, val_loader, device)

        metrics = {
            "train_enc": train_metrics["enc"],
            "train_vq": train_metrics["vq"],
            "train_mse": train_metrics["mse"],
            "train_total": train_total,
            "val_enc": val_metrics["enc"],
            "val_vq": val_metrics["vq"],
            "val_mse": val_metrics["mse"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.info(
            "epoch %4d | lr %.2e | train enc %.5f vq %.5f mse %.5f | val enc %.5f vq %.5f mse %.5f",
            epoch,
            metrics["lr"],
            metrics["train_enc"],
            metrics["train_vq"],
            metrics["train_mse"],
            metrics["val_enc"],
            metrics["val_vq"],
            metrics["val_mse"],
        )

        if epoch % args.save_epochs == 0 or epoch == args.num_epochs:
            _save_checkpoint(
                output_dir / f"vqvae_hand_epoch={epoch:04d}.pt",
                epoch=epoch,
                vqvae=vqvae,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                args=args,
                train_history=train_history,
                split_metadata=split_metadata,
                metrics=metrics,
            )

        selection_mse = metrics[selection_metric]
        if not np.isfinite(selection_mse):
            raise FloatingPointError(
                f"Non-finite {selection_metric} at epoch {epoch}; previous best preserved"
            )
        if selection_mse < best_mse:
            best_mse = selection_mse
            _save_checkpoint(
                output_dir / "vqvae_hand_best.pt",
                epoch=epoch,
                vqvae=vqvae,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                args=args,
                train_history=train_history,
                split_metadata=split_metadata,
                metrics=metrics,
            )

        if epoch % args.codebook_report_epochs == 0:
            for group in range(args.num_groups):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(np.arange(args.codebook_size), usage[group].numpy())
                ax.set_title(f"Group {group} code usage — epoch {epoch}")
                ax.set_xlabel("Code index")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / f"code_usage_g{group}_epoch{epoch:04d}.png")
                plt.close(fig)

    final_metrics = metrics
    _save_checkpoint(
        output_dir / "vqvae_hand_last.pt",
        epoch=args.num_epochs,
        vqvae=vqvae,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        args=args,
        train_history=train_history,
        split_metadata=split_metadata,
        metrics=final_metrics,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training objective")
    ax.set_title("VQ-VAE training objective")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)


def _parse_loss_weight(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",")]


def _load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("YAML config must be a mapping")
    return dict(config.get("vq_vae", config))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--policy-config",
        default=None,
        help="Target Policy YAML; overrides independent data/split options",
    )
    parser.add_argument(
        "--policy-override",
        action="append",
        default=[],
        help="Hydra override for the target Policy (repeatable)",
    )
    parser.add_argument("--zarr_path", default=None)
    parser.add_argument("--action_key", default=None)
    parser.add_argument("--tcp_dim", type=int, default=None)
    parser.add_argument("--hand_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_groups", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--act_scale", type=float, default=None)
    parser.add_argument("--loss_weight", type=_parse_loss_weight, default=None)
    parser.add_argument("--vq_decay", type=float, default=None)
    parser.add_argument("--threshold_ema_dead_code", type=int, default=None)
    parser.add_argument(
        "--kmeans_init",
        type=lambda value: value.lower() in ("true", "1", "yes"),
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
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--max_train_episodes", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_epochs", type=int, default=None)
    parser.add_argument("--codebook_report_epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_parser.add_argument("--policy-config", default=None)
    config_parser.add_argument("--policy-override", action="append", default=[])
    config_args, remaining = config_parser.parse_known_args(argv)
    if config_args.config:
        config_path = Path(config_args.config)
        if not config_path.is_absolute():
            config_path = _project_root / config_path
        parser.set_defaults(**_load_yaml_config(str(config_path)))
    policy_cfg = None
    if config_args.policy_config:
        policy_cfg = load_policy_config(
            config_args.policy_config, config_args.policy_override
        )
        if not config_args.config:
            parser.set_defaults(
                **OmegaConf.to_container(policy_cfg.vq_vae, resolve=True)
            )
    parser.set_defaults(
        policy_config=config_args.policy_config,
        policy_override=config_args.policy_override,
    )
    args = parser.parse_args(remaining)
    if policy_cfg is None and not config_args.config:
        parser.error("DQ codebooks require --policy-config; independent VQ research requires explicit --config")
    if policy_cfg is not None:
        args.zarr_path = str(policy_cfg.dataset.zarr_path)
        args.action_key = policy_cfg.action_key
        args.tcp_dim = int(policy_cfg.agent.tcp_dim)
        args.hand_dim = int(policy_cfg.agent.action_dim) - args.tcp_dim
        args.val_ratio = policy_cfg.dataset.get("val_ratio", 0.0)
        args.max_train_episodes = policy_cfg.dataset.get("max_train_episodes")
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
    if args.zarr_path is None:
        parser.error("--zarr_path is required")
    if args.hand_dim is None or args.hand_dim <= 0:
        buffer = ReplayBuffer.open(args.zarr_path, keys=[args.action_key])
        args.hand_dim = int(buffer[args.action_key].shape[-1] - args.tcp_dim)
    if args.loss_weight is None:
        args.loss_weight = [1.0] * args.hand_dim
    train(args, policy_cfg=policy_cfg)


if __name__ == "__main__":
    main()
