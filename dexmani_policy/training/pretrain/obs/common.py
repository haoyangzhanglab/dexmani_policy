import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import positive_int
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import validate_normalizer_state
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state
from dexmani_policy.training.common.logging import accumulate_metrics


def load_run(method):
    parser = argparse.ArgumentParser(description=f"Pretrain the {method} observation encoder")
    parser.add_argument("--config", help="Config name or relative path under dexmani_policy/configs")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--zarr-path")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.max_updates is not None:
        positive_int(args.max_updates, "max_updates")
    checkpoint = None
    if args.resume_from is not None:
        if args.config is not None or args.output_dir is not None or args.overrides:
            parser.error("Resume reads the saved recipe in its original run; --config, --output-dir and overrides are fresh-only")
        source = Path(args.resume_from).expanduser().resolve(strict=True)
        if source.is_dir():
            source = source / "resume.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(checkpoint["effective_config"])
        cfg.workspace.output_dir = str(source.parent)
    else:
        cfg = load_policy_config(args.config or f"pretrain/obs/{method}", args.overrides)
        if args.output_dir is not None:
            cfg.workspace.output_dir = args.output_dir
    with open_dict(cfg):
        cfg.resume_from = args.resume_from
        cfg.workspace.wandb_enabled = not args.no_wandb
    if args.device is not None:
        cfg.training.device = args.device
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers must be nonnegative")
        cfg.dataloader.num_workers = args.num_workers
    if args.log_interval is not None:
        positive_int(args.log_interval, "--log-interval")
        cfg.training.loop.log_interval_steps = args.log_interval
    if args.zarr_path is not None:
        cfg.zarr_path = args.zarr_path
        cfg.dataset.zarr_path = cfg.zarr_path
    if checkpoint is None:
        for name in ("encoder.pt", "resume.pt"):
            path = Path(cfg.workspace.output_dir) / name
            if path.exists():
                raise FileExistsError(f"Pretraining output already exists: {path}")
    return cfg, checkpoint, args.max_updates


def export_encoder(model, encoder, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "normalizer_state": {key: value.detach().cpu() for key, value in model.normalizer.state_dict().items()},
    }
    with atomic_path(output_path) as temporary:
        torch.save(payload, temporary)


def save_run(model, optimizer, scheduler, cfg, step):
    output = Path(cfg.workspace.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    checkpoint = {
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "global_step": step,
        "rng_state": get_rng_state(cfg.training.device), "effective_config": config,
    }
    with atomic_path(output / "resume.pt") as temporary:
        torch.save(checkpoint, temporary)


def restore_run(model, optimizer, scheduler, checkpoint, *, device) -> int:
    # The optimizer already owns these Parameters, so restore their values in place.
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validate_normalizer_state(model.normalizer)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    set_rng_state(checkpoint["rng_state"], device)
    return checkpoint["global_step"]


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, *, stop_requested: Callable[[], bool]):
    if stop_requested():
        return None
    rng_state, training = get_rng_state(device), model.training
    try:
        model.eval()
        totals: dict[str, torch.Tensor] = {}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            if stop_requested():
                return None
            count = len(next(iter(batch["obs"].values())))
            _, metrics = model(dict_apply(batch, lambda x: x.to(device)))
            accumulate_metrics(totals, {key: value * count for key, value in metrics.items()})
            if stop_requested():
                return None
        return {f"val/{key}": value / len(dataset) for key, value in totals.items()}
    finally:
        model.train(training)
        set_rng_state(rng_state, device)


def main():
    parser = argparse.ArgumentParser(description="Export an observation encoder from a resume checkpoint")
    parser.add_argument("--method", required=True, choices=("afro", "cordvip"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["effective_config"])
    if args.method == "afro":
        from dexmani_policy.training.pretrain.obs.afro import AFROPretrain

        model = AFROPretrain(cfg.normalization)
        encoder = model.visual
    else:
        from dexmani_policy.training.pretrain.obs.cordvip import CordViPPretrain

        model = CordViPPretrain(cfg.normalization, **cfg.model)
        encoder = model.obs_encoder
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validate_normalizer_state(model.normalizer)
    export_encoder(model, encoder, args.output)


if __name__ == "__main__":
    main()
