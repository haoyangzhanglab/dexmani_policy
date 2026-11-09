import argparse
import os
import signal
from collections.abc import Callable
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path, close_resources
from dexmani_policy.common.normalizer import validate_normalizer_state
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state, set_rng_state, set_seed
from dexmani_policy.training.common import training_signal_handlers
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.data import (
    build_train_loader, close_data_loader, epoch_position, fit_normalizer,
)
from dexmani_policy.training.common.logging import ExperimentLogger, accumulate_metrics, to_log_scalars
from dexmani_policy.training.common.optimization import build_scheduler


def load_run(method, argv=None):
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
    args = parser.parse_args(argv)
    checkpoint = None
    if args.resume_from is not None:
        if args.config is not None or args.output_dir is not None or args.overrides:
            parser.error("Resume reads the saved recipe from the specified run; --config, --output-dir and overrides are fresh-only")
        source = Path(args.resume_from).expanduser().resolve(strict=True)
        if source.is_dir():
            directory = source
            source = directory / "checkpoints/latest.pt"
        else:
            directory = source.parent.parent
        checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        cfg = OmegaConf.create(checkpoint["effective_config"])
        cfg.workspace.output_dir = str(directory)
    else:
        cfg = load_policy_config(args.config or f"pretrain/obs/{method}", args.overrides)
        if args.output_dir is not None:
            cfg.workspace.output_dir = args.output_dir
    if args.no_wandb:
        with open_dict(cfg):
            cfg.workspace.wandb_enabled = False
    if args.device is not None:
        cfg.training.device = args.device
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers must be nonnegative")
        cfg.dataloader.num_workers = args.num_workers
    if args.log_interval is not None:
        cfg.training.loop.log_interval_steps = args.log_interval
    if args.zarr_path is not None:
        cfg.zarr_path = args.zarr_path
        cfg.dataset.zarr_path = cfg.zarr_path
    return cfg, checkpoint, args.max_updates


def export_encoder(state, normalization, output_path):
    payload = {
        f"{name}_state": {
            key.removeprefix(f"{name}."): value.detach().cpu()
            for key, value in state.items() if key.startswith(f"{name}.")
        }
        for name in ("encoder", "normalizer")
    }
    payload["normalization"] = dict(normalization)
    if not payload["encoder_state"]:
        raise ValueError("Checkpoint has no encoder.* parameters")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with atomic_path(checkpoint_dir / "latest.pt") as temporary:
        torch.save(checkpoint, temporary)


def restore_run(model, optimizer, scheduler, checkpoint) -> int:
    # The optimizer already owns these Parameters, so restore their values in place.
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validate_normalizer_state(model.normalizer)
    if optimizer.param_groups[0]["param_names"] != checkpoint["optimizer_state"]["param_groups"][0]["param_names"]:
        raise ValueError("Optimizer parameter order changed")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
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


def train(method, argv=None):
    cfg, checkpoint, max_updates = load_run(method, argv)
    resuming = checkpoint is not None
    if resuming and checkpoint["global_step"] >= cfg.training.loop.total_train_steps:
        print(f"Pretraining already completed at step {checkpoint['global_step']}")
        return
    if not resuming:
        for name in ("config.yaml", "metrics.jsonl", "encoder.pt", "checkpoints"):
            path = Path(cfg.workspace.output_dir) / name
            if path.exists():
                raise FileExistsError(f"Pretraining output already exists: {path}")
    if max_updates is not None:
        positive_int(max_updates, "max_updates")
    positive_int(cfg.training.loop.log_interval_steps, "training.loop.log_interval_steps")
    is_afro = method == "afro"
    if is_afro:
        positive_int(cfg.training.num_epochs, "training.num_epochs")
        positive_int(cfg.training.save_every_epochs, "training.save_every_epochs")
        if cfg.training.loop.gradient_accumulation_steps != 1:
            raise ValueError("AFRO VICReg requires accumulation=1")
    else:
        positive_int(cfg.training.save_every_updates, "training.save_every_updates")
        positive_int(cfg.training.loop.total_train_steps, "training.loop.total_train_steps")
        positive_int(cfg.training.loop.gradient_accumulation_steps, "training.loop.gradient_accumulation_steps")

    set_seed(cfg.training.seed)
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    missing = dataset.normalization_fields.keys() - cfg.normalization.keys()
    if missing:
        raise ValueError(f"Missing normalization for inputs: {sorted(missing)}")
    cfg.normalization = {key: cfg.normalization[key] for key in dataset.normalization_fields}
    print(f"[Dataset] {dataset.training_summary()}")
    validation = dataset.get_validation_dataset()
    if validation is not None:
        positive_int(cfg.training.val_every_epochs, "training.val_every_epochs")
        print(f"[Dataset] val windows={len(validation):,}")
    loader = build_train_loader(cfg, dataset)
    batches_per_epoch = loader.batch_sampler.batches_per_epoch
    accumulation = cfg.training.loop.gradient_accumulation_steps
    model: torch.nn.Module
    if is_afro:
        from dexmani_policy.training.pretrain.obs.afro import AFRO

        model = AFRO(cfg.normalization)
        if not resuming:
            with open_dict(cfg.training.loop):
                cfg.training.loop.total_train_steps = cfg.training.num_epochs * batches_per_epoch
    else:
        from dexmani_policy.training.pretrain.obs.cordvip import CordViP

        model = CordViP(cfg.normalization, n_obs_steps=dataset.obs_horizon, **cfg.model)
    if not resuming:
        fit_normalizer(dataset, cfg.normalization, model.normalizer)
    device = torch.device(cfg.training.device)
    model.to(device=device, dtype=torch.float32)
    names, parameters = zip(*((name, p) for name, p in model.named_parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW([dict(params=parameters, param_names=list(names))], **cfg.optimizer)
    total = cfg.training.loop.total_train_steps
    scheduler = build_scheduler(optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps, total)
    step = 0
    rng_state = None
    if resuming:
        step = restore_run(model, optimizer, scheduler, checkpoint)
        rng_state = checkpoint["rng_state"]
    del checkpoint
    epoch, start_batch = epoch_position(step, batches_per_epoch, accumulation)
    if resuming:
        print(f"Resume epoch={epoch}, batch={start_batch}, global_step={step}")
    stop_step = total if max_updates is None else min(total, step + max_updates)
    logger = ExperimentLogger(**cfg.workspace, resume=resuming)
    print(f"[{type(model).__name__}] samples={len(dataset)}, batches_per_epoch={batches_per_epoch}, "
          f"batch={cfg.dataloader.batch_size}, accumulation={accumulation}, total_updates={total}")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            print(f"\nSecond {signal.Signals(signum).name} — forcing exit.", flush=True)
            os._exit(1)
        print(f"\n{signal.Signals(signum).name} — finishing current update, then saving.", flush=True)
        stop_requested = True

    iterator = None
    try:
        logger.save_config(cfg)
        if rng_state is not None:
            set_rng_state(rng_state, device)
        with training_signal_handlers(request_stop):
            initial_step = step
            while step < stop_step and not stop_requested:
                epoch, start_batch = epoch_position(step, batches_per_epoch, accumulation)
                loader.batch_sampler.set_epoch(epoch, start_batch)
                model.train()
                current_epoch = epoch
                iterator = iter(loader)
                for group_start in range(start_batch, batches_per_epoch, accumulation):
                    group_size = min(accumulation, batches_per_epoch - group_start)
                    record = (
                        step == initial_step
                        or (step + 1) % cfg.training.loop.log_interval_steps == 0
                        or step + 1 == stop_step
                    )
                    totals: dict[str, torch.Tensor] = {}
                    optimizer.zero_grad(set_to_none=True)
                    finite_loss = torch.ones((), dtype=torch.bool, device=device)
                    for _ in range(group_size):
                        batch = next(iterator)
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss, metrics = model(batch)
                        finite_loss &= torch.isfinite(loss.detach())
                        (loss / group_size).backward()
                        if record:
                            accumulate_metrics(totals, metrics)
                    if not finite_loss:
                        raise RuntimeError(f"Nonfinite {type(model).__name__} loss at step {step}")
                    if is_afro:
                        # AFRO updates the independent teacher before the online optimizer step.
                        model.update_teacher(current_epoch)
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    if group_start + group_size == batches_per_epoch:
                        epoch = current_epoch + 1
                    if record:
                        logger.log(
                            to_log_scalars(
                                {"train/" + key: value / group_size for key, value in totals.items()}
                                | {
                                    "train/lr": lr,
                                    "train/epoch": current_epoch,
                                }
                            ),
                            step,
                        )
                    if step >= stop_step or stop_requested:
                        break
                    if not is_afro and step % cfg.training.save_every_updates == 0:
                        save_run(model, optimizer, scheduler, cfg, step)
                if stop_requested:
                    break
                if (
                    validation is not None
                    and epoch > current_epoch
                    and epoch % cfg.training.val_every_epochs == 0
                ):
                    metrics = evaluate(
                        model, validation, cfg.dataloader.batch_size, device,
                        stop_requested=lambda: stop_requested,
                    )
                    if metrics is not None:
                        logger.log(to_log_scalars(metrics), step)
                if (
                    is_afro and epoch > current_epoch
                    and epoch % cfg.training.save_every_epochs == 0
                    and step < stop_step and not stop_requested
                ):
                    save_run(model, optimizer, scheduler, cfg, step)
            save_run(model, optimizer, scheduler, cfg, step)
            export_encoder(model.state_dict(), cfg.normalization, Path(cfg.workspace.output_dir) / "encoder.pt")
            print(f"[{type(model).__name__}] Saved raw encoder.pt and checkpoints/latest.pt at epoch={epoch}, step={step}/{total}")
    finally:
        close_resources(lambda: close_data_loader(loader, iterator), logger.close)


def main():
    parser = argparse.ArgumentParser(description="Export an observation encoder from a resume checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    export_encoder(checkpoint["model_state"], checkpoint["effective_config"]["normalization"], args.output)


if __name__ == "__main__":
    main()
