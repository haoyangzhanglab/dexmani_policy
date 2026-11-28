import argparse
from contextlib import ExitStack, contextmanager
from math import isfinite
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import atomic_path, cleanup_on_exit
from dexmani_policy.common.pytorch_util import (
    dict_apply,
    get_rng_state,
    set_rng_state,
    set_seed,
)
from dexmani_policy.training.config import check_fresh_output, load_config
from dexmani_policy.training.data import (
    build_train_loader,
    close_data_loader,
    fit_observation_normalizer,
    record_data_processing,
    record_training_data,
    validate_resume_data,
)
from dexmani_policy.training.logging import (
    ExperimentLogger,
    accumulate_metrics,
    to_log_scalars,
)
from dexmani_policy.training.optimization import (
    backward_microbatches, build_scheduler, clip_gradients, restore_optimizer,
)
from dexmani_policy.training.runtime import lock_run_directory, training_signals


@contextmanager
def open_run(method, argv=None):
    parser = argparse.ArgumentParser(description=f"Pretrain the {method} observation encoder")
    parser.add_argument("--config", help="Config name or relative path under dexmani_policy/configs")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from", metavar="RUN_DIR", help="Resume the run's checkpoints/latest.pt")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--zarr-path")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    resuming = args.resume_from is not None
    if resuming:
        if (args.config is not None or args.output_dir is not None
                or args.zarr_path is not None or args.overrides):
            parser.error("Resume uses the saved recipe and Zarr. Data and recipe options are fresh-only")

    checkpoint = None
    with ExitStack() as resources:
        if resuming:
            directory = resources.enter_context(lock_run_directory(args.resume_from))
            checkpoint = load_checkpoint(directory / "checkpoints/latest.pt")
            cfg = OmegaConf.create(checkpoint.effective_config)
        else:
            cfg = load_config(args.config or f"pretrain/obs/{method}", args.overrides)
            if args.zarr_path is not None:
                cfg.zarr_path = cfg.dataset.zarr_path = args.zarr_path
            output = args.output_dir if args.output_dir is not None else cfg.workspace.output_dir
            directory = resources.enter_context(lock_run_directory(output, create=True))

        cfg.workspace.output_dir = str(directory)
        if args.no_wandb:
            with open_dict(cfg):
                cfg.workspace.wandb_enabled = False
        if args.device is not None:
            cfg.training.device = args.device
        if args.num_workers is not None:
            cfg.dataloader.num_workers = args.num_workers
        if args.log_interval is not None:
            cfg.training.loop.log_interval_steps = args.log_interval
        yield cfg, checkpoint, args


def export_encoder(checkpoint, output_path):
    state = checkpoint.model_state
    payload: dict[str, Any] = {
        f"{name}_state": {
            key.removeprefix(f"{name}."): value.detach().cpu()
            for key, value in state.items() if key.startswith(f"{name}.")
        }
        for name in ("encoder", "normalizer")
    }
    if not payload["encoder_state"]:
        raise ValueError("Checkpoint has no encoder.* parameters")
    payload["normalization"] = checkpoint.effective_config["normalization"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output_path) as temporary:
        torch.save(payload, temporary)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, *, stop):
    if stop.requested:
        return None
    device = torch.device(device)
    rng_state, training = get_rng_state(device), model.training
    try:
        # Reuse the same diffusion noise across validations without seeding other GPUs.
        validation_seed = 0
        torch.set_rng_state(torch.Generator().manual_seed(validation_seed).get_state())
        if device.type == "cuda":
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(validation_seed).get_state(), device)
        model.eval()
        totals: dict[str, torch.Tensor | float] = {}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            if stop.requested:
                return None
            count = next(iter(batch["obs"].values())).shape[0]
            _, metrics = model(dict_apply(batch, lambda x: x.to(device)))
            accumulate_metrics(totals, metrics, weight=count)
        if stop.requested:
            return None
        metrics = to_log_scalars({f"val/{key}": value / len(dataset) for key, value in totals.items()})
        invalid = {key: value for key, value in metrics.items() if not isfinite(value)}
        if invalid:
            raise ValueError(f"Non-finite observation pretraining validation metrics: {invalid}")
        return metrics
    finally:
        model.train(training)
        set_rng_state(rng_state, device)


def train(cfg, build_model, *, checkpoint=None, max_updates=None, before_optimizer_step=None):
    resuming = checkpoint is not None
    step = checkpoint.global_step if resuming else 0
    if resuming and step >= cfg.training.loop.total_train_steps:
        export_encoder(checkpoint, Path(cfg.workspace.output_dir) / "encoder.pt")
        print(f"Exported completed encoder at step {step}")
        return
    if not resuming:
        check_fresh_output(cfg.workspace.output_dir, "encoder.pt")
    if max_updates is not None:
        positive_int(max_updates, "max_updates")
    positive_int(cfg.training.loop.log_interval_steps, "training.loop.log_interval_steps")
    num_epochs = positive_int(cfg.training.num_epochs, "training.num_epochs")
    save_every = positive_int(cfg.training.save_every_epochs, "training.save_every_epochs")
    positive_int(cfg.training.loop.gradient_accumulation_steps, "training.loop.gradient_accumulation_steps")

    set_seed(cfg.training.seed + step)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    if not resuming:
        record_training_data(cfg, dataset.episode_store, dataset.train_mask, dataset.val_mask)
        record_data_processing(cfg, dataset)
    else:
        validate_resume_data(
            dataset.episode_store, dataset.train_mask, dataset.val_mask, saved=checkpoint.effective_config,
        )
    print(f"[Dataset] {dataset.training_summary()}")
    validation = dataset.get_validation_dataset()
    if validation is not None:
        positive_int(cfg.training.val_every_epochs, "training.val_every_epochs")
        print(f"[Dataset] val windows={len(validation):,}")
    loader = build_train_loader(cfg, dataset)
    batches_per_epoch = len(loader)
    accumulation = cfg.training.loop.gradient_accumulation_steps
    model = build_model(dataset)
    updates_per_epoch = (batches_per_epoch + accumulation - 1) // accumulation
    if not resuming:
        with open_dict(cfg.training.loop):
            cfg.training.loop.total_train_steps = num_epochs * updates_per_epoch
        fit_observation_normalizer(dataset, model.normalizer, cfg.normalization)
    device = torch.device(cfg.training.device)
    model.to(device=device, dtype=torch.float32)
    names, parameters = zip(*((name, p) for name, p in model.named_parameters() if p.requires_grad))
    optimizer_options = dict(cfg.optimizer)
    if "betas" in optimizer_options:
        optimizer_options["betas"] = tuple(optimizer_options["betas"])
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "param_names": list(names)}], **optimizer_options,
    )
    total = cfg.training.loop.total_train_steps
    if resuming:
        # The optimizer already owns these Parameters; restore their values in place.
        model.load_state_dict(checkpoint.model_state, strict=True)
        model.normalizer.validate()
        restore_optimizer(optimizer, checkpoint.optimizer_state)
    scheduler = build_scheduler(
        optimizer, cfg.training.lr_scheduler, cfg.training.lr_warmup_steps, total,
        state=checkpoint.scheduler_state if resuming else None,
    )
    del checkpoint
    stop_step = total if max_updates is None else min(total, step + max_updates)

    def forward(batch):
        count = next(iter(batch["obs"].values())).shape[0]
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        loss, metrics = model(batch)
        return loss, metrics, count

    iterator = None
    logger = ExperimentLogger(cfg, resuming=resuming)
    with cleanup_on_exit(logger.close), cleanup_on_exit(lambda: close_data_loader(loader, iterator)):
        print(f"[{type(model).__name__}] samples={len(dataset)}, batches_per_epoch={batches_per_epoch}, "
              f"batch={cfg.dataloader.batch_size}, accumulation={accumulation}, total_updates={total}")
        with training_signals() as stop:
            initial_step = step
            while step < stop_step:
                model.train()
                iterator = iter(loader)
                for group_start in range(0, batches_per_epoch, accumulation):
                    current_epoch = step // updates_per_epoch
                    group_size = min(accumulation, batches_per_epoch - group_start)
                    record = (
                        step == initial_step
                        or (step + 1) % cfg.training.loop.log_interval_steps == 0
                        or step + 1 == stop_step
                    )
                    optimizer.zero_grad(set_to_none=True)
                    finite_loss, group_metrics = backward_microbatches(
                        iterator, group_size, parameters, forward, device=device, log_metrics=record,
                    )
                    clip_gradients(parameters, finite_loss, step=step)
                    if before_optimizer_step is not None:
                        before_optimizer_step(model, current_epoch)
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    if record:
                        metrics = {f"train/{key}": value for key, value in group_metrics.items()}
                        metrics.update({"train/lr": lr, "train/epoch": current_epoch})
                        logger.log(to_log_scalars(metrics), step)
                    epoch = step // updates_per_epoch
                    epoch_end = step % updates_per_epoch == 0
                    if validation is not None and epoch_end and epoch % cfg.training.val_every_epochs == 0:
                        metrics = evaluate(model, validation, cfg.dataloader.batch_size, device, stop=stop)
                        if metrics is not None:
                            logger.log(metrics, step)
                    periodic = epoch_end and epoch % save_every == 0
                    should_stop = stop.requested or step >= stop_step
                    if periodic or should_stop:
                        saved = Checkpoint(
                            global_step=step,
                            model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict(),
                            effective_config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
                        )
                        save_checkpoint(logger.output_dir / "checkpoints/latest.pt", saved)
                    if should_stop:
                        break
                if should_stop:
                    break
            export_encoder(saved, logger.output_dir / "encoder.pt")
            print(f"[{type(model).__name__}] Saved encoder.pt and checkpoints/latest.pt "
                  f"at epoch={epoch}, step={step}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Export an observation encoder from a resume checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    export_encoder(checkpoint, args.output)


if __name__ == "__main__":
    main()
