"""Resume latest.pt in its original run using only explicit execution settings."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int, set_project_root


def prepare_resume(run, *, device=None, num_workers=None, log_interval=None,
                   use_compile=None, compile_mode=None, wandb=False, zarr_path=None):
    directory = Path(run).expanduser().resolve(strict=True)
    source = directory / "checkpoints" / "latest.pt"
    checkpoint = load_checkpoint(source)
    cfg = OmegaConf.create(checkpoint.effective_config)
    if checkpoint.global_step >= cfg.training.loop.total_train_steps:
        print(f"Training already complete at step {checkpoint.global_step}/{cfg.training.loop.total_train_steps}")
        return None
    with open_dict(cfg):
        cfg.workspace.output_dir = str(directory)
        cfg.workspace.wandb_enabled = wandb
        if device is not None:
            cfg.training.device = device
        if num_workers is not None:
            if num_workers < 0:
                raise ValueError("num_workers must be nonnegative")
            cfg.dataloader.num_workers = num_workers
        if log_interval is not None:
            cfg.training.loop.log_interval_steps = positive_int(log_interval, "log_interval")
        if use_compile is not None:
            cfg.training.use_compile = use_compile
        if compile_mode is not None:
            cfg.training.compile_mode = compile_mode
        if zarr_path is not None:
            cfg.zarr_path = str(Path(zarr_path).expanduser().resolve(strict=True))
            cfg.dataset.zarr_path = cfg.zarr_path
    from dexmani_policy.train import build_trainer

    return build_trainer(cfg, checkpoint=checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--compile", dest="use_compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--zarr-path")
    args = parser.parse_args(argv)
    set_project_root()
    options = vars(args).copy()
    max_updates = options.pop("max_updates")
    if max_updates is not None:
        positive_int(max_updates, "max_updates")
    trainer = prepare_resume(**options)
    if trainer is None:
        return
    from dexmani_policy.train import run_trainer

    run_trainer(trainer, max_updates=max_updates)


if __name__ == "__main__":
    main()
