"""Resume latest.pt with optional execution settings and a new total step budget."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.training.runtime import lock_run_directory


def prepare_resume(
    run, *, device=None, num_workers=None, log_interval=None, use_compile=None,
    compile_mode=None, cudnn_benchmark=None, wandb=False, total_steps=None,
):
    directory = Path(run).expanduser().resolve()
    source = directory / "checkpoints" / "latest.pt"
    checkpoint = load_checkpoint(source)
    cfg = OmegaConf.create(checkpoint.effective_config)
    if total_steps is not None:
        cfg.training.loop.total_train_steps = positive_int(total_steps, "total_steps")
    if checkpoint.global_step >= cfg.training.loop.total_train_steps:
        print(f"Training already complete at step {checkpoint.global_step}/{cfg.training.loop.total_train_steps}")
        return None
    cfg.workspace.output_dir = str(directory)
    cfg.workspace.wandb_enabled = wandb
    if device is not None:
        cfg.training.device = device
    if num_workers is not None:
        cfg.dataloader.num_workers = num_workers
    if log_interval is not None:
        cfg.training.loop.log_interval_steps = log_interval
    if use_compile is not None:
        cfg.training.use_compile = use_compile
    if compile_mode is not None:
        cfg.training.compile_mode = compile_mode
    if cudnn_benchmark is not None:
        cfg.training.cudnn_benchmark = cudnn_benchmark
    from dexmani_policy.training.trainer import build_trainer

    return build_trainer(cfg, checkpoint=checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--compile", dest="use_compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    options = vars(args)
    max_updates = options.pop("max_updates")
    with lock_run_directory(args.run):
        trainer = prepare_resume(**options)
        if trainer is None:
            return
        from dexmani_policy.training.trainer import run_trainer

        run_trainer(trainer, max_updates=max_updates)


if __name__ == "__main__":
    main()
