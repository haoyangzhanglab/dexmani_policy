"""Check short training and resume runs for a supplied recipe."""

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.training.config import load_config
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import run_trainer, run_training


def run_smoke(cfg, directory: Path, *, max_updates: int, resume_updates: int) -> tuple[int, int]:
    cfg.workspace.output_dir = str(directory)
    cfg.workspace.wandb_cfg = None
    if cfg.training.loop.total_train_steps <= max_updates:
        raise ValueError("The saved training budget must leave updates for resume; reduce --max-updates")
    step = run_training(cfg, max_updates=max_updates)
    assert step == max_updates, "Fresh update count mismatch"
    trainer = prepare_resume(directory)
    final = run_trainer(trainer, max_updates=resume_updates)
    expected = min(step + resume_updates, cfg.training.loop.total_train_steps)
    assert final == expected, "Resume update count mismatch"
    saved = load_checkpoint(directory / "checkpoints" / "latest.pt")
    assert saved.global_step == final, "Resumed checkpoint mismatch"
    return step, final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Recipe name")
    parser.add_argument("--config-dir", type=Path, help="External recipe directory")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override")
    parser.add_argument("--device", help="Override the recipe device")
    parser.add_argument("--output-dir", type=Path, help="Keep test artifacts here instead of a temporary directory")
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--resume-updates", type=int, default=2)
    args = parser.parse_args(argv)
    positive_int(args.max_updates, "max_updates")
    positive_int(args.resume_updates, "resume_updates")
    output = nullcontext(args.output_dir) if args.output_dir is not None else tempfile.TemporaryDirectory(prefix="dexmani-smoke-")
    with output as directory:
        directory = Path(directory).expanduser().resolve()
        cfg = load_config(args.config, args.override, config_dir=args.config_dir)
        if args.device is not None:
            cfg.training.device = args.device
        step, final = run_smoke(cfg, directory, max_updates=args.max_updates, resume_updates=args.resume_updates)
        print(f"PASS {args.config}: fresh={step}, resumed={final}; run={directory}", flush=True)


if __name__ == "__main__":
    main()
