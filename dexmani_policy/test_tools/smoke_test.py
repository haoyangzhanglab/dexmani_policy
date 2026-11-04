"""Run one recipe through bounded fresh training, saving and original-run resume."""

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int, set_project_root
from dexmani_policy.train import run_trainer, run_training
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.resume import prepare_resume


def run_smoke(cfg, directory: Path, *, max_updates: int, resume_updates: int) -> tuple[int, int]:
    positive_int(max_updates, "max_updates")
    positive_int(resume_updates, "resume_updates")
    cfg.workspace.output_dir = str(directory)
    cfg.workspace.wandb_cfg = None
    if cfg.training.loop.total_train_steps <= max_updates:
        raise ValueError("The saved training budget must leave updates for resume; reduce --max-updates")
    step = run_training(cfg, max_updates=max_updates)
    assert step == max_updates, "Fresh update count mismatch"
    trainer = prepare_resume(directory)
    assert trainer is not None
    assert trainer.global_step == step, "Resume source mismatch"
    final = run_trainer(trainer, max_updates=resume_updates)
    expected = min(step + resume_updates, cfg.training.loop.total_train_steps)
    assert final == expected, "Resume update count mismatch"
    saved = load_checkpoint(directory / "checkpoints" / "latest.pt")
    assert saved.global_step == final, "Resumed checkpoint mismatch"
    return step, final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="One config name or relative path under dexmani_policy/configs")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--resume-updates", type=int, default=2)
    args = parser.parse_args(argv)
    set_project_root()
    output = nullcontext(args.output_dir) if args.output_dir is not None else tempfile.TemporaryDirectory(prefix="dexmani-smoke-")
    with output as directory:
        directory = Path(directory).expanduser().resolve()
        cfg = load_policy_config(args.config, args.override)
        if args.device is not None:
            cfg.training.device = args.device
        step, final = run_smoke(
            cfg, directory, max_updates=args.max_updates, resume_updates=args.resume_updates,
        )
        print(f"PASS {args.config}: fresh={step}, resumed={final}; run={directory}", flush=True)


if __name__ == "__main__":
    main()
