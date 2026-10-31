"""Run one recipe through bounded fresh training, saving and original-run resume."""

import argparse
import tempfile
from contextlib import contextmanager
from pathlib import Path

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int, set_project_root
from dexmani_policy.train import run_training
from dexmani_policy.training.common.config import load_policy_config, prepare_training
from dexmani_policy.training.common.resume import prepare_resume


@contextmanager
def smoke_output(output_dir):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="dexmani-smoke-") as temporary:
            yield Path(temporary)
    else:
        directory = output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        yield directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="One config name or relative path under dexmani_policy/configs")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--resume-updates", type=int, default=2)
    args = parser.parse_args(argv)
    positive_int(args.max_updates, "max_updates")
    positive_int(args.resume_updates, "resume_updates")
    set_project_root()
    with smoke_output(args.output_dir) as directory:
        cfg = load_policy_config(args.config, args.override)
        cfg.workspace.output_dir = str(directory)
        cfg.workspace.wandb_cfg = None
        if args.device is not None:
            cfg.training.device = args.device
        cfg = prepare_training(cfg)
        if cfg.training.loop.total_train_steps <= args.max_updates:
            raise ValueError("The saved training budget must leave updates for resume; reduce --max-updates")
        step = run_training(cfg, max_updates=args.max_updates)
        assert step == args.max_updates, "Fresh update count mismatch"
        saved = load_checkpoint(directory / "checkpoints" / "latest.pt")
        assert saved.global_step == step, "Saved update count mismatch"
        del saved
        resumed_cfg, checkpoint = prepare_resume(directory, device=args.device)
        assert checkpoint.global_step == step, "Resume source mismatch"
        final = run_training(resumed_cfg, checkpoint=checkpoint, max_updates=args.resume_updates)
        expected = min(step + args.resume_updates, cfg.training.loop.total_train_steps)
        assert final == expected, "Resume update count mismatch"
        saved = load_checkpoint(directory / "checkpoints" / "latest.pt")
        assert saved.global_step == final, "Resumed checkpoint mismatch"
        print(f"PASS {args.config}: fresh={step}, resumed={final}; run={directory}", flush=True)


if __name__ == "__main__":
    main()
