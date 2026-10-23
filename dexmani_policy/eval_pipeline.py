"""Freeze test seeds, select a checkpoint, then evaluate the saved plan without video."""

import argparse
from pathlib import Path

from dexmani_policy.common.config import (
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluation.config import add_evaluation_arguments, parse_evaluation_overrides
from dexmani_policy.evaluation.pipeline import run_pipeline

ROOT_DIR = set_project_root()
register_resolvers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Test episodes per task, reserved before selection; otherwise saved config.",
    )
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args(argv)
    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = load_experiment_config(exp_dir)
    run_pipeline(
        exp_dir,
        cfg,
        episodes=args.episodes,
        eval_seed=args.seed,
        inference_steps=args.inference_steps,
        action_steps=args.action_steps,
        overrides=parse_evaluation_overrides(args.overrides),
        use_ema=args.use_ema,
    )


if __name__ == "__main__":
    main()
