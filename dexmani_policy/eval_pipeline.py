"""Freeze test seeds, select a checkpoint, then evaluate the saved plan without video."""

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluation.config import build_evaluation_parser, parse_evaluation_overrides
from dexmani_policy.evaluation.pipeline import run_pipeline


def main(argv=None):
    parser = build_evaluation_parser(__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Test episodes per task, reserved before selection; otherwise saved config.",
    )
    args = parser.parse_args(argv)

    set_project_root()
    register_resolvers()
    exp_dir = PROJECT_ROOT / "experiments" / args.policy_name / args.task_name / args.exp_name
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
