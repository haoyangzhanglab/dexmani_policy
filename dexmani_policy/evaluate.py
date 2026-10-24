"""Evaluate a checkpoint on held-out task seeds, optionally sweeping NFE."""

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluation.config import build_evaluation_parser, parse_evaluation_overrides
from dexmani_policy.evaluation.rollout import evaluate_checkpoint


def main(argv=None):
    parser = build_evaluation_parser(__doc__)
    parser.add_argument("--ckpt-tag", default="best", help="best, latest, or 20pct..100pct.")
    parser.add_argument(
        "--ckpt-path", help="Path inside experiment/checkpoints; overrides --ckpt-tag."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Test episodes per task; defaults to selection record, then saved config. Conflicts fail.",
    )
    parser.add_argument("--no-videos", action="store_true", help="Override saved video settings.")
    parser.add_argument("--selection-record", help="Path to a completed selection_result.json.")
    args = parser.parse_args(argv)

    set_project_root()
    register_resolvers()
    exp_dir = PROJECT_ROOT / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = load_experiment_config(exp_dir)
    evaluate_checkpoint(
        exp_dir,
        cfg,
        checkpoint=args.ckpt_path or args.ckpt_tag,
        episodes=args.episodes,
        use_ema=args.use_ema,
        action_steps=args.action_steps,
        inference_steps_list=[args.inference_steps] if args.inference_steps is not None else None,
        overrides=parse_evaluation_overrides(args.overrides),
        eval_seed=args.seed,
        selection_record=args.selection_record,
        record_video=False if args.no_videos else None,
    )


if __name__ == "__main__":
    main()
