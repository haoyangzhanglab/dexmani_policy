"""Select a milestone checkpoint with fixed-seed evaluation."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import load_experiment_config, register_resolvers, set_project_root
from dexmani_policy.env_runner.evaluation import (
    add_evaluation_arguments,
    parse_eval_overrides,
)
from dexmani_policy.env_runner.selection import select_best_checkpoint

ROOT_DIR = set_project_root()
register_resolvers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument(
        "--initial-episodes", type=int, help="Initial episodes per checkpoint/task."
    )
    parser.add_argument(
        "--batch-size", type=int, help="Extra episodes for tied candidates; 0 disables."
    )
    parser.add_argument(
        "--max-episodes", type=int, help="Per-task episode cap including tie breaks."
    )
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument(
        "--videos", dest="record_video", action="store_true", help="Record candidate rollouts."
    )
    videos.add_argument(
        "--no-videos", dest="record_video", action="store_false", help="Disable videos (default)."
    )
    parser.set_defaults(record_video=False)
    parser.add_argument("--output-dir", help="Empty directory under experiment/eval_ckpt_selector.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Evaluation/environment dotlist; eval.offline.episodes sets the reserved test count.",
    )
    args = parser.parse_args(argv)

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = OmegaConf.merge(load_experiment_config(exp_dir), parse_eval_overrides(args.overrides))
    select_best_checkpoint(
        exp_dir,
        cfg,
        initial_episodes=args.initial_episodes,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        inference_steps=args.inference_steps,
        action_steps=args.action_steps,
        use_ema=args.use_ema,
        eval_seed=args.seed,
        record_video=args.record_video,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
