"""Select a milestone checkpoint with fixed-seed evaluation."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.env_runner.evaluation import (
    add_evaluation_arguments,
    get_eval_param,
    parse_eval_overrides,
)
from dexmani_policy.env_runner.selection import select_best_checkpoint
from dexmani_policy.utils.config import load_experiment_config, register_resolvers, set_project_root

ROOT_DIR = set_project_root()
register_resolvers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--initial-episodes", type=int, help="Initial episodes per checkpoint/task.")
    parser.add_argument(
        "--batch-size", type=int, help="Extra episodes for tied candidates; 0 disables."
    )
    parser.add_argument("--max-episodes", type=int, help="Per-task episode cap including tie breaks.")
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument(
        "--videos", dest="record_video", action="store_true", help="Record candidate rollouts."
    )
    videos.add_argument(
        "--no-videos", dest="record_video", action="store_false", help="Disable videos (default)."
    )
    parser.set_defaults(record_video=False)
    parser.add_argument("--output-dir", help="Empty directory under experiment/eval_ckpt_selector.")
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args(argv)

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = OmegaConf.merge(load_experiment_config(exp_dir), parse_eval_overrides(args.overrides))
    options = {}
    for name, default in (
        ("initial_episodes", 25),
        ("batch_size", 5),
        ("max_episodes", 100),
        ("inference_steps", 10),
        ("use_ema", True),
    ):
        value = getattr(args, name)
        options[name] = (
            value
            if value is not None
            else get_eval_param(cfg, name, "select_best", default=default)
        )
    select_best_checkpoint(
        exp_dir,
        cfg,
        **options,
        eval_seed=args.seed,
        record_video=args.record_video,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
