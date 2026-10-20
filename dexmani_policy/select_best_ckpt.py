"""Select a milestone checkpoint with fixed-seed evaluation."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.env_runner.evaluation import (
    add_inference_steps_argument,
    get_eval_param,
    parse_eval_overrides,
)
from dexmani_policy.env_runner.selection import select_best_checkpoint
from dexmani_policy.utils.config import load_experiment_config, register_resolvers, set_project_root

ROOT_DIR = set_project_root()
register_resolvers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--initial-episodes", type=int)
    parser.add_argument(
        "--batch-size", type=int, help="Extra episodes for tied candidates; 0 disables."
    )
    parser.add_argument("--max-episodes", type=int)
    add_inference_steps_argument(parser)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--ema", dest="use_ema", action="store_true", default=None)
    weights.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--seed", type=int)
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument("--videos", dest="record_video", action="store_true")
    videos.add_argument("--no-videos", dest="record_video", action="store_false")
    parser.set_defaults(record_video=False)
    parser.add_argument("--output-dir", help="Empty directory under experiment/eval_ckpt_selector.")
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args()

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
