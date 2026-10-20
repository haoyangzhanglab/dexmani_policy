"""Evaluate a checkpoint on held-out task seeds, optionally sweeping NFE."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.env_runner.evaluation import (
    add_inference_steps_argument,
    evaluate_checkpoint,
)
from dexmani_policy.utils.config import load_experiment_config, register_resolvers, set_project_root

ROOT_DIR = set_project_root()
register_resolvers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--ckpt-tag", default="best", help="best, latest, or 20pct..100pct.")
    parser.add_argument("--ckpt-path", help="Explicit checkpoint path, overrides --ckpt-tag.")
    parser.add_argument("--episodes", type=int)
    add_inference_steps_argument(parser)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--ema", dest="use_ema", action="store_true", default=None)
    weights.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--selection-record", help="Path to a completed selection_result.json.")
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args()

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = OmegaConf.create(load_experiment_config(exp_dir))
    evaluate_checkpoint(
        exp_dir,
        cfg,
        checkpoint=args.ckpt_path or args.ckpt_tag,
        episodes=args.episodes,
        use_ema=args.use_ema,
        inference_steps_list=[args.inference_steps] if args.inference_steps is not None else None,
        overrides=args.overrides,
        selection_record=args.selection_record,
        record_video=False if args.no_videos else None,
    )


if __name__ == "__main__":
    main()
