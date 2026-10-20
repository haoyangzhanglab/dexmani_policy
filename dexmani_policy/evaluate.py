"""Evaluate a checkpoint on held-out task seeds, optionally sweeping NFE."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.env_runner.evaluation import (
    add_evaluation_arguments,
    evaluate_checkpoint,
)
from dexmani_policy.utils.config import load_experiment_config, register_resolvers, set_project_root

ROOT_DIR = set_project_root()
register_resolvers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--ckpt-tag", default="best", help="best, latest, or 20pct..100pct.")
    parser.add_argument(
        "--ckpt-path", help="Path inside experiment/checkpoints; overrides --ckpt-tag."
    )
    parser.add_argument("--episodes", type=int, help="Episodes per task; defaults to saved config.")
    parser.add_argument("--no-videos", action="store_true", help="Override saved video settings.")
    parser.add_argument("--selection-record", help="Path to a completed selection_result.json.")
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args(argv)

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
