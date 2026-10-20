"""Record viewer videos for explicit or deterministically selected task seeds."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.env_runner.common.seeds import select_eval_seeds, validate_task_seeds
from dexmani_policy.env_runner.evaluation import (
    add_inference_steps_argument,
    build_eval_runner,
    evaluate_policy,
    get_eval_param,
    iter_leaf_env_runners,
    resolve_eval_request,
    resolve_eval_seed,
    task_seed_pools,
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
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Physical seeds for every task; overrides episodes."
    )
    add_inference_steps_argument(parser)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--ema", dest="use_ema", action="store_true", default=None)
    weights.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--output-dir", help="Parent directory for this run's videos.")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=int)
    parser.add_argument("--selection-record", help="Path to a completed selection_result.json.")
    args = parser.parse_args()

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg, use_ema, steps, selection, checkpoint = resolve_eval_request(
        OmegaConf.create(load_experiment_config(exp_dir)),
        exp_dir,
        args.ckpt_tag,
        section="demo",
        use_ema=args.use_ema,
        selection_record=args.selection_record,
        inference_steps_list=[args.inference_steps] if args.inference_steps is not None else None,
    )
    runner = build_eval_runner(cfg)
    if args.seeds is not None:
        pools = task_seed_pools(runner)
        plan = {task: list(args.seeds) for task in pools}
        validate_task_seeds(plan, pools)
        for task, seeds in plan.items():
            if set(seeds) - set(pools[task]):
                raise ValueError(f"Unavailable demo seeds for {task}")
    else:
        episodes = (
            args.episodes
            if args.episodes is not None
            else get_eval_param(cfg, "episodes", "demo", default=5)
        )
        plan = select_eval_seeds(task_seed_pools(runner), resolve_eval_seed(cfg), episodes)
    resolution = args.resolution or get_eval_param(
        cfg, "viewer_resolution", "demo", default=[1280, 960]
    )
    for leaf in iter_leaf_env_runners(runner):
        leaf.render_mode = "human"
        leaf.viewer_resolution = tuple(resolution)
        if args.fps is not None:
            leaf.env_video_fps = args.fps
    root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else exp_dir / "demo_videos"
    )
    evaluate_policy(
        cfg,
        runner,
        checkpoint,
        plan,
        use_ema=use_ema,
        inference_steps_list=steps,
        output_root=root,
        record_video=True,
        selection=selection,
    )


if __name__ == "__main__":
    main()
