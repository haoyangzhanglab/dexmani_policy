"""Record viewer videos for explicit or deterministically selected task seeds."""

import argparse
from pathlib import Path

from dexmani_policy.common.config import load_experiment_config, register_resolvers, set_project_root
from dexmani_policy.env_runner.common.seeds import (
    split_eval_seeds,
    validate_task_seeds,
)
from dexmani_policy.env_runner.evaluation import (
    add_evaluation_arguments,
    build_eval_runner,
    evaluate_policy,
    get_eval_param,
    iter_leaf_env_runners,
    resolve_eval_request,
    task_seed_pools,
)

ROOT_DIR = set_project_root()
register_resolvers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--ckpt-tag", default="best", help="best, latest, or 20pct..100pct.")
    parser.add_argument("--episodes", type=int, help="Episodes per task; defaults to saved config.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Physical seeds for every task; overrides episodes."
    )
    parser.add_argument("--output-dir", help="Parent directory for this run's videos.")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Viewer size; defaults to saved demo config, then 1280 960.",
    )
    parser.add_argument("--fps", type=int, help="Override saved video or environment FPS.")
    parser.add_argument("--selection-record", help="Path to a completed selection_result.json.")
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args(argv)

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg, use_ema, inference_steps, selection, checkpoint_path = resolve_eval_request(
        load_experiment_config(exp_dir),
        exp_dir,
        args.ckpt_tag,
        args.overrides,
        section="demo",
        eval_seed=args.seed,
        use_ema=args.use_ema,
        action_steps=args.action_steps,
        selection_record=args.selection_record,
        inference_steps_list=[args.inference_steps] if args.inference_steps is not None else None,
    )
    runner = build_eval_runner(cfg)
    pools = task_seed_pools(runner)
    if args.seeds is not None:
        plan = {task: list(args.seeds) for task in pools}
        validate_task_seeds(plan, pools)
        for task, seeds in plan.items():
            if set(seeds) - set(pools[task]):
                raise ValueError(f"Unavailable demo seeds for {task}")
    else:
        episodes = get_eval_param(cfg, "episodes", "demo", override=args.episodes, default=5)
        plan, _ = split_eval_seeds(pools, cfg.eval.seed, episodes)

    resolution = args.resolution or get_eval_param(
        cfg, "viewer_resolution", "demo", default=[1280, 960]
    )
    for leaf in iter_leaf_env_runners(runner):
        leaf.render_mode = "human"
        leaf.viewer_resolution = tuple(resolution)
        if args.fps is not None:
            leaf.env_video_fps = args.fps
    output_root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else exp_dir / "demo_videos"
    )
    evaluate_policy(
        cfg,
        runner,
        checkpoint_path,
        plan,
        use_ema=use_ema,
        inference_steps_list=inference_steps,
        action_steps=cfg.eval.demo.action_steps,
        output_root=output_root,
        record_video=True,
        selection=selection,
    )


if __name__ == "__main__":
    main()
