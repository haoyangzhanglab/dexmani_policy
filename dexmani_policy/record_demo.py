"""Record viewer videos for explicit or deterministically selected task seeds."""

import argparse
from pathlib import Path

from dexmani_policy.common.config import (
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.env_runner.common.seeds import (
    split_evaluation_seeds,
    validate_task_seeds,
)
from dexmani_policy.evaluation.config import (
    add_evaluation_arguments,
    parse_evaluation_overrides,
    resolve_evaluation_parameter,
    resolve_evaluation_request,
)
from dexmani_policy.evaluation.rollout import (
    build_evaluation_runner,
    evaluate_policy,
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
    request = resolve_evaluation_request(
        load_experiment_config(exp_dir),
        exp_dir,
        args.ckpt_tag,
        parse_evaluation_overrides(args.overrides),
        section="demo",
        eval_seed=args.seed,
        use_ema=args.use_ema,
        action_steps=args.action_steps,
        selection_record=args.selection_record,
        inference_steps_list=[args.inference_steps] if args.inference_steps is not None else None,
    )
    cfg = request.config
    runner = build_evaluation_runner(cfg)
    pools = task_seed_pools(runner)
    if args.seeds is not None:
        plan = {task: list(args.seeds) for task in pools}
        validate_task_seeds(plan, pools)
        for task, seeds in plan.items():
            if set(seeds) - set(pools[task]):
                raise ValueError(f"Unavailable demo seeds for {task}")
    else:
        episodes = resolve_evaluation_parameter(
            cfg, "episodes", "demo", override=args.episodes, default=5
        )
        plan, _ = split_evaluation_seeds(pools, cfg.eval.seed, episodes)

    resolution = args.resolution or resolve_evaluation_parameter(
        cfg, "viewer_resolution", "demo", default=[1280, 960]
    )
    for leaf in runner.task_runners.values():
        leaf.render_mode = "human"
        leaf.viewer_resolution = tuple(resolution)
        if args.fps is not None:
            leaf.env_video_fps = args.fps
    output_root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else exp_dir / "demo_videos"
    )
    evaluate_policy(
        request,
        runner,
        plan,
        output_root=output_root,
        record_video=True,
    )


if __name__ == "__main__":
    main()
