"""Demo video recording — high-resolution viewer capture for presentations.

Loads a trained checkpoint and records evaluation episodes using the SAPIEN
viewer at high resolution (default 1280×960), suitable for making demo videos.

Key differences from ``eval_best_ckpt.py``:

- Uses ``render_mode="human"`` → viewer is created → video frames come from
  ``get_viewer_rgb()`` at the configured viewer resolution (default 1280×960),
  instead of the 320×240 sensor camera used in headless eval.
- Designed for machines with an X11 ``DISPLAY``. Wayland sessions require
  XWayland. The viewer window will open during recording — this is expected.
- Defaults to a small number of episodes (5), suitable for demo clips.
- With ``--ckpt-tag best``, pins one record and its concrete checkpoint path,
  using its EMA choice and inference step count when present and saved config
  defaults otherwise. Explicit
  ``--ema``/``--no-ema`` and ``--inference-steps`` override these settings.
- ``--selection-record`` pins a selector's own handoff file and rejects
  conflicting checkpoint/EMA/NFE choices. Demo seeds remain non-held-out and
  do not require the selection/test manifest to be available in the demo pool.

Usage
-----
.. code-block:: bash

    # Basic: record 5 episodes from the best checkpoint
    python dexmani_policy/record_demo.py --policy-name dp3 --task-name pour \\
        --exp-name 2026-08-01_12-34-56

    # Specific checkpoint, more episodes, custom output dir
    python dexmani_policy/record_demo.py --policy-name sat --task-name pour \\
        --exp-name 2026-08-01_12-34-56 --ckpt-tag 100pct --episodes 10 \\
        --output-dir ~/Videos/demos

    # Custom viewer resolution (4K)
    python dexmani_policy/record_demo.py --policy-name maniflow --task-name pour \\
        --exp-name 2026-08-01_12-34-56 --resolution 3840 2160

Output
------
Each invocation creates ``<output-dir>/<timestamp>_<random-suffix>/`` with
``eval_config.yaml``, ``result_details.json`` and optional episode videos.
NFE sweeps put results/videos under ``inference_steps<N>/``; multi-task
runners put videos in task subdirectories. Demo seeds are not held-out.
Default output directory: ``experiments/<policy>/<task>/<exp>/demo_videos/``.
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.agents.loader import resolve_best_checkpoint
from dexmani_policy.utils.path import set_project_root
from dexmani_policy.utils.random import set_seed
from dexmani_policy.evaluation.protocol import (
    _get_eval_param,
    save_eval_snapshot, run_eval_plan, plan_size, task_seed_pools, artifact_reference, atomic_json, compute_eval_stats, selection_provenance,
    add_inference_steps_argument,
    build_eval_runner,
    collect_episode_details,
    iter_leaf_env_runners,
    load_ckpt_for_inference,
    resolve_checkpoint_path,
    resolve_eval_seed,
    select_eval_seeds,
    validate_inference_steps,
    validate_task_seeds,
)

ROOT_DIR = set_project_root()
register_resolvers()


def _resolve_demo_inference(
    cfg,
    exp_dir: Path,
    ckpt_tag: str,
    *,
    cli_use_ema: bool | None,
    cli_inference_steps: int | None,
    selection_record=None,
) -> tuple[bool, list[int], tuple[dict, Path] | None]:
    """Return EMA/NFE and a pinned best pair; CLI > record > saved defaults."""
    use_ema = _get_eval_param(cfg, "use_ema", "demo", default=True)
    configured_steps = _get_eval_param(
        cfg, "inference_steps_list", "demo", default=None
    )
    if configured_steps is not None:
        inference_steps_list = list(configured_steps)
    else:
        inference_steps_list = [
            _get_eval_param(cfg, "inference_steps", "demo", default=10)
        ]

    resolved_best = (resolve_best_checkpoint(exp_dir, selection_record) if selection_record
                     else resolve_best_checkpoint(exp_dir) if ckpt_tag == "best" else None)
    if resolved_best is not None:
        inference = resolved_best[0].get("inference", {})
        if not isinstance(inference, dict):
            raise ValueError("Best inference settings must be an object")
        use_ema = inference.get("use_ema", use_ema)
        if "inference_steps" in inference:
            inference_steps_list = [inference["inference_steps"]]

    if cli_use_ema is not None:
        use_ema = cli_use_ema
    if cli_inference_steps is not None:
        inference_steps_list = [cli_inference_steps]

    if selection_record and ckpt_tag != "best":
        requested_path, _ = resolve_checkpoint_path(exp_dir, ckpt_tag)
        if requested_path != resolved_best[1]:
            raise ValueError("Checkpoint override differs from selection handoff")
    if selection_record and (use_ema != resolved_best[0]["inference"]["use_ema"] or
                             inference_steps_list != [resolved_best[0]["inference"]["inference_steps"]]):
        raise ValueError("Selection handoff cannot override raw/EMA or NFE")
    if type(use_ema) is not bool:
        raise ValueError(f"use_ema must resolve to boolean, got {use_ema!r}")
    validate_inference_steps(inference_steps_list)
    return use_ema, inference_steps_list, resolved_best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record high-resolution demo videos from a trained checkpoint.",
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        required=True,
        help="Policy name (e.g., dp3, sat, maniflow).",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name (e.g., pour, stack_cups).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment directory name (timestamp or custom).",
    )
    parser.add_argument(
        "--ckpt-tag",
        type=str,
        default="best",
        help="Checkpoint: best, latest, 20pct..100pct (default: best).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to record (default: from config eval.demo.episodes). "
        "Ignored when --seeds is provided.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Physical seed numbers for every task (e.g. --seeds 5 12 33); no ordinal remapping. "
        "Overrides --episodes. Useful for re-recording specific episodes "
        "from a prior eval run (see result_details.json).",
    )
    add_inference_steps_argument(parser)
    parser.add_argument(
        "--ema",
        dest="use_ema",
        action="store_true",
        default=None,
        help="Use EMA weights (best: selection record; otherwise config).",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Use raw weights instead of EMA.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos (default: exp_dir/demo_videos/).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Viewer window resolution WIDTH HEIGHT (default: 1280 960).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video FPS override (default: auto-detect from env).",
    )

    parser.add_argument("--selection-record", default=None)
    args = parser.parse_args()

    # 1. Locate experiment directory
    exp_dir = (
        (
            Path(ROOT_DIR)
            / "experiments"
            / args.policy_name
            / args.task_name
            / args.exp_name
        )
        .expanduser()
        .resolve()
    )

    if not exp_dir.is_dir():
        cprint(f"Error: experiment directory not found: {exp_dir}", "red")
        sys.exit(1)

    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        cprint(f"Error: config.yaml not found: {cfg_path}", "red")
        sys.exit(1)

    # 2. Load config
    cfg = OmegaConf.load(cfg_path)
    cfg._exp_dir = str(exp_dir)

    eval_seed = resolve_eval_seed(cfg)
    set_seed(eval_seed)

    cprint(f"Device: {cfg.training.device}", "cyan")

    use_ema, inference_steps_list, resolved_best = _resolve_demo_inference(
        cfg,
        exp_dir,
        args.ckpt_tag,
        selection_record=args.selection_record,
        cli_use_ema=args.use_ema,
        cli_inference_steps=args.inference_steps,
    )

    # 3. Build env_runner
    best_info = resolved_best[0] if resolved_best else None
    env_runner = build_eval_runner(cfg)

    # Apply viewer resolution from CLI or config
    _demo_resolution = args.resolution
    if _demo_resolution is None:
        _demo_resolution = _get_eval_param(
            cfg, "viewer_resolution", "demo", default=[1280, 960]
        )
    resolved_resolution = tuple(_demo_resolution)
    resolved_fps = (
        args.fps
        if args.fps is not None
        else _get_eval_param(cfg, "fps", "video", default=None)
    )

    # Switch every leaf to viewer-based rendering for high-res video capture.
    for leaf_runner in iter_leaf_env_runners(env_runner):
        leaf_runner.render_mode = "human"
        leaf_runner.record_video = True
        leaf_runner.viewer_resolution = resolved_resolution
        if resolved_fps is not None:
            leaf_runner.env_video_fps = resolved_fps

    # 4. Resolve output directory
    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = exp_dir / "demo_videos"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base.mkdir(parents=True, exist_ok=True)
    video_save_dir = Path(tempfile.mkdtemp(prefix=timestamp+"_", dir=output_base))

    # 5. Use the pinned best path or resolve a non-best selector
    ckpt_path, ckpt_label = resolve_checkpoint_path(
        exp_dir, str(resolved_best[1]) if resolved_best is not None else args.ckpt_tag
    )
    # 6. Resolve parameters
    demo_episodes = (
        args.episodes
        if args.episodes is not None
        else _get_eval_param(cfg, "episodes", "demo", default=5)
    )

    do_sweep = len(inference_steps_list) > 1

    cprint(f"\nLoading checkpoint: {ckpt_label} (EMA={use_ema})", "cyan")
    agent = load_ckpt_for_inference(ckpt_path, use_ema, cfg=cfg)
    if best_info is not None and "selection_summary" in best_info and agent._checkpoint_global_step != best_info["global_step"]:
        raise ValueError("Best record global_step disagrees with actual checkpoint state")
    cprint("✅ Checkpoint loaded\n", "green")

    # 7. Print recording config
    resolution_str = f"{resolved_resolution[0]}×{resolved_resolution[1]}"
    steps_str = (
        ", ".join(str(d) for d in inference_steps_list)
        if do_sweep
        else str(inference_steps_list[0])
    )
    cprint(f"{'=' * 60}", "cyan")
    cprint("  Demo Video Recording", "cyan")
    cprint(f"  Policy       : {args.policy_name}", "cyan")
    cprint(f"  Task         : {args.task_name}", "cyan")
    cprint(f"  Checkpoint   : {ckpt_label}", "cyan")
    cprint(f"  Episodes     : {demo_episodes}", "cyan")
    cprint(f"  Inference steps: {steps_str}" + (" (sweep)" if do_sweep else ""), "cyan")
    cprint(f"  Resolution   : {resolution_str}", "cyan")
    cprint(f"  Output dir   : {video_save_dir}", "cyan")
    cprint(f"{'=' * 60}\n", "cyan")

    # 8. Select seeds
    pools = task_seed_pools(env_runner)
    if args.seeds is not None:
        eval_seeds = {task: list(args.seeds) for task in pools}
        demo_episodes = len(args.seeds)
        cprint(
            f"  Using {demo_episodes} specified seeds: {eval_seeds}",
            "cyan",
        )
    else:
        eval_seeds = select_eval_seeds(env_runner, eval_seed, demo_episodes)
        demo_episodes = plan_size(eval_seeds)

    validate_task_seeds(eval_seeds, pools)
    for task, seeds in eval_seeds.items():
        if set(seeds) - set(pools[task]):
            raise ValueError(f"Unavailable demo seeds for {task}")
    # Only the actual demo plan is used; selection evidence remains provenance.
    snapshot_ref = save_eval_snapshot(
        video_save_dir, cfg, env_runner, checkpoint=artifact_reference(ckpt_path, exp_dir),
        global_step=agent._checkpoint_global_step, use_ema=use_ema,
        inference_steps_list=inference_steps_list, episodes=demo_episodes,
        shuffle_seed=eval_seed, policy_seed_mode="episode_seed",
        task_seeds=eval_seeds,
        viewer_resolution=list(resolved_resolution), env_video_fps=resolved_fps,
        heldout_from_selection=False,
        **selection_provenance(best_info),
    )

    # 9. Run episodes (sweep or single)
    demo_results: list[dict] = []

    for inference_steps in inference_steps_list:
        if do_sweep:
            sub_dir = video_save_dir / f"inference_steps{inference_steps}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            cprint(
                f"\n--- inference_steps={inference_steps} ---", "cyan", attrs=["bold"]
            )
        else:
            sub_dir = video_save_dir

        result = run_eval_plan(
            env_runner, agent, eval_seeds,
            inference_steps=inference_steps,
            video_save_dir=sub_dir,
        )

        per_seed_details = collect_episode_details(result)
        statistics = compute_eval_stats(result)
        atomic_json(sub_dir / "result_details.json", {
            "eval_config": snapshot_ref, "checkpoint": artifact_reference(ckpt_path, exp_dir),
            "global_step": agent._checkpoint_global_step, "inference_steps": inference_steps,
            "use_ema": use_ema, "episode_details": per_seed_details,
            "statistics": statistics, "heldout_from_selection": False,
        })
        n_success = statistics["n_success"]
        n_total = statistics["n_valid_episodes"]
        sr = statistics["micro_success_rate"]

        if do_sweep:
            cprint(
                f"  Success: {n_success}/{n_total} = {sr:.1%}",
                "green" if sr >= 0.5 else "red",
            )
            demo_results.append(
                {
                    "inference_steps": inference_steps,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": sr,
                }
            )

    # 10. Report
    if do_sweep and demo_results:
        cprint(f"\n{'=' * 60}", "green")
        cprint("  Inference Steps Sweep Summary", "green")
        cprint(f"  {'Inference Steps':<16} {'Success Rate':<18}", "green")
        cprint("  " + "-" * 34, "green")
        for r in demo_results:
            inference_steps = r["inference_steps"]
            sr = f"{r['n_success']}/{r['n_total']} ({r['success_rate']:.1%})"
            cprint(
                f"  {inference_steps:<16} {sr:<18}",
                "green" if r["success_rate"] >= 0.5 else "red",
            )
        cprint(f"{'=' * 60}\n", "green")

    cprint(f"\n{'=' * 60}", "green")
    cprint("  Recording complete!", "green")
    if not do_sweep:
        cprint(
            f"  Success rate : {n_success}/{n_total} = {sr:.1%}",
            "green",
        )
    cprint(f"  Videos saved : {video_save_dir}", "green")

    video_count = len(list(video_save_dir.rglob("*.mp4")))
    cprint(f"  MP4 files    : {video_count}", "green")
    cprint(f"{'=' * 60}\n", "green")


if __name__ == "__main__":
    main()
