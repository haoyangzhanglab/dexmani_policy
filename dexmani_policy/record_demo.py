"""Demo video recording — high-resolution viewer capture for presentations.

Loads a trained checkpoint and records evaluation episodes using the SAPIEN
viewer at high resolution (default 1920×1080), suitable for making demo videos.

Key differences from ``eval_best_ckpt.py``:

- Uses ``render_mode="human"`` → viewer is created → video frames come from
  ``get_viewer_rgb()`` at the viewer window's native resolution (1920×1080),
  instead of the 320×240 sensor camera used in headless eval.
- Designed for machines **with a display** (X11/Wayland). The viewer window
  will open during recording — this is expected.
- Defaults to a small number of episodes (5), suitable for demo clips.

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
Videos are saved to ``<output-dir>/<YYYYmmdd_HHMMSS>/episode_<seed>.mp4``.
Default output directory: ``experiments/<policy>/<task>/<exp>/demo_videos/``.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.training.eval_utils import (
    _get_eval_param,
    build_eval_components,
    collect_episode_details,
    load_ckpt_for_inference,
    resolve_checkpoint_path,
    resolve_eval_seed,
    validate_eval_config,
)

ROOT_DIR = set_project_root()
register_resolvers()


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
        help="Specific seed numbers to record (e.g. --seeds 5 12 33). "
        "Overrides --episodes. Useful for re-recording specific episodes "
        "from a prior eval run (see result_details.json).",
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=None,
        help="DDIM/Euler denoising steps (default: from config).",
    )
    parser.add_argument(
        "--ema",
        dest="use_ema",
        action="store_true",
        default=None,
        help="Use EMA weights (default: from config).",
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
        help="Viewer window resolution WIDTH HEIGHT (default: 1920 1080).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video FPS override (default: auto-detect from env).",
    )

    args = parser.parse_args()

    # ── 1. Locate experiment directory ────────────────────────────────────
    exp_dir = (
        (Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name)
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

    # ── 2. Load config ────────────────────────────────────────────────────
    cfg = OmegaConf.load(cfg_path)
    cfg._exp_dir = str(exp_dir)
    validate_eval_config(cfg)

    eval_seed = resolve_eval_seed(cfg)
    set_seed(eval_seed)

    device = torch.device(cfg.training.device)
    cprint(f"Device: {device}", "cyan")

    # ── 3. Build agent and env_runner ─────────────────────────────────────
    agent, env_runner, checkpoint_store = build_eval_components(cfg, device)

    # Switch to viewer-based rendering for high-res video capture
    env_runner.render_mode = "human"
    env_runner.record_video = True

    # Apply viewer resolution from CLI or config
    _demo_resolution = args.resolution
    if _demo_resolution is None:
        _demo_resolution = _get_eval_param(cfg, "viewer_resolution", "demo", default=[1920, 1080])
    env_runner.viewer_resolution = tuple(_demo_resolution)

    if args.fps is not None:
        env_runner.env_video_fps = args.fps
    else:
        _demo_fps = _get_eval_param(cfg, "fps", "video", default=None)
        if _demo_fps is not None:
            env_runner.env_video_fps = _demo_fps

    # ── 4. Resolve output directory ───────────────────────────────────────
    if args.output_dir:
        output_base = Path(args.output_dir).expanduser().resolve()
    else:
        output_base = exp_dir / "demo_videos"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_save_dir = output_base / timestamp
    video_save_dir.mkdir(parents=True, exist_ok=True)

    # ── 5. Resolve checkpoint ─────────────────────────────────────────────
    ckpt_path, ckpt_label = resolve_checkpoint_path(exp_dir, args.ckpt_tag, checkpoint_store)

    # ── 6. Resolve parameters ─────────────────────────────────────────────
    denoise_steps = (
        args.denoise_steps
        if args.denoise_steps is not None
        else _get_eval_param(cfg, "denoise_steps", "demo", default=10)
    )
    use_ema = (
        args.use_ema if args.use_ema is not None else _get_eval_param(cfg, "use_ema", "demo", default=True)
    )
    demo_episodes = (
        args.episodes if args.episodes is not None else _get_eval_param(cfg, "episodes", "demo", default=5)
    )

    cprint(f"\nLoading checkpoint: {ckpt_label} (EMA={use_ema})", "cyan")
    load_ckpt_for_inference(agent, checkpoint_store, ckpt_path, use_ema)
    agent.to(device)
    agent.eval()
    cprint("✅ Checkpoint loaded\n", "green")

    # ── 7. Print recording config ─────────────────────────────────────────
    _res = env_runner.viewer_resolution
    resolution_str = f"{_res[0]}×{_res[1]}" if _res else "1920×1080 (default)"
    cprint(f"{'=' * 60}", "cyan")
    cprint("  Demo Video Recording", "cyan")
    cprint(f"  Policy       : {args.policy_name}", "cyan")
    cprint(f"  Task         : {args.task_name}", "cyan")
    cprint(f"  Checkpoint   : {ckpt_label}", "cyan")
    cprint(f"  Episodes     : {demo_episodes}", "cyan")
    cprint(f"  Resolution   : {resolution_str}", "cyan")
    cprint(f"  Output dir   : {video_save_dir}", "cyan")
    cprint(f"{'=' * 60}\n", "cyan")

    # ── 8. Select seeds ────────────────────────────────────────────────────
    # Deterministically shuffled with eval_seed, matching eval_best_ckpt and
    # select_best_ckpt conventions — same experiment always picks the same subset.
    if args.seeds:
        eval_seeds = args.seeds
        demo_episodes = len(eval_seeds)
        cprint(
            f"  Using {demo_episodes} specified seeds: {eval_seeds}",
            "cyan",
        )
    else:
        all_seeds = list(env_runner.get_seed_list())
        # Deterministically shuffle (same convention as eval_best_ckpt / select_best_ckpt)
        rng = random.Random(eval_seed)
        rng.shuffle(all_seeds)
        eval_seeds = all_seeds[:demo_episodes]

    env_runner.eval_seeds = eval_seeds

    # ── 9. Run episodes ───────────────────────────────────────────────────

    result = env_runner.run(
        agent,
        denoise_timesteps=denoise_steps,
        eval_episodes=demo_episodes,
        video_save_dir=video_save_dir,
    )

    # ── 10. Report ────────────────────────────────────────────────────────
    per_seed_details = collect_episode_details(result)
    n_success = sum(1 for d in per_seed_details if d.get("success"))
    n_total = len(per_seed_details)

    cprint(f"\n{'=' * 60}", "green")
    cprint("  Recording complete!", "green")
    cprint(f"  Success rate : {n_success}/{n_total} = {n_success / n_total:.1%}" if n_total else "", "green")
    cprint(f"  Videos saved : {video_save_dir}", "green")

    video_count = len(list(video_save_dir.glob("*.mp4")))
    cprint(f"  MP4 files    : {video_count}", "green")
    cprint(f"{'=' * 60}\n", "green")


if __name__ == "__main__":
    main()
