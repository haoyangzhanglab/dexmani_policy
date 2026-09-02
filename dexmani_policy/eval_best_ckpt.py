"""RoboTwin-style checkpoint evaluation — simple, reproducible, no fluff.

Loads a checkpoint and runs it on all evaluation seeds from the seed pool.
Output is a single success rate, matching RoboTwin's ``_result.txt`` format.

Methodology (1:1 RoboTwin ``eval_policy.py``)
----------------------------------------------

1. Load the specified checkpoint (EMA weights by default).
2. Use ``training.seed`` from the experiment config — same seed the
   model was trained with.
3. Read the full evaluation seed pool (~100 seeds from
   ``eval_seeds/<task>.txt`` or ``range(100)``).
4. Run one episode per seed (deterministic env + policy: re-running
   the same seed produces identical results).
5. Output: ``success_rate = n_success / n_total`` and avg steps.

No bootstrap.  No confidence intervals.  No statistics beyond counting.
This is exactly what RoboTwin's ``eval_policy.py`` does — the paper's
bootstrap CIs are computed separately with the Rliable library.

Usage
-----
.. code-block:: bash

    # After select_best_ckpt.sh:
    bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42

    # Specific checkpoint:
    bash scripts/eval/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 \\
        --ckpt-tag 20pct --episodes 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.env_runner.base_runner import EvalEpisodeError
from dexmani_policy.training.eval_utils import (
    _get_eval_param,
    build_eval_components,
    collect_episode_details,
    load_ckpt_for_inference,
    resolve_checkpoint_path,
    resolve_eval_seed,
    validate_denoise_steps,
    validate_eval_config,
)

ROOT_DIR = set_project_root()
register_resolvers()


# ---------------------------------------------------------------------------
# Shared helpers (used by both single-value and sweep paths)
# ---------------------------------------------------------------------------


def _setup_eval(
    cfg,
    exp_dir: Path,
    ckpt_tag_or_path: str,
    use_ema: bool,
    *,
    video_save_dir: Path | None = None,
):
    """Validate config, build components, load checkpoint (shared setup).

    Returns
    -------
    (agent, env_runner, checkpoint_store, ckpt_path, ckpt_label, eval_seed, device)
    """
    validate_eval_config(cfg)
    eval_seed = resolve_eval_seed(cfg)
    set_seed(eval_seed)

    device = torch.device(cfg.training.device)
    agent, env_runner, checkpoint_store = build_eval_components(cfg, device)

    if hasattr(env_runner, "record_video"):
        env_runner.record_video = video_save_dir is not None

    ckpt_path, ckpt_label = resolve_checkpoint_path(exp_dir, ckpt_tag_or_path, checkpoint_store)

    cprint(f"\nLoading checkpoint: {ckpt_label} (EMA={use_ema})", "cyan")
    load_ckpt_for_inference(agent, checkpoint_store, ckpt_path, use_ema)
    agent.to(device)
    agent.eval()
    cprint("✅ Checkpoint loaded\n", "green")

    return agent, env_runner, checkpoint_store, ckpt_path, ckpt_label, eval_seed, device


def _select_eval_seeds(env_runner, eval_seed: int, episodes: int) -> list[int]:
    """Deterministically select *episodes* seeds from the seed pool."""
    all_seeds = list(env_runner.get_seed_list())
    n_total = min(episodes, len(all_seeds))
    if episodes > len(all_seeds):
        cprint(
            f"⚠ Requested {episodes} episodes > {len(all_seeds)} available seeds, using {len(all_seeds)}",
            "yellow",
        )

    rng = random.Random(eval_seed)
    rng.shuffle(all_seeds)
    eval_seeds = all_seeds[:n_total]

    cprint(
        f"Evaluating on {n_total} seeds (eval_seed={eval_seed}, first seed={eval_seeds[0]}) ...",
        "cyan",
    )
    return eval_seeds


def _run_one_timestep(
    agent,
    env_runner,
    eval_seeds: list[int],
    denoise_steps: int,
    video_save_dir: Path | None,
    *,
    exp_dir: Path,
    ckpt_tag_or_path: str,
    ckpt_path: Path,
    ckpt_label: str,
    eval_seed: int,
) -> dict:
    """Run eval at a single denoise step count; save per-value results.

    Saves ``_result.txt`` + ``result_details.json`` into
    *exp_dir*/eval_dexsim (single-value) or a ``denoise_timesteps<N>/``
    subdirectory of *video_save_dir* (sweep).

    Returns a dict with keys: ``success_rate``, ``avg_steps``, ``n_success``,
    ``n_total``, ``per_seed_details``.
    """
    n_total = len(eval_seeds)
    env_runner.eval_seeds = eval_seeds
    result = env_runner.run(
        agent,
        denoise_timesteps=denoise_steps,
        eval_episodes=n_total,
        video_save_dir=video_save_dir,
    )

    per_seed_details: list[dict] = collect_episode_details(result)
    n_success = sum(1 for d in per_seed_details if d.get("success"))
    success_rate = n_success / n_total if n_total > 0 else 0.0

    task_done_steps = [
        d["steps"] for d in per_seed_details if d.get("success") and d.get("steps") is not None
    ]
    avg_steps = float(sum(task_done_steps) / len(task_done_steps)) if task_done_steps else None

    # ── Save results ───────────────────────────────────────────────────
    if video_save_dir is not None:
        # Sweep: save into the per-timestep subdirectory
        save_dir = video_save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = exp_dir / "eval_dexsim"
        save_dir.mkdir(parents=True, exist_ok=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "_result.txt").write_text(f"{success_rate}\n")
    (save_dir / "result_details.json").write_text(
        json.dumps(
            {
                "ckpt_tag": ckpt_tag_or_path,
                "ckpt_path": str(ckpt_path),
                "success_rate": success_rate,
                "n_success": n_success,
                "n_total": n_total,
                "avg_steps": avg_steps,
                "eval_seed": eval_seed,
                "denoise_steps": denoise_steps,
                "per_seed_details": per_seed_details,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "n_success": n_success,
        "n_total": n_total,
        "per_seed_details": per_seed_details,
    }


# ---------------------------------------------------------------------------
# Single-value evaluation (RoboTwin-style)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint_robotwin(
    exp_dir: Path,
    cfg,
    *,
    ckpt_tag_or_path: str = "best",
    episodes: int = 100,
    denoise_steps: int = 10,
    use_ema: bool = True,
    video_save_dir: Path | None = None,
) -> tuple[float, float | None, int, int]:
    """Evaluate a checkpoint and return success rate.

    Parameters
    ----------
    cfg : pre-loaded OmegaConf config with ``_exp_dir`` injected.
    ckpt_tag_or_path : ``"best"``, ``"latest"``, ``"20pct"``, or a path.
        ``"best"`` reads ``best_ckpt.json`` (written by ``select_best_ckpt.py``).
    episodes : number of seeds to evaluate (default: 100).
    denoise_steps : DDIM/Euler inference steps.
    use_ema : use EMA weights if available.

    Returns
    -------
    (success_rate, avg_steps, n_success, n_total)
    """
    agent, env_runner, ckpt_store, ckpt_path, ckpt_label, eval_seed, device = _setup_eval(
        cfg, exp_dir, ckpt_tag_or_path, use_ema, video_save_dir=video_save_dir,
    )
    validate_denoise_steps([denoise_steps], getattr(agent.action_decoder, "solver", None))
    eval_seeds = _select_eval_seeds(env_runner, eval_seed, episodes)

    info = _run_one_timestep(
        agent, env_runner, eval_seeds, denoise_steps, video_save_dir=None,
        exp_dir=exp_dir, ckpt_tag_or_path=ckpt_tag_or_path,
        ckpt_path=ckpt_path, ckpt_label=ckpt_label, eval_seed=eval_seed,
    )

    # ── Report ─────────────────────────────────────────────────────────
    avg_str = f"{info['avg_steps']:.1f}" if info["avg_steps"] is not None else "N/A"
    cprint(f"\n{'=' * 50}", "cyan")
    cprint(f"  Checkpoint   : {ckpt_label}", "cyan")
    cprint(f"  Seeds        : {info['n_total']}", "cyan")
    cprint(f"  Denoise steps: {denoise_steps}", "cyan")
    cprint(f"  Success rate : {info['n_success']}/{info['n_total']} = {info['success_rate']:.1%}", "green")
    cprint(f"  Avg steps    : {avg_str}", "cyan")
    cprint(f"{'=' * 50}\n", "cyan")

    save_dir = exp_dir / "eval_dexsim"
    cprint(f"  Results saved to: {save_dir}/_result.txt", "cyan")

    return info["success_rate"], info["avg_steps"], info["n_success"], info["n_total"]


# ---------------------------------------------------------------------------
# Multi-value sweep evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint_sweep(
    exp_dir: Path,
    cfg,
    *,
    ckpt_tag_or_path: str = "best",
    episodes: int = 100,
    denoise_timesteps_list: list[int],
    use_ema: bool = True,
    video_save_dir: Path | None = None,
) -> list[dict]:
    """Evaluate a checkpoint at multiple denoising step counts.

    The checkpoint is **loaded once** and reused across all denoise values.
    The **same evaluation seeds** are used for every value so the comparison
    is apples-to-apples.

    Results are saved into ``denoise_timesteps<N>/`` subdirectories under
    *video_save_dir* (or ``exp_dir/eval_dexsim/<timestamp>/``), plus an
    aggregate ``eval_summary.json``.
    """
    if not denoise_timesteps_list:
        raise ValueError("denoise_timesteps_list must be non-empty")

    # ── 1. Setup ONCE ──────────────────────────────────────────────────
    agent, env_runner, ckpt_store, ckpt_path, ckpt_label, eval_seed, device = _setup_eval(
        cfg, exp_dir, ckpt_tag_or_path, use_ema, video_save_dir=video_save_dir,
    )
    validate_denoise_steps(denoise_timesteps_list, getattr(agent.action_decoder, "solver", None))

    # ── 2. Same seeds for all denoise values (fair comparison) ──────────
    eval_seeds = _select_eval_seeds(env_runner, eval_seed, episodes)
    n_total = len(eval_seeds)

    # ── 3. Sweep over denoise timesteps ─────────────────────────────────
    sweep_results: list[dict] = []

    for dt in denoise_timesteps_list:
        cprint(f"\n--- denoise_timesteps={dt} ---", "cyan", attrs=["bold"])

        sub_dir = video_save_dir / f"denoise_timesteps{dt}" if video_save_dir else None
        info = _run_one_timestep(
            agent, env_runner, eval_seeds, dt, video_save_dir=sub_dir,
            exp_dir=exp_dir, ckpt_tag_or_path=ckpt_tag_or_path,
            ckpt_path=ckpt_path, ckpt_label=ckpt_label, eval_seed=eval_seed,
        )

        avg_str = f"{info['avg_steps']:.1f}" if info["avg_steps"] is not None else "N/A"
        cprint(
            f"  Success: {info['n_success']}/{n_total} = {info['success_rate']:.1%}  "
            f"Avg steps: {avg_str}",
            "green" if info["success_rate"] >= 0.5 else "red",
        )

        sweep_results.append({"denoise_timesteps": dt, **info})

    # ── 4. Aggregate summary ───────────────────────────────────────────
    _save_sweep_summary(video_save_dir or (exp_dir / "eval_dexsim"), sweep_results, ckpt_label)

    return sweep_results


def _save_sweep_summary(save_dir: Path, sweep_results: list[dict], ckpt_label: str) -> None:
    """Save ``eval_summary.json`` and print a comparison table."""
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": ckpt_label,
        "results": {f"denoise_timesteps{r['denoise_timesteps']}": {
            "success_rate": r["success_rate"],
            "avg_steps": r["avg_steps"],
            "n_success": r["n_success"],
            "n_total": r["n_total"],
        } for r in sweep_results},
    }
    (save_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── Terminal comparison table ──────────────────────────────────────
    cprint(f"\n{'=' * 60}", "cyan", attrs=["bold"])
    cprint("  Denoise Timesteps Sweep Summary", "cyan", attrs=["bold"])
    cprint(f"  Checkpoint: {ckpt_label}", "cyan")
    cprint(f"  {'Denoise Steps':<16} {'Success Rate':<18} {'Avg Steps':<12}", "cyan")
    cprint("  " + "-" * 46, "cyan")
    for r in sweep_results:
        dt = r["denoise_timesteps"]
        sr = f"{r['n_success']}/{r['n_total']} ({r['success_rate']:.1%})"
        avg = f"{r['avg_steps']:.1f}" if r["avg_steps"] is not None else "N/A"
        cprint(f"  {dt:<16} {sr:<18} {avg:<12}", "green" if r["success_rate"] >= 0.5 else "red")
    cprint(f"{'=' * 60}\n", "cyan", attrs=["bold"])
    cprint(f"  Summary saved to: {save_dir}/eval_summary.json", "cyan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboTwin-style checkpoint evaluation (success rate only).",
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt-tag",
        type=str,
        default="best",
        help="Checkpoint: best (reads best_ckpt.json), latest, 20pct..100pct (default: best).",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Direct .pt path (overrides --ckpt-tag).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of seeds to evaluate (default: from config eval.offline).",
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=None,
        help="DDIM / Euler denoising steps (default: from config).",
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
        "--no-videos",
        action="store_true",
        default=False,
        help="Disable video recording (videos are saved by default).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dot-list overrides.",
    )
    args = parser.parse_args()

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

    cfg = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg._exp_dir = str(exp_dir)

    # ── Resolve parameters: CLI > config > hardcoded fallback ────────────
    episodes = (
        args.episodes
        if args.episodes is not None
        else _get_eval_param(cfg, "episodes", "offline", default=100)
    )
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    use_ema = (
        args.use_ema if args.use_ema is not None else _get_eval_param(cfg, "use_ema", "offline", default=True)
    )

    # Denoise steps: CLI --denoise-steps (single value) > config list > config single > default
    if args.denoise_steps is not None:
        denoise_timesteps_list = [args.denoise_steps]
    else:
        dt_list = _get_eval_param(cfg, "denoise_timesteps_list", "offline", default=None)
        if dt_list:
            denoise_timesteps_list = list(dt_list)
        else:
            denoise_steps = _get_eval_param(cfg, "denoise_steps", "offline", default=10)
            denoise_timesteps_list = [denoise_steps]

    do_sweep = len(denoise_timesteps_list) > 1

    # Video saving — configurable via eval.video.enabled, overridable via --no-videos.
    # Single-value: exp_dir/eval_dexsim/_result.txt (backward compat)
    # Sweep:        exp_dir/eval_dexsim/<timestamp>/denoise_timesteps<N>/* + eval_summary.json
    video_enabled = _get_eval_param(cfg, "enabled", "video", default=True)
    video_save_dir = None
    if video_enabled and not args.no_videos:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_save_dir = exp_dir / "eval_dexsim" / timestamp
        video_save_dir.mkdir(parents=True, exist_ok=True)
        cprint(f"\n📹 Video output: {video_save_dir}", "cyan")
    elif do_sweep:
        # Sweep always uses a timestamped dir for per-value subdirectories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_save_dir = exp_dir / "eval_dexsim" / timestamp

    ckpt_tag_or_path = args.ckpt_path if args.ckpt_path else args.ckpt_tag

    try:
        if do_sweep:
            cprint(
                f"\n🔁 Denoise timesteps sweep: {denoise_timesteps_list}", "cyan", attrs=["bold"],
            )
            evaluate_checkpoint_sweep(
                exp_dir,
                cfg,
                ckpt_tag_or_path=ckpt_tag_or_path,
                episodes=episodes,
                denoise_timesteps_list=denoise_timesteps_list,
                use_ema=use_ema,
                video_save_dir=video_save_dir,
            )
        else:
            evaluate_checkpoint_robotwin(
                exp_dir,
                cfg,
                ckpt_tag_or_path=ckpt_tag_or_path,
                episodes=episodes,
                denoise_steps=denoise_timesteps_list[0],
                use_ema=use_ema,
                video_save_dir=video_save_dir,
            )
    except EvalEpisodeError as e:
        cprint(f"Fatal eval error (category={e.category}, seed={e.seed}): {e}", "red")
        sys.exit(1)
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as e:
        cprint(f"Eval failed: {type(e).__name__}: {e}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
