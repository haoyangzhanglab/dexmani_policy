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
    bash scripts/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42

    # Specific checkpoint:
    bash scripts/eval_best_ckpt.sh dp3 pour 2026-07-29_01-53_42 \\
        --ckpt-tag 20pct --episodes 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.select_best_ckpt import discover_milestone_checkpoints
from dexmani_policy.training.eval_utils import (
    build_eval_components,
    collect_episode_details,
    load_ckpt_for_inference,
    read_best_ckpt_json,
    validate_eval_config,
)

ROOT_DIR = set_project_root()
register_resolvers()


# ---------------------------------------------------------------------------
# Main evaluation function (RoboTwin-style)
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

    # ── 1. Validate config ────────────────────────────────────────────
    validate_eval_config(cfg)

    # Use eval.seed (standardized across all three eval scripts — C6 fix).
    # Falls back to training.seed for backward compat with older configs.
    train_seed = (
        cfg.eval.get("seed")
        if hasattr(cfg, "eval") and cfg.eval.get("seed") is not None
        else cfg.training.get("seed", 0)
    )
    set_seed(train_seed)

    device = torch.device(cfg.training.device)

    # ── 2. Build components ───────────────────────────────────────────
    agent, env_runner, checkpoint_store = build_eval_components(cfg, device)

    # ── 3. Resolve & load checkpoint ──────────────────────────────────
    if ckpt_tag_or_path.endswith("pct"):
        milestones = discover_milestone_checkpoints(exp_dir)
        target_pct = int(ckpt_tag_or_path.replace("pct", ""))
        match = [m for m in milestones if m.pct == target_pct]
        if not match:
            available = sorted(m.pct for m in milestones)
            raise FileNotFoundError(f"No {target_pct}% milestone checkpoint. Available: {available}")
        ckpt_path = match[0].path
        ckpt_label = match[0].label
    elif ckpt_tag_or_path == "best":
        # Read best_ckpt.json (written by select_best_ckpt.py)
        best_info = read_best_ckpt_json(exp_dir)
        if best_info:
            ckpt_path = Path(best_info["ckpt_path"])
            ckpt_label = f"best -> {best_info['pct']}% (step={best_info['global_step']})"
            cprint(
                f"  Auto-loaded best checkpoint: {best_info['pct']}% "
                f"(success_rate={best_info['success_rate']:.1%}, "
                f"n_episodes={best_info['n_episodes']})",
                "cyan",
            )
        else:
            # Fallback: try symlink
            ckpt_path = checkpoint_store.resolve_path("best")
            ckpt_label = f"best ({ckpt_path.name})"
            cprint(
                "  ⚠ best_ckpt.json not found — using best.pt symlink. "
                "Run select_best_ckpt.sh first for automatic selection.",
                "yellow",
            )
    elif ckpt_tag_or_path == "latest":
        ckpt_path = checkpoint_store.resolve_path("latest")
        ckpt_label = f"latest ({ckpt_path.name})"
    else:
        ckpt_path = Path(ckpt_tag_or_path)
        if not ckpt_path.is_absolute():
            ckpt_path = exp_dir / "checkpoints" / ckpt_path
        ckpt_label = str(ckpt_path)

    cprint(f"\nLoading checkpoint: {ckpt_label} (EMA={use_ema})", "cyan")
    load_ckpt_for_inference(agent, checkpoint_store, ckpt_path, use_ema)
    agent.to(device)
    agent.eval()
    cprint("✅ Checkpoint loaded\n", "green")

    # ── 4. Run evaluation on seed pool ────────────────────────────────
    all_seeds = list(env_runner.get_seed_list())
    n_total = min(episodes, len(all_seeds))
    if episodes > len(all_seeds):
        cprint(
            f"⚠ Requested {episodes} episodes > {len(all_seeds)} available seeds, using {len(all_seeds)}",
            "yellow",
        )

    # Deterministically select seeds with training seed (reproducible)
    rng = random.Random(train_seed)
    rng.shuffle(all_seeds)
    eval_seeds = all_seeds[:n_total]

    cprint(
        f"Evaluating on {n_total} seeds from pool "
        f"(training_seed={train_seed}, first seed={eval_seeds[0]}) ...",
        "cyan",
    )

    env_runner.eval_seeds = eval_seeds
    result = env_runner.run(
        agent,
        denoise_timesteps=denoise_steps,
        eval_episodes=n_total,
        video_save_dir=None,
    )

    # ── 5. Compute metrics ────────────────────────────────────────────
    per_seed_details: list[dict] = collect_episode_details(result)
    n_success = sum(1 for d in per_seed_details if d.get("success"))
    success_rate = n_success / n_total if n_total > 0 else 0.0

    task_done_steps = [
        d["steps"] for d in per_seed_details if d.get("success") and d.get("steps") is not None
    ]
    avg_steps = float(sum(task_done_steps) / len(task_done_steps)) if task_done_steps else None

    # ── 6. Report ─────────────────────────────────────────────────────
    avg_str = f"{avg_steps:.1f}" if avg_steps is not None else "N/A"
    cprint(f"\n{'=' * 50}", "cyan")
    cprint(f"  Checkpoint   : {ckpt_label}", "cyan")
    cprint(f"  Seeds        : {n_total}", "cyan")
    cprint(f"  Success rate : {n_success}/{n_total} = {success_rate:.1%}", "green")
    cprint(f"  Avg steps    : {avg_str}", "cyan")
    cprint(f"{'=' * 50}\n", "cyan")

    # ── 7. Save result (RoboTwin _result.txt format) ──────────────────
    eval_dir = exp_dir / "eval_robotwin"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # RoboTwin-style: just the float on one line
    result_file = eval_dir / "_result.txt"
    result_file.write_text(f"{success_rate}\n")

    # Also save detailed JSON for record-keeping
    detail_file = eval_dir / "result_details.json"
    detail_file.write_text(
        json.dumps(
            {
                "ckpt_tag": ckpt_tag_or_path,
                "ckpt_path": str(ckpt_path),
                "success_rate": success_rate,
                "n_success": n_success,
                "n_total": n_total,
                "avg_steps": avg_steps,
                "training_seed": train_seed,
                "per_seed_details": per_seed_details,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    cprint(f"  Results saved to: {eval_dir}/_result.txt", "cyan")

    return success_rate, avg_steps, n_success, n_total


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
    _off = cfg.eval.get("offline", {}) if hasattr(cfg, "eval") else {}
    episodes = args.episodes if args.episodes is not None else _off.get("eval_episodes", 100)
    denoise_steps = (
        args.denoise_steps
        if args.denoise_steps is not None
        else ((_off.get("denoise_timesteps_list") or [10])[0])
    )
    use_ema = args.use_ema if args.use_ema is not None else _off.get("use_ema_for_eval", True)

    ckpt_tag_or_path = args.ckpt_path if args.ckpt_path else args.ckpt_tag

    evaluate_checkpoint_robotwin(
        exp_dir,
        cfg,
        ckpt_tag_or_path=ckpt_tag_or_path,
        episodes=episodes,
        denoise_steps=denoise_steps,
        use_ema=use_ema,
    )


if __name__ == "__main__":
    main()
