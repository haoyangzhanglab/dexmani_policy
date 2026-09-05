"""Offline best-checkpoint selector via fixed-seed two-stage evaluation.

Discovers milestone checkpoints from an experiment directory, runs a
deterministic two-stage evaluation to identify the single best checkpoint,
and writes its strict v2 selection record.

Algorithm
---------

**Stage 1 — Initial evaluation**:
    Run ``initial_episodes`` (default 25) on every discovered milestone
    checkpoint, using a fixed, deterministically-shuffled seed slice that is
    identical across checkpoints.

**Stage 2 — Tie-break**:
    When two or more checkpoints share the highest success rate, run a single
    additional batch of ``batch_size`` fresh seeds (the same slice for every
    tied candidate) and merge.  Equal denominators guarantee a fair comparison.

**Tiebreak** (when still tied after Stage 2):
    1. Higher success rate.
    2. Lower ``avg_steps`` (faster task completion).
    3. Higher ``global_step`` (more training).

**Fail-fast**: a load/model/CUDA error on any checkpoint aborts the run (never
silently treated as 0%); if every checkpoint scores 0%, no ``best_ckpt.json``
is written and the run exits non-zero.

Seed management
---------------
The full seed list is deterministically shuffled with the eval seed (same
convention as ``eval_best_ckpt``); ``all_seeds[:initial_episodes]`` are Stage 1
and the next ``batch_size`` are Stage 2.  ``BaseRunner.run_one_episode`` re-seeds
the policy RNG per episode so ``(checkpoint, seed)`` is reproducible.

Usage
-----
.. code-block:: bash

    python dexmani_policy/select_best_ckpt.py \\
        --policy-name dp3 --task-name pour --exp-name 2026-07-29_01-53_35

    bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \\
        --initial-episodes 25 --max-episodes 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.env_runner.base_runner import EvalEpisodeError
from dexmani_policy.training.eval_utils import (
    MilestoneCheckpoint,
    _get_eval_param,
    build_eval_components,
    collect_episode_details,
    discover_milestone_checkpoints,
    load_ckpt_for_inference,
    resolve_eval_seed,
    validate_denoise_steps,
    validate_eval_config,
)

ROOT_DIR = set_project_root()
register_resolvers()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CkptEvalAccum:
    """Accumulated evaluation results for a single checkpoint."""

    ckpt: MilestoneCheckpoint
    success_list: list[bool] = field(default_factory=list)
    episode_details: list[dict] = field(default_factory=list)
    task_done_steps: list[int] = field(default_factory=list)

    # ---- derived ----

    @property
    def success_rate(self) -> float:
        if not self.success_list:
            return 0.0
        return float(np.mean(self.success_list))

    @property
    def n_episodes(self) -> int:
        return len(self.success_list)

    @property
    def avg_steps(self) -> float | None:
        if not self.task_done_steps:
            return None
        return float(np.mean(self.task_done_steps))

    @property
    def success_count(self) -> int:
        return sum(self.success_list)

    def merge(self, result: Dict[str, Any]) -> None:
        """Absorb per-episode results from one ``env_runner.run()`` call."""
        details: List[dict] = collect_episode_details(result)
        for d in details:
            self.episode_details.append(d)
            success = bool(d.get("success", False))
            self.success_list.append(success)
            steps = d.get("steps")
            if success and steps is not None:
                self.task_done_steps.append(steps)


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint(
    agent,
    env_runner,
    checkpoint_store: CheckpointStore,
    ckpt: MilestoneCheckpoint,
    seeds: List[int],
    use_ema: bool,
    denoise_steps: int,
    device: torch.device,
    video_save_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run *len(seeds)* episodes for one checkpoint.  Returns env_runner result dict."""

    load_ckpt_for_inference(agent, checkpoint_store, ckpt.path, use_ema)
    agent.to(device)
    agent.eval()

    env_runner.eval_seeds = list(seeds)
    env_runner.record_video = video_save_dir is not None
    return env_runner.run(
        agent,
        denoise_timesteps=denoise_steps,
        eval_episodes=len(seeds),
        video_save_dir=video_save_dir,
    )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _format_rate(num: int, den: int) -> str:
    pct = (num / den * 100) if den > 0 else 0.0
    return f"{num}/{den} ({pct:.1f}%)"


def _print_table(accumulators: list[CkptEvalAccum], title: str) -> None:
    cprint(title, "cyan", attrs=["bold"])
    header = f"  {'Checkpoint':<24} {'Success':<18} {'Avg Steps':<12}"
    cprint(header, "cyan")
    cprint("  " + "-" * 54, "cyan")
    for a in accumulators:
        sr = _format_rate(a.success_count, a.n_episodes)
        avg = f"{a.avg_steps:.1f}" if a.avg_steps is not None else "N/A"
        cprint(f"  {a.ckpt.label:<24} {sr:<18} {avg:<12}", "cyan")
    print()


def _rank_key(a: CkptEvalAccum) -> tuple[float, float, int]:
    """Sort key: success_rate → lower avg_steps → higher global_step."""
    return (
        a.success_rate,
        -(a.avg_steps if a.avg_steps is not None else float("inf")),
        a.ckpt.global_step,
    )


def select_best_checkpoint(
    exp_dir: Path,
    cfg,  # pre-loaded OmegaConf config (with _exp_dir injected)
    *,
    initial_episodes: int = 25,
    batch_size: int = 5,
    max_episodes: int = 100,
    denoise_steps: int = 10,
    use_ema: bool = True,
    eval_seed: int | None = None,
    video_save_dir: Path | None = None,
) -> tuple[MilestoneCheckpoint, list[CkptEvalAccum]]:
    """Run fixed two-stage evaluation with an optional exact-tie batch.

    Parameters
    ----------
    cfg : OmegaConf config, already loaded and validated, with
        ``_exp_dir`` set to the experiment directory path.

    Returns
    -------
    (best_ckpt, all_accumulators)
        The winning checkpoint and the full accumulator list (for reporting).
    """

    # ── 1. Validate config ────────────────────────────────────────────
    validate_eval_config(cfg)

    seed = resolve_eval_seed(cfg, cli_seed=eval_seed)
    set_seed(seed)

    device = torch.device(cfg.training.device)

    # ── 2. Discover checkpoints ───────────────────────────────────────
    milestones = discover_milestone_checkpoints(exp_dir)
    cprint(f"\nDiscovered {len(milestones)} milestone checkpoint(s):", "cyan")
    for mc in milestones:
        cprint(f"  {mc.label}", "cyan")

    # ── 3. Build components ───────────────────────────────────────────
    agent, env_runner, checkpoint_store = build_eval_components(cfg, device)
    eval_root_dir = exp_dir / "eval_ckpt_selector"

    # A malformed seed source must not cause duplicate environment episodes or
    # duplicate seed metadata in the selection record.
    all_seeds = list(dict.fromkeys(env_runner.get_seed_list()))
    if max_episodes > len(all_seeds):
        cprint(
            f"⚠ max_episodes ({max_episodes}) > available seeds "
            f"({len(all_seeds)}), capping at {len(all_seeds)}",
            "yellow",
        )
        max_episodes = len(all_seeds)
    if initial_episodes > max_episodes:
        initial_episodes = max_episodes

    # Deterministically shuffle seeds (same convention as eval_best_ckpt)
    rng = random.Random(seed)
    rng.shuffle(all_seeds)

    # Fixed, deterministic seed slices — identical for every checkpoint, so
    # equal-denominator comparisons and reproducible results hold.
    phase1_seeds = all_seeds[:initial_episodes]
    tie_seeds = all_seeds[
        initial_episodes : min(initial_episodes + batch_size, max_episodes)
    ]

    # ── 4. Stage 1: initial evaluation on all checkpoints ─────────────
    cprint(
        f"\n{'=' * 60}\n  Phase 1: Initial Evaluation ({len(phase1_seeds)} episodes each)\n{'=' * 60}",
        "cyan",
        attrs=["bold"],
    )

    accumulators: list[CkptEvalAccum] = []
    for mc in milestones:
        cprint(f"  Evaluating {mc.label} ...", "cyan")
        # No try/except: a load/model/CUDA failure is fatal and aborts the run
        # (an errored checkpoint must not be silently treated as 0%).
        result = evaluate_checkpoint(
            agent,
            env_runner,
            checkpoint_store,
            mc,
            phase1_seeds,
            use_ema,
            denoise_steps,
            device,
            video_save_dir=video_save_dir,
        )
        acc = CkptEvalAccum(ckpt=mc)
        acc.merge(result)
        accumulators.append(acc)
        cprint(f"    -> {_format_rate(acc.success_count, acc.n_episodes)}", "green")

    if not accumulators:
        raise RuntimeError("No checkpoints could be evaluated.")

    _print_table(accumulators, "Phase 1 Results:")

    # ── 5. Tie-break (single deterministic batch, same seeds for all tied) ──
    best_rate = max(a.success_rate for a in accumulators)
    tied = [a for a in accumulators if a.success_rate == best_rate]

    tie_break_used = len(tied) > 1 and bool(tie_seeds)
    if tie_break_used:
        cprint(
            f"\n⚠ Tied at top ({len(tied)} checkpoints at {best_rate:.1%}). "
            f"Running a single tie-break batch (+{len(tie_seeds)} episodes each)...",
            "yellow",
        )
        for acc in tied:
            cprint(f"    Evaluating {acc.ckpt.label} ...", "cyan")
            result = evaluate_checkpoint(
                agent,
                env_runner,
                checkpoint_store,
                acc.ckpt,
                tie_seeds,
                use_ema,
                denoise_steps,
                device,
                video_save_dir=video_save_dir,
            )
            acc.merge(result)
            cprint(
                f"      -> {_format_rate(acc.success_count, acc.n_episodes)}", "green"
            )

        # Recompute the tied set on the merged results — every tied candidate
        # consumed the same tie_seeds, so denominators stay equal.
        best_rate = max(a.success_rate for a in tied)
        tied = [a for a in tied if a.success_rate == best_rate]

        _print_table(accumulators, "Tie-break Results:")

    # ── 6. Final selection (single rank key) ──────────────────────────
    best = max(tied, key=_rank_key)

    avg_str = f"{best.avg_steps:.1f}" if best.avg_steps is not None else "N/A"
    cprint(f"\n{'=' * 60}", "green", attrs=["bold"])
    if len(tied) == 1:
        cprint(
            f"  ✅ Best checkpoint: {best.ckpt.label}\n"
            f"     Success: {_format_rate(best.success_count, best.n_episodes)}\n"
            f"     Avg steps: {avg_str}",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            f"  ⚠ Still tied after tie-break. Tiebreak by avg_steps → global_step.\n"
            f"  ✅ Best checkpoint: {best.ckpt.label}\n"
            f"     Success: {_format_rate(best.success_count, best.n_episodes)}\n"
            f"     Avg steps: {avg_str}",
            "yellow",
            attrs=["bold"],
        )
    cprint(f"{'=' * 60}\n", "green", attrs=["bold"])

    # ── 7. All-fail guard ─────────────────────────────────────────────
    if best.success_count == 0:
        eval_root_dir.mkdir(parents=True, exist_ok=True)
        summary_path = eval_root_dir / "best_ckpt_selection.json"
        summary = {
            "best_checkpoint": None,
            "error": "All milestone checkpoints scored 0% success",
            "all_results": [
                {
                    "pct": a.ckpt.pct,
                    "global_step": a.ckpt.global_step,
                    "success_rate": a.success_rate,
                    "avg_steps": a.avg_steps,
                    "n_episodes": a.n_episodes,
                    "path": str(a.ckpt.path),
                }
                for a in accumulators
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        cprint(f"  Summary saved to: {summary_path}", "cyan")
        raise RuntimeError(
            "All milestone checkpoints scored 0% success — refusing to write best_ckpt.json."
        )

    # Save summary JSON
    eval_root_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_checkpoint": {
            "path": str(best.ckpt.path),
            "pct": best.ckpt.pct,
            "global_step": best.ckpt.global_step,
            "success_rate": best.success_rate,
            "avg_steps": best.avg_steps,
            "n_episodes": best.n_episodes,
        },
        "all_results": [
            {
                "pct": a.ckpt.pct,
                "global_step": a.ckpt.global_step,
                "success_rate": a.success_rate,
                "avg_steps": a.avg_steps,
                "n_episodes": a.n_episodes,
                "path": str(a.ckpt.path),
            }
            for a in accumulators
        ],
    }
    summary_path = eval_root_dir / "best_ckpt_selection.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    cprint(f"  Summary saved to: {summary_path}", "cyan")

    # ── Save best_ckpt.json in experiment root (handoff to eval_best_ckpt) ──
    ckpt_relpath = best.ckpt.path.resolve().relative_to(exp_dir.resolve())
    selection_seeds = list(phase1_seeds)
    if tie_break_used:
        selection_seeds.extend(
            seed for seed in tie_seeds if seed not in selection_seeds
        )
    temporal_ensemble_coeff = cfg.env_runner.get("temporal_ensemble_coeff", None)
    best_info = {
        "record_version": 2,
        "ckpt_relpath": str(ckpt_relpath),
        "pct": best.ckpt.pct,
        "global_step": best.ckpt.global_step,
        "success_rate": best.success_rate,
        "avg_steps": best.avg_steps,
        "n_episodes": best.n_episodes,
        "inference": {
            "use_ema": bool(use_ema),
            "denoise_steps": int(denoise_steps),
            "temporal_ensemble_coeff": temporal_ensemble_coeff,
            "policy_seed_mode": "episode_seed",
        },
        "selection": {
            "shuffle_seed": seed,
            "seeds": selection_seeds,
            "initial_episodes": len(phase1_seeds),
            "tie_break_used": tie_break_used,
        },
    }
    best_info_path = exp_dir / "best_ckpt.json"
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    cprint(f"  Best checkpoint record saved to: {best_info_path}", "green")

    return best.ckpt, accumulators


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline best-checkpoint selector via fixed two-stage evaluation "
            "with an optional exact-tie batch."
        ),
    )
    parser.add_argument(
        "--policy-name",
        type=str,
        required=True,
        help="Policy config name (e.g. dp3, maniflow).",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name (e.g. pour, pick_apple_messy).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment timestamp/name under experiments/<policy>/<task>/.",
    )
    parser.add_argument(
        "--initial-episodes",
        type=int,
        default=None,
        help="Phase-1 episodes per checkpoint (default: from config eval.select_best).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Additional episodes in the optional exact-tie stage (default: from config).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Hard cap covering the initial stage plus one optional tie-break batch "
            "(default: from config)."
        ),
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=None,
        help="DDIM / Euler denoising steps at inference (default: from config).",
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
        help="Use raw model weights instead of EMA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Eval seed override (default: training.seed + 1024).",
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
        help="Optional OmegaConf dot-list overrides (merged onto config.yaml).",
    )
    args = parser.parse_args()

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

    # ── Load and validate config once ─────────────────────────────────
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        cprint(f"Error: config.yaml not found: {cfg_path}", "red")
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    # Stash exp_dir so build_eval_components can build paths
    cfg._exp_dir = str(exp_dir)

    # ── Resolve parameters: CLI > config > defaults ───────────────────────
    _sb = cfg.eval.get("select_best", {}) if hasattr(cfg, "eval") else {}
    initial_episodes = (
        args.initial_episodes
        if args.initial_episodes is not None
        else _sb.get("initial_episodes", 25)
    )
    batch_size = (
        args.batch_size if args.batch_size is not None else _sb.get("batch_size", 5)
    )
    max_episodes = (
        args.max_episodes
        if args.max_episodes is not None
        else _sb.get("max_episodes", 100)
    )
    denoise_steps = (
        args.denoise_steps
        if args.denoise_steps is not None
        else _get_eval_param(cfg, "denoise_steps", "select_best", default=10)
    )
    use_ema = (
        args.use_ema
        if args.use_ema is not None
        else _get_eval_param(cfg, "use_ema", "select_best", default=True)
    )

    # Video saving — enabled by default, configurable via eval.video.enabled.
    # Output: exp_dir/eval_ckpt_selector/<timestamp>/episode_<seed>.mp4
    video_enabled = _get_eval_param(cfg, "enabled", "video", default=True)
    video_save_dir = None
    if video_enabled and not args.no_videos:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_save_dir = exp_dir / "eval_ckpt_selector" / timestamp
        Path(video_save_dir).mkdir(parents=True, exist_ok=True)
        cprint(f"\n📹 Video output: {video_save_dir}", "cyan")

    if initial_episodes <= 0 or max_episodes <= 0 or batch_size < 0:
        cprint(
            "Error: initial/max episodes must be positive and batch size non-negative "
            f"(got {initial_episodes}/{max_episodes}/{batch_size})",
            "red",
        )
        sys.exit(1)

    try:
        select_best_checkpoint(
            exp_dir,
            cfg,
            initial_episodes=initial_episodes,
            batch_size=batch_size,
            max_episodes=max_episodes,
            denoise_steps=denoise_steps,
            use_ema=use_ema,
            eval_seed=args.seed,
            video_save_dir=video_save_dir,
        )
    except EvalEpisodeError as e:
        cprint(f"Fatal eval error (category={e.category}, seed={e.seed}): {e}", "red")
        sys.exit(1)
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as e:
        cprint(f"Selection failed: {type(e).__name__}: {e}", "red")
        sys.exit(1)

if __name__ == "__main__":
    main()
