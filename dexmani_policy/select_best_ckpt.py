"""Offline best-checkpoint selector via adaptive elimination evaluation.

Discovers milestone checkpoints from an experiment directory, runs a
batched-elimination evaluation to identify the single best checkpoint,
and optionally symlinks it as ``best.pt``.

Algorithm
---------

**Phase 1 — Initial evaluation**:
    Run ``initial_episodes`` (default 25) on every discovered milestone
    checkpoint.  If one checkpoint has a strictly higher success rate than
    all others, it is selected immediately.

**Phase 2 — Incremental tie-break**:
    When two or more checkpoints share the highest success rate, additional
    batches (``batch_size`` episodes each, using **fresh** seeds) are run
    only on the tied checkpoints.  This repeats until:

    * a unique best emerges, or
    * ``max_episodes`` total episodes have been run on the tied candidates.

**Tiebreak** (when ``max_episodes`` is exhausted with no unique winner):
    1. Higher success rate.
    2. Lower ``avg_steps`` (faster task completion).
    3. Higher ``global_step`` (more training).

Seed management
---------------
``SimRunner.run()`` reads ``self.eval_seeds`` on every call.  We slice
the full seed list so that each call uses a non-overlapping range,
avoiding wasted repeated episodes.

Usage
-----
.. code-block:: bash

    python dexmani_policy/select_best_ckpt.py \\
        --policy-name dp3 --task-name pour --exp-name 2026-07-29_01-53_35

    bash scripts/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \\
        --initial-episodes 25 --max-episodes 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.training.eval_utils import (
    build_eval_components,
    collect_episode_details,
    load_ckpt_for_inference,
    validate_eval_config,
)

ROOT_DIR = set_project_root()
register_resolvers()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_MILESTONE_RE = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    """A discovered milestone checkpoint."""

    path: Path
    pct: int  # 20, 40, 60, 80, 100
    global_step: int

    @property
    def label(self) -> str:
        return f"{self.pct}% (step={self.global_step})"


@dataclass
class CkptEvalAccum:
    """Accumulated evaluation results for a single checkpoint."""

    ckpt: MilestoneCheckpoint
    success_list: list[bool] = field(default_factory=list)
    episode_details: list[dict] = field(default_factory=list)
    task_done_steps: list[int] = field(default_factory=list)
    seed_offset: int = 0

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
# Discovery
# ---------------------------------------------------------------------------


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    """Find milestone checkpoints in *exp_dir/checkpoints/*, sorted by pct."""

    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Make sure the experiment was run with the new step-driven "
            f"training loop (total_train_steps in config)."
        )

    found: list[MilestoneCheckpoint] = []
    for pt_file in sorted(ckpt_dir.glob("epoch=*.pt")):
        m = _MILESTONE_RE.match(pt_file.name)
        if not m:
            continue
        found.append(
            MilestoneCheckpoint(
                path=pt_file,
                pct=int(m.group("pct")),
                global_step=int(m.group("step")),
            )
        )

    if not found:
        raise FileNotFoundError(
            f"No milestone checkpoints found in {ckpt_dir}.\n"
            f"Expected filenames like: epoch=*-step=*-milestone=20pct.pt\n"
            f"Run training with the new step-driven loop first."
        )

    found.sort(key=lambda c: c.pct)
    return found


# build_eval_components and load_ckpt_for_inference have been moved to
# dexmani_policy.training.eval_utils (imported above).


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
) -> Dict[str, Any]:
    """Run *len(seeds)* episodes for one checkpoint.  Returns env_runner result dict."""

    load_ckpt_for_inference(agent, checkpoint_store, ckpt.path, use_ema)
    agent.to(device)
    agent.eval()

    env_runner.eval_seeds = list(seeds)
    return env_runner.run(
        agent,
        denoise_timesteps=denoise_steps,
        eval_episodes=len(seeds),
        video_save_dir=None,
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


def _best_among_tied(
    tied: list[CkptEvalAccum],
) -> CkptEvalAccum:
    """Tiebreak: highest success_rate → lowest avg_steps → highest global_step."""
    return max(
        tied,
        key=lambda a: (
            a.success_rate,
            -(a.avg_steps if a.avg_steps is not None else float("inf")),
            a.ckpt.global_step,
        ),
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
) -> tuple[MilestoneCheckpoint, list[CkptEvalAccum]]:
    """Run adaptive elimination to select the best checkpoint.

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

    seed = (
        eval_seed
        if eval_seed is not None
        else (cfg.eval.get("seed") if hasattr(cfg, "eval") else cfg.training.get("seed", 0))
    )
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

    all_seeds = env_runner.get_seed_list()
    if max_episodes > len(all_seeds):
        cprint(
            f"⚠ max_episodes ({max_episodes}) > available seeds "
            f"({len(all_seeds)}), capping at {len(all_seeds)}",
            "yellow",
        )
        max_episodes = len(all_seeds)
    if initial_episodes > max_episodes:
        initial_episodes = max_episodes

    # ── 4. Phase 1: initial evaluation on all checkpoints ─────────────
    cprint(
        f"\n{'=' * 60}\n  Phase 1: Initial Evaluation ({initial_episodes} episodes each)\n{'=' * 60}",
        "cyan",
        attrs=["bold"],
    )

    accumulators: list[CkptEvalAccum] = []
    phase1_seeds = all_seeds[:initial_episodes]

    for mc in milestones:
        cprint(f"  Evaluating {mc.label} ...", "cyan")
        try:
            result = evaluate_checkpoint(
                agent,
                env_runner,
                checkpoint_store,
                mc,
                phase1_seeds,
                use_ema,
                denoise_steps,
                device,
            )
            acc = CkptEvalAccum(ckpt=mc, seed_offset=initial_episodes)
            acc.merge(result)
            accumulators.append(acc)
            cprint(
                f"    -> {_format_rate(acc.success_count, acc.n_episodes)}",
                "green",
            )
        except Exception as exc:
            cprint(f"    -> SKIPPED: {exc}", "red")
            # Still add a zero-result accumulator so the table is complete
            accumulators.append(CkptEvalAccum(ckpt=mc, seed_offset=initial_episodes))

    if not accumulators:
        raise RuntimeError("No checkpoints could be evaluated.")

    _print_table(accumulators, "Phase 1 Results:")

    # ── 5. Check for unique best ──────────────────────────────────────
    best_rate = max(a.success_rate for a in accumulators)
    tied = [a for a in accumulators if a.success_rate == best_rate]

    skip_phase2 = len(tied) == 1
    if skip_phase2:
        cprint(
            f"✅ Unique best after Phase 1: {tied[0].ckpt.label} "
            f"— {_format_rate(tied[0].success_count, tied[0].n_episodes)}",
            "green",
            attrs=["bold"],
        )

    # ── 6. Phase 2: incremental tie-break ─────────────────────────────
    if not skip_phase2:
        cprint(
            f"\n⚠ Tied at top ({len(tied)} checkpoints at {best_rate:.1%}). Running tie-break rounds...",
            "yellow",
        )
        round_num = 0
        seed_offset = initial_episodes

        while len(tied) > 1 and seed_offset < max_episodes:
            round_num += 1
            batch_end = min(seed_offset + batch_size, max_episodes)
            batch_seeds = all_seeds[seed_offset:batch_end]
            actual_batch = len(batch_seeds)
            if actual_batch == 0:
                break

            cprint(
                f"\n  Tie-break Round {round_num} (+{actual_batch} episodes on {len(tied)} tied ckpt(s))",
                "cyan",
            )

            for acc in tied:
                cprint(f"    Evaluating {acc.ckpt.label} ...", "cyan")
                try:
                    result = evaluate_checkpoint(
                        agent,
                        env_runner,
                        checkpoint_store,
                        acc.ckpt,
                        batch_seeds,
                        use_ema,
                        denoise_steps,
                        device,
                    )
                    acc.merge(result)
                    acc.seed_offset = batch_end
                    cprint(
                        f"      -> {_format_rate(acc.success_count, acc.n_episodes)}",
                        "green",
                    )
                except Exception as exc:
                    cprint(f"      -> batch failed: {exc}", "red")

            seed_offset = batch_end

            # Recompute tied set
            best_rate = max(a.success_rate for a in tied)
            tied = [a for a in tied if a.success_rate == best_rate]

            if len(tied) > 1:
                cprint(
                    "    Still tied: "
                    + ", ".join(
                        f"{a.ckpt.pct}%({_format_rate(a.success_count, a.n_episodes)})" for a in tied
                    ),
                    "yellow",
                )

    # ── 7. Final selection ────────────────────────────────────────────
    best = _best_among_tied(tied)

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
            f"  ⚠ Max episodes reached. Tiebreak by avg_steps → global_step.\n"
            f"  ✅ Best checkpoint: {best.ckpt.label}\n"
            f"     Success: {_format_rate(best.success_count, best.n_episodes)}\n"
            f"     Avg steps: {avg_str}",
            "yellow",
            attrs=["bold"],
        )
    cprint(f"{'=' * 60}\n", "green", attrs=["bold"])

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
    best_info = {
        "ckpt_path": str(best.ckpt.path),
        "ckpt_relpath": str(best.ckpt.path.relative_to(exp_dir)),
        "pct": best.ckpt.pct,
        "global_step": best.ckpt.global_step,
        "success_rate": best.success_rate,
        "avg_steps": best.avg_steps,
        "n_episodes": best.n_episodes,
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
        description="Offline best-checkpoint selector via adaptive evaluation.",
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
        help="Additional episodes per round in Phase 2 (default: from config).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Hard cap on total episodes per checkpoint (default: from config).",
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
        help="Eval seed override (default: from config.yaml eval.seed).",
    )
    parser.add_argument(
        "--link-best",
        action="store_true",
        help="Symlink the best checkpoint as checkpoints/best.pt.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dot-list overrides (merged onto config.yaml).",
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

    # ── Resolve parameters: CLI > config > hardcoded fallback ────────────
    _sb = cfg.eval.get("select_best", {}) if hasattr(cfg, "eval") else {}
    initial_episodes = (
        args.initial_episodes if args.initial_episodes is not None else _sb.get("initial_episodes", 25)
    )
    batch_size = args.batch_size if args.batch_size is not None else _sb.get("batch_size", 5)
    max_episodes = args.max_episodes if args.max_episodes is not None else _sb.get("max_episodes", 100)
    denoise_steps = args.denoise_steps if args.denoise_steps is not None else _sb.get("denoise_steps", 10)
    use_ema = args.use_ema if args.use_ema is not None else _sb.get("use_ema", True)

    best_ckpt, _all = select_best_checkpoint(
        exp_dir,
        cfg,
        initial_episodes=initial_episodes,
        batch_size=batch_size,
        max_episodes=max_episodes,
        denoise_steps=denoise_steps,
        use_ema=use_ema,
        eval_seed=args.seed,
    )

    if args.link_best:
        best_link = exp_dir / "checkpoints" / "best.pt"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(best_ckpt.path.name)
        cprint(f"  🔗 best.pt -> {best_ckpt.path.name}", "green")


if __name__ == "__main__":
    main()
