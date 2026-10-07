"""Offline best-checkpoint selector via fixed-seed two-stage evaluation.

Discovers milestone checkpoints from an experiment directory, runs a
two-stage evaluation on deterministically selected paired seeds,
and records the selected checkpoint, inference settings and selection seeds.

Algorithm
---------

**Stage 1 — Initial evaluation**:
    Evaluate every discovered milestone on the required manifest's selection seeds.
    All candidates use the same seeds.

**Stage 2 — Tie-break**:
    When two or more checkpoints share the highest success rate, run a single
    additional batch from the manifest's tie-break seeds and merge.
    All tied candidates share it; unused tie-break seeds remain reserved.

**Tiebreak** (when still tied after Stage 2):
    1. Higher success rate.
    2. Lower ``avg_steps`` (faster task completion).
    3. Higher ``global_step`` (more training).

**Fail-fast**: a load/model/CUDA error on any checkpoint aborts the run (never
silently treated as 0%). Complete, normal all-zero results are published with
``selection_all_zero=true`` using the same deterministic ranking.
Failed runs keep their own summary and leave the last successful
``best_ckpt.json`` unchanged.

Seed management
---------------
``eval.seed_manifest`` fixes disjoint selection, tie-break and test lists.
New selections require a valid manifest, generated once from the actual pool
with ``scripts/eval/make_seed_manifest.py``. ``BaseRunner.run_one_episode`` re-seeds
the policy RNG per episode. This makes seed selection and RNG initialization
repeatable; it does not guarantee bitwise-identical trajectories across
GPU/driver/kernel environments.

Successful selections save their own ``selection_result.json``; ``--result-file``
also writes a new handoff file for downstream ``--selection-record`` consumers.
``best_ckpt.json`` remains the latest successful selection alias.

Usage
-----
.. code-block:: bash

    python dexmani_policy/select_best_ckpt.py \\
        --policy-name dp3 --task-name pour --exp-name 2026-07-29_01-53_35

    bash scripts/eval/select_best_ckpt.sh dp3 pour 2026-07-29_01-53_35 \\
        eval.seed_manifest=dexmani_policy/configs/eval_protocols/pour.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.utils.path import set_project_root
from dexmani_policy.utils.random import set_seed
from dexmani_policy.env_runner.base_runner import EvalEpisodeError
from dexmani_policy.evaluation.protocol import (
    MilestoneCheckpoint,
    _get_eval_param,
    save_eval_snapshot, run_eval_plan, plan_size, artifact_reference, atomic_json, compute_eval_stats,
    add_inference_steps_argument,
    build_eval_runner,
    collect_episode_details,
    discover_milestone_checkpoints,
    iter_leaf_env_runners,
    load_ckpt_for_inference,
    parse_eval_overrides,
    resolve_eval_seed,
    validate_inference_steps,
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
    episode_details: list[dict] = field(default_factory=list)

    # ---- derived ----

    @property
    def success_rate(self) -> float:
        return self.success_count / self.n_episodes if self.n_episodes else 0.0

    @property
    def n_episodes(self) -> int:
        return len(self.episode_details)

    @property
    def avg_steps(self) -> float | None:
        steps = [d["steps"] for d in self.episode_details if d["success"] and d["steps"] is not None]
        return float(np.mean(steps)) if steps else None

    @property
    def success_count(self) -> int:
        return sum(d["success"] for d in self.episode_details)

    def merge(self, result: Dict[str, Any]) -> None:
        """Absorb per-episode results from one ``env_runner.run()`` call."""
        self.episode_details.extend(collect_episode_details(result))


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint(
    cfg,
    env_runner,
    ckpt: MilestoneCheckpoint,
    seeds: Dict[str, List[int]],
    use_ema: bool,
    inference_steps: int,
    video_save_dir: Path | None = None,
) -> Dict[str, Any]:
    """Evaluate one checkpoint on the complete physical task/seed plan."""

    agent = load_ckpt_for_inference(ckpt.path, use_ema, cfg=cfg)
    if agent._checkpoint_global_step != ckpt.global_step:
        raise ValueError("Checkpoint filename/global_step disagrees with saved training state")

    for leaf_runner in iter_leaf_env_runners(env_runner):
        leaf_runner.record_video = video_save_dir is not None
    return run_eval_plan(
        env_runner, agent, seeds,
        inference_steps=inference_steps,
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
    exp_dir: Path, cfg, *, max_episodes=100,
    inference_steps=10, use_ema=True, eval_seed=None, video_save_dir=None,
    result_file=None,
) -> tuple[MilestoneCheckpoint, list[CkptEvalAccum]]:
    """Fixed initial stage and one exact-tie batch, with durable run evidence."""
    import tempfile
    from dexmani_policy.evaluation.protocol import load_seed_manifest
    if result_file is not None and Path(result_file).exists():
        raise FileExistsError(f"Selection result already exists: {result_file}")
    validate_inference_steps([inference_steps])
    if type(max_episodes) is not int or max_episodes <= 0:
        raise ValueError("max_episodes must be a positive integer")
    exp_dir = Path(exp_dir).resolve()
    root = exp_dir / "eval_ckpt_selector"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    summary_path = run_dir / "best_ckpt_selection.json"
    record = {"selection_id": run_dir.name, "status": "running", "stages": [], "all_results": []}
    accumulators = []
    published = False
    try:
        seed = resolve_eval_seed(cfg, cli_seed=eval_seed)
        set_seed(seed)
        milestones = discover_milestone_checkpoints(exp_dir)
        runner = build_eval_runner(cfg)
        for leaf in iter_leaf_env_runners(runner):
            leaf.record_video = video_save_dir is not None
        requested = {"max_episodes": max_episodes}
        manifest_source = cfg.get("eval", {}).get("seed_manifest")
        if manifest_source is None:
            raise ValueError("New selection requires eval.seed_manifest; generate it with "
                             "scripts/eval/make_seed_manifest.py and pass eval.seed_manifest=<path>")
        try:
            protocol = load_seed_manifest(manifest_source, runner)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Seed manifest not found: {manifest_source}; generate it with "
                "scripts/eval/make_seed_manifest.py or pass eval.seed_manifest=<path>"
            ) from exc
        phase1_seeds = protocol["roles"]["selection"]
        tie_seeds = protocol["roles"]["tie_break"]
        initial_count = plan_size(phase1_seeds)
        if initial_count + plan_size(tie_seeds) > max_episodes:
            raise ValueError(
                f"max_episodes={max_episodes} is below the manifest selection + reserved "
                f"tie-break count ({initial_count}+{plan_size(tie_seeds)}); increase the cap"
            )
        record["eval_config"] = save_eval_snapshot(
            run_dir, cfg, runner, protocol=protocol, use_ema=use_ema, inference_steps=inference_steps,
            shuffle_seed=seed, policy_seed_mode="episode_seed", **requested,
            effective_episode_counts={role: plan_size(seeds) for role, seeds in protocol["roles"].items()},
            phase1_task_seeds=phase1_seeds,
            possible_tie_task_seeds=tie_seeds,
            checkpoints=[{"path": artifact_reference(m.path, exp_dir), "global_step": m.global_step} for m in milestones],
        )
        selection = {"shuffle_seed": seed, "task_seeds": {},
                     "tie_break_used": False}
        selection["protocol"] = "explicit"
        selection["seed_manifest"] = protocol
        record["selection"] = selection

        def dispatch(mc, seeds, phase):
            mapping = seeds
            for task, values in mapping.items():
                selection["task_seeds"].setdefault(task, [])
                selection["task_seeds"][task] = list(dict.fromkeys(selection["task_seeds"][task] + values))
            video = None
            if video_save_dir is not None:
                video = Path(video_save_dir) / run_dir.name / mc.path.stem / phase
                if not hasattr(runner, "runners"):
                    video = video / runner.task_name
            stage = {"checkpoint": artifact_reference(mc.path, exp_dir), "global_step": mc.global_step,
                     "phase": phase, "requested_task_seeds": mapping, "status": "requested",
                     "video_dir": artifact_reference(video, exp_dir) if video is not None else None}
            record["stages"].append(stage)
            atomic_json(summary_path, record)
            result = evaluate_checkpoint(cfg, runner, mc, seeds, use_ema, inference_steps, video_save_dir=video)
            details = [dict(d, checkpoint=stage["checkpoint"], phase=phase,
                            task_name=d.get("task_name", getattr(runner, "task_name", None)))
                       for d in collect_episode_details(result)]
            stage["episode_details"] = details
            stage.update(status="completed", statistics=compute_eval_stats(result))
            atomic_json(summary_path, record)
            return {"episode_details": details}

        for mc in milestones:
            acc = CkptEvalAccum(ckpt=mc)
            acc.merge(dispatch(mc, phase1_seeds, "initial"))
            accumulators.append(acc)
        best_rate = max(a.success_rate for a in accumulators)
        tied = [a for a in accumulators if a.success_rate == best_rate]
        selection["tie_break_used"] = len(tied) > 1 and plan_size(tie_seeds) > 0
        if selection["tie_break_used"]:
            for acc in tied:
                acc.merge(dispatch(acc.ckpt, tie_seeds, "tie_break"))
            best_rate = max(a.success_rate for a in tied)
            tied = [a for a in tied if a.success_rate == best_rate]
        best = max(tied, key=_rank_key)
        def candidate(a):
            return {"ckpt_relpath": artifact_reference(a.ckpt.path, exp_dir), "pct": a.ckpt.pct,
                    "global_step": a.ckpt.global_step, "success_rate": a.success_rate,
                    "success_count": a.success_count, "avg_steps": a.avg_steps,
                    "n_episodes": a.n_episodes, "episode_details": a.episode_details}
        record["all_results"] = [candidate(a) for a in accumulators]
        selection["selection_all_zero"] = all(a.success_count == 0 for a in accumulators)
        record.update(status="success", best_checkpoint=candidate(best),
                      inference={"use_ema": use_ema, "inference_steps": inference_steps,
                                 "policy_seed_mode": "episode_seed"})
        atomic_json(summary_path, record)
        best_info = {k: v for k, v in candidate(best).items() if k != "episode_details"}
        best_info.update(selection_id=run_dir.name,
                         selection_summary=artifact_reference(summary_path, exp_dir),
                         inference={"use_ema": use_ema, "inference_steps": inference_steps,
                                    "policy_seed_mode": "episode_seed"}, selection=selection)
        # Immutable per-selection record is also the pipeline handoff.
        best_info.update(_format="selection.v2", status="success",
                         all_results=record["all_results"], stages=record["stages"])
        result_path = run_dir / "selection_result.json"
        atomic_json(result_path, best_info)
        reference = {"selection_result": artifact_reference(result_path, exp_dir)}
        if result_file is not None:
            atomic_json(result_file, reference, overwrite=False)
        atomic_json(exp_dir / "best_ckpt.json", reference)
        published = True
        _print_table(accumulators, "Selection results:")
        cprint(f"Published best: {best.ckpt.label}, selection={run_dir.name}", "green")
        return best.ckpt, accumulators
    except BaseException as exc:
        if published:
            raise
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            atomic_json(summary_path, record)
        except OSError as save_error:
            print(f"Could not save failed selection summary: {save_error}")
        print(f"Selection {run_dir.name} failed; previous successful best is unchanged.")
        raise


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
    parser.add_argument("--initial-episodes", "--batch-size", dest="removed_budget",
                        action="append", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Hard cap for manifest selection plus reserved tie-break seeds."
        ),
    )
    add_inference_steps_argument(parser)
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
        help="Disable selection video recording (default).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Evaluation/environment dot-list overrides; agent.* is forbidden (saved-config-owned).",
    )
    parser.add_argument("--videos", action="store_true", help="Explicitly record selection candidate videos")
    parser.add_argument("--result-file", default=None)
    args = parser.parse_args()
    if args.removed_budget is not None:
        parser.error("--initial-episodes/--batch-size were removed; episode lists come from eval.seed_manifest")

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

    # ── Load evaluation config; saved Agent configuration is used for every candidate ──
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        cprint(f"Error: config.yaml not found: {cfg_path}", "red")
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, parse_eval_overrides(args.overrides))
    # Keep the experiment path available to saved-config restoration
    cfg._exp_dir = str(exp_dir)

    # ── Resolve parameters: CLI > config > defaults ───────────────────────
    _sb = cfg.eval.get("select_best", {}) if hasattr(cfg, "eval") else {}
    max_episodes = (
        args.max_episodes
        if args.max_episodes is not None
        else _sb.get("max_episodes", 100)
    )
    inference_steps = (
        args.inference_steps
        if args.inference_steps is not None
        else _get_eval_param(cfg, "inference_steps", "select_best", default=10)
    )
    use_ema = (
        args.use_ema
        if args.use_ema is not None
        else _get_eval_param(cfg, "use_ema", "select_best", default=True)
    )

    # Selection videos are opt-in; each run/candidate/stage owns its path.
    video_save_dir = exp_dir / "eval_ckpt_selector" / "videos" if args.videos and not args.no_videos else None


    try:
        select_best_checkpoint(
            exp_dir,
            cfg,
            max_episodes=max_episodes,
            inference_steps=inference_steps,
            use_ema=use_ema,
            result_file=args.result_file,
            eval_seed=args.seed,
            video_save_dir=video_save_dir,
        )
    except EvalEpisodeError as e:
        cprint(f"Fatal eval error (category={e.category}, seed={e.seed}): {e}", "red")
        sys.exit(1)
    except (ValueError, RuntimeError, OSError) as e:
        cprint(f"Selection failed: {type(e).__name__}: {e}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
