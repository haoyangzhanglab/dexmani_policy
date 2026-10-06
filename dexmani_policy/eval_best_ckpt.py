"""Checkpoint-owned evaluation on deterministically selected seeds.

Loads a checkpoint and runs the manifest's complete held-out test role.
Legacy records without a manifest exclude the recorded selection seeds
from a deterministic shuffled pool.
Each invocation writes success metrics and provenance to its own result
directory, including a scalar success rate in ``_result.txt``.

Evaluation protocol
-------------------

1. Resolve ``--selection-record`` or ``best`` once to a record and concrete
   checkpoint. A supplied handoff record rejects conflicting EMA/NFE overrides;
   ordinary ``best`` permits explicit inference overrides.
2. Use the selection record's embedded manifest and complete test role; an
   explicit manifest override must match. Without a pinned protocol, use the
   configured manifest or the legacy shuffled pool (training.seed + 1024).
3. Validate the full runner pool and held-out task/seed identities before
   loading weights; legacy evaluation excludes the recorded selection seeds.
4. Restore the saved Agent from the concrete checkpoint with the resolved
   EMA/raw choice. Single evaluation and NFE sweep share this setup.
5. Run each seed with environment/policy RNG reseeding and save statistics and
   provenance. This does not guarantee identical trajectories across
   GPU/driver/kernels.

The reported metrics are empirical success rates and step averages;
per-task statistics include 95% Wilson intervals.

Usage
-----
.. code-block:: bash

    # Pin the handoff exported by select_best_ckpt.py --result-file:
    bash scripts/eval/eval_best_ckpt.sh dp pick_apple_messy EXP_NAME \\
        --selection-record /absolute/path/selection_record.json

    # Specific checkpoint; a manifest still fixes the full test role:
    bash scripts/eval/eval_best_ckpt.sh dp pick_apple_messy EXP_NAME \\
        --ckpt-tag 20pct
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.agents.loader import resolve_best_checkpoint
from dexmani_policy.utils.path import set_project_root
from dexmani_policy.utils.random import set_seed
from dexmani_policy.env_runner.base_runner import EvalEpisodeError
from dexmani_policy.evaluation.protocol import (
    _get_eval_param,
    save_eval_snapshot, mapped_task_seeds, validate_heldout, artifact_reference, selection_provenance,
    add_inference_steps_argument,
    build_eval_runner,
    fixed_test_seeds,
    collect_episode_details,
    compute_eval_stats,
    iter_leaf_env_runners,
    load_ckpt_for_inference,
    parse_eval_overrides,
    resolve_checkpoint_path,
    resolve_eval_seed,
    validate_inference_steps,
)

ROOT_DIR = set_project_root()
register_resolvers()

# ---------------------------------------------------------------------------
# Shared helpers (used by both single-value and sweep paths)
# ---------------------------------------------------------------------------


def _prepare_result_dir(exp_dir: Path, result_save_dir: Path | None) -> Path:
    """Allocate a unique run directory; explicit directories must be empty."""
    if result_save_dir is None:
        root = exp_dir / "eval_dexsim"
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
        return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    result_save_dir = Path(result_save_dir)
    result_save_dir.mkdir(parents=True, exist_ok=True)
    if any(result_save_dir.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory must be empty: {result_save_dir}"
        )
    return result_save_dir


def _write_result(path: Path, text: str) -> None:
    """Never replace an artifact from a previous invocation."""
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def _setup_eval(
    cfg,
    exp_dir: Path,
    ckpt_tag_or_path: str,
    use_ema: bool,
    *,
    video_save_dir: Path | None = None,
    best_info=None,
    episodes: int,
    selection_seeds: list[int],
):
    """Resolve and validate seeds on the full runner before restoring the Agent.

    Returns
    -------
    (agent, env_runner, ckpt_path, ckpt_label, eval_seed, eval_seeds)
    """
    eval_seed = resolve_eval_seed(cfg)
    set_seed(eval_seed)

    env_runner = build_eval_runner(cfg)
    eval_seeds = fixed_test_seeds(cfg, env_runner, best_info)
    if eval_seeds is None:
        eval_seeds = _select_eval_seeds(
            env_runner, eval_seed, episodes, excluded_seeds=selection_seeds
        )
    elif len(eval_seeds) != episodes:
        cprint(f"Requested {episodes} episodes; manifest fixes {len(eval_seeds)} test seeds "
               "per task. Evaluating the full test role.", "yellow")
    validate_heldout(env_runner, best_info, eval_seeds)

    for leaf_runner in iter_leaf_env_runners(env_runner):
        leaf_runner.record_video = video_save_dir is not None

    ckpt_path, ckpt_label = resolve_checkpoint_path(exp_dir, ckpt_tag_or_path)

    cprint(f"\nLoading checkpoint: {ckpt_label} (EMA={use_ema})", "cyan")
    agent = load_ckpt_for_inference(ckpt_path, use_ema, cfg=cfg)
    cprint("✅ Checkpoint loaded\n", "green")

    return agent, env_runner, ckpt_path, ckpt_label, eval_seed, eval_seeds


def _selection_seeds(best_info) -> list[int]:
    """Require selection seeds so final evaluation can exclude them."""
    selection = best_info.get("selection")
    seeds = selection.get("seeds") if isinstance(selection, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "Held-out best evaluation requires non-empty unique integer "
            "selection.seeds in best_ckpt.json to exclude checkpoint-selection seeds"
        )
    return seeds


def _select_eval_seeds(
    env_runner,
    eval_seed: int,
    episodes: int,
    excluded_seeds: list[int] | None = None,
) -> list[int]:
    """Select deterministic seeds after excluding checkpoint-selection seeds."""
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")

    all_seeds = list(dict.fromkeys(env_runner.get_seed_list()))
    rng = random.Random(eval_seed)
    rng.shuffle(all_seeds)

    excluded = set(excluded_seeds or [])
    eligible_seeds = [seed for seed in all_seeds if seed not in excluded]
    if not eligible_seeds:
        raise RuntimeError(
            "No evaluation seeds remain after excluding checkpoint-selection seeds."
        )
    n_total = min(episodes, len(eligible_seeds))
    if episodes > len(eligible_seeds):
        cprint(
            f"Requested {episodes} episodes, only {len(eligible_seeds)} disjoint "
            f"held-out seeds remain; evaluating all {len(eligible_seeds)}.",
            "yellow",
        )
    eval_seeds = eligible_seeds[:n_total]

    cprint(
        f"Evaluating on {n_total} seeds (eval_seed={eval_seed}, first seed={eval_seeds[0]}) ...",
        "cyan",
    )
    return eval_seeds


def _run_one_inference_setting(
    agent,
    env_runner,
    eval_seeds: list[int],
    inference_steps: int,
    video_save_dir: Path | None,
    *,
    result_save_dir: Path,
    ckpt_tag_or_path: str,
    ckpt_path: Path,
    eval_seed: int,
    selection_seeds_excluded: list[int],
    heldout_from_selection: bool,
    use_ema: bool,
    eval_config: str,
    global_step: int,
) -> dict:
    """Run eval at a single inference step count; save per-value results.

    Saves ``_result.txt`` + ``result_details.json`` into *result_save_dir*.

    Returns a dict with keys: ``success_rate`` (micro), ``macro_success_rate``,
    ``avg_steps``, ``n_success``, ``n_total``, ``per_seed_details``.
    """
    for name in ("_result.txt", "result_details.json"):
        if (result_save_dir / name).exists():
            raise FileExistsError(
                f"Evaluation result already exists: {result_save_dir / name}"
            )
    n_seeds = len(eval_seeds)
    env_runner.eval_seeds = eval_seeds
    result = env_runner.run(
        agent,
        inference_steps=inference_steps,
        eval_episodes=n_seeds,
        video_save_dir=video_save_dir,
    )

    per_seed_details: list[dict] = collect_episode_details(result)
    stats = compute_eval_stats(result)
    # Micro denominator = actual completed (task, seed) episode units; for a
    # single task this equals n_seeds, so the single-task path is unchanged.
    n_success = stats["n_success"]
    n_total = stats["n_valid_episodes"]
    success_rate = stats["micro_success_rate"]
    macro_success_rate = (
        stats["macro_success_rate"]
        if stats["macro_success_rate"] is not None
        else success_rate
    )

    task_done_steps = [
        d["steps"]
        for d in per_seed_details
        if d.get("success") and d.get("steps") is not None
    ]
    avg_steps = (
        float(sum(task_done_steps) / len(task_done_steps)) if task_done_steps else None
    )

    result_save_dir.mkdir(parents=True, exist_ok=True)
    _write_result(result_save_dir / "_result.txt", f"{success_rate}\n")
    _write_result(
        result_save_dir / "result_details.json",
        json.dumps(
            {
                "ckpt_tag": ckpt_tag_or_path,
                "ckpt_path": artifact_reference(ckpt_path, ckpt_path.parent.parent),
                "global_step": global_step,
                "eval_config": eval_config,
                "per_task": stats["per_task"],
                "wilson_95": stats["wilson_95"],
                "success_rate": success_rate,
                "macro_success_rate": macro_success_rate,
                "n_success": n_success,
                "n_total": n_total,
                "n_tasks": stats["n_tasks"],
                "avg_steps": avg_steps,
                "eval_seed": eval_seed,
                "evaluation_seeds": eval_seeds,
                "selection_seeds_excluded": selection_seeds_excluded,
                "heldout_from_selection": heldout_from_selection,
                "use_ema": use_ema,
                "inference_steps": inference_steps,
                "per_seed_details": per_seed_details,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    return {
        "success_rate": success_rate,
        "macro_success_rate": macro_success_rate,
        "avg_steps": avg_steps,
        "n_success": n_success,
        "n_total": n_total,
        "per_seed_details": per_seed_details,
        "evaluation_seeds": eval_seeds,
        "selection_seeds_excluded": selection_seeds_excluded,
        "heldout_from_selection": heldout_from_selection,
        "use_ema": use_ema,
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
    inference_steps: int | None = None,
    use_ema: bool | None = None,
    resolved_best: tuple[dict, Path] | None = None,
    video_save_dir: Path | None = None,
    result_save_dir: Path | None = None,
    dotlist_overrides: list[str] | tuple[str, ...] = (),
) -> tuple[float, float | None, int, int]:
    """Evaluate a checkpoint and return success rate.

    Parameters
    ----------
    cfg : pre-loaded OmegaConf config with ``_exp_dir`` injected.
    ckpt_tag_or_path : ``"best"``, ``"latest"``, ``"20pct"``, or a path.
        ``"best"`` requires the selection record written by ``select_best_ckpt.py``.
    episodes : number of seeds to evaluate (default: 100).
    inference_steps : DDIM/Euler inference steps; None uses best/config defaults.
    use_ema : None uses best/config defaults; missing requested EMA is an error.
    resolved_best : optional (record, concrete path) already resolved by the caller.
        Preserves best held-out semantics even with a concrete ckpt_tag_or_path.

    Returns
    -------
    (success_rate, avg_steps, n_success, n_total)
    """
    cfg, use_ema, steps, resolved_best = _resolve_final_eval_request(
        cfg, exp_dir, ckpt_tag_or_path, list(dotlist_overrides), cli_use_ema=use_ema,
        cli_inference_steps=inference_steps, resolved_best=resolved_best,
    )
    if len(steps) != 1:
        raise ValueError("Single evaluation requires one NFE; use evaluate_checkpoint_sweep")
    inference_steps = steps[0]
    best_info = resolved_best[0] if resolved_best is not None else None
    selection_seeds = _selection_seeds(best_info) if best_info is not None else []
    result_save_dir = _prepare_result_dir(exp_dir, result_save_dir)
    agent, env_runner, ckpt_path, ckpt_label, eval_seed, eval_seeds = _setup_eval(
        cfg,
        exp_dir,
        str(resolved_best[1]) if resolved_best is not None else ckpt_tag_or_path,
        use_ema,
        video_save_dir=video_save_dir,
        best_info=best_info,
        episodes=episodes,
        selection_seeds=selection_seeds,
    )
    if best_info is not None and "selection_summary" in best_info and agent._checkpoint_global_step != best_info["global_step"]:
        raise ValueError("Best record global_step disagrees with actual checkpoint state")
    snapshot_ref = save_eval_snapshot(
        result_save_dir, cfg, env_runner, checkpoint=artifact_reference(ckpt_path, exp_dir),
        global_step=agent._checkpoint_global_step, use_ema=use_ema,
        inference_steps_list=[inference_steps], episodes=episodes,
        effective_episodes=len(eval_seeds), shuffle_seed=eval_seed,
        policy_seed_mode="episode_seed", task_seeds=mapped_task_seeds(env_runner, eval_seeds),
        selection_seeds_excluded=selection_seeds, heldout_from_selection=best_info is not None,
        **selection_provenance(best_info),
    )

    info = _run_one_inference_setting(
        agent,
        env_runner,
        eval_seeds,
        inference_steps,
        video_save_dir=video_save_dir,
        result_save_dir=result_save_dir,
        ckpt_tag_or_path=ckpt_tag_or_path,
        ckpt_path=ckpt_path,
        eval_seed=eval_seed,
        selection_seeds_excluded=selection_seeds,
        heldout_from_selection=best_info is not None,
        use_ema=use_ema,
        eval_config=snapshot_ref, global_step=agent._checkpoint_global_step,
    )

    # ── Report ─────────────────────────────────────────────────────────
    avg_str = f"{info['avg_steps']:.1f}" if info["avg_steps"] is not None else "N/A"
    cprint(f"\n{'=' * 50}", "cyan")
    cprint(f"  Checkpoint   : {ckpt_label}", "cyan")
    cprint(f"  Episodes     : {info['n_total']}", "cyan")
    cprint(f"  Inference steps: {inference_steps}", "cyan")
    cprint(
        f"  Success rate : {info['n_success']}/{info['n_total']} = {info['success_rate']:.1%}",
        "green",
    )
    if (
        info["macro_success_rate"] is not None
        and abs(info["macro_success_rate"] - info["success_rate"]) > 1e-9
    ):
        cprint(f"  Macro SR     : {info['macro_success_rate']:.1%}", "cyan")
    cprint(f"  Avg steps    : {avg_str}", "cyan")
    cprint(f"{'=' * 50}\n", "cyan")

    cprint(f"  Results saved to: {result_save_dir}/_result.txt", "cyan")

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
    inference_steps_list: list[int],
    use_ema: bool | None = None,
    resolved_best: tuple[dict, Path] | None = None,
    video_save_dir: Path | None = None,
    result_save_dir: Path | None = None,
    dotlist_overrides: list[str] | tuple[str, ...] = (),
) -> list[dict]:
    """Evaluate a checkpoint at multiple inference step counts.

    The checkpoint is **loaded once** and reused across all inference step counts.
    The **same evaluation seeds** are used for every value so the comparison
    is apples-to-apples.

    Results are saved into ``inference_steps<N>/`` subdirectories under
    *result_save_dir* (or ``exp_dir/eval_dexsim/<run-id>/``), plus an
    aggregate ``eval_summary.json``.
    """
    validate_inference_steps(inference_steps_list)
    if len(set(inference_steps_list)) != len(inference_steps_list):
        raise ValueError(
            "Sweep inference steps must be distinct to preserve per-value results"
        )
    cfg, use_ema, inference_steps_list, resolved_best = _resolve_final_eval_request(
        cfg, exp_dir, ckpt_tag_or_path, list(dotlist_overrides), cli_use_ema=use_ema,
        cli_inference_steps_list=inference_steps_list, resolved_best=resolved_best,
    )
    best_info = resolved_best[0] if resolved_best is not None else None
    selection_seeds = _selection_seeds(best_info) if best_info is not None else []
    result_save_dir = _prepare_result_dir(exp_dir, result_save_dir)

    # ── 1. Setup ONCE ──────────────────────────────────────────────────
    agent, env_runner, ckpt_path, ckpt_label, eval_seed, eval_seeds = _setup_eval(
        cfg,
        exp_dir,
        str(resolved_best[1]) if resolved_best is not None else ckpt_tag_or_path,
        use_ema,
        video_save_dir=video_save_dir,
        best_info=best_info,
        episodes=episodes,
        selection_seeds=selection_seeds,
    )
    if best_info is not None and "selection_summary" in best_info and agent._checkpoint_global_step != best_info["global_step"]:
        raise ValueError("Best record global_step disagrees with actual checkpoint state")
    snapshot_ref = save_eval_snapshot(
        result_save_dir, cfg, env_runner, checkpoint=artifact_reference(ckpt_path, exp_dir),
        global_step=agent._checkpoint_global_step, use_ema=use_ema,
        inference_steps_list=inference_steps_list, episodes=episodes,
        effective_episodes=len(eval_seeds), shuffle_seed=eval_seed,
        policy_seed_mode="episode_seed", task_seeds=mapped_task_seeds(env_runner, eval_seeds),
        selection_seeds_excluded=selection_seeds, heldout_from_selection=best_info is not None,
        **selection_provenance(best_info),
    )

    # ── 2. Sweep over inference steps ─────────────────────────────────
    sweep_results: list[dict] = []

    for inference_steps in inference_steps_list:
        cprint(f"\n--- inference_steps={inference_steps} ---", "cyan", attrs=["bold"])

        result_sub_dir = result_save_dir / f"inference_steps{inference_steps}"
        video_sub_dir = (
            video_save_dir / f"inference_steps{inference_steps}"
            if video_save_dir is not None
            else None
        )
        info = _run_one_inference_setting(
            agent,
            env_runner,
            eval_seeds,
            inference_steps,
            video_save_dir=video_sub_dir,
            result_save_dir=result_sub_dir,
            ckpt_tag_or_path=ckpt_tag_or_path,
            ckpt_path=ckpt_path,
            eval_seed=eval_seed,
            selection_seeds_excluded=selection_seeds,
            heldout_from_selection=best_info is not None,
            use_ema=use_ema,
            eval_config=snapshot_ref, global_step=agent._checkpoint_global_step,
        )

        avg_str = f"{info['avg_steps']:.1f}" if info["avg_steps"] is not None else "N/A"
        cprint(
            f"  Success: {info['n_success']}/{info['n_total']} = {info['success_rate']:.1%}  "
            f"Avg steps: {avg_str}",
            "green" if info["success_rate"] >= 0.5 else "red",
        )

        sweep_results.append({"inference_steps": inference_steps, **info})

    # ── 4. Aggregate summary ───────────────────────────────────────────
    _save_sweep_summary(result_save_dir, sweep_results, ckpt_label)

    return sweep_results


def _save_sweep_summary(
    save_dir: Path, sweep_results: list[dict], ckpt_label: str
) -> None:
    """Save ``eval_summary.json`` and print a comparison table."""
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": ckpt_label,
        "results": {
            f"inference_steps{r['inference_steps']}": {
                "success_rate": r["success_rate"],
                "avg_steps": r["avg_steps"],
                "n_success": r["n_success"],
                "n_total": r["n_total"],
            }
            for r in sweep_results
        },
    }
    _write_result(
        save_dir / "eval_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )

    # ── Terminal comparison table ──────────────────────────────────────
    cprint(f"\n{'=' * 60}", "cyan", attrs=["bold"])
    cprint("  Inference Steps Sweep Summary", "cyan", attrs=["bold"])
    cprint(f"  Checkpoint: {ckpt_label}", "cyan")
    cprint(f"  {'Inference Steps':<16} {'Success Rate':<18} {'Avg Steps':<12}", "cyan")
    cprint("  " + "-" * 46, "cyan")
    for r in sweep_results:
        inference_steps = r["inference_steps"]
        sr = f"{r['n_success']}/{r['n_total']} ({r['success_rate']:.1%})"
        avg = f"{r['avg_steps']:.1f}" if r["avg_steps"] is not None else "N/A"
        cprint(
            f"  {inference_steps:<16} {sr:<18} {avg:<12}",
            "green" if r["success_rate"] >= 0.5 else "red",
        )
    cprint(f"{'=' * 60}\n", "cyan", attrs=["bold"])
    cprint(f"  Summary saved to: {save_dir}/eval_summary.json", "cyan")


def _present_config_value(cfg, paths: list[str]):
    """Return the first explicitly present value among dotted config paths."""
    for path in paths:
        node = cfg
        for part in path.split("."):
            if not hasattr(node, "__contains__") or part not in node:
                break
            node = node[part]
        else:
            return True, node
    return False, None


def _resolve_final_eval_request(
    cfg,
    exp_dir: Path,
    ckpt_tag_or_path: str,
    dotlist_overrides: list[str],
    *,
    cli_use_ema: bool | None = None,
    cli_inference_steps: int | None = None,
    cli_inference_steps_list: list[int] | None = None,
    resolved_best: tuple[dict, Path] | None = None,
):
    """Resolve explicit call/CLI arguments > dotlist > pinned record > config.

    Return (effective config, EMA choice, NFE list, resolved best or None).
    Callers pass the resolved pair onward without rereading the best alias.
    """
    override_cfg = parse_eval_overrides(dotlist_overrides)
    merged_cfg = OmegaConf.merge(cfg, override_cfg)

    config_use_ema = _get_eval_param(cfg, "use_ema", "offline", default=True)
    configured_steps = _get_eval_param(
        cfg, "inference_steps_list", "offline", default=None
    )
    if configured_steps is not None:
        inference_steps_list = list(configured_steps)
    else:
        inference_steps_list = [
            _get_eval_param(cfg, "inference_steps", "offline", default=10)
        ]
    use_ema = config_use_ema

    if resolved_best is None and ckpt_tag_or_path == "best":
        resolved_best = resolve_best_checkpoint(exp_dir)
    if resolved_best is not None:
        best_info = resolved_best[0]
        inference = best_info.get("inference", {})
        if not isinstance(inference, dict):
            raise ValueError("Best inference settings must be an object")
        use_ema = inference.get("use_ema", use_ema)
        if "inference_steps" in inference:
            inference_steps_list = [inference["inference_steps"]]

    present, value = _present_config_value(
        override_cfg, ["eval.offline.use_ema", "eval.use_ema"]
    )
    if present:
        use_ema = value

    list_present, list_value = _present_config_value(
        override_cfg,
        ["eval.offline.inference_steps_list", "eval.inference_steps_list"],
    )
    step_present, step_value = _present_config_value(
        override_cfg, ["eval.offline.inference_steps", "eval.inference_steps"]
    )
    if list_present and list_value is not None:
        inference_steps_list = list(list_value)
    elif step_present:
        inference_steps_list = [step_value]

    if cli_use_ema is not None:
        use_ema = cli_use_ema
    if cli_inference_steps_list is not None:
        inference_steps_list = list(cli_inference_steps_list)
    if cli_inference_steps is not None:
        inference_steps_list = [cli_inference_steps]
    if not isinstance(use_ema, bool):
        raise ValueError(f"use_ema must resolve to boolean, got {use_ema!r}")
    if "env_runner" not in merged_cfg:
        raise ValueError("Evaluation config is missing env_runner")
    validate_inference_steps(inference_steps_list)
    from dexmani_policy.evaluation.protocol import bind_seed_manifest
    merged_cfg = bind_seed_manifest(merged_cfg, resolved_best[0] if resolved_best else None,
                                    overrides=override_cfg)
    return (
        merged_cfg,
        use_ema,
        inference_steps_list,
        resolved_best,
    )


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
        help="Checkpoint: best (best_ckpt.json), latest, 20pct..100pct (default: best).",
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
        "--no-videos",
        action="store_true",
        default=False,
        help="Disable video recording (videos are saved by default).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Evaluation/environment dot-list overrides; agent.* is forbidden (saved-config-owned).",
    )
    parser.add_argument("--selection-record", default=None)
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

    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        cprint(f"Error: config.yaml not found: {cfg_path}", "red")
        sys.exit(1)

    ckpt_tag_or_path = args.ckpt_path if args.ckpt_path else args.ckpt_tag
    cfg, use_ema, inference_steps_list, resolved_best = _resolve_final_eval_request(
        OmegaConf.load(cfg_path),
        exp_dir,
        ckpt_tag_or_path,
        args.overrides,
        resolved_best=resolve_best_checkpoint(exp_dir, args.selection_record) if args.selection_record else None,
        cli_use_ema=args.use_ema,
        cli_inference_steps=args.inference_steps,
    )
    if args.selection_record and ckpt_tag_or_path != "best":
        requested_path, _ = resolve_checkpoint_path(exp_dir, ckpt_tag_or_path)
        if requested_path != resolved_best[1]:
            raise ValueError("Checkpoint override differs from selection handoff")
    if args.selection_record and (use_ema != resolved_best[0]["inference"]["use_ema"] or
                                  inference_steps_list != [resolved_best[0]["inference"]["inference_steps"]]):
        raise ValueError("Selection handoff cannot override raw/EMA or NFE")
    cfg._exp_dir = str(exp_dir)

    episodes = (
        args.episodes
        if args.episodes is not None
        else _get_eval_param(cfg, "episodes", "offline", default=100)
    )
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")

    do_sweep = len(inference_steps_list) > 1

    # Every invocation owns a new run directory, with or without videos.
    video_enabled = _get_eval_param(cfg, "enabled", "video", default=True)
    record_video = video_enabled and not args.no_videos
    result_save_dir = _prepare_result_dir(exp_dir, None)
    video_save_dir = result_save_dir if record_video else None
    cprint(f"Results directory: {result_save_dir}", "cyan")

    if video_save_dir is not None:
        video_save_dir.mkdir(parents=True, exist_ok=True)
        cprint(f"\n📹 Video output: {video_save_dir}", "cyan")

    try:
        if do_sweep:
            cprint(
                f"\n🔁 Inference steps sweep: {inference_steps_list}",
                "cyan",
                attrs=["bold"],
            )
            evaluate_checkpoint_sweep(
                exp_dir,
                cfg,
                ckpt_tag_or_path=ckpt_tag_or_path,
                resolved_best=resolved_best,
                episodes=episodes,
                inference_steps_list=inference_steps_list,
                use_ema=use_ema,
                video_save_dir=video_save_dir,
                result_save_dir=result_save_dir,
            )
        else:
            evaluate_checkpoint_robotwin(
                exp_dir,
                cfg,
                ckpt_tag_or_path=ckpt_tag_or_path,
                resolved_best=resolved_best,
                episodes=episodes,
                inference_steps=inference_steps_list[0],
                use_ema=use_ema,
                video_save_dir=video_save_dir,
                result_save_dir=result_save_dir,
            )
    except EvalEpisodeError as e:
        cprint(f"Fatal eval error (category={e.category}, seed={e.seed}): {e}", "red")
        sys.exit(1)
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as e:
        cprint(f"Eval failed: {type(e).__name__}: {e}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
