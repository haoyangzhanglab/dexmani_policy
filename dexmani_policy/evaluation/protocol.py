"""Shared evaluation utilities used by the eval entry points.

Saved experiment config owns Agent construction and model-facing semantics.
Evaluation overrides supply environment and protocol controls for
``select_best_ckpt.py``, ``eval_best_ckpt.py`` and ``record_demo.py``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from dexmani_policy.utils.config import (
    validate_action_key_consistency,
    validate_window_contract,
)
from dexmani_policy.agents.loader import load_experiment_config, resolve_checkpoint
from dexmani_policy.datasets.preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.utils.validation import positive_int
from dexmani_policy.utils.atomic import atomic_path


def resolve_selection_checkpoint(experiment_dir, record_path=None):
    """Resolve a current selection result, snapshot, or explicit writer handoff."""
    from dexmani_policy.agents.loader import (
        _read_record, _record_path, _validate_record_inference,
    )

    root = Path(experiment_dir).resolve()
    info = _read_record(Path(record_path) if record_path else root / "best_ckpt.json")
    if record_path is None and info.get("_format") != "best.v1":
        raise ValueError("best_ckpt.json requires best.v1; rerun checkpoint selection")
    snapshot = info if info.get("_format") == "best.v1" else None
    if snapshot is not None:
        _validate_record_inference(snapshot)
        snapshot_path = _record_path(root, snapshot.get("ckpt_relpath"),
                                     checkpoint=True, immutable=True)
    # Only explicit handoffs may contain a bare reference; ordinary best is best.v1.
    if snapshot is not None or (record_path is not None and set(info) == {"selection_result"}):
        info = _read_record(_record_path(root, info.get("selection_result")))
    if info.get("_format") != "selection.v2":
        raise ValueError("Selection evidence requires selection.v2")
    _validate_record_inference(info)
    resolved = _record_path(root, info.get("ckpt_relpath"), checkpoint=True, immutable=True)
    if snapshot is not None:
        if snapshot_path != resolved or snapshot["inference"] != info["inference"]:
            raise ValueError("Best snapshot/selection checkpoint or inference mismatch")
        for key in ("global_step", "pct", "selection_id"):
            if key in snapshot and snapshot[key] != info.get(key):
                raise ValueError(f"Best snapshot/selection mismatch: {key}")
    stages = info.get("stages")
    candidates = info.get("all_results")
    if (info.get("status") != "success"
            or not isinstance(info.get("selection_id"), str) or not info["selection_id"]
            or not isinstance(info.get("selection"), dict)
            or not isinstance(candidates, list) or not candidates
            or not isinstance(stages, list) or not stages
            or any(not isinstance(stage, dict) or stage.get("status") != "completed" for stage in stages)
            or not any(isinstance(candidate, dict) and all(candidate.get(k) == info.get(k)
                       for k in ("ckpt_relpath", "global_step", "pct")) for candidate in candidates)):
        raise ValueError("Selection result lacks successful completed candidate evidence")
    task_seeds = info["selection"].get("task_seeds")
    if not isinstance(task_seeds, dict) or not task_seeds:
        raise ValueError("Selection requires actual task_seeds evidence")
    validate_task_seeds(task_seeds, task_seeds)
    return info, resolved


def resolve_eval_seed(cfg, cli_seed: int | None = None) -> int:
    """Resolve the eval seed.

    1. CLI override (``--seed``)
    2. ``training.seed + 1024`` — shifts eval away from training seed
    """
    if cli_seed is not None:
        return cli_seed
    training_seed = cfg.training.get("seed", 0) if hasattr(cfg, "training") else 0
    return training_seed + 1024


# Saved model inputs and evaluation overrides


def validate_eval_config(cfg, saved: dict) -> None:
    """Validate saved model windows against the current evaluation environment."""
    validate_window_contract(
        saved["agent"]["horizon"],
        saved["agent"]["n_obs_steps"],
        saved["agent"]["n_action_steps"],
    )
    # Dataset/model fields in current config are not historical model semantics.
    validate_action_key_consistency(
        {
            "action_key": saved["action_key"],
            "env_runner": cfg.env_runner,
        }
    )


def parse_eval_overrides(overrides: list[str]):
    """Only evaluation/inference controls may override checkpoint evaluation."""
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key == "eval.seed_manifest":
            raise ValueError("eval.seed_manifest was removed; use evaluation seed and episode counts")
        if key.split(".")[-1] in {"denoise_steps", "denoise_timesteps_list"}:
            raise ValueError(
                "Use inference_steps or inference_steps_list for evaluation"
            )
        if not (
            key.startswith(("eval.", "env_runner."))
            or key in {"training.device", "training.seed"}
        ):
            raise ValueError(
                f"Evaluation override cannot redefine saved inference config: {key}"
            )
        if any(
            part in key.split(".")
            for part in (
                "n_obs_steps",
                "sensor_modalities",
                "rgb_preprocess_size",
                "rgb_random_crop_size",
                "rgb_keep_uint8",
            )
        ):
            raise ValueError(f"Model inputs come from saved dataset config: {key}")
    return OmegaConf.from_dotlist(overrides)


def validate_inference_steps(inference_steps_list) -> None:
    """Reject an empty list or non-positive/non-integer NFE before rollout."""
    if not inference_steps_list:
        raise ValueError("inference_steps_list must be non-empty")
    for nfe in inference_steps_list:
        positive_int(nfe, "inference_steps")


def add_inference_steps_argument(parser) -> None:
    """Add the shared inference-step override to evaluation CLIs."""
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="DDIM/Euler inference steps (best: saved snapshot/selection; otherwise config).",
    )


# Evaluation environment


def build_eval_runner(cfg):
    """Build the evaluation environment with the saved model input recipe."""
    saved = load_experiment_config(cfg._exp_dir)
    validate_eval_config(cfg, saved)
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    dataset = saved["dataset"]
    children = dict(zip(dataset.get("task_names", ()), dataset.get("datasets", ())))
    for runner in iter_leaf_env_runners(env_runner):
        inputs = children[runner.task_name] if children else dataset
        runner.n_obs_steps = saved["agent"]["n_obs_steps"]
        runner.sensor_modalities = list(inputs["sensor_modalities"])
        runner.rgb_preprocessing = rgb_preprocessing_kwargs(inputs)
    return env_runner


def iter_leaf_env_runners(env_runner):
    runners = getattr(env_runner, "runners", None)
    if isinstance(runners, dict):
        return tuple(runners.values())
    return (env_runner,)


# Checkpoint loading for inference


def load_ckpt_for_inference(
    ckpt_path: Path,
    use_ema: bool,
    *,
    cfg,
):
    """Use the shared loader with the original saved config, never eval overrides."""
    from dexmani_policy.agents.loader import restore_policy_agent

    saved = load_experiment_config(cfg._exp_dir)
    validate_eval_config(cfg, saved)
    return restore_policy_agent(
        saved, ckpt_path, use_ema=use_ema, device=cfg.training.device
    )


# Episode details and statistics


def collect_episode_details(result: dict) -> list[dict]:
    """Extract per-episode details from a runner result dict.

    Handles both single-task results (top-level ``episode_details`` key)
    and multi-task results (``episode_details`` nested under each task in
    the ``per_task`` dict).  For multi-task results each detail is tagged
    with a ``task_name`` field.
    """
    # Single-task: top-level episode_details
    if "episode_details" in result:
        details: list[dict] = result.get("episode_details", [])
        if details:
            return details

    # Multi-task: flatten from per_task
    per_task = result.get("per_task", {})
    if per_task:
        all_details: list[dict] = []
        for task_name, task_result in per_task.items():
            task_details = task_result.get("episode_details", [])
            for d in task_details:
                d = dict(d)  # shallow copy so we can mutate safely
                d.setdefault("task_name", task_name)
                all_details.append(d)
        return all_details

    return []


def compute_eval_stats(result: dict) -> dict:
    """Compute success-rate statistics from a runner result dict.

    Handles single-task (top-level ``episode_details``) and multi-task
    (``per_task`` nested) results uniformly.  The evaluation unit is
    ``(task, seed)``: ``micro`` averages over every such unit, ``macro``
    averages over tasks (None for single-task; callers fall back to micro).

    Returns
    -------
    dict with keys:
        micro_success_rate (float): ``n_success / n_valid_episodes`` (0.0 if none)
        macro_success_rate (float | None): mean per-task SR (None for single-task)
        n_success (int): successes across all (task, seed) units
        n_valid_episodes (int): episode units that completed (micro denominator)
        n_tasks (int): 1 for single-task, ``len(per_task)`` for multi-task
        per_task (dict | None): ``{task_name: {success_rate, n_success, n_valid}}``
            (multi-task only)
    """
    per_seed_details = collect_episode_details(result)
    n_success = sum(1 for d in per_seed_details if d.get("success"))
    n_valid = len(per_seed_details)
    micro = (n_success / n_valid) if n_valid > 0 else 0.0

    per_task = result.get("per_task", {})
    per_task_stats: dict | None = None
    macro: float | None = None
    if per_task:
        task_srs: list[float] = []
        per_task_stats = {}
        for task_name, task_result in per_task.items():
            t_details = task_result.get("episode_details", [])
            t_success = sum(1 for d in t_details if d.get("success"))
            t_valid = len(t_details)
            t_sr = (t_success / t_valid) if t_valid > 0 else None
            per_task_stats[task_name] = {
                "success_rate": t_sr,
                "n_success": t_success,
                "n_valid": t_valid,
                "wilson_95": wilson_interval(t_success, t_valid),
            }
            if t_sr is not None:
                task_srs.append(t_sr)
        if task_srs:
            macro = sum(task_srs) / len(task_srs)

    return {
        "micro_success_rate": micro,
        "macro_success_rate": macro,
        "n_success": n_success,
        "n_valid_episodes": n_valid,
        "wilson_95": wilson_interval(n_success, n_valid),
        "n_tasks": len(per_task) if per_task else 1,
        "per_task": per_task_stats,
    }


# Eval config field access


def _get_eval_param(
    cfg, param: str, section: str | None = None, *, default: Any = None
) -> Any:
    """Resolve section → shared eval → default; only None means unspecified."""
    eval_cfg = cfg.get("eval") or {}
    section_cfg = (eval_cfg.get(section) or {}) if section else {}
    for source in (section_cfg, eval_cfg):
        val = source.get(param)
        if val is not None:
            return val
    return default


# Milestone checkpoint discovery

_MILESTONE_RE = re.compile(
    r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$"
)


@dataclass
class MilestoneCheckpoint:
    """A discovered milestone checkpoint."""

    path: Path
    pct: int  # 20, 40, 60, 80, 100
    global_step: int

    @property
    def label(self) -> str:
        return f"{self.pct}% (step={self.global_step})"


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    """Find regular milestone files inside this experiment, sorted by pct."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Make sure the experiment was run with the step-driven "
            f"training loop (total_train_steps in config)."
        )

    found: list[MilestoneCheckpoint] = []
    for pt_file in sorted(ckpt_dir.glob("epoch=*.pt")):
        m = _MILESTONE_RE.match(pt_file.name)
        if not m:
            continue
        if pt_file.is_symlink():
            raise ValueError(f"Milestone checkpoint must not be a symlink: {pt_file}")
        found.append(
            MilestoneCheckpoint(
                path=resolve_checkpoint(exp_dir, pt_file.name),
                pct=int(m.group("pct")),
                global_step=int(m.group("step")),
            )
        )

    if not found:
        raise FileNotFoundError(
            f"No milestone checkpoints found in {ckpt_dir}.\n"
            f"Expected filenames like: epoch=*-step=*-milestone=20pct.pt\n"
            f"Run training with the step-driven loop first."
        )

    found.sort(key=lambda c: c.pct)
    return found


# Checkpoint path resolution


def resolve_checkpoint_path(
    exp_dir: Path,
    ckpt_tag_or_path: str,
) -> tuple[Path, str]:
    """Resolve a checkpoint tag to an absolute path and human-readable label.

    Supported tags:
    - ``"best"`` — reads the ``best_ckpt.json`` selection record
    - ``"latest"`` — ``checkpoints/latest.pt``
    - ``"20pct".."100pct"`` — matched against milestone checkpoints
    - any other string — treated as a filename inside ``checkpoints/``
      (relative) or an absolute path inside the same directory
    """
    if ckpt_tag_or_path.endswith("pct"):
        milestones = discover_milestone_checkpoints(exp_dir)
        target_pct = int(ckpt_tag_or_path.replace("pct", ""))
        match = [m for m in milestones if m.pct == target_pct]
        if not match:
            available = sorted(m.pct for m in milestones)
            raise FileNotFoundError(
                f"No {target_pct}% milestone checkpoint. Available: {available}"
            )
        return match[0].path, match[0].label

    path = resolve_checkpoint(exp_dir, ckpt_tag_or_path)
    return path, f"{ckpt_tag_or_path} ({path.name})"


def wilson_interval(successes, n):
    """95% binomial Wilson interval; not training-seed or macro uncertainty."""
    import math
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / den
    return [max(0., center-half), min(1., center+half)]


def task_seed_pools(runner):
    """Physical seeds in task execution order; no reference-task indirection."""
    return {leaf.task_name: list(leaf.get_seed_list())
            for leaf in iter_leaf_env_runners(runner)}


def select_eval_seeds(runner, eval_seed, episodes, excluded_seeds=None):
    """Shuffle each task's pool and take equal budgets, excluding selection seeds."""
    positive_int(episodes, "episodes")
    pools = task_seed_pools(runner)
    excluded = {} if excluded_seeds is None else excluded_seeds
    if not isinstance(excluded, dict):
        raise ValueError("Excluded selection seeds must be a task-to-seeds mapping")
    if excluded:
        validate_task_seeds(excluded, pools)
    available = {}
    common_count = min(len(pool) for pool in pools.values())
    for task, pool in pools.items():
        pool = list(dict.fromkeys(pool))[:common_count]
        random.Random(eval_seed).shuffle(pool)
        available[task] = [s for s in pool if s not in excluded.get(task, [])]
    count = min(episodes, *(len(pool) for pool in available.values()))
    if count == 0:
        raise RuntimeError("No evaluation seeds remain after excluding checkpoint-selection seeds")
    return {task: pool[:count] for task, pool in available.items()}


def plan_size(plan):
    """Equal per-task budgets retain the existing macro/micro comparison."""
    sizes = {len(seeds) for seeds in plan.values()}
    if len(sizes) != 1:
        raise ValueError("Task seed plans require equal per-task episode counts")
    return sizes.pop()


def validate_task_seeds(plan, tasks):
    if not isinstance(plan, dict) or set(plan) != set(tasks):
        raise ValueError("Task seed plan tasks do not match runner")
    for task, seeds in plan.items():
        if (not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(f"{task}: invalid or duplicate seeds")
    if plan_size(plan) == 0:
        raise ValueError("Task seed plan is empty")


def run_eval_plan(runner, agent, plan, *, inference_steps, video_save_dir=None):
    tasks = list(runner.runners) if hasattr(runner, "runners") else [runner.task_name]
    validate_task_seeds(plan, tasks)
    kwargs = dict(inference_steps=inference_steps, eval_episodes=plan_size(plan),
                  video_save_dir=video_save_dir)
    if hasattr(runner, "runners"):
        result = runner.run(agent, task_seeds=plan, **kwargs)
    else:
        runner.eval_seeds = list(plan[runner.task_name])
        result = runner.run(agent, **kwargs)
    validate_episode_result(result, plan, default_task=getattr(runner, "task_name", None))
    return result


def validate_episode_result(result, plan, *, default_task=None):
    """Validate completed episodes before statistics; normalize supported bools for JSON."""
    from collections import Counter
    import numpy as np

    if not isinstance(result, dict) or result.get("failed_tasks") or any(
        key in result for key in ("error", "error_category")
    ):
        raise RuntimeError("Evaluation contains failed tasks or an exceptional result")
    if "per_task" in result:
        per_task = result["per_task"]
        if not isinstance(per_task, dict) or set(per_task) != set(plan):
            raise RuntimeError("Evaluation per_task results do not match requested tasks")
        for task, child in per_task.items():
            validate_episode_result(child, {task: plan[task]}, default_task=task)
    raw = result.get("episode_details", [])
    if not isinstance(raw, list) or any(not isinstance(d, dict) for d in raw):
        raise RuntimeError("Evaluation episode_details must be a list of objects")
    details = collect_episode_details(result)
    for detail in details:
        if (type(detail.get("seed")) is not int
                or not isinstance(detail.get("success"), (bool, np.bool_))
                or "steps" not in detail
                or (detail["steps"] is not None and
                    (type(detail["steps"]) is not int or detail["steps"] < 0))
                or not isinstance(detail.get("task_name", default_task), str)
                or "error" in detail or "error_category" in detail):
            raise RuntimeError(f"Invalid or exceptional evaluation episode: {detail}")
        detail["success"] = bool(detail["success"])
    expected = Counter((task, seed) for task, seeds in plan.items() for seed in seeds)
    actual = Counter((d.get("task_name", default_task), d["seed"]) for d in details)
    if actual != expected:
        raise RuntimeError("Evaluation episodes do not match the requested task/seed plan")


def validate_heldout(runner, best_info, plan):
    if best_info is None:
        return
    selection = best_info["selection"]
    excluded = selection.get("task_seeds")
    try:
        validate_task_seeds(excluded, plan)
    except ValueError as exc:
        raise ValueError("Selection lacks actual task/seed evidence; rerun selection") from exc
    for task, requested in plan.items():
        if set(requested) & set(excluded[task]):
            raise ValueError(f"Held-out task/seed overlap for {task}; rerun selection")


def selection_provenance(best_info):
    """Keep selection evidence separate from the actual inference overrides."""
    if best_info is None:
        return {}
    return {key: best_info.get(key) for key in
            ("selection_id", "selection_summary", "selection")}


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(exp_dir).resolve()))
    except ValueError:
        return str(path)


def code_version(directory):
    import subprocess
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(directory), *args],
                stderr=subprocess.DEVNULL, text=True).strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}


def save_eval_snapshot(directory, cfg, runner, **request):
    """Persist resolved inputs and the actual evaluation request."""
    import importlib.util
    import importlib.metadata
    import sys
    exp_dir = Path(cfg._exp_dir)
    sim = {"commit": "unknown", "dirty": "unknown", "version": "unknown"}
    try:
        spec = importlib.util.find_spec("dexmani_sim")
        if spec is not None and spec.origin:
            sim.update(code_version(Path(spec.origin).parent))
        sim["version"] = importlib.metadata.version("dexmani_sim")
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        pass
    inputs = {leaf.task_name: {
        "n_obs_steps": leaf.n_obs_steps,
        "sensor_modalities": list(leaf.sensor_modalities),
        "rgb_preprocessing": leaf.rgb_preprocessing,
        "record_video": leaf.record_video,
        "render_mode": getattr(leaf, "render_mode", None),
        "viewer_resolution": getattr(leaf, "viewer_resolution", None),
        "env_video_fps": getattr(leaf, "env_video_fps", None),
        "env_kwargs": (leaf._expand_env_kwargs(leaf.env_kwargs)
                       if hasattr(leaf, "_expand_env_kwargs") else getattr(leaf, "env_kwargs", {})),
    } for leaf in iter_leaf_env_runners(runner)}
    snapshot = {
        "model_config_source": artifact_reference(exp_dir / "config.yaml", exp_dir),
        "effective_config": OmegaConf.to_container(cfg, resolve=True),
        "runner_inputs": inputs,
        "request": request,
        "argv": list(sys.argv),
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": sim,
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with atomic_path(directory / "eval_config.yaml", overwrite=False) as temporary:
        temporary.write_text(OmegaConf.to_yaml(OmegaConf.create(snapshot), resolve=True))
    return artifact_reference(directory / "eval_config.yaml", exp_dir)


def atomic_json(path, record, *, overwrite=True):
    import json
    with atomic_path(path, overwrite=overwrite) as temporary:
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
