"""Shared evaluation utilities used by the eval entry points.

Extracted from ``select_best_ckpt.py`` and ``eval_best_ckpt.py`` to eliminate
duplicated config validation, component construction, and checkpoint loading
logic across the evaluation pipeline.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    build_agent_contract,
    parse_normalization_contract,
    validate_resume_contract,
)
from dexmani_policy.common.config import validate_action_key_consistency, validate_window_contract
from dexmani_policy.common.normalizer import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    validate_normalizer_state,
)
from dexmani_policy.common.pytorch_util import fix_state_dict


def resolve_eval_seed(cfg, cli_seed: int | None = None) -> int:
    """Resolve the eval seed.

    1. CLI override (``--seed``)
    2. ``training.seed + 1024`` — shifts eval away from training seed
    """
    if cli_seed is not None:
        return cli_seed
    training_seed = cfg.training.get("seed", 0) if hasattr(cfg, "training") else 0
    return training_seed + 1024


# ---------------------------------------------------------------------------
# 1. Config validation (was quadruplicated across 4 files)
# ---------------------------------------------------------------------------


def validate_eval_config(cfg, agent_contract: dict) -> None:
    """Validate saved model windows against the current evaluation environment."""
    validate_window_contract(
        agent_contract.get("horizon"),
        agent_contract.get("n_obs_steps"),
        agent_contract.get("n_action_steps"),
    )
    # Dataset/model fields in current config are not historical model semantics.
    validate_action_key_consistency({
        "action_key": agent_contract.get("action_key"),
        "env_runner": cfg.env_runner,
    })


def parse_eval_overrides(overrides: list[str]):
    """Only evaluation/inference controls may override checkpoint evaluation."""
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key == "agent" or key.startswith(("agent.", "agent[")):
            raise ValueError(
                "agent.* overrides are forbidden for checkpoint evaluation; "
                "the checkpoint owns constructor semantics. Use eval.* or "
                "explicit EMA/NFE options for inference ablations."
            )
    return OmegaConf.from_dotlist(overrides)


def validate_denoise_steps(denoise_timesteps_list) -> None:
    """Pre-episode NFE validation for the CLI ``--denoise-steps`` override.

    Fails at startup on an invalid NFE (non-integer or non-positive) instead of
    being swallowed by the per-episode exception layer after ``env.reset``.
    """
    if not denoise_timesteps_list:
        raise ValueError("denoise_timesteps_list must be non-empty")
    for nfe in denoise_timesteps_list:
        if isinstance(nfe, bool) or not isinstance(nfe, int):
            raise ValueError(f"denoise step must be an integer, got {nfe!r}")
        if nfe <= 0:
            raise ValueError(f"denoise step must be positive, got {nfe}")


# ---------------------------------------------------------------------------
# 2. Agent / env_runner construction
# ---------------------------------------------------------------------------


def build_eval_components(cfg) -> tuple[Any, CheckpointStore]:
    """Build the current evaluation environment and experiment checkpoint store."""
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    checkpoint_store = CheckpointStore(Path(cfg._exp_dir) / "checkpoints")
    return env_runner, checkpoint_store


def iter_leaf_env_runners(env_runner):
    runners = getattr(env_runner, "runners", None)
    if isinstance(runners, dict):
        return tuple(runners.values())
    return (env_runner,)


# ---------------------------------------------------------------------------
# 3. Checkpoint loading for inference (shared across eval entry points)
# ---------------------------------------------------------------------------


def load_ckpt_for_inference(
    checkpoint_store: CheckpointStore,
    ckpt_path: Path,
    use_ema: bool,
    *,
    cfg,
):
    """Construct and strictly restore an Agent using only checkpoint semantics.

    Current config supplies the evaluation environment, never the constructor
    or normalization recipe. Each candidate in checkpoint selection is restored
    independently, including behavior-only options absent from its state dict.
    """
    checkpoint = checkpoint_store.load(ckpt_path)
    agent_contract = checkpoint.resume_contract.get("agent")
    agent_config = checkpoint.resume_contract.get("agent_config")
    if type(agent_contract) is not dict or not agent_contract:
        raise RuntimeError("Checkpoint resume_contract.agent must be a non-empty plain dict")
    if type(agent_config) is not dict or not agent_config:
        raise RuntimeError("Checkpoint resume_contract.agent_config must be a non-empty plain dict")
    if not isinstance(agent_config.get("_target_"), str):
        raise RuntimeError("Checkpoint agent_config must declare an Agent _target_")
    validate_eval_config(cfg, agent_contract)
    normalization_spec = parse_normalization_contract(agent_contract.get("normalization"))

    raw_state = checkpoint.model_state
    if use_ema:
        if checkpoint.ema_model_state is None:
            raise RuntimeError(
                f"EMA weights were requested, but checkpoint {ckpt_path} has no EMA state. "
                "Use use_ema=False (or --no-ema) to explicitly load raw model weights."
            )
        raw_state = checkpoint.ema_model_state

    constructor = copy.deepcopy(agent_config)
    is_dqrise = constructor["_target_"] == "dexmani_policy.agents.core.dqrise.DQRISEAgent"
    if is_dqrise:
        # Persistent buffers, not the training-time external NPZ, own this state.
        constructor["codebook_path"] = None
    agent = hydra.utils.instantiate(OmegaConf.create(constructor))
    agent.action_key = agent_contract["action_key"]
    numeric_fields = set(agent.obs_encoder.consumed_observation_fields) - NON_NUMERIC_OBSERVATION_FIELDS
    agent.normalization_spec = parse_normalization_contract(
        agent_contract["normalization"], observation_fields=numeric_fields
    )
    validate_resume_contract(
        {"agent": agent_contract}, {"agent": build_agent_contract(agent)}
    )
    agent.load_state_dict(fix_state_dict(raw_state, is_current_ddp=False), strict=True)
    validate_normalizer_state(agent.normalizer, normalization_spec)
    if is_dqrise:
        manager = agent.codebook_manager
        if (
            not manager.is_loaded
            or not manager.has_hand_normalizer
            or manager.hand_dim != agent.hand_dim
            or manager.num_groups != agent.codebook_num_groups
            or manager.codebook_size != agent.codebook_size
        ):
            raise RuntimeError("DQ-RISE checkpoint has incomplete or inconsistent persistent codebook state")
        agent._validate_codebook_normalizer()
    validate_resume_contract(
        {"agent": agent_contract}, {"agent": build_agent_contract(agent)}
    )
    return agent


# ---------------------------------------------------------------------------
# 4. best_ckpt.json reading (shared across eval entry points)
# ---------------------------------------------------------------------------


def read_best_ckpt_json(exp_dir: Path) -> dict:
    """Read and validate the strict selection record in *exp_dir*."""
    best_json = exp_dir / "best_ckpt.json"
    if not best_json.is_file():
        raise FileNotFoundError(
            f"Selection record not found: {best_json}. Run select_best_ckpt.py "
            "before evaluating 'best', or explicitly select latest/milestone/path."
        )
    try:
        best_info = json.loads(best_json.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"best_ckpt.json is malformed JSON: {e}") from e
    except OSError as e:
        raise OSError(f"best_ckpt.json is unreadable: {e}") from e

    if not isinstance(best_info, dict):
        raise ValueError("best_ckpt.json must contain a JSON object")

    expected_top = {
        "ckpt_relpath",
        "pct",
        "global_step",
        "success_rate",
        "avg_steps",
        "n_episodes",
        "inference",
        "selection",
    }
    if set(best_info) != expected_top:
        raise ValueError("best_ckpt.json does not match current schema")

    inference = best_info["inference"]
    if not isinstance(inference, dict):
        raise ValueError("best_ckpt.json inference must be an object")
    if set(inference) != {"use_ema", "denoise_steps", "policy_seed_mode"}:
        raise ValueError("best_ckpt.json does not match current schema")
    if not isinstance(inference["use_ema"], bool):
        raise ValueError("best_ckpt.json inference.use_ema must be boolean")
    denoise_steps = inference["denoise_steps"]
    if (
        isinstance(denoise_steps, bool)
        or not isinstance(denoise_steps, int)
        or denoise_steps <= 0
    ):
        raise ValueError(
            "best_ckpt.json inference.denoise_steps must be a positive integer"
        )
    if inference["policy_seed_mode"] != "episode_seed":
        raise ValueError(
            "best_ckpt.json inference.policy_seed_mode must be 'episode_seed'"
        )

    selection = best_info["selection"]
    if not isinstance(selection, dict):
        raise ValueError("best_ckpt.json selection must be an object")
    if set(selection) != {
        "shuffle_seed",
        "seeds",
        "initial_episodes",
        "tie_break_used",
    }:
        raise ValueError("best_ckpt.json does not match current schema")
    seeds = selection["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError(
            "best_ckpt.json selection.seeds must be a non-empty list of unique integers"
        )

    ckpt_relpath = best_info["ckpt_relpath"]
    if not isinstance(ckpt_relpath, str) or not ckpt_relpath:
        raise ValueError("best_ckpt.json ckpt_relpath must be a non-empty string")
    relative_path = Path(ckpt_relpath)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "best_ckpt.json ckpt_relpath must be relative to the experiment directory"
        )

    ckpt_path = (exp_dir / relative_path).resolve()
    resolved_exp_dir = exp_dir.resolve()
    if not ckpt_path.is_relative_to(resolved_exp_dir):
        raise ValueError(
            "best_ckpt.json ckpt_relpath resolves outside the experiment directory"
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint recorded by best_ckpt.json does not exist: {ckpt_path}"
        )
    return best_info


# ---------------------------------------------------------------------------
# 5. Episode detail extraction for single-task and multi-task result dicts
# ---------------------------------------------------------------------------


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
        "n_tasks": len(per_task) if per_task else 1,
        "per_task": per_task_stats,
    }


# ---------------------------------------------------------------------------
# 7. Eval config field access
# ---------------------------------------------------------------------------


def _get_eval_param(
    cfg, param: str, section: str | None = None, *, default: Any = None
) -> Any:
    """Read an eval config parameter with three-level fallback.

    Resolution order:
    1. ``cfg.eval.<section>.<param>``   — per-section override
    2. ``cfg.eval.<param>``             — shared top-level
    3. *default*
    """
    eval_cfg = cfg.eval if hasattr(cfg, "eval") else {}
    _has_get = hasattr(eval_cfg, "get")

    # 1. Per-section override
    if section:
        section_cfg = eval_cfg.get(section, {}) if _has_get else {}
        val = section_cfg.get(param) if section_cfg else None
        if val is not None:
            return val

    # 2. Shared top-level
    val = eval_cfg.get(param) if _has_get else None
    if val is not None:
        return val

    # 3. Hardcoded default
    return default


# ---------------------------------------------------------------------------
# 8. Milestone checkpoint discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 9. Unified checkpoint path resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(
    exp_dir: Path,
    ckpt_tag_or_path: str,
    checkpoint_store: CheckpointStore,
) -> tuple[Path, str]:
    """Resolve a checkpoint tag to an absolute path and human-readable label.

    Supported tags:
    - ``"best"`` — reads the strict ``best_ckpt.json`` selection record
    - ``"latest"`` — ``checkpoint_store.resolve_path("latest")``
    - ``"20pct".."100pct"`` — matched against milestone checkpoints
    - any other string — treated as a filename inside ``checkpoints/``
      (relative) or an absolute path
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

    if ckpt_tag_or_path == "best":
        best_info = read_best_ckpt_json(exp_dir)
        ckpt_path = (exp_dir / best_info["ckpt_relpath"]).resolve()
        label = f"best -> {best_info['pct']}% (step={best_info['global_step']})"
        cprint(
            f"  Auto-loaded best checkpoint: {best_info['pct']}% "
            f"(success_rate={best_info['success_rate']:.1%}, "
            f"n_episodes={best_info['n_episodes']})",
            "cyan",
        )
        return ckpt_path, label

    if ckpt_tag_or_path == "latest":
        ckpt_path = checkpoint_store.resolve_path("latest")
        return ckpt_path, f"latest ({ckpt_path.name})"

    # Treat as a path
    ckpt_path = Path(ckpt_tag_or_path)
    if not ckpt_path.is_absolute():
        ckpt_path = exp_dir / "checkpoints" / ckpt_path
    return ckpt_path, str(ckpt_path)
