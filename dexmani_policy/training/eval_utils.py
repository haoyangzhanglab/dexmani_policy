"""Shared evaluation utilities used by all three eval entry points.

Extracted from ``select_best_ckpt.py`` and ``sim_evaluator.py`` to eliminate
~200 lines of duplicated config validation, component construction, and
checkpoint loading logic across the evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import (
    normalize_action_key,
    validate_action_key_consistency,
)
from dexmani_policy.common.pytorch_util import fix_state_dict
from dexmani_policy.training.build_utils import inject_faas_into_agent


# ---------------------------------------------------------------------------
# 1. Config validation (was quadruplicated across 4 files)
# ---------------------------------------------------------------------------


def validate_eval_config(cfg) -> None:
    """Validate the minimal config invariants required for evaluation.

    Replaces the 4 independent copies in ``eval_sim.py``,
    ``select_best_ckpt.py``, ``eval_best_ckpt.py``, and the trainer.
    """
    normalize_action_key(cfg)
    validate_action_key_consistency(cfg)

    if not (cfg.n_obs_steps >= 1 and cfg.n_action_steps >= 1):
        raise ValueError(
            f"n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps} "
            f"must be >= 1"
        )
    if cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps-1+n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) "
            f"exceeds horizon ({cfg.horizon}). The control_action slice "
            f"pred[:, {cfg.n_obs_steps - 1}:{cfg.n_obs_steps - 1 + cfg.n_action_steps}] "
            f"would be out of bounds."
        )


# ---------------------------------------------------------------------------
# 2. Agent / env_runner construction
# ---------------------------------------------------------------------------


def build_eval_components(
    cfg, device: torch.device
) -> tuple[Any, Any, CheckpointStore]:
    """Instantiate agent, env_runner, and checkpoint_store from an OmegaConf config.

    Parameters
    ----------
    cfg : OmegaConf config with ``_exp_dir`` set to the experiment directory.
    device : torch.device (unused by construction but kept for caller convenience).
    """
    agent = hydra.utils.instantiate(cfg.agent)
    agent.action_key = cfg.action_key
    inject_faas_into_agent(agent, cfg)

    env_runner = hydra.utils.instantiate(cfg.env_runner)

    exp_dir = Path(cfg._exp_dir) if hasattr(cfg, "_exp_dir") else None
    ckpt_dir = exp_dir / "checkpoints" if exp_dir else None
    checkpoint_store = CheckpointStore(ckpt_dir)

    return agent, env_runner, checkpoint_store


# ---------------------------------------------------------------------------
# 3. Checkpoint loading for inference (was duplicated between
#    SimEvaluator._load_for_inference and select_best_ckpt.py)
# ---------------------------------------------------------------------------


def load_ckpt_for_inference(
    agent,
    checkpoint_store: CheckpointStore,
    ckpt_path: Path,
    use_ema: bool,
) -> None:
    """Load a checkpoint into *agent* for inference, with validation.

    Validates train_params consistency (n_obs_steps, n_action_steps,
    action_dim, horizon, action_key), FAAS compatibility, EMA selection,
    and normalizer integrity.
    """
    checkpoint = checkpoint_store.load(ckpt_path)

    train_params = checkpoint.train_params
    if train_params is not None:
        for key in ("n_obs_steps", "n_action_steps", "action_dim", "horizon", "action_key"):
            expected = train_params.get(key)
            actual = getattr(agent, key, None)
            if expected is not None and actual is not None and expected != actual:
                raise ValueError(
                    f"Checkpoint train_params.{key}={expected} does not match "
                    f"agent.{key}={actual}."
                )

        ckpt_use_faas = train_params.get("use_faas", False)
        agent_use_faas = getattr(agent, "use_faas", False)
        if ckpt_use_faas != agent_use_faas:
            raise ValueError(
                f"Checkpoint use_faas={ckpt_use_faas} but agent "
                f"use_faas={agent_use_faas}. Use a matching config."
            )

    raw_state = (
        checkpoint.ema_model_state
        if (use_ema and checkpoint.ema_model_state is not None)
        else checkpoint.model_state
    )
    if use_ema and checkpoint.ema_model_state is None and raw_state is checkpoint.model_state:
        cprint(
            "WARNING: EMA weights requested but not found in checkpoint. "
            "Using model weights.",
            "yellow",
        )

    agent.load_state_dict(
        fix_state_dict(raw_state, is_current_ddp=False),
        strict=True,
    )

    if not agent.normalizer.is_fitted(required_keys=["action"]):
        raise RuntimeError(
            "Normalizer is missing required key 'action' after loading checkpoint."
        )

# ---------------------------------------------------------------------------
# 4. best_ckpt.json reading (was duplicated between eval_sim.py and
#    eval_best_ckpt.py)
# ---------------------------------------------------------------------------


def read_best_ckpt_json(exp_dir: Path) -> dict | None:
    """Read ``best_ckpt.json`` from *exp_dir*, resolving relative paths.

    Returns the parsed dict (with ``ckpt_path`` resolved to absolute), or
    ``None`` if the file does not exist or is malformed.
    """
    best_json = exp_dir / "best_ckpt.json"
    if not best_json.is_file():
        return None
    try:
        best_info = json.loads(best_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Resolve relative paths (written by select_best_ckpt.py as ckpt_relpath)
    ckpt_path = Path(best_info.get("ckpt_path", ""))
    if not ckpt_path.is_absolute():
        best_info["ckpt_path"] = str(exp_dir / ckpt_path)
    return best_info


# ---------------------------------------------------------------------------
# 5. Best-checkpoint resolution fallback (fixes eval_sim.py C5)
# ---------------------------------------------------------------------------


def resolve_best_checkpoint(
    exp_dir: Path, checkpoint_store: CheckpointStore
) -> Path:
    """Resolve the 'best' checkpoint with a proper fallback chain.

    1. ``best_ckpt.json`` (from ``select_best_ckpt.py``).
    2. ``best.pt`` symlink.
    3. ``latest.pt`` symlink.
    4. Raise ``FileNotFoundError``.

    The old behaviour of falling through to a filename-score sort on
    milestone checkpoints (all scoring ``-inf``) is removed — that path
    picked an arbitrary checkpoint and was never correct.
    """
    # 1. best_ckpt.json
    best_info = read_best_ckpt_json(exp_dir)
    if best_info:
        path = Path(best_info["ckpt_path"])
        if path.is_file():
            return path

    # 2. best.pt symlink
    best_symlink = exp_dir / "checkpoints" / "best.pt"
    if best_symlink.is_symlink() or best_symlink.is_file():
        return checkpoint_store.resolve_path(str(best_symlink))

    # 3. latest.pt symlink
    try:
        return checkpoint_store.resolve_path("latest")
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"Cannot resolve 'best' checkpoint in {exp_dir}.\n"
        f"  Tried: best_ckpt.json, best.pt symlink, latest.pt symlink.\n"
        f"  Run 'bash scripts/select_best_ckpt.sh ...' first to generate best_ckpt.json."
    )


# ---------------------------------------------------------------------------
# 6. Episode detail extraction (handles both single-task and multi-task
#    result dicts — fixes C2 / C4)
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
