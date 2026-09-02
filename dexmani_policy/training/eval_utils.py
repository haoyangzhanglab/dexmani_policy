"""Shared evaluation utilities used by the eval entry points.

Extracted from ``select_best_ckpt.py`` and ``eval_best_ckpt.py`` to eliminate
duplicated config validation, component construction, and checkpoint loading
logic across the evaluation pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from termcolor import cprint

from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import (
    normalize_action_key,
    validate_action_key_consistency,
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


def validate_eval_config(cfg) -> None:
    """Validate the minimal config invariants required for evaluation.

    Replaces the duplicated copies across eval entry points,
    ``select_best_ckpt.py``, ``eval_best_ckpt.py``, and the trainer.
    """
    normalize_action_key(cfg)
    validate_action_key_consistency(cfg)

    if not (cfg.n_obs_steps >= 1 and cfg.n_action_steps >= 1):
        raise ValueError(f"n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps} must be >= 1")
    if cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps-1+n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) "
            f"exceeds horizon ({cfg.horizon}). The control_action slice "
            f"pred[:, {cfg.n_obs_steps - 1}:{cfg.n_obs_steps - 1 + cfg.n_action_steps}] "
            f"would be out of bounds."
        )


def validate_denoise_steps(denoise_timesteps_list, solver: str | None) -> None:
    """Pre-episode NFE validation, single-sourced with the ActionFlow decoder.

    Guards the CLI ``--denoise-steps`` override so an invalid NFE (e.g. odd
    value with the midpoint solver) fails at startup instead of being swallowed
    by the per-episode exception layer after ``env.reset``.  ``solver`` is
    ``None`` for non-ActionFlow decoders (which take no even-NFE constraint).
    """
    if not denoise_timesteps_list:
        raise ValueError("denoise_timesteps_list must be non-empty")
    for nfe in denoise_timesteps_list:
        if isinstance(nfe, bool) or not isinstance(nfe, int):
            raise ValueError(f"denoise step must be an integer, got {nfe!r}")
        if nfe <= 0:
            raise ValueError(f"denoise step must be positive, got {nfe}")
        if solver == "midpoint" and nfe % 2 != 0:
            raise ValueError(f"midpoint solver requires an even NFE, got {nfe}")


# ---------------------------------------------------------------------------
# 2. Agent / env_runner construction
# ---------------------------------------------------------------------------


def build_eval_components(cfg, device: torch.device) -> tuple[Any, Any, CheckpointStore]:
    """Instantiate agent, env_runner, and checkpoint_store from an OmegaConf config.

    Parameters
    ----------
    cfg : OmegaConf config with ``_exp_dir`` set to the experiment directory.
    device : torch.device (unused by construction but kept for caller convenience).
    """
    agent = hydra.utils.instantiate(cfg.agent)
    agent.action_key = cfg.action_key

    env_runner = hydra.utils.instantiate(cfg.env_runner)

    exp_dir = Path(cfg._exp_dir) if hasattr(cfg, "_exp_dir") else None
    ckpt_dir = exp_dir / "checkpoints" if exp_dir else None
    checkpoint_store = CheckpointStore(ckpt_dir)

    return agent, env_runner, checkpoint_store


# ---------------------------------------------------------------------------
# 3. Checkpoint loading for inference (shared across eval entry points)
# ---------------------------------------------------------------------------


def load_ckpt_for_inference(
    agent,
    checkpoint_store: CheckpointStore,
    ckpt_path: Path,
    use_ema: bool,
) -> None:
    """Load a checkpoint into *agent* for inference, with validation.

    Validates train_params consistency (n_obs_steps, n_action_steps,
    action_dim, horizon, action_key), EMA selection, and normalizer
    integrity.
    """
    checkpoint = checkpoint_store.load(ckpt_path)

    train_params = checkpoint.train_params
    if train_params is not None:
        for key in ("n_obs_steps", "n_action_steps", "action_dim", "horizon", "action_key"):
            expected = train_params.get(key)
            actual = getattr(agent, key, None)
            if expected is not None and actual is not None and expected != actual:
                raise ValueError(
                    f"Checkpoint train_params.{key}={expected} does not match agent.{key}={actual}."
                )

    raw_state = checkpoint.model_state
    if use_ema:
        if checkpoint.ema_model_state is not None:
            raw_state = checkpoint.ema_model_state
        else:
            cprint(
                "WARNING: EMA weights requested but not found in checkpoint. Using model weights.",
                "yellow",
            )

    agent.load_state_dict(
        fix_state_dict(raw_state, is_current_ddp=False),
        strict=True,
    )

    if not agent.normalizer.is_fitted(required_keys=["action"]):
        raise RuntimeError("Normalizer is missing required key 'action' after loading checkpoint.")


# ---------------------------------------------------------------------------
# 4. best_ckpt.json reading (shared across eval entry points)
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
    except json.JSONDecodeError as e:
        raise ValueError(f"best_ckpt.json is malformed JSON: {e}") from e
    except OSError as e:
        raise OSError(f"best_ckpt.json is unreadable: {e}") from e

    # Resolve relative paths (written by select_best_ckpt.py as ckpt_relpath)
    ckpt_path = Path(best_info.get("ckpt_path", ""))
    if not ckpt_path.is_absolute():
        best_info["ckpt_path"] = str(exp_dir / ckpt_path)
    return best_info


# ---------------------------------------------------------------------------
# 5. Best-checkpoint resolution fallback
# ---------------------------------------------------------------------------


def resolve_best_checkpoint(exp_dir: Path, checkpoint_store: CheckpointStore) -> Path:
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
        f"  Run 'bash scripts/eval/select_best_ckpt.sh ...' first to generate best_ckpt.json."
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


# ---------------------------------------------------------------------------
# 7. Eval config field access
# ---------------------------------------------------------------------------


def _get_eval_param(cfg, param: str, section: str | None = None, *, default: Any = None) -> Any:
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
    - ``"best"`` — reads ``best_ckpt.json``, falls back to ``best.pt`` symlink
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
            raise FileNotFoundError(f"No {target_pct}% milestone checkpoint. Available: {available}")
        return match[0].path, match[0].label

    if ckpt_tag_or_path == "best":
        best_info = read_best_ckpt_json(exp_dir)
        if best_info:
            ckpt_path = Path(best_info["ckpt_path"])
            label = f"best -> {best_info['pct']}% (step={best_info['global_step']})"
            cprint(
                f"  Auto-loaded best checkpoint: {best_info['pct']}% "
                f"(success_rate={best_info['success_rate']:.1%}, "
                f"n_episodes={best_info['n_episodes']})",
                "cyan",
            )
        else:
            ckpt_path = checkpoint_store.resolve_path("best")
            label = f"best ({ckpt_path.name})"
            cprint(
                "  ⚠ best_ckpt.json not found — using best.pt symlink. "
                "Run select_best_ckpt.sh first for automatic selection.",
                "yellow",
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
