"""Saved experiment selection and shared strict Agent restore.

Heavy imports stay inside restore; parent-side inspection does not load Torch.
"""


def load_experiment_config(experiment_dir):
    """Read the resolved snapshot without composing current repository YAML."""
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )
    if not isinstance(cfg, dict):
        raise ValueError("Expected a saved experiment config mapping")
    return cfg


def resolve_best_checkpoint(experiment_dir, record_path=None):
    """Read one best record and return it with its validated concrete weight path."""
    import json
    from pathlib import Path

    root = Path(experiment_dir).resolve()
    info = json.loads((Path(record_path) if record_path else root / "best_ckpt.json").read_text())
    if not isinstance(info, dict):
        raise ValueError("best_ckpt.json must contain an object")
    relative = info.get("ckpt_relpath")
    if not isinstance(relative, str) or not relative:
        raise ValueError("best_ckpt.json requires ckpt_relpath")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Best checkpoint must be relative to the experiment")
    if record_path is not None and (path.name == "latest.pt" or (root / path).is_symlink()):
        raise ValueError("Selection handoff must name an immutable milestone checkpoint")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root / "checkpoints") or not resolved.is_file():
        raise ValueError("Best checkpoint must be a file inside experiment/checkpoints")
    if "selection_summary" in info or "selection_id" in info:
        if not isinstance(info.get("selection_id"), str) or not info["selection_id"]:
            raise ValueError("New best record requires a nonempty selection_id")
        relative_summary = info.get("selection_summary")
        if not isinstance(relative_summary, str) or Path(relative_summary).is_absolute() or ".." in Path(relative_summary).parts:
            raise ValueError("Selection summary must be relative to experiment")
        summary_path = (root / relative_summary).resolve(strict=True)
        if not summary_path.is_relative_to(root):
            raise ValueError("Selection summary must remain inside experiment")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "success" or summary.get("selection_id") != info.get("selection_id"):
            raise ValueError("Best record must reference its successful selection")
        selected = summary.get("best_checkpoint", {})
        for key in ("ckpt_relpath", "global_step", "pct"):
            if key not in selected or selected[key] != info.get(key):
                raise ValueError(f"Best/selection mismatch: {key}")
        if "inference" in summary and summary["inference"] != info.get("inference"):
            raise ValueError("Best/selection inference mismatch")
        if record_path is not None and "inference" not in summary:
            raise ValueError("Selection handoff requires immutable inference settings")
        if summary.get("selection") != info.get("selection"):
            raise ValueError("Best/selection seed evidence mismatch")
    else:
        if record_path is not None:
            raise ValueError("Selection handoff requires a successful selection identity")
        import warnings
        warnings.warn("Historical best has no selection identity/summary; provenance is unverified", stacklevel=2)
    return info, resolved


def read_best_ckpt_json(experiment_dir):
    """Read selection essentials; retain the historical dictionary interface."""
    return resolve_best_checkpoint(experiment_dir)[0]


def resolve_checkpoint(experiment_dir, selector="best"):
    from pathlib import Path

    root = Path(experiment_dir).resolve()
    if selector == "best":
        return resolve_best_checkpoint(root)[1]
    elif selector == "latest":
        path = root / "checkpoints/latest.pt"
    else:
        path = Path(selector)
        if not path.is_absolute():
            path = root / "checkpoints" / path
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root / "checkpoints"):
        raise ValueError(
            "Checkpoint must be inside the selected experiment/checkpoints"
        )
    return path


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    """Restore inference from config and tensors, independent of resume semantics."""
    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.agents.normalization import (
        resolve_normalization_spec,
        validate_normalizer_state,
    )
    from dexmani_policy.training.checkpoint import CheckpointStore, fix_state_dict
    from dexmani_policy.utils.config import validate_window_contract

    cfg = OmegaConf.create(saved_config)
    validate_window_contract(
        cfg.agent.horizon, cfg.agent.n_obs_steps, cfg.agent.n_action_steps
    )
    if cfg.action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    checkpoint = CheckpointStore(Path(checkpoint_path).parent).load(checkpoint_path)
    state = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    if state is None:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    agent = hydra.utils.instantiate(cfg.agent)
    agent._checkpoint_global_step = checkpoint.global_step
    agent.action_key = cfg.action_key
    agent.set_normalization_spec(resolve_normalization_spec(cfg))
    agent.load_state_dict(fix_state_dict(state, is_current_ddp=False), strict=True)
    validate_normalizer_state(agent.normalizer, agent.normalization_spec)
    agent.to(device)
    agent.eval()
    return agent
