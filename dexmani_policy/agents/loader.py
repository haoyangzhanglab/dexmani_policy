"""Saved inference snapshot resolution and shared strict Agent restore.

Heavy imports stay inside restore; parent-side inspection does not load Torch.
Selection evidence is validated by evaluation.protocol.
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


def _read_record(path):
    import json

    info = json.loads(path.read_text())
    if not isinstance(info, dict):
        raise ValueError(f"Checkpoint record must contain an object: {path}")
    return info


def _record_path(root, relative, *, checkpoint=False, immutable=False):
    from pathlib import Path

    if not isinstance(relative, str) or not relative:
        raise ValueError("Record requires a nonempty relative file path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Record path must be relative to experiment")
    if immutable and (path.name == "latest.pt" or (root / path).is_symlink()):
        raise ValueError("Selection must name an immutable milestone checkpoint")
    resolved = (root / path).resolve(strict=True)
    directory = root / "checkpoints" if checkpoint else root
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError(f"Record path must be a file inside {directory}")
    return resolved


def _validate_record_inference(info, *, historical=False):
    inference = info.get("inference", {} if historical else None)
    if not isinstance(inference, dict):
        raise ValueError("Record requires inference settings")
    if (not historical or "use_ema" in inference) and type(inference.get("use_ema")) is not bool:
        raise ValueError("inference.use_ema must be a bool")
    steps = inference.get("inference_steps")
    if (not historical or "inference_steps" in inference) and (type(steps) is not int or steps <= 0):
        raise ValueError("inference.inference_steps must be a positive int (not bool)")
    mode = inference.get("policy_seed_mode")
    if (info.get("_format") == "best.v1" or "policy_seed_mode" in inference) and (not isinstance(mode, str) or not mode):
        raise ValueError("Record requires inference.policy_seed_mode")


def _read_selection_reference(root, info):
    try:
        target = _record_path(root, info.get("selection_result"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Selection reference is missing; use an explicit checkpoint with its EMA/NFE settings"
        ) from exc
    result = _read_record(target)
    if result.get("_format") != "selection.v2":
        raise ValueError("Selection reference requires selection.v2")
    return result


def resolve_best_checkpoint(experiment_dir):
    """Resolve saved inference; only unformatted flat records may omit EMA/NFE."""
    from pathlib import Path

    root = Path(experiment_dir).resolve()
    info = _read_record(root / "best_ckpt.json")
    if "_format" in info and info["_format"] not in ("best.v1", "selection.v2"):
        raise ValueError(f"Unsupported best record format: {info['_format']!r}")
    if "_format" not in info and "selection_result" in info:
        info = _read_selection_reference(root, info)
    historical = "_format" not in info
    if info.get("_format") == "selection.v2" and info.get("status") != "success":
        raise ValueError("Selection result must be successful")
    _validate_record_inference(info, historical=historical)
    if historical:
        info.setdefault("inference", {})
    resolved = _record_path(root, info.get("ckpt_relpath"), checkpoint=True, immutable=not historical)
    return info, resolved


def warn_legacy_inference_defaults(info, *, source, overridden=()):
    """Identify saved-config fallback without filling or rewriting the record."""
    import warnings

    missing = {"use_ema", "inference_steps"} - info["inference"].keys() - set(overridden)
    if missing:
        warnings.warn(
            f"Historical flat best: {', '.join(sorted(missing))} from {source}; "
            "provided inference fields come from the record",
            stacklevel=2,
        )


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
    state, global_step = CheckpointStore(Path(checkpoint_path).parent).load_inference(
        checkpoint_path, use_ema=use_ema
    )
    state = fix_state_dict(state)
    cfg = checkpoint_agent_config(cfg, state)
    agent = hydra.utils.instantiate(cfg.agent)
    agent._checkpoint_global_step = global_step
    agent.action_key = cfg.action_key
    agent.set_normalization_spec(resolve_normalization_spec(cfg))
    agent.load_state_dict(state, strict=True)
    validate_normalizer_state(agent.normalizer, agent.normalization_spec)
    agent.to(device)
    agent.eval()
    return agent


def checkpoint_agent_config(cfg, state):
    """Construct from saved weights without loading external initialization assets."""
    import copy
    import torch
    from omegaconf import DictConfig, open_dict

    cfg = copy.deepcopy(cfg)
    agent = cfg.agent
    with open_dict(agent):
        if agent.get('rgb_backbone_name') is not None:
            rgb = dict(agent.get('rgb_backbone_config') or {})
            if (agent.rgb_backbone_name in {'dino', 'clip', 'siglip'}
                    and 'architecture' in rgb
                    and not isinstance(rgb.get('architecture'), (dict, DictConfig))):
                raise ValueError('Saved backbone architecture must be a mapping')
            rgb['load_pretrained'] = False
            agent.rgb_backbone_config = rgb
        if agent.get('_target_') == 'dexmani_policy.agents.core.multi_task.MultiTaskAgent' and agent.get('task_texts') is not None:
            table = state.get('task_emb_table')
            texts = list(dict.fromkeys(agent.task_texts))
            if not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[0] != len(texts):
                raise ValueError('Closed text checkpoint requires the complete saved embedding table and task list')
            agent.text_embed_dim = table.shape[1]
    return cfg
