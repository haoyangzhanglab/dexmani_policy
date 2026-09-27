"""Saved experiment selection, preprocessing and shared strict inference restore.

Heavy imports stay inside restore; parent-side inspection does not load Torch.
"""

from collections.abc import Mapping


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_inference_steps(default_steps: int, inference_steps: int | None) -> int:
    if inference_steps is None:
        return positive_int(default_steps, "num_inference_steps")
    return positive_int(inference_steps, "inference_steps")


def normalize_inference_settings(settings: Mapping) -> dict:
    """Normalize one config/selection ingress mapping without mutating it."""
    result = dict(settings)
    for legacy, canonical in (
        ("denoise_steps", "inference_steps"),
        ("denoise_timesteps_list", "inference_steps_list"),
    ):
        if legacy in result:
            value = result.pop(legacy)
            if canonical in result and (
                type(result[canonical]) is not type(value) or result[canonical] != value
            ):
                raise ValueError(f"Conflicting {canonical} and legacy {legacy}")
            result[canonical] = value
    return result


def load_experiment_config(experiment_dir):
    """Read the resolved snapshot without composing current repository YAML."""
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )
    if not isinstance(cfg, dict) or "real_runtime" not in cfg:
        raise ValueError(
            "Expected a saved experiment config with real_runtime (possibly null)"
        )
    return cfg


def read_best_ckpt_json(experiment_dir):
    """Read selection essentials; extra reporting fields are not a schema."""
    import json
    from pathlib import Path

    root = Path(experiment_dir).resolve()
    info = json.loads((root / "best_ckpt.json").read_text())
    if not isinstance(info, dict):
        raise ValueError("best_ckpt.json must contain an object")
    relative = info.get("ckpt_relpath")
    if not isinstance(relative, str) or not relative:
        raise ValueError("best_ckpt.json requires ckpt_relpath")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Best checkpoint must be relative to the experiment")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root / "checkpoints") or not resolved.is_file():
        raise ValueError("Best checkpoint must be a file inside experiment/checkpoints")
    return info


def resolve_checkpoint(experiment_dir, selector="best"):
    from pathlib import Path

    root = Path(experiment_dir).resolve()
    if selector == "best":
        path = root / read_best_ckpt_json(root)["ckpt_relpath"]
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


def rgb_preprocessing_kwargs(dataset_config):
    """The exact deterministic BaseDataset validation recipe."""
    return {
        "resize_hw": dataset_config.get("rgb_preprocess_size"),
        "center_crop_hw": dataset_config.get("rgb_random_crop_size"),
        "keep_uint8": bool(dataset_config.get("rgb_keep_uint8", False))
        and dataset_config.get("rgb_color_aug") is None,
    }


def restore_policy_agent(saved_config, checkpoint_path, *, use_ema, device):
    """Restore inference from config and tensors, independent of resume semantics."""
    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.common.checkpoint_io import CheckpointStore
    from dexmani_policy.common.config import validate_window_contract
    from dexmani_policy.common.normalizer import validate_normalizer_state
    from dexmani_policy.common.pytorch_util import fix_state_dict
    from dexmani_policy.training.build_utils import resolve_normalization_spec

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
    agent.action_key = cfg.action_key
    agent.normalization_spec = resolve_normalization_spec(cfg)
    agent.load_state_dict(fix_state_dict(state, is_current_ddp=False), strict=True)
    validate_normalizer_state(agent.normalizer, agent.normalization_spec)
    agent.to(device)
    agent.eval()
    return agent
