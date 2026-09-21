"""Hydra config resolvers and validation shared by train / eval entry points."""

import math
import warnings
from numbers import Integral, Real

from omegaconf import DictConfig, OmegaConf


def register_resolvers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OmegaConf.register_new_resolver(
            "eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True
        )
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def validate_window_contract(horizon, n_obs_steps, n_action_steps) -> None:
    """Shared training and checkpoint-evaluation action-window grammar."""
    for name, value in (
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
        ("n_action_steps", n_action_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if n_obs_steps - 1 + n_action_steps > horizon:
        raise ValueError(
            f"n_obs_steps - 1 + n_action_steps ({n_obs_steps - 1 + n_action_steps}) "
            f"exceeds horizon ({horizon})"
        )


def validate_val_ratio(val_ratio) -> None:
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, Real)
        or not math.isfinite(val_ratio)
        or not 0 <= val_ratio < 1
    ):
        raise ValueError(f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio!r}")


def validate_max_train_episodes(value) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError(f"max_train_episodes must be None or a positive integer, got {value!r}")


def validate_dataset_splits(dataset) -> None:
    """Validate split options without constructing datasets or reading Zarr."""
    validate_val_ratio(dataset.get("val_ratio", 0.0))
    validate_max_train_episodes(dataset.get("max_train_episodes"))
    for child in dataset.get("datasets", []):
        validate_dataset_splits(child)


def validate_action_key_consistency(cfg) -> None:
    """Validate that ``action_key`` matches ``env_runner.env_kwargs.control_mode``.

    Raises ValueError if the configuration is contradictory (e.g. joint-space
    ``action_key`` with ``control_mode='ee'`` in the env runner).  This
    prevents silent misconfiguration from CLI overrides.
    """
    action_key = cfg.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be explicitly set to 'action' or 'action_ee'")

    expected_control = "ee" if action_key == "action_ee" else "joint"
    env_runner = cfg.get("env_runner", {})
    task_configs = env_runner.get("task_configs")
    if task_configs is not None:
        for task_index, task_config in enumerate(task_configs):
            env_kwargs = task_config.get("env_kwargs", {})
            actual_control = env_kwargs.get("control_mode", "joint")
            if actual_control != expected_control:
                task_name = task_config.get("task_name", f"index {task_index}")
                raise ValueError(
                    f"action_key='{action_key}' requires control_mode='{expected_control}', "
                    f"but env_runner.task_configs task '{task_name}' has "
                    f"control_mode='{actual_control}'."
                )
    else:
        env_kwargs = env_runner.get("env_kwargs", {})
        if isinstance(env_kwargs, (dict, DictConfig)):
            actual_control = env_kwargs.get("control_mode", "joint")
        else:
            actual_control = "joint"
        if actual_control != expected_control:
            raise ValueError(
                f"action_key='{action_key}' requires control_mode='{expected_control}', "
                f"but env_runner.env_kwargs.control_mode='{actual_control}'. "
                f"Check CLI overrides for env_runner.env_kwargs.control_mode."
            )

    # Guard against CLI overrides that desync dataset.action_key from the
    # top-level action_key (e.g. --dataset.action_key=action while
    # --action_key=action_ee, which would cause a dimension mismatch).
    ds_action_key = cfg.get("dataset", {}).get("action_key")
    if ds_action_key is not None and ds_action_key != action_key:
        raise ValueError(
            f"dataset.action_key='{ds_action_key}' != action_key='{action_key}'. "
            f"They must be the same. Check CLI overrides for dataset.action_key "
            f"and/or action_key."
        )
