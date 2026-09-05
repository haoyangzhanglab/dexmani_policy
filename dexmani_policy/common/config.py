"""Hydra config resolvers and validation shared by train / eval entry points."""

import warnings

from omegaconf import DictConfig, OmegaConf


def register_resolvers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OmegaConf.register_new_resolver(
            "eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True
        )
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def validate_action_key_consistency(cfg) -> None:
    """Validate that ``action_key`` matches ``env_runner.env_kwargs.control_mode``.

    Raises ValueError if the configuration is contradictory (e.g. joint-space
    ``action_key`` with ``control_mode='ee'`` in the env runner).  This
    prevents silent misconfiguration from CLI overrides.
    """
    action_key = cfg.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be explicitly set to 'action' or 'action_ee'")

    env_kwargs = cfg.get("env_runner", {}).get("env_kwargs", {})
    if isinstance(env_kwargs, (dict, DictConfig)):
        actual_control = env_kwargs.get("control_mode", "joint")
    else:
        actual_control = "joint"
    expected_control = "ee" if action_key == "action_ee" else "joint"
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
