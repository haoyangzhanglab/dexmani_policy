"""Hydra config resolvers and validation shared by train / eval entry points."""

import warnings
from omegaconf import OmegaConf


def register_resolvers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True)
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def normalize_action_key(cfg) -> None:
    """Ensure cfg has ``action_key``, filling from legacy ``action_mode`` if needed.

    Backward compat: historical configs used ``action_mode`` (``eef_hand`` /
    ``joint``).  This function translates it to the current ``action_key``
    convention and emits a ``FutureWarning`` so users can update their configs.
    """
    if hasattr(cfg, 'action_key'):
        return
    if hasattr(cfg, 'action_mode'):
        import warnings
        warnings.warn(
            "Config uses deprecated 'action_mode' field. "
            "Please update to 'action_key: action_ee' (for eef_hand) "
            "or 'action_key: action' (for joint space).",
            FutureWarning,
        )
        cfg.action_key = 'action_ee' if cfg.action_mode == 'eef_hand' else 'action'
    else:
        cfg.action_key = 'action'


def validate_action_key_consistency(cfg) -> None:
    """Validate that ``action_key`` matches ``env_runner.env_kwargs.control_mode``.

    Raises ValueError if the configuration is contradictory (e.g. joint-space
    ``action_key`` with ``control_mode='ee'`` in the env runner).  This
    prevents silent misconfiguration from CLI overrides.
    """
    env_kwargs = cfg.get('env_runner', {}).get('env_kwargs', {})
    if isinstance(env_kwargs, dict):
        actual_control = env_kwargs.get('control_mode', 'joint')
    else:
        actual_control = 'joint'
    expected_control = 'ee' if cfg.action_key == 'action_ee' else 'joint'
    if actual_control != expected_control:
        raise ValueError(
            f"action_key='{cfg.action_key}' requires control_mode='{expected_control}', "
            f"but env_runner.env_kwargs.control_mode='{actual_control}'. "
            f"Check CLI overrides for env_runner.env_kwargs.control_mode."
        )
