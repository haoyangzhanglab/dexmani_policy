"""Shared configuration validation functions used by train and eval entry points."""


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
