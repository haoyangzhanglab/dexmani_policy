"""Scalar validation and model observation declarations."""


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_observation_fields(agent, declared, *, require_declaration=False):
    """Compare Dataset inputs with the model's existing consumption declaration."""
    consumed = getattr(agent, "consumed_observation_fields", None)
    if consumed is None:
        consumed = getattr(getattr(agent, "obs_encoder", None), "consumed_observation_fields", None)
    if consumed is None:
        if require_declaration:
            raise ValueError("Real deployment requires consumed_observation_fields on Agent or encoder")
        return
    missing = set(consumed) - set(declared)
    unused = set(declared) - set(consumed)
    if missing or unused:
        raise ValueError(f"Model observation mismatch: missing={sorted(missing)}, unconsumed={sorted(unused)}")
