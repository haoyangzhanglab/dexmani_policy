"""Inference step validation and legacy input-name normalization.

No model or Torch imports: deployment metadata inspection uses this module too.
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
