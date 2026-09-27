"""Runtime inference-step selection shared by action decoders."""

from dexmani_policy.utils.validation import positive_int


def resolve_inference_steps(default_steps: int, inference_steps: int | None) -> int:
    if inference_steps is None:
        return positive_int(default_steps, "num_inference_steps")
    return positive_int(inference_steps, "inference_steps")
