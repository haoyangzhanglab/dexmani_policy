"""Data settings shared by hand representation pretraining and policy training."""

import math

import numpy as np


def shuffled_batches(indices, batch_size, rng, *, drop_last=False):
    """Repeat shuffled passes, optionally dropping each pass's final partial batch."""
    count = int(indices) if isinstance(indices, (int, np.integer)) else len(indices)
    if count == 0:
        raise ValueError("Cannot generate shuffled batches from empty indices")
    if drop_last and count < batch_size:
        raise ValueError("drop_last=True requires at least batch_size indices")
    stop = count - count % batch_size if drop_last else count
    while True:
        order = rng.permutation(indices)
        for start in range(0, stop, batch_size):
            yield order[start : start + batch_size]


def validate_kl_schedule(steps, beta, eligible_step):
    if not math.isfinite(beta) or beta <= 0 or steps < eligible_step:
        raise ValueError(
            f"Hand pretraining requires positive beta and steps >= {eligible_step} "
            "to complete KL warmup/ramp"
        )
