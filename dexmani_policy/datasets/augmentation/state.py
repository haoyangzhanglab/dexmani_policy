import numpy as np
from .base import Aug


class StateNoiseAug(Aug):
    """Small Gaussian noise on proprioceptive state — standard RL/IL practice."""

    def __init__(self, noise_std=0.005, enabled=True, prob=1.0):
        super().__init__(enabled=enabled, prob=prob)
        self.noise_std = noise_std

    def _augment(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape).astype(x.dtype)
