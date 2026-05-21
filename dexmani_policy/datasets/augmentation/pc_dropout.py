import numpy as np
from .base import Aug


class PCDropout(Aug):
    """Randomly drop points — simulates occlusion in dexterous manipulation.

    Dropped points are replaced with zeros. Same dropout mask across all T frames.
    """

    def __init__(self, dropout_ratio=0.1, prob=1.0):
        super().__init__(prob=prob)
        self.dropout_ratio = dropout_ratio

    def _augment(self, x):
        k = int(x.shape[1] * self.dropout_ratio)
        if k == 0:
            return x
        drop_idx = np.random.choice(x.shape[1], size=k, replace=False)
        pc = x.copy()
        pc[:, drop_idx] = 0.0
        return pc
