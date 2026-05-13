import numpy as np


class Aug:
    def __init__(self, enabled=True, prob=1.0):
        self.enabled = enabled
        self.prob = prob

    def __call__(self, x):
        if not self.enabled or np.random.random() > self.prob:
            return x
        return self._augment(x)

    def _augment(self, x):
        raise NotImplementedError
