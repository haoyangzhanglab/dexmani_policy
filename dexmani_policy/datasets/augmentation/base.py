import numpy as np


class Aug:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, x):
        if np.random.random() > self.prob:
            return x
        return self._augment(x)

    def _augment(self, x):
        raise NotImplementedError
