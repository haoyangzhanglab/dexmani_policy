import numpy as np
import torch


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


class PCAug:
    def __init__(self, color_std=0.05):
        self.color_std = color_std

    def __call__(self, pc):
        # (T, N, C) float32 numpy, C >= 6; jitter color channels [..., 3:6], XYZ unchanged
        pc = pc.copy()
        noise = np.random.normal(0, self.color_std, pc[..., 3:6].shape).astype(pc.dtype)
        pc[..., 3:6] = np.clip(pc[..., 3:6] + noise, 0.0, 1.0)
        return pc
