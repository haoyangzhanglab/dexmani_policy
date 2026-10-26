"""Deterministic epoch ordering with a consumed-batch cursor."""

import torch
from torch.utils.data import Sampler


class ResumableSampler(Sampler[int]):
    def __init__(self, dataset, *, batch_size, shuffle=False, seed=0):
        self.num_samples = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.next_micro_step = 0

    def set_epoch(self, epoch, next_micro_step=0):
        if type(next_micro_step) is not int or next_micro_step < 0:
            raise ValueError("next_micro_step must be an int >= 0")
        self.epoch = epoch
        self.next_micro_step = next_micro_step

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        return iter(indices[self.next_micro_step * self.batch_size :])

    def __len__(self):
        return max(0, self.num_samples - self.next_micro_step * self.batch_size)
