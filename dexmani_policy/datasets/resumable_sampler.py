"""Deterministic distributed ordering with a consumed-batch cursor."""

from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, dataset, *, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size
        self.next_micro_step = 0

    def set_epoch(self, epoch, next_micro_step=0):
        super().set_epoch(epoch)
        if type(next_micro_step) is not int or next_micro_step < 0:
            raise ValueError("next_micro_step must be an int >= 0")
        self.next_micro_step = next_micro_step

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.next_micro_step * self.batch_size:])

    def __len__(self):
        return max(0, self.num_samples - self.next_micro_step * self.batch_size)

    def full_num_batches(self, drop_last=False):
        if drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size
