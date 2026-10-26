"""Deterministic distributed ordering with a consumed-batch cursor."""

import numpy as np
from torch.utils.data.distributed import DistributedSampler

from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset


def multitask_indices(
    task_lengths, sample_probs, seed, epoch, *, deterministic=False, sampling_strategy="balanced"
):
    """Map logical positions to task samples with deterministic seed/epoch quotas."""
    rng = np.random.default_rng([seed, 0 if deterministic else epoch])
    total = sum(task_lengths)
    offsets = np.cumsum([0] + list(task_lengths))
    if deterministic and sampling_strategy == "proportional":
        indices = np.arange(total, dtype=np.int64)
    else:
        if sampling_strategy == "weighted":
            probs = np.asarray(sample_probs, dtype=np.float64)
            cumulative = np.r_[0.0, np.cumsum(probs / probs.sum()) * total]
            cumulative = np.minimum(cumulative, total)
            # Trailing zero-weight tasks must not receive floating-point residue.
            cumulative[np.flatnonzero(probs)[-1] + 1 :] = total
            counts = np.diff(np.floor(cumulative + rng.random())).astype(np.int64)
        else:
            counts = np.round(np.asarray(sample_probs) * total).astype(int)
            diff = total - counts.sum()
            if diff > 0:
                counts[rng.choice(len(task_lengths), p=sample_probs)] += diff
            while diff < 0:
                valid = np.where(counts > 0)[0]
                counts[valid[rng.choice(len(valid))]] -= 1
                diff += 1
        parts = []
        for task, (length, count) in enumerate(zip(task_lengths, counts)):
            if count == 0:
                continue
            if count <= length:
                local = rng.permutation(length)[:count]
            else:
                chunks = [rng.permutation(length) for _ in range(count // length)]
                if count % length:
                    chunks.append(rng.permutation(length)[: count % length])
                local = np.concatenate(chunks)
            parts.append(local + offsets[task])
        indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


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
        indices = list(super().__iter__())[self.next_micro_step * self.batch_size :]
        if isinstance(self.dataset, MultiTaskDataset):
            ds = self.dataset
            mapping = multitask_indices(
                ds.task_lengths,
                ds.sample_probs,
                ds.seed,
                self.epoch,
                deterministic=ds.deterministic,
                sampling_strategy=ds.sampling_strategy,
            )
            indices = mapping[np.asarray(indices, dtype=np.int64)].tolist()
        return iter(indices)

    def __len__(self):
        return max(0, self.num_samples - self.next_micro_step * self.batch_size)
