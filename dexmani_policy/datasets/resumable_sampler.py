"""Deterministic distributed ordering with a consumed-batch cursor."""

import hashlib

import numpy as np

from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from torch.utils.data.distributed import DistributedSampler


def multitask_indices(task_lengths, sample_probs, seed, epoch, *, deterministic=False,
                      sampling_strategy="balanced"):
    """Task mapping M_e, expressed as stable concatenated-dataset indices.

    Weighted quotas use unbiased random-offset rounding. Other strategies retain
    their rounding and RNG. DistributedSampler supplies logical positions Q_e,r.
    """
    parts = (seed, "fixed") if deterministic else (seed, epoch)
    raw = "_".join(str(p) for p in parts)
    rng = np.random.default_rng(int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**32))
    total = sum(task_lengths)
    offsets = np.cumsum([0] + list(task_lengths))
    if deterministic and sampling_strategy == "proportional":
        indices = list(range(total))
    else:
        if sampling_strategy == 'weighted':
            probs = np.asarray(sample_probs, dtype=np.float64)
            if (probs.shape != (len(task_lengths),) or not np.isfinite(probs).all()
                    or (probs < 0).any() or not np.isclose(probs.sum(), 1)):
                raise ValueError('weighted sampling requires finite nonnegative probabilities summing to one')
            cumulative = np.r_[0., np.cumsum(probs / probs.sum()) * total]
            cumulative = np.minimum(cumulative, total)
            # Fix the endpoint at the last positive task, including trailing
            # zero probabilities, without awarding floating-point residue to them.
            cumulative[np.flatnonzero(probs)[-1] + 1:] = total
            counts = np.diff(np.floor(cumulative + rng.random())).astype(np.int64)
        else:
            # Preserve balanced/proportional quota rounding and RNG consumption.
            counts = np.round(np.asarray(sample_probs) * total).astype(int)
            diff = total - counts.sum()
            if diff > 0:
                counts[rng.choice(len(task_lengths), p=sample_probs)] += diff
            while diff < 0:
                valid = np.where(counts > 0)[0]
                counts[valid[rng.choice(len(valid))]] -= 1
                diff += 1
        indices = []
        for task, (length, count) in enumerate(zip(task_lengths, counts)):
            if count == 0:
                continue
            if count <= length:
                local = rng.permutation(length)[:count]
            else:
                chunks = [rng.permutation(length) for _ in range(count // length)]
                if count % length:
                    chunks.append(rng.permutation(length)[:count % length])
                local = np.concatenate(chunks)
            indices.extend(int(offsets[task] + i) for i in local)
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
        indices = list(super().__iter__())
        if isinstance(self.dataset, MultiTaskDataset):
            ds = self.dataset
            mapping = multitask_indices(ds.task_lengths, ds.sample_probs, ds.seed, self.epoch,
                                        deterministic=ds.deterministic,
                                        sampling_strategy=ds.sampling_strategy)
            indices = [mapping[position] for position in indices]
        return iter(indices[self.next_micro_step * self.batch_size:])

    def __len__(self):
        return max(0, self.num_samples - self.next_micro_step * self.batch_size)
