import hashlib
import warnings
import multiprocessing as mp
import numpy as np
import torch
from typing import List, Optional
from dexmani_policy.common.normalizer import LinearNormalizer


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List,
        task_names: List[str],
        sampling_strategy: str = 'balanced',
        task_weights: Optional[List[float]] = None,
        normalizer_mode: str = 'shared',
        seed: int = 42,
        deterministic: bool = False,
        task_texts: Optional[List[str]] = None,
        augmentation_cfg=None,
        action_key: str = 'action',
    ):
        assert len(datasets) == len(task_names)
        assert sampling_strategy in ['proportional', 'balanced', 'weighted']
        assert normalizer_mode in ['shared', 'per_task']

        if len(set(task_names)) != len(task_names):
            raise ValueError(
                f"Duplicate task names detected: {task_names}. "
                f"Task names must be unique — duplicates cause silent data misalignment "
                f"in get_validation_dataset() when task_weights are used."
            )

        for i, dataset in enumerate(datasets):
            if len(dataset) == 0:
                raise ValueError(
                    f"Dataset {i} (task '{task_names[i]}') is empty. "
                    f"All datasets must contain at least one sample."
                )

        self.datasets = datasets
        self.task_names = task_names
        self.action_key = action_key

        if augmentation_cfg is not None:
            for dataset in self.datasets:
                if hasattr(dataset, '_build_augmentors'):
                    dataset.augmentation_cfg = augmentation_cfg
                    dataset._build_augmentors()
        self.num_tasks = len(datasets)
        self.normalizer_mode = normalizer_mode
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.task_weights = task_weights
        self.task_texts = task_texts if task_texts is not None else task_names

        self.task_lengths = [len(d) for d in datasets]
        self.cumsum_lengths = np.cumsum([0] + self.task_lengths)
        self.total_length = sum(self.task_lengths)

        self._epoch = 0
        # Manager-backed Value works with both fork and spawn start methods,
        # unlike mp.Value which requires fork for cross-process visibility.
        self._manager = mp.Manager()
        self._epoch_val = self._manager.Value('i', 0)
        self.deterministic = deterministic

        if sampling_strategy == 'proportional':
            self.sample_probs = np.array(self.task_lengths) / self.total_length
        elif sampling_strategy == 'balanced':
            self.sample_probs = np.ones(self.num_tasks) / self.num_tasks
        elif sampling_strategy == 'weighted':
            assert task_weights is not None and len(task_weights) == self.num_tasks
            total = sum(task_weights)
            self.sample_probs = np.array(task_weights) / total

        if sampling_strategy != 'weighted' and task_weights is not None:
            warnings.warn(
                f"task_weights={task_weights} is ignored because "
                f"sampling_strategy='{sampling_strategy}'. "
                f"Set sampling_strategy='weighted' to use task weights."
            )

        if self.deterministic:
            self.fixed_indices = self.generate_fixed_indices()
        else:
            self.epoch_indices = None
            self.current_epoch = -1

        if normalizer_mode == 'shared':
            self.normalizer = self.compute_shared_normalizer()
        else:
            self.normalizers = {name: d.get_normalizer() for name, d in zip(task_names, datasets)}

    def compute_shared_normalizer(self):
        all_joint_states = [d.replay_buffer['joint_state'] for d in self.datasets]
        all_actions = [d.replay_buffer[d.action_key] for d in self.datasets]
        joint_state = np.concatenate(all_joint_states, axis=0)
        action = np.concatenate(all_actions, axis=0)
        return LinearNormalizer.fit_obs_action(joint_state, action, self.action_key, 'limits')

    def __del__(self):
        """Best-effort shutdown of the Manager server process on garbage collection."""
        try:
            if hasattr(self, '_manager'):
                self._manager.shutdown()
        except Exception:
            pass

    def _make_rng(self, *seed_parts: str):
        """Derive a reproducible ``np.random.Generator`` from seed components."""
        raw = "_".join(str(p) for p in seed_parts)
        seed = int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**32)
        return np.random.default_rng(seed)

    def _build_balanced_indices(self, rng):
        """Core index construction shared by fixed and epoch-based generation."""
        target_counts = self.compute_target_counts(rng=rng)
        indices = []
        for task_idx in range(self.num_tasks):
            n_samples = target_counts[task_idx]
            if n_samples == 0:
                continue
            local_indices = self.sample_task_indices(task_idx, n_samples, rng)
            indices.extend((task_idx, int(idx)) for idx in local_indices)
        rng.shuffle(indices)
        return indices

    def generate_fixed_indices(self):
        if self.sampling_strategy == 'proportional':
            indices = []
            for task_idx, task_length in enumerate(self.task_lengths):
                for local_idx in range(task_length):
                    indices.append((task_idx, local_idx))
            rng = self._make_rng(self.seed, "fixed")
            rng.shuffle(indices)
            return indices

        rng = self._make_rng(self.seed, "fixed")
        return self._build_balanced_indices(rng)

    def compute_target_counts(self, rng=None):
        target_counts = np.round(self.sample_probs * self.total_length).astype(int)
        diff = self.total_length - target_counts.sum()
        if diff != 0:
            if rng is None:
                rng = self._make_rng(self.seed, self._epoch, "diff")
            if diff > 0:
                idx = rng.choice(self.num_tasks, p=self.sample_probs)
                target_counts[idx] += diff
            else:
                while diff < 0:
                    valid = np.where(target_counts > 0)[0]
                    if len(valid) == 0:
                        break
                    idx = valid[rng.choice(len(valid))]
                    target_counts[idx] -= 1
                    diff += 1
        return target_counts

    def sample_task_indices(self, task_idx, n_samples, rng):
        task_length = self.task_lengths[task_idx]

        if n_samples <= task_length:
            return rng.permutation(task_length)[:n_samples]

        n_full_passes = n_samples // task_length
        n_remainder = n_samples % task_length

        local_indices = [rng.permutation(task_length) for _ in range(n_full_passes)]
        if n_remainder > 0:
            local_indices.append(rng.permutation(task_length)[:n_remainder])
        return np.concatenate(local_indices)

    def generate_epoch_indices(self):
        rng = self._make_rng(self.seed, self._epoch)
        return self._build_balanced_indices(rng)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # sync epoch from shared memory (visible to persistent_workers)
        self._epoch = self._epoch_val.value

        if self.deterministic:
            task_idx, local_idx = self.fixed_indices[idx]
        else:
            if self.epoch_indices is None or self.current_epoch != self._epoch:
                self.epoch_indices = self.generate_epoch_indices()
                self.current_epoch = self._epoch
            task_idx, local_idx = self.epoch_indices[idx]

        sample = self.datasets[task_idx][local_idx]

        # inject task identity into obs (ManiFlow convention)
        sample['obs']['task_text'] = self.task_texts[task_idx]
        sample['obs']['task_name'] = self.task_names[task_idx]

        return sample

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._epoch_val.value = epoch

    def get_normalizer(self, task_name: Optional[str] = None, **kwargs):
        if self.normalizer_mode == 'shared':
            return self.normalizer
        else:
            if task_name is None:
                raise ValueError("normalizer_mode='per_task' requires task_name argument")
            return self.normalizers[task_name]

    def get_validation_dataset(self):
        val_datasets = [d.get_validation_dataset() for d in self.datasets]
        valid_triples = [
            (d, name, text)
            for d, name, text in zip(val_datasets, self.task_names, self.task_texts)
            if d is not None
        ]
        if not valid_triples:
            return None
        val_ds, val_names, val_texts = zip(*valid_triples)

        if self.task_weights is not None:
            val_weights = [self.task_weights[self.task_names.index(name)] for name in val_names]
        else:
            val_weights = None

        return MultiTaskDataset(
            datasets=list(val_ds),
            task_names=list(val_names),
            sampling_strategy=self.sampling_strategy,
            task_weights=val_weights,
            normalizer_mode=self.normalizer_mode,
            seed=self.seed,
            deterministic=True,
            task_texts=list(val_texts),
            action_key=self.action_key,
        )
