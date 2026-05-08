import numpy as np
import torch
from typing import List, Optional
from dexmani_policy.common.normalizer import LinearNormalizer


class MultiTaskDataset:
    """
    多任务混合训练数据集，每个样本附带 task_id。

    采样机制：
        - 训练集（_deterministic=False）：有放回的随机采样。每个 __getitem__(idx) 调用
          根据 hash(seed, epoch, idx) 独立随机选择任务和样本，不同 idx 可能映射到相同样本。
        - 验证集（_deterministic=True）：无放回的固定顺序遍历，每个样本恰好出现一次。

    详细采样机制见 docs/数据集设计与处理流程.md 第六节。
    """

    def __init__(
        self,
        datasets: List,
        task_names: List[str],
        sampling_strategy: str = 'balanced',
        task_weights: Optional[List[float]] = None,
        normalizer_mode: str = 'shared',
        seed: int = 42,
        _deterministic: bool = False,
        _original_task_ids: Optional[List[int]] = None,
    ):
        assert len(datasets) == len(task_names)
        assert sampling_strategy in ['proportional', 'balanced', 'weighted']
        assert normalizer_mode in ['shared', 'per_task']

        for i, dataset in enumerate(datasets):
            if len(dataset) == 0:
                raise ValueError(
                    f"Dataset {i} (task '{task_names[i]}') is empty. "
                    f"All datasets must contain at least one sample."
                )

        self.datasets = datasets
        self.task_names = task_names
        self.num_tasks = len(datasets)
        self.normalizer_mode = normalizer_mode
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.task_weights = task_weights

        self.task_lengths = [len(d) for d in datasets]
        self.cumsum_lengths = np.cumsum([0] + self.task_lengths)
        self._total_length = sum(self.task_lengths)

        self._epoch = 0
        self._deterministic = _deterministic
        if self._deterministic:
            self._fixed_indices = self._generate_fixed_indices()

        # 验证集专用：保留原始 task_id 映射（用于多任务验证时保持 task_id 一致性）
        self._original_task_ids = _original_task_ids

        if sampling_strategy == 'proportional':
            self.sample_probs = np.array(self.task_lengths) / self._total_length
        elif sampling_strategy == 'balanced':
            self.sample_probs = np.ones(self.num_tasks) / self.num_tasks
        elif sampling_strategy == 'weighted':
            assert task_weights is not None and len(task_weights) == self.num_tasks
            total = sum(task_weights)
            self.sample_probs = np.array(task_weights) / total

        if normalizer_mode == 'shared':
            self.normalizer = self._compute_shared_normalizer()
        else:
            self.normalizers = {name: d.get_normalizer() for name, d in zip(task_names, datasets)}

    def _compute_shared_normalizer(self):
        all_joint_states = []
        all_actions = []

        for i, dataset in enumerate(self.datasets):
            if not hasattr(dataset, 'replay_buffer'):
                raise ValueError(
                    f"Dataset {i} (task '{self.task_names[i]}') does not have 'replay_buffer' attribute."
                )

            replay_buffer = dataset.replay_buffer

            if 'joint_state' not in replay_buffer:
                available_keys = list(replay_buffer.keys())
                raise ValueError(
                    f"Dataset {i} (task '{self.task_names[i]}') missing 'joint_state' field. "
                    f"Available fields: {available_keys}"
                )

            if 'action' not in replay_buffer:
                available_keys = list(replay_buffer.keys())
                raise ValueError(
                    f"Dataset {i} (task '{self.task_names[i]}') missing 'action' field. "
                    f"Available fields: {available_keys}"
                )

            all_joint_states.append(replay_buffer['joint_state'])
            all_actions.append(replay_buffer['action'])

        data = {
            'joint_state': np.concatenate(all_joint_states, axis=0),
            'action': np.concatenate(all_actions, axis=0),
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode='limits')

        from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer
        normalizer['camera_intrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        normalizer['camera_extrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        return normalizer

    def _generate_fixed_indices(self):
        indices = []
        for task_idx, task_length in enumerate(self.task_lengths):
            for local_idx in range(task_length):
                indices.append((task_idx, local_idx))
        return indices

    def __len__(self):
        return self._total_length

    def __getitem__(self, idx):
        if self._deterministic:
            task_idx, local_idx = self._fixed_indices[idx]
        else:
            item_seed = hash((self.seed, self._epoch, idx)) % (2**32)
            item_rng = np.random.default_rng(item_seed)
            task_idx = item_rng.choice(self.num_tasks, p=self.sample_probs)
            local_idx = item_rng.integers(0, self.task_lengths[task_idx])

        sample = self.datasets[task_idx][local_idx]

        # 使用原始 task_id（验证集）或局部 task_idx（训练集）
        if self._original_task_ids is not None:
            sample['task_id'] = self._original_task_ids[task_idx]
        else:
            sample['task_id'] = task_idx

        sample['task_name'] = self.task_names[task_idx]
        return sample

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def get_normalizer(self, task_name: Optional[str] = None):
        if self.normalizer_mode == 'shared':
            return self.normalizer
        else:
            if task_name is None:
                raise ValueError("normalizer_mode='per_task' requires task_name argument")
            return self.normalizers[task_name]

    def get_validation_dataset(self):
        val_datasets = [d.get_validation_dataset() for d in self.datasets]
        valid_pairs = [(d, name, i) for i, (d, name) in enumerate(zip(val_datasets, self.task_names)) if d is not None]
        if not valid_pairs:
            return None
        val_ds, val_names, original_ids = zip(*valid_pairs)
        return MultiTaskDataset(
            datasets=list(val_ds),
            task_names=list(val_names),
            sampling_strategy='proportional',
            normalizer_mode=self.normalizer_mode,
            seed=self.seed,
            _deterministic=True,
            _original_task_ids=list(original_ids),
        )
