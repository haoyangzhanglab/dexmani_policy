import numpy as np
import torch
from typing import List, Dict, Optional
from dexmani_policy.common.normalizer import LinearNormalizer


class MultiTaskDataset:
    """
    多任务混合训练数据集

    支持从多个任务的数据集中按策略采样，每个样本附带 task_id。

    采样策略:
        - 'proportional': 按各任务数据量比例采样
        - 'balanced': 各任务等概率采样
        - 'weighted': 按用户指定权重采样

    Normalizer 策略:
        - 'shared': 所有任务共享一个 normalizer（要求 action space 相同）
        - 'per_task': 每个任务独立 normalizer（需 Agent 支持动态切换）
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
    ):
        assert len(datasets) == len(task_names)
        assert sampling_strategy in ['proportional', 'balanced', 'weighted']
        assert normalizer_mode in ['shared', 'per_task']

        # 检查所有数据集非空
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

        # 构建索引映射: global_idx -> (task_idx, local_idx)
        self.task_lengths = [len(d) for d in datasets]
        self.cumsum_lengths = np.cumsum([0] + self.task_lengths)
        self._total_length = sum(self.task_lengths)

        # 用于确定性采样的 epoch 计数器
        self._epoch = 0

        # 确定性模式：验证集使用固定索引映射
        self._deterministic = _deterministic
        if self._deterministic:
            self._fixed_indices = self._generate_fixed_indices()

        # 计算采样概率
        if sampling_strategy == 'proportional':
            self.sample_probs = np.array(self.task_lengths) / self._total_length
        elif sampling_strategy == 'balanced':
            self.sample_probs = np.ones(self.num_tasks) / self.num_tasks
        elif sampling_strategy == 'weighted':
            assert task_weights is not None and len(task_weights) == self.num_tasks
            total = sum(task_weights)
            self.sample_probs = np.array(task_weights) / total

        # Normalizer 处理
        if normalizer_mode == 'shared':
            self.normalizer = self._compute_shared_normalizer()
        else:
            self.normalizers = {name: d.get_normalizer() for name, d in zip(task_names, datasets)}

    def _compute_shared_normalizer(self):
        """合并所有任务的数据计算共享 normalizer"""
        # 假设所有任务的 action space 相同
        all_joint_states = []
        all_actions = []

        for i, dataset in enumerate(self.datasets):
            # 检查必需字段是否存在
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

        # 添加其他字段（如相机参数）
        from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer
        normalizer['camera_intrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        normalizer['camera_extrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        return normalizer

    def _generate_fixed_indices(self):
        """生成固定的索引映射，按 proportional 策略"""
        indices = []
        for task_idx, task_length in enumerate(self.task_lengths):
            for local_idx in range(task_length):
                indices.append((task_idx, local_idx))
        return indices

    def __len__(self):
        return self._total_length

    def __getitem__(self, idx):
        """动态采样，使用确定性随机数生成确保可复现性"""
        if self._deterministic:
            # 确定性模式：按固定顺序返回
            task_idx, local_idx = self._fixed_indices[idx]
        else:
            # 动态采样模式（训练集）
            # 基于 seed、epoch 和 idx 生成确定性随机数
            item_seed = hash((self.seed, self._epoch, idx)) % (2**32)
            item_rng = np.random.default_rng(item_seed)

            # 根据概率选择任务
            task_idx = item_rng.choice(self.num_tasks, p=self.sample_probs)
            # 从该任务中随机选择样本
            local_idx = item_rng.integers(0, self.task_lengths[task_idx])

        # 从对应任务采样
        sample = self.datasets[task_idx][local_idx]

        # 添加任务信息
        sample['task_id'] = task_idx
        sample['task_name'] = self.task_names[task_idx]

        return sample

    def set_epoch(self, epoch: int):
        """设置当前 epoch，用于确定性采样"""
        self._epoch = epoch

    def get_normalizer(self, task_name: Optional[str] = None):
        """获取 normalizer"""
        if self.normalizer_mode == 'shared':
            return self.normalizer
        else:
            if task_name is None:
                raise ValueError("normalizer_mode='per_task' requires task_name argument")
            return self.normalizers[task_name]

    def get_validation_dataset(self):
        """返回多任务验证集"""
        val_datasets = [d.get_validation_dataset() for d in self.datasets]
        return MultiTaskDataset(
            datasets=val_datasets,
            task_names=self.task_names,
            sampling_strategy='proportional',
            normalizer_mode=self.normalizer_mode,
            seed=self.seed,
            _deterministic=True,
        )
