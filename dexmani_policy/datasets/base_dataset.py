import copy
import torch
import numpy as np

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler, get_val_mask, downsample_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        sensor_modalities=['joint_state'],
        augmentation_cfg=None,
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=sensor_modalities + ['action'],
        )

        # 验证所有必需字段是否存在
        required_keys = set(sensor_modalities + ['action'])
        available_keys = set(self.replay_buffer.keys())
        missing_keys = required_keys - available_keys

        if missing_keys:
            raise ValueError(
                f"Zarr file missing required keys: {sorted(missing_keys)}\n"
                f"Required: {sorted(required_keys)}\n"
                f"Available: {sorted(available_keys)}\n"
                f"Path: {zarr_path}"
            )

        self.sensor_modalities = sensor_modalities
        self.augmentation_cfg = augmentation_cfg

        val_mask = get_val_mask(
            seed=seed,
            val_ratio=val_ratio,
            n_episodes=self.replay_buffer.n_episodes,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(seed=seed, mask=train_mask, max_n=max_train_episodes)
        self.val_mask = val_mask
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """创建验证集，与训练集完全隔离"""
        val_set = copy.deepcopy(self)

        # 重新创建 sampler，使用验证集 mask
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,  # 共享 replay_buffer 节省内存
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        # 验证集不使用数据增强
        val_set.augmentation_cfg = None
        return val_set

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        data = self.apply_augmentation(data)
        return dict_apply(data, torch.from_numpy)

    def sample_to_data(self, sample):
        data = {'obs': {}}
        for modality in self.sensor_modalities:
            data['obs'][modality] = sample[modality]
        data['action'] = sample['action']
        return data

    def apply_augmentation(self, data):
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'joint_state': self.replay_buffer['joint_state'],
            'action': self.replay_buffer['action'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer


def example(zarr_path):
    dataset = BaseDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        sensor_modalities=['joint_state'],
    )
    normalizer = dataset.get_normalizer()
    sample = dataset[0]
    print('joint_state:', sample['obs']['joint_state'].shape)
    print('action     :', sample['action'].shape)
    val_set = dataset.get_validation_dataset()
    print(f'train size: {len(dataset)}  val size: {len(val_set)}')


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
