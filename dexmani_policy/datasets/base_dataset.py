import copy
import torch
import numpy as np
from typing import Dict

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dexmani_policy.datasets.common.sampler import SequenceSampler, get_val_mask, downsample_mask


def create_rgb_normalizer():
    # Imagenet normalization stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    scale = 1.0 / std
    offset = -mean / std

    # 输入数据是[0, 1]范围内的RGB图像，输出数据是经过Imagenet标准化后的图像
    imagenet_normalizer = SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={
            'min': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'max': np.array([1.0, 1.0, 1.0], dtype=np.float32),
            'mean': mean,
            'std': std,
        }
    )     
    return imagenet_normalizer



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
        sensor_modalities=['point_cloud', 'state'],
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, 
            keys=sensor_modalities + ['action'],
        )
        self.sensor_modalities = sensor_modalities

        val_mask = get_val_mask(
            seed=seed,
            val_ratio=val_ratio,
            n_episodes=self.replay_buffer.n_episodes, 
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            seed=seed,
            mask=train_mask, 
            max_n=max_train_episodes,  
        )
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
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set


    def __len__(self) -> int:
        return len(self.sampler)
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


    def _sample_to_data(self, sample):
        data = {'obs': {}}
        for modality in self.sensor_modalities:
            data['obs'][modality] = sample[modality].astype(np.float32)
    
        data['action'] = sample['action'].astype(np.float32)
        return data


    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # 默认观察量不进行归一化
        for modality in self.sensor_modalities:
            normalizer[modality] = SingleFieldLinearNormalizer.create_identity()

        return normalizer


