import copy
import torch
import numpy as np
from typing import Dict

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
        sensor_modalities=['joint_state',],
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
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
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
            data['obs'][modality] = sample[modality]
        data['action'] = sample['action'].astype(np.float32, copy=False)
        return data

    # 子类需要重定义这个函数来适应不同模态的归一化方式
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'joint_state': self.replay_buffer['joint_state'],
            'action': self.replay_buffer['action']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        return normalizer

