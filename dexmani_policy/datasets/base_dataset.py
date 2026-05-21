import copy
import torch

from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler, get_val_mask, downsample_mask


class BaseDataset(torch.utils.data.Dataset):
    DEFAULT_MODALITIES = ['joint_state']

    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        sensor_modalities=None,
        augmentation_cfg=None,
    ):
        super().__init__()

        if sensor_modalities is None:
            sensor_modalities = self.DEFAULT_MODALITIES

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=sensor_modalities + ['action'],
        )

        self.sensor_modalities = sensor_modalities
        self.augmentation_cfg = augmentation_cfg
        self.augmentors = {}
        if augmentation_cfg is not None:
            self._build_augmentors()

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
        if not self.val_mask.any():
            return None

        val_set = copy.copy(self)

        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        # 验证集禁用数据增强：置 None 阻断 apply_augmentation，清空 augmentors 断开共享引用
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        return {
            'obs': {m: sample[m] for m in self.sensor_modalities},
            'action': sample['action'],
        }

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        data = self.apply_augmentation(data)
        return dict_apply(data, torch.from_numpy)

    def _build_augmentors(self):
        pass

    def apply_augmentation(self, data):
        if self.augmentation_cfg is None:
            return data
        for modality, augs in self.augmentors.items():
            if modality not in data['obs'] or augs is None:
                continue
            for aug in augs:
                data['obs'][modality] = aug(data['obs'][modality])
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
