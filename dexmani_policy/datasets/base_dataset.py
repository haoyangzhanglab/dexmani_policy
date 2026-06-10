import copy
from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as TVF

from dexmani_policy.common.normalizer import LinearNormalizer, build_mixed_action_normalizer
from dexmani_policy.common.pytorch_util import dict_apply, ensure_tensor
from dexmani_policy.datasets.augmentation import (
    ImageAug,
    PointColorJitter,
    PointCoordNoiseAug,
    PointDropout,
    StateNoiseAug,
)
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler, get_val_mask, downsample_mask

# (yaml_section, augmentor_class, yaml_key, output_modality)
AUGMENTOR_REGISTRY = [
    ('pc',    PointCoordNoiseAug, 'coord_noise', 'point_cloud'),
    ('pc',    PointColorJitter, 'color',    'point_cloud'),
    ('pc',    PointDropout,     'dropout',  'point_cloud'),
    ('state', StateNoiseAug,    'noise',    'joint_state'),
    ('rgb',   ImageAug,         'color',    'rgb'),
]


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
        action_key='action',
        obs_horizon: Optional[int] = None,
        rgb_preprocess_size: Optional[Tuple[int, int]] = None,
        rgb_random_crop_size: Optional[Tuple[int, int]] = None,
        rgb_color_aug=None,
        rgb_keep_uint8: bool = False,
    ):
        super().__init__()

        if sensor_modalities is None:
            sensor_modalities = self.DEFAULT_MODALITIES

        self.action_key = action_key
        self.obs_horizon = obs_horizon
        self.rgb_preprocess_size = rgb_preprocess_size
        self.rgb_random_crop_size = rgb_random_crop_size
        self.rgb_color_aug = rgb_color_aug
        self.rgb_keep_uint8 = rgb_keep_uint8
        self._is_val = False  # 验证集标记 — get_validation_dataset() 会覆盖

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=sensor_modalities + [action_key],
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
        # 验证集禁用随机性：关闭 augmentation，随机裁剪切回 center_crop
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        val_set._is_val = True
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        return {
            'obs': {
                m: sample[m][:self.obs_horizon] if self.obs_horizon else sample[m]
                for m in self.sensor_modalities
            },
            'action': sample[self.action_key],
        }

    def _preprocess_rgb_cpu(self, rgb_np):
        """rgb_np: (T, H, W, 3) uint8 numpy → (T, 3, H_dst, W_dst) tensor.
        /255 + resize + optional random crop + optional color aug.
        ImageNet normalization is left on GPU.

        Two paths:
        - uint8 fast path: when ``rgb_keep_uint8=True`` and no color aug,
          resize/crop keep uint8 output → 4x less DataLoader→GPU transfer.
        - float32 path (default): current behavior, required for color augmentation.
        """
        rgb = torch.from_numpy(rgb_np)                 # (T, H, W, 3) uint8
        rgb = rgb.permute(0, 3, 1, 2).contiguous()     # (T, 3, H, W)

        # --- uint8 fast path: skip float32 conversion, resize/crop in uint8 ---
        if self.rgb_keep_uint8 and self.rgb_color_aug is None:
            rgb = TVF.resize(rgb, list(self.rgb_preprocess_size), antialias=True)
            if self.rgb_random_crop_size is not None:
                if self._is_val:
                    rgb = TVF.center_crop(rgb, list(self.rgb_random_crop_size))
                else:
                    rgb = TVF.crop(rgb,
                        top=torch.randint(0, rgb.shape[-2] - self.rgb_random_crop_size[0] + 1, (1,)).item(),
                        left=torch.randint(0, rgb.shape[-1] - self.rgb_random_crop_size[1] + 1, (1,)).item(),
                        height=self.rgb_random_crop_size[0],
                        width=self.rgb_random_crop_size[1])
            return rgb.contiguous()                      # uint8

        # --- float32 path: current behavior (needed for color augmentation) ---
        rgb = rgb.float().div_(255.0)                    # (T, 3, H, W) float32 [0,1]
        rgb = TVF.resize(rgb, list(self.rgb_preprocess_size), antialias=True)
        if self.rgb_random_crop_size is not None:
            if self._is_val:
                rgb = TVF.center_crop(rgb, list(self.rgb_random_crop_size))
            else:
                rgb = TVF.crop(rgb,
                    top=torch.randint(0, rgb.shape[-2] - self.rgb_random_crop_size[0] + 1, (1,)).item(),
                    left=torch.randint(0, rgb.shape[-1] - self.rgb_random_crop_size[1] + 1, (1,)).item(),
                    height=self.rgb_random_crop_size[0],
                    width=self.rgb_random_crop_size[1])
        if self.rgb_color_aug is not None and not self._is_val:
            rgb = self.rgb_color_aug(rgb)                # (T, 3, H_dst, W_dst) float32 [0,1]
        return rgb.clamp(0, 1).contiguous()

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        data = self.apply_augmentation(data)

        if self.rgb_preprocess_size is not None and 'rgb' in data['obs']:
            data['obs']['rgb'] = self._preprocess_rgb_cpu(data['obs']['rgb'])

        return dict_apply(data, ensure_tensor)

    def _build_augmentors(self):
        """Build augmentors from ``augmentation_cfg`` using AUGMENTOR_REGISTRY.

        Subclasses that need different augmentors (e.g. RGB-only) can override
        this method, but the registry covers all standard cases.
        """
        self.augmentors = {}
        if self.augmentation_cfg is None:
            return
        for section, cls, key, modality in self.AUGMENTOR_REGISTRY:
            config = (self.augmentation_cfg.get(section) or {}).get(key)
            if config:
                self.augmentors.setdefault(modality, []).append(cls(**config))

    def apply_augmentation(self, data):
        if self.augmentation_cfg is None:
            return data
        for modality, augs in self.augmentors.items():
            if modality not in data['obs'] or augs is None:
                continue
            for aug in augs:
                data['obs'][modality] = aug(data['obs'][modality])
        return data

    def get_normalizer(self, mode='limits'):
        joint_state = self.replay_buffer['joint_state']
        action = self.replay_buffer[self.action_key]
        normalizer = LinearNormalizer()

        if self.action_key == 'action_ee':
            normalizer.fit(data={'joint_state': joint_state}, last_n_dims=1, mode=mode)
            normalizer['action'] = build_mixed_action_normalizer(action)
        else:
            normalizer.fit(data={'joint_state': joint_state, 'action': action},
                           last_n_dims=1, mode=mode)
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
    dataset.get_normalizer()
    sample = dataset[0]
    print('joint_state:', sample['obs']['joint_state'].shape)
    print('action     :', sample['action'].shape)
    val_set = dataset.get_validation_dataset()
    print(f'train size: {len(dataset)}  val size: {len(val_set)}')


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
