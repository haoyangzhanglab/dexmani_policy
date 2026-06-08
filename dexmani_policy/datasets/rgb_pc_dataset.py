import torch
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import (
    RGBAug, PCColorJitter, PCSpatialAug, PCDropout, StateNoiseAug, PC_AUG_CLASSES,
)
from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer


class RGBPCDataset(BaseDataset):

    DEFAULT_MODALITIES = ['joint_state', 'rgb', 'depth', 'point_cloud',
                          'camera_intrinsic', 'camera_extrinsic']

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
        obs_horizon=None,
        rgb_preprocess_size=None,
        rgb_random_crop_size=None,
        rgb_color_aug=None,
        rgb_keep_uint8=False,
    ):
        super().__init__(
            zarr_path=zarr_path,
            seed=seed,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            sensor_modalities=sensor_modalities,
            augmentation_cfg=augmentation_cfg,
            action_key=action_key,
            obs_horizon=obs_horizon,
            rgb_preprocess_size=rgb_preprocess_size,
            rgb_random_crop_size=rgb_random_crop_size,
            rgb_color_aug=rgb_color_aug,
            rgb_keep_uint8=rgb_keep_uint8,
        )

    def _build_augmentors(self):
        self.augmentors = {}
        if self.augmentation_cfg is None:
            return

        rgb_cfg = self.augmentation_cfg.get('rgb')
        if rgb_cfg is not None:
            self.augmentors['rgb'] = [RGBAug(**rgb_cfg)]

        pc_cfg = self.augmentation_cfg.get('pc')
        if pc_cfg is not None:
            pc_augs = []
            for name, cls in PC_AUG_CLASSES.items():
                aug_cfg = pc_cfg.get(name)
                if aug_cfg is not None:
                    pc_augs.append(cls(**aug_cfg))
            if pc_augs:
                self.augmentors['point_cloud'] = pc_augs

        state_cfg = self.augmentation_cfg.get('state')
        if state_cfg is not None:
            self.augmentors['joint_state'] = [StateNoiseAug(**state_cfg)]

    def get_normalizer(self, mode='limits'):
        normalizer = super().get_normalizer(mode=mode)
        normalizer['camera_intrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        normalizer['camera_extrinsic'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        return normalizer


def example(zarr_path):
    dataset = RGBPCDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
    )
    normalizer = dataset.get_normalizer()
    sample = dataset[0]
    obs_n = normalizer.normalize(sample['obs'])
    action_n = normalizer['action'].normalize(sample['action'])
    print('rgb             :', sample['obs']['rgb'].shape, sample['obs']['rgb'].dtype)
    print('depth           :', sample['obs']['depth'].shape, sample['obs']['depth'].dtype)
    print('point_cloud     :', sample['obs']['point_cloud'].shape)
    print('camera_intrinsic:', obs_n['camera_intrinsic'].shape)
    print('camera_extrinsic:', obs_n['camera_extrinsic'].shape)
    print('action          :', action_n.shape)
    print(f'train size: {len(dataset)}  val size: {len(dataset.get_validation_dataset())}')


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
