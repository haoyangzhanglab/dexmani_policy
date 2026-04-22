import torch
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import RGBAug, PCAug
from dexmani_policy.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class RGBPCDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        sensor_modalities=['joint_state', 'rgb', 'depth', 'point_cloud',
                           'camera_intrinsic', 'camera_extrinsic'],
        augmentation_cfg=None,
        use_pc_color=False,
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
        )
        self.use_pc_color = use_pc_color
        rgb_cfg = (augmentation_cfg or {}).get('rgb')
        pc_cfg = (augmentation_cfg or {}).get('pc')
        self.rgb_aug = RGBAug(**rgb_cfg) if rgb_cfg is not None else None
        self.pc_aug = PCAug(**pc_cfg) if (use_pc_color and pc_cfg is not None) else None

    def apply_augmentation(self, data):
        if self.augmentation_cfg is None:
            return data
        if self.rgb_aug is not None:
            data['obs']['rgb'] = self.rgb_aug(data['obs']['rgb'])
        if self.pc_aug is not None:
            data['obs']['point_cloud'] = self.pc_aug(data['obs']['point_cloud'])
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'point_cloud': self.replay_buffer['point_cloud'],
            'joint_state': self.replay_buffer['joint_state'],
            'action': self.replay_buffer['action'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
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
