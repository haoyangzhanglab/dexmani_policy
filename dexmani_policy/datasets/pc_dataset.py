from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import PCAug
from dexmani_policy.common.normalizer import LinearNormalizer


class PCDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        sensor_modalities=['joint_state', 'point_cloud'],
        augmentation_cfg=None,
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
        pc_cfg = (augmentation_cfg or {}).get('pc')
        self.pc_aug = PCAug(**pc_cfg) if pc_cfg else None

    def apply_augmentation(self, data):
        if self.augmentation_cfg is None or self.pc_aug is None:
            return data
        data['obs']['point_cloud'] = self.pc_aug(data['obs']['point_cloud'])
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
    dataset = PCDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
    )
    normalizer = dataset.get_normalizer()
    sample = dataset[0]
    print('point_cloud:', sample['obs']['point_cloud'].shape)
    print('joint_state:', sample['obs']['joint_state'].shape)
    print('action     :', sample['action'].shape)
    print(f'train size: {len(dataset)}  val size: {len(dataset.get_validation_dataset())}')

    aug_dataset = PCDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        enable_color_aug=True,
        augmentation_cfg={'pc': {'color_std': 0.05}},
    )
    print('aug point_cloud:', aug_dataset[0]['obs']['point_cloud'].shape)


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
