from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import RGBAug


class RGBDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        sensor_modalities=['joint_state', 'rgb'],
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
        rgb_cfg = (augmentation_cfg or {}).get('rgb')
        self.rgb_aug = RGBAug(**rgb_cfg) if rgb_cfg is not None else None

    def apply_augmentation(self, data):
        if self.augmentation_cfg is None or self.rgb_aug is None:
            return data
        data['obs']['rgb'] = self.rgb_aug(data['obs']['rgb'])
        return data


def example(zarr_path):
    dataset = RGBDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
    )
    sample = dataset[0]
    print('rgb        :', sample['obs']['rgb'].shape, sample['obs']['rgb'].dtype)
    print('joint_state:', sample['obs']['joint_state'].shape)
    print('action     :', sample['action'].shape)
    print(f'train size: {len(dataset)}  val size: {len(dataset.get_validation_dataset())}')


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
