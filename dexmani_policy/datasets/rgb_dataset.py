from dexmani_policy.datasets.base_dataset import BaseDataset


class RGBDataset(BaseDataset):

    DEFAULT_MODALITIES = ['joint_state', 'rgb']

    def __init__(self, rgb_aug=None, rgb_keep_uint8=False, **kwargs):
        super().__init__(rgb_keep_uint8=rgb_keep_uint8, **kwargs)
        self.rgb_color_aug = rgb_aug


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
