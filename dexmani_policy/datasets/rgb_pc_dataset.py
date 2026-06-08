import torch

from dexmani_policy.common.normalizer import SingleFieldLinearNormalizer
from dexmani_policy.datasets.base_dataset import BaseDataset


class RGBPCDataset(BaseDataset):

    DEFAULT_MODALITIES = ['joint_state', 'rgb', 'depth', 'point_cloud',
                          'camera_intrinsic', 'camera_extrinsic']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
