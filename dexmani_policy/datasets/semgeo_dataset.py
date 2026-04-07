import torch
from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class SemGeoDataset(BaseDataset):
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
    ):
        super().__init__(
            zarr_path=zarr_path,
            seed=seed,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            sensor_modalities=sensor_modalities
        )


    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'point_cloud': self.replay_buffer['point_cloud'],
            'joint_state': self.replay_buffer['joint_state'],
            'action': self.replay_buffer['action']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # 不需要归一化的模态可以不用恒等变换器，直接透传即可（不处理）
        # RGB和depth只能透传，因为当前实现中归一化器会自动把他们的数据类型转换为float32
        normalizer["camera_intrinsic"] = SingleFieldLinearNormalizer().create_identity(dtype=torch.float32)
        normalizer["camera_extrinsic"] = SingleFieldLinearNormalizer().create_identity(dtype=torch.float32)
        return normalizer


def example():
    dataset = SemGeoDataset(
        zarr_path='/home/zhanghaoyang/Desktop/DexMani_Policy/robot_data/sim/pick_apple_messy.zarr',
        seed=0,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        max_train_episodes=50,
        sensor_modalities=['joint_state', 'rgb', 'depth', 'point_cloud', 'camera_intrinsic', 'camera_extrinsic'],
    )
    normalizer = dataset.get_normalizer()
    sample = dataset[0]
    print("RGB shape:", sample['obs']['rgb'].shape, sample['obs']['rgb'].dtype)
    print("Depth shape:", sample['obs']['depth'].shape, sample['obs']['depth'].dtype)

    normalized_sample = {}
    normalized_sample['obs'] = normalizer.normalize(sample['obs'])
    normalized_sample['action'] = normalizer['action'].normalize(sample['action'])

    print("Normalized action shape:", normalized_sample['action'].shape)
    print("Normalized RGB shape:", normalized_sample['obs']['rgb'].shape, normalized_sample['obs']['rgb'].dtype)
    print("Normalized depth shape:", normalized_sample['obs']['depth'].shape, normalized_sample['obs']['depth'].dtype)
    print("Normalized point cloud shape:", normalized_sample['obs']['point_cloud'].shape)
    print("Normalized camera intrinsic shape:", normalized_sample['obs']['camera_intrinsic'].shape)
    print("Normalized camera extrinsic shape:", normalized_sample['obs']['camera_extrinsic'].shape)
    print("Normalized joint state shape:", normalized_sample['obs']['joint_state'].shape)


if __name__ == "__main__":
    example()
