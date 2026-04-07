from dexmani_policy.datasets.base_dataset import BaseDataset

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
        return super().get_normalizer(mode=mode, **kwargs)


def example():
    dataset = RGBDataset(
        zarr_path='/home/zhanghaoyang/Desktop/DexMani_Policy/robot_data/sim/pick_apple_messy.zarr',
        seed=0,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        max_train_episodes=50,
        sensor_modalities=['joint_state', 'rgb'],
    )
    normalizer = dataset.get_normalizer()
    sample = dataset[0]
    normalized_sample = {}
    normalized_sample['obs'] = normalizer.normalize(sample['obs'])
    normalized_sample['action'] = normalizer['action'].normalize(sample['action'])

    print("Normalized action shape:", normalized_sample['action'].shape)
    print("Normalized RGB shape:", normalized_sample['obs']['rgb'].shape, normalized_sample['obs']['rgb'].dtype)
    print("Normalized joint state shape:", normalized_sample['obs']['joint_state'].shape)


if __name__ == "__main__":
    example()