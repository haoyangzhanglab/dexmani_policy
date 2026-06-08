from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import RGBAug, StateNoiseAug


class RGBDataset(BaseDataset):

    DEFAULT_MODALITIES = ['joint_state', 'rgb']

    def __init__(self, rgb_aug=None, rgb_keep_uint8=False, **kwargs):
        super().__init__(rgb_keep_uint8=rgb_keep_uint8, **kwargs)
        self.rgb_color_aug = rgb_aug  # 覆盖 BaseDataset 的 None，使 _preprocess_rgb_cpu 生效

    def _build_augmentors(self):
        self.augmentors = {}
        if self.augmentation_cfg is None:
            return

        # RGB 增强已通过 rgb_aug 参数在 CPU resize 后执行, 不在此处理
        state_cfg = self.augmentation_cfg.get('state')
        if state_cfg is not None:
            self.augmentors['joint_state'] = [StateNoiseAug(**state_cfg)]


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
