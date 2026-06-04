from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.augmentation import (
    PCColorJitter, PCSpatialAug, PCDropout, StateNoiseAug, PC_AUG_CLASSES,
)


class PCDataset(BaseDataset):

    DEFAULT_MODALITIES = ['joint_state', 'point_cloud']

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

    def _build_augmentors(self):
        self.augmentors = {}
        if self.augmentation_cfg is None:
            return

        pc_cfg = self.augmentation_cfg.get('pc')
        if pc_cfg is not None:
            augs = []
            for name, cls in PC_AUG_CLASSES.items():
                aug_cfg = pc_cfg.get(name)
                if aug_cfg is not None:
                    augs.append(cls(**aug_cfg))
            if augs:
                self.augmentors['point_cloud'] = augs

        state_cfg = self.augmentation_cfg.get('state')
        if state_cfg is not None:
            self.augmentors['joint_state'] = [StateNoiseAug(**state_cfg)]

    def get_normalizer(self, mode='limits', **kwargs):
        return super().get_normalizer(mode=mode, **kwargs)


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
        augmentation_cfg={
            'pc': {
                'color': {'brightness': 0.2, 'prob': 0.8},
                'spatial': {'rot_z': 15.0, 'trans_xy': 0.10, 'prob': 0.8},
            },
        },
    )
    print('aug point_cloud:', aug_dataset[0]['obs']['point_cloud'].shape)


if __name__ == '__main__':
    example('robot_data/sim/pick_apple_messy.zarr')
