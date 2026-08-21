import numpy as np

from dexmani_policy.common.normalizer import LinearNormalizer, build_mixed_action_normalizer
from dexmani_policy.datasets.base_dataset import BaseDataset


class PCDataset(BaseDataset):
    DEFAULT_MODALITIES = ["joint_state", "point_cloud"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_normalizer(self, mode="limits"):
        normalizer = LinearNormalizer()
        joint_state, action = self._get_normalizer_data()

        if self.use_aux_ee:
            parts = [self.replay_buffer["action"]]
            parts.append(self.replay_buffer["action_ee"][..., :9])
            action = np.concatenate(parts, axis=-1)
            normalizer.fit(
                data={
                    "joint_state": joint_state,
                    "action": action,
                    "point_cloud": self.replay_buffer["point_cloud"],
                },
                last_n_dims=1,
                mode=mode,
            )
        elif self.action_key == "action_ee":
            # EE-space action: use mixed normalizer (rot6d identity-norm)
            normalizer.fit(
                data={
                    "joint_state": joint_state,
                    "point_cloud": self.replay_buffer["point_cloud"],
                },
                last_n_dims=1,
                mode=mode,
            )
            normalizer["action"] = build_mixed_action_normalizer(action)
        else:
            normalizer.fit(
                data={
                    "joint_state": joint_state,
                    "action": action,
                    "point_cloud": self.replay_buffer["point_cloud"],
                },
                last_n_dims=1,
                mode=mode,
            )
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
    dataset.get_normalizer()
    sample = dataset[0]
    print("point_cloud:", sample["obs"]["point_cloud"].shape)
    print("joint_state:", sample["obs"]["joint_state"].shape)
    print("action     :", sample["action"].shape)
    print(f"train size: {len(dataset)}  val size: {len(dataset.get_validation_dataset())}")

    aug_dataset = PCDataset(
        zarr_path=zarr_path,
        seed=42,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        augmentation_cfg={
            "pc": {
                "color": {"brightness": 0.2, "prob": 0.8},
                "coord_noise": {"noise_std": 0.002, "prob": 0.5},
            },
        },
    )
    print("aug point_cloud:", aug_dataset[0]["obs"]["point_cloud"].shape)


if __name__ == "__main__":
    example("robot_data/sim/pick_apple_messy.zarr")
