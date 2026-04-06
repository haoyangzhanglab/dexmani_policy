import torch
import numpy as np
from typing import Dict, Optional, Sequence

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dexmani_policy.datasets.common.image_processor import ImageProcessor, build_image_processor



class MultiModalDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        seed=42,
        horizon=1,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_train_episodes=None,
        rgb_key: str = "rgb",
        depth_key: Optional[str] = None,
        intrinsics_key: Optional[str] = None,
        camera_to_world_key: Optional[str] = None,
        lowdim_keys: Sequence[str] = ("joint_state",),
        processor_name: str = "dino",
        image_processor: Optional[ImageProcessor] = None,
        keep_camera_time_dim: bool = False,
    ):
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.intrinsics_key = intrinsics_key
        self.camera_to_world_key = camera_to_world_key
        self.lowdim_keys = list(lowdim_keys)
        self.keep_camera_time_dim = keep_camera_time_dim
        self.use_depth = depth_key is not None

        if self.use_depth and intrinsics_key is None:
            raise ValueError("intrinsics_key must be provided when depth_key is used.")

        sensor_modalities = [rgb_key] + self.lowdim_keys
        if self.use_depth:
            sensor_modalities += [depth_key, intrinsics_key]
            if camera_to_world_key is not None:
                sensor_modalities.append(camera_to_world_key)

        super().__init__(
            zarr_path=zarr_path,
            seed=seed,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            sensor_modalities=sensor_modalities,
        )

        self.image_processor = image_processor if image_processor is not None else build_image_processor(processor_name)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return self.sample_to_data(sample)

    def select_camera_value(self, x: np.ndarray):
        if self.keep_camera_time_dim:
            return x
        if x.ndim == 2:
            return x
        return x[0]

    def sample_to_data(self, sample) -> Dict[str, torch.Tensor]:
        data = {"obs": {}}
        images = sample[self.rgb_key]

        if self.use_depth:
            depths = sample[self.depth_key]
            intrinsics = self.select_camera_value(sample[self.intrinsics_key])
            camera_to_world = None
            if self.camera_to_world_key is not None:
                camera_to_world = self.select_camera_value(sample[self.camera_to_world_key])

            proc_out = self.image_processor.process_rgbd(
                images=images,
                depths=depths,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                collapse_repeated_camera=not self.keep_camera_time_dim,
            )

            data["obs"]["image"] = proc_out["image"]
            data["obs"]["depth"] = proc_out["depth"]
            data["obs"]["intrinsics"] = proc_out["intrinsics"]
            if "camera_to_world" in proc_out:
                data["obs"]["camera_to_world"] = proc_out["camera_to_world"]
        else:
            proc_out = self.image_processor.process_image(images)
            data["obs"]["image"] = proc_out["image"]

        for key in self.lowdim_keys:
            data["obs"][key] = torch.from_numpy(sample[key].astype(np.float32))

        data["action"] = torch.from_numpy(sample["action"].astype(np.float32))
        return data

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": self.replay_buffer["action"]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )

        normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer["depth"] = SingleFieldLinearNormalizer.create_identity()
            normalizer["intrinsics"] = SingleFieldLinearNormalizer.create_identity()
            if self.camera_to_world_key is not None:
                normalizer["camera_to_world"] = SingleFieldLinearNormalizer.create_identity()

        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()

        return normalizer


def example():
    dataset = MultiModalDataset(
        zarr_path="your_dataset.zarr",
        horizon=4,
        rgb_key="rgb",
        depth_key="depth",
        intrinsics_key="intrinsics",
        camera_to_world_key="camera_to_world",
        lowdim_keys=("joint_state",),
        processor_name="dino",
        keep_camera_time_dim=False,
    )

    sample = dataset[0]
    print("obs.image:", tuple(sample["obs"]["image"].shape))
    print("obs.depth:", tuple(sample["obs"]["depth"].shape))
    print("obs.intrinsics:", tuple(sample["obs"]["intrinsics"].shape))
    print("action:", tuple(sample["action"].shape))


if __name__ == "__main__":
    example()