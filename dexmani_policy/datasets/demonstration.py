from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_policy.common.config import POINT_CLOUD_FIELDS
from dexmani_policy.common.image_util import preprocess_rgb_train, validate_rgb_preprocessing
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.split import format_split_summary, split_episodes
from dexmani_policy.datasets.transforms import AUGMENTATION_SPECS, PointCoordinateNoise


class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path: str | Path,
        *,
        sensor_modalities: list[str],
        obs_horizon: int,
        seed: int = 42,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_episodes: int = 0,
        max_train_episodes: int | None = None,
        augmentation_cfg: dict | None = None,
        action_key: str = "action",
        rgb_preprocess_size: tuple[int, int] | None = None,
        rgb_random_crop_size: tuple[int, int] | None = None,
        rgb_color_aug: Callable | None = None,
        pc_dim: int | None = None,
    ) -> None:
        super().__init__()

        if "rgb" in sensor_modalities:
            validate_rgb_preprocessing(rgb_preprocess_size, rgb_random_crop_size, rgb_color_aug)

        self.action_key = action_key
        if pc_dim is not None and (type(pc_dim) is not int or pc_dim not in (3, 6)):
            raise ValueError("pc_dim must be 3 (XYZ), 6 (XYZRGB), or None (all stored channels)")
        self.pc_dim = pc_dim
        self.obs_horizon = obs_horizon
        if not 1 <= self.obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        self.rgb_preprocess_size = rgb_preprocess_size
        self.rgb_random_crop_size = rgb_random_crop_size
        self.rgb_color_aug = rgb_color_aug
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.sensor_modalities = sensor_modalities
        self.episode_store = EpisodeStore(zarr_path, keys=self.data_keys, pc_dim=pc_dim)
        self.build_augmentors(augmentation_cfg)
        if {"point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic"} <= set(sensor_modalities):
            if rgb_preprocess_size is not None:
                raise ValueError(
                    "Pinhole RGB must retain its input resolution in Dataset; resize in ImageProcessor"
                )
            if any(
                isinstance(aug, PointCoordinateNoise) and aug.prob > 0 and aug.noise_std > 0
                for aug in self.augmentors.get("point_cloud", ())
            ):
                raise ValueError("Calibrated RGB/point clouds cannot use independent XYZ coordinate noise")

        self.train_mask, self.val_mask = split_episodes(
            self.episode_store.episode_groups, val_episodes=val_episodes,
            seed=seed, max_train_episodes=max_train_episodes,
        )
        if len(self.episode_store[action_key].shape) != 2:
            raise ValueError("Stored actions must have shape [T,D]")
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No training sequence windows: selected episodes are too short")

    @property
    def data_keys(self):
        return list(self.sensor_modalities) + [self.action_key]

    @property
    def consumed_fields(self):
        fields = dict.fromkeys(self.data_keys)
        if self.pc_dim is not None:
            for key in POINT_CLOUD_FIELDS.intersection(fields):
                fields[key] = slice(0, self.pc_dim)
        return fields

    @property
    def action_dim(self):
        return self.episode_store[self.action_key].shape[-1]

    @property
    def action_start_index(self):
        return self.obs_horizon - 1

    def training_summary(self):
        store = self.episode_store
        validation_note = "val 留出但未验证; " if self.val_mask.any() else ""
        return (
            f"{Path(store.path).name}: {format_split_summary(store, self.train_mask, self.val_mask)}; "
            f"{validation_note}train windows={len(self):,}; "
            f"fields={', '.join(self.data_keys)}"
        )

    def build_sampler(self, episode_mask):
        return SequenceSampler(
            self.episode_store, self.horizon, self.pad_before, self.pad_after, episode_mask=episode_mask,
            fields=self.consumed_fields, field_horizons={key: self.obs_horizon for key in self.sensor_modalities},
        )

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        return {
            "obs": {key: sample[key] for key in self.sensor_modalities},
            "action": sample[self.action_key],
        }

    def __getitem__(self, idx):
        return self._prepare_sample(self.sampler.sample_sequence(idx))

    def __getitems__(self, indices):
        # Preserve per-item hooks supplied by dataset and sampler subclasses.
        if type(self).__getitem__ is not DemonstrationDataset.__getitem__ or type(self.sampler) is not SequenceSampler:
            return [self[idx] for idx in indices]
        return [self._prepare_sample(sample) for sample in self.sampler.sample_batch(indices)]

    def _prepare_sample(self, sample):
        data = self.sample_to_data(sample)
        # Each sample has a separate writable row range for in-place augmentation.
        for modality, augmentors in self.augmentors.items():
            values = data["obs"][modality]
            for augmentor in augmentors:
                if np.random.random() <= augmentor.prob:
                    augmentor.augment(values)
        if "rgb" in data["obs"]:
            rgb = data["obs"]["rgb"]
            if self.rgb_preprocess_size is not None:
                data["obs"]["rgb"] = preprocess_rgb_train(
                    rgb,
                    resize_hw=self.rgb_preprocess_size,
                    random_crop_hw=self.rgb_random_crop_size,
                    color_augmentation=self.rgb_color_aug,
                )
            else:
                data["obs"]["rgb"] = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))

        return dict_apply(data, torch.as_tensor)

    def build_augmentors(self, augmentation_cfg):
        self.augmentors: dict[str, list[Any]] = {}
        augmentation = augmentation_cfg or {}
        for section, cls, key, modality in AUGMENTATION_SPECS:
            if modality not in self.sensor_modalities:
                continue
            config = (augmentation.get(section) or {}).get(key)
            if config:
                if (
                    section == "pc" and key == "color"
                    and (self.pc_dim or self.episode_store["point_cloud"].shape[-1]) < 6
                ):
                    if config.get("prob", 1) > 0:
                        raise ValueError(
                            f"XYZ-only or incomplete RGB point clouds cannot use color augmentation ({key}); "
                            "set prob=0 or use XYZRGB channels"
                        )
                    continue
                self.augmentors.setdefault(modality, []).append(cls(**config))

    def iter_normalization_data(self, key):
        field = self.action_key if key == "action" else key
        columns = self.consumed_fields[field]
        for _, values in self.episode_store.iter_chunks(field, columns=columns, episode_mask=self.train_mask):
            yield values
