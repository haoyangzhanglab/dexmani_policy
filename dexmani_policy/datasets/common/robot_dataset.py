from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_policy.common.image_util import (
    preprocess_rgb_train,
    preprocess_rgb_eval,
    validate_rgb_preprocessing,
)
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.common.config import FINGER_VECTOR_FIELDS, POINT_CLOUD_FIELDS, positive_int
from dexmani_policy.datasets.common.augmentation import (
    AUGMENTATION_SPECS,
    random_sample_points,
)
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler
from dexmani_policy.datasets.common.validation import validate_fields


class RobotDataset(torch.utils.data.Dataset):
    training_dataset: "RobotDataset"


    def __init__(
        self,
        zarr_path: str,
        *,
        sensor_modalities: list[str],
        seed: int = 42,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        augmentation_cfg: dict | None = None,
        action_key: str = "action",
        obs_horizon: int | None = None,
        rgb_preprocess_size: tuple[int, int] | None = None,
        rgb_random_crop_size: tuple[int, int] | None = None,
        rgb_color_aug: Callable | None = None,
        rgb_keep_uint8: bool = False,
        split_manifest: str | dict | None = None,
        pc_dim: int | None = None,
    ) -> None:
        super().__init__()

        if "rgb" in sensor_modalities:
            validate_rgb_preprocessing(rgb_preprocess_size, rgb_random_crop_size, rgb_color_aug)

        self.action_key = action_key
        if pc_dim is not None and (type(pc_dim) is not int or pc_dim not in (3, 6)):
            raise ValueError("pc_dim must be 3 (XYZ), 6 (XYZRGB), or None (all stored channels)")
        self.pc_dim = pc_dim
        self.obs_horizon = horizon if obs_horizon is None else obs_horizon
        if not 1 <= self.obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        self.rgb_preprocess_size = rgb_preprocess_size
        self.rgb_random_crop_size = rgb_random_crop_size
        self.rgb_color_aug = rgb_color_aug
        self.rgb_keep_uint8 = rgb_keep_uint8
        self.is_validation = False
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.sensor_modalities = sensor_modalities
        self.replay_buffer = ReplayBuffer(zarr_path, keys=self.data_keys)
        self.zarr_path = self.replay_buffer.path

        if "rgb" in sensor_modalities:
            rgb = self.replay_buffer["rgb"]
            if len(rgb.shape) != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Stored RGB must have shape (T, H, W, 3) and dtype uint8")
        for key in POINT_CLOUD_FIELDS.intersection(sensor_modalities):
            shape = self.replay_buffer[key].shape
            if len(shape) != 3:
                raise ValueError(f"Stored {key} must have shape [T,N,C]")
            if pc_dim is not None and shape[-1] < pc_dim:
                raise ValueError(f"Stored {key} does not satisfy configured pc_dim")
        self.augmentation_cfg = augmentation_cfg
        self.build_augmentors()

        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.replay_buffer, manifest=split_manifest, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
        )
        for key in FINGER_VECTOR_FIELDS.intersection(sensor_modalities):
            if self.replay_buffer[key].shape[1:] != (5, 3):
                raise ValueError(f"{key} requires stored shape (T, 5, 3)")
        if len(self.replay_buffer[action_key].shape) != 2:
            raise ValueError("Stored actions must have shape [T,D]")
        validate_fields(self.replay_buffer, self.train_mask | self.val_mask, self.consumed_fields)
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No training sequence windows: selected episodes are too short")
        self.validation_dataset: RobotDataset | None = None
        self._normalizer: LinearNormalizer | None = None

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
        return self.replay_buffer[self.action_key].shape[-1]

    @property
    def action_start_index(self):
        return self.obs_horizon - 1

    def training_summary(self):
        return (
            f"{Path(self.zarr_path).name}: train episodes={int(self.train_mask.sum()):,}, "
            f"val episodes={int(self.val_mask.sum()):,}, train windows={len(self):,}, "
            f"fields={', '.join(self.data_keys)}"
        )

    def build_sampler(self, episode_mask):
        return SequenceSampler(
            self.replay_buffer, self.horizon, self.pad_before, self.pad_after, episode_mask,
            keys=self.data_keys, key_first_k={key: self.obs_horizon for key in self.sensor_modalities},
        )

    def get_validation_dataset(self):
        if self.is_validation:
            return self
        if self.validation_dataset is not None:
            return self.validation_dataset
        if not self.val_mask.any():
            return None

        val_set = copy.copy(self)
        val_set.training_dataset = self

        val_set.sampler = self.build_sampler(self.val_mask)
        if not len(val_set.sampler):
            return None
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        val_set.random_keep_points = None
        val_set.is_validation = True
        self.validation_dataset = val_set
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        if self.pc_dim is not None:
            for key in POINT_CLOUD_FIELDS.intersection(sample):
                sample[key] = sample[key][..., : self.pc_dim]
        action = sample[self.action_key]
        return {
            "obs": {m: sample[m] for m in self.sensor_modalities},
            "action": action,
        }

    def preprocess_rgb(self, rgb, resize_hw: tuple[int, int]):
        if self.is_validation:
            return preprocess_rgb_eval(
                rgb,
                resize_hw=resize_hw,
                center_crop_hw=self.rgb_random_crop_size,
                keep_uint8=self.rgb_keep_uint8,
                float_spatial_before_uint8=self.rgb_color_aug is not None,
            )
        return preprocess_rgb_train(
            rgb,
            resize_hw=resize_hw,
            random_crop_hw=self.rgb_random_crop_size,
            keep_uint8=self.rgb_keep_uint8,
            color_augmentation=self.rgb_color_aug,
        )

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        self.apply_augmentation(data)
        if self.random_keep_points is not None:
            data["obs"]["point_cloud"] = random_sample_points(
                data["obs"]["point_cloud"],
                self.random_keep_points,
            )

        if self.rgb_preprocess_size is not None and "rgb" in data["obs"]:
            data["obs"]["rgb"] = self.preprocess_rgb(data["obs"]["rgb"], self.rgb_preprocess_size)

        return dict_apply(data, torch.as_tensor)

    def build_augmentors(self):
        self.augmentors: dict[str, list[Any]] = {}
        augmentation = self.augmentation_cfg or {}
        pc_cfg = augmentation.get("pc") or {}
        self.random_keep_points = pc_cfg.get("random_keep_points")
        if self.random_keep_points is not None:
            if "point_cloud" not in self.sensor_modalities:
                raise ValueError("random_keep_points requires the point_cloud modality")
            positive_int(self.random_keep_points, "random_keep_points")
            shape = self.replay_buffer["point_cloud"].shape
            if shape[1] < self.random_keep_points:
                raise ValueError(
                    f"point_cloud N={shape[1]} must be >= random_keep_points={self.random_keep_points}"
                )
        for section, cls, key, modality in AUGMENTATION_SPECS:
            if modality not in self.sensor_modalities:
                continue
            config = (augmentation.get(section) or {}).get(key)
            if config:
                if (
                    section == "pc" and key in {"color", "color_noise"}
                    and (self.pc_dim or self.replay_buffer["point_cloud"].shape[-1]) < 6
                ):
                    if config.get("prob", 1) > 0:
                        raise ValueError(
                            f"XYZ-only or incomplete RGB point clouds cannot use color augmentation ({key}); "
                            "set prob=0 or use XYZRGB channels"
                        )
                    continue
                self.augmentors.setdefault(modality, []).append(cls(**config))

    def apply_augmentation(self, data):
        """ReplayBuffer reads own their storage, so augmentation can run in place."""
        for modality, augs in self.augmentors.items():
            x = data["obs"][modality]
            for aug in augs:
                if np.random.random() <= aug.prob:
                    aug.augment(x)

    def iter_normalization_data(self, key):
        if self.is_validation:
            raise ValueError("Validation must reuse the training normalizer")
        field = self.action_key if key == "action" else key
        columns = self.consumed_fields[field]
        for _, values in self.replay_buffer.iter_chunks(field, columns=columns, episode_mask=self.train_mask):
            yield values

    def get_normalizer(self, spec):
        if self.is_validation:
            return self.training_dataset.get_normalizer(spec)
        if self._normalizer is not None:
            return self._normalizer
        normalizer = LinearNormalizer()
        for key, mode in spec.items():
            if mode == "identity":
                continue
            normalizer.fit_field(
                key, self.iter_normalization_data(key),
                mode="limits" if mode == "auto" else mode,
                last_n_dims=2 if key in FINGER_VECTOR_FIELDS else 1,
            )
            if key == "action" and mode == "auto" and self.action_key == "action_ee":
                field = normalizer[key]
                if field.scale.numel() <= 9:
                    raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
                field.scale[3:9] = 1
                field.offset[3:9] = 0
                for name, value in (("min", -1), ("max", 1), ("mean", 0), ("std", 1)):
                    field.input_stats[name][3:9] = value
        self._normalizer = normalizer
        return normalizer
