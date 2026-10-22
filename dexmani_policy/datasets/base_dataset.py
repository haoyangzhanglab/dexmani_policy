from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from dexmani_policy.common.rgb_preprocessing import (
    preprocess_training_rgb,
    preprocess_validation_rgb,
    validate_rgb_preprocessing,
)
from dexmani_policy.common.runtime import dict_apply, ensure_tensor
from dexmani_policy.datasets.common.augmentation import (
    AUGMENTATION_SPECS,
    random_sample_points,
    validate_pc_color_augmentation,
    validate_random_keep_points,
)
from dexmani_policy.datasets.common.episode_split import (
    downsample_mask,
    get_val_mask,
    validate_split_manifest,
    validate_split_options,
)
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sequence_sampler import SequenceSampler


class BaseDataset(torch.utils.data.Dataset):
    DEFAULT_MODALITIES: ClassVar[list[str]] = ["joint_state"]

    def __init__(
        self,
        zarr_path: str,
        seed: int = 42,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        sensor_modalities: list[str] | None = None,
        augmentation_cfg: dict | None = None,
        action_key: str = "action",
        use_aux_ee: bool = False,
        obs_horizon: int | None = None,
        rgb_preprocess_size: tuple[int, int] | None = None,
        rgb_random_crop_size: tuple[int, int] | None = None,
        rgb_color_aug: Callable | None = None,
        rgb_keep_uint8: bool = False,
        split_manifest: str | None = None,
        saved_split: dict | None = None,
        pc_dim: int | None = None,
    ) -> None:
        super().__init__()

        validate_split_options(
            val_ratio,
            max_train_episodes,
            explicit=split_manifest is not None or saved_split is not None,
        )

        if sensor_modalities is None:
            sensor_modalities = self.DEFAULT_MODALITIES
        if "rgb" in sensor_modalities:
            validate_rgb_preprocessing(rgb_preprocess_size, rgb_random_crop_size, rgb_color_aug)

        self.zarr_path = str(Path(zarr_path).expanduser().resolve())
        self.action_key = action_key
        self.use_aux_ee = use_aux_ee
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

        load_keys = list(sensor_modalities) + [action_key]
        if use_aux_ee:
            load_keys.append("action_ee")

        self.replay_buffer = ReplayBuffer.open(
            self.zarr_path,
            keys=load_keys,
        )

        if "rgb" in sensor_modalities:
            rgb = self.replay_buffer["rgb"]
            if len(rgb.shape) != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Stored RGB must have shape (T, H, W, 3) and dtype uint8")
        if "point_cloud" in sensor_modalities and pc_dim is not None:
            shape = self.replay_buffer["point_cloud"].shape
            if len(shape) != 3 or shape[-1] < pc_dim:
                raise ValueError("Stored point_cloud does not satisfy configured pc_dim")
        self.sensor_modalities = sensor_modalities
        self.augmentation_cfg = augmentation_cfg
        self.build_augmentors()

        self.observation_valid = np.ones(int(self.replay_buffer.episode_ends[-1]), dtype=bool)
        for key in sensor_modalities:
            self.observation_valid &= self.finite_rows(key)
        self.action_valid = self.finite_rows(action_key)
        if use_aux_ee:
            self.action_valid &= self.finite_rows("action_ee", slice(0, 9))
        self.role_lengths = {key: self.obs_horizon for key in sensor_modalities}
        self.role_lengths[action_key] = horizon
        if use_aux_ee:
            self.role_lengths["action_ee"] = horizon

        self.data_recipe = {}
        if "point_cloud" in sensor_modalities and self.pc_dim == 3:
            self.data_recipe.update(point_cloud_channels=3)
        elif (
            "point_cloud" in sensor_modalities
            and self.pc_dim is not None
            and self.pc_dim < self.replay_buffer["point_cloud"].shape[-1]
        ):
            self.data_recipe.update(point_cloud_channels=self.pc_dim)
        if saved_split is not None or split_manifest is not None:
            manifest = (
                saved_split
                if saved_split is not None
                else json.loads(Path(split_manifest).read_text())
            )
            train_mask, val_mask, manifest = validate_split_manifest(
                manifest, self.replay_buffer.attrs, self.replay_buffer.n_episodes
            )
            self.data_recipe["split_manifest"] = manifest
        else:
            val_mask = get_val_mask(
                seed=seed, val_ratio=val_ratio, n_episodes=self.replay_buffer.n_episodes
            )
            train_mask = downsample_mask(seed=seed, mask=~val_mask, max_n=max_train_episodes)
        self.val_mask = val_mask
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.filter_sampler(self.sampler)
        self.validation_dataset = None

    def finite_rows(self, key, columns=None):
        valid = np.ones(int(self.replay_buffer.episode_ends[-1]), dtype=bool)
        if np.issubdtype(self.replay_buffer[key].dtype, np.floating):
            for start, values in self.replay_buffer.iter_chunks(key, columns=columns):
                if key == "point_cloud" and self.pc_dim is not None:
                    values = values[..., : self.pc_dim]
                valid[start : start + len(values)] = (
                    np.isfinite(values).reshape(len(values), -1).all(axis=1)
                )
        return valid

    def filter_sampler(self, sampler):
        sampler.filter_valid(
            self.observation_valid,
            self.action_valid,
            self.obs_horizon,
        )

    def get_validation_dataset(self):
        if self.is_validation:
            return self
        if self.validation_dataset is not None:
            return self.validation_dataset
        if not self.val_mask.any():
            return None

        val_set = copy.copy(self)
        val_set.data_recipe = copy.deepcopy(self.data_recipe)

        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        self.filter_sampler(val_set.sampler)
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        val_set.random_keep_points = None
        val_set.is_validation = True
        self.validation_dataset = val_set
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        if "point_cloud" in sample and self.pc_dim is not None:
            sample["point_cloud"] = sample["point_cloud"][..., : self.pc_dim]
        action = sample[self.action_key]
        if self.use_aux_ee:
            # Auxiliary wrist pose is position (3) followed by rotation (6D).
            action = np.concatenate((action, sample["action_ee"][..., :9]), axis=-1)
        return {
            "obs": {m: sample[m] for m in self.sensor_modalities},
            "action": action,
        }

    def preprocess_rgb(self, rgb):
        if self.is_validation:
            return preprocess_validation_rgb(
                rgb,
                resize_hw=self.rgb_preprocess_size,
                center_crop_hw=self.rgb_random_crop_size,
                keep_uint8=self.rgb_keep_uint8,
                float_spatial_before_uint8=self.rgb_color_aug is not None,
            )
        return preprocess_training_rgb(
            rgb,
            resize_hw=self.rgb_preprocess_size,
            random_crop_hw=self.rgb_random_crop_size,
            keep_uint8=self.rgb_keep_uint8,
            color_augmentation=self.rgb_color_aug,
        )

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx, key_lengths=self.role_lengths)
        data = self.sample_to_data(sample)
        self.apply_augmentation(data)
        if self.random_keep_points is not None:
            data["obs"]["point_cloud"] = random_sample_points(
                data["obs"]["point_cloud"],
                self.random_keep_points,
            )

        if self.rgb_preprocess_size is not None and "rgb" in data["obs"]:
            data["obs"]["rgb"] = self.preprocess_rgb(data["obs"]["rgb"])

        return dict_apply(data, ensure_tensor)

    def build_augmentors(self):
        """Build NumPy transforms; RGB augmentation runs during preprocessing."""
        self.augmentors = {}
        pc_cfg = (self.augmentation_cfg or {}).get("pc") or {}
        self.random_keep_points = pc_cfg.get("random_keep_points")
        pc_dim = None
        if "point_cloud" in self.sensor_modalities:
            pc_dim = self.pc_dim or self.replay_buffer["point_cloud"].shape[-1]
            validate_pc_color_augmentation(pc_dim, self.augmentation_cfg)
        if self.random_keep_points is not None:
            if "point_cloud" not in self.sensor_modalities:
                raise ValueError("random_keep_points requires the point_cloud modality")
            shape = self.replay_buffer["point_cloud"].shape
            if len(shape) != 3:
                raise ValueError("Stored point_cloud must have shape [T,N,C]")
            validate_random_keep_points(self.random_keep_points, num_points=shape[1])
        for section, cls, key, modality in AUGMENTATION_SPECS:
            if modality not in self.sensor_modalities:
                continue
            config = ((self.augmentation_cfg or {}).get(section) or {}).get(key)
            if config:
                if section == "pc" and key in {"color", "color_noise"} and pc_dim < 6:
                    continue
                self.augmentors.setdefault(modality, []).append(cls(**config))

    def apply_augmentation(self, data):
        """ReplayBuffer reads own their storage, so augmentation can run in place."""
        for modality, augs in self.augmentors.items():
            x = data["obs"][modality]
            for aug in augs:
                if np.random.random() <= aug.prob:
                    aug.augment(x)

    def iter_normalization_data(self, key: str):
        """Stream unaugmented fields from the complete loaded replay buffer."""
        if self.is_validation:
            raise ValueError("Validation must reuse the training normalizer")
        if key != "action" and key not in self.sensor_modalities:
            raise KeyError(f"Not an enabled observation field: {key}")
        action = key == "action"
        field = self.action_key if action else key
        num_rows = int(self.replay_buffer.episode_ends[-1])
        step = self.replay_buffer.chunk_rows(field)
        if action and self.use_aux_ee:
            step = min(step, self.replay_buffer.chunk_rows("action_ee"))
        for start in range(0, num_rows, step):
            end = min(start + step, num_rows)
            values = self.replay_buffer.read(field, slice(start, end), cache=False)
            if field == "point_cloud" and self.pc_dim is not None:
                values = values[..., : self.pc_dim]
            if action and self.use_aux_ee:
                aux = self.replay_buffer.read(
                    "action_ee", slice(start, end), columns=slice(0, 9), cache=False
                )
                values = np.concatenate((values, aux), axis=-1)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nonfinite normalization field {key!r} from {self.zarr_path}, "
                    f"source {field!r}, rows [{start}, {end})"
                )
            yield values
