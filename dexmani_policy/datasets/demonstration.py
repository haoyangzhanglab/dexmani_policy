from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from dexmani_policy.common.image_util import (
    preprocess_rgb_train,
    preprocess_rgb_eval,
    validate_rgb_preprocessing,
)
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.common.config import (
    FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, positive_int, require_tactile_source,
)
from dexmani_policy.datasets.transforms import (
    AUGMENTATION_SPECS,
    PointCoordinateNoise,
    random_sample_points,
)
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler


def validate_tactile_source(attrs, source):
    if source == "real":
        if attrs.get("format") != "dexmani.real.canonical":
            raise ValueError("tactile_source=real requires dexmani.real.canonical data")
    elif source == "sim":
        required = {"dt", "physics_dt", "frame_skip", "action_dim", "action_space", "obs_alignment"}
        missing = required - attrs.keys()
        if attrs.get("format") is not None or missing:
            raise ValueError(
                "tactile_source=sim requires sim exporter metadata; "
                f"format={attrs.get('format')!r}, missing={sorted(missing)}"
            )
        physics_dt, skip = attrs["physics_dt"], attrs["frame_skip"]
        if (
            type(skip) is not int or skip <= 0
            or type(physics_dt) not in (float, int) or not math.isfinite(physics_dt) or physics_dt <= 0
            or attrs["obs_alignment"] != "obs[t]_before_action[t]"
            or attrs["action_space"] not in {"joint", "ee"} or attrs["action_dim"] != 19
        ):
            raise ValueError("Invalid sim exporter tactile/timing contract")
    else:
        raise ValueError(f"tactile_source must be sim or real, got {source!r}")
    dt = attrs.get("dt")
    if type(dt) not in (float, int) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Tactile data requires finite positive dt")
    if source == "sim" and not math.isclose(dt, physics_dt * skip, rel_tol=1e-6, abs_tol=0):
        raise ValueError("Sim dt must equal physics_dt * frame_skip")


class DemonstrationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zarr_path: str,
        *,
        sensor_modalities: list[str],
        obs_horizon: int,
        seed: int = 42,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        augmentation_cfg: dict | None = None,
        action_key: str = "action",
        rgb_preprocess_size: tuple[int, int] | None = None,
        rgb_random_crop_size: tuple[int, int] | None = None,
        rgb_color_aug: Callable | None = None,
        rgb_keep_uint8: bool = False,
        split_manifest: dict | DictConfig | str | Path | None = None,
        pc_dim: int | None = None,
        tactile_source: str | None = None,
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
        self.rgb_keep_uint8 = rgb_keep_uint8
        self.training = True
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.sensor_modalities = sensor_modalities
        self.episode_store = EpisodeStore(zarr_path, keys=self.data_keys)
        self.zarr_path = self.episode_store.path
        if require_tactile_source(sensor_modalities, tactile_source) is not None:
            validate_tactile_source(self.episode_store.attrs, tactile_source)

        if "rgb" in sensor_modalities:
            rgb = self.episode_store["rgb"]
            if len(rgb.shape) != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Stored RGB must have shape (T, H, W, 3) and dtype uint8")
        for key in POINT_CLOUD_FIELDS.intersection(sensor_modalities):
            shape = self.episode_store[key].shape
            if len(shape) != 3 or shape[1] == 0:
                raise ValueError(f"Stored {key} must have shape [T,N>0,C]")
            if pc_dim is not None and shape[-1] < pc_dim:
                raise ValueError(f"Stored {key} does not satisfy configured pc_dim")
        self.augmentation_cfg = augmentation_cfg
        self.build_augmentors()
        if {"point_cloud", "rgb", "camera_intrinsic", "camera_extrinsic"} <= set(sensor_modalities):
            if rgb_preprocess_size is not None:
                raise ValueError("Calibrated RGB must stay raw in Dataset; resize in the encoder ImageProcessor")
            if any(
                isinstance(aug, PointCoordinateNoise) and aug.prob > 0 and aug.noise_std > 0
                for aug in self.augmentors.get("point_cloud", ())
            ):
                raise ValueError("Calibrated RGB/point clouds cannot use independent XYZ coordinate noise")

        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.episode_store, manifest=split_manifest, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
        )
        for key in FIELD_FEATURE_SHAPES.keys() & set(sensor_modalities):
            if self.episode_store[key].shape[1:] != FIELD_FEATURE_SHAPES[key]:
                raise ValueError(f"{key} requires trailing shape {FIELD_FEATURE_SHAPES[key]} after its T axis")
            if self.episode_store[key].dtype.kind != "f":
                raise ValueError(f"Stored {key} must be floating point; observations are read as FP32")
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
        return (
            f"{Path(self.zarr_path).name}: train episodes={int(self.train_mask.sum()):,}, "
            f"val episodes={int(self.val_mask.sum()):,}, train windows={len(self):,}, "
            f"fields={', '.join(self.data_keys)}"
        )

    def build_sampler(self, episode_mask):
        return SequenceSampler(
            self.episode_store, self.horizon, self.pad_before, self.pad_after, episode_mask=episode_mask,
            fields=self.consumed_fields, field_horizons={key: self.obs_horizon for key in self.sensor_modalities},
        )

    def get_validation_dataset(self):
        if not self.training:
            return self
        if not self.val_mask.any():
            return None

        val_set = copy.copy(self)
        val_set.training = False
        val_set.sampler = self.build_sampler(self.val_mask)
        if not len(val_set.sampler):
            return None
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        val_set.random_keep_points = None
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        action = sample[self.action_key]
        return {
            "obs": {m: sample[m] for m in self.sensor_modalities},
            "action": action,
        }

    def preprocess_rgb(self, rgb, resize_hw: tuple[int, int]):
        if not self.training:
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

        if "rgb" in data["obs"]:
            if self.rgb_preprocess_size is not None:
                data["obs"]["rgb"] = self.preprocess_rgb(data["obs"]["rgb"], self.rgb_preprocess_size)
            else:
                data["obs"]["rgb"] = preprocess_rgb_eval(
                    data["obs"]["rgb"], resize_hw=None, center_crop_hw=None, keep_uint8=self.rgb_keep_uint8,
                )

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
            shape = self.episode_store["point_cloud"].shape
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

    def apply_augmentation(self, data):
        """EpisodeStore reads own their storage, so augmentation can run in place."""
        for modality, augs in self.augmentors.items():
            x = data["obs"][modality]
            for aug in augs:
                if np.random.random() <= aug.prob:
                    aug.augment(x)

    def iter_normalization_data(self, key):
        field = self.action_key if key == "action" else key
        columns = self.consumed_fields[field]
        for _, values in self.episode_store.iter_chunks(field, columns=columns, episode_mask=self.train_mask):
            yield values
