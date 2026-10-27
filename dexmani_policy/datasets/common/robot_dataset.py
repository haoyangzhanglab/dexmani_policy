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
from dexmani_policy.common.config import FINGER_VECTOR_FIELDS, positive_int
from dexmani_policy.datasets.common.augmentation import (
    AUGMENTATION_SPECS,
    random_sample_points,
)
from dexmani_policy.datasets.common.metadata import describe_hand_data, describe_real_runtime
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler


class RobotDataset(torch.utils.data.Dataset):

    action_normalization_population = "valid_train_windows_unique_source_rows"
    future_action_window = False

    @property
    def action_offset(self):
        return 0

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
        use_aux_ee: bool = False,
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

        self.replay_buffer = ReplayBuffer(zarr_path, keys=load_keys)
        self.zarr_path = self.replay_buffer.path

        if "rgb" in sensor_modalities:
            rgb = self.replay_buffer["rgb"]
            if len(rgb.shape) != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Stored RGB must have shape (T, H, W, 3) and dtype uint8")
        if "point_cloud" in sensor_modalities:
            shape = self.replay_buffer["point_cloud"].shape
            if len(shape) != 3:
                raise ValueError("Stored point_cloud must have shape [T,N,C]")
            if pc_dim is not None and shape[-1] < pc_dim:
                raise ValueError("Stored point_cloud does not satisfy configured pc_dim")
        self.sensor_modalities = sensor_modalities
        self.augmentation_cfg = augmentation_cfg
        self.build_augmentors()

        self.data_recipe: dict[str, Any] = {}
        if (
            "point_cloud" in sensor_modalities
            and self.pc_dim is not None
            and (self.pc_dim == 3 or self.pc_dim < self.replay_buffer["point_cloud"].shape[-1])
        ):
            self.data_recipe.update(point_cloud_channels=self.pc_dim)
        train_mask, val_mask, manifest = split_episodes(
            self.replay_buffer, manifest=split_manifest, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
        )
        self.data_recipe["split_manifest"] = manifest
        self.val_mask = val_mask
        self.train_mask = train_mask

        self.observation_valid = np.ones(int(self.replay_buffer.episode_ends[-1]), dtype=bool)
        for key in sensor_modalities:
            self.observation_valid &= self.finite_rows(key)
        self.action_valid = self.finite_rows(action_key)
        if use_aux_ee:
            self.action_valid &= self.finite_rows("action_ee", slice(0, 9))
        self.role_lengths = {key: self.obs_horizon for key in sensor_modalities}
        self.role_lengths[action_key] = horizon
        self.role_offsets = {action_key: self.action_offset}
        if use_aux_ee:
            self.role_lengths["action_ee"] = horizon
            self.role_offsets["action_ee"] = self.action_offset

        self.sampler = self.build_sampler(train_mask)
        self.validation_dataset: RobotDataset | None = None

    def describe_training_data(
        self, spec, *, task_name, requires_hand_pretraining=False,
    ):
        buffer = self.replay_buffer
        widths = {
            key: int(buffer[key].shape[-1])
            for key, mode in spec.items() if key != "action" and mode != "identity"
        }
        for key in FINGER_VECTOR_FIELDS.intersection(self.sensor_modalities):
            shape = buffer[key].shape
            if shape[1:] != (5, 3):
                raise ValueError(f"{key} requires stored shape (T, 5, 3), got {shape}")
            if key in widths:
                widths[key] = 15
        if "point_cloud" in widths and self.pc_dim is not None:
            widths["point_cloud"] = self.pc_dim
        widths["action"] = int(buffer[self.action_key].shape[-1]) + (9 if self.use_aux_ee else 0)
        description: dict[str, Any] = {
            "normalization_dims": widths,
            "data_recipe": [dict(
                self.data_recipe,
                windows=self.sampler.validity_summary,
                observation_rows=len(self.sampler.observation_source_rows),
                action_rows=len(self.sampler.action_source_rows),
                normalization={
                    "observations": "valid_train_windows_unique_source_rows",
                    "action": self.action_normalization_population,
                },
            )],
        }
        # Episode seeds are acquisition facts, not a hash or proof of scene independence.
        meta = buffer.group['meta']
        seeds = None
        if 'episode_seeds' in meta:
            values = np.asarray(meta['episode_seeds'])
            if values.shape != buffer.episode_ends.shape or values.dtype.kind not in 'iu':
                raise ValueError('episode_seeds must be integer metadata aligned with episodes')
            seeds = values.tolist()
        description['data_recipe'][0]['source_evidence'] = {
            'task_name': buffer.attrs.get('task_name'), 'episode_seeds': seeds,
        }
        runtime = describe_real_runtime(self, task_name)
        if runtime is not None:
            description["real_runtime"] = runtime
        if requires_hand_pretraining:
            description["hand_data"] = self.describe_hand_data()
        return description

    def training_summary(self):
        counts = self.sampler.validity_summary
        summary = (
            f"{Path(self.zarr_path).name}: "
            f"train episodes={int(self.train_mask.sum()):,}, windows={len(self):,}"
        )
        rejected = counts["candidate"] - counts["valid"]
        return summary + (f", rejected={rejected:,}" if rejected else "")

    def describe_hand_data(self):
        return describe_hand_data(self)

    def finite_rows(self, key, columns=None):
        valid = np.repeat(self.train_mask | self.val_mask,
                          np.diff(np.r_[0, self.replay_buffer.episode_ends]))
        if np.issubdtype(self.replay_buffer[key].dtype, np.floating):
            for start, values in self.replay_buffer.iter_chunks(
                key, columns=columns, episode_mask=self.train_mask | self.val_mask
            ):
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
            action_offset=self.action_offset,
        )

    def build_sampler(self, episode_mask):
        sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.action_offset,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
        )
        self.filter_sampler(sampler)
        return sampler

    def get_validation_dataset(self):
        if self.is_validation:
            return self
        if self.validation_dataset is not None:
            return self.validation_dataset
        if not self.val_mask.any():
            return None

        val_set = copy.copy(self)
        val_set.data_recipe = copy.deepcopy(self.data_recipe)

        val_set.sampler = self.build_sampler(self.val_mask)
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
        sample = self.sampler.sample_sequence(
            idx, key_lengths=self.role_lengths, key_offsets=self.role_offsets
        )
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

    def iter_normalization_data(self, key: str):
        """Stream only source rows consumed by valid training windows, once per field."""
        if self.is_validation:
            raise ValueError("Validation must reuse the training normalizer")
        if key != "action" and key not in self.sensor_modalities:
            raise KeyError(f"Not an enabled observation field: {key}")
        action = key == "action"
        field = self.action_key if action else key
        step = self.replay_buffer.chunk_rows(field)
        if action and self.use_aux_ee:
            step = min(step, self.replay_buffer.chunk_rows("action_ee"))
        rows = self.sampler.action_source_rows if action else self.sampler.observation_source_rows
        for block in np.unique(rows // step):
            start = int(block * step)
            selected = rows[np.searchsorted(rows, start) : np.searchsorted(rows, start + step)]
            end = int(selected[-1]) + 1
            values = self.replay_buffer.read(field, slice(start, end))[selected - start]
            if field == "point_cloud" and self.pc_dim is not None:
                values = values[..., : self.pc_dim]
            if action and self.use_aux_ee:
                aux = self.replay_buffer.read(
                    "action_ee", slice(start, end), columns=slice(0, 9)
                )[selected - start]
                values = np.concatenate((values, aux), axis=-1)
            yield values
