from __future__ import annotations

import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torchvision.transforms.functional as TVF

from dexmani_policy.datasets.augmentation import (
    PointColorJitter,
    PointColorNoiseAug,
    PointCoordNoiseAug,
    PointDropout,
    StateNoiseAug,
)
from dexmani_policy.datasets.preprocessing import preprocess_validation_rgb, raw_rgb_tensor
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
    validate_max_train_episodes,
    validate_val_ratio,
)
from dexmani_policy.utils.tensor import dict_apply, ensure_tensor

# (yaml_section, augmentor_class, yaml_key, output_modality)
# Point-cloud/state transforms run on NumPy arrays in this order.
# Image augmentation runs separately on tensors in _preprocess_rgb_cpu.
AUGMENTOR_REGISTRY = [
    ("pc", PointCoordNoiseAug, "coord_noise", "point_cloud"),
    ("pc", PointColorJitter, "color", "point_cloud"),
    ("pc", PointColorNoiseAug, "color_noise", "point_cloud"),
    ("pc", PointDropout, "dropout", "point_cloud"),
    ("state", StateNoiseAug, "noise", "joint_state"),
]


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
        rgb_color_aug: dict | None = None,
        rgb_keep_uint8: bool = False,
        split_manifest: str | None = None,
        saved_split: dict | None = None,
    ) -> None:
        super().__init__()

        validate_val_ratio(val_ratio)
        validate_max_train_episodes(max_train_episodes)
        if ((split_manifest is not None or saved_split is not None)
                and (max_train_episodes is not None or val_ratio != 0)):
            raise ValueError("Explicit split_manifest defines final IDs: set max_train_episodes=null "
                             "and val_ratio=0; prepare a subset manifest for a smaller budget")

        if sensor_modalities is None:
            sensor_modalities = self.DEFAULT_MODALITIES

        self.zarr_path = str(Path(zarr_path).expanduser().resolve())
        self.action_key = action_key
        self.use_aux_ee = use_aux_ee
        self.obs_horizon = horizon if obs_horizon is None else obs_horizon
        if not 1 <= self.obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        self.rgb_preprocess_size = rgb_preprocess_size
        self.rgb_random_crop_size = rgb_random_crop_size
        self.rgb_color_aug = rgb_color_aug
        self.rgb_keep_uint8 = rgb_keep_uint8
        self._is_val = False

        # When EE auxiliary loss is enabled, load action_ee for wrist pose (pos3+rot6d6).
        load_keys = list(sensor_modalities) + [action_key]
        if use_aux_ee:
            load_keys = load_keys + ["action_ee"]

        self.replay_buffer = ReplayBuffer.open(
            self.zarr_path,
            keys=load_keys,
        )

        self.data_revision = self.replay_buffer.data_revision
        n_rows = int(self.replay_buffer.episode_ends[-1])

        def finite_rows(key, columns=None):
            valid = np.ones(n_rows, dtype=bool)
            if np.issubdtype(self.replay_buffer[key].dtype, np.floating):
                for start, values in self.replay_buffer.iter_chunks(
                    key, columns=columns
                ):
                    valid[start : start + len(values)] = (
                        np.isfinite(values).reshape(len(values), -1).all(axis=1)
                    )
            return valid

        self._obs_valid = np.ones(n_rows, dtype=bool)
        for key in sensor_modalities:
            self._obs_valid &= finite_rows(key)
        self._action_valid = finite_rows(action_key)
        if use_aux_ee:
            self._action_valid &= finite_rows("action_ee", slice(0, 9))
        self._role_lengths = {key: self.obs_horizon for key in sensor_modalities}
        self._role_lengths[action_key] = horizon
        if use_aux_ee:
            self._role_lengths["action_ee"] = horizon
        self.data_recipe = {
            "window_validity": "role_finite_v1",
            "normalization": "unique_train_source_rows",
        }
        self.sensor_modalities = sensor_modalities
        self.augmentation_cfg = augmentation_cfg
        self.augmentors = {}
        if augmentation_cfg is not None:
            self._build_augmentors()

        if saved_split is not None:
            from dexmani_policy.datasets.split import restore_split
            train_mask, val_mask, manifest, digest = restore_split(
                saved_split, self.replay_buffer.attrs, self.replay_buffer.n_episodes
            )
        elif split_manifest is None:
            val_mask = get_val_mask(
                seed=seed, val_ratio=val_ratio, n_episodes=self.replay_buffer.n_episodes
            )
            train_mask = downsample_mask(seed=seed, mask=~val_mask, max_n=max_train_episodes)
        else:
            from dexmani_policy.datasets.split import load_split_manifest

            train_mask, val_mask, manifest, digest = load_split_manifest(
                split_manifest, self.replay_buffer.attrs, self.replay_buffer.n_episodes
            )
        self.val_mask = val_mask
        self.train_mask = train_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self._filter_sampler(self.sampler)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._validation_dataset = None
        if split_manifest is not None or saved_split is not None:
            ids, trials = manifest["episode_ids"], manifest["trial_ids"]
            actual_train = [ids[i] for i in np.flatnonzero(train_mask)]
            self.data_recipe["split_manifest"] = {
                "content": manifest,
                "sha256": digest,
                "train_mask": train_mask.tolist(),
                "val_mask": val_mask.tolist(),
                "actual_train_ids": actual_train,
                "train_episodes": len(actual_train),
                "val_episodes": int(val_mask.sum()),
                "train_trials": len({trials[i] for i in actual_train}),
                "val_trials": len({trials[i] for i in manifest["val_ids"]}),
                "train_windows": len(self),
                "holdout": bool(val_mask.any()),
            }

    def _filter_sampler(self, sampler):
        sampler.filter_valid(
            self._obs_valid, self._action_valid, self.obs_horizon,
        )

    def get_validation_dataset(self):
        if self._validation_dataset is not None:
            return self._validation_dataset
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
        self._filter_sampler(val_set.sampler)
        # Validation set disables randomness: no augmentation, random crop → center crop
        val_set.augmentation_cfg = None
        val_set.augmentors = {}
        val_set._is_val = True
        self._validation_dataset = val_set
        return val_set

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        # Build action by concatenating enabled auxiliary modalities.
        # EE aux loss: wrist pose = action_ee[..., :9] = pos(3) + rot6d(6).
        parts = [sample[self.action_key]]
        if self.use_aux_ee:
            parts.append(sample["action_ee"][..., :9])
        action = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        return {
            "obs": {m: sample[m][: self.obs_horizon] for m in self.sensor_modalities},
            "action": action,
        }

    def _preprocess_rgb_cpu(self, rgb_np):
        """Convert HWC uint8 frames to CHW RGB using the dataset recipe.

        Training applies resize, one shared random crop, then color augmentation.
        Validation uses deterministic resize/center crop without augmentation.
        Backbone-specific normalization is left to the Agent's vision path.

        Transport paths:
        - uint8 fast path: when ``rgb_keep_uint8=True`` and no color aug,
          resize/crop keep uint8 output (one byte per channel).
        - float32 spatial/color path (default); with ``rgb_keep_uint8=True``,
          quantize only the final clamped result for transport.
        """
        if self._is_val:
            return preprocess_validation_rgb(
                rgb_np,
                resize_hw=self.rgb_preprocess_size,
                center_crop_hw=self.rgb_random_crop_size,
                keep_uint8=self.rgb_keep_uint8,
                float_spatial_before_uint8=self.rgb_color_aug is not None,
            )

        rgb = raw_rgb_tensor(rgb_np, ndim=4)

        # uint8 fast path: skip float32 conversion, resize/crop in uint8
        if self.rgb_keep_uint8 and self.rgb_color_aug is None:
            rgb = rgb.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W) uint8
            rgb = TVF.resize(rgb, list(self.rgb_preprocess_size), antialias=True)
            if self.rgb_random_crop_size is not None:
                rgb = TVF.crop(
                    rgb,
                    top=torch.randint(
                        0, rgb.shape[-2] - self.rgb_random_crop_size[0] + 1, (1,)
                    ).item(),
                    left=torch.randint(
                        0, rgb.shape[-1] - self.rgb_random_crop_size[1] + 1, (1,)
                    ).item(),
                    height=self.rgb_random_crop_size[0],
                    width=self.rgb_random_crop_size[1],
                )
            return rgb.contiguous()  # uint8

        # Convert contiguous channel-first data before float color augmentation.
        rgb = rgb.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W) uint8
        rgb = rgb.float().div_(255.0)  # (T, 3, H, W) float32 [0,1]
        rgb = TVF.resize(rgb, list(self.rgb_preprocess_size), antialias=True)
        if self.rgb_random_crop_size is not None:
            rgb = TVF.crop(
                rgb,
                top=torch.randint(
                    0, rgb.shape[-2] - self.rgb_random_crop_size[0] + 1, (1,)
                ).item(),
                left=torch.randint(
                    0, rgb.shape[-1] - self.rgb_random_crop_size[1] + 1, (1,)
                ).item(),
                height=self.rgb_random_crop_size[0],
                width=self.rgb_random_crop_size[1],
            )
        if self.rgb_color_aug is not None:
            rgb = self.rgb_color_aug(rgb)  # (T, 3, H_dst, W_dst) float32 [0,1]
        # Clamp owned float data before optional transport quantization.
        rgb = rgb.clamp_(0, 1)
        if self.rgb_keep_uint8:
            rgb = rgb.mul(255).round_().clamp_(0, 255).to(torch.uint8)
        return rgb

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx, key_lengths=self._role_lengths)
        data = self.sample_to_data(sample)
        data = self.apply_augmentation(data)

        if self.rgb_preprocess_size is not None and "rgb" in data["obs"]:
            data["obs"]["rgb"] = self._preprocess_rgb_cpu(data["obs"]["rgb"])

        data = dict_apply(data, ensure_tensor)
        return data

    def _build_augmentors(self):
        """Build augmentors from ``augmentation_cfg`` using AUGMENTOR_REGISTRY.

        Subclasses that need different augmentors (e.g. RGB-only) can override
        this method, but the registry covers all standard cases.
        """
        self.augmentors = {}
        if self.augmentation_cfg is None:
            return
        for section, cls, key, modality in AUGMENTOR_REGISTRY:
            config = (self.augmentation_cfg.get(section) or {}).get(key)
            if config:
                self.augmentors.setdefault(modality, []).append(cls(**config))

    def apply_augmentation(self, data):
        """Apply configured augmentors to the sample dict.

        Augmentors run in registry order: coordinate noise, color jitter,
        optional color noise, then point dropout. Each modality is copied once,
        when its first augmentor triggers; subsequent transforms modify that
        copy in-place. Policy normalization happens later in the Agent.
        """
        if self.augmentation_cfg is None:
            return data
        for modality, augs in self.augmentors.items():
            if modality not in data["obs"]:
                continue
            x = data["obs"][modality]
            copied = False
            for aug in augs:
                if np.random.random() <= aug.prob:
                    if not copied:
                        x = x.copy()
                        copied = True
                    aug._augment(x)
            data["obs"][modality] = x
        return data

    def iter_normalization_data(self, key: str):
        """Fit only unique source rows referenced by valid training windows, by role."""
        if self._is_val:
            raise ValueError("Validation must reuse the training normalizer")
        if key != "action" and key not in self.sensor_modalities:
            raise KeyError(f"Not an enabled observation field: {key}")
        action = key == "action"
        field = self.action_key if action else key
        rows = (
            self.sampler.action_source_rows
            if action
            else self.sampler.observation_source_rows
        )
        step = self.replay_buffer.chunk_rows(field)
        if action and self.use_aux_ee:
            step = min(step, self.replay_buffer.chunk_rows("action_ee"))
        for start in range(0, int(self.replay_buffer.episode_ends[-1]), step):
            end = min(start + step, int(self.replay_buffer.episode_ends[-1]))
            selected = (
                rows[np.searchsorted(rows, start) : np.searchsorted(rows, end)] - start
            )
            if not len(selected):
                continue
            values = self.replay_buffer.read(field, slice(start, end))[selected]
            if action and self.use_aux_ee:
                aux = self.replay_buffer.read(
                    "action_ee", slice(start, end), columns=slice(0, 9)
                )[selected]
                values = np.concatenate((values, aux), axis=-1)
            yield values
