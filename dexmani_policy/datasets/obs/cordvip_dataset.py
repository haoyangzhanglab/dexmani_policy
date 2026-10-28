import copy

import numpy as np
import torch

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.geometry import contact_source_metadata
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.validation import validate_fields


class CordViPDataset(torch.utils.data.Dataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, contact_map_path, *, seed=42, val_ratio=0.1,
                 max_train_episodes=None, split_manifest=None, horizon=12,
                 obs_horizon=4, pad_before=3, pad_after=5):
        if (horizon, obs_horizon, pad_before, pad_after) != (12, 4, 3, 5):
            raise ValueError("CordViP pretraining requires horizon=12, obs_horizon=4 and padding=3/5")
        self.replay_buffer = ReplayBuffer(zarr_path, keys=self.sensor_modalities)
        self.contacts = ReplayBuffer(contact_map_path, keys=["contact_map"])
        self.zarr_path = self.replay_buffer.path
        self.horizon, self.obs_horizon = horizon, obs_horizon
        self.pad_before, self.pad_after = pad_before, pad_after
        for key in self.sensor_modalities[:2]:
            shape = self.replay_buffer[key].shape
            if len(shape) != 3 or shape[2] < 3:
                raise ValueError(f"CordViP requires {key}[T,N,C>=3]")
        if self.replay_buffer["object_pointcloud"].shape[1] != 1024:
            raise ValueError("CordViP contact head predicts 1024 object point labels")
        if self.replay_buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("CordViP requires arm7+hand12 joint_state")
        if self.contacts["contact_map"].shape != (int(self.replay_buffer.episode_ends[-1]), 1024):
            raise ValueError("Contact labels must match source rows and object point slots")
        if not np.array_equal(self.contacts.episode_ends, self.replay_buffer.episode_ends):
            raise ValueError("Contact episode boundaries differ from the source")
        source = contact_source_metadata(self.replay_buffer)
        if self.contacts.attrs["source"] != source:
            raise ValueError("Contact source identity or geometry/point-order recipe mismatch")
        if self.contacts.attrs["recipe"] != {"gamma": 1.0, "theta": 10.0, "knn": 30}:
            raise ValueError("Contact labels require gamma=1, theta=10, KNN=30")
        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.replay_buffer, manifest=split_manifest, seed=seed,
            val_ratio=val_ratio, max_train_episodes=max_train_episodes,
        )
        mask = self.train_mask | self.val_mask
        fields = {"hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None}
        validate_fields(self.replay_buffer, mask, fields)
        validate_fields(self.contacts, mask, {"contact_map": None})
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No CordViP training windows")
        self.training_dataset: CordViPDataset | None = None
        self._normalizer = None

    def build_sampler(self, mask):
        return SequenceSampler(
            self.replay_buffer, self.horizon, self.pad_before, self.pad_after, mask,
            key_first_k={key: self.obs_horizon for key in self.sensor_modalities},
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        for key in self.sensor_modalities[:2]:
            sample[key] = sample[key][..., :3]
        rows = self.sampler.sample_source_rows(index)[:self.obs_horizon]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        contact = self.contacts.read("contact_map", slice(start, stop))[rows - start]
        return {
            "obs": {key: torch.as_tensor(value) for key, value in sample.items()},
            "contact_map": torch.as_tensor(contact),
        }

    def get_normalizer(self, spec):
        if self.training_dataset is not None:
            return self.training_dataset.get_normalizer(spec)
        if self._normalizer is None:
            if set(spec) != set(self.sensor_modalities):
                raise ValueError("CordViP normalization must declare its three observation fields")
            normalizer = LinearNormalizer()
            for key, mode in spec.items():
                if mode == "identity":
                    continue
                columns = None if key == "joint_state" else slice(0, 3)
                chunks = (values for _, values in self.replay_buffer.iter_chunks(
                    key, columns=columns, episode_mask=self.train_mask,
                ))
                normalizer.fit_field(key, chunks, mode=mode)
            self._normalizer = normalizer
        return self._normalizer

    def get_validation_dataset(self):
        if not self.val_mask.any():
            return None
        result = copy.copy(self)
        result.sampler = self.build_sampler(self.val_mask)
        result.training_dataset = self
        return result if len(result) else None
