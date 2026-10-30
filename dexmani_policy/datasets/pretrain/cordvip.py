import copy

import torch

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.split import split_episodes


class CordViPPretrainDataset(torch.utils.data.Dataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, *, seed=42, val_ratio=0.1,
                 max_train_episodes=None, split_manifest: dict | None = None, horizon=12,
                 obs_horizon=4, pad_before=3, pad_after=5):
        if not 1 <= obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        self.episode_store = EpisodeStore(zarr_path, keys=(*self.sensor_modalities, "contact_map"))
        self.horizon, self.obs_horizon = horizon, obs_horizon
        self.pad_before, self.pad_after = pad_before, pad_after
        for key in self.sensor_modalities[:2]:
            shape = self.episode_store[key].shape
            if len(shape) != 3 or shape[1] < 1 or shape[2] < 3:
                raise ValueError(f"CordViP requires {key}[T,N>0,C>=3]")
        if self.episode_store["object_pointcloud"].shape[1] != 1024:
            raise ValueError("CordViP contact head predicts 1024 object point labels")
        if self.episode_store["joint_state"].shape[1:] != (19,):
            raise ValueError("CordViP requires arm7+hand12 joint_state")
        if self.episode_store["contact_map"].shape[1:] != (1024,):
            raise ValueError("Contact labels must match the 1024 object point slots")
        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.episode_store, manifest=split_manifest, seed=seed,
            val_ratio=val_ratio, max_train_episodes=max_train_episodes,
        )
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No CordViP training windows")
        self._normalizer: LinearNormalizer | None = None
        self.training_dataset: CordViPPretrainDataset | None = None

    def build_sampler(self, mask):
        fields = {"hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3),
                  "joint_state": None, "contact_map": None}
        return SequenceSampler(
            self.episode_store, self.horizon, self.pad_before, self.pad_after, mask,
            fields=fields, field_horizons={key: self.obs_horizon for key in fields},
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        return {
            "obs": {key: torch.as_tensor(sample[key]) for key in self.sensor_modalities},
            "contact_map": torch.as_tensor(sample["contact_map"]),
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
                columns = self.sampler.fields[key]
                chunks = (values for _, values in self.episode_store.iter_chunks(
                    key, columns=columns, episode_mask=self.train_mask,
                ))
                normalizer.fit_field(key, chunks, mode=mode)
            self._normalizer = normalizer
        return self._normalizer

    def get_validation_dataset(self):
        if self.training_dataset is not None:
            return self
        if not self.val_mask.any():
            return None
        result = copy.copy(self)
        result.training_dataset = self
        result.sampler = self.build_sampler(self.val_mask)
        return result if len(result) else None
