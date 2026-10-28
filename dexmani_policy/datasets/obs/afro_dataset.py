import copy

import torch

from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.validation import validate_fields


class AFRODataset(torch.utils.data.Dataset):
    sensor_modalities = ("point_cloud",)

    def __init__(self, zarr_path, *, seed=42, val_ratio=0.1, max_train_episodes=None,
                 split_manifest=None, horizon=33, pad_before=0, pad_after=7):
        if (horizon, pad_before, pad_after) != (33, 0, 7):
            raise ValueError("AFRO pretraining requires horizon=33 and padding=0/7")
        self.replay_buffer = ReplayBuffer(zarr_path, keys=self.sensor_modalities)
        self.zarr_path = self.replay_buffer.path
        shape = self.replay_buffer["point_cloud"].shape
        if len(shape) != 3 or shape[1] != 1024 or shape[2] < 3:
            raise ValueError("AFRO requires point_cloud[T,1024,C>=3]")
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.replay_buffer, manifest=split_manifest, seed=seed,
            val_ratio=val_ratio, max_train_episodes=max_train_episodes,
        )
        validate_fields(self.replay_buffer, self.train_mask | self.val_mask, {"point_cloud": slice(0, 3)})
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No AFRO training windows")
        self.training_dataset: AFRODataset | None = None
        self._normalizer = None

    def build_sampler(self, mask):
        return SequenceSampler(self.replay_buffer, self.horizon, self.pad_before, self.pad_after, mask)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        return {"obs": {"point_cloud": torch.as_tensor(sample["point_cloud"][..., :3])}}

    def get_normalizer(self, spec):
        if self.training_dataset is not None:
            return self.training_dataset.get_normalizer(spec)
        if self._normalizer is None:
            if set(spec) != set(self.sensor_modalities):
                raise ValueError("AFRO normalization must declare only point_cloud")
            normalizer = LinearNormalizer()
            if spec["point_cloud"] != "identity":
                chunks = (values for _, values in self.replay_buffer.iter_chunks(
                    "point_cloud", columns=slice(0, 3), episode_mask=self.train_mask,
                ))
                normalizer.fit_field("point_cloud", chunks, mode=spec["point_cloud"])
            self._normalizer = normalizer
        return self._normalizer

    def get_validation_dataset(self):
        if not self.val_mask.any():
            return None
        result = copy.copy(self)
        result.sampler = self.build_sampler(self.val_mask)
        result.training_dataset = self
        return result if len(result) else None
