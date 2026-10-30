import copy

import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.split import split_episodes


class AFROPretrainDataset(torch.utils.data.Dataset):
    sensor_modalities = ("point_cloud",)

    def __init__(self, zarr_path, *, frame_stride, seed=42, val_ratio=0.1, max_train_episodes=None,
                 split_manifest: dict | None = None, horizon=33, pad_before=0, pad_after=7):
        self.frame_stride = positive_int(frame_stride, "frame_stride")
        if len(range(0, horizon, self.frame_stride)) < 2:
            raise ValueError("AFRO frame_stride must leave at least two sampled frames")
        self.episode_store = EpisodeStore(zarr_path, keys=self.sensor_modalities)
        shape = self.episode_store["point_cloud"].shape
        if len(shape) != 3 or shape[2] < 3:
            raise ValueError("AFRO requires point_cloud[T,N,C>=3]")
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.episode_store, manifest=split_manifest, seed=seed,
            val_ratio=val_ratio, max_train_episodes=max_train_episodes,
        )
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No AFRO training windows")
        self._normalizer: LinearNormalizer | None = None
        self.training_dataset: AFROPretrainDataset | None = None

    def build_sampler(self, mask):
        return SequenceSampler(self.episode_store, self.horizon, self.pad_before, self.pad_after, mask)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        rows = self.sampler.sample_source_rows(index)[::self.frame_stride]
        points = self.episode_store.read("point_cloud", rows, columns=slice(0, 3))
        return {"obs": {"point_cloud": torch.as_tensor(points)}}

    def get_normalizer(self, spec):
        if self.training_dataset is not None:
            return self.training_dataset.get_normalizer(spec)
        if self._normalizer is None:
            if set(spec) != set(self.sensor_modalities):
                raise ValueError("AFRO normalization must declare only point_cloud")
            normalizer = LinearNormalizer()
            if spec["point_cloud"] != "identity":
                chunks = (values for _, values in self.episode_store.iter_chunks(
                    "point_cloud", columns=slice(0, 3), episode_mask=self.train_mask,
                ))
                normalizer.fit_field("point_cloud", chunks, mode=spec["point_cloud"])
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
