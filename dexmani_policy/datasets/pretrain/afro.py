import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.common import ObservationPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler


class AFROPretrainDataset(ObservationPretrainDataset):
    sensor_modalities = ("point_cloud",)

    def __init__(self, zarr_path, *, frame_stride, seed=42, val_ratio=0.1, max_train_episodes=None,
                 split_manifest: dict | None = None, horizon=33, pad_before=0, pad_after=7):
        self.frame_stride = positive_int(frame_stride, "frame_stride")
        if len(range(0, horizon, self.frame_stride)) < 2:
            raise ValueError("AFRO frame_stride must leave at least two sampled frames")
        buffer = EpisodeStore(zarr_path, keys=self.sensor_modalities)
        shape = buffer["point_cloud"].shape
        if len(shape) != 3 or shape[2] < 3:
            raise ValueError("AFRO requires point_cloud[T,N,C>=3]")
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        super().__init__(
            buffer, normalization_fields={"point_cloud": slice(0, 3)},
            seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_manifest=split_manifest,
        )

    def build_sampler(self, mask):
        return SequenceSampler(self.episode_store, self.horizon, self.pad_before, self.pad_after, episode_mask=mask)

    def __getitem__(self, index):
        rows = self.sampler.sample_source_rows(index)[::self.frame_stride]
        points = self.episode_store.read("point_cloud", rows, columns=slice(0, 3))
        return {"obs": {"point_cloud": torch.as_tensor(points)}}
