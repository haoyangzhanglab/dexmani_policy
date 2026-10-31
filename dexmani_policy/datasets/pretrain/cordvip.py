from pathlib import Path

import torch
from omegaconf import DictConfig

from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler


class CordViPPretrainDataset(ObservationPretrainDataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, *, seed=42, val_ratio=0.1,
                 max_train_episodes=None, split_manifest: dict | DictConfig | str | Path | None = None, horizon=12,
                 obs_horizon=4, pad_before=3, pad_after=5):
        if not 1 <= obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        buffer = EpisodeStore(zarr_path, keys=(*self.sensor_modalities, "contact_map"))
        self.horizon, self.obs_horizon = horizon, obs_horizon
        self.pad_before, self.pad_after = pad_before, pad_after
        for key in self.sensor_modalities[:2]:
            shape = buffer[key].shape
            if len(shape) != 3 or shape[1] < 1 or shape[2] < 3:
                raise ValueError(f"CordViP requires {key}[T,N>0,C>=3]")
        if buffer["object_pointcloud"].shape[1] != 1024:
            raise ValueError("CordViP contact head predicts 1024 object point labels")
        if buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("CordViP requires arm7+hand12 joint_state")
        if buffer["contact_map"].shape[1:] != (1024,):
            raise ValueError("Contact labels must match the 1024 object point slots")
        super().__init__(
            buffer, normalization_fields={
                "hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None,
            }, seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_manifest=split_manifest,
        )

    def build_sampler(self, mask):
        fields = {**self.normalization_fields, "contact_map": None}
        return SequenceSampler(
            self.episode_store, self.horizon, self.pad_before, self.pad_after, episode_mask=mask,
            fields=fields, field_horizons={key: self.obs_horizon for key in fields},
        )

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        return {
            "obs": {key: torch.as_tensor(sample[key]) for key in self.sensor_modalities},
            "contact_map": torch.as_tensor(sample["contact_map"]),
        }
