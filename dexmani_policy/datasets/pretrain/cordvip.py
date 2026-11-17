import torch

from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler


class CordViPPretrainDataset(ObservationPretrainDataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, *, contact_loss_weight, seed=42, val_episodes=0,
                 max_train_episodes=None, horizon=4,
                 obs_horizon=4, pad_before=None, pad_after=0):
        if not 1 <= obs_horizon <= horizon:
            raise ValueError("obs_horizon must satisfy 1 <= N <= horizon")
        self.contact_loss_weight = contact_loss_weight
        keys = list(self.sensor_modalities)
        if contact_loss_weight > 0:
            keys.append("contact_map")
        buffer = EpisodeStore(zarr_path, keys=keys)
        self.horizon, self.obs_horizon = horizon, obs_horizon
        self.pad_before = obs_horizon - 1 if pad_before is None else pad_before
        self.pad_after = pad_after
        for key in self.sensor_modalities[:2]:
            shape = buffer[key].shape
            if len(shape) != 3 or shape[1] < 1 or shape[2] < 3:
                raise ValueError(f"CordViP requires {key}[T,N>0,C>=3]")
        if contact_loss_weight > 0:
            if buffer["object_pointcloud"].shape[1] != 1024:
                raise ValueError("CordViP contact head predicts 1024 object point labels")
            if buffer["contact_map"].shape[1:] != (1024,):
                raise ValueError("Contact labels must match the 1024 object point slots")
            processing = buffer.attrs.get("data_processing") or {}
            geometry = processing.get("object_pointcloud", {}).get("geometry_sha256")
            if not isinstance(geometry, str) or not geometry:
                raise ValueError("Contact supervision requires a shared object geometry_sha256 for fixed point slots")
        super().__init__(
            buffer, normalization_fields={
                "hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None,
            }, seed=seed, val_episodes=val_episodes, max_train_episodes=max_train_episodes,
        )

    def build_sampler(self, mask):
        fields = dict(self.normalization_fields)
        if self.contact_loss_weight > 0:
            fields["contact_map"] = None
        return SequenceSampler(
            self.episode_store, self.horizon, self.pad_before, self.pad_after, episode_mask=mask,
            fields=fields, field_horizons={key: self.obs_horizon for key in fields},
        )

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        batch = {
            "obs": {key: torch.as_tensor(sample[key]) for key in self.sensor_modalities},
        }
        if self.contact_loss_weight > 0:
            batch["contact_map"] = torch.as_tensor(sample["contact_map"])
        return batch
