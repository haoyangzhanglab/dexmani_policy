import torch

from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler


class CordViPPretrainDataset(ObservationPretrainDataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, *, contact_loss_weight, seed=42, val_episodes=0,
                 max_train_episodes=None, obs_horizon=4, pad_before=None, pad_after=0):
        keys = (*self.sensor_modalities, "contact_map") if contact_loss_weight > 0 else self.sensor_modalities
        buffer = EpisodeStore(zarr_path, keys=keys, pc_dim=3)
        self.obs_horizon = obs_horizon
        self.pad_before = obs_horizon - 1 if pad_before is None else pad_before
        self.pad_after = pad_after
        if "contact_map" in keys:
            contact_map = buffer["contact_map"]
            if contact_map.shape != buffer["object_pointcloud"].shape[:2]:
                raise ValueError("Contact labels must match the object point slots in every frame")
            if contact_map.dtype.kind != "f":
                raise ValueError(f"{buffer.path}: contact_map must be floating point")
        super().__init__(
            buffer, normalization_fields={
                "hand_pointcloud": slice(0, 3), "object_pointcloud": slice(0, 3), "joint_state": None,
            }, seed=seed, val_episodes=val_episodes, max_train_episodes=max_train_episodes,
        )

    def build_sampler(self, mask):
        fields = dict(self.normalization_fields)
        if "contact_map" in self.episode_store.data:
            fields["contact_map"] = None
        return SequenceSampler(
            self.episode_store, self.obs_horizon, self.pad_before, self.pad_after,
            episode_mask=mask, fields=fields,
        )

    def sample_to_data(self, sample):
        batch = super().sample_to_data(sample)
        if "contact_map" in sample:
            batch["contact_map"] = torch.as_tensor(sample["contact_map"])
        return batch
