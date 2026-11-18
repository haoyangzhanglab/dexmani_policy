import torch

from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.preprocess.contact_map import require_contact_geometry
from dexmani_policy.datasets.pretrain import ObservationPretrainDataset
from dexmani_policy.datasets.sampler import SequenceSampler


class CordViPPretrainDataset(ObservationPretrainDataset):
    sensor_modalities = ("hand_pointcloud", "object_pointcloud", "joint_state")

    def __init__(self, zarr_path, *, contact_loss_weight, seed=42, val_episodes=0,
                 max_train_episodes=None, obs_horizon=4, pad_before=None, pad_after=0):
        self.contact_loss_weight = contact_loss_weight
        keys = list(self.sensor_modalities)
        if contact_loss_weight > 0:
            keys.append("contact_map")
        buffer = EpisodeStore(zarr_path, keys=keys, pc_dim=3)
        self.obs_horizon = obs_horizon
        self.pad_before = obs_horizon - 1 if pad_before is None else pad_before
        self.pad_after = pad_after
        if contact_loss_weight > 0:
            require_contact_geometry(buffer)
            if buffer["contact_map"].shape[1:] != buffer["object_pointcloud"].shape[1:2]:
                raise ValueError("Contact labels must match the 1024 object point slots")
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
            self.episode_store, self.obs_horizon, self.pad_before, self.pad_after,
            episode_mask=mask, fields=fields,
        )

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        batch: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            "obs": {key: torch.as_tensor(sample[key]) for key in self.sensor_modalities},
        }
        if self.contact_loss_weight > 0:
            batch["contact_map"] = torch.as_tensor(sample["contact_map"])
        return batch
