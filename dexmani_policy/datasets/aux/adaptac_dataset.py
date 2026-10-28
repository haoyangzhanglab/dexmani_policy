import numpy as np
import torch

from dexmani_policy.datasets.common.robot_dataset import RobotDataset


class AdapTacDataset(RobotDataset):
    def get_normalizer(self, spec):
        if self.is_validation or self._normalizer is not None:
            return super().get_normalizer(spec)
        key = "finger_tactile_force"
        normalizer = super().get_normalizer({name: mode for name, mode in spec.items() if name != key})
        normalizer.fit_field(key, self.iter_normalization_data(key), mode=spec[key], last_n_dims=1)
        field = normalizer[key]
        # The shared normalizer stores finger-vector fields as (5, 3).
        field.scale = field.scale.expand(5, 3).clone()
        field.offset = field.offset.expand(5, 3).clone()
        return normalizer

    def build_sampler(self, episode_mask):
        sampler = super().build_sampler(episode_mask)
        start, end, sample_start = sampler.indices.T
        query = np.clip(start + self.obs_horizon - 1 - sample_start, start, end - 1)
        episode_ends = self.replay_buffer.episode_ends
        ends = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        sampler.indices = sampler.indices[query + 1 < ends]
        return sampler

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.obs_horizon - 1]
        episode_ends = self.replay_buffer.episode_ends
        end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        steps = self.horizon - self.obs_horizon + 1
        # action[t] starts at the query; its resulting force is observed at t+1.
        force = self.replay_buffer.read(
            "finger_tactile_force", slice(query + 1, min(query + steps + 1, end))
        ).sum(axis=1, dtype=np.float32)
        rows = np.minimum(np.arange(steps), len(force) - 1)
        data["future_force"] = torch.from_numpy(force[rows])
        return data
