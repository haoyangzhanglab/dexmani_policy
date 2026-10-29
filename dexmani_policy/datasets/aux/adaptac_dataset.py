import numpy as np
import torch

from dexmani_policy.datasets.common.robot_dataset import RobotDataset


class AdapTacDataset(RobotDataset):
    def __init__(self, *args, tactile_source, sensor_modalities, **kwargs):
        if "finger_tactile_force" not in sensor_modalities:
            raise ValueError("AdapTac requires finger_tactile_force for its 15D future target and guide")
        super().__init__(*args, tactile_source=tactile_source, sensor_modalities=sensor_modalities, **kwargs)

    def build_sampler(self, episode_mask):
        sampler = super().build_sampler(episode_mask)
        start, end, sample_start = sampler.indices.T
        query = np.clip(start + self.action_start_index - sample_start, start, end - 1)
        episode_ends = self.replay_buffer.episode_ends
        ends = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        sampler.indices = sampler.indices[query + 1 < ends]
        return sampler

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        episode_ends = self.replay_buffer.episode_ends
        end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        steps = self.horizon - self.action_start_index
        # Next observation row; real SDK hold can retain an older sensor sample.
        force = self.replay_buffer.read(
            "finger_tactile_force", slice(query + 1, min(query + steps + 1, end))
        )
        rows = np.arange(steps)
        data["future_force"] = torch.from_numpy(force[np.minimum(rows, len(force) - 1)])
        data["future_force_valid"] = torch.from_numpy(rows < len(force))
        return data
