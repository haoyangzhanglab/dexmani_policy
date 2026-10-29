import numpy as np
import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.common.robot_dataset import RobotDataset


class InterActDexDataset(RobotDataset):
    def __init__(self, *args, n_action_steps, sensor_modalities, **kwargs):
        self.n_action_steps = positive_int(n_action_steps, "n_action_steps")
        if "finger_tactile_force" not in sensor_modalities:
            raise ValueError("InterAct-Dex requires finger_tactile_force for its future target")
        super().__init__(*args, sensor_modalities=sensor_modalities, **kwargs)
        if self.action_start_index + self.n_action_steps > self.horizon:
            raise ValueError("action_start_index + n_action_steps must not exceed horizon")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        episode_ends = self.replay_buffer.episode_ends
        end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        # Read the final real row even when there is no successor to supervise.
        force = self.replay_buffer.read(
            "finger_tactile_force",
            slice(min(query + 1, end - 1), min(query + self.n_action_steps + 1, end)),
        )
        rows = np.arange(self.n_action_steps)
        data["future_force"] = torch.from_numpy(force[np.minimum(rows, len(force) - 1)])
        data["future_force_valid"] = torch.from_numpy(rows < end - query - 1)
        return data
