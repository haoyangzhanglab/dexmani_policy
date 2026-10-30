import numpy as np
import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.demonstration import DemonstrationDataset


def read_future_force(store, query, steps):
    episode_ends = store.episode_ends
    end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
    # The terminal query uses the final real row as fill, with no valid supervision.
    force = store.read(
        "finger_tactile_force",
        slice(min(query + 1, end - 1), min(query + steps + 1, end)),
    )
    rows = np.arange(steps)
    return force[np.minimum(rows, len(force) - 1)], rows < end - query - 1


class AdapTacDataset(DemonstrationDataset):
    def __init__(self, *args, tactile_source, sensor_modalities, **kwargs):
        if "finger_tactile_force" not in sensor_modalities:
            raise ValueError("AdapTac requires finger_tactile_force for its 15D future target and guide")
        super().__init__(*args, tactile_source=tactile_source, sensor_modalities=sensor_modalities, **kwargs)

    def build_sampler(self, episode_mask):
        sampler = super().build_sampler(episode_mask)
        start, end, sample_start = sampler.indices.T
        query = np.clip(start + self.action_start_index - sample_start, start, end - 1)
        episode_ends = self.episode_store.episode_ends
        ends = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        sampler.indices = sampler.indices[query + 1 < ends]
        return sampler

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        force, valid = read_future_force(self.episode_store, query, self.horizon - self.action_start_index)
        data["future_force"] = torch.from_numpy(force)
        data["future_force_valid"] = torch.from_numpy(valid)
        return data


class InterActDexDataset(DemonstrationDataset):
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
        force, valid = read_future_force(self.episode_store, query, self.n_action_steps)
        data["future_force"] = torch.from_numpy(force)
        data["future_force_valid"] = torch.from_numpy(valid)
        return data
