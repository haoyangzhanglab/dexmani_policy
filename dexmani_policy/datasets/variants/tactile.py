import numpy as np
import torch

from dexmani_policy.common.config import validate_future_force_offsets
from dexmani_policy.datasets.demonstration import DemonstrationDataset


def read_future_force(store, query, offsets):
    episode_ends = store.episode_ends
    end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
    target_rows = query + np.asarray(offsets, dtype=np.int64)
    # The terminal query uses the final real row as fill, with no valid supervision.
    rows = np.minimum(target_rows, end - 1)
    force = store.read(
        "finger_tactile_force",
        slice(rows[0], rows[-1] + 1),
    )
    return force[rows - rows[0]], target_rows < end


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
        offsets = range(1, self.horizon - self.action_start_index + 1)
        force, valid = read_future_force(self.episode_store, query, offsets)
        data["future_force"] = torch.from_numpy(force)
        data["future_force_valid"] = torch.from_numpy(valid)
        return data


class InterActDexDataset(DemonstrationDataset):
    def __init__(self, *args, future_force_offsets, sensor_modalities, **kwargs):
        self.future_force_offsets = validate_future_force_offsets(future_force_offsets)
        if "finger_tactile_force" not in sensor_modalities:
            raise ValueError("InterAct-Dex requires finger_tactile_force for its future target")
        super().__init__(*args, sensor_modalities=sensor_modalities, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        force, valid = read_future_force(self.episode_store, query, self.future_force_offsets)
        data["future_force"] = torch.from_numpy(force)
        data["future_force_valid"] = torch.from_numpy(valid)
        return data
