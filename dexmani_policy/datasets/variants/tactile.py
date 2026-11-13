import numpy as np
import torch

from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES
from dexmani_policy.datasets.demonstration import DemonstrationDataset


class AdapTacDataset(DemonstrationDataset):
    def __init__(self, *args, sensor_modalities, **kwargs):
        fields = TACTILE_FEATURE_SHAPES.keys() & set(sensor_modalities)
        if len(fields) != 1:
            raise ValueError(f"AdapTac requires exactly one tactile field, got {sorted(fields)}")
        self.force_key = next(iter(fields))
        super().__init__(*args, sensor_modalities=sensor_modalities, **kwargs)

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
        query = int(self.sampler.sample_source_rows(idx)[self.action_start_index])
        episode_ends = self.episode_store.episode_ends
        end = int(episode_ends[np.searchsorted(episode_ends, query, side="right")])
        future_steps = self.horizon - self.action_start_index
        force = self.episode_store.read(self.force_key, slice(query + 1, min(query + 1 + future_steps, end)))
        steps = np.arange(future_steps)
        # The sampler keeps at least one future frame; repeat its last force for invalid tail steps.
        data["future_force"] = torch.from_numpy(force[np.minimum(steps, len(force) - 1)])
        data["future_force_valid"] = torch.from_numpy(steps < len(force))
        return data
