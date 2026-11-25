import numpy as np

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
        query = sampler.source_rows(slice(None), self.action_start_index)
        sampler.indices = sampler.indices[query + 1 < sampler.indices[:, 2]]
        return sampler

    def _samples_to_data(self, indices, samples):
        data = super()._samples_to_data(indices, samples)
        query = self.sampler.source_rows(indices, self.action_start_index)
        ends = self.sampler.indices[indices, 2]
        future_steps = self.horizon - self.action_start_index
        rows = query[:, None] + np.arange(1, future_steps + 1)
        valid = rows < ends[:, None]
        force = self.episode_store.read_rows_by_chunk(self.force_key, np.minimum(rows, ends[:, None] - 1))
        force[~valid] = 0
        for item, values, mask in zip(data, force, valid):
            item["future_force"] = values
            item["future_force_valid"] = mask
            yield item
