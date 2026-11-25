import numpy as np

from dexmani_policy.datasets.demonstration import DemonstrationDataset


class ODEDataset(DemonstrationDataset):
    def _samples_to_data(self, indices, samples):
        data = super()._samples_to_data(indices, samples)
        rows = self.sampler.source_rows(indices, self.action_start_index)
        windows, starts, _ = self.sampler.indices[indices].T
        first = rows == starts
        previous_offset = rows - windows - 1
        outside = (previous_offset < 0) & ~first
        previous = np.empty((len(samples), self.action_dim), dtype=np.float32)
        if outside.any():
            previous[outside] = self.episode_store.read_rows_by_chunk(self.action_key, rows[outside] - 1)
        for i, item in enumerate(data):
            if first[i]:
                # Copy the physical state before in-place observation augmentation.
                previous[i] = item["obs"]["joint_state"][0]
            elif not outside[i]:
                previous[i] = item["action"][previous_offset[i]]
            item["prev_action"] = previous[i]
            yield item
