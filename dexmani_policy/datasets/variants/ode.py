import torch

from dexmani_policy.datasets.demonstration import DemonstrationDataset


class ODEDataset(DemonstrationDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        start, _, sample_start = self.sampler.indices[idx]
        row = self.sampler.sample_source_row(idx, self.action_start_index)
        if row > start:
            slot = 0 if row - 1 == start else sample_start + row - 1 - start
            previous = data["action"][slot]
        else:
            ends = self.episode_store.episode_ends
            if row == 0 or row == ends[ends.searchsorted(row)]:
                previous = self.episode_store.read("joint_state", row)
            else:
                previous = self.episode_store.read(self.action_key, row - 1)
        data["prev_action"] = torch.as_tensor(previous, dtype=torch.float32)
        return data
