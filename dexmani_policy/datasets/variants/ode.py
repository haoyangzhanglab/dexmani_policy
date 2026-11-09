import numpy as np
import torch

from dexmani_policy.datasets.demonstration import DemonstrationDataset


class ODEDataset(DemonstrationDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        row = int(self.sampler.sample_source_rows(idx)[self.obs_horizon - 1])
        ends = self.episode_store.episode_ends
        episode = int(np.searchsorted(ends, row, side="right"))
        start = int(ends[episode - 1]) if episode else 0
        # Read raw state at episode start, outside observation augmentation.
        key, previous = ("joint_state", row) if row == start else (self.action_key, row - 1)
        data["prev_action"] = torch.as_tensor(self.episode_store.read(key, previous), dtype=torch.float32)
        return data
