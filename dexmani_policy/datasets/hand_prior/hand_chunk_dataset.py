"""Full future hand chunks, independent of policy observations and visual validity."""

import numpy as np

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.training.common.hand_representation import describe_hand_chunks


class HandChunkDataset(DemonstrationDataset):
    def __init__(
        self, zarr_path, *, data_semantics, horizon=32, seed=0, val_ratio=0.1,
        max_train_episodes=None, split_manifest=None, saved_split=None, source_id=None,
    ):
        if positive_int(horizon, "horizon") < 3:
            raise ValueError("HandVAE horizon must be at least 3 for second differences")
        super().__init__(
            zarr_path, sensor_modalities=["joint_state"], horizon=horizon, obs_horizon=1,
            seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_manifest=split_manifest, saved_split=saved_split,
        )
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)

    def finite_rows(self, key, columns=None):
        return super().finite_rows(key, columns=slice(7, 19))

    def raw_hand_batch(self, indices):
        rows = self.sampler.source_rows(self.sampler.indices[indices])
        actions = self.replay_buffer.read("action", rows.reshape(-1), columns=slice(7, 19))
        q0 = self.replay_buffer.read("joint_state", rows[:, 0], columns=slice(7, 19))
        return actions.reshape(-1, self.horizon, 12), q0

    def __getitem__(self, idx):
        actions, q0 = self.raw_hand_batch(np.array([idx]))
        return {"action": actions[0], "reference_state": q0[0]}
