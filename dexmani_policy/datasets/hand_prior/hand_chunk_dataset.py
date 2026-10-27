import numpy as np

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_chunks
from dexmani_policy.datasets.common.robot_dataset import RobotDataset


class HandChunkDataset(RobotDataset):
    def __init__(
        self, zarr_path, *, data_semantics, horizon=32, seed=0, val_ratio=0.1,
        max_train_episodes=None, split_manifest=None, source_id=None,
    ):
        if positive_int(horizon, "horizon") < 3:
            raise ValueError("HandVAE horizon must be at least 3 for second differences")
        super().__init__(
            zarr_path, sensor_modalities=["joint_state"], horizon=horizon, obs_horizon=1,
            seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes,
            split_manifest=split_manifest,
        )
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)
        self.hand_actions = self.replay_buffer.read("action", slice(None), columns=slice(7, 19))
        self.hand_states = self.replay_buffer.read("joint_state", slice(None), columns=slice(7, 19))
        self.hand_actions.flags.writeable = self.hand_states.flags.writeable = False

    def finite_rows(self, key, columns=None):
        return super().finite_rows(key, columns=slice(7, 19))

    def raw_hand_batch(self, indices):
        rows = self.sampler.source_rows(self.sampler.indices[indices])
        return self.hand_actions[rows], self.hand_states[rows[:, 0]]

    def __getitem__(self, idx):
        actions, q0 = self.raw_hand_batch(np.array([idx]))
        return {"action": actions[0], "reference_state": q0[0]}
