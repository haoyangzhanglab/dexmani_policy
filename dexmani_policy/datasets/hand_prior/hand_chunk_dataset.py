import copy

import numpy as np
from torch.utils.data import Dataset

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.sampler import SequenceSampler
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.validation import check_finite
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_chunks


class HandChunkDataset(Dataset):
    def __init__(
        self, zarr_path, *, data_semantics, horizon=32, seed=0, val_ratio=0.1,
        max_train_episodes=None, split_manifest=None, source_id=None,
    ):
        if positive_int(horizon, "horizon") < 3:
            raise ValueError("HandVAE horizon must be at least 3 for second differences")
        self.horizon = horizon
        self.replay_buffer = ReplayBuffer(zarr_path, keys=["action", "joint_state"])
        self.zarr_path = self.replay_buffer.path
        self.train_mask, self.val_mask, self.split_manifest = split_episodes(
            self.replay_buffer, manifest=split_manifest, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
        )
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)
        selected = self.train_mask | self.val_mask
        self.sampler = SequenceSampler(self.replay_buffer, horizon, episode_mask=self.train_mask)
        if not len(self.sampler):
            raise ValueError("No training hand chunks: selected episodes are too short")
        self.is_validation = False
        self.validation_dataset = None
        # Only the small hand arrays are shared; no visual policy state is needed.
        self.hand_actions = np.zeros((int(self.replay_buffer.episode_ends[-1]), 12), dtype=np.float32)
        self.hand_states = np.zeros_like(self.hand_actions)
        for key, values in (("action", self.hand_actions), ("joint_state", self.hand_states)):
            for start, chunk in self.replay_buffer.iter_chunks(key, columns=slice(7, 19), episode_mask=selected):
                check_finite(self.replay_buffer, key, start, chunk, columns=slice(7, 19))
                values[start:start + len(chunk)] = chunk
            values.flags.writeable = False

    def __len__(self):
        return len(self.sampler)

    def get_validation_dataset(self):
        if self.is_validation:
            return self
        if self.validation_dataset is None and self.val_mask.any():
            sampler = SequenceSampler(self.replay_buffer, self.horizon, episode_mask=self.val_mask)
            if len(sampler):
                validation = copy.copy(self)
                validation.sampler = sampler
                validation.is_validation = True
                self.validation_dataset = validation
        return self.validation_dataset

    def raw_hand_batch(self, indices):
        rows = self.sampler.source_rows(self.sampler.indices[indices])
        return self.hand_actions[rows], self.hand_states[rows[:, 0]]

    def __getitem__(self, idx):
        actions, q0 = self.raw_hand_batch(np.array([idx]))
        return {"action": actions[0], "reference_state": q0[0]}
