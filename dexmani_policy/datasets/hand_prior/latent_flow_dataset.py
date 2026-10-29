import numpy as np

from dexmani_policy.datasets.common.sampler import SequenceSampler
from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_chunks


class LatentFlowDataset(RobotDataset):
    """Historical observations paired with full future commands at the query state."""

    @property
    def action_start_index(self):
        return 0

    @property
    def action_offset(self):
        return self.obs_horizon - 1

    def __init__(self, zarr_path, *, sensor_modalities, data_semantics, source_id=None, **kwargs):
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("LatentFlow requires joint action=7+12 without auxiliary targets")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LatentFlow reference requires joint_state observations")
        horizon = positive_int(kwargs.get("horizon", 32), "horizon")
        obs_horizon = positive_int(kwargs.get("obs_horizon", 2), "obs_horizon")
        if horizon < 3:
            raise ValueError("LatentFlow horizon must be at least 3")
        if kwargs.get("pad_before", obs_horizon - 1) != obs_horizon - 1 or kwargs.get("pad_after", 0) != 0:
            raise ValueError("LatentFlow pads only observation history, never future actions")
        kwargs.update(horizon=horizon, obs_horizon=obs_horizon, pad_before=obs_horizon - 1, pad_after=0)
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)

    def build_sampler(self, episode_mask):
        return SequenceSampler(
            self.replay_buffer, self.horizon + self.action_offset,
            self.pad_before, self.pad_after, episode_mask,
            fields=self.consumed_fields, key_first_k={key: self.obs_horizon for key in self.sensor_modalities},
        )

    def sample_to_data(self, sample):
        sample[self.action_key] = sample[self.action_key][self.action_offset:]
        data = super().sample_to_data(sample)
        data["action_context"] = {"reference_state": sample["joint_state"][-1].copy()}
        return data

    def iter_normalization_data(self, key):
        if key != "action":
            yield from super().iter_normalization_data(key)
            return
        if self.is_validation:
            raise ValueError("Validation must reuse the training normalizer")
        # Bound both source reads and target tensors; keep every window/reference pair.
        indices = self.sampler.indices
        episodes = np.searchsorted(self.replay_buffer.episode_ends, indices[:, 0], side="right")
        boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(indices)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for start in range(left, right, 1024):
                rows = self.sampler.source_rows(indices[start:min(start + 1024, right)])
                future = rows[:, self.action_offset:]
                first, last = int(future[0, 0]), int(future[-1, -1]) + 1
                commands = self.replay_buffer.read("action", slice(first, last))
                states = self.replay_buffer.read("joint_state", slice(first, int(future[-1, 0]) + 1))
                relative = commands[future - first] - states[future[:, 0] - first, None]
                yield relative.reshape(-1, self.action_dim)
