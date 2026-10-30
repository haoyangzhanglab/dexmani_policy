import numpy as np

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.hand_motion import describe_hand_chunks


class LatentFlowDataset(DemonstrationDataset):
    @property
    def action_start_index(self):
        return 0

    @property
    def action_offset(self):
        return self.obs_horizon - 1

    def __init__(self, zarr_path, *, sensor_modalities, data_semantics, obs_horizon, horizon=32, **kwargs):
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("LatentFlow requires joint action=7+12 without auxiliary targets")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LatentFlow reference requires joint_state observations")
        horizon = positive_int(horizon, "horizon")
        obs_horizon = positive_int(obs_horizon, "obs_horizon")
        if horizon < 3:
            raise ValueError("LatentFlow horizon must be at least 3")
        if kwargs.get("pad_before", obs_horizon - 1) != obs_horizon - 1 or kwargs.get("pad_after", 0) != 0:
            raise ValueError("LatentFlow pads only observation history, never future actions")
        kwargs.update(horizon=horizon, obs_horizon=obs_horizon, pad_before=obs_horizon - 1, pad_after=0)
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)
        self.hand_data = describe_hand_chunks(self, data_semantics)

    def build_sampler(self, episode_mask):
        return SequenceSampler(
            self.episode_store, self.horizon + self.action_offset,
            self.pad_before, self.pad_after, episode_mask,
            fields=self.consumed_fields, field_horizons={key: self.obs_horizon for key in self.sensor_modalities},
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
        if self.training_dataset is not None:
            raise ValueError("Validation must reuse the training normalizer")
        indices = self.sampler.indices
        episodes = np.searchsorted(self.episode_store.episode_ends, indices[:, 0], side="right")
        boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(indices)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for start in range(left, right, 1024):
                windows = indices[start:min(start + 1024, right)]
                query = windows[:, 0] + self.action_offset - windows[:, 2]
                first, stop = int(query[0]), int(query[-1]) + 1
                commands = self.episode_store.read("action", slice(first, stop + self.horizon - 1))
                states = self.episode_store.read("joint_state", slice(first, stop))
                # Queries are consecutive within an episode; window views avoid repeated gathers.
                chunks = np.lib.stride_tricks.sliding_window_view(commands, self.horizon, axis=0)
                relative = chunks.swapaxes(1, 2) - states[:, None]
                yield relative.reshape(-1, self.action_dim)
