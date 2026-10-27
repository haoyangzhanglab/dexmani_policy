import numpy as np
import torch

from dexmani_policy.datasets.common.robot_dataset import RobotDataset


class LAMPDataset(RobotDataset):
    """Demonstration windows with strictly past hand commands."""

    def __init__(self, zarr_path, *, sensor_modalities, history_steps=8, **kwargs):
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError("LAMPDataset requires joint action=7+12 without auxiliary targets")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        self.history_steps = history_steps
        self.history_valid = None
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)

    def prepare_lamp_source(self):
        buffer = self.replay_buffer
        if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("LAMP requires action and joint_state with shape [T,19]")
        self.episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
        self.episode_starts = np.r_[0, self.episode_ends[:-1]]
        selected = self.train_mask | self.val_mask
        self.hand_actions = np.zeros((int(self.episode_ends[-1]), 12), dtype=np.float32)
        for start, values in buffer.iter_chunks("action", columns=slice(7, 19), episode_mask=selected):
            self.hand_actions[start:start + len(values)] = values
        self.hand_state_bootstrap = np.zeros((len(self.episode_ends), 12), dtype=np.float32)
        self.history_valid = np.zeros(len(self.hand_actions), dtype=bool)
        for episode in np.flatnonzero(selected):
            start, end = self.episode_starts[episode], self.episode_ends[episode]
            self.hand_state_bootstrap[episode] = buffer.read(
                "joint_state", slice(int(start), int(start) + 1), columns=slice(7, 19),
            )[0]
            finite_hand = np.isfinite(self.hand_actions[start:end]).all(axis=-1)
            local_rows = np.arange(end - start)
            invalid_prefix = np.r_[0, np.cumsum(~finite_hand)]
            valid = (
                invalid_prefix[local_rows]
                == invalid_prefix[np.maximum(0, local_rows - self.history_steps)]
            )
            valid[: self.history_steps] &= np.isfinite(self.hand_state_bootstrap[episode]).all()
            self.history_valid[start:end] = valid

    def filter_sampler(self, sampler):
        # RobotDataset construction dispatches here before valid source rows are known.
        if self.history_valid is None:
            self.prepare_lamp_source()
        sampler.filter_valid(
            self.observation_valid, self.action_valid & self.history_valid, self.obs_horizon
        )

    def history_for_rows(self, rows):
        rows = np.asarray(rows)
        rows = rows.astype(np.int64, copy=False)
        episode = np.searchsorted(self.episode_ends, rows, side="right")
        starts = self.episode_starts[episode]
        indices = rows[:, None] + np.arange(-self.history_steps, 0)
        past = indices >= starts[:, None]
        safe_indices = np.maximum(indices, starts[:, None])
        return np.where(
            past[..., None],
            self.hand_actions[safe_indices],
            self.hand_state_bootstrap[episode, None],
        )

    def history_for_frame(self, row):
        return self.history_for_rows(np.asarray([row]))[0]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        rows = self.sampler.sample_source_rows(idx)
        query = rows[self.obs_horizon - 1]
        data["action_context"] = {"hand_history": torch.from_numpy(self.history_for_frame(query))}
        return data
