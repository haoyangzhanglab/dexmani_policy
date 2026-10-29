import numpy as np
import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.common.validation import check_finite
from dexmani_policy.datasets.hand_prior.data import (
    compact_episode_offsets, describe_lamp_data, load_hand_array,
)


def load_hand_history(buffer, episode_mask):
    if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
        raise ValueError("LAMP requires action and joint_state with shape [T,19]")
    actions = load_hand_array(buffer, "action", episode_mask)
    selected = np.flatnonzero(episode_mask)
    bootstrap = np.empty((len(selected), 12), dtype=np.float32)
    starts = np.r_[0, buffer.episode_ends[:-1]]
    for index, episode in enumerate(selected):
        start = int(starts[episode])
        state = buffer.read("joint_state", slice(start, start + 1), columns=slice(7, 19))
        check_finite(buffer, "joint_state", start, state, columns=slice(7, 19))
        bootstrap[index] = state[0]
    actions.flags.writeable = bootstrap.flags.writeable = False
    offsets = compact_episode_offsets(buffer.episode_ends, episode_mask)
    episode_ends = (buffer.episode_ends + offsets)[selected]
    return actions, bootstrap, episode_ends, offsets


def hand_history_for_rows(actions, bootstrap, episode_ends, rows, history_steps):
    rows = np.asarray(rows, dtype=np.int64)
    episode = np.searchsorted(episode_ends, rows, side="right")
    starts = np.r_[0, episode_ends[:-1]][episode]
    indices = rows[:, None] + np.arange(-history_steps, 0)
    return np.where(
        (indices >= starts[:, None])[..., None],
        actions[np.maximum(indices, starts[:, None])], bootstrap[episode, None],
    )


class LAMPDataset(RobotDataset):
    """Demonstration windows with strictly past hand commands."""

    def __init__(self, zarr_path, *, sensor_modalities, history_steps=8, **kwargs):
        self.history_steps = positive_int(history_steps, "history_steps")
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("LAMPDataset requires joint action=7+12")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        if any(kwargs.get(key, 1) != 1 for key in ("horizon", "obs_horizon")):
            raise ValueError("LAMP requires horizon=obs_horizon=1")
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)
        self.hand_data = describe_lamp_data(
            self.replay_buffer.attrs, action_key=self.action_key, horizon=self.horizon,
            obs_horizon=self.obs_horizon, pad_before=self.pad_before, pad_after=self.pad_after,
            history_steps=self.history_steps,
        )
        (self.hand_actions, self.hand_state_bootstrap,
         self.hand_episode_ends, self.hand_row_offsets) = load_hand_history(
            self.replay_buffer, self.train_mask | self.val_mask)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[0]
        episode = np.searchsorted(self.replay_buffer.episode_ends, query, side="right")
        query += self.hand_row_offsets[episode]
        history = hand_history_for_rows(
            self.hand_actions, self.hand_state_bootstrap, self.hand_episode_ends,
            [query], self.history_steps,
        )[0]
        data["action_context"] = {"hand_history": torch.from_numpy(history)}
        return data
