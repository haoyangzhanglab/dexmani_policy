import math

import numpy as np
import torch

from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.common.hand_representation import validate_hand_coordinates


def describe_lamp_data(attrs, *, action_key, horizon, obs_horizon, pad_before, pad_after, history_steps):
    validate_hand_coordinates(attrs)
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("LAMP hand history requires finite positive dt")

    return {
        "action_key": action_key, "dt": float(dt),
        "horizon": horizon, "obs_horizon": obs_horizon,
        "pad_before": pad_before, "pad_after": pad_after,
        "history_steps": history_steps,
    }


def load_hand_history(buffer, episode_mask):
    if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
        raise ValueError("LAMP requires action and joint_state with shape [T,19]")
    actions = np.zeros((int(buffer.episode_ends[-1]), 12), dtype=np.float32)
    for start, values in buffer.iter_chunks("action", columns=slice(7, 19), episode_mask=episode_mask):
        actions[start:start + len(values)] = values
    bootstrap = np.zeros((buffer.n_episodes, 12), dtype=np.float32)
    starts = np.r_[0, buffer.episode_ends[:-1]]
    for episode in np.flatnonzero(episode_mask):
        start = int(starts[episode])
        bootstrap[episode] = buffer.read("joint_state", slice(start, start + 1), columns=slice(7, 19))[0]
    actions.flags.writeable = bootstrap.flags.writeable = False
    return actions, bootstrap


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
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("LAMPDataset requires joint action=7+12")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        if any(kwargs.get(key, 1) != 1 for key in ("horizon", "obs_horizon")):
            raise ValueError("LAMP requires horizon=obs_horizon=1")
        self.history_steps = history_steps
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)
        self.hand_data = describe_lamp_data(
            self.replay_buffer.attrs, action_key=self.action_key, horizon=self.horizon,
            obs_horizon=self.obs_horizon, pad_before=self.pad_before, pad_after=self.pad_after,
            history_steps=self.history_steps,
        )
        self.hand_actions, self.hand_state_bootstrap = load_hand_history(
            self.replay_buffer, self.train_mask | self.val_mask,
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[0]
        history = hand_history_for_rows(
            self.hand_actions, self.hand_state_bootstrap, self.replay_buffer.episode_ends,
            [query], self.history_steps,
        )[0]
        data["action_context"] = {"hand_history": torch.from_numpy(history)}
        return data
