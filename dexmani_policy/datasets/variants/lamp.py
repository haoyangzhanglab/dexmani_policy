import numpy as np
import torch

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.hand_semantics import validate_hand_coordinates
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import check_finite
from dexmani_policy.datasets.hand_motion import (
    episode_row_offsets, load_hand_array, recorded_dt,
)


def describe_lamp_data(attrs, *, action_key, horizon, obs_horizon, pad_before, pad_after, history_steps):
    validate_hand_coordinates(attrs)
    return {
        "action_key": action_key, "dt": recorded_dt(attrs),
        "horizon": horizon, "obs_horizon": obs_horizon,
        "pad_before": pad_before, "pad_after": pad_after,
        "history_steps": history_steps,
    }


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
    offsets = episode_row_offsets(buffer.episode_ends, episode_mask)
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


class LAMPDataset(DemonstrationDataset):
    def __init__(self, zarr_path, *, sensor_modalities, obs_horizon, history_steps=8, **kwargs):
        self.history_steps = positive_int(history_steps, "history_steps")
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("LAMPDataset requires joint action=7+12")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        if obs_horizon != 1 or kwargs.get("horizon", 1) != 1:
            raise ValueError("LAMP requires horizon=obs_horizon=1")
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, obs_horizon=obs_horizon, **kwargs)
        self.hand_data = describe_lamp_data(
            self.episode_store.attrs, action_key=self.action_key, horizon=self.horizon,
            obs_horizon=self.obs_horizon, pad_before=self.pad_before, pad_after=self.pad_after,
            history_steps=self.history_steps,
        )
        (self.hand_actions, self.hand_state_bootstrap,
         self.hand_episode_ends, self.hand_row_offsets) = load_hand_history(
            self.episode_store, self.train_mask | self.val_mask)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[0]
        episode = np.searchsorted(self.episode_store.episode_ends, query, side="right")
        query += self.hand_row_offsets[episode]
        history = hand_history_for_rows(
            self.hand_actions, self.hand_state_bootstrap, self.hand_episode_ends,
            [query], self.history_steps,
        )[0]
        data["action_context"] = {"hand_history": torch.from_numpy(history)}
        return data
