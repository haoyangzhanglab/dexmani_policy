"""Point-cloud windows with strictly past hand commands."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from dexmani_policy.datasets.pc_dataset import PCDataset
from dexmani_policy.utils.hand_artifact import array_sha256

OBS_ALIGNMENT = "obs[t]_before_action[t]"


class LampDataset(PCDataset):
    """Pair observations/actions with strictly past physical hand commands."""

    def __init__(
        self,
        zarr_path,
        *,
        history_steps=8,
        obs_alignment=None,
        **kwargs,
    ):
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError(
                "LampDataset currently requires joint action=7+12 without auxiliary targets"
            )
        modalities = kwargs.get("sensor_modalities") or self.DEFAULT_MODALITIES
        if "joint_state" not in modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        self.history_steps = history_steps
        self._alignment_assertion = obs_alignment
        self._lamp_ready = False
        super().__init__(zarr_path, **kwargs)
        self._lamp_train_source_rows = self.sampler.action_source_rows.copy()

    def prepare_lamp_source(self):
        if self._lamp_ready:
            return
        buffer = self.replay_buffer
        attrs = buffer.attrs
        asserted = self._alignment_assertion
        if asserted == "obs_before_action":
            asserted = OBS_ALIGNMENT
        if asserted is not None and asserted != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported asserted obs_alignment: {asserted!r}")
        stored = attrs.get("obs_alignment")
        if stored is not None and stored != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported stored obs_alignment: {stored!r}")
        if stored is None and asserted is None:
            raise ValueError(
                "LAMP requires obs_alignment='obs[t]_before_action[t]' in data attributes "
                "or an explicit obs_alignment='obs_before_action' assertion"
            )
        dt = attrs.get("dt")
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (float, int))
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("LAMP data requires a finite positive dt attribute")
        if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("LAMP requires action and joint_state with shape [T,19]")
        self.episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
        self._episode_starts = np.r_[0, self.episode_ends[:-1]]
        self.hand_actions = np.asarray(
            buffer.read("action", slice(None), columns=slice(7, 19)), dtype=np.float32
        )
        self.hand_state_bootstrap = np.concatenate(
            [
                buffer.read("joint_state", slice(int(start), int(start) + 1), columns=slice(7, 19))
                for start in self._episode_starts
            ]
        ).astype(np.float32)
        finite_hand = np.isfinite(self.hand_actions).all(axis=-1)
        self.history_valid = np.empty(len(self.hand_actions), dtype=bool)
        for episode, (start, end) in enumerate(zip(self._episode_starts, self.episode_ends)):
            local_rows = np.arange(end - start)
            invalid_prefix = np.r_[0, np.cumsum(~finite_hand[start:end])]
            valid = (
                invalid_prefix[local_rows]
                == invalid_prefix[np.maximum(0, local_rows - self.history_steps)]
            )
            valid[: self.history_steps] &= np.isfinite(self.hand_state_bootstrap[episode]).all()
            self.history_valid[start:end] = valid
        self.lamp_source_metadata = {
            "history_steps": self.history_steps,
            "history": "strictly_past_hand_command_targets",
            "bootstrap": "episode_initial_measured_hand_pose",
            "obs_alignment": OBS_ALIGNMENT,
            "alignment_source": "data_attribute" if stored is not None else "explicit_assertion",
            "dt": float(dt),
            "action_key": "action",
            "hand_slice": [7, 19],
            "data_revision": self.data_revision,
            "num_frames": len(self.hand_actions),
            "hand_actions_sha256": array_sha256(self.hand_actions),
            "bootstrap_sha256": array_sha256(self.hand_state_bootstrap),
            "episode_ends_sha256": array_sha256(self.episode_ends),
        }
        canonical = json.dumps(
            self.lamp_source_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.lamp_source_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        self.data_recipe["lamp_history"] = {
            "source_fingerprint": self.lamp_source_fingerprint,
            "history_steps": self.history_steps,
            "bootstrap": "episode_initial_measured_hand_pose",
        }
        self._lamp_ready = True

    def _filter_sampler(self, sampler):
        # Called inside BaseDataset.__init__, before normalizer source rows exist.
        self.prepare_lamp_source()
        sampler.filter_valid(
            self._obs_valid, self._action_valid & self.history_valid, self.obs_horizon
        )

    def history_for_rows(self, rows):
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or (rows < 0).any()
            or (rows >= len(self.hand_actions)).any()
        ):
            raise ValueError("History rows must be one-dimensional source-frame indices")
        rows = rows.astype(np.int64, copy=False)
        episode = np.searchsorted(self.episode_ends, rows, side="right")
        starts = self._episode_starts[episode]
        indices = rows[:, None] + np.arange(-self.history_steps, 0)
        past = indices >= starts[:, None]
        safe_indices = np.maximum(indices, starts[:, None])
        return np.where(
            past[..., None],
            self.hand_actions[safe_indices],
            self.hand_state_bootstrap[episode, None],
        ).astype(np.float32, copy=False)

    def history_for_frame(self, row):
        return self.history_for_rows(np.asarray([row]))[0]

    @property
    def lamp_metadata(self):
        validation = self.get_validation_dataset()
        return {
            "source_fingerprint": self.lamp_source_fingerprint,
            "source_metadata": dict(self.lamp_source_metadata),
            "train_mask": self.train_mask.tolist(),
            "val_mask": self.val_mask.tolist(),
            "train_source_rows": self._lamp_train_source_rows.tolist(),
            "val_source_rows": []
            if validation is None
            else validation.sampler.action_source_rows.tolist(),
        }

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        data = super().__getitem__(idx)
        rows = self.sampler.source_rows(self.sampler.indices[idx : idx + 1])[0]
        query = rows[self.obs_horizon - 1]
        data["action_context"] = {"hand_history": torch.from_numpy(self.history_for_frame(query))}
        return data
