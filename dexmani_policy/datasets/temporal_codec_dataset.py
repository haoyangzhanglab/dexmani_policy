"""Policy windows with the raw measured hand pose used by the temporal codec."""

from __future__ import annotations

import numpy as np
import torch

from dexmani_policy.datasets.lamp_dataset import OBS_ALIGNMENT
from dexmani_policy.datasets.pc_dataset import PCDataset
from dexmani_policy.utils.hand_artifact import array_sha256, metadata_sha256


class TemporalCodecDataset(PCDataset):
    """Keep physical actions and raw measured q0 at the policy's query observation."""

    def __init__(self, zarr_path, *, obs_alignment=None, **kwargs):
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError(
                "TemporalCodecDataset requires joint action=7+12 without auxiliary targets"
            )
        if "joint_state" not in (kwargs.get("sensor_modalities") or self.DEFAULT_MODALITIES):
            raise ValueError("Temporal codec reference requires joint_state observations")
        super().__init__(zarr_path, **kwargs)
        if self.horizon < 3:
            raise ValueError("Temporal codec horizon must be at least 3")
        buffer = self.replay_buffer
        asserted = OBS_ALIGNMENT if obs_alignment == "obs_before_action" else obs_alignment
        if asserted is not None and asserted != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported asserted obs_alignment: {asserted!r}")
        stored = buffer.attrs.get("obs_alignment")
        if stored is not None and stored != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported stored obs_alignment: {stored!r}")
        if stored is None and asserted is None:
            raise ValueError(
                "Temporal codec requires data attribute obs_alignment='obs[t]_before_action[t]' "
                "or an explicit obs_alignment='obs_before_action' assertion"
            )
        dt = buffer.attrs.get("dt")
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Temporal codec requires a positive finite dt attribute")
        if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("Temporal codec requires action and joint_state with shape [T,19]")
        self.hand_actions = np.asarray(
            buffer.read("action", slice(None), columns=slice(7, 19)), dtype=np.float32
        )
        self.hand_states = np.asarray(
            buffer.read("joint_state", slice(None), columns=slice(7, 19)), dtype=np.float32
        )
        self.codec_source_metadata = {
            "reference": "raw_measured_hand_at_query_observation",
            "obs_alignment": OBS_ALIGNMENT,
            "alignment_source": (
                "data_attribute" if stored is not None else "explicit_assertion"
            ),
            "dt": float(dt),
            "action_key": "action",
            "hand_slice": [7, 19],
            "data_revision": self.data_revision,
            "num_frames": len(self.hand_actions),
            "hand_actions_sha256": array_sha256(self.hand_actions),
            "hand_states_sha256": array_sha256(self.hand_states),
            "episode_ends_sha256": array_sha256(
                np.asarray(buffer.episode_ends, dtype=np.int64)
            ),
        }
        self.data_recipe["temporal_codec_reference"] = {
            "source_fingerprint": metadata_sha256(self.codec_source_metadata),
            "query_index": self.obs_horizon - 1,
        }

    def raw_hand_batch(self, indices):
        """Return raw [B,H,12] commands and [B,12] measured q0 without PC reads."""
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or (indices < 0).any()
            or (indices >= len(self)).any()
        ):
            raise ValueError("Hand batch indices must be one-dimensional valid window indices")
        rows = self.sampler.source_rows(self.sampler.indices[indices.astype(np.int64, copy=False)])
        return self.hand_actions[rows], self.hand_states[rows[:, self.obs_horizon - 1]]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        data = super().__getitem__(idx)
        rows = self.sampler.source_rows(self.sampler.indices[idx : idx + 1])[0]
        reference = self.hand_states[rows[self.obs_horizon - 1]].copy()
        data["action_context"] = {"hand_reference": torch.from_numpy(reference)}
        return data
