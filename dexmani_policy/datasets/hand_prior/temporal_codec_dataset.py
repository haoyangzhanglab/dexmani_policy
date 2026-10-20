import numpy as np
import torch

from dexmani_policy.datasets.point_cloud_dataset import PointCloudDataset


class TemporalCodecDataset(PointCloudDataset):
    """Point-cloud windows with the raw measured hand pose before the query action."""

    def __init__(self, zarr_path, **kwargs):
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError(
                "TemporalCodecDataset requires joint action=7+12 without auxiliary targets"
            )
        modalities = kwargs.get("sensor_modalities")
        if modalities is not None and "joint_state" not in modalities:
            raise ValueError("Temporal codec reference requires joint_state observations")
        super().__init__(zarr_path, **kwargs)
        if self.horizon < 3:
            raise ValueError("Temporal codec horizon must be at least 3")
        buffer = self.replay_buffer
        if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("Temporal codec requires action and joint_state with shape [T,19]")
        self.hand_actions = buffer.read("action", slice(None), columns=slice(7, 19)).astype(
            np.float32, copy=False
        )
        self.hand_states = buffer.read("joint_state", slice(None), columns=slice(7, 19)).astype(
            np.float32, copy=False
        )

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
        data = super().__getitem__(idx)
        rows = self.sampler.sample_source_rows(idx)
        reference = self.hand_states[rows[self.obs_horizon - 1]].copy()
        data["action_context"] = {"hand_reference": torch.from_numpy(reference)}
        return data
