from pathlib import Path

import numpy as np
import torch
import zarr

from dexmani_policy.datasets.demonstration import DemonstrationDataset


class ODEDataset(DemonstrationDataset):
    def __init__(self, zarr_path, *, action_key="action", **kwargs):
        path = Path(zarr_path).expanduser().resolve()
        root = zarr.open_group(str(path), mode="r")
        if root.attrs.get("format") == "dexmani.real.canonical":
            processing = root.attrs.get("data_processing") or {}
            semantics = processing.get("action_ode", {}).get("semantics")
            if semantics != "held_accepted_joint_target" or "action_ode" not in root["data"]:
                raise ValueError(
                    f"{path}: Real ODE requires action_ode with recorded accepted joint targets. "
                    "Export with --fields joint_state point_cloud action_ode from Raw with submission status"
                )
            action_key = "action_ode"
        super().__init__(path, action_key=action_key, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        row = int(self.sampler.sample_source_rows(idx)[self.obs_horizon - 1])
        ends = self.episode_store.episode_ends
        episode = int(np.searchsorted(ends, row, side="right"))
        start = int(ends[episode - 1]) if episode else 0
        # Read raw state at episode start, outside observation augmentation.
        key, previous = ("joint_state", row) if row == start else (self.action_key, row - 1)
        data["prev_action"] = torch.as_tensor(self.episode_store.read(key, previous), dtype=torch.float32)
        return data
