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
            if semantics != "held_accepted_joint_target":
                raise ValueError(
                    f"{path}: Real ODE requires action_ode with recorded accepted joint targets. "
                    "Export with --fields joint_state point_cloud action_ode from Raw with submission status"
                )
            action_key = "action_ode"
        super().__init__(path, action_key=action_key, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        rows = self.sampler.sample_source_rows(idx)
        row = int(rows[self.action_start_index])
        ends = self.episode_store.episode_ends
        episode = int(np.searchsorted(ends, row, side="right"))
        start = int(ends[episode - 1]) if episode else 0
        if row == start:
            previous = self.episode_store.read("joint_state", row)
        else:
            slot = int(np.searchsorted(rows, row - 1))
            previous = data["action"][slot] if rows[slot] == row - 1 else self.episode_store.read(
                self.action_key, row - 1,
            )
        data["prev_action"] = torch.as_tensor(previous, dtype=torch.float32)
        return data
