from pathlib import Path

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
        start, _, sample_start = self.sampler.indices[idx]
        row = self.sampler.sample_source_row(idx, self.action_start_index)
        if row > start:
            slot = 0 if row - 1 == start else sample_start + row - 1 - start
            previous = data["action"][slot]
        else:
            ends = self.episode_store.episode_ends
            if row == 0 or row == ends[ends.searchsorted(row)]:
                previous = self.episode_store.read("joint_state", row)
            else:
                previous = self.episode_store.read(self.action_key, row - 1)
        data["prev_action"] = torch.as_tensor(previous, dtype=torch.float32)
        return data
