import numpy as np
import torch

from dexmani_policy.common.config import TACTILE_FEATURE_SHAPES, positive_int
from dexmani_policy.datasets.demonstration import DemonstrationDataset


def read_future_force(store, query, offsets, *, field):
    episode_ends = store.episode_ends
    end = episode_ends[np.searchsorted(episode_ends, query, side="right")]
    target_rows = query + np.asarray(offsets, dtype=np.int64)
    # Targets past the episode end use the final real row as invalid fill.
    rows = np.minimum(target_rows, end - 1)
    valid = target_rows < end
    start = int(rows.min())
    stop = int(rows.max()) + 1
    force = store.read(field, slice(start, stop))
    if len(rows) == 1 or (
        isinstance(offsets, range) and offsets.step == 1 and target_rows[-1] < end
    ):
        return force, valid
    return force[rows - start], valid


class AdapTacDataset(DemonstrationDataset):
    def __init__(self, *args, sensor_modalities, **kwargs):
        fields = TACTILE_FEATURE_SHAPES.keys() & set(sensor_modalities)
        if len(fields) != 1:
            raise ValueError(f"AdapTac requires exactly one tactile field, got {sorted(fields)}")
        self.force_key = next(iter(fields))
        super().__init__(*args, sensor_modalities=sensor_modalities, **kwargs)

    def build_sampler(self, episode_mask):
        sampler = super().build_sampler(episode_mask)
        start, end, sample_start = sampler.indices.T
        query = np.clip(start + self.action_start_index - sample_start, start, end - 1)
        episode_ends = self.episode_store.episode_ends
        ends = episode_ends[np.searchsorted(episode_ends, query, side="right")]
        sampler.indices = sampler.indices[query + 1 < ends]
        return sampler

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        offsets = range(1, self.horizon - self.action_start_index + 1)
        force, valid = read_future_force(self.episode_store, query, offsets, field=self.force_key)
        data["future_force"] = torch.from_numpy(force)
        data["future_force_valid"] = torch.from_numpy(valid)
        return data


class InterActDexDataset(DemonstrationDataset):
    def __init__(self, *args, future_force_horizon, include_future_force, sensor_modalities, **kwargs):
        self.future_force_horizon = positive_int(future_force_horizon, "future_force_horizon")
        self.include_future_force = include_future_force
        if TACTILE_FEATURE_SHAPES.keys() & set(sensor_modalities) != {"finger_tactile_force"}:
            raise ValueError("InterAct-Dex requires finger_tactile_force as its only tactile field")
        super().__init__(*args, sensor_modalities=sensor_modalities, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if not self.include_future_force:
            return data
        query = self.sampler.sample_source_rows(idx)[self.action_start_index]
        force, valid = read_future_force(
            self.episode_store, query, (self.future_force_horizon,), field="finger_tactile_force",
        )
        data["future_force"] = torch.from_numpy(force[0])
        data["future_force_valid"] = torch.from_numpy(valid)[0]
        return data
