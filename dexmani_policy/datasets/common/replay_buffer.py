import os
from pathlib import Path

import numpy as np
import zarr


class ReplayBuffer:
    """Read-only Zarr arrays with process-local handles."""

    def __init__(self, zarr_path, *, keys=None):
        self.path = str(Path(zarr_path).expanduser().resolve())
        self.group = zarr.open_group(self.path, mode="r")
        fields = self.group["data"]
        self.field_names = tuple(dict.fromkeys(fields.keys() if keys is None else keys))
        missing = set(self.field_names) - set(fields.keys())
        if missing:
            raise ValueError(f"Missing required data fields in {self.path}: {sorted(missing)}")
        self.data = {key: fields[key] for key in self.field_names}
        self.episode_ends = np.asarray(self.group["meta"]["episode_ends"])
        ends = self.episode_ends
        if (
            ends.ndim != 1
            or ends.size == 0
            or ends.dtype.kind not in "iu"
            or ends[0] <= 0
            or np.any(ends[1:] <= ends[:-1])
        ):
            raise ValueError(
                "episode_ends must be nonempty 1D positive, strictly increasing integers (not bool)"
            )
        for key, values in self.data.items():
            shape = values.shape
            if values.dtype.kind not in "buif":
                raise ValueError(f"Data field {key!r} must be real numeric, got {values.dtype}")
            if not shape or shape[0] != ends[-1]:
                raise ValueError(
                    f"Data field {key!r}: shape={shape}, expected time dimension {ends[-1]}"
                )
        self.episode_ends = ends.astype(np.int64, copy=False)
        self.attrs = dict(self.group.attrs)
        self.pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["group"] = state["pid"] = state["data"] = None
        return state

    def get_arrays(self):
        if self.group is None or self.pid != os.getpid():
            self.group = zarr.open_group(self.path, mode="r")
            self.data = {key: self.group["data"][key] for key in self.field_names}
            self.pid = os.getpid()
        return self.data

    def read(self, key, rows, *, columns=None):
        arr = self[key]
        selection = rows if columns is None else (rows,) + (slice(None),) * (arr.ndim - 2) + (columns,)
        dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
        return np.asarray(arr[selection], dtype=dtype)

    def chunk_rows(self, key):
        return self[key].chunks[0]

    def iter_chunks(self, key, *, columns=None, episode_mask=None):
        step = self.chunk_rows(key)
        mask = np.ones(self.n_episodes, dtype=bool) if episode_mask is None else episode_mask
        # Merge adjacent episodes, then bound reads by both selected spans and chunks.
        first = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        last = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        starts = np.r_[0, self.episode_ends[:-1]]
        for left, right in zip(starts[first], self.episode_ends[last]):
            start = int(left)
            while start < right:
                stop = min(int(right), (start // step + 1) * step)
                yield start, self.read(key, slice(start, stop), columns=columns)
                start = stop

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self.field_names

    def __getitem__(self, key):
        return self.get_arrays()[key]

    def __contains__(self, key):
        return key in self.field_names
