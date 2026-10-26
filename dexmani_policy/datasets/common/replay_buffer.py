import os

import numpy as np
import zarr

READ_CACHE_BYTES = 16 * 1024 * 1024


class ReplayBuffer:
    """Selected read-only arrays; process-local Zarr handles and bounded reads."""

    def __init__(self, zarr_path, *, keys=None):
        self.path = os.path.abspath(os.path.expanduser(zarr_path))
        self.group = zarr.open_group(self.path, mode="r")
        fields = self.group["data"]
        self.field_names = tuple(dict.fromkeys(fields.keys() if keys is None else keys))
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
            if not shape or shape[0] != ends[-1]:
                raise ValueError(
                    f"Data field {key!r}: shape={shape}, expected time dimension {ends[-1]}"
                )
        self.attrs = dict(self.group.attrs)
        self.pid = os.getpid()
        self.read_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["group"] = state["pid"] = state["data"] = None
        state["read_cache"] = {}
        return state

    def get_arrays(self):
        if self.group is None or self.pid != os.getpid():
            self.group = zarr.open_group(self.path, mode="r")
            self.data = {key: self.group["data"][key] for key in self.field_names}
            self.read_cache = {}
            self.pid = os.getpid()
        return self.data

    def read(self, key, rows, *, columns=None, cache=True):
        arr = self[key]
        selection = rows if columns is None else (rows, columns)
        dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
        if cache and isinstance(rows, slice) and rows.step in (None, 1):
            start, stop, _ = rows.indices(arr.shape[0])
            step = self.chunk_rows(key)
            block_bytes = step * int(np.prod(arr.shape[1:])) * arr.dtype.itemsize
            # Retention bounds do not bound decoder allocations, even for a tail slice.
            if start < stop and block_bytes <= READ_CACHE_BYTES:
                result = None
                for block in range(start // step * step, stop, step):
                    cached = self.read_cache.get(key)
                    if cached is None or cached[0] != block:
                        self.read_cache.pop(key, None)
                        cached = (block, arr[block : block + step])
                        self.read_cache[key] = cached
                    left, right = max(start, block), min(stop, block + step)
                    values = cached[1][left - block : right - block]
                    if columns is not None:
                        values = values[:, columns]
                    if result is None:
                        result = np.empty((stop - start, *values.shape[1:]), dtype=dtype)
                    result[left - start : right - start] = values
                return result
        # Augmentation must not mutate cached storage.
        return np.array(arr[selection], dtype=dtype, copy=True)

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
                yield start, self.read(key, slice(start, stop), columns=columns, cache=False)
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
