import os

import numpy as np
import zarr

READ_CACHE_BYTES = 16 * 1024 * 1024


def validate_episode_metadata(episode_ends, data):
    ends = np.asarray(episode_ends)
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
    for key, values in data.items():
        shape = values.shape
        if not shape or shape[0] != ends[-1]:
            raise ValueError(
                f"Data field {key!r}: shape={shape}, expected time dimension {ends[-1]}"
            )


class ReplayBuffer:
    """Selected read-only arrays; process-local Zarr handles and bounded reads."""

    def __init__(self, root, *, keys=None):
        self.field_names = tuple(dict.fromkeys(root["data"].keys() if keys is None else keys))
        self.data = {key: root["data"][key] for key in self.field_names}
        self.episode_ends = np.asarray(root["meta"]["episode_ends"])
        validate_episode_metadata(self.episode_ends, self.data)
        self.attrs = dict(root.get("attrs", {}) if isinstance(root, dict) else root.attrs)
        revision = self.attrs.get("data_revision")
        if "data_revision" in self.attrs and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("Zarr data_revision must be a nonempty string when present")
        self.data_revision = revision
        self.path = None
        self.group = None
        self.pid = None
        self.read_cache = {}

    @classmethod
    def open(cls, zarr_path, keys=None):
        path = os.path.abspath(os.path.expanduser(zarr_path))
        group = zarr.open_group(path, mode="r")
        buffer = cls(group, keys=keys)
        buffer.path = path
        buffer.group = group
        buffer.pid = os.getpid()
        return buffer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["group"] = state["pid"] = None
        state["read_cache"] = {}
        if self.path is not None:
            state["data"] = None
        return state

    def get_arrays(self):
        if self.path is None:
            return self.data
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
        if self.path is not None and cache and isinstance(rows, slice) and rows.step in (None, 1):
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
        # No mutable alias escapes, including in-memory buffers and direct large-block reads.
        return np.array(arr[selection], dtype=dtype, copy=True)

    def chunk_rows(self, key):
        """Scan physical time blocks; only in-memory arrays need an artificial bound."""
        arr = self[key]
        if isinstance(arr, zarr.Array):
            return arr.chunks[0]
        row_bytes = max(1, int(np.prod(arr.shape[1:])) * arr.dtype.itemsize)
        return max(1, READ_CACHE_BYTES // row_bytes)

    def iter_chunks(self, key, *, columns=None):
        step = self.chunk_rows(key)
        for start in range(0, int(self.episode_ends[-1]), step):
            yield start, self.read(key, slice(start, start + step), columns=columns, cache=False)

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self.field_names

    def __getitem__(self, key):
        return self.get_arrays()[key]

    def __contains__(self, key):
        return key in self.field_names
