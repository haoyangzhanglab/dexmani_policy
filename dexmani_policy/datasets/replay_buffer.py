import os

import numpy as np
import zarr


def validate_episode_metadata(episode_ends, data, keys=None):
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
    for key in data.keys() if keys is None else keys:
        if key not in data:
            raise ValueError(f"Missing data field: {key}")
        shape = data[key].shape
        if not shape or shape[0] != ends[-1]:
            raise ValueError(
                f"Data field {key!r}: shape={shape}, expected time dimension {ends[-1]}"
            )


class ReplayBuffer:
    """Selected read-only arrays; process-local Zarr handles and bounded reads."""

    def __init__(self, root, *, keys=None):
        if (
            "data" not in root
            or "meta" not in root
            or "episode_ends" not in root["meta"]
        ):
            raise ValueError("ReplayBuffer requires data and meta/episode_ends")
        self._keys = tuple(dict.fromkeys(root["data"].keys() if keys is None else keys))
        self.meta = {
            k: np.asarray(v) if isinstance(v, np.ndarray) else v[...]
            for k, v in root["meta"].items()
        }
        validate_episode_metadata(self.meta["episode_ends"], root["data"], self._keys)
        self.attrs = dict(
            root.get("attrs", {}) if isinstance(root, dict) else root.attrs
        )
        revision = self.attrs.get("data_revision")
        if "data_revision" in self.attrs and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError(
                "Zarr data_revision must be a nonempty string when present"
            )
        self.data_revision = revision
        self.row_info = {}
        rows = root.get("row_info", {})
        if "dispatch_status" in rows:
            status = rows["dispatch_status"]
            if (
                status.shape != (int(self.episode_ends[-1]), 2)
                or status.dtype != np.uint8
            ):
                raise ValueError("dispatch_status must be uint8 (rows, 2)")
            self.row_info["dispatch_status"] = np.array(status[:], copy=True)
        self._path = None
        self._group = None
        self._pid = None
        self._data = {key: root["data"][key] for key in self._keys}
        self._read_cache = {}

    @classmethod
    def open(cls, zarr_path, keys=None):
        path = os.path.abspath(os.path.expanduser(zarr_path))
        group = zarr.open_group(path, mode="r")
        buffer = cls(group, keys=keys)
        buffer._path = path
        buffer._data = None
        return buffer

    def observation_timestamps(self):
        """Read only the integer evidence required by the Real time recipe."""
        root = zarr.open_group(self._path, mode="r") if self._path else None
        if root is None or "row_info/observation_timestamp_ns" not in root:
            raise ValueError("Time filtering requires row_info/observation_timestamp_ns; unknown Raw times cannot be reconstructed")
        stamps = root["row_info/observation_timestamp_ns"]
        if stamps.shape != (int(self.episode_ends[-1]),) or stamps.dtype.kind not in "iu":
            raise ValueError("observation_timestamp_ns must be a 1D integer array with one timestamp per source row")
        return np.asarray(stamps[:])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_group"] = state["_pid"] = None
        state["_read_cache"] = {}
        if self._path is not None:
            state["_data"] = None
        return state

    def _arrays(self):
        if self._path is None:
            return self._data
        if self._group is None or self._pid != os.getpid():
            self._group = zarr.open_group(self._path, mode="r")
            self._data = {key: self._group["data"][key] for key in self._keys}
            self._read_cache = {}
            self._pid = os.getpid()
        return self._data

    def read(self, key, rows, *, columns=None):
        arr = self[key]
        selection = rows if columns is None else (rows, columns)
        values = None
        if (
            self._path is not None
            and isinstance(rows, slice)
            and rows.step in (None, 1)
        ):
            start, stop, _ = rows.indices(arr.shape[0])
            step = self.chunk_rows(key)
            block = start // step * step
            if start < stop <= block + step:
                cached = self._read_cache.get(key)
                if cached is None or cached[0] != block:
                    # Exactly one bounded decoded block per selected field,
                    # released on PID change/pickle. Adjacent windows reuse it.
                    cached = (block, arr[block : block + step])
                    self._read_cache[key] = cached
                values = cached[1][start - block : stop - block]
                if columns is not None:
                    values = values[:, columns]
        if values is None:
            values = arr[selection]
        dtype = np.float32 if np.issubdtype(values.dtype, np.floating) else values.dtype
        # Callers may augment windows in place; no mutable alias escapes even
        # when this buffer was constructed from an in-memory NumPy dictionary.
        return np.array(values, dtype=dtype, copy=True)

    def chunk_rows(self, key):
        arr = self[key]
        row_bytes = max(1, int(np.prod(arr.shape[1:])) * max(arr.dtype.itemsize, 4))
        limit = max(1, (16 * 1024 * 1024) // row_bytes)
        physical = arr.chunks[0] if isinstance(arr, zarr.Array) else limit
        return max(1, min(physical, limit))

    def iter_chunks(self, key, *, columns=None):
        step = self.chunk_rows(key)
        for start in range(0, int(self.episode_ends[-1]), step):
            yield start, self.read(key, slice(start, start + step), columns=columns)

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._arrays()[key]

    def __contains__(self, key):
        return key in self._keys
