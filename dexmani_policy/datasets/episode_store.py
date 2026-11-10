from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, TACTILE_FEATURE_SHAPES


class EpisodeStore:
    """Read-only arrays from a local Zarr 2 directory."""

    def __init__(self, zarr_path, *, keys=None):
        self.path = str(Path(zarr_path).expanduser().resolve())
        group = zarr.open_group(zarr.DirectoryStore(self.path), mode="r")
        fields = group["data"]
        self.data = {key: fields[key] for key in (fields.keys() if keys is None else keys)}
        self.attrs = dict(group.attrs)
        if (
            self.attrs.get("format") == "dexmani.real.canonical"
            and TACTILE_FEATURE_SHAPES.keys() & self.data.keys()
            and self.attrs.get("tactile_scale_divisor") != 255.0
        ):
            raise ValueError(
                f"Canonical Zarr {self.path}: tactile_scale_divisor must be 255.0, "
                f"got {self.attrs.get('tactile_scale_divisor')!r}"
            )
        ends = np.asarray(group["meta"]["episode_ends"])
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
            feature_shape = FIELD_FEATURE_SHAPES.get(key, TACTILE_FEATURE_SHAPES.get(key))
            if feature_shape is not None:
                if shape[1:] != feature_shape:
                    raise ValueError(f"{key} requires trailing shape {feature_shape} after its T axis")
                if values.dtype.kind != "f":
                    raise ValueError(f"Stored {key} must be floating point; observations are read as FP32")
        self.episode_ends = ends.astype(np.int64, copy=False)

    def read(self, key, rows, *, columns=None):
        arr = self[key]
        selection = rows if columns is None else (rows,) + (slice(None),) * (arr.ndim - 2) + (columns,)
        dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
        return np.asarray(arr[selection], dtype=dtype)

    def iter_chunks(self, key, *, columns=None, episode_mask=None):
        step = self[key].chunks[0]
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
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]


def check_finite(buffer, key, start, values, columns=None):
    invalid = ~np.isfinite(values)
    if not invalid.any():
        return
    location = [int(index) for index in np.unravel_index(invalid.argmax(), invalid.shape)]
    row = int(start + location[0])
    episode = int(np.searchsorted(buffer.episode_ends, row, side="right"))
    if columns is not None:
        location[-1] = int(np.arange(buffer[key].shape[-1])[columns][location[-1]])
    raise ValueError(
        f"Nonfinite data: path={buffer.path}, field={key}, episode={episode}, "
        f"row={row}, channel={tuple(location[1:])}"
    )
