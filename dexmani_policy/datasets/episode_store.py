from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.common.config import (
    FIXED_OBSERVATION_SHAPES,
    POINT_CLOUD_FIELDS,
    validate_observation_shape,
)


class EpisodeStore:
    """Read-only arrays from a local Zarr 2 directory."""

    def __init__(self, zarr_path, *, keys, pc_dim=None):
        self.path = str(Path(zarr_path).expanduser().resolve())
        group = zarr.open_group(zarr.DirectoryStore(self.path), mode="r")
        fields = group["data"]
        self.data = {key: fields[key] for key in keys}
        self.attrs = dict(group.attrs)
        self.meta = group["meta"]
        ends = np.asarray(self.meta["episode_ends"])
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
            validate_observation_shape(key, shape[1:], width=pc_dim if key in POINT_CLOUD_FIELDS else None)
            if key == "rgb" and values.dtype != np.uint8:
                raise ValueError("Stored RGB must have dtype uint8")
            if key in FIXED_OBSERVATION_SHAPES and values.dtype.kind != "f":
                raise ValueError(f"Stored {key} must be floating point; observations are read as FP32")
        self.episode_ends = ends.astype(np.int64, copy=False)
        if self.attrs.get("format") == "dexmani.real.canonical":
            sources = self.attrs.get("source_episodes")
            if (
                not isinstance(sources, list) or len(sources) != self.n_episodes
                or any(not isinstance(source, str) or not source for source in sources)
            ):
                raise ValueError(f"{self.path}: source_episodes must identify every Real segment")
            self.episode_groups = sources
        else:
            # Current simulation stores contain independent episodes.
            self.episode_groups = list(range(self.n_episodes))

    def read(self, key, rows, *, columns=None):
        arr = self[key]
        selection = rows if columns is None else (rows,) + (slice(None),) * (arr.ndim - 2) + (columns,)
        dtype = np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype
        return np.asarray(arr[selection], dtype=dtype)

    def read_rows_by_chunk(self, key, rows, *, columns=None):
        # A single Zarr selection would load every requested compressed chunk before decoding.
        step = self[key].chunks[0]
        order = rows.argsort()
        chunks = rows[order] // step
        splits = np.flatnonzero(chunks[1:] != chunks[:-1]) + 1
        output = None
        for positions in np.split(order, splits):
            selected = rows[positions]
            start = selected[0]
            block = self.read(key, slice(start, selected[-1] + 1), columns=columns)
            if output is None:
                output = np.empty((len(rows), *block.shape[1:]), dtype=block.dtype)
            output[positions] = block[selected - start]
        return output

    def iter_chunks(self, key, *, episode_mask, columns=None):
        step = self[key].chunks[0]
        # Merge adjacent episodes, then bound reads by both selected spans and chunks.
        first = np.flatnonzero(episode_mask & ~np.r_[False, episode_mask[:-1]])
        last = np.flatnonzero(episode_mask & ~np.r_[episode_mask[1:], False])
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

    @property
    def source_path(self):
        return self.attrs.get("source_zarr_path", self.path)

    def source_groups(self, episode_mask=None):
        return {
            source for index, source in enumerate(self.episode_groups)
            if episode_mask is None or episode_mask[index]
        }

    def count_groups(self, episode_mask=None):
        return len(self.source_groups(episode_mask))

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
        f"source={buffer.episode_groups[episode]!r}, "
        f"row={row}, channel={tuple(location[1:])}"
    )
