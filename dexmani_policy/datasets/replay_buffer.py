import logging
import os
from functools import cached_property
from typing import Optional

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def validate_episode_metadata(episode_ends, data, keys=None):
    ends = np.asarray(episode_ends)
    if (ends.ndim != 1 or ends.size == 0 or ends.dtype.kind not in "iu"
            or ends[0] <= 0 or np.any(ends[1:] <= ends[:-1])):
        raise ValueError("episode_ends must be nonempty 1D positive, strictly increasing integers (not bool)")
    for key in data.keys() if keys is None else keys:
        if key not in data:
            raise ValueError(f"Missing data field: {key}")
        shape = data[key].shape
        if not shape or shape[0] != ends[-1]:
            raise ValueError(f"Data field {key!r}: shape={shape}, expected time dimension {ends[-1]}")


class ReplayBuffer:
    def __init__(self, root):
        if "data" not in root or "meta" not in root:
            raise ValueError(
                f"Invalid root structure: missing 'data' or 'meta'. Available keys: {list(root.keys())}"
            )
        if "episode_ends" not in root["meta"]:
            raise ValueError(
                f"Invalid root structure: missing 'episode_ends' in meta. "
                f"Available meta keys: {list(root['meta'].keys())}"
            )
        validate_episode_metadata(root["meta"]["episode_ends"], root["data"])
        attrs = root.get("attrs", {}) if isinstance(root, dict) else root.attrs
        revision = attrs.get("data_revision")
        if "data_revision" in attrs and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("Zarr data_revision must be a nonempty string when present")
        self.data_revision = revision
        self.root = root

    @classmethod
    def copy_from_path(cls, zarr_path, keys: Optional[list] = None):
        group = zarr.open(os.path.expanduser(zarr_path), "r")

        meta = {}
        for key, value in group["meta"].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]

        if keys is None:
            keys = list(group["data"].keys())
        validate_episode_metadata(meta["episode_ends"], group["data"], keys)
        data = {}
        for key in keys:
            arr = group["data"][key]
            arr_data = arr[:]
            if arr_data.dtype != np.float32 and np.issubdtype(arr_data.dtype, np.floating):
                data[key] = arr_data.astype(np.float32)
            else:
                data[key] = arr_data

        buffer = cls(root={"meta": meta, "data": data, "attrs": dict(group.attrs)})
        for key, value in buffer.items():
            logger.info(
                "%-12s  shape=%-12s  dtype=%-8s  range %7.2f ~ %7.2f",
                key,
                str(value.shape),
                str(value.dtype),
                float(value.min()),
                float(value.max()),
            )
        logger.info("-" * 55)
        return buffer

    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
