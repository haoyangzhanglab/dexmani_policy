import os
import zarr
import numpy as np
from termcolor import cprint
from functools import cached_property
from typing import Optional

class ReplayBuffer:
    def __init__(self, root):
        if 'data' not in root or 'meta' not in root:
            raise ValueError(
                f"Invalid root structure: missing 'data' or 'meta'. "
                f"Available keys: {list(root.keys())}"
            )
        if 'episode_ends' not in root['meta']:
            raise ValueError(
                f"Invalid root structure: missing 'episode_ends' in meta. "
                f"Available meta keys: {list(root['meta'].keys())}"
            )
        for key, value in root['data'].items():
            if value.shape[0] != root['meta']['episode_ends'][-1]:
                raise ValueError(
                    f"Data shape mismatch for key '{key}': "
                    f"shape[0]={value.shape[0]}, expected={root['meta']['episode_ends'][-1]}"
                )
        self.root = root

    @classmethod
    def copy_from_path(cls, zarr_path, keys: Optional[list] = None):
        group = zarr.open(os.path.expanduser(zarr_path), 'r')

        meta = dict()
        for key, value in group['meta'].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]

        if keys is None:
            keys = list(group['data'].keys())
        data = dict()
        for key in keys:
            arr = group['data'][key]
            arr_data = arr[:]
            if arr_data.dtype != np.float32 and np.issubdtype(arr_data.dtype, np.floating):
                data[key] = arr_data.astype(np.float32)
            else:
                data[key] = arr_data

        buffer = cls(root={'meta': meta, 'data': data})
        for key, value in buffer.items():
            cprint(
                f'Replay Buffer: {key}, shape {value.shape}, dtype {value.dtype}, '
                f'range {value.min():.2f}~{value.max():.2f}', 'green'
            )
        cprint("--------------------------", 'green')
        return buffer

    @cached_property
    def data(self):
        return self.root['data']

    @cached_property
    def meta(self):
        return self.root['meta']

    @property
    def episode_ends(self):
        return self.meta['episode_ends']

    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
