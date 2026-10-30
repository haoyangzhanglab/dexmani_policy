"""Scan stored floating-point fields in chunks and locate nonfinite values."""

import argparse

import numpy as np

from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


def check_data(zarr_path):
    store = EpisodeStore(zarr_path)
    for key in store.keys():
        if np.issubdtype(store[key].dtype, np.floating):
            for start, values in store.iter_chunks(key):
                check_finite(store, key, start, values)
            print(f"Finite: {key}")
    print(f"Checked {store.n_episodes} episodes: {store.path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True)
    args = parser.parse_args()
    check_data(args.zarr_path)


if __name__ == "__main__":
    main()
