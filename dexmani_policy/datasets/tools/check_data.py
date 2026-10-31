"""Check nonempty stored fields and finite values after training's FP32 conversion."""

import argparse

import numpy as np

from dexmani_policy.datasets.episode_store import EpisodeStore, check_finite


def check_data(zarr_path):
    store = EpisodeStore(zarr_path)
    for key in store.keys():
        if any(size == 0 for size in store[key].shape):
            raise ValueError(f"Empty data field: path={store.path}, field={key}, shape={store[key].shape}")
        if np.issubdtype(store[key].dtype, np.floating):
            # EpisodeStore.read uses the same FP32 conversion as training.
            with np.errstate(over="ignore"):
                for start, values in store.iter_chunks(key):
                    check_finite(store, key, start, values)
            print(f"Finite after FP32 conversion: {key}")
    print(f"Checked {store.n_episodes} episodes: {store.path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True)
    args = parser.parse_args()
    check_data(args.zarr_path)


if __name__ == "__main__":
    main()
