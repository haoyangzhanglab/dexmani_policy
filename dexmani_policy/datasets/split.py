import numpy as np

from dexmani_policy.common.config import positive_int


def split_episodes(
    n_episodes: int,
    *,
    seed: int = 42,
    max_train_episodes: int | None = None,
    val_episodes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    positive_int(n_episodes, "n_episodes")
    if max_train_episodes is not None:
        positive_int(max_train_episodes, "max_train_episodes")
    if type(val_episodes) is not int or val_episodes < 0:
        raise ValueError("val_episodes must be a nonnegative integer")
    train_count = (n_episodes - val_episodes if max_train_episodes is None
                   else min(max_train_episodes, n_episodes))
    if train_count <= 0 or train_count + val_episodes > n_episodes:
        raise ValueError(
            f"Cannot select {train_count} train and {val_episodes} val episodes from {n_episodes}; "
            "require nonempty train and enough remaining episodes for val"
        )
    order = np.random.default_rng(seed).permutation(n_episodes)
    train_mask = np.zeros(n_episodes, dtype=bool)
    val_mask = np.zeros(n_episodes, dtype=bool)
    train_mask[order[:train_count]] = True
    val_mask[order[train_count:train_count + val_episodes]] = True
    return train_mask, val_mask
