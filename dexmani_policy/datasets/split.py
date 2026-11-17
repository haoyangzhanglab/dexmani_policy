import numpy as np

from dexmani_policy.common.config import positive_int


def format_split_summary(store, train_mask, val_mask):
    parts = [f"source groups={store.count_groups():,}, segments={store.n_episodes:,}"]
    for name, mask in (("train", train_mask), ("val", val_mask), ("unused", ~(train_mask | val_mask))):
        parts.append(f"{name} groups={store.count_groups(mask):,}/segments={int(mask.sum()):,}")
    return "; ".join(parts)


def split_episodes(
    episode_groups,
    *,
    seed: int = 42,
    max_train_episodes: int | None = None,
    val_episodes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # A recorded source can contain several stored continuous segments.
    groups = {source: index for index, source in enumerate(dict.fromkeys(episode_groups))}
    n_groups = len(groups)
    positive_int(n_groups, "number of source groups")
    if max_train_episodes is not None:
        positive_int(max_train_episodes, "max_train_episodes")
    if type(val_episodes) is not int or val_episodes < 0:
        raise ValueError("val_episodes must be a nonnegative integer")
    if val_episodes >= n_groups:
        raise ValueError("val_episodes must be smaller than the number of source groups")
    available = n_groups - val_episodes
    train_count = available if max_train_episodes is None else min(max_train_episodes, available)
    order = np.random.default_rng(seed).permutation(n_groups)
    train_mask = np.zeros(n_groups, dtype=bool)
    val_mask = np.zeros(n_groups, dtype=bool)
    train_mask[order[:train_count]] = True
    val_mask[order[available:]] = True
    indices = np.asarray([groups[source] for source in episode_groups], dtype=np.int64)
    return train_mask[indices], val_mask[indices]
