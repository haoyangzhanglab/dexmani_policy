import numpy as np

from dexmani_policy.datasets.episode_store import check_finite


def load_hand_array(store, episode_mask):
    """Read each selected hand target once, independently of policy windows."""
    if store["action"].shape[1:] != (19,):
        raise ValueError("VQ pretraining requires action shape [T,19]")
    count = np.diff(np.r_[0, store.episode_ends])[episode_mask].sum()
    values = np.empty((count, 12), dtype=np.float32)
    cursor = 0
    for start, chunk in store.iter_chunks("action", columns=slice(7, 19), episode_mask=episode_mask):
        check_finite(store, "action", start, chunk, columns=slice(7, 19))
        values[cursor:cursor + len(chunk)] = chunk
        cursor += len(chunk)
    return values
