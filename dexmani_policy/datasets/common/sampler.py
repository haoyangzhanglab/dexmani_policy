import numba
import numpy as np
from typing import Optional
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer

######################################################################################################
#   从 ReplayBuffer中按episode安全地切出固定长度的时序片段，并支持训练/验证划分、随机下采样以及边界padding的序列采样
######################################################################################################

@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, 
    sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, 
    pad_after: int=0,
    debug:bool=True,
) -> np.ndarray:
    
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue

        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        if max_start < min_start:
            min_required_length = sequence_length - pad_before - pad_after
            raise ValueError(
                f"Episode {i} is too short: length={episode_length}, "
                f"min_required={min_required_length} "
                f"(sequence_length={sequence_length} - pad_before={pad_before} - pad_after={pad_after})"
            )

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0) and (end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
            
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    train_mask = mask

    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train

    return train_mask



class SequenceSampler:
    def __init__(
        self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
    ):
        super().__init__()

        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if not np.any(episode_mask):
            raise ValueError(
                f"All episodes are masked out. Cannot create dataset. "
                f"episode_mask.sum()=0, n_episodes={len(episode_ends)}"
            )

        indices = create_indices(episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask
            )

        self.keys = list(keys)
        self.indices = indices 
        self.key_first_k = key_first_k
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        
            
    def __len__(self):
        return len(self.indices)


    def sample_sequence(self, idx):
        result = dict()
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to copy replay buffer key '{key}' into a truncated sample: "
                        f"slice=[{buffer_start_idx}:{buffer_start_idx + k_data}], "
                        f"sample_shape={sample.shape}, buffer_shape={input_arr.shape}."
                    ) from e

            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data

        return result
