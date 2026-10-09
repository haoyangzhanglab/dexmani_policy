import math
from numbers import Integral, Real

import numba
import numpy as np

from dexmani_policy.datasets.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    # Internal Numba kernel: SequenceSampler validates parameters and filters mask.
    assert episode_mask.shape == episode_ends.shape

    indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue

        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        assert max_start >= min_start

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            assert start_offset >= 0
            assert end_offset >= 0
            assert (sample_end_idx - sample_start_idx) == (
                buffer_end_idx - buffer_start_idx
            )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )

    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    validate_val_ratio(val_ratio)
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    validate_max_train_episodes(max_n)
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
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: np.ndarray | None = None,
    ):
        super().__init__()

        if (isinstance(sequence_length, bool)
                or not isinstance(sequence_length, Integral) or sequence_length < 1):
            raise ValueError("sequence_length must be a positive integer (not bool)")
        for name, value in (("pad_before", pad_before), ("pad_after", pad_after)):
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not 0 <= value < sequence_length):
                raise ValueError(f"{name} must be an integer with 0 <= {name} < sequence_length")
        sequence_length, pad_before, pad_after = map(int, (sequence_length, pad_before, pad_after))

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        else:
            episode_mask = np.array(episode_mask, dtype=bool, copy=True)

        if not np.any(episode_mask):
            raise ValueError(
                f"All episodes are masked out. Cannot create dataset. "
                f"episode_mask.sum()=0, n_episodes={len(episode_ends)}"
            )

        min_required_length = sequence_length - pad_before - pad_after
        total_masked = int(episode_mask.sum())
        for i in range(len(episode_ends)):
            if not episode_mask[i]:
                continue
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            episode_length = episode_ends[i] - start_idx
            if episode_length < min_required_length:
                episode_mask[i] = False

        n_skipped = total_masked - int(episode_mask.sum())
        if n_skipped > 0:
            print(
                f"SequenceSampler: skipped {n_skipped}/{total_masked} episodes "
                f"({100 * n_skipped / total_masked:.1f}%) — too short "
                f"(min required={min_required_length} frames, "
                f"sequence={sequence_length} - pad_before={pad_before} - pad_after={pad_after})"
            )

        if not np.any(episode_mask):
            raise ValueError(
                f"All episodes filtered out. "
                f"min_required_length={min_required_length}, n_episodes={len(episode_ends)}"
            )

        indices = create_indices(
            episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )

        self.keys = list(replay_buffer.keys())
        self.indices = indices
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.indices)

    def source_rows(self, indices=None):
        """Original source rows, including repeated boundary rows used by padding."""
        indices = self.indices if indices is None else indices
        start, end, sample_start, _ = np.asarray(indices).T
        return np.clip(
            start[:, None] + np.arange(self.sequence_length) - sample_start[:, None],
            start[:, None],
            end[:, None] - 1,
        )

    def filter_valid(self, obs_valid, action_valid, n_obs_steps):
        # Process bounded batches: a full RGB buffer or windows×horizon copy is unnecessary.
        kept = []
        counts = {
            "candidate": len(self.indices),
            "observation": 0,
            "action": 0,
        }
        obs_rows = np.zeros(len(obs_valid), dtype=bool)
        action_rows = np.zeros(len(action_valid), dtype=bool)
        for offset in range(0, len(self.indices), 8192):
            indices = self.indices[offset : offset + 8192]
            rows = self.source_rows(indices)
            obs_ok = obs_valid[rows[:, :n_obs_steps]].all(axis=1)
            action_ok = action_valid[rows].all(axis=1)
            for name, ok in (
                ("observation", obs_ok),
                ("action", action_ok),
            ):
                counts[name] += int((~ok).sum())
            keep = obs_ok & action_ok
            kept.append(indices[keep])
            obs_rows[rows[keep, :n_obs_steps].reshape(-1)] = True
            action_rows[rows[keep].reshape(-1)] = True
        self.indices = np.concatenate(kept, axis=0)
        counts["valid"] = len(self.indices)
        self.validity_summary = counts
        self.observation_source_rows = np.flatnonzero(obs_rows)
        self.action_source_rows = np.flatnonzero(action_rows)
        print(f"SequenceSampler window validity (rejections may overlap): {counts}")
        if not len(self.indices):
            raise ValueError(f"Zero valid training/validation windows: {counts}")

    def sample_sequence(self, idx, *, key_lengths=None):
        result = {}
        source = self.source_rows(self.indices[idx : idx + 1])[0]
        for key in self.keys:
            length = self.sequence_length if key_lengths is None else key_lengths[key]
            rows = source[:length]
            start, end = int(rows[0]), int(rows[-1]) + 1
            values = self.replay_buffer.read(key, slice(start, end))
            result[key] = values if end - start == len(rows) else values[rows - start]
        return result


def validate_val_ratio(val_ratio) -> None:
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, Real)
        or not math.isfinite(val_ratio)
        or not 0 <= val_ratio < 1
    ):
        raise ValueError(
            f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio!r}"
        )


def validate_max_train_episodes(value) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError(
            f"max_train_episodes must be None or a positive integer, got {value!r}"
        )


def validate_dataset_splits(dataset) -> None:
    """Validate split options without constructing datasets or reading Zarr."""
    validate_val_ratio(dataset.get("val_ratio", 0.0))
    validate_max_train_episodes(dataset.get("max_train_episodes"))
    for child in dataset.get("datasets", []):
        validate_dataset_splits(child)
