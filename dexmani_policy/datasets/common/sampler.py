import numba
import numpy as np

from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_sequence_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    count = 0
    start = 0
    for episode, end in enumerate(episode_ends):
        if episode_mask[episode]:
            count += max(0, end - start - sequence_length + pad_before + pad_after + 1)
        start = end

    indices = np.empty((count, 3), dtype=np.int64)
    cursor = 0
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue

        start_idx = episode_ends[i - 1] if i else 0
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            sample_start_idx = buffer_start_idx - (idx + start_idx)
            indices[cursor] = (buffer_start_idx, buffer_end_idx, sample_start_idx)
            cursor += 1

    return indices


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: np.ndarray | None = None,
    ):
        if sequence_length < 1 or not 0 <= pad_before < sequence_length or not 0 <= pad_after < sequence_length:
            raise ValueError("Require positive sequence_length and 0 <= padding < sequence_length")

        episode_ends = np.asarray(replay_buffer.episode_ends, dtype=np.int64)
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        else:
            episode_mask = np.asarray(episode_mask, dtype=bool)

        indices = create_sequence_indices(
            episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )
        if not len(indices):
            raise ValueError("No sequence windows: selected episodes are empty or too short")

        self.indices = indices
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.indices)

    def source_rows(self, indices=None):
        """Original source rows, including repeated boundary rows used by padding."""
        indices = self.indices if indices is None else indices
        start, end, sample_start = np.asarray(indices).T
        return np.clip(
            start[:, None] + np.arange(self.sequence_length) - sample_start[:, None],
            start[:, None],
            end[:, None] - 1,
        )

    def filter_valid(self, obs_valid, action_valid, n_obs_steps, *, action_offset=0):
        # Bound temporary source-row arrays independently of dataset length.
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
            future_rows = rows[:, action_offset:]
            action_ok = action_valid[future_rows].all(axis=1)
            for name, ok in (
                ("observation", obs_ok),
                ("action", action_ok),
            ):
                counts[name] += int((~ok).sum())
            keep = obs_ok & action_ok
            kept.append(indices[keep])
            obs_rows[rows[keep, :n_obs_steps].reshape(-1)] = True
            action_rows[future_rows[keep].reshape(-1)] = True
        self.indices = np.concatenate(kept, axis=0)
        counts["valid"] = len(self.indices)
        self.validity_summary = counts
        self.observation_source_rows = np.flatnonzero(obs_rows)
        self.action_source_rows = np.flatnonzero(action_rows)
        if not len(self.indices):
            raise ValueError(f"Zero valid training/validation windows: {counts}")

    def sample_source_rows(self, idx):
        return self.source_rows(self.indices[idx][None])[0]

    def sample_sequence(self, idx, *, key_lengths=None, key_offsets=None):
        result = {}
        rows = self.sample_source_rows(idx)
        for key in self.replay_buffer.keys():
            length = self.sequence_length if key_lengths is None else key_lengths[key]
            offset = 0 if key_offsets is None else key_offsets.get(key, 0)
            selected = rows[offset : offset + length]
            start, end = int(selected[0]), int(selected[-1]) + 1
            values = self.replay_buffer.read(key, slice(start, end))
            result[key] = values if end - start == length else values[selected - start]
        return result
