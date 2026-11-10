import numpy as np

from dexmani_policy.datasets.episode_store import EpisodeStore


def window_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    episode_starts = np.r_[0, episode_ends[:-1]]
    counts = np.maximum(episode_ends - episode_starts - sequence_length + pad_before + pad_after + 1, 0)
    counts[~episode_mask] = 0
    indices = np.empty((counts.sum(), 3), dtype=np.int64)
    cursor = 0
    for episode in np.flatnonzero(counts):
        start_idx, end_idx = episode_starts[episode], episode_ends[episode]
        window_starts = np.arange(-pad_before, counts[episode] - pad_before, dtype=np.int64)
        next_cursor = cursor + counts[episode]
        indices[cursor:next_cursor, 0] = np.maximum(window_starts, 0) + start_idx
        indices[cursor:next_cursor, 1] = np.minimum(window_starts + sequence_length + start_idx, end_idx)
        indices[cursor:next_cursor, 2] = np.maximum(-window_starts, 0)
        cursor = next_cursor

    return indices


class SequenceSampler:
    def __init__(
        self,
        episode_store: EpisodeStore,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        *,
        episode_mask: np.ndarray,
        fields: dict[str, slice | None] | None = None,
        field_horizons: dict[str, int] | None = None,
    ):
        if sequence_length < 1 or not 0 <= pad_before < sequence_length or not 0 <= pad_after < sequence_length:
            raise ValueError("Require positive sequence_length and 0 <= padding < sequence_length")

        episode_ends = episode_store.episode_ends
        episode_mask = np.asarray(episode_mask, dtype=bool)
        if episode_mask.shape != episode_ends.shape:
            raise ValueError("episode_mask must have one entry per episode")

        self.indices = window_indices(
            episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )
        self.episode_store = episode_store
        self.sequence_length = sequence_length
        self.fields = dict.fromkeys(episode_store.keys()) if fields is None else dict(fields)
        self.field_horizons = dict(field_horizons or {})
        if any(key not in self.fields or not 1 <= k <= sequence_length for key, k in self.field_horizons.items()):
            raise ValueError("field_horizons requires selected keys and 1 <= K <= sequence_length")

    def __len__(self):
        return len(self.indices)

    def sample_source_rows(self, idx):
        """Original source rows, including repeated boundary rows used by padding."""
        start, end, sample_start = self.indices[idx]
        return np.clip(
            start + np.arange(self.sequence_length) - sample_start,
            start,
            end - 1,
        )

    def sample_sequence(self, idx):
        result = {}
        rows = self.sample_source_rows(idx)
        for key, columns in self.fields.items():
            length = self.field_horizons.get(key, self.sequence_length)
            selected = rows[:length]
            start, end = int(selected[0]), int(selected[-1]) + 1
            values = self.episode_store.read(key, slice(start, end), columns=columns)
            result[key] = values if end - start == length else values[selected - start]
        return result
