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

        self.indices = window_indices(
            episode_store.episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )
        self.episode_store = episode_store
        self.sequence_length = sequence_length
        self.fields = dict.fromkeys(episode_store.keys()) if fields is None else dict(fields)
        self.field_horizons = dict(field_horizons or {})

    def __len__(self):
        return len(self.indices)

    def sample_source_row(self, idx, offset):
        start, end, sample_start = self.indices[idx]
        return int(min(max(start + offset - sample_start, start), end - 1))

    def sample_source_rows(self, idx):
        """Original source rows, including repeated boundary rows used by padding."""
        start, end, sample_start = self.indices[idx].T[..., None]
        return np.clip(
            start + np.arange(self.sequence_length) - sample_start,
            start,
            end - 1,
        )

    def sample_batch(self, indices: list[int]) -> list[dict[str, np.ndarray]]:
        samples: list[dict[str, np.ndarray]] = [{} for _ in indices]
        if not samples:
            return samples

        source_rows = self.sample_source_rows(indices)
        for key, columns in self.fields.items():
            length = self.field_horizons.get(key, self.sequence_length)
            rows = source_rows[:, :length]
            values = self.episode_store.read_rows_by_chunk(key, rows.ravel(), columns=columns)
            for sample, value in zip(samples, values.reshape(*rows.shape, *values.shape[1:])):
                sample[key] = value
        return samples

    def sample_sequence(self, idx):
        result = {}
        start, end, sample_start = self.indices[idx]
        for key, columns in self.fields.items():
            length = self.field_horizons.get(key, self.sequence_length)
            # Even a horizon wholly inside left padding needs the first source frame.
            before = min(sample_start, length - 1)
            stop = min(end, start + length - before)
            values = self.episode_store.read(key, slice(start, stop), columns=columns)
            if len(values) != length:
                padded = np.empty((length, *values.shape[1:]), dtype=values.dtype)
                padded[:before] = values[0]
                padded[before:before + len(values)] = values
                padded[before + len(values):] = values[-1]
                values = padded
            result[key] = values
        return result
