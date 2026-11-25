import numpy as np

from dexmani_policy.datasets.episode_store import EpisodeStore


def window_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """Rows are (window start before clipping, episode start, exclusive episode end)."""
    episode_starts = np.r_[0, episode_ends[:-1]]
    counts = np.maximum(episode_ends - episode_starts - sequence_length + pad_before + pad_after + 1, 0)
    counts[~episode_mask] = 0
    indices = np.empty((counts.sum(), 3), dtype=np.int64)
    cursor = 0
    for episode in np.flatnonzero(counts):
        start_idx, end_idx = episode_starts[episode], episode_ends[episode]
        window_starts = np.arange(-pad_before, counts[episode] - pad_before, dtype=np.int64)
        next_cursor = cursor + counts[episode]
        indices[cursor:next_cursor, 0] = start_idx + window_starts
        indices[cursor:next_cursor, 1] = start_idx
        indices[cursor:next_cursor, 2] = end_idx
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
        frame_stride: int = 1,
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
        self.frame_stride = frame_stride

    def __len__(self):
        return len(self.indices)

    def source_rows(self, indices, offsets):
        window, start, end = np.moveaxis(self.indices[indices], -1, 0)
        return np.clip(window + offsets, start, end - 1)

    def sample_batch(self, indices: list[int]) -> list[dict[str, np.ndarray]]:
        samples: list[dict[str, np.ndarray]] = [{} for _ in indices]
        if not samples:
            return samples

        source_rows = self.source_rows(np.asarray(indices)[:, None], np.arange(self.sequence_length))
        for key, columns in self.fields.items():
            length = self.field_horizons.get(key, self.sequence_length)
            rows = source_rows[:, :length:self.frame_stride]
            values = self.episode_store.read_rows_by_chunk(key, rows, columns=columns)
            for sample, value in zip(samples, values):
                sample[key] = value
        return samples
