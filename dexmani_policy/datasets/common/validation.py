import numpy as np


def check_finite(buffer, key, start, values, columns=None):
    invalid = ~np.isfinite(values)
    if not invalid.any():
        return
    location = list(np.unravel_index(invalid.argmax(), invalid.shape))
    row = int(start + location[0])
    episode = int(np.searchsorted(buffer.episode_ends, row, side="right"))
    if columns is not None:
        location[-1] = int(np.arange(buffer[key].shape[-1])[columns][location[-1]])
    raise ValueError(
        f"Nonfinite data: path={buffer.path}, field={key}, episode={episode}, "
        f"row={row}, channel={tuple(location[1:])}"
    )


def validate_fields(buffer, episode_mask, fields):
    """One admission scan over selected real frames, independent of sample windows."""
    for key, columns in fields.items():
        if np.issubdtype(buffer[key].dtype, np.floating):
            for start, values in buffer.iter_chunks(key, columns=columns, episode_mask=episode_mask):
                check_finite(buffer, key, start, values, columns)
