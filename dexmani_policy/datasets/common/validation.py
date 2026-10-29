import math

import numpy as np


def validate_tactile_source(attrs, source):
    if source == "real":
        if attrs.get("format") != "dexmani.real.canonical":
            raise ValueError("tactile_source=real requires dexmani.real.canonical data")
    elif source == "sim":
        required = {
            "task_name", "dt", "physics_dt", "frame_skip", "action_dim",
            "action_space", "obs_alignment", "robot_setup_json",
        }
        missing = required - attrs.keys()
        if attrs.get("format") is not None or missing:
            raise ValueError(
                "tactile_source=sim requires sim exporter metadata; "
                f"format={attrs.get('format')!r}, missing={sorted(missing)}"
            )
        physics_dt, skip = attrs["physics_dt"], attrs["frame_skip"]
        if (
            type(skip) is not int or skip <= 0
            or type(physics_dt) not in (float, int) or not math.isfinite(physics_dt) or physics_dt <= 0
            or attrs["obs_alignment"] != "obs[t]_before_action[t]"
            or attrs["action_space"] not in {"joint", "ee"} or attrs["action_dim"] != 19
        ):
            raise ValueError("Invalid sim exporter tactile/timing contract")
    else:
        raise ValueError(f"tactile_source must be sim or real, got {source!r}")
    dt = attrs.get("dt")
    if type(dt) not in (float, int) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Tactile data requires finite positive dt")
    if source == "sim" and not math.isclose(dt, physics_dt * skip, rel_tol=1e-6, abs_tol=0):
        raise ValueError("Sim dt must equal physics_dt * frame_skip")


def check_finite(buffer, key, start, values, columns=None):
    invalid = ~np.isfinite(values)
    if not invalid.any():
        return
    location = [int(index) for index in np.unravel_index(invalid.argmax(), invalid.shape)]
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
