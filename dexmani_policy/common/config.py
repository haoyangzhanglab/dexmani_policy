import os
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTILE_FEATURE_SHAPES = {
    "finger_tactile_force": (5, 3),
    "taxel_tactile_force": (5, 120, 3),
}
FIELD_FEATURE_SHAPES = {
    "fingertip_xyz": (5, 3),
    "camera_intrinsic": (9,),
    "camera_extrinsic": (12,),
    **TACTILE_FEATURE_SHAPES,
}
POINT_CLOUD_FIELDS = frozenset({"point_cloud", "hand_pointcloud", "object_pointcloud"})


def set_project_root() -> None:
    os.chdir(PROJECT_ROOT)


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_future_force_offsets(values) -> tuple[int, ...]:
    offsets = tuple(positive_int(value, "future_force_offset") for value in values)
    if not offsets or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("future_force_offsets must be nonempty, positive and strictly increasing")
    return offsets


def validate_action_window(horizon, n_obs_steps, n_action_steps, *, action_start_index) -> None:
    for name, value in (
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
        ("n_action_steps", n_action_steps),
    ):
        positive_int(value, name)
    if action_start_index + n_action_steps > horizon:
        raise ValueError(
            f"action_start_index + n_action_steps ({action_start_index + n_action_steps}) "
            f"exceeds horizon ({horizon})"
        )


def require_tactile_source(fields: Iterable[str], source: str | None) -> str | None:
    """Only force observations require an explicit production source."""
    if TACTILE_FEATURE_SHAPES.keys() & set(fields):
        if source not in ("real", "sim"):
            raise ValueError("Tactile observations require tactile_source=real or sim")
        return source
    return None


def validate_observation_fields(agent, declared):
    consumed = set(agent.consumed_observation_fields)
    declared = set(declared)
    missing = consumed - declared
    unused = declared - consumed
    if missing or unused:
        raise ValueError(
            f"Model observation mismatch: missing={sorted(missing)}, unconsumed={sorted(unused)}"
        )
