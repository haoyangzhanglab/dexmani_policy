import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTILE_FEATURE_SHAPES = {
    "finger_tactile_force": (5, 3),
    "taxel_tactile_force": (5, 120, 3),
}
FIXED_OBSERVATION_SHAPES: dict[str, tuple[int, ...]] = {
    **TACTILE_FEATURE_SHAPES,
    "joint_state": (19,),
    "eef_pose": (9,),
    "fingertip_xyz": (5, 3),
    "camera_intrinsic": (9,),
    "camera_extrinsic": (12,),
}
POINT_CLOUD_FIELDS = frozenset({"point_cloud", "hand_pointcloud", "object_pointcloud"})


def validate_ddim_steps(steps: int, name: str) -> None:
    if positive_int(steps, name) == 1:
        raise ValueError(f"{name} must be at least 2 for leading DDIM; 1 NFE starts at t=0")


def validate_observation_shape(name: str, shape: tuple[int, ...], *, width: int | None = None) -> None:
    """Validate one frame before point-cloud channel selection."""
    if name in FIXED_OBSERVATION_SHAPES:
        expected = FIXED_OBSERVATION_SHAPES[name]
        if shape != expected:
            raise ValueError(f"{name} requires frame shape {expected}, got {shape}")
    elif name in POINT_CLOUD_FIELDS:
        channels = 1 if width is None else width
        if len(shape) != 2 or not shape[0] or shape[1] < channels:
            raise ValueError(f"{name} requires frame shape (N>0, C>={channels}), got {shape}")
    elif name == "rgb":
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0:
            raise ValueError(f"RGB requires frame shape (H, W, 3), got {shape}")
    elif name == "proprio" and (len(shape) != 1 or not shape[0]):
        raise ValueError(f"proprio requires one vector per frame, got {shape}")
    elif width is not None and (not shape or shape[-1] != width):
        raise ValueError(f"{name} requires feature width {width}, got frame shape {shape}")


def set_project_root() -> None:
    os.chdir(PROJECT_ROOT)


def positive_float(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_action_window(horizon: int, n_obs_steps: int, n_action_steps: int) -> None:
    for name, value in (
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
        ("n_action_steps", n_action_steps),
    ):
        positive_int(value, name)
    future_steps = horizon - n_obs_steps + 1
    if n_action_steps > future_steps:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) exceeds future_steps ({future_steps}) "
            "for the configured horizon and n_obs_steps"
        )
