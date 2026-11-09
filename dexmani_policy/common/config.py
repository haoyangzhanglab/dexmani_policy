import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTILE_FEATURE_SHAPES = {
    "finger_tactile_force": (5, 3),
    "taxel_tactile_force": (5, 120, 3),
}
FIELD_FEATURE_SHAPES = {
    "joint_state": (19,),
    "eef_pose": (9,),
    "fingertip_xyz": (5, 3),
    "camera_intrinsic": (9,),
    "camera_extrinsic": (12,),
}
POINT_CLOUD_FIELDS = frozenset({"point_cloud", "hand_pointcloud", "object_pointcloud"})


def validate_normalization_spec(fields, spec):
    fields = set(fields)
    missing = fields - spec.keys()
    unknown = spec.keys() - fields
    if missing or unknown:
        raise ValueError(f"Invalid normalization fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key, mode in spec.items():
        if mode not in {"identity", "limits"}:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}; expected identity or limits")
        if key == "action" and mode != "limits":
            raise ValueError("action must use limits")
        if key in TACTILE_FEATURE_SHAPES and mode != "identity":
            raise ValueError(f"{key} must use identity")


def set_project_root() -> None:
    os.chdir(PROJECT_ROOT)


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
