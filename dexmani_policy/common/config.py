import math
import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FINGER_VECTOR_FIELDS = frozenset({"fingertip_xyz", "finger_tactile_force"})
POINT_CLOUD_FIELDS = frozenset({"point_cloud", "hand_pointcloud", "object_pointcloud"})


def set_project_root() -> None:
    os.chdir(PROJECT_ROOT)


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: LAUNCH_ID, replace=True)
    OmegaConf.register_new_resolver(
        "eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True
    )
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def load_policy_config(config, overrides=()):
    import hydra

    register_resolvers()
    root = PROJECT_ROOT / "dexmani_policy" / "configs"
    path = Path(config).expanduser()
    if path.is_file():
        path = path.resolve()
        if not path.is_relative_to(root):
            root = path.parent
        name = path.relative_to(root).with_suffix("").as_posix()
    else:
        name = str(config)
    with hydra.initialize_config_dir(config_dir=str(root), version_base=None):
        return hydra.compose(config_name=name, overrides=list(overrides))


def load_experiment_config(experiment_dir):
    return OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


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


def validate_action_mode(cfg) -> None:
    action_key = cfg["action_key"]
    expected_control = {"action": "joint", "action_ee": "ee"}[action_key]
    env_runner = cfg["env_runner"]
    env_kwargs = env_runner.get("env_kwargs") or {}
    actual_control = env_kwargs.get("control_mode", "joint")
    if actual_control != expected_control:
        raise ValueError(
            f"Task {env_runner.get('task_name')!r}: action_key={action_key!r} requires "
            f"control_mode={expected_control!r}, got {actual_control!r}"
        )

    ds_action_key = cfg["dataset"]["action_key"]
    if ds_action_key != action_key:
        raise ValueError(
            f"dataset.action_key={ds_action_key!r} differs from action_key={action_key!r}"
        )


def validate_normalization_config(cfg):
    spec = cfg["normalization"]
    for key, mode in spec.items():
        if mode not in {"identity", "limits", "gaussian", "auto"}:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        if mode == "auto" and key != "action":
            raise ValueError("Normalization auto is only supported for action")
    if spec.get("rgb", "identity") != "identity":
        raise ValueError("RGB preprocessing requires normalization.rgb=identity")
    expected = set(cfg["dataset"]["sensor_modalities"]) | {"action"}
    if spec.keys() != expected:
        raise ValueError(
            f"Normalization field mismatch: missing={sorted(expected - spec.keys())}, "
            f"extra={sorted(spec.keys() - expected)}"
        )


def normalization_feature_shapes(dims):
    return {
        key: (5, 3) if key in FINGER_VECTOR_FIELDS else (width,)
        for key, width in dims.items()
    }


def validate_depth_recipe(recipe):
    shape = recipe["shape"]
    if len(shape) != 2:
        raise ValueError("Depth recipe shape must be [H,W]")
    for value in shape:
        positive_int(value, "depth.shape")
    scale = recipe["depth_scale_m"]
    if (
        recipe["dtype"] != "uint16"
        or recipe["unit"] != "raw_counts"
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Depth requires uint16 raw_counts and finite positive depth_scale_m")
    return recipe


def validate_observation_fields(agent, declared):
    consumed = set(agent.consumed_observation_fields)
    declared = set(declared)
    missing = consumed - declared
    unused = declared - consumed
    if missing or unused:
        raise ValueError(
            f"Model observation mismatch: missing={sorted(missing)}, unconsumed={sorted(unused)}"
        )
