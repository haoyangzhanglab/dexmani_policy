import math
import os
from datetime import datetime
from numbers import Integral
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FINGER_VECTOR_FIELDS = frozenset({"fingertip_xyz", "finger_tactile_force"})
ALLOWED_NORMALIZATION_MODES = frozenset({"identity", "limits", "gaussian", "auto"})


def set_project_root() -> None:
    os.chdir(PROJECT_ROOT)


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: LAUNCH_ID, replace=True)
    OmegaConf.register_new_resolver(
        "eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True
    )
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def load_policy_config(config, overrides=()):
    """Compose a repository config name or explicit YAML with the same Hydra rules."""
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


def validate_action_window(horizon, n_obs_steps, n_action_steps, *, action_start_index=None) -> None:
    for name, value in (
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
        ("n_action_steps", n_action_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    start = n_obs_steps - 1 if action_start_index is None else action_start_index
    if start + n_action_steps > horizon:
        raise ValueError(
            f"action_start_index + n_action_steps ({start + n_action_steps}) "
            f"exceeds horizon ({horizon})"
        )


def validate_action_key_consistency(cfg) -> None:
    """Keep policy, dataset and environment action coordinates consistent."""
    action_key = cfg.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be explicitly set to 'action' or 'action_ee'")

    expected_control = "ee" if action_key == "action_ee" else "joint"
    env_runner = cfg.get("env_runner", {})
    env_kwargs = env_runner.get("env_kwargs") or {}
    actual_control = env_kwargs.get("control_mode", "joint")
    if actual_control != expected_control:
        raise ValueError(
            f"Task {env_runner.get('task_name')!r}: action_key={action_key!r} requires "
            f"control_mode={expected_control!r}, got {actual_control!r}"
        )

    ds_action_key = cfg["dataset"].get("action_key", "action")
    if ds_action_key != action_key:
        raise ValueError(
            f"dataset.action_key={ds_action_key!r} differs from action_key={action_key!r}"
        )


def load_experiment_config(experiment_dir):
    """Read the resolved snapshot without composing current repository YAML."""
    return OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )


def resolve_normalization_spec(cfg, *, observation_fields=None):
    spec = cfg["normalization"]
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    for key, mode in spec.items():
        if mode not in ALLOWED_NORMALIZATION_MODES:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        if mode == "auto" and key != "action":
            raise ValueError("Normalization auto is only supported for action")
    if not {"joint_state", "action"} <= spec.keys():
        raise ValueError("Normalization must declare joint_state and action")
    if spec.get("rgb", "identity") != "identity":
        raise ValueError("RGB preprocessing requires normalization.rgb=identity")
    if observation_fields is not None:
        expected = set(observation_fields) | {"action"}
        if spec.keys() != expected:
            raise ValueError(
                f"Normalization field mismatch: missing={sorted(expected - spec.keys())}, "
                f"extra={sorted(spec.keys() - expected)}"
            )
    return dict(spec)


def normalization_dims_from_config(cfg):
    dims = dict(cfg["normalization_dims"])
    agent = cfg["agent"]
    expected = {
        "action": agent["action_dim"],
        "eef_pose": 9,
        "fingertip_xyz": 15,
        "finger_tactile_force": 15,
        "taxel_tactile_force": 3,
        "camera_intrinsic": 9,
        "camera_extrinsic": 12,
        "tactile_valid": 5,
    }
    if "state_dim" in agent:
        expected["joint_state"] = agent["state_dim"]
    dataset = cfg["dataset"]
    if dataset.get("pc_dim") is not None:
        expected["point_cloud"] = dataset["pc_dim"]
    for key, mode in cfg["normalization"].items():
        if mode != "identity" and key in expected and dims[key] != expected[key]:
            raise ValueError(f"Saved {key} feature width must be {expected[key]}")
    return dims


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_depth_recipe(recipe):
    """Depth is stored as raw counts; the scale defines their physical unit."""
    if not isinstance(recipe, dict):
        raise ValueError("Depth requires a saved shape/dtype/unit/scale recipe")
    shape = recipe["shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
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
