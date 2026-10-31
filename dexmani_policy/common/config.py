import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: LAUNCH_ID, replace=True)


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
        return assemble_policy_config(hydra.compose(config_name=name, overrides=list(overrides)))


def assemble_policy_config(cfg):
    """Pass research inputs to consumers and derive fixed data/model dimensions once."""
    if "policy_name" not in cfg:
        return cfg
    with open_dict(cfg):
        action_key = cfg.action_key
        if action_key not in ("action", "action_ee"):
            raise ValueError("action_key must be action or action_ee")
        auxiliary = cfg.policy_name == "r3d" and cfg.use_aux_ee
        if auxiliary and action_key != "action":
            raise ValueError("R3D auxiliary targets require joint action")
        cfg.action_dim = 28 if auxiliary else (21 if action_key == "action_ee" else 19)
        for key in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
            cfg.agent[key] = cfg[key]
        cfg.dataset.update(action_key=action_key, zarr_path=cfg.zarr_path,
                           horizon=cfg.horizon, obs_horizon=cfg.n_obs_steps,
                           pad_before=cfg.n_obs_steps - 1, pad_after=cfg.n_action_steps - 1)
        if cfg.policy_name == "r3d":
            cfg.dataset["_target_"] = ("dexmani_policy.datasets.variants.r3d.R3DDataset" if auxiliary
                                      else "dexmani_policy.datasets.demonstration.DemonstrationDataset")
        if "pc_dim" in cfg.agent:
            cfg.dataset.pc_dim = cfg.agent.pc_dim
        if "future_force_offsets" in cfg:
            cfg.agent.future_force_offsets = cfg.future_force_offsets
            cfg.dataset.future_force_offsets = cfg.future_force_offsets
        if "tactile_source" in cfg:
            cfg.agent.tactile_source = cfg.tactile_source
            cfg.dataset.tactile_source = cfg.tactile_source
        cfg.env_runner.task_name = cfg.task_name
        cfg.env_runner.env_kwargs = {**(cfg.env_runner.env_kwargs or {}), "control_mode": "ee" if action_key == "action_ee" else "joint"}
    return cfg


def read_split_manifest(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).expanduser().read_text())
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


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


def validate_action_mode(cfg) -> None:
    import hydra

    cls = hydra.utils.get_class(cfg["agent"]["_target_"])
    if cfg["action_key"] not in cls.supported_action_keys:
        raise ValueError(f"{cls.__name__} does not support {cfg['action_key']}")


def validate_normalization_config(cfg):
    spec = cfg["normalization"]
    for key, mode in spec.items():
        if mode not in {"identity", "limits", "gaussian", "auto", "rms", "max_abs"}:
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

    if cfg["normalization"]["action"] not in (("auto", "limits") if cfg["action_key"] == "action" else ("auto",)):
        raise ValueError("Action normalization is fixed by the action representation")
    method = cfg["policy_name"]
    if method == "r3d" and spec["point_cloud"] != "limits":
        raise ValueError("R3D requires limits normalization before its clamp")
    if method in {"sat", "dqrise", "adaptac", "interact-dex"} and spec["point_cloud"] != "identity":
        raise ValueError("Metric geometry requires normalization.point_cloud=identity")
    if method in {"sat", "dqrise"}:
        for key in ("rgb", "camera_intrinsic", "camera_extrinsic"):
            if spec[key] != "identity":
                raise ValueError(f"Calibrated geometry requires normalization.{key}=identity")
        if spec["joint_state"] != "limits":
            raise ValueError("Spatial joint_state requires limits normalization")
    if method == "interact-dex":
        for key in ("eef_pose", "fingertip_xyz"):
            if spec[key] != "identity":
                raise ValueError(f"Metric geometry requires normalization.{key}=identity")
    if "finger_tactile_force" in spec and spec["finger_tactile_force"] not in {"rms", "max_abs"}:
        raise ValueError("finger_tactile_force requires rms or max_abs")
    if "taxel_tactile_force" in spec and spec["taxel_tactile_force"] not in {"rms", "max_abs"}:
        raise ValueError("taxel_tactile_force requires rms or max_abs")


def require_tactile_source(fields: Iterable[str], source: str | None) -> str | None:
    """Only force observations require an explicit production source."""
    if TACTILE_FEATURE_SHAPES.keys() & set(fields):
        if source not in ("real", "sim"):
            raise ValueError("Tactile observations require tactile_source=real or sim")
        return source
    return None


def normalization_feature_shapes(dims):
    return {
        key: FIELD_FEATURE_SHAPES.get(key, (width,))
        for key, width in dims.items()
    }


def validate_observation_fields(agent, declared):
    consumed = set(agent.consumed_observation_fields)
    declared = set(declared)
    missing = consumed - declared
    unused = declared - consumed
    if missing or unused:
        raise ValueError(
            f"Model observation mismatch: missing={sorted(missing)}, unconsumed={sorted(unused)}"
        )
