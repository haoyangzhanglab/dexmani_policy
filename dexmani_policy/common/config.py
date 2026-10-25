"""Configuration loading, numerical settings and dataset path resolution."""

import argparse
import copy
import json
import math
import os
import re
from datetime import datetime
from numbers import Integral
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TASK_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]+(?:\+[a-zA-Z0-9_-]+)*")
FINGER_VECTOR_FIELDS = frozenset({"fingertip_xyz", "finger_tactile_force"})
NON_NUMERIC_OBSERVATION_FIELDS = frozenset({"task_text", "task_name"})
ALLOWED_NORMALIZATION_MODES = frozenset({"identity", "limits", "gaussian", "auto"})

ARTIFACT_LOCATION_FIELDS = {
    "lamp.prior_path",
    "codec.path",
    "codebook_path",
    "agent.prior_path",
    "agent.codec_path",
    "agent.codebook_path",
}

RESUME_OPERATIONAL_FIELDS = {
    "resume_from",
    "max_updates",
    "workspace.output_dir",
    "workspace.wandb_cfg",
    "training.device",
    "training.gpu_ids",
    "training.use_compile",
    "training.compile_mode",
    "training.loop.log_interval_steps",
    "dataloader.num_workers",
    "dataloader.persistent_workers",
    "dataloader.prefetch_factor",
}


def set_project_root() -> str:
    """Resolve entry-point relative paths from the repository root."""
    os.chdir(PROJECT_ROOT)
    return str(PROJECT_ROOT)


def resolve_resume_source(source):
    if source is None:
        return None
    path = PROJECT_ROOT / Path(source).expanduser()
    if path.is_dir():
        path = path / "checkpoints" / "latest.pt"
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Resume source must be a checkpoint file: {path}")
    return str(path)


def recipe_content(config):
    """Compare research settings, excluding supported run and location changes."""
    plain = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config

    def visit(value, path=""):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if (
                    child_path in RESUME_OPERATIONAL_FIELDS
                    or child_path in ARTIFACT_LOCATION_FIELDS
                    or child_path == "hand_data.provenance.zarr_path"
                    or re.fullmatch(r"dataset\.(?:datasets\.\d+\.)?zarr_path", child_path)
                ):
                    continue
                result[key] = visit(child, child_path)
            return result
        if isinstance(value, (list, tuple)):
            return [visit(child, f"{path}.{i}") for i, child in enumerate(value)]
        return value

    return visit(plain)


def validate_source_recipe(saved, current):
    """Require the same experiment recipe, allowing supported run/location changes."""
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    differences.append(f"{path}.{key}: missing field")
                else:
                    compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f"{path}: different lengths")
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{i}]")
        elif type(left) is not type(right) or left != right:
            differences.append(f"{path}: checkpoint={left!r}, config={right!r}")

    compare(recipe_content(saved), recipe_content(current), "config")
    if differences:
        raise ValueError("Checkpoint source recipe mismatch:\n" + "\n".join(differences))


def resolve_input_recipe(cfg, *, overrides, resume_checkpoint=None):
    """Use the saved recipe on resume, applying only explicit operational/location overrides."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    source = resolve_resume_source(cfg.get("resume_from"))
    if source is None:
        return copy.deepcopy(cfg)
    if resume_checkpoint is None:
        from dexmani_policy.common.checkpoint import read_checkpoint

        source_config = read_checkpoint(source)["effective_config"]
    else:
        source_config = resume_checkpoint.effective_config
    result = OmegaConf.create(source_config)
    with open_dict(result):
        # The execution limit belongs to this invocation.
        result.pop("max_updates", None)
        for override in OverridesParser.create().parse_overrides(list(overrides)):
            key = override.key_or_group
            if key.startswith("hydra."):
                continue
            if override.is_delete() or override.is_sweep_override():
                raise ValueError(f"Unsupported resume override: {override.input_line}")
            if key == "task_name":
                requested, identity = override.value(), result.get("task_name")
                if (
                    not isinstance(requested, str)
                    or not TASK_NAME_PATTERN.fullmatch(requested)
                    or requested != identity
                ):
                    raise ValueError(
                        f"Resume task identity mismatch: requested={requested!r}, saved={identity!r}"
                    )
                continue
            data_path = re.fullmatch(r"dataset\.(?:datasets\.\d+\.)?zarr_path", key) is not None
            if (
                key not in RESUME_OPERATIONAL_FIELDS
                and key not in ARTIFACT_LOCATION_FIELDS
                and not key.startswith("workspace.wandb_cfg.")
                and not data_path
            ):
                raise ValueError(
                    f"Resume recipe is owned by saved config; forbidden override: {key}"
                )
            if key == "resume_from":
                continue
            OmegaConf.update(result, key, override.value(), merge=False, force_add=True)
        result.resume_from = source
    return result


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


def validate_config_references(cfg):
    """Reject detached overrides of the public input references used by fresh runs."""
    if cfg.get("resume_from") is not None:
        return  # Saved recipes allow the existing explicit asset/data relocation overrides.
    references = []
    for name in ("tcp_dim", "joint_dim", "ee_dim", "codebook_path"):
        if name in cfg and name in cfg.agent:
            references.append((f"agent.{name}", name))
    if "datasets" not in cfg.dataset and "zarr_path" in cfg:
        references.append(("dataset.zarr_path", "zarr_path"))
    if "lamp" in cfg:
        references.extend((f"agent.{target}", f"lamp.{source}") for target, source in (
            ("prior_path", "prior_path"), ("history_steps", "history_steps"),
            ("latent_dim", "latent_dim"), ("prior_hidden_dim", "hidden_dim"),
        ))
        references.append(("dataset.history_steps", "lamp.history_steps"))
    if "codec" in cfg:
        references.extend((f"agent.{target}", f"codec.{source}") for target, source in (
            ("codec_path", "path"), ("latent_dim", "latent_dim"), ("codec_hidden_dim", "hidden_dim"),
            ("codec_n_layers", "n_layers"), ("codec_n_heads", "n_heads"), ("codec_mlp_ratio", "mlp_ratio"),
        ))
    if "data_semantics" in cfg:
        references.append(("dataset.data_semantics", "data_semantics"))
    for target, source in references:
        if OmegaConf.select(cfg, target) != OmegaConf.select(cfg, source):
            raise ValueError(f"{target} must reference {source}; override {source} instead")


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
    tasks = env_runner.get("task_configs")
    for index, task in enumerate([env_runner] if tasks is None else tasks):
        env_kwargs = task.get("env_kwargs") or {}
        actual_control = env_kwargs.get("control_mode", "joint")
        if actual_control != expected_control:
            name = task.get("task_name", f"index {index}")
            raise ValueError(
                f"Task {name!r}: action_key={action_key!r} requires "
                f"control_mode={expected_control!r}, got {actual_control!r}"
            )

    dataset = cfg.get("dataset")
    if dataset is None:
        return
    for child in [dataset, *dataset.get("datasets", ())]:
        ds_action_key = child.get("action_key", "action")
        if ds_action_key != action_key:
            raise ValueError(
                f"dataset.action_key={ds_action_key!r} differs from action_key={action_key!r}"
            )


def load_experiment_config(experiment_dir):
    """Read the resolved snapshot without composing current repository YAML."""
    return OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )


def validate_normalization_spec(spec, *, observation_fields=None):
    """Validate declared modes and optional exact numeric-field coverage."""
    if not isinstance(spec, dict):
        raise ValueError("Normalization must map field names to modes")
    for key, mode in spec.items():
        if not isinstance(key, str) or not key or key in NON_NUMERIC_OBSERVATION_FIELDS:
            raise ValueError(f"Invalid numeric normalization field: {key!r}")
        if not isinstance(mode, str) or mode not in ALLOWED_NORMALIZATION_MODES:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        if mode == "auto" and key != "action":
            raise ValueError("Normalization auto is only supported for action")
    if not {"joint_state", "action"} <= spec.keys():
        raise ValueError("Normalization must declare joint_state and action")
    if spec.get("rgb", "identity") != "identity":
        raise ValueError("RGB preprocessing requires normalization.rgb=identity")
    if observation_fields is not None:
        expected = set(observation_fields) | {"action"}
        if set(spec) != expected:
            raise ValueError(
                f"Normalization field mismatch: missing={sorted(expected - spec.keys())}, "
                f"extra={sorted(spec.keys() - expected)}"
            )
    return dict(spec)


def resolve_normalization_spec(cfg, *, observation_fields=None):
    spec = cfg.get("normalization")
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    return validate_normalization_spec(spec, observation_fields=observation_fields)


def normalization_dims_from_config(cfg):
    """Check recorded affine widths against the policy's observation and action layout."""
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
    reference = dataset["datasets"][0] if "datasets" in dataset else dataset
    if reference.get("pc_dim") is not None:
        expected["point_cloud"] = reference["pc_dim"]
    for key, mode in cfg["normalization"].items():
        if mode != "identity" and key in expected and dims[key] != expected[key]:
            raise ValueError(f"Saved {key} feature width must be {expected[key]}")
    return dims


def validate_dq_normalization(spec):
    """DQ/VQ hand prototypes require the existing bounded action coordinates."""
    if spec.get("action") not in {"auto", "limits"}:
        raise ValueError("DQ/VQ hand prototypes require action normalization auto or limits")


def validate_action_clipping(spec, clip_sample):
    if type(clip_sample) is not bool:
        raise TypeError("agent.clip_sample must be a bool")
    if spec.get("action") == "gaussian" and clip_sample:
        raise ValueError("Gaussian action normalization requires agent.clip_sample=false.")


def validate_sparse3d_config(cfg, spec, *, default_num_patches=96):
    """Preserve metric XYZ and enough points for Sparse3D patch geometry."""
    if spec.get("point_cloud") != "identity":
        raise ValueError(
            "Sparse3DEncoder requires normalization.point_cloud: identity (metric XYZ)"
        )
    encoder = cfg.agent.get("pc_encoder_config") or {}
    sizes = {
        name: positive_int(encoder.get(name, default), name)
        for name, default in (("num_patches", default_num_patches), ("group_size", 32))
    }
    point_limit = positive_int(cfg.agent.get("num_points", 1024), "agent.num_points")
    if point_limit < max(sizes.values()):
        raise ValueError("agent.num_points must be at least max(num_patches, group_size)")
    shared_augmentation = cfg.dataset.get("augmentation_cfg") if "datasets" in cfg.dataset else None
    for child in cfg.dataset.get("datasets", [cfg.dataset]):
        augmentation = (
            shared_augmentation
            if shared_augmentation is not None
            else child.get("augmentation_cfg")
        )
        augmentation = augmentation or {}
        keep_points = (augmentation.get("pc") or {}).get("random_keep_points")
        if keep_points is not None:
            positive_int(keep_points, "random_keep_points")
            for name, size in sizes.items():
                if keep_points < size:
                    raise ValueError(f"random_keep_points={keep_points} must be >= {name}={size}")


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


def resolve_dataset_paths(config_name: str, overrides: list[str]) -> list[Path]:
    """Resolve fresh or saved dataset paths without opening data or importing models."""
    cfg = load_policy_config(config_name, overrides)
    cfg = resolve_input_recipe(cfg, overrides=overrides)
    datasets = cfg.dataset.get("datasets", [cfg.dataset])
    return list(
        dict.fromkeys(
            (PROJECT_ROOT / Path(child.zarr_path).expanduser()).resolve() for child in datasets
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Resolve training dataset paths.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--check", action="store_true", help="Require existing dataset directories."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides passed to training.")
    args = parser.parse_args()
    paths = resolve_dataset_paths(args.config_name, args.overrides)
    print(json.dumps({"dataset_paths": [str(path) for path in paths]}, indent=2))
    if args.check:
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise SystemExit("Missing dataset directories:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
