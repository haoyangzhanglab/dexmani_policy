import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    TACTILE_FEATURE_SHAPES,
    positive_int,
    require_tactile_source,
)

CONFIG_ROOT = PROJECT_ROOT / "dexmani_policy" / "configs"
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: LAUNCH_ID, replace=True)


def config_name(config):
    path = Path(config)
    if path.is_absolute() or ".." in path.parts or not str(config) or str(config).startswith("~"):
        raise ValueError("Config must be a name or relative path under dexmani_policy/configs")
    return path.with_suffix("").as_posix() if path.suffix == ".yaml" else path.as_posix()


def load_policy_config(config, overrides=()):
    name = config_name(config)
    register_resolvers()
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = hydra.compose(config_name=name, overrides=list(overrides))
    return assemble_policy_config(cfg)


def assemble_policy_config(cfg):
    """Derive fixed recipe fields; YAML only exposes research choices."""
    if "policy_name" not in cfg:
        return cfg
    with open_dict(cfg):
        action_key = cfg.action_key
        if action_key not in ("action", "action_ee"):
            raise ValueError("action_key must be action or action_ee")
        method = cfg.policy_name
        auxiliary = method == "r3d" and cfg.use_aux_ee
        if auxiliary and action_key != "action":
            raise ValueError("R3D auxiliary targets require joint action")

        fixed = {"action": "limits"}
        if "rgb" in cfg.dataset.sensor_modalities:
            fixed["rgb"] = "identity"
        if method == "r3d":
            fixed["point_cloud"] = "limits"
        if method in {"sat", "dqrise", "adaptac", "interact-dex"}:
            fixed["point_cloud"] = "identity"
        if method in {"sat", "dqrise"}:
            fixed.update(joint_state="limits", camera_intrinsic="identity", camera_extrinsic="identity")
        if method == "interact-dex":
            fixed.update(eef_pose="identity", fingertip_xyz="identity")
        choices = cfg.get("normalization", {})
        overlap = fixed.keys() & choices.keys()
        if overlap:
            raise ValueError(f"Normalization fields are fixed by the recipe; remove inputs: {sorted(overlap)}")
        cfg.normalization = {**choices, **fixed}

        cfg.action_dim = 28 if auxiliary else (21 if action_key == "action_ee" else 19)
        for key in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
            cfg.agent[key] = cfg[key]
        cfg.dataset.update(action_key=action_key, zarr_path=cfg.zarr_path,
                           horizon=cfg.horizon, obs_horizon=cfg.n_obs_steps,
                           pad_before=cfg.n_obs_steps - 1, pad_after=cfg.n_action_steps - 1)
        if method == "r3d":
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


def prepare_training(cfg):
    if cfg.get("resume_from") is not None:
        raise ValueError("Use python -m dexmani_policy.training.common.resume --run <run>")
    validate_config(cfg)
    from dexmani_policy.agents.obs_encoder.rgb.build import prepare_initialization
    prepare_initialization(cfg)
    return cfg


def validate_config(cfg):
    spec = cfg.normalization
    expected = set(cfg.dataset.sensor_modalities) | {"action"}
    if spec.keys() != expected:
        raise ValueError(
            f"Normalization field mismatch: missing={sorted(expected - spec.keys())}, "
            f"extra={sorted(spec.keys() - expected)}"
        )
    for key, mode in spec.items():
        allowed = {"rms", "max_abs"} if key in TACTILE_FEATURE_SHAPES else {"identity", "limits", "gaussian", "rms", "max_abs"}
        if mode not in allowed:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}; expected {sorted(allowed)}")
    require_tactile_source(cfg.dataset.sensor_modalities, cfg.get("tactile_source"))

    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    cls = hydra.utils.get_class(cfg.agent._target_)
    if cfg.action_key not in cls.supported_action_keys:
        raise ValueError(f"{cls.__name__} does not support {cfg.action_key}")

    print("Config validation passed")


def assemble_effective_config(cfg, data_description, model, *, checkpoint=None):
    metadata = model.inference_metadata
    if checkpoint is not None:
        if checkpoint.effective_config["normalization_dims"] != data_description["normalization_dims"]:
            raise ValueError("Resume data/model mismatch: normalization_dims")
        for key in ("action_start_index", "control_action_dim"):
            if checkpoint.effective_config["policy_metadata"][key] != metadata[key]:
                raise ValueError(f"Resume data/model mismatch: policy_metadata.{key}")

    effective = copy.deepcopy(cfg)
    with open_dict(effective):
        effective.normalization_dims = data_description["normalization_dims"]
        effective.normalizer_shapes = {key: list(field.scale.shape) for key, field in model.normalizer.fields.items()}
        effective.dataset.split_manifest = data_description["split_manifest"]
        effective.policy_metadata = metadata
        effective.deployment_metadata = data_description["deployment_metadata"]
        if checkpoint is None:
            effective.agent.update(model.effective_config)
    OmegaConf.resolve(effective)
    OmegaConf.set_readonly(effective, True)
    return effective


def resolve_dataset_paths(config_name: str, overrides: list[str]) -> list[Path]:
    """Resolve the fresh recipe's dataset path without opening data or importing models."""
    cfg = load_policy_config(config_name, overrides)
    return [(PROJECT_ROOT / Path(cfg.dataset.zarr_path).expanduser()).resolve()]


def main():
    parser = argparse.ArgumentParser(description="Resolve training dataset paths.")
    parser.add_argument("--config-name", required=True, help="Config name or relative path under dexmani_policy/configs")
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
