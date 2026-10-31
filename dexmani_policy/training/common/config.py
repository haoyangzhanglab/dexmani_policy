import argparse
import copy
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_policy_config,
    positive_int,
    require_tactile_source,
    validate_action_mode,
    validate_normalization_config,
)


def prepare_training(cfg):
    if cfg.get("resume_from") is not None:
        raise ValueError("Use python -m dexmani_policy.training.common.resume --run <run>")
    validate_config(cfg)
    from dexmani_policy.agents.obs_encoder.rgb.build import prepare_initialization
    prepare_initialization(cfg)
    return cfg


def validate_config(cfg):
    validate_normalization_config(cfg)
    require_tactile_source(cfg.dataset.sensor_modalities, cfg.get("tactile_source"))

    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    validate_action_mode(cfg)

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
