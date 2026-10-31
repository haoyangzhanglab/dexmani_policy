import argparse
import copy
import json
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import (
    PROJECT_ROOT,
    assemble_policy_config,
    load_policy_config,
    positive_int,
    validate_action_mode,
    validate_normalization_config,
    validate_tactile_config,
)
from dexmani_policy.training.common.resume import resolve_input_recipe, resolve_resume_source


def prepare_training(cfg):
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = load_checkpoint(source, mmap=True) if source is not None else None
    overrides = HydraConfig.get().overrides.task if source is not None else []
    effective_config = resolve_input_recipe(
        cfg, overrides=overrides, saved_config=None if checkpoint is None else checkpoint.effective_config,
    )
    with open_dict(effective_config):
        effective_config.resume_from = source
        # Resume writes into this invocation's directory, never the source run.
        effective_config.workspace.output_dir = cfg.workspace.output_dir
    assemble_policy_config(effective_config)
    validate_config(effective_config)
    return effective_config, checkpoint


def validate_config(cfg):
    validate_normalization_config(cfg)
    validate_tactile_config(cfg)

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
    """Resolve fresh or saved dataset paths without opening data or importing models."""
    cfg = load_policy_config(config_name, overrides)
    source = resolve_resume_source(cfg.get("resume_from"))
    saved = load_checkpoint(source, mmap=True).effective_config if source is not None else None
    cfg = resolve_input_recipe(cfg, overrides=overrides, saved_config=saved)
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
