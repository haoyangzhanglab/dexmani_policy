import argparse
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_policy_config,
    normalization_dims_from_config,
    positive_int,
    resolve_normalization_spec,
    validate_action_key_consistency,
)
from dexmani_policy.training.common.resume import resolve_input_recipe, resolve_resume_source

if TYPE_CHECKING:
    from dexmani_policy.agents.policies.base import BasePolicy
    from dexmani_policy.datasets.common.robot_dataset import RobotDataset


def prepare_training(cfg):
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = load_checkpoint(source) if source is not None else None
    overrides = HydraConfig.get().overrides.task if source is not None else []
    effective_config = resolve_input_recipe(
        cfg, overrides=overrides, saved_config=None if checkpoint is None else checkpoint.effective_config,
    )
    with open_dict(effective_config):
        effective_config.resume_from = source
        # Resume writes into this invocation's directory, never the source run.
        effective_config.workspace.output_dir = cfg.workspace.output_dir
    validate_config(effective_config)
    return effective_config, checkpoint


def validate_config(cfg):
    agent_class: type[BasePolicy] = hydra.utils.get_class(cfg.agent._target_)
    for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if cfg.agent[name] != cfg[name]:
            raise ValueError(f"agent.{name} must match {name}")
    aux = cfg.get("use_aux_ee", False)
    if cfg.agent.get("use_aux_ee", False) != aux:
        raise ValueError("agent.use_aux_ee must match use_aux_ee")
    dataset = cfg.dataset
    if dataset.action_key not in agent_class.supported_action_keys:
        raise ValueError(f"{agent_class.__name__} does not support action_key={dataset.action_key}")
    if "history_steps" in cfg.agent and dataset.get("history_steps") != cfg.agent.history_steps:
        raise ValueError("Dataset/policy history_steps must match")
    if dataset.get("pc_dim") is not None and dataset.pc_dim != cfg.agent.get("pc_dim"):
        raise ValueError("Dataset pc_dim must match the actual Agent point-cloud channels")
    dataset_class: type[RobotDataset] = hydra.utils.get_class(dataset._target_)
    from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
    from dexmani_policy.datasets.r3d_dataset import R3DAuxDataset
    if agent_class.future_action_window != issubclass(dataset_class, LatentFlowDataset):
        raise ValueError("Policy and Dataset action windows must be used together")
    if dataset.get("sensor_modalities") is None:
        raise ValueError("dataset.sensor_modalities must be explicit")
    for field, name in (("horizon", "horizon"), ("obs_horizon", "n_obs_steps")):
        if field not in dataset:
            raise ValueError(f"dataset.{field} must explicitly reference {name}")
        if dataset[field] != cfg[name]:
            raise ValueError(f"dataset.{field} must match {name}")
    if issubclass(dataset_class, R3DAuxDataset) != aux:
        raise ValueError("R3D auxiliary targets require R3DAuxDataset")
    resolve_normalization_spec(cfg, observation_fields=dataset.sensor_modalities)
    if aux:
        if cfg.action_key != "action":
            raise ValueError("use_aux_ee=true requires action_key='action' (joint primary)")
        if cfg.get("joint_dim") is None or cfg.get("ee_dim") is None:
            raise ValueError("use_aux_ee=true requires joint_dim and ee_dim")
        if cfg.ee_dim != 9 or cfg.action_dim != cfg.joint_dim + cfg.ee_dim:
            raise ValueError("Auxiliary actions require joint_dim + 9 full prediction dimensions")
        for name in ("joint_dim", "ee_dim"):
            if cfg.agent.get(name) != cfg[name]:
                raise ValueError(f"agent.{name} must match {name}")

    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    validate_action_key_consistency(cfg)

    print("Config validation passed")


def assemble_effective_config(cfg, data_description, model, *, checkpoint=None):
    effective = copy.deepcopy(cfg)
    with open_dict(effective):
        effective.update({key: value for key, value in data_description.items() if key != "split_manifest"})
        effective.dataset.split_manifest = data_description["split_manifest"]
        if "real_runtime" not in data_description:
            effective.pop("real_runtime", None)
        effective.policy_metadata = model.inference_metadata
        if checkpoint is None:
            effective.agent.update(model.effective_config)
        if effective.training.lr_scheduler == "cosine_min_lr":
            effective.training.lr_min_ratio = effective.training.get("lr_min_ratio", 0.1)
    normalization_dims_from_config(effective)
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
