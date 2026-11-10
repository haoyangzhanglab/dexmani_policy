from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.config import PROJECT_ROOT, positive_int

if TYPE_CHECKING:
    from dexmani_policy.agents.policies.base import Policy

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
    """Derive action layouts and sampling windows from the recipe."""
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
        if method == "interact-dex":
            cfg.agent.future_force_horizon = cfg.future_force_horizon
            cfg.dataset.include_future_force = cfg.agent.future_force_loss_weight > 0
        cfg.env_runner.task_name = cfg.task_name
        cfg.env_runner.env_kwargs = {**(cfg.env_runner.env_kwargs or {}), "control_mode": "ee" if action_key == "action_ee" else "joint"}
    return cfg


def prepare_training(cfg):
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        path = Path(cfg.workspace.output_dir) / name
        if path.exists():
            raise FileExistsError(f"Training output already exists: {path}")
    validate_config(cfg)
    if cfg.policy_name in {"dp", "sat", "dqrise", "dexlatent"}:
        name = cfg.agent.rgb_backbone_name
        if name in ("dino", "siglip"):
            rgb = dict(cfg.agent.get("rgb_backbone_config") or {})
            if rgb.get("architecture") is None:
                source = cfg.initialization.rgb
                if source is None:
                    raise ValueError("Fresh HF RGB requires initialization.rgb or an explicit architecture")
                from transformers import AutoConfig, SiglipVisionConfig

                config_class = SiglipVisionConfig if name == "siglip" else AutoConfig
                rgb["architecture"] = config_class.from_pretrained(source).to_dict()
            with open_dict(cfg.agent):
                cfg.agent.rgb_backbone_config = rgb
    return cfg


def validate_normalization(cfg):
    spec = cfg.normalization

    physical = set()
    if cfg.policy_name in {"sat", "dqrise", "dexlatent"}:
        physical = {"point_cloud", "camera_intrinsic", "camera_extrinsic"}
    elif cfg.policy_name == "interact-dex":
        physical = {"point_cloud", "eef_pose", "fingertip_xyz"}
    elif cfg.policy_name == "adaptac":
        physical = {"point_cloud"}
    if cfg.policy_name in {"dp", "sat", "dqrise", "dexlatent"}:
        physical.add("rgb")
    for key in physical:
        if key in spec and spec[key] != "identity":
            raise ValueError(f"{key} must use identity for image preprocessing or physical geometry")


def validate_config(cfg):
    if not 0 <= cfg.training.max_grad_norm < float("inf"):
        raise ValueError("training.max_grad_norm must be finite and nonnegative")
    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    cls = cast("type[Policy]", hydra.utils.get_class(cfg.agent._target_))
    if cfg.action_key not in cls.supported_action_keys:
        raise ValueError(f"{cls.__name__} does not support {cfg.action_key}")


def assemble_effective_config(cfg, model):
    with open_dict(cfg):
        cfg.pop("max_updates", None)
        cfg.normalization_dims = model.normalization_dims
        cfg.agent.update(model.effective_config)
    OmegaConf.resolve(cfg)
    return cfg
