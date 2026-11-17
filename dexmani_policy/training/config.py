from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.config import PROJECT_ROOT, positive_int

CONFIG_ROOT = PROJECT_ROOT / "dexmani_policy" / "configs"
LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: LAUNCH_ID, replace=True)


def load_config(config, overrides=(), *, config_dir=None):
    path = Path(config)
    if path.is_absolute() or ".." in path.parts or not str(config) or str(config).startswith("~"):
        raise ValueError("Config must be a name or relative path under the config directory")
    name = path.with_suffix("").as_posix() if path.suffix == ".yaml" else path.as_posix()
    register_resolvers()
    directory = CONFIG_ROOT if config_dir is None else Path(config_dir).expanduser().resolve()
    search = [] if config_dir is None else [f'hydra.searchpath=["file://{CONFIG_ROOT}"]']
    with hydra.initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = hydra.compose(config_name=name, overrides=[*search, *overrides])
    return assemble_policy_config(cfg) if "policy_name" in cfg else cfg


def assemble_policy_config(cfg):
    """Assemble the shared dataset, window and environment protocol."""
    with open_dict(cfg):
        action_key = cfg.action_key
        if action_key not in ("action", "action_ee", "action_rotvec"):
            raise ValueError("action_key must be action, action_ee or action_rotvec")
        for key in ("horizon", "n_obs_steps", "n_action_steps"):
            cfg.agent[key] = cfg[key]
        cfg.dataset.update(
            action_key=action_key, zarr_path=cfg.zarr_path,
            horizon=cfg.horizon, obs_horizon=cfg.n_obs_steps,
            pad_before=cfg.n_obs_steps - 1, pad_after=cfg.n_action_steps - 1,
        )
        if "pc_dim" in cfg.agent:
            cfg.dataset.pc_dim = cfg.agent.pc_dim
        cfg.env_runner.task_name = cfg.task_name
        if action_key != "action_rotvec":
            cfg.env_runner.env_kwargs = {
                **(cfg.env_runner.env_kwargs or {}),
                "control_mode": "ee" if action_key == "action_ee" else "joint",
            }
    return cfg


def normalize_output_dir(cfg) -> Path:
    output_dir = Path(cfg.workspace.output_dir).expanduser().resolve(strict=False)
    cfg.workspace.output_dir = str(output_dir)
    return output_dir


def check_fresh_output(output_dir, *extra_files):
    output_dir = Path(output_dir)
    for name in ("config.yaml", "metrics.jsonl", "checkpoints", *extra_files):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"Training output already exists: {path}")


def validate_normalization(policy, spec, field_dims):
    for key in policy.required_identity_fields.intersection(field_dims):
        if key in spec and spec[key] != "identity":
            raise ValueError(f"{key} must use identity for image preprocessing or physical geometry")


def validate_config(cfg):
    if not 0 <= cfg.training.max_grad_norm < float("inf"):
        raise ValueError("training.max_grad_norm must be finite and nonnegative")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    cls = hydra.utils.get_class(cfg.agent._target_)
    if cfg.action_key not in cls.supported_action_keys:
        raise ValueError(f"{cls.__name__} does not support {cfg.action_key}")
