"""Configuration and model checks at training boundaries."""

import inspect

import hydra
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.common.config import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    positive_int,
    resolve_normalization_spec,
    validate_action_key_consistency,
    validate_action_window,
)
from dexmani_policy.common.rgb_preprocessing import validate_rgb_preprocessing
from dexmani_policy.datasets.common.augmentation import (
    validate_pc_color_augmentation,
    validate_random_keep_points,
)
from dexmani_policy.datasets.common.episode_split import validate_split_options
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset


def validate_ema_batchnorm(model) -> None:
    """Reject BatchNorm whose parameters or running statistics can change."""
    unsafe = []
    for name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        has_trainable_params = any(p.requires_grad for p in module.parameters(recurse=False))
        updates_running_stats = bool(module.training and module.track_running_stats)
        if has_trainable_params or updates_running_stats:
            unsafe.append(name or "<root>")
    if unsafe:
        raise ValueError(f"EMA requires frozen BatchNorm or GroupNorm; active modules: {unsafe}")


def resolve_numeric_observation_fields(cfg) -> set:
    """Require the same numeric observation fields across tasks."""
    fields = [
        set(child.sensor_modalities) - NON_NUMERIC_OBSERVATION_FIELDS
        for child in cfg.dataset.get("datasets", [cfg.dataset])
    ]
    if any(child != fields[0] for child in fields[1:]):
        raise ValueError(
            f"Shared normalization requires matching numeric fields: {[sorted(s) for s in fields]}"
        )
    return fields[0]


def validate_model_config(cfg):
    """Check normalization and the selected policy's encoder/decoder constraints."""
    rgb_transport = {
        bool(child.get("rgb_keep_uint8", False))
        for child in cfg.dataset.get("datasets", [cfg.dataset])
        if "rgb" in child.sensor_modalities
    }
    if len(rgb_transport) > 1:
        raise ValueError("RGB child datasets must have consistent rgb_keep_uint8")
    spec = resolve_normalization_spec(
        cfg, observation_fields=resolve_numeric_observation_fields(cfg)
    )
    agent_class = hydra.utils.get_class(cfg.agent._target_)
    if agent_class.requires_action_affine and spec["action"] == "identity":
        raise ValueError("This policy requires fitted action affine parameters")
    agent_class.validate_model_config(cfg, spec)


def validate_policy_layout(cfg):
    """Check shared windows and action layouts before opening data."""
    validate_action_window(cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps)
    positive_int(cfg.action_dim, "action_dim")
    for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if type(cfg.agent[name]) is not type(cfg[name]) or cfg.agent[name] != cfg[name]:
            raise ValueError(f"agent.{name} must match {name}")
    aux = cfg.get("use_aux_ee", False)
    if cfg.agent.get("use_aux_ee", False) != aux:
        raise ValueError("agent.use_aux_ee must match use_aux_ee")
    dataset_defaults = inspect.signature(DemonstrationDataset).parameters
    for index, child in enumerate(cfg.dataset.get("datasets", [cfg.dataset])):
        layout = dict(child)
        if issubclass(hydra.utils.get_class(child._target_), DemonstrationDataset):
            if child.get("sensor_modalities") is None:
                raise ValueError(f"dataset[{index}].sensor_modalities must be explicit")
            layout.setdefault("horizon", dataset_defaults["horizon"].default)
            if layout.get("obs_horizon") is None:
                layout["obs_horizon"] = layout["horizon"]
        for field, name in (("horizon", "horizon"), ("obs_horizon", "n_obs_steps")):
            if field in layout and (
                type(layout[field]) is not type(cfg[name]) or layout[field] != cfg[name]
            ):
                raise ValueError(f"dataset[{index}].{field} must match {name}")
        if child.get("use_aux_ee", False) != aux:
            raise ValueError(f"dataset[{index}].use_aux_ee must match use_aux_ee")
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


def validate_config(cfg):
    """Validate shared training constraints without opening data or model artifacts."""
    agent_pc_dim = cfg.agent.get("pc_dim")
    for child in cfg.dataset.get("datasets", [cfg.dataset]):
        validate_split_options(
            child.get("val_ratio", 0),
            child.get("max_train_episodes"),
            explicit=child.get("split_manifest") is not None
            or child.get("saved_split") is not None,
        )
        validate_rgb_preprocessing(
            child.get("rgb_preprocess_size"),
            child.get("rgb_random_crop_size"),
            child.get("rgb_color_aug"),
        )
        if child.get("pc_dim") is not None and child.pc_dim != agent_pc_dim:
            raise ValueError("Dataset pc_dim must match the actual Agent point-cloud channels")
        augmentation = child.get("augmentation_cfg")
        validate_pc_color_augmentation(agent_pc_dim, augmentation)
        keep_points = ((augmentation or {}).get("pc") or {}).get("random_keep_points")
        validate_random_keep_points(keep_points)
    if "datasets" in cfg.dataset:
        # Hydra constructs children before applying shared augmentation.
        augmentation = cfg.dataset.get("augmentation_cfg")
        validate_pc_color_augmentation(agent_pc_dim, augmentation)
        validate_random_keep_points(((augmentation or {}).get("pc") or {}).get("random_keep_points"))
    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")
    validate_policy_layout(cfg)

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    validate_model_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")
