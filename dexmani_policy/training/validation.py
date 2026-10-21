"""Configuration and model checks at training boundaries."""

import inspect

import hydra
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.datasets.base_dataset import BaseDataset
from dexmani_policy.datasets.common.augmentation import (
    validate_pc_color_augmentation,
    validate_random_keep_points,
)
from dexmani_policy.datasets.common.episode_split import validate_split_options
from dexmani_policy.datasets.common.rgb_preprocessing import validate_rgb_preprocessing
from dexmani_policy.utils.config import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    positive_int,
    resolve_normalization_spec,
    validate_action_clipping,
    validate_action_key_consistency,
    validate_dq_normalization,
    validate_action_window,
)

HAND_POLICY_TARGETS = frozenset(
    {
        "dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
        "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy",
        "dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy",
    }
)


SPARSE3D_POLICY_TARGETS = frozenset(
    {
        "dexmani_policy.agents.core.sat_policy.SATPolicy",
        "dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
        "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy",
    }
)


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
        set(child.get("sensor_modalities", ("joint_state",))) - NON_NUMERIC_OBSERVATION_FIELDS
        for child in cfg.dataset.get("datasets", [cfg.dataset])
    ]
    if any(child != fields[0] for child in fields[1:]):
        raise ValueError(
            f"Shared normalization requires matching numeric fields: {[sorted(s) for s in fields]}"
        )
    return fields[0]


def validate_encoder_normalization(cfg, spec) -> None:
    """Keep Sparse3D position content and geometric features in metric XYZ."""
    agent = cfg.get("agent", {})
    if agent.get("_target_") in SPARSE3D_POLICY_TARGETS and spec.get("point_cloud") != "identity":
        raise ValueError(
            "Sparse3DEncoder requires normalization.point_cloud: identity "
            "(position content and geometric features require metric XYZ)."
        )


def validate_normalization_config(cfg):
    """Validate normalization fields and their encoder/decoder coordinate constraints."""
    from dexmani_policy.agents.core.base_policy import UNetDiffusionPolicy
    from dexmani_policy.agents.core.dqrise_policy import DQRISEPolicy
    from dexmani_policy.agents.core.multitask_policy import MultiTaskPolicy
    from dexmani_policy.agents.core.r3d_policy import R3DPolicy

    rgb_transport = {
        bool(child.get("rgb_keep_uint8", False))
        for child in cfg.dataset.get("datasets", [cfg.dataset])
        if "rgb" in child.get("sensor_modalities", [])
    }
    if len(rgb_transport) > 1:
        raise ValueError("RGB child datasets must have consistent rgb_keep_uint8")
    spec = resolve_normalization_spec(
        cfg, observation_fields=resolve_numeric_observation_fields(cfg)
    )
    agent_class = hydra.utils.get_class(cfg.agent._target_)
    if issubclass(agent_class, DQRISEPolicy):
        validate_dq_normalization(spec)
    uses_diffusion = issubclass(agent_class, (UNetDiffusionPolicy, DQRISEPolicy, R3DPolicy))
    if issubclass(agent_class, MultiTaskPolicy):
        uses_diffusion = cfg.agent.get("action_decoder_type", "diffusion") == "diffusion"
    if uses_diffusion:
        validate_action_clipping(spec, cfg.agent.get("clip_sample", True))
    validate_encoder_normalization(cfg, spec)


def validate_policy_layout(cfg):
    """Check shared windows and action layouts before opening data."""
    from dexmani_policy.agents.core.base_policy import UNetDiffusionPolicy
    from dexmani_policy.agents.core.dqrise_policy import DQRISEPolicy

    validate_action_window(cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps)
    positive_int(cfg.action_dim, "action_dim")
    for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if type(cfg.agent[name]) is not type(cfg[name]) or cfg.agent[name] != cfg[name]:
            raise ValueError(f"agent.{name} must match {name}")
    aux = cfg.get("use_aux_ee", False)
    if cfg.agent.get("use_aux_ee", False) != aux:
        raise ValueError("agent.use_aux_ee must match use_aux_ee")
    dataset_defaults = inspect.signature(BaseDataset).parameters
    for index, child in enumerate(cfg.dataset.get("datasets", [cfg.dataset])):
        layout = dict(child)
        if issubclass(hydra.utils.get_class(child._target_), BaseDataset):
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

    agent_class = hydra.utils.get_class(cfg.agent._target_)
    if issubclass(agent_class, (UNetDiffusionPolicy, DQRISEPolicy)):
        constructor = DQRISEPolicy if issubclass(agent_class, DQRISEPolicy) else UNetDiffusionPolicy
        defaults = inspect.signature(constructor).parameters["down_dims"].default
        down_dims = cfg.agent.get("down_dims", defaults)
        if not down_dims:
            raise ValueError("UNet down_dims must be nonempty")
        divisor = 2 ** (len(down_dims) - 1)
        if cfg.horizon % divisor:
            raise ValueError(f"UNet horizon must be divisible by {divisor}, got {cfg.horizon}")


def validate_hand_config(cfg):
    """Align representation preparation and policy training before opening data."""
    target = cfg.agent.get("_target_")
    if target not in HAND_POLICY_TARGETS:
        return
    if cfg.dataset.get("use_aux_ee", False):
        raise ValueError("Hand representations do not support auxiliary action targets")

    if target == "dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy":
        layout = {"action": (7, 19), "action_ee": (9, 21)}.get(cfg.action_key)
        if (cfg.agent.tcp_dim, cfg.agent.action_dim) != layout:
            raise ValueError("Supported DQ-RISE layouts are action=7+12 or action_ee=9+12")
    else:
        if cfg.action_key != "action" or cfg.action_dim != 19 or cfg.agent.state_dim != 19:
            raise ValueError("LAMP/Temporal require joint action/state=7+12")

    if target == "dexmani_policy.agents.core.lamp_policy.LAMPPolicy":
        if (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) != (1, 1, 1):
            raise ValueError("LAMP requires H=Nobs=C=1")
        if cfg.dataset.history_steps != cfg.agent.history_steps:
            raise ValueError("LAMP dataset/policy history_steps must match")
        if cfg.agent.get("num_inference_steps", 1) != 1 or cfg.eval.inference_steps != 1:
            raise ValueError("LAMP requires inference_steps=1 in agent/eval")
        if cfg.eval.get("inference_steps_list") not in (None, [1]):
            raise ValueError("LAMP inference_steps_list must be null or [1]")


def validate_config(cfg):
    """Validate shared training constraints without opening data or model artifacts."""
    agent_pc_dim = cfg.agent.get("pc_dim")
    geometry_sizes = {}
    if cfg.agent.get("_target_") in SPARSE3D_POLICY_TARGETS:
        pc_config = cfg.agent.get("pc_encoder_config") or {}
        geometry_sizes = dict(
            num_patches=pc_config.get("num_patches", 96), group_size=pc_config.get("group_size", 32)
        )
        point_limit = cfg.agent.get("num_points", 1024)
        positive_int(point_limit, "agent.num_points")
        for name, size in geometry_sizes.items():
            positive_int(size, name)
        if point_limit < max(geometry_sizes.values()):
            raise ValueError("agent.num_points must be at least max(num_patches, group_size)")
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
        validate_pc_color_augmentation(agent_pc_dim, child.get("augmentation_cfg"))
        augmentation = child.get("augmentation_cfg")
        if "datasets" in cfg.dataset and cfg.dataset.get("augmentation_cfg") is not None:
            # Hydra constructs each child before the shared augmentation is applied.
            validate_random_keep_points(
                ((augmentation or {}).get("pc") or {}).get("random_keep_points")
            )
            augmentation = cfg.dataset.augmentation_cfg
        keep_points = ((augmentation or {}).get("pc") or {}).get("random_keep_points")
        validate_random_keep_points(keep_points, **geometry_sizes)
    if "datasets" in cfg.dataset:
        # Hydra constructs children before MultiTaskDataset applies its shared augmentation.
        validate_pc_color_augmentation(agent_pc_dim, cfg.dataset.get("augmentation_cfg"))
    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        positive_int(cfg.training.loop[name], f"training.loop.{name}")
    if cfg.training.lr_warmup_steps < 0:
        raise ValueError("training.lr_warmup_steps must be nonnegative")
    validate_policy_layout(cfg)
    validate_hand_config(cfg)

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    validate_normalization_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")
