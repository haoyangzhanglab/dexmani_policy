"""Configuration and model checks at training boundaries."""

import hydra

from dexmani_policy.common.config import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    positive_int,
    resolve_normalization_spec,
    validate_action_key_consistency,
)


def validate_config(cfg):
    """Check shared training constraints before opening data or model artifacts."""
    agent_class = hydra.utils.get_class(cfg.agent._target_)
    for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if type(cfg.agent[name]) is not type(cfg[name]) or cfg.agent[name] != cfg[name]:
            raise ValueError(f"agent.{name} must match {name}")
    aux = cfg.get("use_aux_ee", False)
    if cfg.agent.get("use_aux_ee", False) != aux:
        raise ValueError("agent.use_aux_ee must match use_aux_ee")
    for index, child in enumerate(cfg.dataset.get("datasets", [cfg.dataset])):
        if child.action_key not in agent_class.supported_action_keys:
            raise ValueError(f"{agent_class.__name__} does not support action_key={child.action_key}")
        if "history_steps" in cfg.agent and child.get("history_steps") != cfg.agent.history_steps:
            raise ValueError("Dataset/policy history_steps must match")
        if child.get("pc_dim") is not None and child.pc_dim != cfg.agent.get("pc_dim"):
            raise ValueError("Dataset pc_dim must match the actual Agent point-cloud channels")
        dataset_class = hydra.utils.get_class(child._target_)
        if agent_class.future_action_window != dataset_class.future_action_window:
            raise ValueError("Policy and Dataset action windows must be used together")
        if child.get("sensor_modalities") is None:
            raise ValueError(f"dataset[{index}].sensor_modalities must be explicit")
        for field, name in (("horizon", "horizon"), ("obs_horizon", "n_obs_steps")):
            if field not in child:
                raise ValueError(f"dataset[{index}].{field} must explicitly reference {name}")
            if type(child[field]) is not type(cfg[name]) or child[field] != cfg[name]:
                raise ValueError(f"dataset[{index}].{field} must match {name}")
        if child.get("use_aux_ee", False) != aux:
            raise ValueError(f"dataset[{index}].use_aux_ee must match use_aux_ee")
    reference = cfg.dataset.get("datasets", [cfg.dataset])[0]
    fields = set(reference.sensor_modalities) - NON_NUMERIC_OBSERVATION_FIELDS
    resolve_normalization_spec(cfg, observation_fields=fields)
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
