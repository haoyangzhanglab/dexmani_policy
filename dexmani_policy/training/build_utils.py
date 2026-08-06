"""Shared build functions for training/eval entry points."""

import hydra

from dexmani_policy.common.config import normalize_action_key, validate_action_key_consistency
from dexmani_policy.common.pytorch_util import print_param_count
from dexmani_policy.training.lr_scheduler import compute_num_training_steps, get_scheduler

__all__ = [
    "build_dataset_and_normalizer",
    "build_model_and_ema",
    "build_scheduler",
    "build_optimizer_and_scheduler",
    "validate_config",
    "inject_faas_into_agent",
    "compute_num_training_steps",
]

# joint_state arm is ALWAYS 7D arm joint angles, independent of action_key.
# This is separate from tcp_dim (7 for joint mode, 9 for action_ee mode).
STATE_ARM_DIM = 7

# ---------------------------------------------------------------------------
# Dataset & Normalizer
# ---------------------------------------------------------------------------


def build_dataset_and_normalizer(cfg):
    """Instantiate dataset and extract its normalizer.

    The caller is responsible for resolving OmegaConf interpolations before
    calling this function (DDP paths call ``OmegaConf.resolve(cfg)`` in the
    parent process before ``mp.spawn``).
    """
    dataset = hydra.utils.instantiate(cfg.dataset)

    # FAAS: inject attributes BEFORE get_normalizer() so the normalizer is
    # fitted on FAAS-converted replay buffer data.
    if cfg.get("use_faas", False):
        from dexmani_policy.common.faas_mapper import FAASHandMapper

        dataset.use_faas = True
        dataset.faas_mapper = FAASHandMapper()
        dataset.tcp_dim = cfg.tcp_dim

    normalizer = dataset.get_normalizer()
    if hasattr(dataset, "normalizer_mode") and dataset.normalizer_mode == "per_task":
        raise NotImplementedError(
            "normalizer_mode='per_task' requires per-task normalizer loading, "
            "which is not yet integrated into the standard training entry. "
            "Use normalizer_mode='shared' or call get_normalizer(task_name=...) manually."
        )
    return dataset, normalizer


# ---------------------------------------------------------------------------
# FAAS Injection (shared across train / eval / smoke-test entry points)
# ---------------------------------------------------------------------------


def inject_faas_into_agent(agent, cfg):
    """Post-construction FAAS attribute injection for any entry point.

    Called by ``build_model_and_ema`` (training), ``eval_sim.py`` (eval),
    and ``smoke_test.py`` (smoke).  Must be called before the agent is used
    for inference so that ``predict_action`` / ``compute_action_mse`` can
    detect ``use_faas`` and apply the correct conversions.
    """
    agent.use_faas = cfg.get("use_faas", False)
    if not agent.use_faas:
        return
    from dexmani_policy.common.faas_mapper import FAASHandMapper

    agent.tcp_dim = cfg.tcp_dim
    agent.hand_dim = cfg.get("hand_dim", cfg.get("faas_hand_dim", 32))
    agent.faas_mapper = FAASHandMapper()


# ---------------------------------------------------------------------------
# Model & EMA
# ---------------------------------------------------------------------------


def build_model_and_ema(cfg, device, normalizer):
    """Instantiate the agent model and, if configured, its EMA twin."""
    model = hydra.utils.instantiate(cfg.agent)
    inject_faas_into_agent(model, cfg)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.get("action_key", "action")
    model.to(device)
    print_param_count(model)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        ema_model = hydra.utils.instantiate(cfg.agent)
        inject_faas_into_agent(ema_model, cfg)
        ema_model.load_normalizer_from_dataset(normalizer)
        ema_model.action_key = model.action_key
        ema_model.to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    return model, ema_model, ema_updater


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------


def build_scheduler(cfg, optimizer, batches_per_epoch, last_epoch=-1):
    """Build the LR scheduler with the correct total step count."""
    total_steps = compute_num_training_steps(cfg, batches_per_epoch)
    return get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch,
    )


def build_optimizer_and_scheduler(cfg, model, batches_per_epoch, last_epoch=-1):
    """Build optimizer (via the agent's ``configure_optimizer``) and LR scheduler."""
    # Guard: tail batches are silently dropped when gradient_accumulation_steps
    # does not divide the dataloader.  Warn if this is ever configured.
    grad_accum = cfg.get("training", {}).get("loop", {}).get("gradient_accumulation_steps", 1)
    if grad_accum > 1 and batches_per_epoch % grad_accum != 0:
        print(
            f"[WARNING] gradient_accumulation_steps ({grad_accum}) does not divide "
            f"batches_per_epoch ({batches_per_epoch}). {batches_per_epoch % grad_accum} "
            f"tail batch(es) will be silently discarded at epoch end."
        )
    optimizer = model.configure_optimizer(**cfg.optimizer)
    scheduler = build_scheduler(cfg, optimizer, batches_per_epoch, last_epoch)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


def _validate_moe_config(cfg):
    """Check MoE expert count vs top_k consistency."""
    agent_cfg = cfg.agent
    if "num_experts" not in agent_cfg:
        return
    num_experts = agent_cfg.get("num_experts", 0)
    top_k = agent_cfg.get("top_k", 0)
    assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"


def _validate_augmentation_consistency(cfg):
    """Warn/error when PC color augmentation is configured but pc_dim < 6."""
    agent_cfg = cfg.agent
    pc_dim = agent_cfg.get("pc_dim")
    if pc_dim is None or pc_dim >= 6:
        return

    aug_cfg = cfg.dataset.get("augmentation_cfg")
    if aug_cfg is None:
        return

    pc_color = aug_cfg.get("pc", {}).get("color")
    pc_color_noise = aug_cfg.get("pc", {}).get("color_noise")
    missing_rgb = (
        f"PC color augmentation requires agent.pc_dim >= 6, got {pc_dim}. "
        f"The encoder only reads the first {pc_dim} channels (XYZ), "
        f"while the augmentation modifies channels 3:6 (RGB). "
        f"Either set agent.pc_dim=6 or remove the augmentation key."
    )
    if pc_color is not None:
        assert pc_dim >= 6, missing_rgb
    if pc_color_noise is not None:
        assert pc_dim >= 6, f"PC color_noise augmentation: {missing_rgb}"


def _validate_aux_config(cfg):
    """Validate use_aux_ee consistency.

    When enabled, action_dim = joint_dim + ee_dim = 19 + 9 = 28
    (wrist pose: pos3 + rot6d6 from action_ee[:9]).
    """
    use_aux_ee = cfg.get("use_aux_ee", False)

    if use_aux_ee:
        if cfg.get("action_key", "action") != "action":
            raise ValueError(
                f"use_aux_ee=true requires action_key='action' (joint primary), "
                f"got action_key='{cfg.action_key}'. "
                f"The EE wrist action is auxiliary — change action_key to 'action'."
            )
        if cfg.get("joint_dim") is None or cfg.get("ee_dim") is None:
            raise ValueError("use_aux_ee=true requires joint_dim and ee_dim in config.")


def _validate_faas_config(cfg):
    """Validate FAAS-specific config constraints.

    Preconditions checked:
    - ``use_aux_ee`` is mutually exclusive with ``use_faas``
    - ``tcp_dim`` exists and is 7 or 9
    - ``action_dim == tcp_dim + hand_dim``
    - ``state_dim == STATE_ARM_DIM + hand_dim`` (joint_state arm always 7D)
    - Normalizer mode is ``limits`` (gaussian is numerically unstable on
      zero-padded FAAS dimensions)
    - MultiTaskDataset + use_faas is blocked (not yet supported)
    """
    if not cfg.get("use_faas", False):
        return

    # Mutual exclusion
    if cfg.get("use_aux_ee", False):
        raise ValueError(
            "use_faas=true is incompatible with use_aux_ee=true. "
            "FAAS uses its own hand space and does not support EE auxiliary loss."
        )

    # tcp_dim
    tcp_dim = cfg.get("tcp_dim")
    if tcp_dim is None:
        raise ValueError("use_faas=true requires tcp_dim in config.")
    if tcp_dim not in (7, 9):
        raise ValueError(f"tcp_dim must be 7 (joint) or 9 (action_ee), got {tcp_dim}.")

    # Dimension consistency
    # NOTE: cfg.hand_dim is the *native* hand dim (12), used for smoke test
    # DQ-RISE compat.  cfg.faas_hand_dim is the FAAS hand dim (32).
    # For validation we use faas_hand_dim as the canonical field.
    faas_hand_dim = cfg.get("faas_hand_dim", 32)
    expected_action_dim = tcp_dim + faas_hand_dim
    actual_action_dim = cfg.get("action_dim")
    if actual_action_dim != expected_action_dim:
        raise ValueError(
            f"action_dim={actual_action_dim}, expected {tcp_dim}+{faas_hand_dim}={expected_action_dim}. "
            f"Check your dimension overrides."
        )

    # state_dim: joint_state arm is always 7D, not tcp_dim
    expected_state_dim = STATE_ARM_DIM + faas_hand_dim
    # state_dim may be inside agent: due to Hydra deep merge
    agent_cfg = cfg.get("agent", cfg)
    actual_state_dim = agent_cfg.get("state_dim", cfg.get("state_dim"))
    if actual_state_dim != expected_state_dim:
        raise ValueError(
            f"state_dim={actual_state_dim}, expected {STATE_ARM_DIM}+{faas_hand_dim}={expected_state_dim}. "
            f"Note: joint_state arm is always 7D (STATE_ARM_DIM), not tcp_dim ({tcp_dim})."
        )

    # Normalizer mode must be limits
    # (gaussian mode produces unstable scale values on zero-padded FAAS dims)
    normalizer_mode = cfg.get("normalizer_mode", "limits")
    if normalizer_mode != "limits":
        raise ValueError(
            f"use_faas=true requires normalizer_mode='limits', got '{normalizer_mode}'. "
            f"Gaussian normalizer is numerically unstable on zero-padded FAAS dimensions."
        )

    # MultiTaskDataset guard (not yet supported)
    dataset_target = cfg.get("dataset", {}).get("_target_", "")
    if "MultiTaskDataset" in dataset_target:
        raise NotImplementedError(
            "use_faas=true with MultiTaskDataset is not yet supported. Use single-task datasets only."
        )

    print("FAAS config validation passed")


def validate_config(cfg):
    """Validate common training config constraints.

    Called by all entry points before training or evaluation.
    """
    normalize_action_key(cfg)

    if cfg.n_obs_steps > cfg.horizon:
        raise ValueError(f"n_obs_steps ({cfg.n_obs_steps}) cannot exceed horizon ({cfg.horizon})")
    if cfg.n_action_steps > cfg.horizon:
        raise ValueError(f"n_action_steps ({cfg.n_action_steps}) cannot exceed horizon ({cfg.horizon})")
    if cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps-1 + n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) "
            f"exceeds horizon ({cfg.horizon})"
        )

    if cfg.optimizer.get("obs_lr") is not None:
        assert cfg.optimizer.obs_lr >= 0, "optimizer.obs_lr must be non-negative (0 means freeze)"

    _validate_moe_config(cfg)
    _validate_augmentation_consistency(cfg)
    _validate_aux_config(cfg)
    _validate_faas_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")
