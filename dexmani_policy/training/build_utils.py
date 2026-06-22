"""Shared build functions for training/eval entry points."""

import hydra
from termcolor import cprint

from dexmani_policy.common.config import normalize_action_key, validate_action_key_consistency
from dexmani_policy.common.pytorch_util import print_param_count
from dexmani_policy.training.lr_scheduler import get_scheduler, compute_num_training_steps

__all__ = [
    "build_dataset_and_normalizer",
    "build_model_and_ema",
    "build_scheduler",
    "build_optimizer_and_scheduler",
    "validate_config",
    "compute_num_training_steps",
]

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

    normalizer = dataset.get_normalizer()
    if hasattr(dataset, 'normalizer_mode') and dataset.normalizer_mode == 'per_task':
        raise NotImplementedError(
            "normalizer_mode='per_task' requires per-task normalizer loading, "
            "which is not yet integrated into the standard training entry. "
            "Use normalizer_mode='shared' or call get_normalizer(task_name=...) manually."
        )
    return dataset, normalizer

# ---------------------------------------------------------------------------
# Model & EMA
# ---------------------------------------------------------------------------

def build_model_and_ema(cfg, device, normalizer):
    """Instantiate the agent model and, if configured, its EMA twin."""
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.get('action_key', 'action')
    model.to(device)
    print_param_count(model)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        ema_model = hydra.utils.instantiate(cfg.agent)
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
    optimizer = model.configure_optimizer(**cfg.optimizer)
    scheduler = build_scheduler(cfg, optimizer, batches_per_epoch, last_epoch)
    return optimizer, scheduler

# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------

def _validate_moe_config(cfg):
    """Check MoE expert count vs top_k consistency."""
    agent_cfg = cfg.agent
    if 'num_experts' not in agent_cfg:
        return
    num_experts = agent_cfg.get('num_experts', 0)
    top_k = agent_cfg.get('top_k', 0)
    assert top_k <= num_experts, \
        f"top_k ({top_k}) must be <= num_experts ({num_experts})"

def _validate_augmentation_consistency(cfg):
    """Warn/error when PC color augmentation is configured but pc_dim < 6."""
    agent_cfg = cfg.agent
    pc_dim = agent_cfg.get('pc_dim')
    if pc_dim is None or pc_dim >= 6:
        return

    aug_cfg = cfg.dataset.get('augmentation_cfg')
    if aug_cfg is None:
        return

    pc_color = aug_cfg.get('pc', {}).get('color')
    pc_color_noise = aug_cfg.get('pc', {}).get('color_noise')
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

def validate_config(cfg):
    """Validate common training config constraints.

    Called by all entry points before training or evaluation.
    """
    normalize_action_key(cfg)

    if cfg.n_obs_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps ({cfg.n_obs_steps}) cannot exceed horizon ({cfg.horizon})"
        )
    if cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_action_steps ({cfg.n_action_steps}) cannot exceed horizon ({cfg.horizon})"
        )
    if cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps-1 + n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) "
            f"exceeds horizon ({cfg.horizon})"
        )

    if cfg.optimizer.get('obs_lr') is not None:
        assert cfg.optimizer.obs_lr >= 0, "optimizer.obs_lr must be non-negative (0 means freeze)"

    _validate_moe_config(cfg)
    _validate_augmentation_consistency(cfg)
    validate_action_key_consistency(cfg)

    cprint("Config validation passed", "green")
