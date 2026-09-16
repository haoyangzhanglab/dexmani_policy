"""Shared build functions for training/eval entry points."""

import hydra
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.common.config import validate_action_key_consistency
from dexmani_policy.common.pytorch_util import print_param_count
from dexmani_policy.training.lr_scheduler import (
    compute_num_training_steps,
    get_scheduler,
)

__all__ = [
    "build_dataset_and_normalizer",
    "build_model_and_ema",
    "build_scheduler",
    "build_optimizer_and_scheduler",
    "validate_config",
    "compute_num_training_steps",
    "validate_gradient_accumulation",
    "print_training_recipe",
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
    if hasattr(dataset, "normalizer_mode") and dataset.normalizer_mode == "per_task":
        raise NotImplementedError(
            "normalizer_mode='per_task' requires per-task normalizer loading, "
            "which is not yet integrated into the standard training entry. "
            "Use normalizer_mode='shared' or call get_normalizer(task_name=...) manually."
        )
    return dataset, normalizer


# ---------------------------------------------------------------------------
# Model & EMA
# ---------------------------------------------------------------------------


def _validate_ema_batchnorm_compatibility(model) -> None:
    """Reject BatchNorm whose parameters or running statistics can change."""
    unsafe = []
    for name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        has_trainable_params = any(
            p.requires_grad for p in module.parameters(recurse=False)
        )
        updates_running_stats = bool(module.training and module.track_running_stats)
        if has_trainable_params or updates_running_stats:
            unsafe.append(name or "<root>")
    if unsafe:
        names = "\n".join(f"  {name}" for name in unsafe[:8])
        if len(unsafe) > 8:
            names += f"\n  ... ({len(unsafe) - 8} more)"
        raise ValueError(
            "EMA is incompatible with active/trainable BatchNorm in this repository:\n"
            f"{names}\n"
            "EMA does not maintain BatchNorm running statistics consistently.\n"
            "Use group_norm, frozen_bn, freeze the BatchNorm backbone, or disable EMA."
        )


def build_model_and_ema(cfg, device, normalizer, rank=0):
    """Instantiate the agent model and, if configured, its EMA twin.

    ``rank`` gates whether a local EMA is built: only rank 0 (or every rank when
    the EMA teacher feeds the consistency loss) needs a full EMA copy. Non-rank-0
    workers receive ``ema_model=None`` — the Trainer guards every EMA site with
    ``self.use_ema = (ema_model is not None)``, so this is safe end-to-end.
    """
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.action_key
    model.to(device)

    if cfg.training.use_ema:
        _validate_ema_batchnorm_compatibility(model)

    ema_model = None
    ema_updater = None
    need_local_ema = cfg.training.use_ema and (
        rank == 0 or cfg.training.get("use_ema_teacher_for_consistency", False)
    )
    if need_local_ema:
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


def build_scheduler(cfg, optimizer, last_epoch=-1):
    """Build the LR scheduler with the correct total step count."""
    total_steps = compute_num_training_steps(cfg)
    return get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch,
        lr_min_ratio=cfg.training.get("lr_min_ratio", 0.1),
    )


def validate_gradient_accumulation(
    batches_per_epoch: int, gradient_accumulation_steps: int
) -> None:
    if batches_per_epoch <= 0:
        raise ValueError("train loader must contain at least one batch")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")


def build_optimizer_and_scheduler(cfg, model, batches_per_epoch, last_epoch=-1):
    """Build optimizer (via the agent's ``configure_optimizer``) and LR scheduler."""
    grad_accum = (
        cfg.get("training", {}).get("loop", {}).get("gradient_accumulation_steps", 1)
    )
    validate_gradient_accumulation(batches_per_epoch, grad_accum)
    optimizer = model.configure_optimizer(**cfg.optimizer)
    print_param_count(model)
    scheduler = build_scheduler(cfg, optimizer, last_epoch)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training Recipe
# ---------------------------------------------------------------------------


def print_training_recipe(cfg, *, world_size: int, batches_per_epoch: int) -> None:
    """Print configured training budgets; sample counts are nominal, not exact."""
    per_device_batch = int(cfg.dataloader.batch_size)
    grad_accum = int(cfg.training.get("loop", {}).get("gradient_accumulation_steps", 1))
    total_train_steps = int(cfg.training.loop.total_train_steps)
    nominal_global_batch = per_device_batch * world_size * grad_accum
    updates_per_epoch = (batches_per_epoch + grad_accum - 1) // grad_accum
    remainder = batches_per_epoch % grad_accum
    last_group_microbatches = grad_accum if remainder == 0 else remainder
    last_group_global_batch = per_device_batch * world_size * last_group_microbatches
    has_partial_accumulation_group = last_group_microbatches != grad_accum
    lr = cfg.optimizer.lr
    obs_lr = cfg.optimizer.get("obs_lr")
    obs_lr = lr if obs_lr is None else obs_lr

    rows = [
        ("per-device batch", per_device_batch),
        ("world size", world_size),
        ("gradient accumulation", grad_accum),
        ("nominal global batch", nominal_global_batch),
        ("batches / epoch", batches_per_epoch),
        ("optimizer updates / epoch", updates_per_epoch),
        ("partial accum group", "yes" if has_partial_accumulation_group else "no"),
        ("last group micro-batches", last_group_microbatches),
        ("last group global batch", last_group_global_batch),
        ("total optimizer updates", total_train_steps),
        ("nominal sample budget", nominal_global_batch * total_train_steps),
        ("drop_last", cfg.dataloader.drop_last),
        ("learning rate", f"{lr:.3e}"),
        ("obs learning rate", f"{obs_lr:.3e}"),
        ("warmup updates", cfg.training.lr_warmup_steps),
        ("scheduler", cfg.training.lr_scheduler),
        ("precision", "bf16" if cfg.training.get("use_bfloat16", False) else "fp32"),
        ("torch.compile", cfg.training.get("use_compile", False)),
        ("EMA", cfg.training.use_ema),
    ]
    print("=" * 60)
    print("Training Recipe")
    print("-" * 60)
    for label, value in rows:
        print(f"{label:<26}: {value}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


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
        raise ValueError(missing_rgb)
    if pc_color_noise is not None:
        raise ValueError(f"PC color_noise augmentation: {missing_rgb}")


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


def validate_config(cfg):
    """Validate common training config constraints.

    Called by all entry points before training or evaluation.
    """
    training_cfg = cfg.get("training", {})
    if training_cfg.get(
        "use_ema_teacher_for_consistency", False
    ) and not training_cfg.get("use_ema", False):
        raise ValueError(
            "use_ema_teacher_for_consistency=true requires training.use_ema=true"
        )

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

    if cfg.optimizer.get("obs_lr") is not None:
        if cfg.optimizer.obs_lr < 0:
            raise ValueError(
                "optimizer.obs_lr must be non-negative "
                "(0 freezes obs_encoder parameters)"
            )

    _validate_augmentation_consistency(cfg)
    _validate_aux_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")
