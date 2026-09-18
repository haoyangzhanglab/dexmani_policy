"""Shared build functions for training/eval entry points."""

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.common.config import validate_action_key_consistency
from dexmani_policy.common.normalizer import (
    LinearNormalizer,
    NON_NUMERIC_OBSERVATION_FIELDS,
    build_mixed_action_normalizer,
    validate_normalization_spec,
    validate_normalizer_state,
)
from dexmani_policy.common.pytorch_util import print_param_count
from dexmani_policy.training.lr_scheduler import (
    compute_num_training_steps,
    get_scheduler,
)

__all__ = [
    "build_dataset_and_normalizer",
    "build_normalizer",
    "resolve_normalization_spec",
    "attach_normalization_spec",
    "build_model_and_ema",
    "build_scheduler",
    "build_optimizer_and_scheduler",
    "validate_config",
    "compute_num_training_steps",
    "validate_gradient_accumulation",
    "print_training_recipe",
]

# ---------------------------------------------------------------------------
# Normalization spec & builder
# ---------------------------------------------------------------------------

def resolve_normalization_spec(cfg) -> dict:
    """Extract and validate the top-level feature-level normalization spec.

    Delegates the mode grammar to the shared ``validate_normalization_spec`` so
    training and deployment can never diverge on normalization semantics.
    Returns a plain ``{field: mode}`` mapping.
    """
    normalization = cfg.get("normalization")
    if normalization is None:
        raise ValueError(
            "config.normalization is required: every Policy config must declare a "
            "top-level `normalization:` mapping (e.g. {joint_state: limits, action: auto})."
        )
    if isinstance(normalization, DictConfig):
        normalization = OmegaConf.to_container(normalization, resolve=True)
    if not isinstance(normalization, dict):
        raise ValueError("config.normalization must be a mapping of field -> mode")

    return validate_normalization_spec(normalization)


def build_normalizer(dataset, spec: dict, action_key: str) -> LinearNormalizer:
    """Build a ``LinearNormalizer`` from a dataset and the resolved normalization spec.

    ``identity`` fields register no params.  ``action:auto`` keeps the current
    ``action`` / ``action_ee`` / ``use_aux_ee`` semantics via the existing
    ``build_mixed_action_normalizer`` or ``limits``.  All other fields use full
    dataset statistics (single-chunk ``fit_field`` fast path, or streaming
    ``fit_field_chunks`` for multi-chunk datasets).
    """
    normalizer = LinearNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        if key == "action" and mode == "auto":
            action = np.concatenate(list(dataset.iter_normalization_data("action")), axis=0)
            if action_key == "action_ee":
                normalizer["action"] = build_mixed_action_normalizer(action)
            else:
                normalizer.fit_field("action", action, mode="limits")
            continue

        chunks = list(dataset.iter_normalization_data(key))
        if len(chunks) == 1:
            normalizer.fit_field(key, chunks[0], mode=mode)
        else:
            normalizer.fit_field_chunks(key, chunks, mode=mode)

    return normalizer


def attach_normalization_spec(model, cfg) -> None:
    """Attach the resolved semantic normalization spec to a model (and its EMA twin)."""
    model.normalization_spec = resolve_normalization_spec(cfg)


def build_dataset_and_normalizer(cfg):
    """Instantiate the dataset and build the normalizer from the config-driven spec.

    The caller is responsible for resolving OmegaConf interpolations before
    calling this function (DDP paths call ``OmegaConf.resolve(cfg)`` in the
    parent process before ``mp.spawn``).
    """
    spec = resolve_normalization_spec(cfg)
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = build_normalizer(dataset, spec, cfg.action_key)
    validate_normalizer_state(normalizer, spec)
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

    ``rank`` gates whether a local EMA is built: rank 0 always owns the evaluation
    EMA, while every rank owns one when the model declares an EMA-dependent loss.
    Non-rank-0
    workers receive ``ema_model=None`` — the Trainer guards every EMA site with
    ``self.use_ema = (ema_model is not None)``, so this is safe end-to-end.
    """
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.action_key
    attach_normalization_spec(model, cfg)
    model.to(device)

    requires_ema_for_loss = model.requires_ema_for_loss
    if requires_ema_for_loss and not cfg.training.use_ema:
        raise ValueError(
            f"{type(model.action_decoder).__name__} requires training.use_ema=true"
        )

    if cfg.training.use_ema:
        _validate_ema_batchnorm_compatibility(model)

    ema_model = None
    ema_updater = None
    need_local_ema = cfg.training.use_ema and (
        rank == 0 or requires_ema_for_loss
    )
    if need_local_ema:
        ema_model = hydra.utils.instantiate(cfg.agent)
        ema_model.load_normalizer_from_dataset(normalizer)
        ema_model.action_key = model.action_key
        attach_normalization_spec(ema_model, cfg)
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


_METRIC_POINTNEXT_ENCODER_TYPES = frozenset({"pointnext", "pointnext_tokenizer"})


def _numeric_modalities(sensor_modalities) -> set:
    return {
        modality
        for modality in sensor_modalities
        if modality not in NON_NUMERIC_OBSERVATION_FIELDS
    }


def _resolve_numeric_observation_fields(cfg) -> set:
    """Return the exact numeric observation field set for coverage checking.

    Single-task: the dataset's own ``sensor_modalities``.  MultiTask: all child
    datasets must declare the *same* numeric modality set — a mismatch is a
    startup fail-fast, never a silently-constructed union.
    """
    dataset = cfg.get("dataset", {})
    sensor_modalities = dataset.get("sensor_modalities")
    if sensor_modalities is not None:
        return _numeric_modalities(sensor_modalities)

    child_datasets = dataset.get("datasets")
    if child_datasets is None:
        return set()

    child_sets = [_numeric_modalities(child.get("sensor_modalities", [])) for child in child_datasets]
    distinct = {frozenset(s) for s in child_sets}
    if len(distinct) > 1:
        raise ValueError(
            "MultiTask child datasets declare inconsistent numeric observation "
            f"field sets: {[sorted(s) for s in child_sets]}. Shared Policy "
            "normalization requires every task to observe the same numeric "
            "fields; per-task normalization is not supported."
        )
    return child_sets[0] if child_sets else set()


def _validate_encoder_normalization_contract(cfg, spec) -> None:
    """Reject metric-sensitive PointNext point-cloud encoders paired with a
    non-identity ``point_cloud`` normalization mode.

    PointNeXT (``encoder_type in {pointnext, pointnext_tokenizer}``) uses FPS and
    fixed ball-query radii directly on point-cloud coordinates; per-axis min-max
    (``limits``) rescaling would change Euclidean geometry underneath a radius
    that stays fixed. This single rule covers every agent exposing
    ``agent.encoder_type`` (DP3, DQ-RISE, ManiFlow, SAT) — no per-agent special
    case is kept.
    """
    agent = cfg.get("agent", {})
    encoder_type = agent.get("encoder_type")
    if encoder_type in _METRIC_POINTNEXT_ENCODER_TYPES and spec.get("point_cloud") != "identity":
        raise ValueError(
            f"agent.encoder_type={encoder_type!r} requires normalization.point_cloud: "
            "identity (PointNeXT FPS/ball-query radii need metric-space coordinates; "
            "per-axis min-max would change Euclidean geometry)."
        )


def _validate_normalization_config(cfg):
    """Validate the feature-level normalization spec and its sensor-modality coverage."""
    spec = resolve_normalization_spec(cfg)
    numeric_fields = _resolve_numeric_observation_fields(cfg)
    # Re-validate with exact-coverage enforcement (missing AND extra fields),
    # reusing the same shared grammar `resolve_normalization_spec` already applied.
    validate_normalization_spec(dict(spec), observation_fields=numeric_fields)
    _validate_encoder_normalization_contract(cfg, spec)


def validate_config(cfg):
    """Validate common training config constraints.

    Called by all entry points before training or evaluation.
    """
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
    _validate_normalization_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")
