"""Dataset, model and optimizer construction for training and integration smoke."""

import copy
import math
from collections.abc import Mapping

import hydra
from omegaconf import open_dict
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.agents.normalization import (
    NON_NUMERIC_OBSERVATION_FIELDS,
    LinearNormalizer,
    build_mixed_action_normalizer_chunks,
    resolve_normalization_spec,
    uses_diffusion_config,
    validate_action_clipping,
    validate_dq_normalization,
    validate_normalization_spec,
    validate_normalizer_state,
)
from dexmani_policy.datasets.sampler import validate_dataset_splits
from dexmani_policy.training.logging import print_param_count
from dexmani_policy.training.lr_scheduler import get_scheduler
from dexmani_policy.utils.config import (
    validate_action_key_consistency,
    validate_window_contract,
)
from dexmani_policy.utils.validation import positive_int

__all__ = [
    "build_dataset_and_normalizer",
    "build_model_and_ema",
    "build_normalizer",
    "build_optimizer_and_scheduler",
    "compile_models",
    "print_training_recipe",
    "validate_config",
    "validate_gradient_accumulation",
]


def build_normalizer(dataset, spec: dict, action_key: str) -> LinearNormalizer:
    """Fit the resolved spec on unique source rows from valid training windows.

    ``identity`` registers no parameters. ``action:auto`` uses mixed normalization
    for EEF actions and limits for joint actions, including enabled auxiliary
    targets. All fitted fields use the dataset's role-specific training rows.
    """
    normalizer = LinearNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        if key == "action" and mode == "auto":
            if action_key == "action_ee":
                normalizer["action"] = build_mixed_action_normalizer_chunks(
                    dataset.iter_normalization_data("action")
                )
                continue
            mode = "limits"

        normalizer.fit_field_chunks(
            key, dataset.iter_normalization_data(key), mode=mode
        )

    return normalizer


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None):
    """Instantiate the dataset and build the normalizer from the config-driven spec.

    The caller is responsible for resolving OmegaConf interpolations before
    calling this function (DDP paths call ``OmegaConf.resolve(cfg)`` in the
    parent process before ``mp.spawn``).
    """
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        from pathlib import Path

        from dexmani_policy.training.checkpoint import CheckpointStore
        path = Path(cfg.resume_from)
        resume_checkpoint = CheckpointStore(path.parent).load(path)
    spec = resolve_normalization_spec(cfg)
    dataset_cfg = copy.deepcopy(cfg.dataset)
    saved_recipes = getattr(resume_checkpoint, "resume_contract", {}).get("data_recipe")
    if cfg.get("resume_from") is not None and saved_recipes is None:
        raise ValueError("Full resume requires saved data_recipe")
    if saved_recipes is not None:
        children = dataset_cfg.get("datasets", [dataset_cfg])
        if len(children) != len(saved_recipes):
            raise ValueError("Saved data_recipe task count does not match dataset")
        for child, recipe in zip(children, saved_recipes):
            if "split_manifest" in recipe:
                with open_dict(child):
                    child.saved_split = recipe["split_manifest"]
    dataset = hydra.utils.instantiate(dataset_cfg)
    runtime = _capture_real_runtime(dataset, cfg)
    with open_dict(cfg):
        cfg.data_identity = capture_data_identity(dataset)
        children = getattr(dataset, "datasets", [dataset])
        cfg.data_recipe = [
            dict(
                child.data_recipe,
                windows=child.sampler.validity_summary,
                observation_rows=len(child.sampler.observation_source_rows),
                action_rows=len(child.sampler.action_source_rows),
            )
            for child in children
        ]
        if runtime is not None:
            cfg.real_runtime = runtime
        else:
            cfg.pop("real_runtime", None)
    if cfg.get("resume_from") is not None:
        from dexmani_policy.training.checkpoint import fix_state_dict

        state = fix_state_dict(resume_checkpoint.model_state)
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(
            {
                key.removeprefix("normalizer."): value
                for key, value in state.items()
                if key.startswith("normalizer.")
            }
        )
        with open_dict(cfg):
            cfg.normalizer_source = "saved_checkpoint"
    else:
        normalizer = build_normalizer(dataset, spec, cfg.action_key)
    validate_normalizer_state(normalizer, spec)
    return dataset, normalizer


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


def build_model_and_ema(cfg, device, normalizer, rank=0, *, checkpoint=None):
    """Build the Agent and the local EMA required by evaluation or training loss.

    With EMA enabled, rank 0 owns the evaluation copy; EMA-dependent losses
    require a copy on every rank. Resume restores raw and EMA independently.
    """
    model_cfg = cfg
    if checkpoint is not None:
        from dexmani_policy.agents.loader import checkpoint_agent_config
        from dexmani_policy.training.checkpoint import fix_state_dict
        model_cfg = checkpoint_agent_config(
            cfg, fix_state_dict(checkpoint.model_state)
        )
    model = hydra.utils.instantiate(model_cfg.agent)
    from dexmani_policy.utils.validation import validate_observation_fields
    if "datasets" in cfg.dataset:
        for child in cfg.dataset.datasets:
            validate_observation_fields(model, (*child.get("sensor_modalities", ("joint_state",)), "task_text"))
    else:
        validate_observation_fields(model, cfg.dataset.get("sensor_modalities", ("joint_state",)))
    # Store local HF architecture and closed-text dimensions in the owning config.
    rgb = getattr(getattr(model, "obs_encoder", None), "backbone", None)
    with open_dict(cfg.agent):
        if hasattr(rgb, "architecture"):
            if cfg.agent.get("rgb_backbone_config") is None:
                cfg.agent.rgb_backbone_config = {}
            with open_dict(cfg.agent.rgb_backbone_config):
                cfg.agent.rgb_backbone_config.architecture = rgb.architecture
        if getattr(model, "task_emb_table", None) is not None:
            cfg.agent.text_embed_dim = model.text_embed_dim
    if checkpoint is None:
        model.initialize_training()
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.action_key
    model.set_normalization_spec(resolve_normalization_spec(cfg))
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
    need_local_ema = cfg.training.use_ema and (rank == 0 or requires_ema_for_loss)
    if need_local_ema:
        # Copy the already constructed, initialized model before compile/DDP.
        # A checkpoint supplies raw and historical EMA weights independently below.
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    if checkpoint is not None:
        from dexmani_policy.training.resume import restore_model_weights
        restore_model_weights(checkpoint, model, ema_model, device)

    if rank == 0:
        from dexmani_policy.training.logging import print_storage_dtypes
        print_storage_dtypes(model, ema_model, autocast=cfg.training.get("use_bfloat16", False))
    return model, ema_model, ema_updater


# Optimizer & Scheduler


def validate_gradient_accumulation(
    batches_per_epoch: int, gradient_accumulation_steps: int
) -> None:
    if batches_per_epoch <= 0:
        raise ValueError("train loader must contain at least one batch")
    positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")


def build_optimizer_and_scheduler(cfg, model, batches_per_epoch, last_epoch=-1, *, verbose=True):
    """Build optimizer (via the agent's ``configure_optimizer``) and LR scheduler."""
    grad_accum = (
        cfg.get("training", {}).get("loop", {}).get("gradient_accumulation_steps", 1)
    )
    validate_gradient_accumulation(batches_per_epoch, grad_accum)
    optimizer = model.configure_optimizer(**cfg.optimizer)
    if verbose:
        print_param_count(model)
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=positive_int(cfg.training.loop.total_train_steps, "total_train_steps"),
        last_epoch=last_epoch,
        **({"lr_min_ratio": cfg.training.get("lr_min_ratio", 0.1)}
           if cfg.training.lr_scheduler == "cosine_min_lr" else {}),
    )
    return optimizer, scheduler


# Training Recipe


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
        (
            "temporal sampling",
            "recorded_rows (no interpolation; eligibility follows data_recipe)",
        ),
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


# Config Validation


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

    child_sets = [
        _numeric_modalities(child.get("sensor_modalities", []))
        for child in child_datasets
    ]
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
    if (
        encoder_type in _METRIC_POINTNEXT_ENCODER_TYPES
        and spec.get("point_cloud") != "identity"
    ):
        raise ValueError(
            f"agent.encoder_type={encoder_type!r} requires normalization.point_cloud: "
            "identity (PointNeXT FPS/ball-query radii need metric-space coordinates; "
            "per-axis min-max would change Euclidean geometry)."
        )


def _validate_normalization_config(cfg):
    """Validate the feature-level normalization spec and its sensor-modality coverage."""
    rgb_transport = set()
    for child in cfg.dataset.get("datasets", [cfg.dataset]):
        if "rgb" in child.get("sensor_modalities", []):
            keep_uint8 = bool(child.get("rgb_keep_uint8", False))
            rgb_transport.add(keep_uint8)
            if keep_uint8 and (cfg.get("normalization") or {}).get("rgb") != "identity":
                raise ValueError(
                    "rgb_keep_uint8 requires normalization.rgb=identity because RGB "
                    "must remain uint8 until the vision preprocessing boundary."
                )
    if len(rgb_transport) > 1:
        raise ValueError("RGB child datasets must have consistent rgb_keep_uint8")
    spec = resolve_normalization_spec(cfg)
    from dexmani_policy.agents.core.dqrise import DQRISEAgent

    if issubclass(hydra.utils.get_class(cfg.agent._target_), DQRISEAgent):
        validate_dq_normalization(spec)
    if uses_diffusion_config(cfg.agent):
        validate_action_clipping(spec, cfg.agent.get("clip_sample", True))
    numeric_fields = _resolve_numeric_observation_fields(cfg)
    # Re-validate with exact-coverage enforcement (missing AND extra fields),
    # reusing the same shared grammar `resolve_normalization_spec` already applied.
    validate_normalization_spec(dict(spec), observation_fields=numeric_fields)
    _validate_encoder_normalization_contract(cfg, spec)


def validate_config(cfg):
    """Validate common training config constraints.

    Called by training and config-only smoke entry points.
    """
    if cfg.get("max_updates") is not None:
        positive_int(cfg.max_updates, "max_updates")
    for name in ("total_train_steps", "log_interval_steps", "gradient_accumulation_steps"):
        value = cfg.training.get("loop", {}).get(name, 100 if name == "log_interval_steps" else 1)
        positive_int(value, name)
    validate_window_contract(cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps)
    validate_dataset_splits(cfg.dataset)

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError(
            "optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)"
        )

    _validate_augmentation_consistency(cfg)
    _validate_aux_config(cfg)
    _validate_normalization_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")


def compile_models(model, ema_model=None, **compile_kwargs):
    """torch.compile the backbone of *model* and optionally *ema_model*.

    The shared ``compile_backbone()`` protocol is defined in
    :class:`~dexmani_policy.agents.core.base.BaseAgent`.

    Keyword arguments are forwarded to :func:`torch.compile`; defaults to
    ``mode='reduce-overhead'``.
    """
    compile_kwargs.setdefault("mode", "reduce-overhead")
    model.compile_backbone(**compile_kwargs)
    if ema_model is not None:
        ema_model.compile_backbone(**compile_kwargs)


def _capture_real_runtime(dataset, cfg) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    buffer = getattr(dataset, "replay_buffer", None)
    if buffer is None:
        return None
    attrs = buffer.attrs
    if attrs.get("format") != "dexmani.real.canonical":
        if attrs.get("domain") == "real" or str(attrs.get("format", "")).startswith(
            "dexmani.real."
        ):
            raise ValueError("Real training requires format='dexmani.real.canonical'")
        return None
    if attrs.get("task_name") != cfg.task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    dt = attrs.get("dt")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Real canonical dt must be finite and positive")
    runtime = {"dt": float(dt)}
    if "point_cloud" in dataset.sensor_modalities:
        cloud = attrs.get("pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real point_cloud requires root pointcloud_config")
        count = positive_int(cloud.get("num_points"), "pointcloud_config.num_points")
        shape = buffer["point_cloud"].shape
        if len(shape) != 3 or shape[1] != count:
            raise ValueError("Stored point count disagrees with pointcloud_config")
        encoder = cfg.agent.get("pc_encoder_config") or {}
        for configured in (cfg.agent.get("num_points"), encoder.get("num_points")):
            if configured is not None and configured != count:
                raise ValueError(
                    "Agent point count disagrees with the actual training cloud"
                )
        runtime["pointcloud"] = dict(cloud)
    if "fingertip_points" in dataset.sensor_modalities:
        links = attrs.get("fingertip_link_names")
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError(
                "Real fingertip_points requires five distinct non-empty link names"
            )
        runtime["fingertip_link_names"] = list(links)
    return runtime


def capture_data_identity(dataset):
    """Producer-declared revisions captured once while loading, never scanned at save."""
    if hasattr(dataset, "task_names") and hasattr(dataset, "datasets"):
        return {
            "tasks": {
                name: capture_data_identity(child)
                for name, child in zip(dataset.task_names, dataset.datasets)
            }
        }
    return {"revision": getattr(dataset, "data_revision", None)}
