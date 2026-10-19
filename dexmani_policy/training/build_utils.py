"""Dataset, model and optimizer construction for training and integration smoke."""

import copy
import math
from collections.abc import Mapping
from pathlib import Path

import hydra
from omegaconf import open_dict
from torch.nn.modules.batchnorm import _BatchNorm

from dexmani_policy.agents.core.normalizer import AffineNormalizer, validate_normalizer_state
from dexmani_policy.datasets.augmentation import (
    validate_pc_color_augmentation,
    validate_random_keep_points,
)
from dexmani_policy.datasets.sampler import validate_dataset_splits
from dexmani_policy.training.logging import print_param_count
from dexmani_policy.training.lr_scheduler import get_scheduler
from dexmani_policy.utils.config import (
    FINGER_VECTOR_FIELDS,
    NON_NUMERIC_OBSERVATION_FIELDS,
    normalization_dims_from_config,
    resolve_normalization_spec,
    validate_action_clipping,
    validate_action_key_consistency,
    validate_dq_normalization,
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
]


def build_normalizer(dataset, spec: dict, action_key: str) -> AffineNormalizer:
    """Fit valid training rows, preserving rot6d coordinates for action_ee auto mode."""
    normalizer = AffineNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        mixed_action = key == "action" and mode == "auto" and action_key == "action_ee"
        mode = "limits" if mode == "auto" else mode
        normalizer.fit_field(key, dataset.iter_normalization_data(key), mode=mode)
        if mixed_action:
            field = normalizer[key]
            if field.params_dict["scale"].numel() <= 9:
                raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
            field.params_dict["scale"][3:9] = 1
            field.params_dict["offset"][3:9] = 0
            for name, value in (("min", -1), ("max", 1), ("mean", 0), ("std", 1)):
                field.input_stats[name][3:9] = value

    return normalizer


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None, verbose=True):
    """Build data and normalization from resolved config; restore saved stats on resume."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        from dexmani_policy.training.checkpoint import CheckpointStore

        path = Path(cfg.resume_from)
        resume_checkpoint = CheckpointStore(path.parent).load(path)
    if resume_checkpoint is not None:
        from dexmani_policy.utils.config import validate_source_recipe

        validate_source_recipe(resume_checkpoint.effective_config, cfg)
    spec = resolve_normalization_spec(cfg)
    dataset_cfg = copy.deepcopy(cfg.dataset)
    saved_recipes = (
        resume_checkpoint.resume_contract.get("data_recipe")
        if resume_checkpoint is not None
        else None
    )
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
    runtime = capture_real_runtime(dataset, cfg)
    with open_dict(cfg):
        cfg.data_identity = capture_data_identity(dataset)
        children = getattr(dataset, "datasets", [dataset])
        dims = []
        for child in children:
            widths = {
                key: int(child.replay_buffer[key].shape[-1])
                for key, mode in spec.items()
                if key != "action" and mode != "identity"
            }
            for key in FINGER_VECTOR_FIELDS.intersection(child.sensor_modalities):
                shape = child.replay_buffer[key].shape
                if len(shape) != 3 or shape[1:] != (5, 3):
                    raise ValueError(f"{key} requires stored shape (T, 5, 3), got {shape}")
                if key in widths:
                    widths[key] = 15
            if "point_cloud" in widths and getattr(child, "pc_dim", None) is not None:
                widths["point_cloud"] = child.pc_dim
            widths["action"] = int(child.replay_buffer[child.action_key].shape[-1]) + (
                9 if child.use_aux_ee else 0
            )
            dims.append(widths)
        if any(widths != dims[0] for widths in dims):
            raise ValueError("Child datasets must have matching normalization feature widths")
        cfg.normalization_dims = dims[0]
        if getattr(dataset, "sampling_strategy", None) == "weighted":
            cfg.sampling_recipe = {"weighted_quota": "cumulative_random_offset_v1"}
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
        normalizer = AffineNormalizer()
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
    validate_normalizer_state(normalizer, spec, normalization_dims_from_config(cfg))
    if verbose:
        summaries = []
        for child in children:
            counts = child.sampler.validity_summary
            summary = (
                f"{Path(child.zarr_path).name}: "
                f"train episodes={int(child.train_mask.sum()):,}, windows={len(child):,}"
            )
            rejected = counts["candidate"] - counts["valid"]
            if rejected:
                summary += f", rejected={rejected:,}"
            summaries.append(summary)
        print(f"[Dataset] {' | '.join(summaries)}")
    return dataset, normalizer


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
        names = "\n".join(f"  {name}" for name in unsafe[:8])
        if len(unsafe) > 8:
            names += f"\n  ... ({len(unsafe) - 8} more)"
        raise ValueError(
            "EMA is incompatible with active/trainable BatchNorm in this repository:\n"
            f"{names}\n"
            "EMA does not maintain BatchNorm running statistics consistently.\n"
            "Use group_norm, frozen_bn, freeze the BatchNorm backbone, or disable EMA."
        )


def build_model_and_ema(cfg, device, normalizer, rank=0, *, checkpoint=None, dataset=None):
    """Build the policy and local EMA for rank-0 evaluation or per-rank training targets."""
    model_cfg = cfg
    if checkpoint is not None:
        from dexmani_policy.training.checkpoint import checkpoint_agent_config, fix_state_dict

        model_cfg = checkpoint_agent_config(cfg, fix_state_dict(checkpoint.model_state))
    model = hydra.utils.instantiate(model_cfg.agent)
    from dexmani_policy.utils.validation import validate_observation_fields

    if "datasets" in cfg.dataset:
        for child in cfg.dataset.datasets:
            validate_observation_fields(
                model, (*child.get("sensor_modalities", ("joint_state",)), "task_text")
            )
    else:
        validate_observation_fields(model, cfg.dataset.get("sensor_modalities", ("joint_state",)))
    # Store local HF architecture and closed-text dimensions in the owning config.
    rgb = getattr(getattr(model, "obs_encoder", None), "backbone", None)
    with open_dict(cfg.agent):
        # Bind constructor defaults so future changes cannot reinterpret saved weights.
        import inspect

        for name, parameter in inspect.signature(type(model)).parameters.items():
            if name not in cfg.agent and parameter.default is not inspect.Parameter.empty:
                cfg.agent[name] = parameter.default
        if checkpoint is None and hasattr(rgb, "effective_config"):
            cfg.agent.rgb_backbone_config = rgb.effective_config
        pc_encoder = getattr(getattr(model, "obs_encoder", None), "pc_encoder", None)
        if type(pc_encoder).__name__ in {"PointNetDense", "Sparse3DEncoder"}:
            config = dict(pc_encoder.effective_config)
            for name, parameter in inspect.signature(type(pc_encoder)).parameters.items():
                if (
                    name != "input_channels"
                    and name not in config
                    and parameter.default is not inspect.Parameter.empty
                ):
                    config[name] = getattr(pc_encoder, name, parameter.default)
            cfg.agent.pc_encoder_config = config
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
    model.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    model.validate_normalizer()
    model.to(device)
    if checkpoint is None:
        bind_action_space_artifact(cfg, model, dataset)

    requires_ema_for_loss = model.requires_ema_for_loss
    if requires_ema_for_loss and not cfg.training.use_ema:
        raise ValueError(f"{type(model.action_decoder).__name__} requires training.use_ema=true")

    if cfg.training.use_ema:
        validate_ema_batchnorm(model)

    ema_model = None
    ema_updater = None
    need_local_ema = cfg.training.use_ema and (rank == 0 or requires_ema_for_loss)
    if need_local_ema:
        # Copy before compile/DDP; resume restores raw and EMA independently below.
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)
        with open_dict(cfg.ema):
            for name, parameter in inspect.signature(type(ema_updater)).parameters.items():
                if (
                    name != "model"
                    and name not in cfg.ema
                    and parameter.default is not inspect.Parameter.empty
                ):
                    cfg.ema[name] = parameter.default

    if checkpoint is not None:
        from dexmani_policy.training.resume import restore_model_weights

        restore_model_weights(checkpoint, model, ema_model, device)
        bind_action_space_artifact(cfg, model, dataset)

    return model, ema_model, ema_updater


def bind_action_space_artifact(cfg, model, dataset):
    from omegaconf import OmegaConf
    from dexmani_policy.utils.hand_artifact import (
        policy_representation, source_contract,
    )

    if policy_representation(model) is None:
        return
    if dataset is None:
        raise ValueError("Hand policies require the actual training dataset at artifact binding")
    record = model.action_space_artifact
    current = source_contract(dataset)
    saved = record["source_contract"]
    if saved != current:
        changed = sorted(
            key for key in saved.keys() | current.keys()
            if key not in saved or key not in current or saved[key] != current[key]
        )
        raise ValueError(f"Hand artifact source mismatch: {changed}; regenerate for this data recipe")
    with open_dict(cfg):
        cfg.action_space_artifact = OmegaConf.create(record)


def build_optimizer_and_scheduler(cfg, model, *, verbose=True):
    """Build initial optimizer/scheduler; resume loads their complete saved states."""
    optimizer = model.configure_optimizer(**cfg.optimizer)
    import inspect

    with open_dict(cfg.optimizer):
        for name, parameter in inspect.signature(model.configure_optimizer).parameters.items():
            if name not in cfg.optimizer and parameter.default is not inspect.Parameter.empty:
                cfg.optimizer[name] = parameter.default
    if cfg.training.lr_scheduler == "cosine_min_lr":
        with open_dict(cfg.training):
            cfg.training.lr_min_ratio = cfg.training.get("lr_min_ratio", 0.1)
    if verbose:
        print_param_count(model)
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=positive_int(cfg.training.loop.total_train_steps, "total_train_steps"),
        **(
            {"lr_min_ratio": cfg.training.get("lr_min_ratio", 0.1)}
            if cfg.training.lr_scheduler == "cosine_min_lr"
            else {}
        ),
    )
    return optimizer, scheduler


def print_training_recipe(
    cfg, *, world_size: int, batches_per_epoch: int, samples_per_rank: int
) -> None:
    """Distinguish nominal batches from the windows actually processed."""
    positive_int(batches_per_epoch, "batches_per_epoch")
    per_device_batch = int(cfg.dataloader.batch_size)
    grad_accum = int(cfg.training.get("loop", {}).get("gradient_accumulation_steps", 1))
    total_train_steps = int(cfg.training.loop.total_train_steps)
    nominal_global_batch = per_device_batch * world_size * grad_accum
    updates_per_epoch = (batches_per_epoch + grad_accum - 1) // grad_accum
    remainder = batches_per_epoch % grad_accum
    last_group_microbatches = grad_accum if remainder == 0 else remainder
    epoch_samples = min(samples_per_rank, batches_per_epoch * per_device_batch)
    last_group_global_batch = world_size * (
        epoch_samples - (updates_per_epoch - 1) * grad_accum * per_device_batch
    )
    full_epochs, partial_updates = divmod(total_train_steps, updates_per_epoch)
    processed_budget = world_size * (
        full_epochs * epoch_samples
        + min(partial_updates * grad_accum * per_device_batch, epoch_samples)
    )
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
        ("processed window budget", processed_budget),
        ("window counting", "includes sampler repeats / DDP padding; not unique source rows"),
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


def validate_aux_config(cfg):
    """Require joint-primary actions when supervising an auxiliary wrist pose."""
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


SPARSE3D_POLICY_TARGETS = frozenset(
    {
        "dexmani_policy.agents.core.sat_policy.SATPolicy",
        "dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
        "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy",
    }
)


def numeric_modalities(sensor_modalities) -> set:
    return {
        modality for modality in sensor_modalities if modality not in NON_NUMERIC_OBSERVATION_FIELDS
    }


def resolve_numeric_observation_fields(cfg) -> set:
    """Require the same numeric observation fields across tasks."""
    dataset = cfg.get("dataset", {})
    sensor_modalities = dataset.get("sensor_modalities")
    if sensor_modalities is not None:
        return numeric_modalities(sensor_modalities)

    child_datasets = dataset.get("datasets")
    if child_datasets is None:
        return set()

    child_sets = [
        numeric_modalities(child.get("sensor_modalities", [])) for child in child_datasets
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


def validate_encoder_normalization(cfg, spec) -> None:
    """Keep Sparse3D position content and geometric features in metric XYZ."""
    agent = cfg.get("agent", {})
    if agent.get("_target_") in SPARSE3D_POLICY_TARGETS and spec.get("point_cloud") != "identity":
        raise ValueError(
            "Sparse3DEncoder requires normalization.point_cloud: identity "
            "(position content and geometric features require metric XYZ)."
        )


def uses_diffusion_config(agent_config):
    """Recognize known Diffusion constructors; runtime checks cover custom agents."""
    if not agent_config.get("_target_"):
        return False
    from hydra.utils import get_class

    from dexmani_policy.agents.core.base_policy import UNetDiffusionPolicy
    from dexmani_policy.agents.core.dqrise_policy import DQRISEPolicy
    from dexmani_policy.agents.core.multitask_policy import MultiTaskPolicy
    from dexmani_policy.agents.core.r3d_policy import R3DPolicy

    cls = get_class(agent_config["_target_"])
    if cls is MultiTaskPolicy:
        return agent_config.get("action_decoder_type", "diffusion") == "diffusion"
    from dexmani_policy.agents.core.diffusion_policy import DiffusionPolicy
    from dexmani_policy.agents.core.dp3_policy import DP3Policy

    return cls in (UNetDiffusionPolicy, DiffusionPolicy, DP3Policy, DQRISEPolicy, R3DPolicy)


def validate_normalization_config(cfg):
    """Validate the feature-level normalization spec and its sensor-modality coverage."""
    rgb_transport = set()
    for child in cfg.dataset.get("datasets", [cfg.dataset]):
        if "rgb" in child.get("sensor_modalities", []):
            keep_uint8 = bool(child.get("rgb_keep_uint8", False))
            rgb_transport.add(keep_uint8)
    if len(rgb_transport) > 1:
        raise ValueError("RGB child datasets must have consistent rgb_keep_uint8")
    spec = resolve_normalization_spec(
        cfg, observation_fields=resolve_numeric_observation_fields(cfg)
    )
    from dexmani_policy.agents.core.dqrise_policy import DQRISEPolicy

    if issubclass(hydra.utils.get_class(cfg.agent._target_), DQRISEPolicy):
        validate_dq_normalization(spec)
    if uses_diffusion_config(cfg.agent):
        validate_action_clipping(spec, cfg.agent.get("clip_sample", True))
    validate_encoder_normalization(cfg, spec)


def validate_hand_config(cfg):
    """Align representation preparation and policy training before opening data."""
    target = cfg.agent.get("_target_")
    if target not in {
        "dexmani_policy.agents.core.lamp_policy.LAMPPolicy",
        "dexmani_policy.agents.core.temporal_latent_flow_policy.TemporalLatentFlowPolicy",
        "dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy",
    }:
        return
    for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
        if type(cfg.agent[name]) is not type(cfg[name]) or cfg.agent[name] != cfg[name]:
            raise ValueError(f"Hand policy agent.{name} must match {name}")
    for field, name in (("horizon", "horizon"), ("obs_horizon", "n_obs_steps")):
        if (
            type(cfg.dataset[field]) is not type(cfg[name])
            or cfg.dataset[field] != cfg[name]
        ):
            raise ValueError("Hand dataset H/Nobs must match the policy config")
    if cfg.dataset.get("use_aux_ee", False):
        raise ValueError("Hand representations do not support auxiliary action targets")

    if target == "dexmani_policy.agents.core.dqrise_policy.DQRISEPolicy":
        layout = {"action": (7, 19), "action_ee": (9, 21)}.get(cfg.action_key)
        if (cfg.agent.tcp_dim, cfg.agent.action_dim) != layout:
            raise ValueError(
                "Supported DQ-RISE layouts are action=7+12 or action_ee=9+12"
            )
    else:
        if (
            cfg.action_key != "action"
            or cfg.action_dim != 19
            or cfg.agent.state_dim != 19
        ):
            raise ValueError("LAMP/Temporal require joint action/state=7+12")

    if target == "dexmani_policy.agents.core.lamp_policy.LAMPPolicy":
        if (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) != (1, 1, 1):
            raise ValueError("LAMP requires H=Nobs=C=1")
        if (
            cfg.dataset.history_steps != cfg.agent.history_steps
            or cfg.lamp.history_steps != cfg.agent.history_steps
        ):
            raise ValueError("LAMP dataset/prior/policy history_steps must match")
        if (
            cfg.agent.get("num_inference_steps", 1) != 1
            or cfg.eval.inference_steps != 1
        ):
            raise ValueError("LAMP requires inference_steps=1 in agent/eval")
        if cfg.eval.get("inference_steps_list") not in (None, [1]):
            raise ValueError("LAMP inference_steps_list must be null or [1]")

    paths = []
    if "lamp" in cfg:
        paths.append(("lamp.prior_path", cfg.lamp.prior_path, cfg.agent.prior_path))
    if "codec" in cfg:
        paths.append(("codec.path", cfg.codec.path, cfg.agent.codec_path))
    if "codebook_path" in cfg:
        paths.append(("codebook_path", cfg.codebook_path, cfg.agent.codebook_path))
    for name, artifact_path, agent_path in paths:
        if (
            Path(artifact_path).expanduser().resolve()
            != Path(agent_path).expanduser().resolve()
        ):
            raise ValueError(f"{name} must match the agent artifact path")


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
        value = cfg.training.get("loop", {}).get(name, 100 if name == "log_interval_steps" else 1)
        positive_int(value, name)
    validate_window_contract(cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps)
    validate_hand_config(cfg)
    validate_dataset_splits(cfg.dataset)

    if cfg.optimizer.get("obs_lr") is not None and cfg.optimizer.obs_lr < 0:
        raise ValueError("optimizer.obs_lr must be non-negative (0 freezes obs_encoder parameters)")

    validate_aux_config(cfg)
    validate_normalization_config(cfg)
    validate_action_key_consistency(cfg)

    print("Config validation passed")


def compile_models(model, ema_model=None, **compile_kwargs):
    """Compile action backbones with reduce-overhead mode by default."""
    compile_kwargs.setdefault("mode", "reduce-overhead")
    model.compile_backbone(**compile_kwargs)
    if ema_model is not None:
        ema_model.compile_backbone(**compile_kwargs)


def capture_real_runtime(dataset, cfg) -> dict | None:
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
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
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
        # Preserve the stored point count; encoder FPS selects its own subset.
        runtime["pointcloud"] = dict(cloud)
    if "fingertip_xyz" in dataset.sensor_modalities:
        links = attrs.get("fingertip_link_names")
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Real fingertip_xyz requires five distinct non-empty link names")
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
