"""Dataset, model and optimizer construction for training and integration smoke."""

import copy
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.agents.core.normalizer import AffineNormalizer, validate_normalizer_state
from dexmani_policy.datasets.common.distributed_sampler import ResumableDistributedSampler
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset
from dexmani_policy.training.common.checkpoint import (
    CheckpointStore,
    checkpoint_agent_config,
    unwrap_state_dict,
)
from dexmani_policy.training.common.ema import EMA
from dexmani_policy.training.common.logging import print_param_count, print_training_recipe
from dexmani_policy.training.common.scheduler import get_scheduler
from dexmani_policy.training.resume import build_resume_contract, restore_model_weights
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.validation import (
    HAND_POLICY_TARGETS,
    validate_config,
    validate_ema_batchnorm,
)
from dexmani_policy.training.workspace import Workspace, prepare_output_dir
from dexmani_policy.utils.config import (
    FINGER_VECTOR_FIELDS,
    normalization_dims_from_config,
    positive_int,
    resolve_input_recipe,
    resolve_normalization_spec,
    resolve_resume_source,
    validate_observation_fields,
)
from dexmani_policy.utils.hand_representation import describe_hand_data
from dexmani_policy.utils.runtime import worker_init_fn


def prepare_training(cfg):
    """Resolve the saved recipe before validating and preparing a new run."""
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = CheckpointStore(Path(source).parent).load(source) if source is not None else None
    overrides = HydraConfig.get().overrides.task if source is not None else []
    effective_config = resolve_input_recipe(
        cfg, overrides=overrides, resume_checkpoint=checkpoint
    )
    with open_dict(effective_config):
        effective_config.resume_from = source
        # Resume writes into this invocation's directory, never the source run.
        effective_config.workspace.output_dir = cfg.workspace.output_dir
    validate_config(effective_config)
    prepare_output_dir(effective_config.workspace.output_dir)
    return effective_config, checkpoint


def bind_defaults(config, constructor):
    """Bind constructor defaults so saved weights retain their original recipe."""
    with open_dict(config):
        for name, parameter in inspect.signature(constructor).parameters.items():
            if name not in config and parameter.default is not inspect.Parameter.empty:
                config[name] = parameter.default


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
    """Build data and fit fresh statistics; full resume restores the model's normalizer."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        raise ValueError("Resume requires the checkpoint loaded by the training entry point")
    spec = resolve_normalization_spec(cfg)
    dataset_cfg = copy.deepcopy(cfg.dataset)
    saved_recipes = (
        resume_checkpoint.effective_config["data_recipe"] if resume_checkpoint is not None else None
    )
    if saved_recipes is not None:
        children = dataset_cfg.get("datasets", [dataset_cfg])
        if len(children) != len(saved_recipes):
            raise ValueError("Saved data_recipe task count does not match dataset")
        for child, recipe in zip(children, saved_recipes):
            if "split_manifest" in recipe:
                with open_dict(child):
                    child.saved_split = recipe["split_manifest"]
    dataset = hydra.utils.instantiate(dataset_cfg)
    children = dataset.datasets if isinstance(dataset, MultiTaskDataset) else [dataset]
    reference = children[0]
    runtime = capture_real_runtime(dataset, cfg)
    with open_dict(cfg):
        cfg.data_identity = capture_data_identity(dataset)
        # MultiTaskDataset already enforces a shared output layout.
        widths = {
            key: int(reference.replay_buffer[key].shape[-1])
            for key, mode in spec.items()
            if key != "action" and mode != "identity"
        }
        for key in FINGER_VECTOR_FIELDS.intersection(reference.sensor_modalities):
            shape = reference.replay_buffer[key].shape
            if shape[1:] != (5, 3):
                raise ValueError(f"{key} requires stored shape (T, 5, 3), got {shape}")
            if key in widths:
                widths[key] = 15
        if "point_cloud" in widths and reference.pc_dim is not None:
            widths["point_cloud"] = reference.pc_dim
        widths["action"] = int(reference.replay_buffer[reference.action_key].shape[-1]) + (
            9 if reference.use_aux_ee else 0
        )
        cfg.normalization_dims = widths
        if isinstance(dataset, MultiTaskDataset) and dataset.sampling_strategy == "weighted":
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
    normalizer = None
    if resume_checkpoint is None:
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


def build_model_and_ema(cfg, device, normalizer, rank=0, *, dataset, checkpoint=None):
    """Build raw weights and EMA for rank-0 checkpoints or per-rank training targets."""
    model_cfg = cfg
    if checkpoint is not None:
        model_cfg = checkpoint_agent_config(cfg, unwrap_state_dict(checkpoint.model_state))
    model = hydra.utils.instantiate(model_cfg.agent)
    if "datasets" in cfg.dataset:
        for child in cfg.dataset.datasets:
            validate_observation_fields(
                model, (*child.get("sensor_modalities", ("joint_state",)), "task_text")
            )
    else:
        validate_observation_fields(model, cfg.dataset.get("sensor_modalities", ("joint_state",)))
    rgb = getattr(getattr(model, "obs_encoder", None), "backbone", None)
    bind_defaults(cfg.agent, type(model))
    with open_dict(cfg.agent):
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
    model.action_key = cfg.action_key
    model.set_normalization_spec(
        resolve_normalization_spec(cfg), normalization_dims_from_config(cfg)
    )
    if checkpoint is None:
        model.initialize_training()
        model.load_normalizer_from_dataset(normalizer)
        model.validate_normalizer()

    validate_hand_pretraining(cfg, model, dataset, checkpoint=checkpoint)
    if checkpoint is None:
        model.to(device)

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
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)
        bind_defaults(cfg.ema, type(ema_updater))

    if checkpoint is not None:
        restore_model_weights(checkpoint, model, ema_model, device)

    return model, ema_model, ema_updater


def validate_hand_pretraining(cfg, model, dataset, *, checkpoint=None):
    if cfg.agent.get("_target_") not in HAND_POLICY_TARGETS:
        return
    current = describe_hand_data(dataset)
    saved = model.hand_data if checkpoint is None else checkpoint.effective_config["hand_data"]
    changed = [key for key in current if saved[key] != current[key]]
    if changed:
        raise ValueError(
            f"Hand pretraining data mismatch: {changed}; regenerate for this data recipe"
        )
    with open_dict(cfg):
        cfg.hand_data = current


def build_optimizer_and_scheduler(cfg, model, *, verbose=True):
    """Build initial optimizer/scheduler; resume loads their complete saved states."""
    optimizer = model.configure_optimizer(**cfg.optimizer)
    bind_defaults(cfg.optimizer, model.configure_optimizer)
    if cfg.training.lr_scheduler == "cosine_min_lr":
        with open_dict(cfg.training):
            cfg.training.lr_min_ratio = cfg.training.get("lr_min_ratio", 0.1)
    if verbose:
        print_param_count(model)
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=cfg.training.loop.total_train_steps,
        **(
            {"lr_min_ratio": cfg.training.get("lr_min_ratio", 0.1)}
            if cfg.training.lr_scheduler == "cosine_min_lr"
            else {}
        ),
    )
    return optimizer, scheduler


def compile_models(model, ema_model=None, **compile_kwargs):
    """Compile action backbones with reduce-overhead mode by default."""
    compile_kwargs.setdefault("mode", "reduce-overhead")
    model.compile_backbone(**compile_kwargs)
    if ema_model is not None:
        ema_model.compile_backbone(**compile_kwargs)


def capture_real_runtime(dataset, cfg) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    if isinstance(dataset, MultiTaskDataset):
        return None
    buffer = dataset.replay_buffer
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
    if isinstance(dataset, MultiTaskDataset):
        return {
            "tasks": {
                name: capture_data_identity(child)
                for name, child in zip(dataset.task_names, dataset.datasets)
            }
        }
    return {"revision": dataset.data_revision}


def loader_options(cfg):
    options = OmegaConf.to_container(cfg.dataloader, resolve=True)
    if options.get("num_workers", 0) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    return options


def build_train_loader(cfg, dataset, *, rank=0, world_size=1):
    options = loader_options(cfg)
    sampler = ResumableDistributedSampler(
        dataset,
        batch_size=options["batch_size"],
        num_replicas=world_size,
        rank=rank,
        shuffle=options.pop("shuffle", False),
        seed=cfg.training.seed,
        drop_last=False,
    )
    n, b = sampler.num_samples, options["batch_size"]
    a = cfg.training.loop.gradient_accumulation_steps
    if a > 1 and not options.get("drop_last", False) and n % b and ((n + b - 1) // b) % a != 1:
        raise ValueError(
            f"Mixed micro-batch sizes in accumulation group: N={n}, B={b}, A={a}; "
            "use drop_last=True or gradient_accumulation_steps=1"
        )
    generator = torch.Generator().manual_seed(cfg.training.seed + rank)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **options,
    )
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader is empty; check dataset length, batch_size and drop_last"
        )
    return loader


@dataclass
class TrainingComponents:
    """Per-process training components; only rank 0 owns a workspace."""

    device: torch.device
    model: torch.nn.Module
    ema_model: torch.nn.Module | None
    ema_updater: EMA | None
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: DataLoader
    workspace: Workspace | None


def build_train_components(cfg, *, resume_checkpoint=None):
    """Build components, reusing a source already validated by CheckpointStore.load."""
    if cfg.training.get("num_gpus", 1) != 1:
        raise ValueError("Single-device training requires num_gpus=1; use train_ddp.py for DDP")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires GPU.")

    device = torch.device(cfg.training.device)

    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=resume_checkpoint)

    train_loader = build_train_loader(cfg, dataset)

    model, ema_model, ema_updater = build_model_and_ema(
        cfg, device, normalizer, checkpoint=resume_checkpoint, dataset=dataset
    )

    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    print_training_recipe(
        cfg,
        world_size=1,
        batches_per_epoch=len(train_loader),
        samples_per_rank=train_loader.sampler.num_samples,
    )

    workspace = hydra.utils.instantiate(cfg.workspace)

    return TrainingComponents(
        device=device,
        model=model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        workspace=workspace,
    )


def build_trainer(cfg, components):
    sampler = components.train_loader.sampler
    return Trainer(
        device=components.device,
        model=components.model,
        ema_model=components.ema_model,
        ema_updater=components.ema_updater,
        optimizer=components.optimizer,
        scheduler=components.scheduler,
        train_loader=components.train_loader,
        workspace=components.workspace,
        train_loop_cfg=TrainLoopConfig(**OmegaConf.to_container(cfg.training.loop, resolve=True)),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        use_bfloat16=cfg.training.get("use_bfloat16", False),
        is_main_process=sampler.rank == 0,
        distributed=sampler.num_replicas > 1,
        resume_contract=build_resume_contract(components.train_loader),
        effective_config=cfg,
    )
