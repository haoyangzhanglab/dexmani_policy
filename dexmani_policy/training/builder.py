"""Model and training component construction."""

import copy
import inspect
from dataclasses import dataclass
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint import (
    load_training_checkpoint,
    prepare_restore_config,
)
from dexmani_policy.common.config import (
    normalization_dims_from_config,
    resolve_input_recipe,
    resolve_normalization_spec,
    resolve_resume_source,
    validate_observation_fields,
)
from dexmani_policy.common.runtime import close_resources, set_seed
from dexmani_policy.common.experiments import archive_resume_milestones
from dexmani_policy.training.common.ema import EMA
from dexmani_policy.training.common.logging import print_param_count, print_training_recipe
from dexmani_policy.training.common.scheduler import get_scheduler
from dexmani_policy.training.data import build_train_loader, close_data_loader
from dexmani_policy.training.resume import restore_model_weights
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.validation import validate_config
from dexmani_policy.training.workspace import Workspace


def prepare_training(cfg):
    """Resolve the saved recipe before validating the training configuration."""
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = load_training_checkpoint(source) if source is not None else None
    overrides = HydraConfig.get().overrides.task if source is not None else []
    effective_config = resolve_input_recipe(cfg, overrides=overrides, resume_checkpoint=checkpoint)
    with open_dict(effective_config):
        effective_config.resume_from = source
        # Resume writes into this invocation's directory, never the source run.
        effective_config.workspace.output_dir = cfg.workspace.output_dir
    validate_config(effective_config)
    return effective_config, checkpoint


def bind_defaults(config, constructor):
    """Bind constructor defaults so saved weights retain their original recipe."""
    with open_dict(config):
        for name, parameter in inspect.signature(constructor).parameters.items():
            if name not in config and parameter.default is not inspect.Parameter.empty:
                config[name] = parameter.default


def build_model_and_ema(cfg, device, normalizer, data_description, rank=0, *, checkpoint=None):
    """Build raw weights and EMA for rank-0 checkpoints or per-rank training targets."""
    model_cfg = cfg.agent if checkpoint is None else prepare_restore_config(cfg.agent)
    model = hydra.utils.instantiate(model_cfg)
    validate_observation_fields(model, cfg.dataset.sensor_modalities)
    model.set_normalization_spec(
        resolve_normalization_spec(cfg), data_description["normalization_dims"]
    )
    if checkpoint is None:
        model.initialize_training(data_description)
        model.load_normalizer(normalizer)
    if checkpoint is not None and dict(cfg.policy_metadata) != model.inference_metadata:
        raise ValueError("Restored policy metadata differs from saved construction")
    if checkpoint is None:
        model.to(device)

    requires_ema_for_loss = model.requires_ema_for_loss
    if requires_ema_for_loss and not cfg.training.use_ema:
        raise ValueError(f"{type(model.action_decoder).__name__} requires training.use_ema=true")

    if cfg.training.use_ema:
        active_batchnorm = [
            name or "<root>" for name, module in model.named_modules()
            if isinstance(module, _BatchNorm) and (
                any(p.requires_grad for p in module.parameters(recurse=False))
                or (module.training and module.track_running_stats)
            )
        ]
        if active_batchnorm:
            raise ValueError(
                f"EMA requires frozen BatchNorm or GroupNorm; active modules: {active_batchnorm}"
            )

    # Freeze before EMA chooses which parameters need FP32 accumulation.
    obs_lr = cfg.optimizer.get("obs_lr")
    if (cfg.optimizer.lr if obs_lr is None else obs_lr) == 0:
        model.obs_encoder.requires_grad_(False)
    ema_model = None
    ema_updater = None
    need_local_ema = cfg.training.use_ema and (rank == 0 or requires_ema_for_loss)
    if need_local_ema:
        # Copy before compile/DDP; resume restores raw and EMA independently below.
        ema_model = copy.deepcopy(model)
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    if checkpoint is not None:
        restore_model_weights(checkpoint, model, ema_model, device)

    return model, ema_model, ema_updater


def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    obs_lr: float | None = None,
    obs_weight_decay: float | None = None,
    betas: tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Require each trainable parameter once and retain its name for resume checks."""
    obs_lr = obs_lr if obs_lr is not None else lr
    obs_weight_decay = obs_weight_decay if obs_weight_decay is not None else weight_decay
    groups = model.optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
    groups = [group for group in groups if group["params"]]
    if any("clip_group" in group for group in groups) and not all(
        isinstance(group.get("clip_group"), str) and group["clip_group"] for group in groups
    ):
        raise ValueError("Every optimizer group must name its clipping group")
    names = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    param_ids = [id(param) for group in groups for param in group["params"]]
    assigned = set(param_ids)
    if len(param_ids) != len(assigned):
        raise ValueError("Optimizer groups contain duplicate parameters")
    if assigned - names.keys():
        raise ValueError("Optimizer groups must contain only this policy's trainable parameters")
    missing = names.keys() - assigned
    if missing:
        raise ValueError(
            "Trainable parameters missing from optimizer: "
            f"{sorted(names[key] for key in missing)}"
        )

    for group in groups:
        group["param_names"] = [names[id(p)] for p in group["params"]]
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )


def build_optimizer_and_scheduler(cfg, model, *, verbose=True):
    """Build initial optimizer/scheduler; resume loads their complete saved states."""
    optimizer = build_optimizer(model, **cfg.optimizer)
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


def compile_training_models(cfg, components):
    if cfg.training.get("use_compile", False):
        mode = cfg.training.get("compile_mode", "reduce-overhead")
        for model in (components.model, components.ema_model):
            if model is not None:
                model.compile_backbone(mode=mode)


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
    effective_config: Any

    def close(self):
        callbacks = [lambda: close_data_loader(self.train_loader)]
        if self.workspace is not None:
            callbacks.append(self.workspace.close)
        close_resources(*callbacks)


def assemble_effective_config(cfg, data_description, model, *, checkpoint=None):
    effective = copy.deepcopy(cfg)
    with open_dict(effective):
        effective.update(data_description)
        if "real_runtime" not in data_description:
            effective.pop("real_runtime", None)
        effective.policy_metadata = model.inference_metadata
        if checkpoint is None:
            bind_defaults(effective.agent, type(model))
            effective.agent.update(model.effective_config)
            bind_defaults(effective.optimizer, build_optimizer)
            if cfg.training.use_ema:
                bind_defaults(effective.ema, hydra.utils.get_class(cfg.ema._target_))
        if effective.training.lr_scheduler == "cosine_min_lr":
            effective.training.lr_min_ratio = effective.training.get("lr_min_ratio", 0.1)
    normalization_dims_from_config(effective)
    OmegaConf.resolve(effective)
    OmegaConf.set_readonly(effective, True)
    return effective


def build_training_components(
    cfg, device, dataset, normalizer, *, data_description, rank=0, world_size=1, checkpoint=None
):
    """Build one rank's components before compile or DDP wrapping."""
    train_loader = build_train_loader(cfg, dataset, rank=rank, world_size=world_size)
    workspace = None
    try:
        model, ema_model, ema_updater = build_model_and_ema(
            cfg, device, normalizer, data_description, rank=rank, checkpoint=checkpoint
        )

        # DDP initialization is shared; augmentation uses a separate seed on each rank.
        if world_size > 1:
            set_seed(cfg.training.seed + rank)
        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=rank == 0)

        cfg = assemble_effective_config(cfg, data_description, model, checkpoint=checkpoint)
        if rank == 0:
            print_training_recipe(
                cfg,
                world_size=world_size,
                batches_per_epoch=len(train_loader),
                samples_per_rank=train_loader.sampler.num_samples,
            )
            workspace = hydra.utils.instantiate(cfg.workspace)

        if checkpoint is not None:
            archive_error = None
            if rank == 0:
                try:
                    archive_resume_milestones(cfg.resume_from, workspace.output_dir, checkpoint)
                except Exception as error:
                    archive_error = error
            if world_size > 1:
                import torch.distributed as dist

                status = [str(archive_error) if archive_error is not None else None]
                dist.broadcast_object_list(status, src=0)
                if rank != 0 and status[0] is not None:
                    raise RuntimeError(f"Rank 0 milestone archive failed: {status[0]}")
            if archive_error is not None:
                raise archive_error

    except BaseException:
        callbacks = [lambda: close_data_loader(train_loader)]
        if workspace is not None:
            callbacks.append(workspace.close)
        close_resources(*callbacks)
        raise

    return TrainingComponents(
        device=device,
        model=model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        workspace=workspace,
        effective_config=cfg,
    )


def build_trainer(components):
    cfg = components.effective_config
    trainer = Trainer(
        components,
        train_loop_cfg=TrainLoopConfig(**OmegaConf.to_container(cfg.training.loop, resolve=True)),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        use_bfloat16=cfg.training.get("use_bfloat16", False),
        effective_config=cfg,
    )
    if components.ema_updater is not None and cfg.get("resume_from") is not None:
        # DDP may have synchronized raw weights after restoring raw/EMA independently.
        averaged = dict(components.ema_model.named_parameters())
        for name, parameter in trainer.raw_model.named_parameters():
            if not parameter.requires_grad and (
                parameter.dtype != averaged[name].dtype
                or not torch.equal(parameter, averaged[name])
            ):
                raise ValueError(f"Raw/EMA frozen parameter mismatch: {name}")
    return trainer
