"""Shared loader, resume contract and state restoration for both entry points."""

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.common.checkpoint_io import (
    build_agent_contract, validate_ema_resume_state, validate_resume_contract,
)
from dexmani_policy.common.pytorch_util import fix_state_dict, optimizer_to, set_rng_state, worker_init_fn
from dexmani_policy.datasets.resumable_sampler import ResumableDistributedSampler


def loader_options(cfg):
    options = OmegaConf.to_container(cfg.dataloader, resolve=True)
    if options.get("num_workers", 0) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    return options


def build_train_loader(cfg, dataset, *, rank=0, world_size=1):
    options = loader_options(cfg)
    sampler = ResumableDistributedSampler(
        dataset, batch_size=options["batch_size"], num_replicas=world_size,
        rank=rank, shuffle=options.pop("shuffle", False), seed=cfg.training.seed,
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(cfg.training.seed + rank)
    return DataLoader(dataset, sampler=sampler, generator=generator,
                      worker_init_fn=worker_init_fn, **options)


def build_resume_contract(cfg, model, train_loader, *, world_size=1):
    def plain(section):
        return OmegaConf.to_container(section, resolve=True)

    training = plain(cfg.training)
    for key in ("device", "gpu_ids", "num_gpus", "use_compile", "compile_mode", "fast_grad_finite_check"):
        training.pop(key, None)
    training["loop"].pop("log_interval_steps", None)
    training["loop"].setdefault("gradient_accumulation_steps", 1)
    training.setdefault("lr_min_ratio", 0.1)
    training["num_training_steps"] = training["loop"]["total_train_steps"]
    return {
        "agent": build_agent_contract(model),
        "agent_config": plain(cfg.agent),
        "dataset": plain(cfg.dataset),
        "loader": {key: loader_options(cfg).get(key, False)
                   for key in ("batch_size", "shuffle", "drop_last")},
        "dataset_length": len(train_loader.dataset),
        "batches_per_epoch": len(train_loader),
        "world_size": world_size,
        "optimizer": plain(cfg.optimizer),
        "ema": plain(cfg.ema),
        "training": training,
    }


def restore_training_state(checkpoint, *, resume_contract, model, ema_model,
                           ema_updater, optimizer, scheduler, device, rank=0):
    """Restore before compile/DDP; return the next unconsumed batch cursor."""
    validate_resume_contract(checkpoint.resume_contract, resume_contract)
    validate_ema_resume_state(checkpoint, require_ema=ema_model is not None)
    world_size = resume_contract["world_size"]
    if len(checkpoint.rng_states) != world_size or not 0 <= rank < world_size:
        raise ValueError("Checkpoint RNG rank count does not match world_size")
    batches = resume_contract["batches_per_epoch"]
    accum = resume_contract["training"]["loop"]["gradient_accumulation_steps"]
    cursor = checkpoint.next_micro_step
    if not 0 <= cursor < batches or cursor % accum:
        raise ValueError("Checkpoint next_micro_step is not a normalized accumulation boundary")
    model.load_state_dict(fix_state_dict(checkpoint.model_state, False), strict=True)
    if ema_model is not None:
        ema_model.load_state_dict(fix_state_dict(checkpoint.ema_model_state, False), strict=True)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(checkpoint.scheduler_state)
    if ema_updater is not None:
        ema_updater.optimization_step = checkpoint.ema_updater_step
        if checkpoint.ema_decay is not None:
            ema_updater.decay = float(checkpoint.ema_decay)
    set_rng_state(checkpoint.rng_states[rank])
    return checkpoint.global_step, checkpoint.epoch, cursor


def validate_gpu_ids(num_gpus, gpu_ids, available_gpus):
    if type(num_gpus) is not int or num_gpus <= 1:
        raise ValueError("DDP num_gpus must be an integer > 1")
    ids = list(range(num_gpus)) if gpu_ids is None else list(gpu_ids)
    if len(ids) != num_gpus:
        raise ValueError("Length of gpu_ids must match num_gpus")
    if any(type(i) is not int or i < 0 or i >= available_gpus for i in ids):
        raise ValueError(f"gpu_ids must contain integers in [0, {available_gpus})")
    if len(set(ids)) != len(ids):
        raise ValueError("gpu_ids must not contain duplicates")
    return ids
