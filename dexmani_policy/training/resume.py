"""Shared loader, resume contract and state restoration for both entry points."""

from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.datasets.resumable_sampler import ResumableDistributedSampler
from dexmani_policy.training.checkpoint import TrainCheckpoint, fix_state_dict
from dexmani_policy.utils.random import set_rng_state, worker_init_fn


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
    a = cfg.training.get("loop", {}).get("gradient_accumulation_steps", 1)
    if a > 1 and not options.get("drop_last", False) and n % b and ((n + b - 1) // b) % a != 1:
        raise ValueError(
            f"Mixed micro-batch sizes in accumulation group: N={n}, B={b}, A={a}; "
            "use drop_last=True or gradient_accumulation_steps=1"
        )
    generator = torch.Generator().manual_seed(cfg.training.seed + rank)
    return DataLoader(
        dataset,
        sampler=sampler,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **options,
    )


def build_resume_contract(cfg, model, train_loader, *, world_size=1):
    def plain(section):
        return OmegaConf.to_container(section, resolve=True)

    training = plain(cfg.training)
    for key in (
        "device",
        "gpu_ids",
        "num_gpus",
        "use_compile",
        "compile_mode",
        "fast_grad_finite_check",  # Ignored historical config field.
    ):
        training.pop(key, None)
    training["loop"].pop("log_interval_steps", None)
    training["loop"].setdefault("gradient_accumulation_steps", 1)
    training.setdefault("lr_min_ratio", 0.1)
    return {
        "agent": build_agent_contract(model),
        "agent_config": plain(cfg.agent),
        "dataset": plain(cfg.dataset),
        "loader": {
            key: loader_options(cfg).get(key, False)
            for key in ("batch_size", "shuffle", "drop_last")
        },
        "data_identity": plain(cfg.data_identity) if "data_identity" in cfg else {"revision": None},
        "dataset_length": len(train_loader.dataset),
        "batches_per_epoch": len(train_loader),
        "world_size": world_size,
        "optimizer": plain(cfg.optimizer),
        "ema": plain(cfg.ema),
        "training": training,
    }


def restore_training_state(
    checkpoint,
    *,
    resume_contract,
    model,
    ema_model,
    ema_updater,
    optimizer,
    scheduler,
    device,
    rank=0,
    source_config=None,
):
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
        raise ValueError(
            "Checkpoint next_micro_step is not a normalized accumulation boundary"
        )
    model.load_state_dict(fix_state_dict(checkpoint.model_state, False), strict=True)
    # Normalizers reconstruct their ParameterDict from checkpoint tensors.
    # A CPU-loaded checkpoint must not leave these parameters on CPU before
    # DDP wraps the restored model or broadcasts normalization state.
    model.to(device)
    if ema_model is not None:
        ema_model.load_state_dict(
            fix_state_dict(checkpoint.ema_model_state, False), strict=True
        )
        ema_model.to(device)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(checkpoint.scheduler_state)
    if ema_updater is not None:
        ema_updater.optimization_step = checkpoint.ema_updater_step
        if checkpoint.ema_decay is not None:
            ema_updater.decay = float(checkpoint.ema_decay)
    set_rng_state(checkpoint.rng_states[rank], device=device, source_config=source_config,
                  rank=rank, world_size=world_size)
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


def optimizer_to(
    optimizer: torch.optim.Optimizer, device: torch.device | str
) -> torch.optim.Optimizer:
    """Move all tensor state in an optimizer to the given device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


NORMALIZATION_CONTRACT_VERSION = 1


def make_normalization_contract(spec) -> Dict[str, Any]:
    """Build the versioned semantic normalization contract (types only, no numbers)."""
    return {"version": NORMALIZATION_CONTRACT_VERSION, "fields": dict(spec)}


def build_agent_contract(model) -> Dict[str, Any]:
    """Training-only facts for exact resume."""
    return {
        "n_obs_steps": model.n_obs_steps,
        "n_action_steps": model.n_action_steps,
        "action_dim": model.action_dim,
        "horizon": model.horizon,
        "action_key": model.action_key,
        "tcp_dim": getattr(model, "tcp_dim", None),
        "hand_dim": getattr(model, "hand_dim", None),
        "control_action_dim": model.control_action_dim,
        "use_aux_ee": bool(getattr(model, "use_aux_ee", False)),
        "normalization": make_normalization_contract(
            getattr(model, "normalization_spec", {})
        ),
    }


def validate_resume_contract(saved, current) -> None:
    """Report all missing, extra and changed values, including nested keys."""
    import copy
    from dexmani_policy.agents.normalization import uses_diffusion_config
    saved, current = copy.deepcopy(saved), copy.deepcopy(current)
    for contract in (saved, current):
        agent_cfg = contract.get("agent_config", {})
        if uses_diffusion_config(agent_cfg):
            agent_cfg.setdefault("clip_sample", True)
    validate_data_identity(saved.pop("data_identity", None), current.pop("data_identity", None))
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}"
                if key not in left:
                    differences.append(f"{child}: missing in checkpoint")
                elif key not in right:
                    differences.append(f"{child}: unexpected checkpoint key")
                else:
                    compare(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(
                    f"{path}: length saved={len(left)}, current={len(right)}"
                )
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{i}]")
        elif type(left) is not type(right) or left != right:
            differences.append(f"{path}: saved={left!r}, current={right!r}")

    compare(saved, current, "resume_contract")
    if differences:
        raise ValueError("Resume contract mismatch:\n" + "\n".join(differences))


def validate_ema_resume_state(
    checkpoint: TrainCheckpoint, *, require_ema: bool
) -> None:
    """Require the complete EMA state needed to resume EMA training."""
    if not require_ema:
        return
    if checkpoint.ema_model_state is None:
        raise RuntimeError("Resume checkpoint is missing required ema_model_state")

    step = checkpoint.ema_updater_step
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError(
            "Resume checkpoint ema_updater_step must be an int (not bool) >= 0"
        )


def load_resume_source_config(checkpoint_path):
    """Historical evidence only; never substitute the destination config."""
    from pathlib import Path
    source = Path(checkpoint_path).resolve().parent.parent / "config.yaml"
    return OmegaConf.to_container(OmegaConf.load(source), resolve=True) if source.is_file() else None


def validate_data_identity(saved, current, path="data_identity"):
    import warnings
    if saved is None:
        warnings.warn(f"{path}: 数据身份未验证 (historical revision unavailable)", stacklevel=2)
        return
    if "tasks" in saved:
        current_tasks = (current or {}).get("tasks", {})
        for task, identity in saved["tasks"].items():
            validate_data_identity(identity, current_tasks.get(task), f"{path}.{task}")
        return
    previous = saved.get("revision")
    actual = (current or {}).get("revision")
    if previous is not None:
        if actual != previous:
            raise ValueError(f"{path}: data_revision changed or lost: saved={previous!r}, current={actual!r}")
    else:
        warnings.warn(f"{path}: 数据身份未验证 (historical revision unavailable)", stacklevel=2)
