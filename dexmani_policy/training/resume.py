"""Restore training from the saved recipe, loader geometry and state dictionaries."""

import warnings

from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from dexmani_policy.training.common.checkpoint import TrainCheckpoint, unwrap_state_dict
from dexmani_policy.utils.config import resolve_input_recipe, validate_source_recipe
from dexmani_policy.utils.hand_artifact import restore_artifact_metadata
from dexmani_policy.utils.random import set_rng_state


def build_resume_contract(train_loader):
    """Capture only loader geometry that is absent from the saved config."""
    return {
        "dataset_length": len(train_loader.dataset),
        "batches_per_epoch": len(train_loader),
        "world_size": train_loader.sampler.num_replicas,
    }


def restore_training_state(
    checkpoint,
    *,
    resume_contract,
    effective_config,
    model,
    ema_updater,
    optimizer,
    scheduler,
    device,
    rank=0,
):
    """Restore optimizer/scheduler/EMA-updater/RNG after model weights and optimizer construction."""
    validate_source_recipe(checkpoint.effective_config, effective_config)
    for key, value in resume_contract.items():
        if checkpoint.resume_contract[key] != value:
            raise ValueError(f"Resume DataLoader {key} changed")
    identity = checkpoint.effective_config["data_identity"]
    for name, source in identity.get("tasks", {"dataset": identity}).items():
        if source["revision"] is None:
            warnings.warn(f"{name}: 数据身份未验证 (dataset has no data_revision)", stacklevel=2)
    world_size = resume_contract["world_size"]
    if len(checkpoint.rng_states) != world_size or not 0 <= rank < world_size:
        raise ValueError("Checkpoint RNG rank count does not match world_size")
    batches = resume_contract["batches_per_epoch"]
    loop = checkpoint.effective_config["training"]["loop"]
    accum = loop["gradient_accumulation_steps"]
    cursor = checkpoint.next_micro_step
    if not 0 <= cursor < batches or cursor % accum:
        raise ValueError("Checkpoint next_micro_step is not a normalized accumulation boundary")
    total = loop["total_train_steps"]
    updates_per_epoch = (batches + accum - 1) // accum
    if checkpoint.global_step != checkpoint.epoch * updates_per_epoch + cursor // accum:
        raise ValueError("Checkpoint step/epoch/cursor disagree with full loader geometry")
    if checkpoint.global_step > total:
        raise ValueError("Checkpoint global_step exceeds the saved total plan")
    names = {id(p): n for n, p in model.named_parameters()}
    if any(id(p) not in names for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("Optimizer must be constructed from the restored model parameters")
    for current, saved in zip(optimizer.param_groups, checkpoint.optimizer_state["param_groups"]):
        if saved.get("param_names") != [names[id(p)] for p in current["params"]]:
            raise ValueError("Optimizer parameter name/order mismatch")
    optimizer.load_state_dict(checkpoint.optimizer_state)
    scheduler.load_state_dict(checkpoint.scheduler_state)
    if ema_updater is not None:
        ema_updater.optimization_step = checkpoint.ema_updater_step
        if checkpoint.ema_decay is not None:
            ema_updater.decay = float(checkpoint.ema_decay)
    set_rng_state(checkpoint.rng_states[rank], device=device)
    return checkpoint.global_step, checkpoint.epoch, cursor


def restore_model_weights(checkpoint, model, ema_model, device):
    validate_ema_resume_state(checkpoint, require_ema=ema_model is not None)
    state = unwrap_state_dict(checkpoint.model_state)
    model.load_state_dict(state, strict=True)
    model.validate_normalizer()
    restore_artifact_metadata(model, checkpoint.effective_config)
    # Normalizer restoration creates parameters on CPU; move them before DDP wrapping.
    model.to(device)
    if ema_model is not None:
        ema_model.load_state_dict(unwrap_state_dict(checkpoint.ema_model_state), strict=True)
        ema_model.validate_normalizer()
        restore_artifact_metadata(ema_model, checkpoint.effective_config)
        ema_model.to(device)


def validate_ema_resume_state(checkpoint: TrainCheckpoint, *, require_ema: bool) -> None:
    """Require the complete EMA state needed to resume EMA training."""
    if not require_ema:
        return
    if checkpoint.ema_model_state is None:
        raise RuntimeError("Resume checkpoint is missing required ema_model_state")

    step = checkpoint.ema_updater_step
    if type(step) is not int or step < 0:
        raise RuntimeError("Resume checkpoint ema_updater_step must be an int (not bool) >= 0")


def resolve_training_config(cfg, *, overrides=None, resume_checkpoint=None):
    """Resolve the input recipe and bind this invocation's output directory."""
    if overrides is None:
        overrides = HydraConfig.get().overrides.task if cfg.get("resume_from") is not None else []
    result = resolve_input_recipe(cfg, overrides=overrides, resume_checkpoint=resume_checkpoint)
    if result.get("resume_from") is not None:
        with open_dict(result):
            # The destination is this invocation's new run, never the source output.
            result.workspace.output_dir = cfg.workspace.output_dir
    return result
