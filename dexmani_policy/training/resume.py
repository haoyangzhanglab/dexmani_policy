"""Restore training from the saved recipe, loader geometry and state dictionaries."""

from dexmani_policy.common.checkpoint import load_policy_state
from dexmani_policy.common.config import validate_source_recipe
from dexmani_policy.common.runtime import set_rng_state


def build_resume_contract(train_loader):
    """Capture only loader geometry that is absent from the saved config."""
    return {
        "dataset_length": len(train_loader.dataset),
        "batches_per_epoch": len(train_loader),
        "world_size": train_loader.sampler.num_replicas,
    }


def restore_training_state(checkpoint, components, *, effective_config):
    """Restore optimizer, scheduler, EMA updater and RNG after model construction."""
    resume_contract = build_resume_contract(components.train_loader)
    rank = components.train_loader.sampler.rank
    optimizer = components.optimizer
    scheduler = components.scheduler
    ema_updater = components.ema_updater
    validate_source_recipe(checkpoint.effective_config, effective_config)
    for key, value in resume_contract.items():
        if checkpoint.resume_contract[key] != value:
            raise ValueError(f"Resume DataLoader {key} changed")
    world_size = resume_contract["world_size"]
    if len(checkpoint.rng_states) != world_size:
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
    for current, saved in zip(optimizer.param_groups, checkpoint.optimizer_state["param_groups"]):
        if saved["param_names"] != current["param_names"]:
            raise ValueError("Optimizer parameter name/order mismatch")
    optimizer.load_state_dict(checkpoint.optimizer_state)
    scheduler.load_state_dict(checkpoint.scheduler_state)
    if ema_updater is not None:
        ema_updater.optimization_step = checkpoint.ema_updater_step
    set_rng_state(checkpoint.rng_states[rank], device=components.device)
    return checkpoint.global_step, checkpoint.epoch, cursor


def restore_model_weights(checkpoint, model, ema_model, device):
    if ema_model is not None:
        if checkpoint.ema_model_state is None:
            raise RuntimeError("Resume checkpoint is missing required ema_model_state")
        if type(checkpoint.ema_updater_step) is not int or checkpoint.ema_updater_step < 0:
            raise RuntimeError("Resume checkpoint ema_updater_step must be an int >= 0")
    load_policy_state(model, checkpoint.model_state)
    model.validate_normalizer()
    # Normalizer restoration creates parameters on CPU; move them before DDP wrapping.
    model.to(device)
    if ema_model is not None:
        load_policy_state(ema_model, checkpoint.ema_model_state)
        ema_model.validate_normalizer()
        ema_model.to(device)
