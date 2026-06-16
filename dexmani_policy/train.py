import os
import pathlib

os.chdir(str(pathlib.Path(__file__).parent.parent))

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from termcolor import cprint

from dexmani_policy.common.config import register_resolvers, validate_action_key_consistency
from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn, print_param_count
from dexmani_policy.training.lr_scheduler import get_scheduler
from dexmani_policy.training.trainer import Trainer

register_resolvers()


def build_dataset_and_normalizer(cfg, rank=None, world_size=None):
    if rank is not None:
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        dataset = hydra.utils.instantiate(dataset_cfg)
    else:
        dataset = hydra.utils.instantiate(cfg.dataset)

    normalizer = dataset.get_normalizer()
    if hasattr(dataset, 'normalizer_mode') and dataset.normalizer_mode == 'per_task':
        raise NotImplementedError(
            "normalizer_mode='per_task' requires per-task normalizer loading, "
            "which is not yet integrated into the standard training entry. "
            "Use normalizer_mode='shared' or call get_normalizer(task_name=...) manually."
        )
    return dataset, normalizer


def build_model_and_ema(cfg, device, normalizer):
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.action_key = cfg.get('action_key', 'action')
    model.to(device)
    print_param_count(model)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        ema_model = hydra.utils.instantiate(cfg.agent)
        ema_model.load_normalizer_from_dataset(normalizer)
        ema_model.action_key = model.action_key
        ema_model.to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    return model, ema_model, ema_updater


def build_scheduler(cfg, optimizer, batches_per_epoch, last_epoch=-1):
    accum_steps = max(1, int(cfg.training.loop.get('gradient_accumulation_steps', 1)))
    steps_per_epoch = -(-batches_per_epoch // accum_steps)
    total_steps = steps_per_epoch * cfg.training.loop.num_epochs
    return get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch,
    )


def build_optimizer_and_scheduler(cfg, model, batches_per_epoch, last_epoch=-1):
    optimizer = model.configure_optimizer(**cfg.optimizer)
    scheduler = build_scheduler(cfg, optimizer, batches_per_epoch, last_epoch)
    return optimizer, scheduler


def _validate_augmentation_consistency(cfg):
    agent_cfg = cfg.agent
    pc_dim = agent_cfg.get('pc_dim')
    if pc_dim is None or pc_dim >= 6:
        return

    aug_cfg = cfg.dataset.get('augmentation_cfg')
    if aug_cfg is None:
        return

    pc_color = aug_cfg.get('pc', {}).get('color')
    pc_color_noise = aug_cfg.get('pc', {}).get('color_noise')
    missing_rgb = (
        f"PC color augmentation requires agent.pc_dim >= 6, got {pc_dim}. "
        f"The encoder only reads the first {pc_dim} channels (XYZ), "
        f"while the augmentation modifies channels 3:6 (RGB). "
        f"Either set agent.pc_dim=6 or remove the augmentation key."
    )
    if pc_color is not None:
        assert pc_dim >= 6, missing_rgb
    if pc_color_noise is not None:
        assert pc_dim >= 6, f"PC color_noise augmentation: {missing_rgb}"


def validate_config(cfg):
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

    if cfg.optimizer.get('obs_lr') is not None:
        assert cfg.optimizer.obs_lr >= 0, "optimizer.obs_lr must be non-negative (0 means freeze)"

    agent_cfg = cfg.agent
    if 'num_experts' in agent_cfg:
        num_experts = agent_cfg.get('num_experts', 0)
        top_k = agent_cfg.get('top_k', 0)
        assert top_k <= num_experts, \
            f"top_k ({top_k}) must be <= num_experts ({num_experts})"

    _validate_augmentation_consistency(cfg)

    validate_action_key_consistency(cfg)

    cprint("Config validation passed", "green")


def build_train_components(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires GPU.")

    device = torch.device(cfg.training.device)

    dataset, normalizer = build_dataset_and_normalizer(cfg)

    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    val_dataset = dataset.get_validation_dataset()
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            worker_init_fn=worker_init_fn,
            **cfg.val_dataloader,
        )

    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)

    batches_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, batches_per_epoch)
    accum_steps = max(1, int(cfg.training.loop.get('gradient_accumulation_steps', 1)))
    num_training_steps = -(-batches_per_epoch // accum_steps) * cfg.training.loop.num_epochs

    workspace = hydra.utils.instantiate(cfg.workspace)

    env_runner = None
    if cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    return {
        'device': device,
        'model': model,
        'ema_model': ema_model,
        'ema_updater': ema_updater,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'workspace': workspace,
        'env_runner': env_runner,
        'num_training_steps': num_training_steps,
    }


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    validate_config(cfg)

    set_seed(cfg.training.seed)
    components = build_train_components(cfg)
    components['workspace'].save_hydra_config(cfg)

    trainer = Trainer(
        device=components['device'],
        model=components['model'],
        ema_model=components['ema_model'],
        ema_updater=components['ema_updater'],
        optimizer=components['optimizer'],
        scheduler=components['scheduler'],
        train_loader=components['train_loader'],
        val_loader=components['val_loader'],
        env_runner=components['env_runner'],
        workspace=components['workspace'],
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
        max_grad_norm=cfg.training.get('max_grad_norm', 1.0),
        use_bfloat16=cfg.training.get('use_bfloat16', False),
        use_compile=cfg.training.get('use_compile', False),
        num_training_steps=components['num_training_steps'],
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()