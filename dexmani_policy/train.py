import os
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import hydra
import torch
from torch.utils.data import DataLoader

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    validate_config,
    compute_num_training_steps,
)
from dexmani_policy.training.trainer import Trainer

register_resolvers()

@dataclass
class TrainingComponents:
    """Assembled training pipeline components (single-GPU).

    Returned by :func:`build_train_components` and consumed by :func:`main`.
    """
    device: torch.device
    model: torch.nn.Module
    ema_model: Optional[torch.nn.Module]
    ema_updater: Optional[Any]
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    workspace: Any
    env_runner: Optional[Any]
    num_training_steps: int

def build_train_components(cfg):
    """Assemble all training components for single-GPU training."""
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
    num_training_steps = compute_num_training_steps(cfg, batches_per_epoch)

    workspace = hydra.utils.instantiate(cfg.workspace)

    env_runner = None
    if cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    return TrainingComponents(
        device=device,
        model=model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        workspace=workspace,
        env_runner=env_runner,
        num_training_steps=num_training_steps,
    )

@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    validate_config(cfg)

    set_seed(cfg.training.seed)
    comp = build_train_components(cfg)
    comp.workspace.save_hydra_config(cfg)

    trainer = Trainer(
        device=comp.device,
        model=comp.model,
        ema_model=comp.ema_model,
        ema_updater=comp.ema_updater,
        optimizer=comp.optimizer,
        scheduler=comp.scheduler,
        train_loader=comp.train_loader,
        val_loader=comp.val_loader,
        env_runner=comp.env_runner,
        workspace=comp.workspace,
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
        max_grad_norm=cfg.training.get('max_grad_norm', 1.0),
        use_bfloat16=cfg.training.get('use_bfloat16', False),
        use_compile=cfg.training.get('use_compile', False),
        num_training_steps=comp.num_training_steps,
    )
    trainer.train(resume_tag="latest")

if __name__ == "__main__":
    main()
