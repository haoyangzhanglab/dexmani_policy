import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from typing import Any, Optional

import hydra
import torch
from torch.utils.data import DataLoader

from dexmani_policy.utils.config import register_resolvers
from dexmani_policy.utils.path import set_project_root
from dexmani_policy.utils.random import set_seed

ROOT_DIR = set_project_root()
from omegaconf import OmegaConf, open_dict

from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    print_training_recipe,
    validate_config,
)
from dexmani_policy.training.resume import (
    build_resume_contract, build_train_loader, restore_training_state,
)
from dexmani_policy.training.run_identity import claim_run, resolve_resume_source
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig

register_resolvers()


@dataclass
class TrainingComponents:
    """Single-device components shared by training and integration smoke."""

    device: torch.device
    model: torch.nn.Module
    ema_model: Optional[torch.nn.Module]
    ema_updater: Optional[Any]
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: DataLoader
    workspace: Any
    batches_per_epoch: int
    resume_checkpoint: Any = None


def build_train_components(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires GPU.")

    device = torch.device(cfg.training.device)

    from pathlib import Path
    from dexmani_policy.training.checkpoint import CheckpointStore
    source = cfg.get("resume_from")
    checkpoint = CheckpointStore(Path(source).parent).load(source) if source else None
    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)

    train_loader = build_train_loader(cfg, dataset)

    model, ema_model, ema_updater = build_model_and_ema(
        cfg, device, normalizer, checkpoint=checkpoint
    )

    batches_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, batches_per_epoch)
    print_training_recipe(cfg, world_size=1, batches_per_epoch=batches_per_epoch)

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
        batches_per_epoch=batches_per_epoch,
        resume_checkpoint=checkpoint,
    )


def build_trainer(cfg, comp):
    return Trainer(
        device=comp.device,
        model=comp.model,
        ema_model=comp.ema_model,
        ema_updater=comp.ema_updater,
        optimizer=comp.optimizer,
        scheduler=comp.scheduler,
        train_loader=comp.train_loader,
        workspace=comp.workspace,
        train_loop_cfg=TrainLoopConfig(
            **OmegaConf.to_container(cfg.training.loop, resolve=True)
        ),
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        use_bfloat16=cfg.training.get("use_bfloat16", False),
        use_compile=cfg.training.get("use_compile", False),
        compile_mode=cfg.training.get("compile_mode", "reduce-overhead"),
        resume_contract=build_resume_contract(cfg, comp.model, comp.train_loader),
        batches_per_epoch=comp.batches_per_epoch,
    )


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    from dexmani_policy.training.resume import resolve_training_config
    cfg = resolve_training_config(cfg)
    validate_config(cfg)
    with open_dict(cfg):
        cfg.resume_from = resolve_resume_source(cfg.get("resume_from"))
        cfg.workspace.claim_token = claim_run(
            cfg.workspace.output_dir, resume_from=cfg.resume_from
        )

    set_seed(cfg.training.seed)
    comp = build_train_components(cfg)

    trainer = build_trainer(cfg, comp)
    resume_state = None
    if comp.resume_checkpoint is not None:
        resume_state = restore_training_state(
            comp.resume_checkpoint, resume_contract=trainer.resume_contract,
            model=comp.model, ema_model=comp.ema_model, ema_updater=comp.ema_updater,
            optimizer=comp.optimizer, scheduler=comp.scheduler, device=comp.device,
        )
    comp.resume_checkpoint = None
    comp.workspace.save_hydra_config(cfg)
    trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))


if __name__ == "__main__":
    main()
