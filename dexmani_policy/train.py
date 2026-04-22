import os
import sys
import pathlib

# 设置项目根目录，以便在训练脚本中正确导入模块，并且在运行训练脚本时保持当前工作目录为项目根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


import copy
import hydra
import torch
import warnings
from omegaconf import OmegaConf
from typing import Any, Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.datasets.augmentation import worker_init_fn
from dexmani_policy.training.trainer import Trainer
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.lr_scheduler import get_scheduler

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class TrainComponents:
    device: torch.device
    model: Any
    ema_model: Optional[Any]
    ema_updater: Optional[Any]
    optimizer: Any
    scheduler: Any
    train_loader: DataLoader
    val_loader: DataLoader
    workspace: TrainWorkspace
    env_runner: Optional[Any]


def build_train_components(cfg) -> TrainComponents:
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = dataset.get_normalizer()

    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    val_loader = DataLoader(
        dataset.get_validation_dataset(),
        **cfg.val_dataloader,
    )

    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(model)
        except Exception as e:
            import warnings
            warnings.warn(f"copy.deepcopy(model) failed ({e}), falling back to fresh instantiation. EMA weights will be random until checkpoint is loaded.")
            ema_model = hydra.utils.instantiate(cfg.agent)

        ema_model.load_normalizer_from_dataset(normalizer)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    optimizer = model.configure_optimizer(**cfg.optimizer)

    total_steps = len(train_loader) * cfg.training.loop.num_epochs
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    workspace = hydra.utils.instantiate(cfg.workspace)

    env_runner = None
    if cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    return TrainComponents(
        device=torch.device(cfg.training.device),
        model=model,
        ema_model=ema_model,
        ema_updater=ema_updater,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        workspace=workspace,
        env_runner=env_runner,
    )


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    set_seed(cfg.training.seed)
    components = build_train_components(cfg)
    components.workspace.save_hydra_config(cfg)

    trainer = Trainer(
        device=components.device,
        model=components.model,
        ema_model=components.ema_model,
        ema_updater=components.ema_updater,
        optimizer=components.optimizer,
        scheduler=components.scheduler,
        train_loader=components.train_loader,
        val_loader=components.val_loader,
        env_runner=components.env_runner,
        workspace=components.workspace,
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()