import copy
import hydra
import torch
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.trainer import Trainer
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.lr_scheduler import get_scheduler

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


def build_train_components(cfg):
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = dataset.get_normalizer()

    train_loader = DataLoader(dataset, **cfg.dataloader)
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
        except Exception:
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

    return {
        "device": torch.device(cfg.training.device),
        "model": model,
        "ema_model": ema_model,
        "ema_updater": ema_updater,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "workspace": workspace,
        "env_runner": env_runner,
    }



@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    set_seed(cfg.training.seed)
    components = build_train_components(cfg)
    components["workspace"].save_hydra_config(cfg)

    trainer = Trainer(
        device=components["device"],
        model=components["model"],
        ema_model=components["ema_model"],
        ema_updater=components["ema_updater"],
        optimizer=components["optimizer"],
        scheduler=components["scheduler"],
        train_loader=components["train_loader"],
        val_loader=components["val_loader"],
        env_runner=components["env_runner"],
        workspace=components["workspace"],
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()