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

    train_loader = DataLoader(dataset, **cfg.training.dataloader)
    val_loader = DataLoader(
        dataset.get_validation_dataset(),
        **cfg.training.val_dataloader,
    )

    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        try:
            ema_model = copy.deepcopy(model)
        except Exception:
            ema_model = hydra.utils.instantiate(cfg.policy)

        ema_model.set_normalizer(normalizer)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.training.ema, model=ema_model)

    optimizer = model.get_optimizer(**cfg.training.optimizer)

    total_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    env_runner = None
    if cfg.training.eval_every > 0:
        env_runner = hydra.utils.instantiate(cfg.eval.env_runner)

    workspace = TrainWorkspace(
        cfg,
        output_dir=Path(HydraConfig.get().runtime.output_dir),
    )

    return {
        "device": torch.device(cfg.training.device),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "ema_model": ema_model,
        "ema_updater": ema_updater,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "env_runner": env_runner,
        "workspace": workspace,
    }



@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    set_seed(cfg.training.seed)

    components = build_train_components(cfg)
    loss_kwargs_fn = (
        hydra.utils.instantiate(cfg.training.loss_kwargs_fn)
        if cfg.training.get("loss_kwargs_fn") is not None
        else None
    )

    trainer = Trainer(
        device=components["device"],
        model=components["model"],
        optimizer=components["optimizer"],
        scheduler=components["scheduler"],
        train_loader=components["train_loader"],
        val_loader=components["val_loader"],
        workspace=components["workspace"],
        total_epochs=cfg.training.num_epochs,
        val_every=cfg.training.val_every,
        eval_every=cfg.training.eval_every,
        sample_every=cfg.training.sample_every,
        env_runner=components["env_runner"],
        ema_model=components["ema_model"],
        ema_updater=components["ema_updater"],
        log_every_steps=cfg.training.log_every_steps,
        loss_kwargs_fn=loss_kwargs_fn,
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()