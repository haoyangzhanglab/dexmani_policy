import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


import hydra
import warnings
from omegaconf import OmegaConf

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.train import build_train_components
from dexmani_policy.training.multi_task_trainer import MultiTaskTrainer

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    set_seed(cfg.training.seed)
    components = build_train_components(cfg)
    components.workspace.save_hydra_config(cfg)

    trainer = MultiTaskTrainer(
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
