import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)


import hydra

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.common.resolver import register_resolvers
from dexmani_policy.train import build_train_components, validate_config
from dexmani_policy.training.trainer import Trainer

register_resolvers()


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
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()
