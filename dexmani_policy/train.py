"""Train a policy on one device, optionally restoring the full training state."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra

from dexmani_policy.common.config import register_resolvers, set_project_root
from dexmani_policy.common.runtime import set_seed
from dexmani_policy.training.builder import (
    build_train_components,
    build_trainer,
    compile_models,
    prepare_training,
)
from dexmani_policy.training.resume import restore_training_state

register_resolvers()
set_project_root()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    cfg, checkpoint = prepare_training(cfg)
    set_seed(cfg.training.seed)
    components = build_train_components(cfg, resume_checkpoint=checkpoint)
    try:
        trainer = build_trainer(cfg, components)
        resume_state = None
        if checkpoint is not None:
            resume_state = restore_training_state(
                checkpoint,
                resume_contract=trainer.resume_contract,
                effective_config=cfg,
                ema_updater=components.ema_updater,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
                device=components.device,
            )
        del checkpoint

        components.workspace.save_hydra_config(cfg)
        if cfg.training.get("use_compile", False):
            compile_models(
                components.model,
                components.ema_model,
                mode=cfg.training.get("compile_mode", "reduce-overhead"),
            )
        trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))
    finally:
        components.workspace.close()


if __name__ == "__main__":
    main()
