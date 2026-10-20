import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
from omegaconf import open_dict

from dexmani_policy.training.builder import (
    build_train_components,
    build_trainer,
    compile_models,
)
from dexmani_policy.training.common.checkpoint import CheckpointStore
from dexmani_policy.training.resume import resolve_training_config, restore_training_state
from dexmani_policy.training.validation import validate_config
from dexmani_policy.training.workspace import prepare_output_dir
from dexmani_policy.utils.config import register_resolvers, resolve_resume_source, set_project_root
from dexmani_policy.utils.runtime import set_seed

register_resolvers()
set_project_root()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    source = resolve_resume_source(cfg.get("resume_from"))
    checkpoint = CheckpointStore(Path(source).parent).load(source) if source is not None else None
    cfg = resolve_training_config(cfg, resume_checkpoint=checkpoint)
    validate_config(cfg)
    with open_dict(cfg):
        cfg.resume_from = source
    prepare_output_dir(cfg.workspace.output_dir)

    set_seed(cfg.training.seed)
    comp = build_train_components(cfg, resume_checkpoint=checkpoint)
    try:
        trainer = build_trainer(cfg, comp)
        resume_state = None
        if checkpoint is not None:
            resume_state = restore_training_state(
                checkpoint,
                resume_contract=trainer.resume_contract,
                effective_config=cfg,
                model=comp.model,
                ema_updater=comp.ema_updater,
                optimizer=comp.optimizer,
                scheduler=comp.scheduler,
                device=comp.device,
            )
        del checkpoint
        comp.workspace.save_hydra_config(cfg)
        if cfg.training.get("use_compile", False):
            compile_models(
                comp.model,
                comp.ema_model,
                mode=cfg.training.get("compile_mode", "reduce-overhead"),
            )
        trainer.train(resume_state=resume_state, max_updates=cfg.get("max_updates"))
    finally:
        comp.workspace.close()


if __name__ == "__main__":
    main()
