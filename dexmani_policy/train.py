"""Train a policy on one device, optionally restoring the full training state."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hydra
import torch

from dexmani_policy.common.config import register_resolvers, set_project_root
from dexmani_policy.common.runtime import set_seed
from dexmani_policy.training.builder import (
    build_training_components,
    build_trainer,
    compile_models,
    prepare_training,
)
from dexmani_policy.training.data import build_dataset_and_normalizer
from dexmani_policy.training.resume import restore_training_state

register_resolvers()
set_project_root()


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    cfg, checkpoint = prepare_training(cfg)
    set_seed(cfg.training.seed)
    if cfg.training.get("num_gpus", 1) != 1:
        raise ValueError("Single-device training requires num_gpus=1; use train_ddp.py for DDP")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires GPU.")
    dataset, normalizer = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    components = build_training_components(
        cfg, torch.device(cfg.training.device), dataset, normalizer, checkpoint=checkpoint
    )
    try:
        trainer = build_trainer(cfg, components)
        resume_state = None
        if checkpoint is not None:
            resume_state = restore_training_state(checkpoint, components, effective_config=cfg)
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
