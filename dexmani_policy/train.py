"""Train a policy on one device, optionally restoring the full training state."""

import os
from contextlib import closing

import hydra

from dexmani_policy.common.config import set_project_root
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.common.config import (
    assemble_policy_config, prepare_training, register_resolvers,
)
from dexmani_policy.training.common.data import build_dataset
from dexmani_policy.training.trainer import PolicyTrainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def build_trainer(cfg, *, checkpoint=None):
    if checkpoint is None:
        cfg = prepare_training(cfg)
    set_seed(cfg.training.seed)
    dataset, data_description = build_dataset(cfg)
    return PolicyTrainer(cfg, dataset, data_description, checkpoint=checkpoint)


def run_trainer(trainer, *, max_updates=None):
    with closing(trainer):
        trainer.compile_models()
        trainer.train(max_updates=max_updates)
        return trainer.global_step


def run_training(cfg, *, max_updates=None):
    return run_trainer(build_trainer(cfg), max_updates=max_updates)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    set_project_root()
    _train()


if __name__ == "__main__":
    main()
