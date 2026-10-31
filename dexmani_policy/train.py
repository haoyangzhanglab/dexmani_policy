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
    cfg = prepare_training(assemble_policy_config(cfg))
    run_training(cfg, max_updates=cfg.get("max_updates"))


def run_training(cfg, *, checkpoint=None, max_updates=None):
    set_seed(cfg.training.seed)
    dataset, data_description = build_dataset(cfg)
    with closing(PolicyTrainer(cfg, dataset, data_description, checkpoint=checkpoint)) as trainer:
        trainer.compile_models()
        trainer.train(max_updates=max_updates)
        return trainer.global_step


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    set_project_root()
    _train()


if __name__ == "__main__":
    main()
