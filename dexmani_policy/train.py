"""Train a policy on one device, optionally restoring the full training state."""

import os
from contextlib import closing

import hydra

from dexmani_policy.common.config import register_resolvers, set_project_root
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.common.config import prepare_training
from dexmani_policy.training.common.data import build_dataset_and_normalizer
from dexmani_policy.training.trainer import PolicyTrainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg, checkpoint = prepare_training(cfg)
    set_seed(cfg.training.seed)
    dataset, normalizer, data_description = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint)
    with closing(PolicyTrainer(
        cfg, dataset, normalizer, data_description, checkpoint=checkpoint
    )) as trainer:
        del checkpoint
        trainer.compile_models()
        trainer.train(max_updates=cfg.get("max_updates"))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    set_project_root()
    _train()


if __name__ == "__main__":
    main()
