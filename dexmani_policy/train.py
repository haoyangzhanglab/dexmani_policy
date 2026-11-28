"""Start a new policy run on one device with optional weight initialization."""

import os

import hydra
from omegaconf import open_dict

from dexmani_policy.training.config import assemble_policy_config, register_resolvers
from dexmani_policy.training.runtime import lock_run_directory
from dexmani_policy.training.trainer import build_trainer, run_trainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    with open_dict(cfg):
        max_updates = cfg.pop("max_updates", None)
    with lock_run_directory(cfg.workspace.output_dir, create=True):
        trainer = build_trainer(assemble_policy_config(cfg))
        run_trainer(trainer, max_updates=max_updates)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    _train()


if __name__ == "__main__":
    main()
