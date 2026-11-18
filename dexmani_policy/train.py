"""Start a new policy run on one device with optional weight initialization."""

import os

import hydra

from dexmani_policy.training.config import assemble_policy_config, register_resolvers
from dexmani_policy.training.trainer import run_training


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    _train()


if __name__ == "__main__":
    main()
