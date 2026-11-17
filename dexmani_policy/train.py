"""Train a policy on one device, optionally restoring the full training state."""

import os

import hydra

from dexmani_policy.common.config import set_project_root
from dexmani_policy.training.config import assemble_policy_config, register_resolvers
from dexmani_policy.training.trainer import run_training


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    set_project_root()
    _train()


if __name__ == "__main__":
    main()
