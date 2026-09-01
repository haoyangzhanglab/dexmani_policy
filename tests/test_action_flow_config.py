"""Regression tests for ActionFlow's joint-state/action-space contract."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dexmani_policy.common.config import register_resolvers


CONFIG_DIR = Path(__file__).resolve().parents[1] / "dexmani_policy" / "configs"


def _compose_action_flow(config_name: str, action_key: str):
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass

    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=[f"action_key={action_key}"])
        cfg.workspace.output_dir = "/tmp/action_flow_config_test"
        OmegaConf.resolve(cfg)
    return cfg


class ActionFlowConfigTest(unittest.TestCase):
    def test_joint_and_ee_configs_preserve_19d_joint_state(self):
        for config_name in ("action_flow", "ddp/action_flow"):
            for action_key, expected_action_dim in (("action", 19), ("action_ee", 21)):
                with self.subTest(config_name=config_name, action_key=action_key):
                    cfg = _compose_action_flow(config_name, action_key)
                    self.assertEqual(cfg.action_dim, expected_action_dim)
                    self.assertEqual(cfg.agent.action_dim, expected_action_dim)
                    self.assertEqual(cfg.agent.state_dim, 19)
