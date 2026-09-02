"""Per-policy import smoke test (C4: eager barrels cleared → direct imports).

After clearing the eager ``__init__.py`` barrels, every agent and obs-encoder
must still import via its direct module path (the Hydra ``_target_`` contract),
and the barrels themselves must stay lightweight (no eager re-export).
"""

from __future__ import annotations

import importlib
import unittest

AGENT_MODULES = [
    "dexmani_policy.agents.core.action_flow",
    "dexmani_policy.agents.core.dp",
    "dexmani_policy.agents.core.dp3",
    "dexmani_policy.agents.core.dqrise",
    "dexmani_policy.agents.core.maniflow",
    "dexmani_policy.agents.core.moe",
    "dexmani_policy.agents.core.multi_task",
    "dexmani_policy.agents.core.r3d",
    "dexmani_policy.agents.core.sat",
]

OBS_ENCODER_MODULES = [
    "dexmani_policy.agents.obs_encoder.rgb.r3m",
    "dexmani_policy.agents.obs_encoder.rgb.resnet",
    "dexmani_policy.agents.obs_encoder.pointcloud.pointnet",
    "dexmani_policy.agents.obs_encoder.pointcloud.pointnet_dense",
    "dexmani_policy.agents.obs_encoder.pointcloud.pointnext",
    "dexmani_policy.agents.obs_encoder.pointcloud.pointnext_tokenizer",
    "dexmani_policy.agents.obs_encoder.pointcloud.registry",
]


class TestPerPolicyImportSmoke(unittest.TestCase):
    def test_all_agent_modules_import(self):
        for mod in AGENT_MODULES:
            with self.subTest(module=mod):
                importlib.import_module(mod)

    def test_obs_encoder_direct_imports(self):
        for mod in OBS_ENCODER_MODULES:
            with self.subTest(module=mod):
                importlib.import_module(mod)

    def test_barrels_are_lightweight(self):
        """Cleared barrels must not eagerly re-export their concrete names."""
        cases = [
            ("dexmani_policy.agents.core", ["DP3Agent", "ActionFlowAgent"]),
            ("dexmani_policy.agents.obs_encoder.rgb", ["R3M", "ResNet"]),
            ("dexmani_policy.agents.obs_encoder.pointcloud", ["PointNet", "PointNextEncoder"]),
            ("dexmani_policy.agents.obs_encoder.text", ["CLIPTextEncoder", "T5TextEncoder"]),
            ("dexmani_policy.agents.obs_encoder.plugins", ["MoE", "TokenCompressor"]),
        ]
        for barrel, names in cases:
            mod = importlib.import_module(barrel)
            for name in names:
                self.assertFalse(hasattr(mod, name), f"{barrel} must not re-export {name}")


if __name__ == "__main__":
    unittest.main()
