"""Config-only validation for the normalization contract across real Policy configs.

Uses the same Hydra compose path as ``smoke_test.py --config-only`` (no GPU/data
required) to prove: all 7 default base configs pass ``validate_config``, and the
documented invalid combinations fail fast at config-validation time.
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.training.build_utils import validate_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "dexmani_policy", "configs")

register_resolvers()


def _compose(config_name, overrides=None):
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        cfg.workspace.output_dir = "/tmp/smoke_test_output"
        OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    "config_name",
    ["dp", "dp3", "dqrise", "maniflow", "r3d", "sat", "multitask_dit"],
)
def test_default_configs_validate(config_name):
    cfg = _compose(config_name)
    validate_config(cfg)  # must not raise


def test_invalid_configs_fail_fast():
    with pytest.raises(ValueError, match="action"):
        validate_config(_compose("dp", ["normalization.action=identity"]))

    with pytest.raises(ValueError, match="rgb"):
        validate_config(_compose("dp", ["normalization.rgb=limits"]))

    with pytest.raises(ValueError, match="pointnext"):
        validate_config(
            _compose(
                "dp3",
                ["agent.encoder_type=pointnext", "normalization.point_cloud=limits"],
            )
        )

    with pytest.raises(ValueError, match="pointnext"):
        validate_config(
            _compose(
                "dqrise",
                ["agent.encoder_type=pointnext", "normalization.point_cloud=limits"],
            )
        )


def test_pointnext_identity_configs_validate():
    # The same PointNeXT override is legal once point_cloud stays identity.
    validate_config(
        _compose(
            "dp3", ["agent.encoder_type=pointnext", "normalization.point_cloud=identity"]
        )
    )
    validate_config(
        _compose(
            "dqrise",
            ["agent.encoder_type=pointnext", "normalization.point_cloud=identity"],
        )
    )
