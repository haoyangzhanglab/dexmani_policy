from unittest.mock import patch

import pytest
import torch

from dexmani_policy.agents.policies import dp
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.initialization import build_policy

from tests.helpers import (
    SmallRGBWeights, build_small_rgb, check_observation_identity_constraints, check_recipe_composition,
    tiny_hf_rgb_config,
)


@pytest.mark.parametrize("tune_mode", [None, "freeze", "lora", "full"])
@pytest.mark.parametrize("has_source", [False, True])
def test_dp_rgb_initialization(tmp_path, make_demo, monkeypatch, tune_mode, has_source):
    monkeypatch.setattr(dp, "build_rgb_encoder", build_small_rgb)
    if tune_mode is not None:
        monkeypatch.setattr(SmallRGBWeights, "tune_mode", tune_mode, raising=False)
    dataset = DemonstrationDataset(make_demo(), sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    source = tmp_path / "rgb.pt"
    rgb = SmallRGBWeights()
    torch.save(rgb.state_dict(), source)
    cfg = load_policy_config("dp", [
        "training.device=cpu", "agent.action_dim=19", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        f"initialization.rgb={source if has_source else 'null'}", "agent.down_dims=[16,32]",
    ])
    with patch.object(SmallRGBWeights, "initialize_weights", autospec=True,
                      side_effect=SmallRGBWeights.initialize_weights) as initialize:
        if not has_source and tune_mode in {"freeze", "lora"}:
            with pytest.raises(ValueError, match="random backbone.*initialization.rgb.*tune_mode=full"):
                build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
            initialize.assert_not_called()
            return
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    assert isinstance(model.obs_encoder, dp.DPEncoder)
    if has_source:
        initialize.assert_called_once_with(model.obs_encoder.rgb_encoder, str(source))
        torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())
    else:
        initialize.assert_not_called()


@pytest.mark.parametrize("kind,tune_mode", [
    ("resnet", "freeze"), ("resnet", "full"),
    ("dino", "freeze"), ("dino", "lora"), ("dino", "full"),
    ("siglip", "freeze"), ("siglip", "lora"), ("siglip", "full"),
    ("dinov3", "freeze"), ("dinov3", "lora"), ("dinov3", "full"),
])
def test_fresh_rgb_without_weights(make_demo, kind, tune_mode):
    dataset = DemonstrationDataset(make_demo(), sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    cfg = load_policy_config("dp", [
        "training.device=cpu", "agent.action_dim=19", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "initialization.rgb=null", "agent.down_dims=[16,32]",
    ])
    rgb_config = {"tune_mode": tune_mode}
    if kind in {"dino", "siglip"}:
        rgb_config.update(architecture=tiny_hf_rgb_config(kind).to_dict(), image_size=[8, 8])
    elif kind == "dinov3":
        rgb_config["architecture"] = {"img_size": 32, "embed_dim": 24, "depth": 1, "num_heads": 3}
    # Exercise the default frozen ViT as well as explicitly frozen ResNet.
    if kind != "resnet" and tune_mode == "freeze":
        del rgb_config["tune_mode"]
    cfg.agent.rgb_backbone_name = kind
    cfg.agent.rgb_backbone_config = rgb_config
    if tune_mode in {"freeze", "lora"}:
        with pytest.raises(ValueError, match="random backbone.*initialization.rgb.*tune_mode=full"):
            build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    else:
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
        assert model.obs_encoder.rgb_encoder.tune_mode == "full"


def test_recipe_composition():
    check_recipe_composition("dp")
    check_observation_identity_constraints("dp", ['finger_tactile_force', 'rgb', 'taxel_tactile_force'])
