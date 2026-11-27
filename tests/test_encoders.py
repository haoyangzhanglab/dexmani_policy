import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.agents.obs_encoder.rgb.resnet import convert_group_norm


def test_bn_to_gn_copies_learned_affines():
    batch_norm = nn.BatchNorm2d(4).double()
    with torch.no_grad():
        batch_norm.weight.copy_(torch.tensor([2., 0.5, 1.5, -1.]))
        batch_norm.bias.fill_(0.4)
        batch_norm.running_mean.fill_(5)
    model = convert_group_norm(nn.Sequential(batch_norm))
    assert isinstance(model[0], nn.GroupNorm)
    torch.testing.assert_close(model[0].weight, batch_norm.weight)
    torch.testing.assert_close(model[0].bias, batch_norm.bias)
    assert set(model[0].state_dict()) == {"weight", "bias"}


def test_warm_start_inherits_rgb_architecture():
    from dexmani_policy.training.initialization import inherit_rgb_architecture

    agent = OmegaConf.create({"rgb_backbone_name": "dino", "rgb_backbone_config": {}})
    saved = {"rgb_backbone_name": "dino", "rgb_backbone_config": {"architecture": {"hidden_size": 16}}}
    inherit_rgb_architecture(agent, saved)
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
