import math
from contextlib import closing
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.training.data import fit_action_normalizer
from dexmani_policy.training.trainer import build_trainer


@pytest.mark.parametrize("action_dim", [22, 44])
def test_centered_rotvec_action_coordinates(action_dim):
    actions = torch.linspace(0, 1, 3 * action_dim).view(3, action_dim)
    arms = actions.unflatten(-1, (action_dim // 22, 22))
    arms[..., 3] = torch.tensor([math.pi - 0.1, 0.3 - math.pi, 0.15 - math.pi])[:, None]
    arms[..., 4:6] = 0
    expected = torch.tensor([-1.0, 1.0, 0.25])[:, None].expand(-1, action_dim // 22)

    class Decoder(nn.Module):
        def compute_loss(self, condition, target, **kwargs):
            torch.testing.assert_close(target[0].unflatten(-1, (-1, 22))[..., 3], expected)
            return target.sum(), {}

        def predict_action(self, condition, action_shape, *, rtc_prefix, **kwargs):
            assert rtc_prefix.shape == (1, 3, action_dim)
            torch.testing.assert_close(rtc_prefix[0].unflatten(-1, (-1, 22))[..., 3], expected)
            return rtc_prefix

    dataset = SimpleNamespace(
        action_key="action_rotvec", iter_normalization_data=lambda key: iter((actions[:1], actions[1:])),
    )
    policy = Policy(nn.Identity(), Decoder(), 3, 1, 3, action_dim, center_rotvec=True)
    policy.supports_rtc = True
    fit_action_normalizer(dataset, policy.normalizer, center_rotvec=True)
    policy.compute_loss({"obs": {}, "action": actions[None]})
    output = policy._predict_from_condition(torch.zeros(1, 1), rtc_prefix=actions[None])
    torch.testing.assert_close(output["future_actions"][0], actions)


def test_non_rgb_training_ignores_rgb_source(tmp_path, small_policy_config):
    cfg = small_policy_config(tmp_path / "non_rgb", "+initialization.rgb=unused")
    with closing(build_trainer(cfg)) as trainer:
        assert not hasattr(trainer.model.obs_encoder, "rgb_encoder")
        assert "prior_data" not in trainer.cfg
