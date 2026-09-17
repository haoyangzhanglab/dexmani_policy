import torch
import torch.nn as nn

from dexmani_policy.agents.core.base import BaseAgent


class _Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)


def _agent(dropout_probs):
    agent = BaseAgent(
        obs_encoder=_Dummy(),
        action_decoder=_Dummy(),
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        action_dim=3,
        modality_dropout_probs=dropout_probs,
    )
    agent.normalizer.fit_field("joint_state", torch.randn(100, 3), mode="limits")
    agent.normalizer.fit_field("action", torch.randn(100, 3), mode="limits")
    return agent


def test_identity_modality_dropout_applied_in_train():
    agent = _agent({"point_cloud": 1.0, "rgb": 1.0, "joint_state": 0.0})
    agent.train()
    out = agent.preprocess({
        "joint_state": torch.randn(2, 2, 3),
        "point_cloud": torch.ones(2, 2, 4, 3),
        "rgb": torch.ones(2, 2, 3, 4, 4),
    })
    # p=1.0 -> dropped (zeroed) for identity modalities, joint_state untouched
    assert (out["point_cloud"] == 0).all()
    assert (out["rgb"] == 0).all()
    assert (out["joint_state"] != 0).any()


def test_identity_modality_dropout_noop_in_eval():
    agent = _agent({"point_cloud": 1.0})
    agent.eval()
    pc = torch.ones(2, 2, 4, 3)
    out = agent.preprocess({"joint_state": torch.randn(2, 2, 3), "point_cloud": pc})
    assert (out["point_cloud"] == 1).all()


def test_no_point_cloud_clamp_in_preprocess():
    agent = _agent({})
    pc = torch.tensor([[[[5.0, -5.0, 0.0]]]])  # values far outside [-1, 1]
    out = agent.preprocess({"joint_state": torch.randn(1, 2, 3), "point_cloud": pc})
    assert (out["point_cloud"] == pc).all()
