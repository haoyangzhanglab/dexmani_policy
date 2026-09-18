from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.consistency_flow import (
    ConsistencyFlowMatch,
)
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
from dexmani_policy.agents.action_decoders.time_sampler import shift_time_to_noise


class ConstantVelocity(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x, timestep, context, **kwargs):
        self.calls += 1
        return torch.full_like(x, self.value)


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shuffle = None

    def forward(self, x, timestep, context, shuffle=False):
        self.shuffle = shuffle
        return torch.zeros_like(x)


class ConsistencyBackbone(nn.Module):
    def forward(self, x, timestep, target_t, context):
        return torch.zeros_like(x)


def test_time_shift_identity_and_noise_bias():
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    torch.testing.assert_close(shift_time_to_noise(t, 1.0), t)

    shifted = shift_time_to_noise(t, 4.0)
    assert shifted[0] == 0
    assert shifted[-1] == 1
    assert torch.all(shifted[1:-1] < t[1:-1])

    with pytest.raises(ValueError):
        shift_time_to_noise(t, 0.5)


@pytest.mark.parametrize("num_steps", [1, 2, 4])
def test_rectified_flow_euler_uses_exact_nfe(num_steps):
    model = ConstantVelocity(value=2.0)
    flow = RectifiedFlow(model, num_inference_steps=num_steps)

    template = torch.zeros(3, 5, 2)
    cond = torch.zeros(3, 7, 4)

    torch.manual_seed(0)
    expected_noise = torch.randn_like(template)
    torch.manual_seed(0)
    result = flow.predict_action(cond, template)

    torch.testing.assert_close(result, expected_noise + 2.0)
    assert model.calls == num_steps


def test_rectified_flow_forwards_only_explicit_model_kwargs():
    model = RecordingModel()
    flow = RectifiedFlow(model, num_inference_steps=2)
    flow.time_sampler.sample = lambda batch_size, mode, device: torch.full(
        (batch_size,), 0.5, device=device
    )

    actions = torch.zeros(2, 4, 3)
    cond = torch.zeros(2, 6, 8)
    loss, logs = flow.compute_loss(
        cond,
        actions,
        dim_groups={"unused": (0, 1)},
        model_kwargs={"shuffle": True},
    )

    assert torch.isfinite(loss)
    assert model.shuffle is True
    assert {"loss", "loss_action", "pred_v_magnitude", "target_v_magnitude", "t_mean"} <= set(logs)


def test_consistency_flow_requires_ema_decoder():
    flow = ConsistencyFlowMatch(
        ConsistencyBackbone(),
        num_inference_steps=4,
    )
    actions = torch.zeros(2, 4, 3)
    cond = torch.zeros(2, 5, 8)

    with pytest.raises(RuntimeError, match="EMA action decoder"):
        flow.compute_loss(cond, actions)


def test_consistency_flow_batch_one_falls_back_to_flow_loss():
    backbone = ConsistencyBackbone()
    flow = ConsistencyFlowMatch(backbone, num_inference_steps=4)
    ema_decoder = SimpleNamespace(model=ConsistencyBackbone())

    actions = torch.zeros(1, 4, 3)
    cond = torch.zeros(1, 5, 8)
    with pytest.warns(UserWarning, match="batch_size < 2"):
        loss, logs = flow.compute_loss(
            cond,
            actions,
            ema_decoder=ema_decoder,
        )

    assert torch.isfinite(loss)
    assert logs["has_consistency"] == 0
