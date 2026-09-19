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
    assert {
        "loss",
        "loss_action",
        "pred_v_magnitude",
        "target_v_magnitude",
        "t_mean",
    } <= set(logs)


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


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_rectified_flow_override_and_time_grid(steps):
    class RecordingVelocity(ConstantVelocity):
        def forward(self, x, timestep, context):
            times.append(timestep.clone())
            return super().forward(x, timestep, context)

    times = []
    model = RecordingVelocity(2)
    flow = RectifiedFlow(model, num_inference_steps=7)
    template = torch.zeros(2, 4, 3)
    torch.manual_seed(8)
    noise = torch.randn_like(template)
    torch.manual_seed(8)
    result = flow.predict_action(torch.zeros(2, 5, 8), template, steps)
    torch.testing.assert_close(result, noise + 2)
    assert model.calls == steps
    for i, t in enumerate(times):
        torch.testing.assert_close(t, torch.full((2,), i / steps))


def test_rectified_flow_interpolation_target_and_shift(monkeypatch):
    noise = torch.full((2, 4, 3), -2.0)
    actions = torch.full_like(noise, 3.0)
    time = torch.tensor([0.25, 0.75])
    shifted = time / (1 + 3 * (1 - time))

    class Oracle(nn.Module):
        def forward(self, x, timestep, context):
            torch.testing.assert_close(timestep, shifted)
            torch.testing.assert_close(
                x, (-2 + 5 * shifted[:, None, None]).expand_as(x)
            )
            return torch.full_like(x, 5.0)

    flow = RectifiedFlow(Oracle(), time_shift_alpha=4)
    monkeypatch.setattr(torch, "randn_like", lambda x: noise)
    monkeypatch.setattr(flow.time_sampler, "sample", lambda *a, **kw: time)
    loss, _ = flow.compute_loss(torch.zeros(2, 3, 4), actions)
    assert loss.item() == 0
    assert flow.requires_ema_for_loss is False


@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("times", [[0.0, 0.5], [0.9, 1.0]])
def test_consistency_target_math_and_teacher_is_detached(monkeypatch, mode, times):
    flow = ConsistencyFlowMatch(
        ConsistencyBackbone(), num_inference_steps=4, target_t_sample_mode=mode
    )
    actions = torch.full((2, 4, 3), 3.0)
    noise = torch.full_like(actions, -2.0)
    t, dt = torch.tensor(times), torch.tensor([0.25, 0.75])
    next_t = (t + dt).clamp(max=1.0)
    teacher_calls = []

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, x, timestep, target_t, context):
            assert not torch.is_grad_enabled()
            torch.testing.assert_close(x, (-2 + 5 * next_t[:, None, None]).expand_as(x))
            torch.testing.assert_close(timestep, next_t)
            torch.testing.assert_close(
                target_t, dt if mode == "relative" else (next_t + dt).clamp(max=1.0)
            )
            teacher_calls.append(True)
            return self.weight * torch.ones_like(x)

    samples = iter([t, dt])
    monkeypatch.setattr(flow.time_sampler, "sample", lambda *a, **kw: next(samples))
    monkeypatch.setattr(torch, "randn_like", lambda x: noise)
    teacher = Teacher()
    result = flow.get_consistency_velocity(actions, torch.zeros(2, 5, 8), teacher)
    xt, xt_next = -2 + 5 * t, -2 + 5 * next_t
    expected = (xt_next + 2 * (1 - next_t) - xt) / (1 - t).clamp(min=0.25)
    torch.testing.assert_close(
        result["vt_target"], expected[:, None, None].expand_as(actions)
    )
    torch.testing.assert_close(result["target_t"], dt if mode == "relative" else next_t)
    assert not result["vt_target"].requires_grad
    assert teacher.weight.grad is None and len(teacher_calls) == 1


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_consistency_flow_training_and_inference_time_arguments(monkeypatch, mode):
    calls = []

    class Backbone(nn.Module):
        def forward(self, x, timestep, target_t, context):
            calls.append((timestep.clone(), target_t.clone()))
            return torch.ones_like(x)

    flow = ConsistencyFlowMatch(
        Backbone(), num_inference_steps=4, target_t_sample_mode=mode
    )
    monkeypatch.setattr(
        flow.time_sampler, "sample", lambda *a, **kw: torch.tensor([0.25, 0.75])
    )
    targets = flow.get_flow_velocity(torch.ones(2, 4, 3))
    torch.testing.assert_close(
        targets["target_t"], torch.zeros(2) if mode == "relative" else targets["t"]
    )
    x = torch.zeros(2, 4, 3)
    result = flow.sample_ode(x, 4, torch.zeros(2, 5, 8))
    torch.testing.assert_close(result, torch.ones_like(x))
    for i, (t, target) in enumerate(calls):
        torch.testing.assert_close(t, torch.full((2,), i / 4))
        torch.testing.assert_close(
            target, torch.full((2,), 0.25 if mode == "relative" else (i + 1) / 4)
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_consistency_missing_ema_never_falls_back(batch_size):
    flow = ConsistencyFlowMatch(ConsistencyBackbone())
    assert flow.requires_ema_for_loss is True
    with pytest.raises(RuntimeError, match="EMA"):
        flow.compute_loss(torch.zeros(batch_size, 5, 8), torch.zeros(batch_size, 4, 3))
