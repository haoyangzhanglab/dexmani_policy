from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.rtc import clean_estimate, guided_step, prefix_weights
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.normalization import LinearNormalizer, SingleFieldLinearNormalizer
from dexmani_policy.deployment.runtime import LoadedPolicy


class Coupled(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.7))

    def forward(self, x, timestep, context):
        return self.gain * x.sin() + 0.3 * x.roll(1, dims=-1) + 0.2


@pytest.mark.parametrize("kind", ["sample", "epsilon", "v_prediction"])
def test_off_is_original_exact(kind):
    decoder = Diffusion(
        Coupled(), num_training_steps=20, num_inference_steps=4, prediction_type=kind
    )
    template = torch.zeros(2, 6, 3)
    cond = torch.zeros(2, 1)
    torch.manual_seed(72)
    original = decoder.predict_action(cond, template)
    original_rng = torch.get_rng_state()
    for prefix in (None, torch.ones(2, 4, 2), torch.empty(2, 0, 2)):
        torch.manual_seed(72)
        result = decoder.predict_action(
            cond, template, rtc_prefix=prefix, rtc_guidance_cap=0, history_offset=1
        )
        assert torch.equal(original, result)
        assert torch.equal(original_rng, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["sample", "epsilon", "v_prediction"])
@pytest.mark.parametrize("clip", [False, True])
def test_vjp_finite_difference_and_original_base(kind, clip):
    decoder = Diffusion(
        Coupled().double(),
        num_training_steps=20,
        num_inference_steps=4,
        prediction_type=kind,
        clip_sample=clip,
    )
    scheduler = decoder.noise_scheduler
    scheduler.set_timesteps(4)
    t = scheduler.timesteps[1]
    x = torch.tensor([[[2.3, -0.4, 3.2]]], dtype=torch.float64)
    cond = torch.zeros(1, 1, dtype=x.dtype)
    weight = torch.tensor([[[0.4, 0, 0]]], dtype=x.dtype)
    target = torch.tensor([[[2.0, float("nan"), float("nan")]]], dtype=x.dtype)
    a = scheduler.alphas_cumprod[t].double().sqrt()
    b = (1 - scheduler.alphas_cumprod[t].double()).sqrt()

    def f(value):
        return clean_estimate(decoder.model(value, t, cond), value, a, b, kind)

    residual = (0.4 * (2 - f(x)[0, 0, 0])).detach()
    g = torch.zeros_like(x)
    eps = 1e-6
    for i in range(3):
        delta = torch.zeros_like(x)
        delta[..., i] = eps
        g[..., i] = residual * (f(x + delta)[..., 0] - f(x - delta)[..., 0]) / (2 * eps)
    assert g[..., 2].abs().item() > 1e-6  # Unconstrained auxiliary latent is coupled.
    prev = int(t) - 20 // 4
    ap = scheduler.alphas_cumprod[prev].double().sqrt()
    bp = (1 - scheduler.alphas_cumprod[prev].double()).sqrt()
    base = scheduler.step(decoder.model(x, t, cond), t, x).prev_sample
    expected = base + (ap * b - a * bp) * min(3.0, 1 / (a * b)) * g
    with torch.inference_mode():
        actual = guided_step(decoder.model, scheduler, x, t, cond, target, weight, 3.0, final=False)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    assert decoder.model.gain.grad is None
    assert actual.grad_fn is None
    # Equivalent VP/flow coordinate coefficient.
    torch.testing.assert_close(
        (ap + bp) * (a + b) * (ap / (ap + bp) - a / (a + b)), ap * b - a * bp
    )


@pytest.mark.parametrize("clip", [True, False])
def test_final_projection_and_zero_weight(clip):
    model = nn.Module()
    model.forward = lambda x, timestep, context: 2 * x
    decoder = Diffusion(model, num_training_steps=20, num_inference_steps=4, clip_sample=clip)
    scheduler = decoder.noise_scheduler
    scheduler.set_timesteps(4)
    x = torch.ones(1, 2, 2) * 3
    t = scheduler.timesteps[-1]
    target = torch.ones_like(x) * 1000
    actual = guided_step(model, scheduler, x, t, x, target, torch.ones_like(x), 100.0, final=True)
    assert (actual.abs().max() <= 1) if clip else (actual.abs().max() > 1)
    t = scheduler.timesteps[0]
    actual = guided_step(model, scheduler, x, t, x, target, torch.zeros_like(x), 100.0, final=False)
    torch.testing.assert_close(
        actual, scheduler.step(model(x, t, x), t, x).prev_sample, rtol=0, atol=0
    )


def test_weights_and_constant_denoiser():
    w = prefix_weights(4, 2, device="cpu", dtype=torch.float64)
    assert torch.equal(w[:2], torch.ones(2, dtype=torch.float64))
    assert 0 < w[3] < w[2] < 1
    assert torch.equal(prefix_weights(4, 4, device="cpu", dtype=torch.float32), torch.ones(4))
    model = nn.Module()
    model.forward = lambda x, timestep, context: torch.zeros_like(x)
    decoder = Diffusion(model, num_training_steps=20, num_inference_steps=4)
    with torch.inference_mode():
        output = decoder.predict_action(
            torch.zeros(1, 1),
            torch.zeros(1, 6, 3),
            rtc_prefix=torch.ones(1, 4, 2),
            delay_steps=2,
            history_offset=1,
            rtc_guidance_cap=2,
        )
    assert torch.isfinite(output).all()


class Encoder(nn.Module):
    def forward(self, obs):
        return obs["joint_state"].reshape(1, -1), {}


class AuxAgent(BaseAgent):
    @property
    def control_action_dim(self):
        return 19


def test_full_future_control_normalizer_and_outer_inference():
    agent = AuxAgent(
        Encoder(),
        Diffusion(Coupled(), num_training_steps=20, num_inference_steps=4),
        horizon=16,
        n_obs_steps=2,
        n_action_steps=3,
        action_dim=28,
    ).eval()
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
        np.arange(28, dtype=np.float32) + 1, np.arange(28, dtype=np.float32)
    )
    normalizer["joint_state"] = SingleFieldLinearNormalizer.create_identity()
    agent.load_normalizer_from_dataset(normalizer)
    info = SimpleNamespace(
        observation_fields=("joint_state",),
        inference_steps=4,
        action_mode="joint",
        n_obs_steps=2,
        n_action_steps=3,
        horizon=16,
    )
    loaded = LoadedPolicy(agent, {"dataset": {}}, info, device="cpu", seed=0)
    loaded.configure_rtc(2.0)
    obs = {"joint_state": np.zeros((2, 19), np.float32)}
    loaded.reset_episode()
    result = loaded.predict(obs)
    assert result.shape == (15, 19)
    seen = []
    original = agent.action_decoder.predict_action

    def capture(*args, **kwargs):
        seen.append(kwargs["rtc_prefix"].clone())
        return original(*args, **kwargs)

    agent.action_decoder.predict_action = capture
    with torch.inference_mode():
        result = loaded.predict(obs, rtc_prefix=np.ones((5, 19)), delay_steps=2)
    expected = np.arange(19) * 2 + 1
    np.testing.assert_allclose(seen[0][0].numpy(), np.broadcast_to(expected, (5, 19)))
    assert result.shape == (15, 19) and np.isfinite(result).all()
    assert all(p.grad is None for p in agent.parameters())


def test_invalid_scheduler():
    decoder = Diffusion(Coupled(), num_training_steps=20, num_inference_steps=4)
    decoder.noise_scheduler.register_to_config(thresholding=True)
    with pytest.raises(NotImplementedError):
        decoder.predict_action(
            torch.zeros(1, 1),
            torch.zeros(1, 6, 3),
            rtc_prefix=torch.ones(1, 4, 2),
            rtc_guidance_cap=2,
        )


@pytest.mark.parametrize("backbone", ["unet", "one_way"])
def test_real_backbone_vjp(backbone):
    torch.set_num_threads(1)
    if backbone == "unet":
        from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D

        model = ConditionalUnet1D(
            19, 8, diffusion_step_embed_dim=16, down_dims=[16, 32], kernel_size=3, n_groups=4
        )
        cond = torch.randn(2, 8)
        dims = 19
    else:
        from dexmani_policy.agents.action_decoders.backbone.one_way_transformer import (
            OneWayTransformerBackbone,
        )

        model = OneWayTransformerBackbone(
            8,
            28,
            2,
            4,
            24,
            16,
            timestep_embed_dim=16,
            embedding_dim=16,
            depth=1,
            num_heads=2,
            mlp_dim=32,
            use_aux_ee=True,
            joint_dim=19,
            ee_dim=9,
        )
        cond = torch.randn(2, 4, 24)
        dims = 28
    model.eval()
    decoder = Diffusion(model, num_training_steps=20, num_inference_steps=3)
    with torch.inference_mode():
        result = decoder.predict_action(
            cond,
            torch.zeros(2, 8, dims),
            rtc_prefix=torch.zeros(2, 5, 19),
            delay_steps=2,
            history_offset=1,
            rtc_guidance_cap=2.0,
        )
    assert result.shape == (2, 8, dims) and torch.isfinite(result).all()
    assert all(p.grad is None for p in model.parameters())


def test_history_and_tail_have_no_endpoint_residual():
    decoder = Diffusion(Coupled(), num_training_steps=20, num_inference_steps=4, clip_sample=False)
    template = torch.zeros(2, 8, 3)
    cond = torch.zeros(2, 1)
    torch.manual_seed(11)
    original = decoder.predict_action(cond, template)
    torch.manual_seed(11)
    guided = decoder.predict_action(
        cond,
        template,
        rtc_prefix=torch.ones(2, 4, 2),
        delay_steps=2,
        history_offset=1,
        rtc_guidance_cap=1.0,
    )
    torch.testing.assert_close(guided[:, :1], original[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(guided[:, 5:], original[:, 5:], rtol=0, atol=0)
    assert not torch.equal(guided[:, 1:5], original[:, 1:5])


def test_bridge_future_starts_at_n_minus_one():
    class TemporalAgent(nn.Module):
        def predict_action(self, obs, inference_steps=None):
            return {
                "pred_action": torch.arange(16, dtype=torch.float32)[None, :, None].expand(
                    1, 16, 19
                )
            }

    info = SimpleNamespace(
        observation_fields=("joint_state",),
        inference_steps=1,
        action_mode="joint",
        n_obs_steps=2,
        n_action_steps=3,
        horizon=16,
    )
    policy = LoadedPolicy(TemporalAgent(), {"dataset": {}}, info, device="cpu", seed=0)
    output = policy.predict({"joint_state": np.zeros((2, 19), np.float32)})
    np.testing.assert_array_equal(output[:, 0], np.arange(1, 16))


def test_eef_bridge_mixed_affine_prefix():
    from dexmani_policy.agents.normalization import build_mixed_action_normalizer

    agent = BaseAgent(
        Encoder(),
        Diffusion(Coupled(), num_training_steps=20, num_inference_steps=3),
        horizon=8,
        n_obs_steps=2,
        n_action_steps=3,
        action_dim=21,
    ).eval()
    physical = np.arange(210, dtype=np.float32).reshape(10, 21) / 100
    normalizer = LinearNormalizer()
    normalizer["action"] = build_mixed_action_normalizer(physical)
    normalizer["joint_state"] = SingleFieldLinearNormalizer.create_identity()
    agent.load_normalizer_from_dataset(normalizer)
    info = SimpleNamespace(
        observation_fields=("joint_state",),
        inference_steps=3,
        action_mode="eef",
        n_obs_steps=2,
        n_action_steps=3,
        horizon=8,
    )
    policy = LoadedPolicy(agent, {"dataset": {}}, info, device="cpu", seed=0)
    policy.configure_rtc(2.0)
    prefix = physical[:4]
    seen = []
    predict = agent.action_decoder.predict_action

    def capture(*args, **kwargs):
        seen.append(kwargs["rtc_prefix"].detach().numpy())
        return predict(*args, **kwargs)

    agent.action_decoder.predict_action = capture
    output = policy.predict(
        {"joint_state": np.zeros((2, 19), np.float32)}, rtc_prefix=prefix, delay_steps=2
    )
    assert output.shape == (7, 21) and np.isfinite(output).all()
    np.testing.assert_allclose(seen[0][0, :, 3:9], prefix[:, 3:9])
    np.testing.assert_allclose(seen[0][0], normalizer["action"].normalize(prefix).numpy())


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("delay,cap", [(0, 2.0), (3, 2.0), (0, 0.0)])
def test_execution_warmup_uses_requested_rtc_delay(mode, delay, cap):
    agent = AuxAgent(
        Encoder(),
        Diffusion(Coupled(), num_training_steps=20, num_inference_steps=2),
        horizon=8,
        n_obs_steps=2,
        n_action_steps=3,
        action_dim=28,
    ).eval()
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["joint_state"] = SingleFieldLinearNormalizer.create_identity()
    agent.load_normalizer_from_dataset(normalizer)
    info = SimpleNamespace(
        observation_fields=("joint_state",),
        inference_steps=2,
        action_mode="joint",
        n_obs_steps=2,
        n_action_steps=3,
        horizon=8,
    )
    policy = LoadedPolicy(agent, {"dataset": {}}, info, device="cpu", seed=0)
    calls = []
    original = policy.predict

    def observe_call(observation, *, rtc_prefix=None, delay_steps=0):
        calls.append((rtc_prefix is not None, delay_steps))
        return original(observation, rtc_prefix=rtc_prefix, delay_steps=delay_steps)

    policy.predict = observe_call
    durations = policy.configure_execution(
        mode, cap if mode == "rtc" else None, warmup=True, rtc_delay=delay
    )
    assert len(durations) == 1 and np.isfinite(durations[0]) and durations[0] >= 0
    assert calls == ([(False, 0), (True, delay)] if mode == "rtc" and cap > 0 else [(False, 0)])
    assert all(p.grad is None for p in agent.parameters())
