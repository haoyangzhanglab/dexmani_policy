import math
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones import ditx
from dexmani_policy.agents.action_decoders.backbones.modules import (
    FlowTimestepEmbedding, SinusoidalTimestepEmbedding, TimestepMLP,
)
from dexmani_policy.agents.action_decoders.diffusion import Diffusion, ForceDiffusion
from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow
from dexmani_policy.agents.action_decoders.sampling.time_sampling import TimeSampler

from tests.helpers import check_context_cache


def test_context_cache_outputs_gradients_and_new_observation_consistencyditx(monkeypatch):
    torch.manual_seed(4)
    layout = {"horizon": 4, "action_dim": 19}
    backbone = ditx.ConsistencyDiTX(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, **layout,
    ).eval()
    check_context_cache(monkeypatch, backbone, ConsistencyFlow, {"target_t": 0.2})


def test_flow_backbone_time_and_qk_defaults_ditx():
    layout = {"horizon": 4, "action_dim": 19}
    model = ditx.DiTX(n_obs_steps=2, obs_token_dim=8, hidden_dim=16, n_layers=2, n_head=2, **layout)
    assert isinstance(model.timestep_embedder.net[0], FlowTimestepEmbedding)
    attentions = [module for module in model.ditx_blocks.modules() if isinstance(module, ditx.CrossAttention)]
    assert len(attentions) == (2)
    for attention in attentions:
        assert isinstance(attention.q_norm, nn.LayerNorm)
        assert isinstance(attention.k_norm, nn.LayerNorm)
        assert attention.q_norm.normalized_shape == attention.k_norm.normalized_shape == (8,)


def test_flow_backbone_time_and_qk_defaults_consistencyditx():
    layout = {"horizon": 4, "action_dim": 19}
    model = ditx.ConsistencyDiTX(n_obs_steps=2, obs_token_dim=8, hidden_dim=16, n_layers=2, n_head=2, **layout)
    assert isinstance(model.timestep_embedder.net[0], FlowTimestepEmbedding)
    assert isinstance(model.target_t_embedder.net[0], FlowTimestepEmbedding)
    attentions = [module for module in model.ditx_blocks.modules() if isinstance(module, ditx.CrossAttention)]
    assert len(attentions) == (2)
    for attention in attentions:
        assert isinstance(attention.q_norm, nn.LayerNorm)
        assert isinstance(attention.k_norm, nn.LayerNorm)
        assert attention.q_norm.normalized_shape == attention.k_norm.normalized_shape == (8,)


@pytest.mark.parametrize("steps", [10, 20])
def test_ddim_actual_time_grid(steps):
    visited = []

    class Denoiser(nn.Module):
        def forward(self, x, timestep, context):
            visited.append(timestep.item())
            return x * 0.5

    cond, shape = torch.zeros(1, 1, 2), (1, 4, 2)
    decoder = Diffusion(Denoiser(), num_inference_steps=steps)
    result = decoder.predict_action(cond, shape, generator=torch.Generator().manual_seed(9))
    stride = 100 // steps
    assert visited == list(range(100 - stride, -1, -stride))
    assert result.isfinite().all()
    with pytest.raises(ValueError, match="t=0"):
        Diffusion(Denoiser(), num_inference_steps=1)
    with pytest.raises(ValueError, match="t=0"):
        decoder.predict_action(cond, shape, inference_steps=1)
    with pytest.raises(ValueError, match="t=0"):
        decoder.predict_action(cond, shape, inference_steps=1, rtc_prefix=torch.zeros(1, 2, 2), rtc_guidance_cap=1)


def test_ddim_training_grid_bounds_direct_sampling():
    with pytest.raises(ValueError, match="num_training_steps"):
        ForceDiffusion(nn.Identity(), num_training_steps=4, num_inference_steps=5)
    decoder = Diffusion(nn.Identity(), num_training_steps=4, num_inference_steps=4)
    assert decoder.effective_config == {"num_training_steps": 4}
    with pytest.raises(ValueError, match="num_training_steps"):
        decoder.predict_action(torch.zeros(1, 1, 2), (1, 4, 2), inference_steps=5)


@pytest.mark.parametrize("prediction_type", ["sample", "epsilon"])
@pytest.mark.parametrize("clip_range,prefix_value", [(1.0, 1.0), (1.0, 0.25), (0.5, 0.25)])
def test_ddim_rtc_clipped_reference(monkeypatch, prediction_type, clip_range, prefix_value):
    class Denoiser(nn.Module):
        def forward(self, x, timestep, context):
            clean = (2 if timestep > 0 else 1) * x
            if prediction_type == "sample":
                return clean
            a = decoder.noise_scheduler.alphas_cumprod[timestep]
            return (x - a.sqrt() * clean) / (1 - a).sqrt()

    decoder = Diffusion(Denoiser(), num_training_steps=100, num_inference_steps=2,
                        prediction_type=prediction_type, clip_sample=True)
    decoder.noise_scheduler.register_to_config(clip_sample_range=clip_range)
    shape = (1, 1, 1)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.ones(shape))
    with torch.inference_mode():
        actual = decoder.predict_action(
            torch.zeros(shape), shape, rtc_prefix=torch.full(shape, prefix_value), rtc_guidance_cap=0.8,
        )
    reference = 1.0
    soft_weight = 0.5 * math.expm1(0.5) / math.expm1(1)
    for current, previous in [(50, 0), (0, -50)]:
        alphas = decoder.noise_scheduler.alphas_cumprod
        a, a_prev = float(alphas[current]), float(alphas[previous]) if previous >= 0 else 1.0
        slope = 2 if current > 0 else 1
        raw_clean = slope * reference
        clean = max(-clip_range, min(clip_range, raw_clean))
        # DDIM derives epsilon before clipping x0; the VJP also retains the raw slope.
        noise = (reference - math.sqrt(a) * raw_clean) / math.sqrt(1 - a)
        strength = (math.sqrt(a_prev * (1 - a)) - math.sqrt(a * (1 - a_prev))) * min(
            0.8, 1 / math.sqrt(a * (1 - a)),
        )
        reference = (math.sqrt(a_prev) * clean + math.sqrt(1 - a_prev) * noise
                     + strength * slope * soft_weight * (prefix_value - clean))
    reference = max(-clip_range, min(clip_range, reference))
    torch.testing.assert_close(actual, torch.full(shape, reference))
    if clip_range == prefix_value == 1:
        assert actual.item() == pytest.approx(0.98640904, abs=1e-6)


def test_ddim_rtc_matches_two_step_reference():
    class Denoiser(nn.Module):
        def forward(self, x, timestep, context):
            return (0.3 + 0.001 * timestep) * x

    decoder = Diffusion(Denoiser(), num_inference_steps=2, clip_sample=False)
    prefix, shape = torch.full((1, 1, 1), -0.4), (1, 2, 1)
    with torch.inference_mode():
        actual = decoder.predict_action(
            torch.zeros(1, 1, 1), shape, generator=torch.Generator().manual_seed(18),
            rtc_prefix=prefix, history_offset=1, rtc_guidance_cap=0.8,
        )
    reference = torch.randn(shape, generator=torch.Generator().manual_seed(18))
    alpha_bar = decoder.noise_scheduler.alphas_cumprod
    soft_weight = 0.5 * math.expm1(0.5) / math.expm1(1)
    for current, previous in [(50, 0), (0, -50)]:
        a, a_prev = float(alpha_bar[current]), float(alpha_bar[previous]) if previous >= 0 else 1.0
        scale = 0.3 + 0.001 * current
        clean = scale * reference
        residual = prefix - clean[:, 1:]
        predicted_noise = (reference - math.sqrt(a) * clean) / math.sqrt(1 - a)
        reference = math.sqrt(a_prev) * clean + math.sqrt(1 - a_prev) * predicted_noise
        strength = (math.sqrt(a_prev * (1 - a)) - math.sqrt(a * (1 - a_prev))) * min(
            0.8, 1 / math.sqrt(a * (1 - a)),
        )
        reference[:, 1:] += scale * soft_weight * strength * residual
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("dim", [2, 8, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_flow_time_embedding_reference_and_precision(dim, dtype, device):
    embedding = FlowTimestepEmbedding(dim).to(device=device, dtype=dtype)
    timestep = torch.tensor([0., 0.004, 0.1, 0.5, 0.9, 1., 1.1], device=device, dtype=dtype)
    fraction = torch.linspace(0, 1, dim // 2, device=device, dtype=torch.float32)
    period = 0.004 * (4.0 / 0.004) ** fraction
    angles = timestep.float()[:, None] * (2 * math.pi / period)[None, :]
    expected = torch.cat((angles.sin(), angles.cos()), dim=-1)
    assert embedding.frequencies.dtype == torch.float32
    assert not embedding.state_dict()
    for autocast in (False, True):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            actual = embedding(timestep)
        assert actual.shape == (len(timestep), dim)
        assert actual.dtype == torch.float32 and actual.device == timestep.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mlp = TimestepMLP(dim, 16, embedding=embedding).to(device=device, dtype=dtype)
    output = mlp(timestep)
    assert output.dtype == dtype and output.device == timestep.device
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in mlp.parameters())


def test_discrete_time_embedding_unchanged():
    model = TimestepMLP(128, 16)
    assert isinstance(model.net[0], SinusoidalTimestepEmbedding)
    times = torch.tensor([0., 1., 50., 999.])
    frequencies = torch.exp(torch.arange(64) * (-math.log(10000) / 63))
    angles = times[:, None] * frequencies[None, :]
    torch.testing.assert_close(model.net[0](times), torch.cat((angles.sin(), angles.cos()), dim=-1))
    torch.testing.assert_close(model(times), model.net(times), rtol=0, atol=0)


def test_cross_attention_qk_normalization_cache():
    attention = ditx.CrossAttention(16, num_heads=2, qk_norm=True).eval()
    query, context = torch.randn(2, 4, 16), torch.randn(2, 6, 16)
    q = attention.q(query).unflatten(-1, (2, 8)).transpose(1, 2)
    k, v = attention.kv(context).unflatten(-1, (2, 2, 8)).permute(2, 0, 3, 1, 4).unbind(0)
    expected = F.scaled_dot_product_attention(attention.q_norm(q), attention.k_norm(k), v)
    expected = attention.proj(expected.transpose(1, 2).flatten(2))
    torch.testing.assert_close(attention(query, context), expected)
    capture = Mock(return_value=None)
    with attention.k_norm.register_forward_pre_hook(capture):
        cache = attention.prepare_kv(context)
        torch.testing.assert_close(cache[1], v)
        for _ in range(10):
            torch.testing.assert_close(attention(query, None, context_kv=cache), expected)
    assert capture.call_count == 1


class ConstantVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(value))

    def forward(self, x, timestep, context, **kwargs):
        self.last_input = x.detach()
        self.last_time = timestep
        self.last_target_time = kwargs.get("target_t")
        return x * 0.0 + self.velocity

    def prepare_context_kv(self, context):
        return None


@pytest.mark.parametrize("decoder_class", [RectifiedFlow, ConsistencyFlow])
@pytest.mark.parametrize("steps", [5, 10])
def test_flow_euler_grid_and_runtime_override(decoder_class, steps):
    decoder = decoder_class(ConstantVelocity(0.25)).eval()
    assert decoder.num_inference_steps == 10
    cond, shape = torch.zeros(2, 3, 8), (2, 4, 19)
    with patch.object(decoder.model, "forward", wraps=decoder.model.forward) as calls:
        result = decoder.predict_action(cond, shape, inference_steps=steps,
                                        generator=torch.Generator().manual_seed(37))
    assert calls.call_count == steps
    actual_times = torch.stack([call.kwargs["timestep"] for call in calls.call_args_list])
    torch.testing.assert_close(actual_times, (torch.arange(steps) / steps)[:, None].expand(-1, 2))
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(37))
    torch.testing.assert_close(result, noise + 0.25)
    with patch.object(decoder.model, "forward", wraps=decoder.model.forward) as calls:
        decoder.predict_action(cond, shape)
    assert calls.call_count == 10


def test_flow_direction_and_rtc_against_reference():
    cond, action = torch.zeros(1, 1, 2), torch.full((1, 2, 1), 0.7)
    decoder = RectifiedFlow(ConstantVelocity(0.25)).eval()
    torch.manual_seed(31)
    noise, t = torch.randn_like(action), torch.rand(1)
    torch.manual_seed(31)
    loss, _ = decoder.compute_loss(cond, action)
    torch.testing.assert_close(decoder.model.last_input, torch.lerp(noise, action, t))
    torch.testing.assert_close(loss, (0.25 - (action - noise)).square().mean())

    def generator():
        return torch.Generator().manual_seed(31)

    plain = decoder.predict_action(cond, action.shape, inference_steps=1, generator=generator())
    torch.testing.assert_close(plain, noise + 0.25)
    prefix = torch.full((1, 1, 1), 0.9)
    guided = decoder.predict_action(
        cond, action.shape, inference_steps=2, generator=generator(),
        rtc_prefix=prefix, history_offset=1, rtc_guidance_cap=0.2,
    )
    expected = noise.clone()
    weight = 0.5 * math.expm1(0.5) / math.expm1(1)
    for t in (0., 0.5):
        residual = prefix - (expected[:, 1:] + (1 - t) * 0.25)
        expected += 0.125
        expected[:, 1:] += 0.1 * weight * residual
    torch.testing.assert_close(guided, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_steps,batch_size", [
    (1, 3), (8, 8), (10, 1), (10, 3), (10, 10), (10, 32), (16, 7),
])
def test_maniflow_discrete_time_distribution(device, num_steps, batch_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(37)
    sampler = TimeSampler(num_steps)
    times = torch.stack([sampler.sample(batch_size, "discrete", device) for _ in range(1024)])
    assert times.shape == (1024, batch_size) and times.dtype == torch.float32
    # Official policy.sample_t uses the entire uniform grid even for one sample.
    grid = torch.arange(num_steps, device=device).float() / num_steps
    torch.testing.assert_close(times.unique(), grid, rtol=0, atol=0)
    counts = (times * num_steps).round().long().flatten().bincount(minlength=num_steps)
    torch.testing.assert_close(
        counts.float() / times.numel(), torch.full_like(grid, 1 / num_steps), rtol=0, atol=0.04,
    )


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_consistency_teacher_target_matches_reference(monkeypatch, mode):
    decoder = ConsistencyFlow(ConstantVelocity(0.2), target_t_sample_mode=mode)
    teacher = ConstantVelocity(-0.4)
    t, dt = torch.tensor([0.2, 0.9]), torch.tensor([0.4, 0.3])
    times = iter((t, dt))
    monkeypatch.setattr(decoder.time_sampler, "sample", lambda *args, **kwargs: next(times))
    target = torch.tensor([[[0.8]], [[-0.3]]])
    torch.manual_seed(8)
    noise = torch.randn_like(target)
    torch.manual_seed(8)
    result = decoder.sample_consistency_targets(target, torch.zeros(2, 1, 2), teacher)
    start = torch.lerp(noise, target, t[:, None, None])
    next_t = torch.minimum(t + dt, torch.ones_like(t))
    next_action = torch.lerp(noise, target, next_t[:, None, None])
    final = next_action - 0.4 * (1 - next_t[:, None, None])
    torch.testing.assert_close(teacher.last_input, next_action)
    torch.testing.assert_close(teacher.last_time, next_t)
    torch.testing.assert_close(teacher.last_target_time, dt if mode == "relative" else next_t + dt)
    torch.testing.assert_close(result["target_t"], dt if mode == "relative" else next_t)
    torch.testing.assert_close(result["vt_target"], (final - start) / (1 - t[:, None, None]))
    assert not result["vt_target"].requires_grad


@pytest.mark.parametrize("groups", [9, 12])
def test_adaln_modulation_matches_independent_layers(groups):
    torch.manual_seed(4)
    hidden, layers = 8, 3
    model = ditx.AdaLNModulation(hidden, layers, groups)
    condition = torch.randn(2, hidden, requires_grad=True)
    weight, bias = model.projection.weight, model.projection.bias
    expected = torch.stack([
        F.linear(F.silu(condition), w, b).unflatten(-1, (groups, hidden))
        for w, b in zip(weight.chunk(layers), bias.chunk(layers))
    ], dim=1)
    actual = model(condition)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    inputs = (condition, weight, bias)
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, probe), torch.autograd.grad(expected, inputs, probe),
    )


def test_rectified_training_pair_preserves_sampling():
    from dexmani_policy.agents.action_decoders.sampling.time_sampling import shift_time_to_noise

    actions = torch.randn(3, 16, 19)
    flow = RectifiedFlow(nn.Identity(), time_shift_alpha=2.)
    torch.manual_seed(37)
    noise = torch.randn_like(actions)
    time = shift_time_to_noise(torch.rand(3), 2.)
    torch.manual_seed(37)
    xt, actual_time, velocity = flow.sample_training_pair(actions)
    torch.testing.assert_close(actual_time, time)
    torch.testing.assert_close(xt, (1-time[:, None, None])*noise + time[:, None, None]*actions)
    torch.testing.assert_close(velocity, actions-noise)
