"""Run decoder checks: python -m dexmani_policy.test_tools.decoder_checks."""

import argparse
import io
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import ConsistencyDiTX, CoupledDiTX, DiTX
from dexmani_policy.agents.action_decoders.backbones.one_way_transformer import OneWayTransformer
from dexmani_policy.agents.action_decoders.sampling.time_sampling import TimeSampler, euler_time_grid
from dexmani_policy.agents.hand_prior.action_io import SATActionIO
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.agents.policies.dp3 import DP3
from dexmani_policy.training.common.optimization import build_optimizer


def check_time(device):
    grid = euler_time_grid(1024, torch.empty((), device=device, dtype=torch.bfloat16))
    assert grid.dtype == torch.float32 and grid.unique().numel() == 1024
    assert grid[0] == 0 and grid[-1] < 1 and (grid[1:] > grid[:-1]).all()
    sampler = TimeSampler(10)
    rng = torch.random.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device)
    expected = 0.999 * torch.distributions.Beta(
        torch.tensor(1.0, device=device), torch.tensor(1.5, device=device),
    ).sample(torch.Size((128,)))
    if device.type == "cpu":
        torch.random.set_rng_state(rng)
    else:
        torch.cuda.set_rng_state(rng, device)
    assert torch.equal(expected, sampler.sample(128, "beta", device))
    for value in (0, -1, float("nan"), float("inf"), 1e-99, 1e99):
        try:
            TimeSampler(10, beta_alpha=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid FP32 beta concentration: {value}")


def check_precision(device):
    models = (
        (DiTX(16, 19, 2, 256, hidden_dim=32, n_head=4, n_layers=1, p_drop_attn=0.0,
              qkv_bias=True, qk_norm=True, action_io=SATActionIO(16, 32)),
         "ditx_blocks", (2, 16, 19), (2, 194, 256)),
        (DiT(4, 3, 16, n_emb=32, num_heads=4, n_layers=1), "blocks", (2, 4, 3), (2, 16)),
        (ConsistencyDiTX(4, 3, 2, 16, hidden_dim=32, n_head=4, n_layers=1),
         "ditx_blocks", (2, 4, 3), (2, 4, 16)),
    )
    for model, blocks, action_shape, context_shape in models:
        model.to(device)
        seen = []
        hook = getattr(model, blocks)[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].dtype))
        kwargs = dict(x=torch.randn(action_shape, device=device), timestep=torch.rand(2, device=device),
                      context=torch.randn(context_shape, device=device))
        if isinstance(model, ConsistencyDiTX):
            kwargs["target_t"] = torch.rand(2, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(**kwargs)
        hook.remove()
        assert seen == [torch.bfloat16] and torch.isfinite(output).all()


def check_decoder_precision(device):
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion
    from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.1, device=device))

        def prepare_context_kv(self, context):
            return None

        def forward(self, x, timestep, context, **kwargs):
            self.input = x.detach()
            return x * self.gain.to(x.dtype)

    cond = torch.zeros(2, 4, device=device, dtype=torch.bfloat16)
    shape = (2, 4, 3)
    for kind in ("sample", "epsilon", "v_prediction"):
        model = Head()
        decoder = Diffusion(model, prediction_type=kind)
        for dtype in (torch.float32, torch.bfloat16):
            actions = torch.zeros(shape, device=device, dtype=dtype)
            noise = torch.ones_like(actions)
            t = torch.zeros(2, device=device, dtype=torch.long)
            with patch("torch.randn_like", return_value=noise), patch("torch.randint", return_value=t):
                loss, _ = decoder.compute_loss(cond, actions)
            a = decoder.noise_scheduler.alphas_cumprod[0].float()
            expected = ((1 - a).sqrt() * noise).to(dtype)
            torch.testing.assert_close(model.input, expected, rtol=0, atol=0)
            assert model.input.abs().min() > 0
            target = actions if kind == "sample" else noise if kind == "epsilon" else (a.sqrt() * noise).to(dtype)
            torch.testing.assert_close(loss, F.mse_loss((expected * model.gain.to(dtype)).float(), target.float()))
            loss.backward()
            assert model.gain.grad is not None and torch.isfinite(model.gain.grad)

    for sampling_decoder in (Diffusion(Head()), RectifiedFlow(Head()), ConsistencyFlow(Head())):
        sampling_decoder.eval()
        def sample(**kwargs):
            return sampling_decoder.predict_action(cond, shape, 4,
                generator=torch.Generator(device=device).manual_seed(10), **kwargs)
        with torch.inference_mode():
            ordinary = sample()
            assert ordinary.dtype == torch.float32
            torch.testing.assert_close(ordinary, sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=0), rtol=0, atol=0)
            guided = sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=2)
            assert guided.dtype == torch.float32 and torch.isfinite(guided).all()


def check_token_decoders(device):
    coupled = CoupledDiTX(
        horizon=8, action_dim=19, n_obs_steps=2, obs_token_dim=16,
        hidden_dim=32, n_head=4, n_layers=2, p_drop_attn=0, interaction_horizon=3,
    ).to(device).eval()
    # Zero-initialized gates and readout would hide a broken attention mask.
    for block in coupled.ditx_blocks:
        torch.nn.init.normal_(block.adaln_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(coupled.final_layer.ffn_final.fc2.weight, std=0.1)
    actions = torch.randn(2, 8, 19, device=device)
    context = torch.randn(2, 18, 16, device=device)
    timestep = torch.rand(2, device=device)
    outside = actions.clone()
    outside[:, [0, 4, 5, 6, 7]] += 3
    changed_body = context.clone()
    changed_body.unflatten(1, (2, 9))[:, -1, :6] += 2
    auxiliary = OneWayTransformer(
        horizon=8, action_dim=28, n_obs_steps=2, obs_token_dim=48, pc_pe_dim=32,
        embedding_dim=32, depth=2, num_heads=4, mlp_dim=64,
        use_aux_ee=True, joint_dim=19, ee_dim=9,
    ).to(device).eval()
    joint_ee = torch.randn(2, 8, 28, device=device)
    changed_ee = joint_ee.clone()
    changed_ee[..., 19:] += 3
    observations = torch.randn(2, 8, 48, device=device)
    for bf16 in (False, True):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity, features = coupled(actions, timestep, context, return_features=True)
            perturbed, perturbed_features = coupled(outside, timestep, context, return_features=True)
            cached_kv = coupled.prepare_context_kv(context)
            cached = coupled(actions, timestep, context, context_kv=cached_kv, return_features=True)
            dynamic = coupled(actions, timestep, changed_body, context_kv=cached_kv, return_features=True)
            prediction = auxiliary(joint_ee, timestep, observations)
            perturbed_ee = auxiliary(changed_ee, timestep, observations)
        torch.testing.assert_close(velocity[:, 1:4], perturbed[:, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(features[:, 8:], perturbed_features[:, 8:], rtol=0, atol=0)
        assert (velocity[:, 4:] - perturbed[:, 4:]).abs().sum() > 0
        torch.testing.assert_close(cached, (velocity, features), rtol=0, atol=0)
        assert (dynamic[1][:, 8:] - features[:, 8:]).abs().sum() > 0
        torch.testing.assert_close(prediction[..., :19], perturbed_ee[..., :19], rtol=0, atol=0)
        assert (prediction[..., 19:] - perturbed_ee[..., 19:]).abs().sum() > 0


def check_ddim_clipping(device):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from dexmani_policy.common.normalizer import FieldNormalizer

    limits = FieldNormalizer(torch.tensor([0.5], device=device), torch.tensor([-2.], device=device))
    physical = torch.tensor([[[2.], [3.], [5.], [6.]]], device=device)
    bounded = limits.normalize(physical)
    outside = torch.tensor([[[-2.], [-0.5], [0.5], [2.]]], device=device)
    for clip in (True, False):
        scheduler = DDIMScheduler(num_train_timesteps=10, prediction_type="sample", clip_sample=clip)
        scheduler.set_timesteps(2, device=device)
        def clean_sample(value):
            return scheduler.step(value, scheduler.timesteps[0], torch.zeros_like(value)).pred_original_sample
        torch.testing.assert_close(limits.unnormalize(clean_sample(bounded)), physical, rtol=0, atol=0)
        expected = outside.clamp(-1, 1) if clip else outside
        torch.testing.assert_close(clean_sample(outside), expected, rtol=0, atol=0)


def check_policies(device):
    boundary = Policy(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 3, 19)
    assert boundary.action_start_index == 1 and boundary.future_steps == 3
    assert Policy(torch.nn.Identity(), torch.nn.Identity(), 1, 1, 1, 19).future_steps == 1
    for horizon, obs_steps, action_steps in ((4, 2, 4), (4, 5, 1), (0, 1, 1), (4, 0, 1), (4, 2, 0)):
        try:
            Policy(torch.nn.Identity(), torch.nn.Identity(), horizon, obs_steps, action_steps, 19)
        except ValueError:
            pass
        else:
            raise AssertionError("Policy accepted an invalid action window")
    kwargs: dict[str, Any] = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, encoder_type="dp3",
                  pc_dim=3, pc_out_dim=16, state_dim=19, num_points=64, state_out_dim=8,
                  down_dims=(32, 64), diffusion_step_embed_dim=32, n_groups=8,
                  num_training_steps=10, num_inference_steps=2)
    policy = DP3(**kwargs)
    spec = dict(joint_state="limits", point_cloud="identity", action="limits")
    dims = dict(joint_state=19, point_cloud=3, action=19)
    policy.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    policy.to(device)
    with torch.no_grad():
        policy.normalizer["action"].scale.copy_(torch.linspace(0.5, 1.5, 19, device=device))
        policy.normalizer["action"].offset.copy_(torch.linspace(-0.2, 0.2, 19, device=device))
    obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
               point_cloud=torch.rand(2, 2, 64, 3, device=device))
    optimizer = build_optimizer(policy, lr=1e-4, weight_decay=1e-6)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = policy.compute_loss(dict(obs=obs, action=torch.randn(2, 8, 19, device=device)))
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    policy.eval()
    prefix = torch.full((2, 3, 19), 0.1234567, device=device)
    with patch.object(policy, "_encode_obs", return_value=torch.zeros(2, 4, device=device, dtype=torch.bfloat16)), \
         patch.object(policy.action_decoder, "predict_action", return_value=torch.zeros(2, 8, 19, device=device)) as sample:
        policy.predict_action(obs, rtc_prefix=prefix, rtc_guidance_cap=2)
    normalized_prefix = sample.call_args.kwargs["rtc_prefix"]
    assert normalized_prefix.dtype == torch.float32
    torch.testing.assert_close(normalized_prefix, policy.normalizer["action"].normalize(prefix), rtol=0, atol=0)
    with torch.inference_mode():
        ordinary = policy.predict_action(
            obs, generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        zero = policy.predict_action(
            obs, rtc_prefix=ordinary[:, :3], rtc_guidance_cap=0,
            generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        guided = policy.predict_action(obs, rtc_prefix=ordinary[:, :3], delay_steps=1,
                                       rtc_guidance_cap=2)["future_actions"]
    assert ordinary.shape == zero.shape == guided.shape == (2, 7, 19)
    torch.testing.assert_close(ordinary, zero, rtol=0, atol=0)
    assert torch.isfinite(guided).all()
    storage = io.BytesIO()
    torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict()), storage)
    storage.seek(0)
    saved = torch.load(storage, map_location=device, weights_only=True)
    restored = DP3(**kwargs)
    restored.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    restored.load_state_dict(saved["model"], strict=True, assign=True)
    restored.to(device)
    new_optimizer = build_optimizer(restored, lr=1e-4, weight_decay=1e-6)
    assert [g["param_names"] for g in optimizer.param_groups] == [g["param_names"] for g in new_optimizer.param_groups]
    new_optimizer.load_state_dict(saved["optimizer"])
    actions = torch.randn(2, 8, 19, device=device)
    for model, opt in ((policy, optimizer), (restored, new_optimizer)):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(29)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            resumed_loss, _ = model.compute_loss(dict(obs=obs, action=actions))
        resumed_loss.backward()
        opt.step()
    for value, replay in zip(policy.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(value, replay, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_time, check_precision, check_decoder_precision, check_token_decoders, check_ddim_clipping, check_policies,):
        check(args.device)
        print(f"PASS {check.__name__} ({args.device})", flush=True)


if __name__ == "__main__":
    main()
