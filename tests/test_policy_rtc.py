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


def boundary_policy(mode='joint'):
    class Agent(nn.Module):
        consumed_observation_fields = ("joint_state",)
        def __init__(self):
            super().__init__()
            self.value = None
            self.calls = []
            self.outputs = []
            self.resets = 0
        def reset_episode(self): self.resets += 1
        def predict_action(self, obs, **kwargs):
            self.calls.append(kwargs)
            value = self.value if self.value is not None else torch.randn(1, 4, 24)
            self.outputs.append(value.clone())
            return {'pred_action': value}
    info = SimpleNamespace(observation_fields=('joint_state',), inference_steps=2,
                           action_mode=mode, n_obs_steps=2, n_action_steps=2, horizon=4)
    policy = LoadedPolicy(Agent(), {'dataset': {}}, info, device='cpu', seed=0)
    policy.agent.resets = 0
    return policy


@pytest.mark.parametrize('mode,dimensions', [('joint',19), ('eef',21)])
@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_predict_returns_nonfinite_control_future_for_real_admission(mode, dimensions, bad):
    policy = boundary_policy(mode)
    policy.agent.value = torch.zeros(1,4,24)
    policy.agent.value[0,1,dimensions-1] = bad
    output = policy.predict({'joint_state': np.zeros((2,19),np.float32)})
    assert output.dtype == np.float64 and output.shape == (3, dimensions)
    assert np.isnan(output[0, dimensions-1]) if np.isnan(bad) else output[0, dimensions-1] == bad


@pytest.mark.parametrize('mode,dimensions', [('joint',19), ('eef',21)])
def test_predict_slices_control_future_and_accepts_strides(mode, dimensions):
    policy = boundary_policy(mode)
    value = torch.zeros(1,24,4).transpose(1,2)
    assert not value.is_contiguous()
    value[0,0,:] = float('nan'); value[0,1:,dimensions:] = float('inf')
    policy.agent.value = value
    output = policy.predict({'joint_state': np.zeros((19,2),np.float32).T})
    assert output.shape == (3,dimensions) and output.dtype == np.float64
    assert np.isfinite(output).all()
    policy.agent.value = torch.zeros(1,3,24)
    with pytest.raises(ValueError, match='Policy future'):
        policy.predict({'joint_state': np.zeros((2,19),np.float32)})


@pytest.mark.parametrize('cap,delay', [(0,0),(0,2),(1,2)])
def test_warmup_zero_cap_uses_plain_path(cap, delay):
    policy = boundary_policy(); policy.rtc_guidance_cap = cap
    torch.manual_seed(23)
    durations = policy.warmup(samples=2, rtc_delay=delay)
    assert set(durations) == {'bootstrap', 'steady'} and policy.agent.resets == 2
    assert all(len(group) == 2 for group in durations.values())
    assert len(policy.agent.calls) == (3 if cap == 0 else 6)
    if cap == 0:
        assert durations['steady'] is durations['bootstrap']
        assert all('rtc_prefix' not in call for call in policy.agent.calls)
    else:
        assert all('rtc_prefix' not in call for call in policy.agent.calls[:3])
        guided = policy.agent.calls[3:]
        assert all(call['delay_steps'] == delay for call in guided)
        torch.testing.assert_close(guided[0]['rtc_prefix'], guided[-1]['rtc_prefix'])
    obs = {'joint_state': np.zeros((2,19),np.float32)}
    first = policy.predict(obs)
    policy.warmup(samples=3,rtc_delay=delay)
    np.testing.assert_array_equal(policy.predict(obs), first)


@pytest.mark.parametrize('delay', [-1,4,True,1.5])
def test_warmup_invalid_delay_and_failure_reset(delay):
    policy = boundary_policy()
    with pytest.raises(ValueError, match='rtc_delay'): policy.warmup(samples=1,rtc_delay=delay)
    assert not policy.agent.calls
    policy.agent.value = torch.full((1,4,24),float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'): policy.warmup(samples=1)
    assert policy.agent.resets == 2


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
    consumed_observation_fields = ("joint_state",)
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
        consumed_observation_fields = ("joint_state",)
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
@pytest.mark.parametrize("delay,cap", [(0, 2.0), (3, 2.0), (0, 0.0), (3, 0.0)])
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
    policy.configure_execution(mode, cap if mode == "rtc" else None)
    assert not calls
    durations = policy.warmup(samples=1, rtc_delay=delay if mode == "rtc" else 0)
    assert all(len(group) == 1 and np.isfinite(group[0]) and group[0] >= 0 for group in durations.values())
    assert calls == ([(False, 0)] * 2 + [(True, delay)] * 2 if mode == "rtc" and cap > 0 else [(False, 0)] * 2)
    assert all(p.grad is None for p in agent.parameters())


@pytest.mark.parametrize('cap', [0., 2.])
def test_warmup_initialization_excluded_and_all_outputs_checked(monkeypatch, cap):
    policy = boundary_policy()
    policy.rtc_guidance_cap = cap
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr('time.perf_counter_ns', lambda: clock.now)
    original = policy.predict
    calls = []
    def predict(obs, **kwargs):
        guided = kwargs.get('rtc_prefix') is not None
        clock.now += 1_000_000_000 if guided not in calls else 10_000_000
        calls.append(guided)
        return original(obs, **kwargs)
    policy.predict = predict
    durations = policy.warmup(samples=2, rtc_delay=2)
    assert durations == {'bootstrap': (.01,.01), 'steady': (.01,.01)}
    assert calls == ([False]*3 + [True]*3 if cap else [False]*3)
    for bad_call in range(len(calls)):
        counter = [0]
        def bad_predict(obs, *, _counter=counter, _bad_call=bad_call, **kwargs):
            output = original(obs, **kwargs)
            if _counter[0] == _bad_call:
                output[0,0] = np.nan
            _counter[0] += 1
            return output
        policy.predict = bad_predict
        with pytest.raises(ValueError, match='Nonfinite'):
            policy.warmup(samples=2, rtc_delay=2)


def test_loaded_input_declaration_and_structural_errors():
    from dexmani_policy.utils.validation import validate_observation_fields
    agent = SimpleNamespace(obs_encoder=SimpleNamespace(consumed_observation_fields=('joint_state','point_cloud')))
    with pytest.raises(ValueError, match="missing=.*point_cloud.*unconsumed=.*tactile_force"):
        validate_observation_fields(agent, ('joint_state','tactile_force'))
    validate_observation_fields(agent, ('point_cloud','joint_state'), require_declaration=True)
    with pytest.raises(ValueError, match='consumed_observation_fields'):
        validate_observation_fields(object(), ('joint_state',), require_declaration=True)
    policy = boundary_policy()
    for value in (None, [], {'pred_action': None}, {'pred_action': torch.zeros(4,19)}, {'pred_action': torch.zeros(1,4,19,dtype=torch.int64)}):
        policy.agent.predict_action = lambda *a, _value=value, **kw: _value
        with pytest.raises(ValueError, match='Policy future'):
            policy.predict({'joint_state':np.zeros((2,19),dtype='f4')})



def test_explicit_inspection_without_eval_and_nonlatest_discovery(tmp_path, monkeypatch):
    import yaml

    from dexmani_policy.deployment import runtime
    exp=tmp_path/'policy'/'task'/'run'
    (exp/'checkpoints').mkdir(parents=True)
    (exp/'checkpoints'/'epoch_1.pt').touch()
    cfg={'policy_name':'policy','task_name':'task','action_key':'action',
         'agent':{'horizon':4,'n_obs_steps':1,'n_action_steps':2},
         'dataset':{'sensor_modalities':['joint_state']},'real_runtime':{'dt':.1}}
    (exp/'config.yaml').write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runtime,'_EXPERIMENTS_ROOT',tmp_path)
    assert runtime.list_experiments() == ('policy/task/run',)
    info=runtime.inspect_policy(exp,checkpoint='epoch_1.pt',weights='raw',inference_steps=2)
    assert info.inference_steps==2 and info.weights=='raw'
    with pytest.raises(ValueError, match='use_ema'):
        runtime.inspect_policy(exp,checkpoint='epoch_1.pt',inference_steps=2)
    for section,target in [('agent','dexmani_policy.agents.core.multi_task.MultiTaskAgent'),
                           ('dataset','dexmani_policy.datasets.multi_task_dataset.MultiTaskDataset')]:
        cfg[section]['_target_']=target
        with pytest.raises(NotImplementedError, match='task selection.*child numerical'):
            runtime.inspect_policy(exp,config=cfg,checkpoint='epoch_1.pt',weights='raw',inference_steps=2)
        del cfg[section]['_target_']


@pytest.mark.parametrize('declared', [('joint_state','point_cloud'), ('joint_state',), ('joint_state','point_cloud','tactile_force')])
def test_training_model_boundary_checks_actual_inputs(monkeypatch, declared):
    from omegaconf import OmegaConf
    from test_infra_resume import TinyPolicy

    from dexmani_policy.training.build_utils import build_model_and_ema
    model = TinyPolicy(clip_sample=False)
    model.obs_encoder.consumed_observation_fields = ('joint_state','point_cloud')
    monkeypatch.setattr('hydra.utils.instantiate', lambda *a, **kw: model)
    cfg = OmegaConf.create({'agent':{}, 'dataset':{'sensor_modalities':list(declared)},
        'action_key':'action', 'normalization':{'action':'limits','joint_state':'identity','point_cloud':'identity'},
        'training':{'use_ema':False}})
    normalizer = LinearNormalizer()
    normalizer.fit_field('action', np.array([[-1.], [1.]], dtype='f4'), mode='limits')
    if declared != ('joint_state','point_cloud'):
        with pytest.raises(ValueError, match='Model observation mismatch'):
            build_model_and_ema(cfg, 'cpu', normalizer)
    else:
        actual, ema, _ = build_model_and_ema(cfg, 'cpu', normalizer)
        assert actual is model and ema is None
