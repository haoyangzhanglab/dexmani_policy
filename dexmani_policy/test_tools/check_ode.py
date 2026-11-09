"""Targeted ODE checks; optional --upstream points to the pinned official source."""

import argparse
import ast
import copy
import shutil
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.ode import ODE
from dexmani_policy.agents.policies.base import Policy, parameter_groups
from dexmani_policy.agents.policies.ode import ODEPolicy
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.inference import prepare_observations
from dexmani_policy.common.normalizer import FieldNormalizer
from dexmani_policy.datasets.variants.ode import ODEDataset
from dexmani_policy.deployment import load_policy
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.test_tools.smoke_test import make_source, run_smoke
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.common.optimization import build_optimizer


def check_data(root):
    path = root / 'rows.zarr'
    source = make_source(path)
    for key, offset in [('action', 100), ('joint_state', 1000)]:
        source[f'data/{key}'][:] = np.broadcast_to(np.arange(12)[:, None] + offset, (12, 19))
    for n in (1, 2):
        dataset = ODEDataset(
            str(path), sensor_modalities=['joint_state', 'point_cloud'], obs_horizon=n,
            horizon=n - 1 + 8, pad_before=n - 1, pad_after=7,
            augmentation_cfg={'state': {'noise': {'noise_std': 0.1, 'prob': 1.0}}},
        )
        assert len(dataset) == 12
        for index in range(12):
            sample = dataset[index]
            row = index
            start, end = (0, 6) if row < 6 else (6, 12)
            expected = 1000 + row if row == start else 100 + row - 1
            torch.testing.assert_close(sample['prev_action'], torch.full((19,), float(expected)))
            targets = sample['action'][n - 1:]
            rows = np.minimum(np.arange(row, row + 8), end - 1)
            torch.testing.assert_close(targets, torch.tensor(rows + 100, dtype=torch.float32)[:, None].expand(-1, 19))
            assert not torch.equal(sample['obs']['joint_state'][-1], torch.full((19,), float(1000 + row)))
    print('PASS data: N=1/2, both episode starts, raw state under augmentation, right padding')


def official_core(path):
    # Extract only the official continuous core, never instantiate the image/gripper policy.
    tree = ast.parse(path.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    policy = classes['ODEImagePolicy']
    methods = {node.name: node for node in policy.body if isinstance(node, ast.FunctionDef)}
    loss = methods['compute_loss']
    selected = []
    for node in loss.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'pose_loss', 'side_loss'}:
            selected.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'self.side_k':
            selected.append(node)
    fn = ast.parse('def loss(self, a0, s, c, target):\n    traj = self.integrate(a0, s, c, self.n_action_steps)\n').body[0]
    fn.body += selected + ast.parse('return pose_loss + side_loss').body
    reference = ast.ClassDef(name='Reference', bases=[ast.parse('nn.Module', mode='eval').body], keywords=[],
                             body=[methods[name] for name in ('_f', 'integrate', '_start_feat', '_side')] + [fn], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[classes['_FiLMLayer'], classes['VectorField'], reference], type_ignores=[]))
    scope = dict(torch=torch, nn=nn, F=F, POSE_DIM=6)
    exec(compile(module, str(path), 'exec'), scope)
    return scope


def check_core(upstream):
    torch.manual_seed(17)
    namespace = official_core(upstream) if upstream else None
    for solver in ('euler', 'rk4'):
        for side in (0, 12):
            decoder = ODE(7, 19, 19, 8, cond_dim=9, hidden_dim=16, num_layers=2,
                          solver=solver, side_head=side, perturb_prob=0)
            with torch.no_grad():
                for p in decoder.parameters():
                    p.normal_(0, 0.08)
            condition = torch.randn(2, 7, requires_grad=True)
            a0, state, target = torch.randn(2, 19), torch.randn(2, 19), torch.randn(2, 8, 19)
            c = decoder.cond_net(condition)
            film = decoder.model.precompute(c)
            loss, metrics = decoder.compute_loss(condition, a0, state, target, perturb_std=0.01)
            assert torch.isfinite(loss) and all(not v.requires_grad for v in metrics.values())
            assert decoder.compute_loss(condition, a0, state, target, perturb_std=0.01, log_metrics=False)[1] == {}
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
            if namespace:
                ref = namespace['Reference']()
                ref.field = namespace['VectorField'](9, 19, 16, 2, 19, 19)
                ref.field.load_state_dict(decoder.model.state_dict())
                ref.cond_net = copy.deepcopy(decoder.cond_net)
                ref.pose_dim, ref.n_action_steps, ref.substeps = 19, 8, 1
                ref.grip_ode, ref.solver, ref.vel_scale = False, solver, decoder.vel_scale
                ref.side_k, ref.spread_min, ref.mix_min = side, decoder.spread_min, decoder.mix_min
                if side:
                    ref.side_head = copy.deepcopy(decoder.side_head)
                rc = ref.cond_net(condition.detach())
                for m in (1, 3):
                    offset = torch.randn(2, 8, 19) * 0.01
                    actual = decoder.integrate(a0, state, film, substeps=m, step_offset=offset)
                    expected = ref.integrate(a0, state, rc, 8, substeps=m, step_offset=offset)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                reference_loss = ref.loss(a0, state, rc, target)
                torch.testing.assert_close(loss, reference_loss, rtol=2e-5, atol=2e-6)
                reference_loss.backward()
                reference_params = dict(ref.named_parameters())
                for name, param in decoder.named_parameters():
                    other = reference_params[name.replace('model.', 'field.', 1)]
                    torch.testing.assert_close(param.grad, other.grad, rtol=3e-5, atol=3e-5)
            decoder.zero_grad(set_to_none=True)
            condition.grad = None
            if side:
                # Remove only the main loss to backpropagate the actual diversity branch alone.
                with patch('torch.nn.functional.mse_loss', return_value=torch.tensor(0.)):
                    side_loss, _ = decoder.compute_loss(condition, a0, state, target, perturb_std=0.01)
                side_loss.backward()
                for module in (decoder.model, decoder.cond_net):
                    assert all(p.grad is None for p in module.parameters())
                assert condition.grad is None
                assert sum(p.grad.abs().sum() for p in decoder.side_head.parameters()) > 0
                _, logw, mean, width = decoder._side(a0, state, film)
                for center in (True, False):
                    g = torch.Generator().manual_seed(29)
                    before = torch.get_rng_state()
                    eps = decoder._draw_eps(logw, mean, width, center=center, generator=g)
                    assert torch.equal(before, torch.get_rng_state())
                    expected_g = torch.Generator().manual_seed(29)
                    k = torch.multinomial(logw.exp(), 1, generator=expected_g)[:, 0]
                    expected = mean[torch.arange(2), k]
                    if not center:
                        expected = expected + width[torch.arange(2), k] * torch.randn((2, 19), generator=expected_g)
                    torch.testing.assert_close(eps, expected, rtol=0, atol=0)
                    assert torch.equal(g.get_state(), expected_g.get_state())
            else:
                assert not hasattr(decoder, 'side_head')
            with patch.object(decoder.model, 'precompute', wraps=decoder.model.precompute) as precompute, patch.object(decoder.model, 'forward', wraps=decoder.model.forward) as field:
                result = decoder.predict_action(condition, a0, state, 3, generator=torch.Generator().manual_seed(1))
                assert result.shape == (2, 8, 19)
                assert precompute.call_count == 1
                assert field.call_count == 8 * 3 * (1 if solver == 'euler' else 4) + bool(side)
    print('PASS core: rollout/loss/gradients' + (' against supplied upstream' if upstream else ' (official comparison skipped)'))
    print('PASS diversity: detached gradients, generator-only draws, component centres, FiLM reuse/NFE')


def small_policy(**kwargs):
    return ODEPolicy(9, 2, 8, 19, num_points=8, pc_out_dim=8, cond_dim=9, hidden_dim=16, num_layers=2, **kwargs)


def check_normalization_perturbation():
    policy = small_policy()
    policy.configure_normalizer({'action': 19, 'joint_state': 19, 'point_cloud': 6},
                                {'action': (19,), 'joint_state': (19,), 'point_cloud': (6,)})
    policy.normalizer.fields['action'] = FieldNormalizer(torch.linspace(2, 4, 19), torch.ones(19) * 3)
    policy.normalizer.fields['joint_state'] = FieldNormalizer(torch.ones(19) * 7, torch.ones(19) * -2)
    obs = {'point_cloud': torch.randn(2, 2, 8, 6), 'joint_state': torch.randn(2, 2, 19)}
    batch = {'obs': obs, 'action': torch.randn(2, 9, 19), 'prev_action': torch.randn(2, 19)}
    with patch.object(policy.action_decoder, 'compute_loss', return_value=(torch.tensor(0.), {})) as call:
        policy.compute_loss(batch)
        _, a0, s, target = call.call_args.args
        torch.testing.assert_close(a0, policy.normalizer['action'].normalize(batch['prev_action']))
        torch.testing.assert_close(s, policy.normalizer['joint_state'].normalize(obs['joint_state'][:, -1]))
        torch.testing.assert_close(target, policy.normalizer['action'].normalize(batch['action'][:, 1:9]))
        torch.testing.assert_close(call.call_args.kwargs['perturb_std'], 0.01 * policy.normalizer['action'].scale)
    # Zero output field preserves a0, including values outside fitted [-1,1] limits.
    for prev in (None, batch['prev_action'] * 10):
        prediction = policy.predict_action(obs, prev_action=prev)['future_actions']
        expected = obs['joint_state'][:, -1] if prev is None else prev
        torch.testing.assert_close(prediction, expected[:, None].expand(-1, 8, -1))
    decoder = policy.action_decoder
    condition, state = policy._condition_state(obs)
    std = policy.perturb_std * policy.normalizer['action'].scale
    with patch('torch.rand', return_value=torch.tensor([[0.1], [0.9]])), patch('torch.randn_like', return_value=torch.ones_like(a0)), patch.object(decoder, 'integrate', wraps=decoder.integrate) as rollout:
        loss, _ = decoder.compute_loss(condition, a0, state, target, perturb_std=std)
        delta = torch.stack([std, torch.zeros_like(std)])
        torch.testing.assert_close(rollout.call_args.args[0], a0 + delta)
        expected_target = target + delta[:, None] * (0.7 ** torch.arange(1, 9))[None, :, None]
        expected = F.mse_loss((a0 + delta)[:, None].expand(-1, 8, -1), expected_target) / 0.05 ** 2
        torch.testing.assert_close(loss, expected)
    decoder.eval()
    rng = torch.get_rng_state()
    decoder.compute_loss(condition, a0, state, target, perturb_std=std)
    assert torch.equal(rng, torch.get_rng_state())
    with np.testing.assert_raises(NotImplementedError):
        policy.predict_action(obs, rtc_prefix=torch.zeros(2, 1, 19))
    with np.testing.assert_raises(ValueError):
        ODEPolicy(10, 2, 8, 19)
    print('PASS affine separation, target alignment, contraction perturbation, eval RNG, RTC/window rejection')


def check_groups():
    for side in (0, 12):
        agent = small_policy(side_head=side)
        build_optimizer(agent, lr=1e-4, weight_decay=1e-6)
    for recipe, overrides in [
        ('dp3', ['agent.down_dims=[16,32]', 'agent.diffusion_step_embed_dim=16']),
        ('maniflow', ['agent.n_layers=2', 'agent.hidden_dim=32', 'agent.n_head=4']),
    ]:
        cfg = load_policy_config(recipe, ['horizon=4', 'n_obs_steps=2', 'n_action_steps=2', 'agent.num_points=8'] + overrides)
        agent = hydra.utils.instantiate(cfg.agent, _convert_='all')
        old = parameter_groups(agent.action_decoder.model, lr=1e-4, weight_decay=1e-6)
        old += parameter_groups(agent.obs_encoder, lr=2e-4, weight_decay=2e-6)
        new = agent.optimizer_groups(1e-4, 2e-4, 1e-6, 2e-6)
        names = {id(p): name for name, p in agent.named_parameters()}
        def describe(groups):
            return [(g['lr'], g['weight_decay'], [names[id(p)] for p in g['params']]) for g in groups]
        assert describe(old) == describe(new)
    print('PASS optimizer: ODE coverage; Diffusion/Flow ordered global names and group options unchanged')


def check_execution():
    agent = small_policy()
    agent.normalization_dims = {'joint_state': 19}
    seen = []
    def predict(obs, *args, prev_action=None, **kwargs):
        seen.append(None if prev_action is None else prev_action.clone())
        base = 0 if prev_action is None else prev_action[0, 0].item()
        return {'future_actions': torch.arange(base + 1, base + 9).reshape(1, 8, 1).expand(1, 8, 19)}
    class Env:
        def __init__(self, limit, report_action=True):
            self.limit = limit
            self.report_action = report_action
        def reset(self, seed):
            self.action_cnt = 0
            self.truncate = seed == 2
            return {'joint_state': np.zeros(19, np.float32)}, {}
        def step(self, action):
            self.action_cnt += 1
            executed = np.clip(action, -self.limit, self.limit)
            ended = self.action_cnt == 11
            info = {'success': True, 'success_condition': False}
            if self.report_action:
                info['executed_joint_action'] = executed
            return ({'joint_state': np.zeros(19, np.float32)}, 0,
                    ended and not self.truncate, ended and self.truncate, info)
    runner = SimRunner('fake', 2, ['joint_state'], {})
    with patch.object(agent, 'predict_action', side_effect=predict):
        for limit in (np.inf, 4.5):
            seen.clear()
            for seed in (1, 2):
                env = Env(limit)
                runner._run_episode(agent, env, seed, action_steps=8)
                assert env.action_cnt == 11
            assert len(seen) == 4 and seen[0] is None and seen[2] is None
            expected = torch.full((1, 19), min(8., limit))
            torch.testing.assert_close(seen[1], expected)
            torch.testing.assert_close(seen[3], expected)
    with np.testing.assert_raises(ValueError):
        next(runner.run(agent, seeds=[1], inference_steps=1, action_steps=7))
    other = Policy(nn.Identity(), nn.Linear(1, 1), 9, 2, 3, 19)
    other.normalization_dims = {'joint_state': 19}
    env = Env(np.inf, report_action=False)
    with patch.object(other, 'predict_action', return_value={'future_actions': torch.zeros(1, 8, 19)}) as call:
        runner._run_episode(other, env, 1, action_steps=3)
        assert env.action_cnt == 11 and call.call_count == 4
        assert all('prev_action' not in item.kwargs for item in call.call_args_list)
    print('PASS execution: last executed target, episode reset, termination, full ODE chunks; other policies need no carry')


def check_lifecycle(root, device, compile_model):
    for side in (0, 12):
        source_path = root / f'source-{side}.zarr'
        source = make_source(source_path)
        cfg = load_policy_config('ode', [
            f'zarr_path={source_path}', f'training.device={device}', f'agent.side_head={side}',
            'agent.num_points=8', 'agent.pc_out_dim=8', 'agent.cond_dim=16', 'agent.hidden_dim=32', 'agent.num_layers=2',
            f'training.use_compile={str(compile_model).lower()}', 'training.lr_warmup_steps=0',
            'training.loop.total_train_steps=4', 'training.loop.log_interval_steps=1',
            'dataset.augmentation_cfg=null', 'dataloader.batch_size=2', 'dataloader.num_workers=0', 'dataloader.pin_memory=false',
        ])
        run = root / f'run-{side}'
        run_smoke(cfg, run, max_updates=2, resume_updates=2)
        checkpoint = load_checkpoint(run / 'checkpoints/latest.pt')
        assert checkpoint.ema_model_state is not None
        assert not any('_orig_mod' in key for key in checkpoint.model_state)
        observation = {key: source[f'data/{key}'][:2] for key in cfg.dataset.sensor_modalities}
        shutil.rmtree(source_path)
        with closing(load_policy(run, device=device)) as runtime:
            assert runtime.info.requires_prev_action
            previous = np.full(19, 0.3, np.float32)
            saved_previous = previous.copy()
            rng = runtime.generator.get_state()
            eager = runtime.predict(observation, prev_action=previous)
            runtime.generator.set_state(rng)
            if compile_model:
                runtime.agent.compile_backbone(mode='reduce-overhead')
                compiled = runtime.predict(observation, prev_action=previous)
                np.testing.assert_allclose(compiled, eager, rtol=2e-4, atol=2e-5)
                runtime.generator.set_state(rng)
            runtime.warmup(observation)
            assert torch.equal(rng, runtime.generator.get_state())
            np.testing.assert_array_equal(previous, saved_previous)
            output = runtime.predict(observation, prev_action=previous)
            np.testing.assert_allclose(output, eager, rtol=2e-4, atol=2e-5)
            assert output.shape == (8, 19) and np.isfinite(output).all()
            tensors = prepare_observations(observation, normalization_dims=runtime.agent.normalization_dims,
                                           n_obs_steps=2, rgb_preprocessing={}, device=device)
            assert runtime.agent.predict_action(tensors)['future_actions'].shape == (1, 8, 19)
            with np.testing.assert_raises(ValueError):
                runtime.predict(observation, prev_action=np.zeros((1, 19)))
        print(f'PASS lifecycle side_head={side}: updates/EMA/save/resume/checkpoint-only runtime/warmup, compile={compile_model}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix='dexmani-ode-check-') as temporary:
        root = Path(temporary)
        check_data(root)
        check_core(args.upstream)
        check_normalization_perturbation()
        check_groups()
        check_execution()
        check_lifecycle(root, args.device, args.compile)


if __name__ == '__main__':
    main()
