import copy

import numpy as np
import pytest
import torch

from dexmani_policy.agents.core.lamp_flow import LampFlowAgent
from dexmani_policy.agents.loader import restore_policy_agent
from dexmani_policy.env_runner.base_runner import BaseRunner


def tiny_agent():
    torch.manual_seed(7)
    config = dict(
        horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
        prior_path="/missing/lamp-prior.pt", latent_path="/missing/lamp-labels.pt",
        history_steps=3, latent_dim=2, prior_hidden_dim=16,
        pc_dim=3, num_points=16, state_out_dim=8,
        pc_encoder_config=dict(stem_channels=8, token_channels=16, num_patches=4,
                               patch_radii=[1.0], patch_neighbors=[4]),
        n_layers=1, hidden_dim=32, n_head=4, p_drop_attn=0., num_inference_steps=2,
    )
    model = LampFlowAgent(**config)
    model.normalizer.fit({'action': torch.randn(30, 19), 'joint_state': torch.randn(30, 19)})
    model.set_normalization_spec({'action': 'limits', 'joint_state': 'limits', 'point_cloud': 'identity'})
    model.prior.hand_scale.copy_(torch.linspace(.5, 2., 12))
    model.prior.hand_offset.fill_(.2)
    model.prior.latent_scale.copy_(torch.tensor([.7, 1.3]))
    model.prior.ready.fill_(True)
    model.latent_ready.fill_(True)
    return model, config


def batch_for(model):
    z = torch.randn(2, 4, 2)
    with torch.no_grad():
        hand = model.prior.unnormalize_hand(model.prior.decode(z))
    return dict(
        obs={'point_cloud': torch.randn(2, 2, 16, 3), 'joint_state': torch.randn(2, 2, 19)},
        action_context={'hand_history': torch.randn(2, 3, 12)},
        hand_latent_target=z,
        action=torch.cat((torch.randn(2, 4, 7), hand), dim=-1),
    )


def test_frozen_decoder_backprop_and_optimizer_update():
    model, _ = tiny_agent()
    model.train()
    batch = batch_for(model)
    prior_before = copy.deepcopy(model.prior.state_dict())
    groups = model.get_optim_param_groups(1e-3, 1e-3, 0., 0.)
    optimizer = torch.optim.AdamW(groups)
    model._check_params_in_optimizer(optimizer)
    _, metrics = model.compute_loss(batch)
    metrics['loss_decoded_hand'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.model.parameters())
    assert all(p.grad is None for p in model.prior.parameters())
    assert not model.prior.training
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    before = {k: p.clone() for k, p in model.action_decoder.model.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in model.action_decoder.model.named_parameters())
    for key, expected in prior_before.items():
        torch.testing.assert_close(model.prior.state_dict()[key], expected, rtol=0, atol=0)


def test_latent_weight_changes_only_latent_supervision():
    model, _ = tiny_agent()
    model.eval()
    batch = batch_for(model)
    torch.manual_seed(31)
    original, original_metrics = model.compute_loss(batch)
    model.latent_loss_weight = 0.1
    torch.manual_seed(31)
    reduced, metrics = model.compute_loss(batch)
    for key in ('loss_arm', 'loss_latent', 'loss_decoded_hand'):
        torch.testing.assert_close(metrics[key], original_metrics[key], rtol=0, atol=0)
    # Compare parameter gradients as well: decoder supervision must stay connected.
    removed = 0.9 * model.hand_dim / model.action_dim * metrics['loss_latent']
    torch.testing.assert_close(original - reduced, removed)
    parameters = list(model.action_decoder.model.parameters())
    expected = torch.autograd.grad(original, parameters)
    actual = torch.autograd.grad(reduced + removed, parameters)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_latent_weight(weight):
    _, config = tiny_agent()
    with pytest.raises(ValueError, match='latent_loss_weight'):
        LampFlowAgent(**config, latent_loss_weight=weight)


def test_physical_actions_and_full_checkpoint_without_external_assets(tmp_path):
    model, config = tiny_agent()
    model.eval()
    batch = batch_for(model)
    with torch.no_grad():
        mu, _ = model.prior.encode_history(batch['action_context']['hand_history'])
        offsets = (batch['hand_latent_target'] - mu[:, None]) / model.prior.latent_scale
        hand = model.prior.unnormalize_hand(model._decode_hand(offsets, mu))
        torch.testing.assert_close(hand, batch['action'][..., 7:])
    torch.manual_seed(21)
    result = model.predict_action(batch['obs'], action_context=batch['action_context'])
    assert result['pred_action'].shape == (2, 4, 19)
    torch.testing.assert_close(result['control_action'], result['pred_action'][:, 1:3])
    torch.testing.assert_close(result['tail'], result['pred_action'][:, 3:])
    assert torch.isfinite(result['pred_action']).all()
    saved = dict(action_key='action', agent=dict(
        _target_='dexmani_policy.agents.core.lamp_flow.LampFlowAgent', **config),
        normalization=dict(model.normalization_spec), dataset={'pc_dim': 3})
    checkpoint = tmp_path / 'checkpoints' / 'policy.pt'
    checkpoint.parent.mkdir()
    torch.save(dict(_format='simple.v4', effective_config=saved,
                    weights={'model': model.state_dict(), 'ema_model': model.state_dict()},
                    state={'global_step': 1}), checkpoint)
    for use_ema in (False, True):
        restored = restore_policy_agent(saved, checkpoint, use_ema=use_ema, device='cpu')
        torch.manual_seed(21)
        actual = restored.predict_action(batch['obs'], action_context=batch['action_context'])
        torch.testing.assert_close(actual['pred_action'], result['pred_action'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='hand_history'):
        model.predict_action(batch['obs'])


@pytest.mark.parametrize('key,value', [('prior.latent_scale', 0.),
                                     ('prior.hand_offset', float('nan')),
                                     ('prior.ready', False), ('latent_ready', False)])
def test_restore_rejects_invalid_embedded_prior(key, value):
    model, config = tiny_agent()
    state = copy.deepcopy(model.state_dict())
    state[key].fill_(value)
    with pytest.raises(ValueError, match='LAMP'):
        LampFlowAgent(**config).load_state_dict(state)


class HistoryAgent(torch.nn.Module):
    history_steps, hand_dim, action_dim = 3, 12, 19

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.histories = []

    def predict_action(self, obs_dict, *, action_context, **kwargs):
        self.histories.append(action_context['hand_history'].clone())
        return {'control_action': torch.full((1, 2, 19), 100.)}


class ClippingEnv:
    def reset(self, **kwargs):
        self.action_cnt = 0
        return {'joint_state': np.full(19, -5., dtype=np.float32)}, {}

    def step(self, action):
        self.action_cnt += 1
        executed = np.full(19, self.action_cnt, dtype=np.float32)
        # Measured state differs from the executed command, as under tracking lag.
        return ({'joint_state': executed - .5}, 0., self.action_cnt == 3, False,
                {'executed_joint_action': executed})


def test_runner_history_uses_only_executed_commands_and_resets():
    runner = BaseRunner(2, 1, sensor_modalities=['joint_state'])
    agent = HistoryAgent()
    for seed in (1, 2):
        runner.run_one_episode(agent, ClippingEnv(), seed)
    for first, second in (agent.histories[:2], agent.histories[2:]):
        torch.testing.assert_close(first, torch.full((1, 3, 12), -5.))
        torch.testing.assert_close(second[0, :, 0], torch.tensor([-5., 1., 2.]))
