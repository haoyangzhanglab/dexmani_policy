import copy
from unittest.mock import patch

import pytest
import torch

from dexmani_policy.agents.core.temporal_codec_flow import TemporalCodecFlowAgent
from dexmani_policy.agents.loader import restore_policy_agent


def tiny_agent(**overrides):
    torch.manual_seed(7)
    config = dict(
        horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
        codec_path="/missing/temporal-codec.pt", latent_dim=2,
        codec_hidden_dim=16, codec_n_layers=1, codec_n_heads=4,
        pc_dim=3, num_points=16, state_out_dim=8,
        pc_encoder_config=dict(stem_channels=8, token_channels=16, num_patches=4,
                               patch_radii=[1.0], patch_neighbors=[4]),
        n_layers=1, hidden_dim=32, n_head=4, p_drop_attn=0., num_inference_steps=2,
    )
    config.update(overrides)
    model = TemporalCodecFlowAgent(**config)
    model.normalizer.fit({'action': torch.randn(30, 19), 'joint_state': torch.randn(30, 19)})
    model.set_normalization_spec({'action': 'limits', 'joint_state': 'limits', 'point_cloud': 'identity'})
    model.codec.action_mean.copy_(torch.linspace(-.2, .2, 12))
    model.codec.action_std.copy_(torch.linspace(.5, 2., 12))
    model.codec.ready.fill_(True)
    return model, config


def batch_for(model):
    obs = {'point_cloud': torch.randn(2, 2, 16, 3), 'joint_state': torch.randn(2, 2, 19)}
    return dict(obs=obs, action=torch.randn(2, model.horizon, 19),
                action_context={'hand_reference': obs['joint_state'][:, 1, 7:].clone()})


def test_decoded_loss_updates_policy_through_frozen_codec():
    model, _ = tiny_agent()
    model.train()
    batch = batch_for(model)
    before = copy.deepcopy(model.codec.state_dict())
    optimizer = torch.optim.AdamW(model.get_optim_param_groups(1e-3, 1e-3, 0., 0.))
    model._check_params_in_optimizer(optimizer)
    loss, metrics = model.compute_loss(batch)
    torch.testing.assert_close(loss, metrics['loss_arm'] + .25 * metrics['loss_latent']
                               + .75 * metrics['loss_decoded_hand'])
    metrics['loss_decoded_hand'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.model.parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    assert not model.codec.training
    weights = {k: p.clone() for k, p in model.action_decoder.model.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(weights[k], p) for k, p in model.action_decoder.model.named_parameters())
    for key, expected in before.items():
        torch.testing.assert_close(model.codec.state_dict()[key], expected, rtol=0, atol=0)


def test_bc_requires_unaugmented_reference_and_uses_posterior_mean():
    model, _ = tiny_agent()
    batch = batch_for(model)
    reference = batch['action_context']['hand_reference']
    batch['obs']['joint_state'] += 5  # Observation augmentation must not move q0.
    seen = []
    encode = model.codec.encode

    def capture(actions, q0):
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(q0, reference)
        mu, log_var = encode(actions, q0)
        seen.append(mu)
        return mu, log_var

    with patch.object(model.codec, 'encode', side_effect=capture):
        model.compute_loss(batch)
        model.compute_loss(batch)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    del batch['action_context']
    with pytest.raises(ValueError, match='hand_reference'):
        model.compute_loss(batch)


def test_clean_estimate_uses_noise_to_data_time_direction():
    model, _ = tiny_agent()
    batch = batch_for(model)
    reference = batch['action_context']['hand_reference']
    with torch.no_grad():
        mu, _ = model.codec.encode(batch['action'][..., 7:], reference)
        arm = model.normalizer['action'].normalize(batch['action'])[..., :7]
        target = torch.cat((arm, mu), -1)
    noise = torch.randn_like(target)
    time = torch.tensor([.2, .8])
    decode = model.codec.decode

    def check_decode(z, q0):
        torch.testing.assert_close(z, mu, atol=1e-6, rtol=1e-6)
        return decode(z, q0)

    with patch('torch.randn_like', return_value=noise), \
            patch.object(model.action_decoder.time_sampler, 'sample', return_value=time), \
            patch.object(model.action_decoder.model, 'forward', return_value=target - noise), \
            patch.object(model.codec, 'decode', side_effect=check_decode):
        _, metrics = model.compute_loss(batch)
    assert metrics['loss_arm'].item() == 0
    assert metrics['loss_latent'].item() == 0


def test_inference_decodes_full_horizon_without_encoder_and_then_crops():
    model, _ = tiny_agent()
    model.eval()
    batch = batch_for(model)
    native = torch.randn(2, model.horizon, model.model_action_dim)
    reference = batch['action_context']['hand_reference']
    with torch.no_grad():
        expected_hand = model.codec.decode_actions(native[..., 7:], reference)
        affine = model.normalizer['action'].params_dict
        expected_arm = (native[..., :7] - affine['offset'][:7]) / affine['scale'][:7]
    with patch.object(model.action_decoder, 'predict_action', return_value=native), \
            patch.object(model.codec, 'encode', side_effect=AssertionError('No encoder at inference')):
        result = model.predict_action(batch['obs'])
    torch.testing.assert_close(result['pred_action'][..., :7], expected_arm)
    torch.testing.assert_close(result['pred_action'][..., 7:], expected_hand)
    torch.testing.assert_close(result['control_action'], result['pred_action'][:, 1:3])
    torch.testing.assert_close(result['tail'], result['pred_action'][:, 3:])


@pytest.mark.parametrize('space', [None, 'physical'])
def test_policy_checkpoint_embeds_codec_and_restores_raw_and_ema(tmp_path, space):
    # Missing field exercises old checkpoints; explicit space must also survive.
    model, config = tiny_agent(**({} if space is None else {'decoded_loss_space': space}))
    model.eval()
    batch = batch_for(model)
    torch.manual_seed(21)
    expected = model.predict_action(batch['obs'])
    saved = dict(action_key='action', agent=dict(
        _target_='dexmani_policy.agents.core.temporal_codec_flow.TemporalCodecFlowAgent', **config),
        normalization=dict(model.normalization_spec), dataset={'pc_dim': 3})
    checkpoint = tmp_path / 'checkpoints' / 'policy.pt'
    checkpoint.parent.mkdir()
    torch.save(dict(_format='simple.v4', effective_config=saved,
                    weights={'model': model.state_dict(), 'ema_model': model.state_dict()},
                    state={'global_step': 1}), checkpoint)
    for use_ema in (False, True):
        restored = restore_policy_agent(saved, checkpoint, use_ema=use_ema, device='cpu')
        assert restored.decoded_loss_space == (space or 'standardized')
        torch.manual_seed(21)
        actual = restored.predict_action(batch['obs'])
        torch.testing.assert_close(actual['pred_action'], expected['pred_action'], rtol=0, atol=0)


@pytest.mark.parametrize('space', ['standardized', 'physical'])
def test_decoded_metric_matches_physical_command_error_and_gradient(space):
    model, _ = tiny_agent(decoded_loss_space=space)
    batch = batch_for(model)
    reference = batch['action_context']['hand_reference']
    with torch.no_grad():
        mu, _ = model.codec.encode(batch['action'][..., 7:], reference)
    noise = torch.randn(2, model.horizon, model.model_action_dim)
    time = torch.tensor([.2, .8])
    velocity = torch.randn_like(noise, requires_grad=True)
    with patch('torch.randn_like', return_value=noise), \
            patch.object(model.action_decoder.time_sampler, 'sample', return_value=time), \
            patch.object(model.action_decoder.model, 'forward', return_value=velocity):
        _, metrics = model.compute_loss(batch)
    # Reference uses physical commands, independently undoing preprocessing.
    t = time[:, None, None]
    clean = t * mu + (1-t) * (noise[..., 7:] + velocity[..., 7:])
    physical_error = model.codec.decode_actions(clean, reference) - batch['action'][..., 7:]
    expected_error = physical_error if space == 'physical' else physical_error / model.codec.action_std
    expected = expected_error.square().mean()
    torch.testing.assert_close(metrics['loss_decoded_hand'], expected)
    actual_grad, = torch.autograd.grad(metrics['loss_decoded_hand'], velocity)
    expected_grad, = torch.autograd.grad(expected, velocity)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-6)
    assert actual_grad[..., 7:].abs().sum() > 0
    assert not actual_grad[..., :7].any()
    assert all(p.grad is None for p in model.codec.parameters())


def test_decoded_metric_only_changes_auxiliary_supervision():
    model, config = tiny_agent()
    physical = TemporalCodecFlowAgent(**config, decoded_loss_space='physical')
    physical.load_state_dict(model.state_dict())
    physical.set_normalization_spec(model.normalization_spec)
    model.eval()
    physical.eval()
    batch = batch_for(model)
    torch.manual_seed(25)
    _, reference = model.compute_loss(batch)
    torch.manual_seed(25)
    _, candidate = physical.compute_loss(batch)
    for key in ('loss_arm', 'loss_latent', 'loss_action'):
        torch.testing.assert_close(reference[key], candidate[key], rtol=0, atol=0)
    torch.manual_seed(25)
    expected = model.predict_action(batch['obs'])
    torch.manual_seed(25)
    actual = physical.predict_action(batch['obs'])
    torch.testing.assert_close(expected['pred_action'], actual['pred_action'], rtol=0, atol=0)


def test_invalid_decoded_loss_space_is_rejected():
    with pytest.raises(ValueError, match='decoded_loss_space'):
        tiny_agent(decoded_loss_space='unknown')


@pytest.mark.parametrize('key,value', [('codec.action_std', 0.),
                                     ('codec.action_mean', float('nan')),
                                     ('codec.ready', False)])
def test_restore_rejects_invalid_embedded_codec(key, value):
    model, config = tiny_agent()
    state = copy.deepcopy(model.state_dict())
    state[key].fill_(value)
    with pytest.raises(ValueError):
        TemporalCodecFlowAgent(**config).load_state_dict(state)
