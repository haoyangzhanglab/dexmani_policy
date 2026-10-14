import hashlib
import io

import pytest
import torch

from dexmani_policy.agents.hand_prior.temporal_codec import TemporalHandCodec, load_codec


def small_codec(**overrides):
    options = dict(hand_dim=2, horizon=4, latent_dim=2, hidden_dim=8,
                   n_layers=2, n_heads=2, mlp_ratio=2)
    options.update(overrides)
    torch.manual_seed(9)
    return TemporalHandCodec(**options)


def test_relative_standardization_q0_token_and_physical_decode():
    codec = small_codec()
    codec.action_mean.copy_(torch.tensor([0.3, -0.5]))
    codec.action_std.copy_(torch.tensor([2., 4.]))
    q0 = torch.tensor([[1., 2.], [3., 4.]])
    action = torch.randn(2, 4, 2)
    expected = (action - q0[:, None] - codec.action_mean) / codec.action_std
    torch.testing.assert_close(codec.normalize_actions(action, q0), expected)
    torch.testing.assert_close(codec.normalize_actions(action + 3, q0 + 3), expected)
    seen = {}
    handles = [layer.register_forward_pre_hook(
        lambda module, inputs, name=name: seen.__setitem__(name, inputs[0].detach().clone()))
        for name, layer in (("encoder", codec.encoder_q0), ("decoder", codec.decoder_q0))]
    mu, log_var = codec.encode(action, q0)
    relative = codec.decode(mu, q0)
    raw = codec.decode_actions(mu, q0)
    for handle in handles:
        handle.remove()
    assert mu.shape == log_var.shape == (2, 4, 2)
    torch.testing.assert_close(seen["encoder"], q0)
    torch.testing.assert_close(seen["decoder"], q0)
    torch.testing.assert_close(raw, relative * codec.action_std + codec.action_mean + q0[:, None])


def test_target_conditioned_encoder_and_frozen_decoder_are_bidirectional():
    codec = small_codec().freeze()
    actions = torch.randn(2, 4, 2, requires_grad=True)
    q0 = torch.randn(2, 2)
    mu, _ = codec.encode(actions, q0)
    (encoder_gradient,) = torch.autograd.grad(mu[:, 0].square().sum(), actions)
    assert encoder_gradient[:, -1].abs().sum() > 1e-7
    z = mu.detach().requires_grad_()
    output = codec.decode(z, q0)
    (decoder_gradient,) = torch.autograd.grad(output[:, 0].square().sum(), z)
    assert decoder_gradient[:, -1].abs().sum() > 1e-7
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in codec.parameters())
    altered_q0 = codec.decode(z.detach(), q0 + 0.5)
    assert not torch.allclose(output.detach(), altered_q0)


def test_weighted_huber_uses_joint_count_denominator_and_temporal_differences():
    codec = small_codec(horizon=3)
    codec.joint_weights.copy_(torch.tensor([2., 0.5]))
    pred = torch.tensor([[[0., 1.], [0., 2.], [0., 4.]]], requires_grad=True)
    target = torch.zeros_like(pred)
    terms = codec.reconstruction_terms(pred, target)
    assert float(terms["position"].detach()) == pytest.approx(2.75 / 6)
    assert float(terms["velocity"].detach()) == pytest.approx(1. / 4)
    assert float(terms["acceleration"].detach()) == pytest.approx(0.25 / 2)
    loss = codec.reconstruction_loss(pred, target)
    assert float(loss.detach()) == pytest.approx(2.75 / 6 + 0.2 / 4 + 0.05 * 0.25 / 2)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    codec.joint_weights.mul_(2)
    torch.testing.assert_close(codec.reconstruction_loss(pred.detach(), target), loss.detach() * 2)


def test_deterministic_posterior_and_decoder_across_train_eval_modes():
    codec = small_codec()
    actions, q0 = torch.randn(2, 4, 2), torch.randn(2, 2)
    codec.train()
    mu_train, log_var_train = codec.encode(actions, q0)
    decoded_train = codec.decode(mu_train, q0)
    codec.eval()
    mu_eval, log_var_eval = codec.encode(actions, q0)
    torch.testing.assert_close(mu_train, mu_eval)
    torch.testing.assert_close(log_var_train, log_var_eval)
    torch.testing.assert_close(decoded_train, codec.decode(mu_eval, q0))
    assert all(module.p == 0 for module in codec.modules() if isinstance(module, torch.nn.Dropout))


def test_full_horizon_contract_and_second_difference_requirement():
    codec = small_codec()
    with pytest.raises(ValueError, match="at least 3"):
        small_codec(horizon=2)
    with pytest.raises(ValueError, match="actions must have shape"):
        codec.encode(torch.zeros(1, 3, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="z must have shape"):
        codec.decode(torch.zeros(1, 3, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="q0 must have shape"):
        codec.encode(torch.zeros(1, 4, 2), torch.zeros(1, 1, 2))


def test_artifact_and_tensor_only_restore_after_external_file_removal(tmp_path):
    codec = small_codec()
    codec.ready.fill_(True)
    codec.action_mean.copy_(torch.tensor([0.3, -0.5]))
    codec.action_std.copy_(torch.tensor([2., 4.]))
    codec.joint_weights.copy_(torch.tensor([1., 0.5]))
    path = tmp_path / "codec.pt"
    torch.save({"_format": "temporal_hand_codec.v1", "model_config": codec.model_config,
                "state_dict": codec.state_dict(), "metadata": {"source": "fixture"}}, path)
    restored, payload = load_codec(path)
    assert payload["_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "joint_weights" not in restored.model_config
    assert all(torch.is_tensor(value) for value in restored.state_dict().values())
    state = io.BytesIO()
    torch.save(restored.state_dict(), state)
    state.seek(0)
    embedded = TemporalHandCodec(**restored.model_config)
    embedded.load_state_dict(torch.load(state, weights_only=True))
    embedded.validate_state()
    path.unlink()
    actions, q0 = torch.randn(2, 4, 2), torch.randn(2, 2)
    mu, log_var = codec.encode(actions, q0)
    for candidate in (restored, embedded):
        torch.testing.assert_close(candidate.encode(actions, q0), (mu, log_var))
        torch.testing.assert_close(candidate.decode_actions(mu, q0), codec.decode_actions(mu, q0))


@pytest.mark.parametrize("mutation", ["not_ready", "zero_std", "negative_weight", "nan_model"])
def test_load_rejects_unusable_codec_state(tmp_path, mutation):
    codec = small_codec()
    codec.ready.fill_(True)
    if mutation == "not_ready":
        codec.ready.fill_(False)
    elif mutation == "zero_std":
        codec.action_std.zero_()
    elif mutation == "negative_weight":
        codec.joint_weights[0] = -1
    else:
        with torch.no_grad():
            codec.fc_mu.weight.fill_(float("nan"))
    path = tmp_path / "codec.pt"
    torch.save({"_format": "temporal_hand_codec.v1", "model_config": codec.model_config,
                "state_dict": codec.state_dict()}, path)
    with pytest.raises(ValueError):
        load_codec(path)
