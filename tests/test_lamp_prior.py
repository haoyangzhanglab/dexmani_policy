import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexmani_policy.agents.hand_prior.lamp import HandMotionPrior, file_sha256, load_prior
from dexmani_policy.training.export_lamp_latents import split_diagnostics


def test_frozen_prior_preserves_decoder_input_gradients_and_physical_affine():
    torch.manual_seed(7)
    prior = HandMotionPrior(hand_dim=3, history_steps=2, latent_dim=2, hidden_dim=8).freeze()
    prior.hand_scale.copy_(torch.tensor([2., 3., 4.]))
    prior.hand_offset.copy_(torch.tensor([-1., 2., 1.]))
    raw = torch.randn(4, 2, 3)
    torch.testing.assert_close(prior.unnormalize_hand(prior.normalize_hand(raw)), raw)
    mu, log_var = prior.encode_history(raw)
    assert mu.shape == log_var.shape == (4, 2)
    z = mu.detach().requires_grad_()
    prior.decode(z).square().sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in prior.parameters())


def linear_prior():
    prior = HandMotionPrior(hand_dim=2, history_steps=2, latent_dim=2, hidden_dim=4)
    with torch.no_grad():
        prior.fc_mu.weight.zero_()
        prior.fc_mu.bias.zero_()
    prior.decoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        prior.decoder.weight.copy_(torch.eye(2))
    return prior.freeze()


@pytest.mark.parametrize("center_penalty", [0.0, 1e-4])
def test_inversion_keeps_per_sample_best_including_initial_and_batch_independence(center_penalty):
    prior = linear_prior()
    targets = torch.tensor([[0., 0.], [0.1, -0.1], [2., -3.]])
    histories = torch.zeros(3, 2, 2)
    initial = prior.invert(targets, histories, steps=0, center_penalty=center_penalty)
    torch.testing.assert_close(initial, torch.zeros_like(targets))
    with torch.no_grad():
        batch = prior.invert(targets, histories, steps=15, lr=0.5, center_penalty=center_penalty)
    if center_penalty == 0:
        default = prior.invert(targets, histories, steps=15, lr=0.5)
        torch.testing.assert_close(default, batch, atol=0, rtol=0)
    single = torch.cat([prior.invert(targets[i:i+1], histories[i:i+1], steps=15, lr=0.5,
                                    center_penalty=center_penalty)
                        for i in range(3)])
    torch.testing.assert_close(batch, single, atol=1e-7, rtol=1e-6)
    final_error = (prior.decode(batch) - targets).square().mean(-1)
    initial_error = (prior.decode(initial) - targets).square().mean(-1)
    assert (final_error <= initial_error).all()
    assert final_error[-1] < initial_error[-1] * 0.1
    torch.testing.assert_close(batch[0], initial[0])
    assert all(parameter.grad is None for parameter in prior.parameters())


def test_inversion_selects_penalized_optimum_around_source_mean_not_raw_reconstruction():
    prior = linear_prior()
    prior.fc_mu.bias.copy_(torch.tensor([3., -2.]))
    targets = torch.tensor([[5., -5.]])
    histories = torch.zeros(1, 2, 2)
    center, _ = prior.encode_history(histories)
    # D is the identity: the regularized minimizer is known analytically.
    # Large steps visit candidates with better raw reconstruction but worse J.
    anchored = prior.invert(targets, histories, steps=250, lr=0.8, center_penalty=4.)
    expected = center + (targets - center) / 5.
    torch.testing.assert_close(anchored, expected, atol=1e-3, rtol=0)
    unanchored = prior.invert(targets, histories, steps=250, lr=0.8, center_penalty=0.)
    torch.testing.assert_close(unanchored, targets, atol=1e-3, rtol=0)
    assert ((anchored - targets).square().mean(-1)
            <= (center - targets).square().mean(-1)).all()


@pytest.mark.parametrize("center_penalty", [-1., float("nan"), float("inf")])
def test_inversion_rejects_invalid_center_penalty(center_penalty):
    with pytest.raises(ValueError, match="center_penalty"):
        linear_prior().invert(torch.zeros(1, 2), torch.zeros(1, 2, 2),
                              center_penalty=center_penalty)


def test_prior_artifact_and_embedded_tensor_state_restore_without_files(tmp_path):
    torch.manual_seed(8)
    prior = HandMotionPrior(hand_dim=3, history_steps=2, latent_dim=2, hidden_dim=8)
    prior.ready.fill_(True)
    prior.hand_scale.copy_(torch.tensor([2., 3., 4.]))
    prior.latent_scale.copy_(torch.tensor([0.5, 2.]))
    path = tmp_path / "prior.pt"
    torch.save({"_format": "lamp_prior.v1", "model_config": prior.model_config,
                "model_state_dict": prior.state_dict(), "metadata": {}}, path)
    restored, payload = load_prior(path)
    assert payload["_file_sha256"] == file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(torch.is_tensor(value) for value in restored.state_dict().values())
    state = io.BytesIO()
    torch.save(restored.state_dict(), state)
    state.seek(0)
    embedded = HandMotionPrior(**restored.model_config)
    embedded.load_state_dict(torch.load(state, weights_only=True))
    path.unlink()
    history = torch.randn(4, 2, 3)
    for candidate in (restored, embedded):
        torch.testing.assert_close(candidate.encode_history(history), prior.encode_history(history))
        torch.testing.assert_close(candidate.decode(torch.ones(4, 2)), prior.decode(torch.ones(4, 2)))
        assert bool(candidate.ready)


@pytest.mark.parametrize("mutation", ["not_ready", "zero_scale", "nan_weight"])
def test_load_prior_rejects_unusable_artifacts(tmp_path, mutation):
    prior = HandMotionPrior(hand_dim=2, history_steps=2, latent_dim=2, hidden_dim=4)
    prior.ready.fill_(True)
    if mutation == "not_ready":
        prior.ready.fill_(False)
    elif mutation == "zero_scale":
        prior.hand_scale.zero_()
    else:
        with torch.no_grad():
            prior.fc_mu.weight.fill_(float("nan"))
    path = tmp_path / "prior.pt"
    torch.save({"_format": "lamp_prior.v1", "model_config": prior.model_config,
                "model_state_dict": prior.state_dict()}, path)
    with pytest.raises(ValueError):
        load_prior(path)


def test_latent_jump_diagnostics_exclude_episode_boundaries_and_missing_rows():
    dataset = SimpleNamespace(episode_ends=np.array([3, 6]),
                              hand_actions=np.array([[0., 0.], [0., 0.], [1., 1.],
                                                     [1., 1.], [2., 2.], [1., 1.]]))
    rows = np.array([0, 1, 2, 3, 5])
    latents = torch.tensor([[0.], [1.], [2.], [100.], [101.], [102.]])
    errors = torch.ones(6)
    diagnostics = split_diagnostics(dataset, rows, latents, errors, errors * 2, torch.ones(6, 2))
    assert diagnostics["adjacent_same_episode_pairs"] == 2
    assert diagnostics["latent_jump_mean"] == diagnostics["latent_jump_p95"] == 1.
    assert diagnostics["latent_jump_p99"] == diagnostics["latent_jump_max"] == 1.
    assert diagnostics["same_target_adjacent_pairs"] == 1
    assert diagnostics["same_target_latent_jump_mean"] == diagnostics["same_target_latent_jump_p95"] == 1.
    assert diagnostics["same_target_latent_jump_p99"] == diagnostics["same_target_latent_jump_max"] == 1.
    assert diagnostics["reconstruction_nonworse_initial_fraction"] == 1.
    assert diagnostics["max_reconstruction_increase"] == -1.
    assert diagnostics["physical_joint_rmse"] == [1., 1.]
    assert diagnostics["physical_joint_mae"] == [1., 1.]
    singleton = split_diagnostics(dataset, np.array([0]), latents, errors, errors, torch.ones(6, 2))
    assert singleton["same_target_adjacent_pairs"] == singleton["adjacent_same_episode_pairs"] == 0
    assert singleton["latent_jump_max"] is singleton["same_target_latent_jump_max"] is None
