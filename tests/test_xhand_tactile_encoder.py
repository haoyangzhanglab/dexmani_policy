"""Single-frame XHand encoder contracts; no policy or simulator dependencies."""

import pytest
import torch
from dexmani_policy.agents.obs_encoder.tactile import XHandForceEncoder, XHandTaxelEncoder


@pytest.mark.parametrize("encoder_type,per_finger_shape", [
    (XHandForceEncoder, (3,)),
    (XHandTaxelEncoder, (120, 3)),
])
def test_shapes_independent_frames_and_gradients(encoder_type, per_finger_shape):
    torch.manual_seed(7)
    encoder = encoder_type()
    x = torch.randn(2, 3, 5, *per_finger_shape, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 3, 5, 64)
    flat = encoder(x.detach().reshape(6, 5, *per_finger_shape))
    torch.testing.assert_close(y.detach(), flat.reshape(2, 3, 5, 64))
    assert encoder.out_dim == 64 and encoder.num_tokens == 5
    assert encoder.consumed_observation_fields == (encoder.input_key,)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.parametrize("encoder_type,per_finger_shape", [
    (XHandForceEncoder, (3,)),
    (XHandTaxelEncoder, (120, 3)),
])
def test_per_finger_independence_and_invalid_nan(encoder_type, per_finger_shape):
    encoder = encoder_type().eval()
    x = torch.randn(2, 5, *per_finger_shape)
    with torch.no_grad():
        baseline = encoder(x)
        mutated = x.clone()
        mutated[:, 0] += 100.0
        changed = encoder(mutated)
        torch.testing.assert_close(changed[:, 1:], baseline[:, 1:])
        valid = torch.ones(2, 5, dtype=torch.bool)
        valid[:, 1] = False
        masked = x.clone()
        masked[:, 1] = float("nan")
        y = encoder(masked, valid)
        assert torch.isfinite(y).all()
        assert torch.count_nonzero(y[:, 1]) == 0
        torch.testing.assert_close(y[:, [0, 2, 3, 4]], baseline[:, [0, 2, 3, 4]])


@pytest.mark.parametrize("encoder_type,per_finger_shape", [
    (XHandForceEncoder, (3,)),
    (XHandTaxelEncoder, (120, 3)),
])
def test_validation(encoder_type, per_finger_shape):
    encoder = encoder_type()
    x = torch.randn(2, 5, *per_finger_shape)
    with pytest.raises(ValueError, match="expected nonempty"):
        encoder(x[:, :4])
    with pytest.raises(ValueError, match="tactile_valid"):
        encoder(x, torch.ones(2, 5))
    bad = x.clone()
    bad[:, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        encoder(bad)


def test_architectures_are_minimal():
    force = XHandForceEncoder()
    taxel = XHandTaxelEncoder()
    assert [(p.in_features, p.out_features) for p in force.net if isinstance(p, torch.nn.Linear)] == [
        (3, 64), (64, 64), (64, 64),
    ]
    assert [(p.in_channels, p.out_channels, p.kernel_size, p.padding) for p in taxel.net if isinstance(p, torch.nn.Conv1d)] == [
        (3, 32, (3,), (1,)), (32, 64, (3,), (1,)),
    ]
    assert isinstance(taxel.net[-1], torch.nn.AdaptiveAvgPool1d)
    assert sum(p.numel() for p in force.parameters()) == 8576
    assert sum(p.numel() for p in taxel.parameters()) == 6528


@pytest.mark.parametrize("encoder_type,per_finger_shape", [
    (XHandForceEncoder, (3,)),
    (XHandTaxelEncoder, (120, 3)),
])
def test_cpu_bfloat16_autocast(encoder_type, per_finger_shape):
    encoder = encoder_type()
    x = torch.randn(1, 5, *per_finger_shape)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = encoder(x)
    assert y.shape == (1, 5, 64)
    assert torch.isfinite(y).all()
