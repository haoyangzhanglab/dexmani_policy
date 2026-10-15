"""ReTouch patch encoder contract tests.

The artificial patch map below is for testing only. It does NOT represent the
physical T30/T16 taxel order or ReTouch's unreleased Figure 6 assignment.
"""

import pytest
import torch

from dexmani_policy.agents.obs_encoder.tactile import XHandReTouchEncoder


def _fake_map():
    return torch.arange(120, dtype=torch.long)[None].expand(5, -1).clone() // 24


def test_patch_encoder_shape_frame_independence_and_gradient():
    torch.manual_seed(31)
    encoder = XHandReTouchEncoder(_fake_map())
    x = torch.randn(2, 3, 5, 120, 3, requires_grad=True)
    tokens = encoder(x)
    assert tokens.shape == (2, 3, 5, 64)
    flat = encoder(x.detach().reshape(6, 5, 120, 3)).reshape(2, 3, 5, 64)
    torch.testing.assert_close(tokens.detach(), flat, atol=0, rtol=0)
    assert encoder.out_dim == 64
    assert encoder.num_tokens == 5
    assert encoder.consumed_observation_fields == ("taxel_tactile_force",)
    tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_patch_encoder_fingers_independent_and_invalid_nan_masked():
    model = XHandReTouchEncoder(_fake_map()).eval()
    x = torch.randn(3, 5, 120, 3)
    with torch.no_grad():
        baseline = model(x)
        perturbed = x.clone()
        perturbed[:, 3] += 12
        output = model(perturbed)
        torch.testing.assert_close(output[:, :3], baseline[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(output[:, 4:], baseline[:, 4:], atol=0, rtol=0)
        valid = torch.ones(3, 5, dtype=torch.bool)
        valid[:, 1] = False
        masked = x.clone()
        masked[:, 1] = float("nan")
        out = model(masked, valid)
        assert torch.isfinite(out).all()
        assert not out[:, 1].any()
        torch.testing.assert_close(out[:, [0, 2, 3, 4]], baseline[:, [0, 2, 3, 4]], atol=0, rtol=0)


def test_patch_encoder_rejects_bad_maps():
    with pytest.raises(ValueError, match="integer"):
        XHandReTouchEncoder(_fake_map().float())
    with pytest.raises(ValueError, match="integer"):
        XHandReTouchEncoder(torch.zeros(2, 120, dtype=torch.long))
    with pytest.raises(ValueError, match="values"):
        XHandReTouchEncoder(torch.full((5, 120), 5, dtype=torch.long))
    with pytest.raises(ValueError, match="at least one"):
        XHandReTouchEncoder(torch.zeros(5, 120, dtype=torch.long))


def test_patch_encoder_validity_and_finite_checks():
    encoder = XHandReTouchEncoder(_fake_map())
    x = torch.randn(1, 5, 120, 3)
    with pytest.raises(ValueError, match="expected nonempty"):
        encoder(x[:, :4])
    with pytest.raises(ValueError, match="tactile_valid"):
        encoder(x, torch.ones(1, 5))
    x[:, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        encoder(x)


def test_patch_encoder_checkpoint_restores_region_map():
    original_map = _fake_map()
    model = XHandReTouchEncoder(original_map)
    state = model.state_dict()
    assert "patch_mask" in state
    restored = XHandReTouchEncoder(torch.flip(original_map, (1,)))
    restored.load_state_dict(state)
    assert torch.equal(restored.patch_mask, model.patch_mask)
    x = torch.randn(1, 5, 120, 3)
    torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)


def test_patch_encoder_preserves_patch_location_information():
    torch.manual_seed(0)
    model = XHandReTouchEncoder(_fake_map()).eval()
    x = torch.zeros(1, 5, 120, 3)
    x[:, 0, 0, 0] = 20
    y = x.clone()
    y[:, 0, 70, 0] = 20
    y[:, 0, 0, 0] = 0
    with torch.no_grad():
        a, b = model(x), model(y)
    assert not torch.allclose(a[:, 0], b[:, 0])
    torch.testing.assert_close(a[:, 1:], b[:, 1:], atol=0, rtol=0)


def test_patch_encoder_cpu_bfloat16_autocast():
    model = XHandReTouchEncoder(_fake_map())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(torch.randn(1, 5, 120, 3))
    assert out.shape == (1, 5, 64) and torch.isfinite(out).all()
