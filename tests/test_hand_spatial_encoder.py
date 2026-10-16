"""World-frame wrist/fingertip feature encoding: geometry, AMP, and gradients."""

import pytest
import torch

from dexmani_policy.agents.obs_encoder.proprio.hand_spatial_encoder import HandSpatialEncoder


@pytest.mark.parametrize("n_freqs", [0, 1, 2, 4])
@pytest.mark.parametrize("shape", [(1, 6, 3), (8, 6, 3)])
def test_shapes_and_gradients(shape, n_freqs):
    torch.manual_seed(11)
    encoder = HandSpatialEncoder(token_dim=48, num_frequencies=n_freqs)
    xyz = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
    output = encoder(xyz)
    assert output.shape == (*shape[:2], 48)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(xyz.grad).all()
    for p in encoder.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert encoder.relative_proj.weight.grad.abs().sum() > 0


def test_world_frame_offset_stays_in_world_axes():
    torch.manual_seed(13)
    encoder = HandSpatialEncoder(token_dim=16)
    observed = []
    hook = encoder.relative_proj.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach())
    )
    xyz = torch.randn(2, 6, 3) * 0.1
    shift = torch.tensor([0.4, -0.5, 0.7])
    encoder(xyz)
    encoder(xyz + shift)
    hook.remove()
    torch.testing.assert_close(observed[0], observed[1], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(observed[0][:, 0]).item() == 0


def test_zero_init_matches_global_only():
    torch.manual_seed(16)
    encoder = HandSpatialEncoder(token_dim=32)
    p = torch.randn(2, 6, 3)
    assert torch.count_nonzero(encoder.relative_proj.weight).item() == 0
    z = encoder(p)
    assert z.shape == (2, 6, 32)
    assert torch.isfinite(z).all()
    with torch.no_grad():
        encoder.relative_proj.weight.normal_(std=0.1)
    z_changed = encoder(p)
    assert not torch.allclose(z, z_changed)


def test_bf16_autocast():
    torch.manual_seed(19)
    encoder = HandSpatialEncoder(token_dim=48)
    p = torch.randn(4, 6, 3, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = encoder(p)
        loss = result.float().square().mean()
    loss.backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"token_dim": 0}, {"hidden_dim": -5},
    {"position_scale": 0}, {"position_scale": float("nan")},
    {"hand_scale": 0}, {"hand_scale": float("inf")},
    {"num_frequencies": -1},
])
def test_bad_init(kwargs):
    with pytest.raises(ValueError):
        HandSpatialEncoder(**kwargs)


@pytest.mark.parametrize("p", [torch.empty(2, 5, 3), torch.empty(2, 6, 4), torch.ones(2, 6, 3, dtype=torch.long)])
def test_bad_input(p):
    with pytest.raises((TypeError, ValueError)):
        HandSpatialEncoder()(p)


def test_parameters():
    model = HandSpatialEncoder(token_dim=192, hidden_dim=128)
    assert sum(p.numel() for p in model.parameters()) == 28928

def test_optional_offset_ablation_and_identity():
    torch.manual_seed(29)
    model = HandSpatialEncoder(token_dim=32, use_wrist_offset=False)
    xyz = torch.zeros(2, 6, 3)
    tokens = model(xyz)
    assert tokens.shape == (2, 6, 32)
    assert model.relative_proj is None
    assert len(model.KEYPOINTS) == 6
    assert not torch.equal(tokens[:, 0], tokens[:, 1])


def test_configurable_relative_branch_matches_absolute_only_at_init():
    torch.manual_seed(35)
    combined = HandSpatialEncoder(token_dim=32, num_frequencies=2)
    absolute = HandSpatialEncoder(token_dim=32, num_frequencies=2, use_wrist_offset=False)
    absolute.load_state_dict({k: v for k, v in combined.state_dict().items()
                              if k != "relative_proj.weight"}, strict=True)
    xyz = torch.randn(3, 6, 3)
    torch.testing.assert_close(combined(xyz), absolute(xyz), rtol=0, atol=0)


def test_small_offset_update_survives_bf16():
    model = HandSpatialEncoder(token_dim=32)
    xyz = torch.zeros(1, 6, 3)
    xyz[:, 1:, 0] = 0.1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        before = model(xyz).detach()
    with torch.no_grad():
        model.relative_proj.weight[0, 0] = 1e-5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        after = model(xyz).detach()
    assert after.dtype == torch.float32
    assert (after[:, 1:] - before[:, 1:]).abs().max() > 1e-6
    torch.testing.assert_close(after[:, 0], before[:, 0], atol=0, rtol=0)


def test_nonfinite_positions_are_rejected():
    model = HandSpatialEncoder()
    for bad in (float("nan"), float("inf"), -float("inf")):
        xyz = torch.zeros(2, 6, 3)
        xyz[0, 2, 1] = bad
        with pytest.raises(ValueError, match="finite"):
            model(xyz)


def test_historical_frames_are_independent():
    torch.manual_seed(41)
    model = HandSpatialEncoder(token_dim=24)
    xyz = torch.randn(2, 3, 6, 3)
    actual = model(xyz.reshape(6, 6, 3)).reshape(2, 3, 6, 24)
    for b in range(2):
        for t in range(3):
            torch.testing.assert_close(actual[b, t], model(xyz[b, t:t+1])[0])


def test_strict_checkpoint_roundtrip(tmp_path):
    model = HandSpatialEncoder().eval()
    assert sum(p.numel() for p in model.parameters()) == 28928
    path = tmp_path / "hand.pt"
    torch.save(model.state_dict(), path)
    target = HandSpatialEncoder().eval()
    target.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    xyz = torch.randn(2, 6, 3)
    torch.testing.assert_close(model(xyz), target(xyz), rtol=0, atol=0)
