import pytest
import torch

from dexmani_policy.agents.obs_encoder.pointcloud.ops import (
    farthest_point_sample, index_points, knn_indices, sample_point_patches,
)
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import (
    LocalGeometryEmbedding, PointTransformerEncoder, SpatialContextEncoder,
)
from dexmani_policy.agents.obs_encoder.pointcloud.rope import RotaryPositionEmbedding3D


def assert_gradients(module):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("input_channels", [3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_local_geometry(device, input_channels, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("PyTorch 2.4 CPU LayerNorm does not support direct FP16")
    model = LocalGeometryEmbedding(input_channels, 192, 0.1).to(device=device, dtype=dtype)
    xyz = torch.randn(2, 8, 32, 3, device=device) * 0.1
    attrs = torch.randn(2, 8, 32, input_channels - 3, device=device)
    output = model(xyz, attrs)
    assert output.shape == (2, 8, 192)
    assert torch.isfinite(output).all()
    order = torch.randperm(32, device=device)
    torch.testing.assert_close(output, model(xyz[:, :, order], attrs[:, :, order]))
    (output.float() * torch.randn_like(output.float())).mean().backward()
    assert_gradients(model)


@pytest.mark.parametrize("channels", [192, 384])
def test_spatial_context(device, channels):
    model = SpatialContextEncoder(channels=channels).to(device)
    tokens = torch.randn(2, 16, channels, device=device, requires_grad=True)
    xyz = torch.randn(2, 16, 3, device=device)
    output = model(tokens, xyz)
    assert output.shape == tokens.shape
    torch.testing.assert_close(output[:1], model(tokens[:1], xyz[:1]), atol=2e-6, rtol=2e-5)
    changed_tokens, changed_xyz = tokens.detach().clone(), xyz.clone()
    changed_tokens[1] *= 10
    changed_xyz[1] += 100
    torch.testing.assert_close(output[:1], model(changed_tokens, changed_xyz)[:1])
    (output * torch.randn_like(output)).mean().backward()
    assert_gradients(model)
    assert torch.isfinite(tokens.grad).all()


def test_rope(device):
    rope = RotaryPositionEmbedding3D(24).to(device)
    xyz = torch.randn(2, 16, 3, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        cos, sin = rope(xyz)
    assert cos.shape == sin.shape == (2, 1, 16, 24)
    assert cos.dtype == sin.dtype == torch.float32
    frequencies = 100.0 ** (-torch.arange(0, 8, 2, device=device).float() / 8)
    angles = ((xyz - xyz.mean(1, keepdim=True)) / 0.05)[..., None] * frequencies
    angles = angles.repeat_interleave(2, -1).flatten(-2).unsqueeze(1)
    torch.testing.assert_close(cos, angles.cos())
    torch.testing.assert_close(sin, angles.sin())
    x = torch.randn(2, 8, 16, 24, device=device)
    rotated = rope.apply_rotary(x, cos, sin)
    even, odd = x[..., 0::2], x[..., 1::2]
    cos_pair, sin_pair = cos[..., 0::2], sin[..., 0::2]
    torch.testing.assert_close(rotated[..., 0::2], even * cos_pair - odd * sin_pair)
    torch.testing.assert_close(rotated[..., 1::2], odd * cos_pair + even * sin_pair)
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    with pytest.raises(ValueError, match="divisible by 6"):
        RotaryPositionEmbedding3D(16)


@pytest.mark.parametrize("input_channels", [3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_encoder(device, input_channels, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("FP16 AMP is tested on CUDA")
    model = PointTransformerEncoder(input_channels=input_channels).to(device).eval()
    points = torch.randn(2, 1024, input_channels, device=device)
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        output = model(points)
        tokens = output["tokens"].float()
        loss = (tokens * torch.randn_like(tokens)).mean()
    assert output["tokens"].shape == (2, 128, 192)
    assert output["xyz"].shape == (2, 128, 3)
    assert output["xyz"].dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in output.values())
    loss.backward()
    assert_gradients(model)


@pytest.mark.parametrize("input_channels", [3, 6])
def test_encoder_checkpoint(device, input_channels, tmp_path):
    model = PointTransformerEncoder(input_channels=input_channels).to(device).eval()
    points = torch.randn(2, 1024, input_channels, device=device)
    assert model.effective_config == dict(
        channels=192, num_patches=128, group_size=32, local_hidden_dim=64,
        depth=2, num_heads=8, mlp_ratio=2.0,
        geometry_scale=0.1, rope_scale=0.05, rope_base=100.0,
    )
    assert all(key.startswith(("patch_encoder.", "spatial_encoder.")) for key in model.state_dict())
    path = tmp_path / "encoder.pt"
    torch.save(model.state_dict(), path)
    restored = PointTransformerEncoder(
        input_channels=input_channels, **model.effective_config,
    ).to(device).eval()
    restored.load_state_dict(torch.load(path, weights_only=True, map_location=device), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(points), restored(points), rtol=0, atol=0)


def test_sampling(device, monkeypatch):
    import pytorch3d.ops

    calls = []
    native = pytorch3d.ops.sample_farthest_points

    def record(**kwargs):
        calls.append(kwargs["random_start_point"])
        return native(**kwargs)

    monkeypatch.setattr(pytorch3d.ops, "sample_farthest_points", record)
    xyz = torch.randn(2, 64, 3, device=device)
    xyz[1] += 20
    config = dict(use_random=True, use_random_start=True, use_shuffle_output=True)
    centers, indices = sample_point_patches(xyz, 8, 4, config, training=False)
    assert calls[-1] is False
    torch.testing.assert_close(centers, sample_point_patches(xyz, 8, 4, config, training=False)[0])
    sample_point_patches(xyz, 8, 4, config, training=True)
    assert calls[-1] is True
    assert indices.shape == (2, 8, 4)
    assert ((indices >= 0) & (indices < 64)).all()
    expected = torch.cdist(centers, xyz).topk(4, largest=False).indices
    torch.testing.assert_close(indices.sort(-1).values, expected.sort(-1).values)
    for batch in range(2):
        single = knn_indices(4, xyz[batch : batch + 1], centers[batch : batch + 1])
        torch.testing.assert_close(indices[batch : batch + 1], single)
    gathered = index_points(xyz, indices)
    for batch in range(2):
        torch.testing.assert_close(gathered[batch], xyz[batch][indices[batch]])
    points = torch.cat((xyz, torch.arange(64, device=device).expand(2, -1)[..., None]), -1)
    ordered = farthest_point_sample(
        points, 8, use_random=True, use_random_start=False, use_shuffle_output=False,
    )
    shuffled = farthest_point_sample(
        points, 8, use_random=True, use_random_start=False, use_shuffle_output=True,
    )
    assert not torch.equal(ordered, shuffled)
    torch.testing.assert_close(ordered[..., 3].sort().values, shuffled[..., 3].sort().values)
    for batch in range(2):
        torch.testing.assert_close(shuffled[batch], points[batch, shuffled[batch, :, 3].long()])
    torch.testing.assert_close(centers, farthest_point_sample(xyz, 8, use_random=False))
    with pytest.raises(ValueError, match="N must be"):
        sample_point_patches(xyz, 65, 4)


def test_config_composition(device):
    config = dict(
        channels=96, num_patches=16, group_size=8, local_hidden_dim=32,
        depth=1, num_heads=4, mlp_ratio=1.5,
        geometry_scale=0.2, rope_scale=0.07, rope_base=200.0,
    )
    model = PointTransformerEncoder(input_channels=3, **config).to(device).eval()
    assert model.effective_config == config
    points = torch.randn(2, 64, 3, device=device)
    patches = model.patch_encoder(points)
    expected = model.spatial_encoder(patches["tokens"], patches["xyz"])
    output = model(points)
    assert output["tokens"].shape == (2, 16, 96)
    assert model.patch_encoder.geometry_embedding.context_mlp[0].in_features == 64
    torch.testing.assert_close(output["tokens"], expected, rtol=0, atol=0)
    torch.testing.assert_close(output["xyz"], patches["xyz"], rtol=0, atol=0)
