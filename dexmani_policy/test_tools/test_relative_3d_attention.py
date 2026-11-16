"""Hand-to-scene attention and geometry/tactile component regressions."""

import pytest
import torch
from torch import nn

from dexmani_policy.agents.obs_encoder.interaction.relative_3d_attention import (
    Relative3DCrossAttention,
)
from dexmani_policy.agents.obs_encoder.interaction.tactile_gated_fusion import TactileGatedFusion
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import PointTransformerEncoder
from dexmani_policy.agents.obs_encoder.proprio import HandGeometryEncoder


@pytest.fixture(params=["cpu", "cuda"])
def attention(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    return Relative3DCrossAttention().to(device)


def make_inputs(batch=2, queries=6, points=17, *, device="cpu", dtype=torch.float32):
    return (
        torch.randn(batch, queries, 192, device=device, dtype=dtype),
        torch.randn(batch, points, 192, device=device, dtype=dtype),
        torch.randn(batch, queries, 3, device=device) * 0.1,
        torch.randn(batch, points, 3, device=device) * 0.1,
    )


@pytest.mark.parametrize("batch,queries,points", [(1, 6, 128), (3, 4, 37), (1, 1, 1)])
def test_output_shape(attention, batch, queries, points):
    device = attention.q.weight.device
    output = attention(*make_inputs(batch, queries, points, device=device))
    assert output.shape == (batch, queries, 192)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@torch.no_grad()
def test_zero_bias_matches_standard_cross_attention(attention):
    device = attention.q.weight.device
    assert torch.count_nonzero(attention.position_bias[-1].weight) == 0
    assert torch.count_nonzero(attention.position_bias[-1].bias) == 0
    baseline = nn.MultiheadAttention(192, 4, dropout=0.0, batch_first=True, device=device)
    baseline.in_proj_weight.copy_(torch.cat((attention.q.weight, attention.kv.weight)))
    baseline.in_proj_bias.copy_(torch.cat((attention.q.bias, attention.kv.bias)))
    baseline.out_proj.load_state_dict(attention.out.state_dict())

    inputs = make_inputs(device=device)
    body, scene = inputs[:2]
    memory = attention.memory_norm(scene)
    context, _ = baseline(attention.query_norm(body), memory, memory, need_weights=False)
    expected = attention.norm(body + context)
    torch.testing.assert_close(attention(*inputs), expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_translation_invariance_with_learned_bias(attention):
    nn.init.normal_(attention.position_bias[-1].weight, std=0.1)
    device = attention.q.weight.device
    body, scene, anchors, xyz = make_inputs(device=device)
    translation = anchors.new_tensor([0.7, -0.3, 1.2])
    expected = attention(body, scene, anchors, xyz)
    shifted = attention(body, scene, anchors + translation, xyz + translation)
    torch.testing.assert_close(shifted, expected, rtol=1e-5, atol=1e-6)

    changed_xyz = xyz.clone()
    changed_xyz[:, 0] += translation
    changed = attention(body, scene, anchors, changed_xyz)
    assert not torch.allclose(changed, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("amp_dtype,input_dtype", [
    (None, torch.float32),
    (torch.float16, torch.float32),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
])
def test_forward_backward(attention, amp_dtype, input_dtype):
    device = attention.q.weight.device
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    inputs = make_inputs(device=device, dtype=input_dtype)
    for value in inputs[:2]:
        value.requires_grad_()
    optimizer = torch.optim.SGD(attention.parameters(), lr=0.1)
    for step in range(2):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = attention(*inputs)
        assert output.shape == inputs[0].shape
        assert output.dtype == (input_dtype if device.type == "cpu" else torch.float32)
        assert torch.isfinite(output).all()
        output[..., 0].float().square().mean().backward()
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert attention.position_bias[-1].weight.grad.abs().sum() > 0
        for value in inputs[:2]:
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
            value.grad = None
        # Zero initialization delays first-layer learning until the last layer updates.
        if step == 0:
            assert torch.count_nonzero(attention.position_bias[0].weight.grad) == 0
        else:
            assert attention.position_bias[0].weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_geometry_stays_fp32_under_autocast(attention, amp_dtype):
    device = attention.q.weight.device
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    dtypes = []
    handle = attention.position_bias.register_forward_hook(
        lambda _module, inputs, output: dtypes.append((inputs[0].dtype, output.dtype))
    )
    try:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            attention(*make_inputs(device=device))
    finally:
        handle.remove()
    assert dtypes == [(torch.float32, torch.float32)]


@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_geometry_tactile_composition(attention, amp_dtype):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    device = attention.q.weight.device
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    hand_encoder = HandGeometryEncoder().to(device)
    scene_encoder = PointTransformerEncoder(
        num_patches=16, group_size=8, fps_random_config={"use_random": False},
    ).to(device)
    fusion = TactileGatedFusion().to(device)
    nn.init.normal_(attention.position_bias[-1].weight, std=0.1)

    position = torch.randn(2, 2, 3, device=device) * 0.1
    rotation = position.new_tensor([1, 0, 0, 0, 1, 0]).expand(2, 2, -1)
    pose = torch.cat((position, rotation), dim=-1)
    fingertips = position.unsqueeze(-2) + torch.randn(2, 2, 5, 3, device=device) * 0.05
    points = torch.cat((
        position.flatten(0, 1).unsqueeze(1) + torch.randn(4, 64, 3, device=device) * 0.1,
        torch.rand(4, 64, 3, device=device),
    ), dim=-1)
    force = torch.randn(4, 5, 3, device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        hand = hand_encoder(pose, fingertips)
        scene = scene_encoder(points)
        geometry = attention(
            hand["tokens"].flatten(0, 1), scene["tokens"],
            hand["anchors_base"].flatten(0, 1), scene["xyz"],
        )
        output = fusion(geometry, force)
        zero_force = fusion(geometry, torch.zeros_like(force))

    torch.testing.assert_close(
        hand["anchors_base"], torch.cat((position.unsqueeze(-2), fingertips), dim=-2),
        rtol=0, atol=0,
    )
    assert hand["anchors_base"].dtype == scene["xyz"].dtype == torch.float32
    assert output.shape == (4, 6, 192)
    assert output.dtype == geometry.dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(zero_force, geometry, rtol=0, atol=0)
    output[..., 0].float().square().mean().backward()
    for module in (hand_encoder, scene_encoder, attention, fusion):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, (type(module).__name__, name)
            assert torch.isfinite(parameter.grad).all(), (type(module).__name__, name)


@pytest.mark.parametrize("name,shape", [
    ("body_tokens", (6, 192)),
    ("body_tokens", (2, 6, 191)),
    ("body_tokens", (0, 6, 192)),
    ("body_tokens", (2, 0, 192)),
    ("scene_tokens", (1, 17, 192)),
    ("scene_tokens", (2, 0, 192)),
    ("anchors_base", (2, 5, 3)),
    ("scene_xyz_base", (2, 17, 4)),
])
def test_invalid_input_shapes(name, shape):
    names = ("body_tokens", "scene_tokens", "anchors_base", "scene_xyz_base")
    inputs = dict(zip(names, make_inputs()))
    inputs[name] = torch.zeros(shape)
    with pytest.raises(ValueError, match=name):
        Relative3DCrossAttention()(**inputs)
