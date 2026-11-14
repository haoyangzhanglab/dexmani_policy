import pytest
import torch

from dexmani_policy.agents.obs_encoder.proprio import HandGeometryEncoder, _rotation_6d_to_matrix
from dexmani_policy.agents.obs_encoder.interaction.relative_3d_attention import (
    Relative3DCrossAttention,
)


def make_geometry(shape, device):
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    position = torch.randn(*shape, 3, device=device) * 0.25
    pose = torch.cat(
        (position, rotation[:, 0].expand(*shape, 3), rotation[:, 1].expand(*shape, 3)), -1
    )
    tips = position.unsqueeze(-2) + torch.randn(*shape, 5, 3, device=device) * 0.1
    return pose, tips


@pytest.mark.parametrize(
    "name,cls",
    [
        ("wrist_scale", HandGeometryEncoder),
        ("finger_scale", HandGeometryEncoder),
        ("geometry_scale", Relative3DCrossAttention),
    ],
)
@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_scale(name, cls, scale):
    with pytest.raises(ValueError, match=name):
        cls(**{name: scale})


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_hand_shapes_and_scales(device, shape):
    pose, tips = make_geometry(shape, device)
    model = HandGeometryEncoder().to(device)
    output = model(pose, tips)
    rotation = _rotation_6d_to_matrix(pose[..., 3:])
    local = (tips - pose[..., None, :3]) @ rotation
    wrist = torch.cat((pose[..., :3] / 0.25, rotation[..., :, 0], rotation[..., :, 1]), -1)
    expected = torch.cat((model.wrist_mlp(wrist).unsqueeze(-2), model.finger_mlp(local / 0.10)), -2)
    torch.testing.assert_close(output["tokens"], model.norm(expected + model.body_identity[0]))
    assert output["tokens"].shape == (*shape, 6, 192)
    assert output["anchors_base"].shape == (*shape, 6, 3)
    flattened = model(pose.reshape(-1, 9), tips.reshape(-1, 5, 3))
    torch.testing.assert_close(output["tokens"].reshape(-1, 6, 192), flattened["tokens"])
    torch.testing.assert_close(output["anchors_base"].reshape(-1, 6, 3), flattened["anchors_base"])
    for wrist_scale, finger_scale in ((0.5, 0.1), (0.25, 0.2)):
        custom = HandGeometryEncoder(wrist_scale, finger_scale).to(device)
        custom.load_state_dict(model.state_dict(), strict=True)
        changed = custom(pose, tips)
        torch.testing.assert_close(
            changed["anchors_base"], torch.cat((pose[..., None, :3], tips), -2)
        )
        expected_wrist = torch.cat(
            (pose[..., :3] / wrist_scale, rotation[..., :, 0], rotation[..., :, 1]), -1
        )
        expected = torch.cat(
            (model.wrist_mlp(expected_wrist).unsqueeze(-2), model.finger_mlp(local / finger_scale)),
            -2,
        )
        torch.testing.assert_close(changed["tokens"], model.norm(expected + model.body_identity[0]))


@pytest.mark.parametrize(
    "pose_shape,tip_shape",
    [
        ((), (5, 3)),
        ((2, 8), (2, 5, 3)),
        ((2, 9), (2, 3, 5)),
        ((2, 9), (1, 5, 3)),
        ((2, 3, 9), (2, 5, 3)),
    ],
)
def test_invalid_shapes(pose_shape, tip_shape):
    with pytest.raises(ValueError, match="eef_pose|fingertip_xyz|Leading dimensions"):
        HandGeometryEncoder()(torch.zeros(pose_shape), torch.zeros(tip_shape))


def test_rotation_and_rigid_transform(device):
    identity = torch.eye(3, device=device)
    quarter = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    for matrix in (identity, quarter):
        r6 = torch.cat((matrix[:, 0], matrix[:, 1]))
        torch.testing.assert_close(_rotation_6d_to_matrix(r6), matrix)
    rotation = _rotation_6d_to_matrix(torch.randn(16, 6, device=device))
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation, identity.expand(16, 3, 3), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(rotation.det(), torch.ones(16, device=device))
    pose, tips = make_geometry((2, 3), device)
    r = _rotation_6d_to_matrix(pose[..., 3:])
    q = rotation[0]
    t = torch.tensor([0.2, -0.3, 0.4], device=device)
    position = pose[..., :3] @ q.T + t
    transformed_r = q @ r
    transformed_pose = torch.cat((position, transformed_r[..., :, 0], transformed_r[..., :, 1]), -1)
    transformed_tips = tips @ q.T + t
    torch.testing.assert_close(
        (tips - pose[..., None, :3]) @ r,
        (transformed_tips - position.unsqueeze(-2)) @ transformed_r,
    )
    model = HandGeometryEncoder().to(device)
    original = model(pose, tips)["tokens"][..., 1:, :]
    transformed = model(transformed_pose, transformed_tips)["tokens"][..., 1:, :]
    torch.testing.assert_close(original, transformed, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("scale", [0.1, 0.2])
def test_relative_attention(device, scale):
    model = Relative3DCrossAttention(scale).to(device)
    torch.nn.init.normal_(model.relative_value.weight, std=0.05)
    body, scene = torch.randn(2, 6, 192, device=device), torch.randn(2, 8, 192, device=device)
    anchors, xyz = torch.randn(2, 6, 3, device=device), torch.randn(2, 8, 3, device=device)
    q = model.q(model.query_norm(body)).reshape(2, 6, 4, 48).transpose(1, 2)
    k, v = (
        model.kv(model.memory_norm(scene)).reshape(2, 8, 2, 4, 48).permute(2, 0, 3, 1, 4).unbind(0)
    )
    delta = (xyz[:, None] - anchors[:, :, None]) / scale
    bias = model.position_bias(delta).permute(0, 3, 1, 2)
    weights = (q @ k.transpose(-1, -2) / (48**0.5) + bias).softmax(-1)
    # Direct pairwise values independently check the split scene/anchor computation.
    values = model.relative_value(delta).reshape(2, 6, 8, 4, 48).permute(0, 3, 1, 2, 4)
    context = weights @ v + (weights[..., None] * values).sum(-2)
    expected = model.norm(body + model.out(context.transpose(1, 2).reshape(2, 6, 192)))
    actual = model(body, scene, anchors, xyz)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert actual.shape == (2, 6, 192)
    torch.testing.assert_close(
        actual[:1],
        model(body[:1], scene[:1], anchors[:1], xyz[:1]),
        atol=3e-6,
        rtol=3e-5,
    )
    assert Relative3DCrossAttention().geometry_scale == 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_amp_gradients(device, dtype):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("FP16 AMP tested on CUDA")
    encoder, attention = HandGeometryEncoder().to(device), Relative3DCrossAttention().to(device)
    geometry_dtypes = []

    def record_dtype(module, inputs):
        geometry_dtypes.append(inputs[0].dtype)

    for module in (
        encoder.wrist_mlp,
        encoder.finger_mlp,
        attention.position_bias,
        attention.relative_value,
    ):
        module.register_forward_pre_hook(record_dtype)
    pose, tips = make_geometry((2,), device)
    pose.requires_grad_()
    tips.requires_grad_()
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        hand = encoder(pose, tips)
        output = attention(
            hand["tokens"],
            torch.randn(2, 8, 192, device=device),
            hand["anchors_base"],
            torch.randn(2, 8, 3, device=device),
        )
    assert geometry_dtypes == [torch.float32] * 4
    assert hand["anchors_base"].dtype == torch.float32
    assert torch.isfinite(output).all()
    (output * torch.randn_like(output)).mean().backward()
    for parameter in (*encoder.parameters(), *attention.parameters(), pose, tips):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in attention.parameters()) > 0


@pytest.mark.parametrize("cls", [HandGeometryEncoder, Relative3DCrossAttention])
def test_checkpoint(cls, tmp_path):
    model = cls()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = cls()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(model.state_dict(), restored.state_dict())
