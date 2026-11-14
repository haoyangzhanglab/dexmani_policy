"""Focused projection, sampling and shared-encoder regression checks."""

import pytest
import torch

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import (
    ImageProjection, pixels_to_grid, project_to_image,
)
from dexmani_policy.agents.obs_encoder.pointcloud.rgb_patch_fusion import RGBPatchFusion
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.common.image_util import resize_image_transform


@pytest.fixture(params=[("cpu", False), ("cuda", False), ("cuda", True)])
def projection_device(request):
    device, tf32 = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    yield device
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("resize", [False, True])
def test_projection_coordinates(projection_device, dtype, resize):
    device = projection_device
    points = torch.tensor([[[1, 2, 2], [-1, 0, 2]]], device=device, dtype=dtype).expand(2, -1, -1)
    K = torch.tensor([
        [[8, 0, 31.5], [0, 8, 15.5], [0, 0, 1]],
        [[16, 0, 31.5], [0, 4, 15.5], [0, 0, 1]],
    ], device=device, dtype=dtype)
    T = torch.eye(3, 4, device=device, dtype=dtype).expand(2, -1, -1).clone()
    hw = (16, 32) if resize else (32, 64)
    A = resize_image_transform((32, 64), hw, device=device).to(dtype).expand(2, -1, -1)
    expected = [[[0.125, 0.5], [-0.125, 0]], [[0.25, 0.25], [-0.25, 0]]]
    if resize:
        T[:] = T.new_tensor([[0, -1, 0, 0.5], [1, 0, 0, -0.25], [0, 0, 1, 2]])
        expected = [[[-0.09375, 0.09375], [0.03125, -0.15625]], [[-0.1875, 0.046875], [0.0625, -0.078125]]]
    values = [value.requires_grad_() for value in (points, K, T, A)]
    original = [value.detach().clone() for value in values]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = project_to_image(*values, hw)
    torch.testing.assert_close(actual.grid_xy, torch.tensor(expected, device=device), rtol=0, atol=0)
    assert actual.valid.all()
    assert actual.grid_xy.dtype == torch.float32
    assert not actual.grid_xy.requires_grad
    assert torch.backends.cuda.matmul.allow_tf32 == tf32
    for value, before in zip(values, original):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.parametrize("scenario", ["invalid_points", "invalid_camera", "zero_denominator"])
def test_projection_invalid_inputs(projection_device, scenario):
    device = projection_device
    points = torch.zeros(4, 8, 3, device=device)
    points[..., 2] = 1
    K = torch.tensor(
        [[8, 0, 31.5], [0, 8, 15.5], [0, 0, 1]], device=device,
    ).expand(4, -1, -1).clone()
    T = torch.eye(3, 4, device=device).expand(4, -1, -1).clone()
    A = torch.eye(3, device=device).expand(4, -1, -1).clone()
    expected_valid = torch.ones(4, 8, device=device, dtype=torch.bool)
    if scenario == "invalid_points":
        points[:, :5, 2] = points.new_tensor([-1, 0, float("nan"), float("inf"), -float("inf")])
        expected_valid[:, :5] = False
    elif scenario == "invalid_camera":
        T[0, 0, 0] = float("nan")
        K[1, 2, 2] = float("inf")
        A[2, 2, 0] = -float("inf")
        expected_valid[:3] = False
    else:
        K[:, 2] = 0
        expected_valid[:] = False
    actual = project_to_image(points, K, T, A, (32, 64))
    assert torch.equal(actual.valid, expected_valid)
    assert (actual.grid_xy == 0).all()


def test_pixel_centers_and_adjacent_edges(projection_device):
    device = projection_device
    lower = torch.tensor(-0.5, device=device)
    upper = torch.tensor(7.5, device=device)
    pixels = torch.stack([
        torch.stack((lower, lower)),
        torch.tensor([3.5, 1.5], device=device),
        torch.stack((torch.nextafter(upper, lower), lower)),
        torch.stack((upper, lower)),
        torch.stack((torch.nextafter(lower, -upper), lower)),
        torch.stack((torch.nextafter(lower, upper), lower)),
    ])[None]
    projection = pixels_to_grid(pixels, torch.ones(1, 6, dtype=torch.bool, device=device), (4, 8))
    assert projection.valid.tolist() == [[True, True, True, False, False, True]]
    assert projection.grid_xy[0, 0].tolist() == [-1, -1]
    assert projection.grid_xy[0, 1].tolist() == [0, 0]
    assert projection.grid_xy[0, 2, 0] < 1
    points = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    actual = project_to_image(
        points, torch.eye(3, device=device)[None], torch.eye(3, 4, device=device)[None],
        torch.eye(3, device=device)[None], (4, 8),
    )
    assert torch.equal(actual.valid, projection.valid)
    torch.testing.assert_close(actual.grid_xy, projection.grid_xy, rtol=0, atol=0)


@pytest.mark.parametrize("device,dtype,amp", [
    ("cpu", torch.float32, False), ("cpu", torch.bfloat16, True),
    ("cuda", torch.float32, False), ("cuda", torch.bfloat16, True),
    ("cuda", torch.float16, True), ("cuda", torch.bfloat16, False),
    ("cuda", torch.float16, False),
])
@pytest.mark.parametrize("prefix", [0, 5])
@pytest.mark.parametrize("validity", ["all", "none", "partial"])
def test_fusion(device, dtype, amp, prefix, validity):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(29)
    weights_dtype = torch.float32 if amp else dtype
    fusion = RGBPatchFusion(384).to(device=device, dtype=weights_dtype)
    geometry = torch.randn(4, 128, 192, device=device, dtype=dtype, requires_grad=True)
    features = torch.randn(4, 196 + prefix, 384, device=device, dtype=dtype, requires_grad=True)
    rgb_grid = features[:, prefix:].unflatten(1, (14, 14))
    valid = torch.ones(4, 128, device=device, dtype=torch.bool)
    if validity == "none":
        valid[:] = False
    elif validity == "partial":
        valid[:, ::2] = False
    projection = ImageProjection(torch.rand(4, 128, 2, device=device) * 2 - 1, valid)
    with torch.autocast(device, dtype=dtype, enabled=amp):
        output = fusion(geometry, rgb_grid, projection)
    assert output.shape == (4, 128, 384)
    assert torch.isfinite(output).all()
    assert (output[..., 192:][~valid] == 0).all()
    output.float().square().mean().backward()
    for value in (geometry, features, *fusion.parameters()):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    if validity == "none":
        assert (features.grad == 0).all()


@pytest.mark.parametrize("prefix", [0, 5])
def test_fusion_sampling_values(prefix):
    torch.manual_seed(29)
    fusion = RGBPatchFusion(4, 2)
    features = torch.randn(2, 4 + prefix, 4)
    rgb_grid = features[:, prefix:].unflatten(1, (2, 2))
    geometry = torch.randn(2, 3, 2)
    coordinates = torch.tensor([[[-0.5, -0.5], [0, 0], [0.5, 0.5]]]).expand(2, -1, -1)
    valid = torch.tensor([[True, True, False]]).expand(2, -1)
    sampled = torch.stack((rgb_grid[:, 0, 0], rgb_grid.mean((1, 2)), rgb_grid[:, 1, 1]), dim=1)
    semantic = fusion.semantic_projection(fusion.rgb_norm(sampled))
    semantic[:, 2] = 0
    output = fusion(geometry, rgb_grid, ImageProjection(coordinates, valid))
    torch.testing.assert_close(output[..., :2], fusion.geometry_norm(geometry))
    torch.testing.assert_close(output[..., 2:], semantic)


def make_observation(device):
    torch.manual_seed(61)
    points = torch.randn(4, 1024, 3, device=device) * 0.1
    points[..., 2] += 1
    return {
        "point_cloud": points,
        "rgb": torch.randint(0, 256, (4, 3, 224, 224), device=device, dtype=torch.uint8),
        "joint_state": torch.randn(4, 19, device=device),
        "camera_intrinsic": torch.tensor(
            [[160, 0, 111.5], [0, 160, 111.5], [0, 0, 1]], device=device,
        ).expand(4, -1, -1),
        "camera_extrinsic": torch.eye(3, 4, device=device).expand(4, -1, -1),
    }


@pytest.fixture(scope="module")
def encoder():
    if not torch.cuda.is_available():
        pytest.skip("Full encoder checks require CUDA and PyTorch3D")
    return RGBPointObservationEncoder(fps_random_config={"use_random": False}).cuda()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_encoder_forward_backward(encoder, dtype):
    encoder.train()
    encoder.zero_grad(set_to_none=True)
    obs = make_observation("cuda")
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        output = encoder(obs)
    assert output.shape == (2, 258, 384) and torch.isfinite(output).all()
    output[..., 0].float().square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in encoder.rgb_encoder.backbone.parameters())
    assert not encoder.rgb_encoder.backbone.training
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.rgb_fusion.semantic_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["sat", "dqrise", "dexlatent"])
def test_shared_policy_interface(kind, encoder):
    from dexmani_policy.agents.policies.dexlatent import DexLatent
    from dexmani_policy.agents.policies.dqrise import DQRISE
    from dexmani_policy.agents.policies.sat import SAT

    policy_class = {"sat": SAT, "dqrise": DQRISE, "dexlatent": DexLatent}[kind]
    kwargs = {"num_codes": 8} if kind == "dqrise" else {}
    # The action decoder is outside this check; retain the full default observation encoder.
    policy = policy_class(n_layers=1, hidden_dim=64, n_head=4, **kwargs).cuda().eval()
    assert isinstance(policy.obs_encoder, RGBPointObservationEncoder)
    policy.obs_encoder.load_state_dict(encoder.state_dict(), strict=True)
    obs = {key: value.unflatten(0, (2, 2)) for key, value in make_observation("cuda").items()}
    with torch.no_grad():
        output = policy._encode_obs(obs)
    assert output.shape == (2, 258, 384) and torch.isfinite(output).all()
