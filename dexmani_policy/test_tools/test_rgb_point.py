"""Focused projection, sampling and shared-encoder regression checks."""

import pytest
import torch

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import (
    ImageProjection, pixels_to_grid, project_to_image,
)
from dexmani_policy.agents.obs_encoder.pointcloud.rgb_patch_fusion import RGBPatchFusion
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder
from dexmani_policy.common.image_util import resize_image_transform


@pytest.fixture(params=[("cpu", False), ("cuda", True)])
def projection_device(request):
    device, tf32 = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        yield device
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


@pytest.mark.parametrize("resize", [False, True])
def test_projection_coordinates(projection_device, resize):
    device = projection_device
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    points = torch.tensor([[[1, 2, 2], [-1, 0, 2]]], device=device, dtype=torch.float32).expand(2, -1, -1)
    K = torch.tensor([
        [[8, 0, 31.5], [0, 8, 15.5], [0, 0, 1]],
        [[16, 0, 31.5], [0, 4, 15.5], [0, 0, 1]],
    ], device=device, dtype=torch.float32)
    T = torch.eye(3, 4, device=device, dtype=torch.float32).expand(2, -1, -1).clone()
    hw = (16, 32) if resize else (32, 64)
    A = resize_image_transform((32, 64), hw, device=device).expand(2, -1, -1)
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


@pytest.mark.parametrize("scenario", ["invalid_points", "invalid_camera", "zero_denominator", "tiny_denominator"])
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
        points[:, :, 2] = points.new_tensor([
            -1, 0, 0.5e-6, 1e-6, float("nan"), float("inf"), -float("inf"), 2e-6,
        ])
        expected_valid[:, :7] = False
    elif scenario == "invalid_camera":
        T[0, 0, 0] = float("nan")
        K[1, 2, 2] = float("inf")
        A[2, 2, 0] = -float("inf")
        expected_valid[:3] = False
    else:
        K[:, 2] = 0
        if scenario == "tiny_denominator":
            K[:, 2, 2] = 1e-6
        expected_valid[:] = False
    actual = project_to_image(points, K, T, A, (32, 64))
    assert torch.equal(actual.valid, expected_valid)
    assert torch.isfinite(actual.grid_xy).all()
    assert (actual.grid_xy[~actual.valid] == 0).all()


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
        torch.tensor([float("nan"), 0], device=device),
        torch.tensor([0, float("inf")], device=device),
    ])[None]
    projection = pixels_to_grid(pixels, torch.ones(1, 8, dtype=torch.bool, device=device), (4, 8))
    assert projection.valid.tolist() == [[True, True, True, False, False, True, False, False]]
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


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cuda", torch.bfloat16),
])
@pytest.mark.parametrize("neighbors", [1, 32])
def test_fusion(device, dtype, neighbors):
    prefix = 5  # Preserve the non-contiguous view after removing DINO prefix tokens.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    torch.manual_seed(29)
    fusion = RGBPatchFusion(384).to(device)
    geometry = torch.randn(4, 128, 192, device=device, dtype=dtype, requires_grad=True)
    features = torch.randn(4, 196 + prefix, 384, device=device, dtype=dtype, requires_grad=True)
    rgb_grid = features[:, prefix:].unflatten(1, (14, 14))
    valid = torch.ones(4, 128, neighbors, device=device, dtype=torch.bool)
    valid[:, 0] = False
    if neighbors > 1:
        valid[:, 1, ::2] = False
    projection = ImageProjection(torch.rand(4, 128, neighbors, 2, device=device) * 2 - 1, valid)
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        output = fusion(geometry, rgb_grid, projection)
    assert output.shape == (4, 128, 384)
    assert torch.isfinite(output).all()
    assert (output[..., 192:][~valid.any(-1)] == 0).all()
    output.float().square().mean().backward()
    for value in (geometry, features, *fusion.parameters()):
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_fusion_sampling_values():
    prefix = 5
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
    output = fusion(geometry, rgb_grid, ImageProjection(coordinates.unsqueeze(2), valid.unsqueeze(2)))
    torch.testing.assert_close(output[..., :2], fusion.geometry_norm(geometry))
    torch.testing.assert_close(output[..., 2:], semantic)


def test_group_mean_valid_neighbors():
    fusion = RGBPatchFusion(4, 2)
    rgb = torch.tensor([[
        [[1.0, 3.0, 5.0, 8.0], [7.0, -2.0, 4.0, 1.0]],
        [[9.0, 2.0, 6.0, 0.0], [3.0, 5.0, 1.0, 4.0]],
    ]], requires_grad=True)
    geometry = torch.randn(1, 3, 2)
    grid = torch.tensor([[
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]],
        [[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]],
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]],
    ]])
    valid = torch.tensor([[[True, True, False], [False, True, False], [False, False, False]]])
    averaged = torch.stack((
        (rgb[0, 0, 0] + rgb[0, 0, 1]) / 2, rgb[0, 1, 1], torch.zeros(4),
    ))[None]
    expected = fusion.semantic_projection(fusion.rgb_norm(averaged))
    expected = torch.where(valid.any(2)[..., None], expected, 0.0)
    geometry = geometry.repeat(1, 4, 1)[:, :10]
    grid = grid.repeat(1, 4, 1, 1)[:, :10]
    valid = valid.repeat(1, 4, 1)[:, :10]
    expected = expected.repeat(1, 4, 1)[:, :10]
    actual = fusion(geometry, rgb, ImageProjection(grid, valid))
    torch.testing.assert_close(actual[..., :2], fusion.geometry_norm(geometry))
    torch.testing.assert_close(actual[..., 2:], expected)
    assert torch.isfinite(actual).all()
    actual.square().sum().backward()
    assert torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
    assert (rgb.grad[0, 1, 0] == 0).all()


def make_observation(device):
    torch.manual_seed(61)
    points = torch.randn(4, 1024, 3, device=device) * 0.1
    points[..., 2] += 1
    return {
        "point_cloud": points,
        "rgb": torch.randint(0, 256, (4, 3, 240, 320), device=device, dtype=torch.uint8),
        "joint_state": torch.randn(4, 19, device=device),
        "camera_intrinsic": torch.tensor(
            [[160, 0, 159.5], [0, 160, 119.5], [0, 0, 1]], device=device,
        ).expand(4, -1, -1).flatten(1),
        "camera_extrinsic": torch.eye(3, 4, device=device).expand(4, -1, -1).flatten(1),
    }


@pytest.fixture(scope="module", params=["center", "group_mean"])
def encoder(request):
    if not torch.cuda.is_available():
        pytest.skip("Full encoder gradient checks require CUDA")
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    return RGBPointObservationEncoder(
        fps_random_config={"use_random": False}, rgb_semantic_sampling=request.param,
    ).cuda()


def test_encoder_forward_backward(encoder):
    encoder.train()
    encoder.zero_grad(set_to_none=True)
    obs = make_observation("cuda")
    output = encoder(obs)
    assert output.shape == (2, 258, 384) and torch.isfinite(output).all()
    output[..., 0].float().square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in encoder.rgb_encoder.backbone.parameters())
    assert not encoder.rgb_encoder.backbone.training
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.rgb_fusion.semantic_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["sat", "dqrise", "dexlatent"])
@pytest.mark.parametrize("mode", ["center", "group_mean"])
def test_policy_observation_roundtrip(kind, mode):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.common.checkpoint import restore_policy
    from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
    from dexmani_policy.common.inference import prepare_observations
    from dexmani_policy.common.normalizer import normalization_feature_shapes
    from dexmani_policy.training.common.config import load_config

    config = load_config(kind)
    assert config.agent.rgb_semantic_sampling == "center"
    config.agent.rgb_semantic_sampling = mode
    agent_config = OmegaConf.to_container(config.agent, resolve=True)
    if kind == "dqrise":
        agent_config["num_codes"] = 8
    policy = hydra.utils.instantiate(agent_config)
    dims = {
        "action": 19, "joint_state": 19, "point_cloud": 3,
        "rgb": 3, "camera_intrinsic": 9, "camera_extrinsic": 12,
    }
    policy.configure_normalizer(dims, normalization_feature_shapes(dims, config.normalization))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    raw = {key: value[:2] for key, value in make_observation("cpu").items()}
    raw["rgb"] = raw["rgb"].permute(0, 2, 3, 1)
    obs = prepare_observations(
        {key: value.numpy() for key, value in raw.items()},
        normalization_dims=dims, n_obs_steps=2,
        rgb_preprocessing=rgb_preprocessing_kwargs(config.dataset), device=device,
    )
    # Keep pinhole RGB at input resolution until the encoder's tracked resize.
    assert obs["rgb"].shape == (1, 2, 3, 240, 320)
    with torch.no_grad():
        expected = policy._encode_obs(obs)
    weights = policy.state_dict()
    effective = {
        "agent": {**agent_config, **policy.effective_config},
        "normalization_dims": dims,
        "normalization": dict(config.normalization),
    }
    del policy
    restored = restore_policy(effective, weights, device)
    assert restored.effective_config["rgb_semantic_sampling"] == mode
    with torch.no_grad():
        actual = restored._encode_obs(obs)
    assert actual.shape == (1, 258, 384) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
