"""Multimodal wiring, frame conversion and end-to-end gradient checks."""

from unittest.mock import patch

import pytest
import torch

from dexmani_policy.agents.obs_encoder import multimodal_interaction as multimodal
from dexmani_policy.agents.obs_encoder.multimodal_interaction import MultimodalInteractionEncoder
from dexmani_policy.agents.obs_encoder.rgb_point import RGBPointObservationEncoder


@pytest.fixture
def rgb_config():
    return {
        "tune_mode": "full",
        "image_size": (32, 32),
        "architecture": {
            "model_name": "vit_small_patch16_dinov3",
            "img_size": 32, "embed_dim": 48, "depth": 1, "num_heads": 4,
        },
    }


@pytest.fixture(params=[
    ("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16),
])
def execution(request):
    device, dtype = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    torch.manual_seed(17)
    return torch.device(device), dtype


@pytest.fixture(params=["center", "group_mean"])
def encoder(request, execution, rgb_config):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    device, _ = execution
    return MultimodalInteractionEncoder(
        pc_dim=6, num_points=64,
        patch_config={"num_patches": 16, "group_size": 8},
        rgb_backbone_config=rgb_config, rgb_semantic_sampling=request.param,
    ).to(device).eval()


def make_observation(device, pc_dim=6, num_points=96):
    points = torch.randn(2, num_points, 3, device=device) * 0.06
    points[..., 2] += 1
    if pc_dim == 6:
        points = torch.cat((points, torch.rand_like(points)), dim=-1)
    transform = torch.tensor([
        [[0, -1, 0, 0.35], [1, 0, 0, -0.20], [0, 0, 1, -0.60]],
        [[1, 0, 0, -0.25], [0, 0, -1, 0.70], [0, 1, 0, 0.30]],
    ], device=device)
    position = points.new_tensor([[0.34, -0.19, 0.42], [-0.24, -0.32, 0.31]])
    rotation = points.new_tensor([1, 0, 0, 0, 1, 0]).expand(2, -1)
    obs = {
        "point_cloud": points,
        "rgb": torch.randint(0, 256, (2, 3, 24, 40), dtype=torch.uint8, device=device),
        "camera_intrinsic": points.new_tensor(
            [[24, 0, 19.5], [0, 24, 11.5], [0, 0, 1]],
        ).expand(2, -1, -1).flatten(1),
        "camera_extrinsic": points.new_tensor([
            [[1, 0, 0, 0.10], [0, 1, 0, 0], [0, 0, 1, 0.20]],
            [[0, -1, 0, -0.04], [1, 0, 0, 0.03], [0, 0, 1, 0.10]],
        ]).flatten(1),
        "eef_pose": torch.cat((position, rotation), dim=-1),
        "fingertip_xyz": position[:, None] + torch.randn(2, 5, 3, device=device) * 0.04,
        "finger_tactile_force": torch.randn(2, 5, 3, device=device) * 3,
    }
    return obs, transform


def test_default_scene_token_count(rgb_config):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    model = MultimodalInteractionEncoder(
        num_points=128, rgb_backbone_config=rgb_config,
    ).eval()
    obs, transform = make_observation("cpu", pc_dim=3, num_points=160)
    with torch.no_grad():
        output = model(obs, T_base_from_world=transform)
    assert output["scene_tokens"].shape == (2, 128, 192)


def test_forward_backward_and_zero_force(encoder, execution):
    device, dtype = execution
    encoder.train()
    obs, transform = make_observation(device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        output = encoder(obs, T_base_from_world=transform)
    assert {key: tuple(value.shape) for key, value in output.items()} == {
        "scene_tokens": (2, 16, 192), "scene_xyz_base": (2, 16, 3),
        "hand_tokens": (2, 6, 192), "hand_anchors_base": (2, 6, 3),
    }
    assert all(torch.isfinite(value).all() for value in output.values())
    loss = (
        output["scene_tokens"][..., 0].float().square().mean()
        + output["hand_tokens"][..., 0].float().square().mean()
    )
    loss.backward()
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert encoder.rgb_fusion.semantic_projection.weight.grad.abs().sum() > 0

    geometry = []
    obs["finger_tactile_force"] = torch.zeros_like(obs["finger_tactile_force"])
    with (
        encoder.relative_attention.register_forward_hook(
            lambda _module, _inputs, output: geometry.append(output),
        ),
        torch.no_grad(),
        torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32),
    ):
        zero_output = encoder(obs, T_base_from_world=transform)
    torch.testing.assert_close(zero_output["hand_tokens"], geometry[0], rtol=0, atol=0)


def test_base_geometry_and_world_projection(encoder, execution):
    device, dtype = execution
    obs, transform = make_observation(device)
    original_points = obs["point_cloud"].clone()
    world = multimodal.preprocess_point_cloud(obs["point_cloud"], encoder.num_points, training=False)
    captured: dict[str, torch.Tensor] = {}

    def track_patches(_module, inputs, output):
        captured.update(points_base=inputs[0], neighbor_idx=output["neighbor_idx"])

    with (
        patch.object(
            multimodal, "preprocess_point_cloud", wraps=multimodal.preprocess_point_cloud,
        ) as preprocess,
        patch.object(multimodal, "project_to_image", wraps=multimodal.project_to_image) as project,
        encoder.patch_encoder.register_forward_hook(track_patches),
        torch.no_grad(),
        torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32),
    ):
        output = encoder(obs, T_base_from_world=transform)

    preprocess.assert_called_once()
    assert world.shape == (2, 64, 6)
    homogeneous = torch.cat((world[..., :3].double(), torch.ones_like(world[..., :1])), dim=-1)
    expected_base = torch.einsum("bij,bnj->bni", transform.double(), homogeneous).float()
    torch.testing.assert_close(captured["points_base"][..., :3], expected_base, rtol=0, atol=2e-7)
    torch.testing.assert_close(captured["points_base"][..., 3:], world[..., 3:], rtol=0, atol=0)
    assert captured["points_base"].dtype == torch.float32

    centers = output["scene_xyz_base"]
    center_idx = (expected_base[:, None] - centers[:, :, None]).square().sum(-1).argmin(-1)
    batch = torch.arange(2, device=device)[:, None]
    torch.testing.assert_close(centers, expected_base[batch, center_idx], rtol=0, atol=2e-7)
    expected_anchors = torch.cat((obs["eef_pose"][:, None, :3], obs["fingertip_xyz"]), dim=1)
    torch.testing.assert_close(output["hand_anchors_base"], expected_anchors, rtol=0, atol=0)
    assert centers.dtype == output["hand_anchors_base"].dtype == torch.float32

    if encoder.rgb_semantic_sampling == "center":
        expected_world = world[batch, center_idx, :3]
    else:
        neighbors = captured["neighbor_idx"]
        expected_world = world[batch[:, :, None], neighbors, :3].flatten(1, 2)
    project.assert_called_once()
    projection_inputs = project.call_args.kwargs
    torch.testing.assert_close(projection_inputs["points_world"], expected_world, rtol=0, atol=2e-7)
    assert projection_inputs["points_world"].dtype == torch.float32
    torch.testing.assert_close(
        projection_inputs["T_camera_from_world"], obs["camera_extrinsic"].reshape(2, 3, 4),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        projection_inputs["intrinsics"], obs["camera_intrinsic"].reshape(2, 3, 3), rtol=0, atol=0,
    )
    torch.testing.assert_close(obs["point_cloud"], original_points, rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_rigid_transforms_stay_fp32_under_autocast(input_dtype):
    obs, transform = make_observation("cpu", pc_dim=3, num_points=4)
    world = obs["point_cloud"].to(input_dtype)
    transform = transform.to(input_dtype)
    homogeneous = torch.cat((world.double(), torch.ones_like(world[..., :1])), dim=-1)
    expected = torch.einsum("bij,bnj->bni", transform.double(), homogeneous).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base = multimodal._transform_points(world, transform)
        recovered = multimodal._inverse_transform_points(base, transform)
    assert base.dtype == recovered.dtype == torch.float32
    torch.testing.assert_close(base, expected, rtol=0, atol=2e-7)
    torch.testing.assert_close(recovered, world.float(), rtol=0, atol=2e-7)


def test_rigid_transforms_with_tf32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(19)
    rotation = torch.linalg.qr(torch.randn(2, 3, 3, dtype=torch.float64, device="cuda")).Q.float()
    transform = torch.cat((rotation, torch.randn(2, 3, 1, device="cuda")), dim=-1)
    world = torch.randn(2, 1024, 3, device="cuda")
    expected = (world.double() @ rotation.double().transpose(1, 2) + transform[:, None, :, 3]).float()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        base = multimodal._transform_points(world, transform)
        recovered = multimodal._inverse_transform_points(base, transform)
    torch.testing.assert_close(base, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(recovered, world, rtol=0, atol=1e-6)


@pytest.mark.parametrize("backbone", ["dino", "dinov3", "siglip"])
def test_current_config_and_weights_roundtrip(rgb_config, backbone):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    if backbone != "dinov3":
        rgb_config["architecture"] = {
            "hidden_size": 48, "intermediate_size": 96, "num_hidden_layers": 1,
            "num_attention_heads": 4, "image_size": 32, "patch_size": 16,
        }
        if backbone == "dino":
            rgb_config["architecture"]["model_type"] = "dinov2"
    encoder = MultimodalInteractionEncoder(
        pc_dim=6, num_points=64,
        patch_config={"num_patches": 16, "group_size": 8, "geometry_scale": 0.12},
        spatial_config={"depth": 1, "rope_scale": 0.07},
        hand_geometry_config={"wrist_scale": 0.30, "finger_scale": 0.06},
        relative_geometry_scale=0.20, rgb_semantic_sampling="group_mean",
        rgb_backbone_name=backbone, rgb_backbone_config=rgb_config,
    ).eval()
    restored = MultimodalInteractionEncoder(**encoder.effective_config).eval()
    restored.load_state_dict(encoder.state_dict(), strict=True)
    assert restored.effective_config == encoder.effective_config
    obs, transform = make_observation("cpu")
    with torch.no_grad():
        expected = encoder(obs, T_base_from_world=transform)
        actual = restored(obs, T_base_from_world=transform)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("encoder_class", [MultimodalInteractionEncoder, RGBPointObservationEncoder])
def test_patch_rgb_backbone_required(encoder_class):
    with pytest.raises(ValueError, match="rgb_backbone_name"):
        encoder_class(rgb_backbone_name="resnet")


@pytest.mark.parametrize("kwargs,error,match", [
    ({"patch_config": {"channels": 384}}, TypeError, "channels"),
    ({"spatial_config": {"channels": 192}}, TypeError, "channels"),
    ({"rgb_semantic_sampling": "unknown"}, ValueError, "rgb_semantic_sampling"),
])
def test_invalid_config(rgb_config, kwargs, error, match):
    with pytest.raises(error, match=match):
        MultimodalInteractionEncoder(rgb_backbone_config=rgb_config, **kwargs)


def test_transform_is_required_and_has_exact_shape(rgb_config):
    encoder = MultimodalInteractionEncoder(rgb_backbone_config=rgb_config)
    obs = {"point_cloud": torch.zeros(2, 4, 3)}
    with pytest.raises(TypeError, match="T_base_from_world"):
        encoder(obs)
    for shape in ((2, 4, 4), (3, 4), (1, 3, 4)):
        with pytest.raises(ValueError, match="T_base_from_world"):
            encoder(obs, T_base_from_world=torch.zeros(shape))
