"""Multimodal wiring, Base-aligned geometry and end-to-end gradient checks."""

from unittest.mock import patch

import pytest
import torch

from dexmani_policy.agents.obs_encoder import multimodal_interaction as multimodal
from dexmani_policy.agents.obs_encoder.multimodal_interaction import MultimodalInteractionEncoder


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


@pytest.fixture(params=[
    ("center", 3), ("center", 6), ("group_mean", 3), ("group_mean", 6),
])
def encoder(request, execution, rgb_config):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    device, _ = execution
    sampling, pc_dim = request.param
    return MultimodalInteractionEncoder(
        pc_dim=pc_dim, num_points=64,
        patch_config={"num_patches": 16, "group_size": 8},
        rgb_backbone_config=rgb_config, rgb_semantic_sampling=sampling,
    ).to(device).eval()


def make_observation(device, point_channels=6, num_points=96):
    points = torch.randn(2, num_points, 3, device=device) * 0.06
    points[..., 2] += 1
    if point_channels > 3:
        attributes = torch.rand(2, num_points, point_channels - 3, device=device)
        points = torch.cat((points, attributes), dim=-1)
    position = points.new_tensor([[0.34, -0.19, 0.42], [-0.24, -0.32, 0.31]])
    rotation = points.new_tensor([1, 0, 0, 0, 1, 0]).expand(2, -1)
    return {
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


def test_default_scene_token_count(rgb_config):
    pytest.importorskip("pytorch3d.ops", reason="Point grouping requires PyTorch3D")
    model = MultimodalInteractionEncoder(
        num_points=128, rgb_backbone_config=rgb_config,
    ).eval()
    obs = make_observation("cpu", point_channels=3, num_points=160)
    with torch.no_grad():
        output = model(obs)
    assert output["scene_tokens"].shape == (2, 128, 192)


def test_forward_backward_and_zero_force(encoder, execution):
    device, dtype = execution
    encoder.train()
    obs = make_observation(device, point_channels=encoder.patch_encoder.input_channels)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        output = encoder(obs)
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
        zero_output = encoder(obs)
    torch.testing.assert_close(zero_output["hand_tokens"], geometry[0], rtol=0, atol=0)


def test_base_aligned_geometry_contract(encoder, execution):
    device, dtype = execution
    obs = make_observation(device, point_channels=9)
    original = {key: value.clone() for key, value in obs.items()}
    points = multimodal.preprocess_point_cloud(
        obs["point_cloud"], encoder.num_points,
        encoder.patch_encoder.fps_random_config, training=False,
    )
    captured = {}

    def track_patches(_module, inputs, output):
        captured.update(patch_input=inputs[0], patches=output)

    def track_hand(_module, inputs, output):
        captured.update(hand_inputs=inputs, hand=output)

    with (
        patch.object(multimodal, "preprocess_point_cloud", return_value=points) as preprocess,
        patch.object(multimodal, "project_to_image", wraps=multimodal.project_to_image) as project,
        patch.object(
            encoder.relative_attention, "forward", wraps=encoder.relative_attention.forward,
        ) as attention,
        encoder.patch_encoder.register_forward_hook(track_patches),
        encoder.hand_encoder.register_forward_hook(track_hand),
        torch.no_grad(),
        torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32),
    ):
        output = encoder(obs)

    preprocess.assert_called_once_with(
        obs["point_cloud"], encoder.num_points,
        encoder.patch_encoder.fps_random_config, training=False,
    )
    assert captured["patch_input"] is points
    patches = captured["patches"]
    assert output["scene_xyz_base"] is patches["xyz"]

    if encoder.rgb_semantic_sampling == "center":
        expected_xyz = patches["xyz"]
    else:
        expected_xyz = multimodal.index_points(points[..., :3], patches["neighbor_idx"]).flatten(1, 2)
    project.assert_called_once()
    projection_inputs = project.call_args.kwargs
    torch.testing.assert_close(projection_inputs["points_world"], expected_xyz, rtol=0, atol=0)
    torch.testing.assert_close(
        projection_inputs["T_camera_from_world"], obs["camera_extrinsic"].reshape(2, 3, 4),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        projection_inputs["intrinsics"], obs["camera_intrinsic"].reshape(2, 3, 3), rtol=0, atol=0,
    )

    assert captured["hand_inputs"][0] is obs["eef_pose"]
    assert captured["hand_inputs"][1] is obs["fingertip_xyz"]
    hand = captured["hand"]
    assert output["hand_anchors_base"] is hand["anchors_base"]
    attention.assert_called_once()
    attention_inputs = attention.call_args.kwargs
    assert attention_inputs["body_tokens"] is hand["tokens"]
    assert attention_inputs["scene_tokens"] is output["scene_tokens"]
    assert attention_inputs["anchors_base"] is hand["anchors_base"]
    assert attention_inputs["scene_xyz_base"] is patches["xyz"]
    for key in obs:
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("backbone", ["dino", "dinov3", "siglip"])
def test_config_and_weights_roundtrip(rgb_config, backbone):
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
    obs = make_observation("cpu")
    with torch.no_grad():
        expected = encoder(obs)
        actual = restored(obs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("name,value", [
    ("rgb_backbone_name", "resnet"),
    ("rgb_semantic_sampling", "unknown"),
])
def test_invalid_config(rgb_config, name, value):
    with pytest.raises(ValueError, match=name):
        MultimodalInteractionEncoder(rgb_backbone_config=rgb_config, **{name: value})
