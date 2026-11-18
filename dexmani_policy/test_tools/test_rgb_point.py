import pytest
import torch

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import ImageProjection, project_to_image
from dexmani_policy.agents.obs_encoder.pointcloud.rgb_patch_fusion import RGBPatchFusion
from dexmani_policy.common.image_util import resize_image_transform


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resized_camera_projection_coordinates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    points = torch.tensor([[[1, 2, 2], [-1, 0, 2], [0, 0, -3]]], device=device, dtype=torch.float32)
    intrinsics = points.new_tensor([[[8, 0, 31.5], [0, 8, 15.5], [0, 0, 1]]])
    extrinsics = points.new_tensor([[[0, -1, 0, 0.5], [1, 0, 0, -0.25], [0, 0, 1, 2]]])
    resize = resize_image_transform((32, 64), (16, 32), device=device).unsqueeze(0)
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = device == "cuda"
        with torch.autocast(device, dtype=torch.bfloat16):
            projection = project_to_image(points, intrinsics, extrinsics, resize, (16, 32))
        assert torch.backends.cuda.matmul.allow_tf32 == (device == "cuda")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    assert projection.valid.tolist() == [[True, True, False]]
    expected = points.new_tensor([[[-0.09375, 0.09375], [0.03125, -0.15625], [0, 0]]])
    torch.testing.assert_close(projection.grid_xy, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sampling", ["center", "group_mean"])
def test_patch_sampling_values_and_gradients(sampling):
    fusion = RGBPatchFusion(4, 2)
    rgb = torch.tensor([[[[1., 3., 5., 8.], [7., -2., 4., 1.]],
                         [[9., 2., 6., 0.], [3., 5., 1., 4.]]]], requires_grad=True)
    geometry = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    if sampling == "center":
        grid = torch.tensor([[[[-0.5, -0.5]], [[0, 0]], [[0.5, 0.5]]]])
        valid = torch.tensor([[[True], [True], [False]]])
        sampled = torch.stack((rgb[:, 0, 0], rgb.mean(dim=(1, 2)), rgb[:, 1, 1]), dim=1)
    else:
        grid = torch.tensor([[[[-0.5, -0.5], [0.5, -0.5]], [[0.5, 0.5], [-0.5, 0.5]],
                              [[-0.5, -0.5], [0.5, 0.5]]]])
        valid = torch.tensor([[[True, True], [True, False], [False, False]]])
        sampled = torch.stack((rgb[:, 0].mean(dim=1), rgb[:, 1, 1], rgb[:, 0, 0]), dim=1)

    actual = fusion(geometry, rgb, ImageProjection(grid, valid))
    semantic = fusion.semantic_projection(fusion.rgb_norm(sampled))
    expected = torch.cat((fusion.geometry_norm(geometry), torch.where(valid.any(-1)[..., None], semantic, 0)), -1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), rgb),
        torch.autograd.grad(expected.square().sum(), rgb),
    )
