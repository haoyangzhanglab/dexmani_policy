"""定向 CPU 检查；运行：python -m scripts.utils.check_point_patch_semantics。

PointPatch 接线检查固定 FPS/KNN 输出，实际执行 PatchEncoder、融合、RoPE
和反向传播；不将此检查视为 PyTorch3D/CUDA、预训练 backbone 或策略验证。
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.point_patch import PointPatchEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.semantic_fusion import (
    PointPatchSemanticFusion,
    build_patch_image_weights,
    project_points_to_images,
    resize_crop_transform,
)


class SemanticFusionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.uv = torch.rand(2, 3, 17, 2) * torch.tensor([20., 12.]) - 0.5
        self.weight = torch.rand(2, 3, 17)
        self.weight[self.weight < 0.3] = 0
        self.idx = torch.randint(17, (2, 5, 8))
        self.features = torch.randn(2, 3, 15, 7)

    def test_matrix_matches_grid_sample_and_feature_gradient(self):
        uv = self.uv.clone()
        uv[0, 0, 0] = torch.tensor([-0.5, -0.5])
        uv[0, 1, 1] = torch.tensor([19.5, 5.])
        uv[1, 0, 2] = float("nan")
        features = self.features.requires_grad_()
        geometry = build_patch_image_weights(self.idx, uv, self.weight, (12, 20), (3, 5))
        actual = geometry["pool_weight"] @ features.flatten(1, 2)

        # 独立 oracle：官方 grid_sample 按点插值，再对视角和成员点池化。
        in_bounds = (uv[..., 0] >= -0.5) & (uv[..., 0] < 19.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] < 11.5)
        confidence = torch.where(in_bounds & torch.isfinite(uv).all(-1), self.weight, 0)
        totals = confidence.sum(1, keepdim=True)
        normalized = confidence / torch.where(totals > 0, totals, 1)
        grid = 2 * (torch.nan_to_num(uv) + 0.5) / torch.tensor([20., 12.]) - 1
        sampled = F.grid_sample(
            features.permute(0, 1, 3, 2).reshape(6, 7, 3, 5),
            grid.reshape(6, 17, 1, 2), mode="bilinear", padding_mode="border", align_corners=False,
        ).reshape(2, 3, 7, 17).permute(0, 1, 3, 2)
        point_features = (sampled * normalized.unsqueeze(-1)).sum(1)
        valid_count = index_points(totals[:, 0] > 0, self.idx).sum(-1).clamp_min(1)
        expected = index_points(point_features, self.idx).sum(-2) / valid_count.unsqueeze(-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        probe = torch.randn_like(actual)
        grad_actual = torch.autograd.grad((actual * probe).sum(), features, retain_graph=True)[0]
        grad_expected = torch.autograd.grad((expected * probe).sum(), features)[0]
        torch.testing.assert_close(grad_actual, grad_expected, atol=2e-6, rtol=2e-6)

    def test_view_count_does_not_bias_member_pooling(self):
        uv = torch.tensor([[[[0., 0.], [1., 0.]], [[0., 0.], [1., 0.]]]])
        weight = torch.tensor([[[1., 1.], [1., 0.]]])
        idx = torch.tensor([[[0, 1]]])
        features = torch.tensor([[[10.], [30.], [10.], [30.]]])
        geometry = build_patch_image_weights(idx, uv, weight, (1, 2), (1, 2))
        torch.testing.assert_close(geometry["pool_weight"] @ features, torch.tensor([[[20.]]]))
        weight[:, :, 1] = 0
        geometry = build_patch_image_weights(idx, uv, weight, (1, 2), (1, 2))
        torch.testing.assert_close(geometry["pool_weight"] @ features, torch.tensor([[[10.]]]))
        self.assertEqual(geometry["semantic_coverage"].item(), 0.5)

    def test_all_invalid_is_exact_geometry_identity_despite_bias(self):
        fusion = PointPatchSemanticFusion(7, 24)
        with torch.no_grad():
            fusion.proj.bias.fill_(9)
        local = torch.randn(2, 5, 24)
        result = fusion(local, self.idx, self.features, self.uv * float("nan"), self.weight,
                        (12, 20), (3, 5), return_intermediate=True)
        self.assertTrue(torch.equal(result["fused_token"], local))
        self.assertEqual(result["semantic_gate"].count_nonzero().item(), 0)
        self.assertEqual(result["pool_weight"].count_nonzero().item(), 0)

    def test_permuting_views_and_point_order_preserves_correspondence(self):
        original = build_patch_image_weights(self.idx, self.uv, self.weight, (12, 20), (3, 5))
        expected = original["pool_weight"] @ self.features.flatten(1, 2)
        view_perm = torch.tensor([2, 0, 1])
        point_perm = torch.randperm(17)
        inverse = torch.argsort(point_perm)
        permuted = build_patch_image_weights(
            inverse[self.idx], self.uv[:, view_perm][:, :, point_perm],
            self.weight[:, view_perm][:, :, point_perm], (12, 20), (3, 5),
        )
        actual = permuted["pool_weight"] @ self.features[:, view_perm].flatten(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_projection_extrinsic_resize_crop_and_occlusion(self):
        xyz = torch.tensor([[[.25, 0., 2.], [.25, 0., 0.], [100., 0., 2.], [.25, 0., 2.2], [.25, 0., float("nan")]]])
        depth = torch.ones(1, 1, 8, 10)
        intrinsic = torch.tensor([[[[4., 0., 4.], [0., 4., 3.], [0., 0., 1.]]]])
        extrinsic = torch.eye(4).reshape(1, 1, 4, 4)
        extrinsic[..., 0, 3] = -.25
        extrinsic[..., 2, 3] = -1
        transform = resize_crop_transform((8, 10), (16, 20), (2, 3))
        result = project_points_to_images(xyz, depth, intrinsic, extrinsic, transform, depth_atol=.01, depth_rtol=0)
        torch.testing.assert_close(result["point_uv"][0, 0, 0], torch.tensor([5.5, 4.5]))
        torch.testing.assert_close(result["view_weight"], torch.tensor([[[1., 0., 0., 0., 0.]]]))
        for invalid_depth in (0., float("nan")):
            depth[..., 3, 4] = invalid_depth
            result = project_points_to_images(xyz, depth, intrinsic, extrinsic)
            self.assertEqual(result["view_weight"].count_nonzero().item(), 0)
        with self.assertRaises(TypeError):
            project_points_to_images(xyz, torch.ones_like(depth, dtype=torch.int32), intrinsic, extrinsic)

    def test_crop_outside_removes_previously_visible_point(self):
        transform = resize_crop_transform((8, 10), (8, 10), (0, 5))
        raw_uv = torch.tensor([1., 3., 1.])
        uv = (transform @ raw_uv)[:2].reshape(1, 1, 1, 2)
        geometry = build_patch_image_weights(torch.zeros(1, 1, 1, dtype=torch.long), uv,
                                             torch.ones(1, 1, 1), (8, 5), (2, 1))
        self.assertFalse(geometry["semantic_valid_mask"].item())

    def test_negative_member_index_is_rejected(self):
        indices = self.idx.clone()
        indices[0, 0, 0] = -1
        with self.assertRaises(RuntimeError):
            build_patch_image_weights(indices, self.uv, self.weight, (12, 20), (3, 5))

    def test_projection_diagnostics_keep_residuals_before_visibility_filter(self):
        xyz = torch.tensor([[[0., 0., 1.], [0., 0., 1.5]]])
        intrinsic = torch.eye(3).reshape(1, 1, 3, 3)
        extrinsic = torch.eye(4).reshape(1, 1, 4, 4)
        result = project_points_to_images(
            xyz, torch.ones(1, 1, 2, 2), intrinsic, extrinsic, return_intermediate=True,
        )
        self.assertTrue(result["in_image"].all())
        self.assertTrue(result["depth_valid"].all())
        torch.testing.assert_close(result["depth_residual"], torch.tensor([[[0., 0.5]]]))
        torch.testing.assert_close(result["view_weight"], torch.tensor([[[1., 0.]]]))

    def test_fusion_gradients_and_cpu_bfloat16(self):
        fusion = PointPatchSemanticFusion(7, 24)
        features = self.features.clone().requires_grad_()
        local = torch.randn(2, 5, 24, requires_grad=True)
        uv = self.uv.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = fusion(local, self.idx, features, uv, self.weight, (12, 20), (3, 5), True)
            loss = (result["fused_token"] * torch.randn_like(local)).sum()
        loss.backward()
        for gradient in (features.grad, local.grad, fusion.proj.weight.grad, fusion.gate.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNone(uv.grad)
        self.assertEqual(result["pool_weight"].dtype, torch.float32)

    def test_point_patch_integration_without_repeating_context_blocks(self):
        def fixed_fps(points, count, **kwargs):
            indices = torch.arange(count).expand(points.shape[0], -1)
            return index_points(points, indices), indices

        def fixed_knn(count, support, query):
            return torch.cdist(query, support).topk(count, largest=False).indices

        options = dict(token_channels=24, num_patches=4, group_size=4, depth=2, num_heads=4)
        base = PointPatchEncoder(**options).eval()
        fused = PointPatchEncoder(**options, semantic_channels=7).eval()
        missing, unexpected = fused.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(name.startswith("semantic_fusion.") for name in missing))
        self.assertEqual(unexpected, [])
        points = torch.randn(2, 17, 6)
        kwargs = dict(image_tokens=self.features, point_uv=self.uv, image_hw=(12, 20), patch_grid_size=(3, 5))
        prefix = "dexmani_policy.agents.obs_encoder.pointcloud.point_patch."
        calls = []
        hooks = [block.register_forward_hook(lambda *args: calls.append(1)) for block in fused.blocks]
        try:
            with patch(prefix + "farthest_point_sample", side_effect=fixed_fps), patch(prefix + "knn_point", side_effect=fixed_knn):
                original = base(points, return_intermediate=True)
                absent = fused(points, view_weight=torch.zeros_like(self.weight), **kwargs)
                present = fused(points, return_intermediate=True, view_weight=self.weight, **kwargs)
            self.assertTrue(torch.equal(original["patch_token"], absent["patch_token"]))
            torch.testing.assert_close(original["local_token"], present["local_token"])
            self.assertEqual(present["patch_token"].shape, (2, 4, 24))
            self.assertEqual(len(calls), 4)  # 两次 fused forward，每次只执行两个 block。
            (present["patch_token"] * torch.randn_like(present["patch_token"])).sum().backward()
            self.assertGreater(fused.semantic_fusion.proj.weight.grad.abs().sum().item(), 0)
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
