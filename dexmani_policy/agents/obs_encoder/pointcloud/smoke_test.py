import unittest

import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import MultiStagePointNet, PointNet
from dexmani_policy.agents.obs_encoder.pointcloud.pointnet_dense import PointNetDense
from dexmani_policy.agents.obs_encoder.pointcloud.registry import (
    build_pc_global_encoder,
    build_pc_patch_tokenizer,
)


class PointNetTest(unittest.TestCase):
    def test_shapes(self):
        for channels in (3, 6):
            with self.subTest(channels=channels), torch.no_grad():
                model = PointNet(input_channels=channels, output_channels=128).eval()
                result = model(torch.randn(2, 1024, channels))
                self.assertEqual(set(result), {"global_token"})
                self.assertEqual(result["global_token"].shape, (2, 128))
                self.assertTrue(torch.isfinite(result["global_token"]).all())
                self.assertEqual(model.out_dim, 128)
                self.assertEqual(model.out_shape, (1, 128))

    def test_unsupported_channels(self):
        for channels in (0, 2, 4, 5, 7):
            with self.subTest(channels=channels), self.assertRaisesRegex(ValueError, "input_channels"):
                PointNet(input_channels=channels, output_channels=128)

    def test_input_validation(self):
        model = PointNet(input_channels=6, output_channels=128)
        for shape in ((2, 6), (2, 4, 1024, 6), (2, 1024, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    def test_permutation_invariance(self):
        for channels in (3, 6):
            with self.subTest(channels=channels), torch.no_grad():
                model = PointNet(input_channels=channels, output_channels=128).eval()
                pointcloud = torch.randn(2, 1024, channels)
                permutation = torch.randperm(1024)
                torch.testing.assert_close(
                    model(pointcloud)["global_token"],
                    model(pointcloud[:, permutation])["global_token"],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_topology(self):
        for channels in (3, 6):
            with self.subTest(channels=channels), torch.no_grad():
                model = PointNet(input_channels=channels, output_channels=128).eval()
                expected_types = [nn.Linear, nn.LayerNorm, nn.ReLU] * 3
                expected_dims = [(channels, 64), (64, 128), (128, 256)]
                if channels == 6:
                    expected_types.append(nn.Linear)
                    expected_dims.append((256, 512))
                self.assertEqual([type(layer) for layer in model.mlp], expected_types)
                self.assertEqual(
                    [
                        (layer.in_features, layer.out_features)
                        for layer in model.mlp if isinstance(layer, nn.Linear)
                    ],
                    expected_dims,
                )
                self.assertEqual(
                    [layer.normalized_shape for layer in model.mlp if isinstance(layer, nn.LayerNorm)],
                    [(64,), (128,), (256,)],
                )
                projection = model.output_projection
                self.assertEqual([type(layer) for layer in projection], [nn.Linear, nn.LayerNorm])
                self.assertEqual(
                    (projection[0].in_features, projection[0].out_features),
                    (512 if channels == 6 else 256, 128),
                )
                self.assertEqual(projection[1].normalized_shape, (128,))
                pointcloud = torch.randn(2, 32, channels)
                expected = projection(model.mlp(pointcloud).max(dim=1).values)
                torch.testing.assert_close(model(pointcloud)["global_token"], expected)

    def test_registry(self):
        default = build_pc_global_encoder("dp3", pc_dim=6)
        override = build_pc_global_encoder("dp3", pc_dim=6, config={"output_channels": 128})
        idp3 = build_pc_global_encoder("idp3", pc_dim=6)
        self.assertIsInstance(default, PointNet)
        self.assertEqual(default.out_dim, 256)
        self.assertIsInstance(override, PointNet)
        self.assertEqual(override.out_dim, 128)
        self.assertIsInstance(idp3, MultiStagePointNet)
        self.assertEqual(idp3.out_dim, 256)


class PointNetDenseTest(unittest.TestCase):
    def test_topology_and_dense_equivariance(self):
        for channels in (3, 6):
            with self.subTest(channels=channels), torch.no_grad():
                model = PointNetDense(channels, num_points=32).eval()
                types = [nn.Linear, nn.LayerNorm, nn.ReLU] * 3
                dims = [(channels, 64), (64, 128), (128, 256)]
                if channels == 6:
                    types.append(nn.Linear)
                    dims.append((256, 512))
                self.assertEqual([type(layer) for layer in model.mlp], types)
                self.assertEqual(
                    [(m.in_features, m.out_features) for m in model.mlp if isinstance(m, nn.Linear)],
                    dims,
                )
                self.assertEqual([type(m) for m in model.final_proj], [nn.Linear, nn.LayerNorm])
                self.assertEqual(model.final_proj[0].in_features, 512 if channels == 6 else 256)
                self.assertEqual(model.final_proj[0].out_features, 128)
                self.assertEqual(model.final_proj[1].normalized_shape, (128,))
                self.assertFalse(model.supports_global_token)
                pc = torch.randn(2, 32, channels)
                result = model(pc)
                self.assertEqual(result.shape, (2, 32, 128))
                self.assertEqual(model.out_shape, (32, 128))
                permutation = torch.randperm(32)
                torch.testing.assert_close(model(pc[:, permutation]), result[:, permutation])
                # A point's feature cannot depend on any other point (no pooling).
                torch.testing.assert_close(model(pc[:, :1]), result[:, :1])

    def test_registry_and_validation(self):
        self.assertIsInstance(build_pc_patch_tokenizer("pointnet_dense", 6), PointNetDense)
        with self.assertRaisesRegex(ValueError, "hidden_dims"):
            build_pc_patch_tokenizer("pointnet_dense", 6, {"hidden_dims": [64]})
        for channels in (0, 2, 4, 5, 7):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                PointNetDense(channels)
        model = PointNetDense(6)
        for shape in ((2, 6), (2, 2, 32, 6), (2, 32, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))


if __name__ == "__main__":
    unittest.main()
