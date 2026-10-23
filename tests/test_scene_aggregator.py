"""CPU-only correctness tests for geometry-aware scene token refinement."""

import unittest

import torch

from dexmani_policy.agents.obs_encoder.interaction.scene_aggregator import SceneAggregator


class SceneAggregatorRefinementTest(unittest.TestCase):
    def make_inputs(self, *, masked=False):
        torch.manual_seed(11)
        tokens = torch.randn(2, 32, 32)
        centers = torch.randn(2, 32, 3) * 0.1
        valid = torch.ones(2, 32, dtype=torch.bool)
        if masked:
            valid[:, -4:] = False
        return tokens, centers, valid

    def make_encoder(self, **kwargs):
        config = dict(token_channels=32, num_scene_tokens=6, route_dim=24,
                      relation_dim=32, refinement_dim=32)
        return SceneAggregator(**(config | kwargs))

    def test_opt_in_preserves_default_path(self):
        tokens, centers, valid = self.make_inputs()
        baseline = self.make_encoder().eval()
        refined = self.make_encoder(use_cross_refinement=True).eval()
        report = refined.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(report.unexpected_keys)
        self.assertTrue(all(key.startswith("cross_refinement.") or key == "refinement_gain" for key in report.missing_keys))
        with torch.no_grad():
            refined.refinement_gain.zero_()
            original = baseline(tokens, centers, patch_mask=valid)
            upgraded = refined(tokens, centers, patch_mask=valid)
        for key in original:
            torch.testing.assert_close(original[key], upgraded[key])

    def test_gradient_and_masked_values(self):
        tokens, centers, valid = self.make_inputs(masked=True)
        tokens.requires_grad_()
        model = self.make_encoder(use_cross_refinement=True)
        outputs = model(tokens, centers, patch_mask=valid, return_intermediate=True)
        self.assertEqual(outputs["scene_token"].shape, (2, 6, 32))
        self.assertEqual(outputs["refinement_update"].shape, (2, 6, 32))
        self.assertTrue(torch.isfinite(outputs["scene_token"]).all())
        outputs["scene_token"].square().mean().backward()
        self.assertIsNotNone(model.cross_refinement.q.weight.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

        corrupted_tokens, corrupted_centers = tokens.detach().clone(), centers.clone()
        corrupted_tokens[~valid] = float("nan")
        corrupted_centers[~valid] = float("inf")
        with torch.no_grad():
            clean = model(tokens.detach(), centers, patch_mask=valid)
            corrupted = model(corrupted_tokens, corrupted_centers, patch_mask=valid)
        for key in clean:
            torch.testing.assert_close(clean[key], corrupted[key])

    def test_permutation_invariance(self):
        tokens, centers, valid = self.make_inputs()
        permutation = torch.randperm(tokens.shape[1])
        model = self.make_encoder(use_cross_refinement=True).eval()
        with torch.no_grad():
            original = model(tokens, centers, patch_mask=valid)
            shuffled = model(tokens[:, permutation], centers[:, permutation], patch_mask=valid[:, permutation])
        for key in original:
            torch.testing.assert_close(original[key], shuffled[key], atol=1e-5, rtol=1e-5)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.make_encoder(use_cross_refinement=True, refinement_dim=31)


if __name__ == "__main__":
    unittest.main()
