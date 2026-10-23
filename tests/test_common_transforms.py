import unittest

import numpy as np
import torch

from dexmani_policy.common.experiments import validate_inference_settings
from dexmani_policy.common.normalization import AffineNormalizer
from dexmani_policy.common.rgb_preprocessing import (
    preprocess_training_rgb,
    preprocess_validation_rgb,
)


class RGBPreprocessingTest(unittest.TestCase):
    def setUp(self):
        image = np.arange(45, dtype=np.uint8).reshape(3, 5, 3) * np.uint8(5)
        self.rgb = np.stack([image, image])

    def test_no_resize_preserves_raw_layout_and_dtype(self):
        rgb = self.rgb[:, ::2]
        output, transform = preprocess_validation_rgb(
            rgb, resize_hw=None, center_crop_hw=None, keep_uint8=False, return_transform=True,
        )
        torch.testing.assert_close(output, torch.from_numpy(rgb))
        torch.testing.assert_close(transform, torch.eye(3))
        self.assertTrue(output.is_contiguous())

    def test_validation_resize_crop_values_and_geometry(self):
        expected = torch.tensor([
            [23.75, 36.25, 48.75],
            [70.625, 83.125, 95.625],
            [126.875, 139.375, 151.875],
        ])
        expected = torch.stack([expected + offset for offset in (0, 5, 10)]) / 255
        expected_transform = torch.tensor([
            [6 / 5, 0, -1.9], [0, 4 / 3, 1 / 6], [0, 0, 1],
        ])
        original = self.rgb.copy()
        for keep_uint8, float_spatial in ((False, False), (True, False), (True, True)):
            with self.subTest(keep_uint8=keep_uint8, float_spatial=float_spatial):
                output, transform = preprocess_validation_rgb(
                    self.rgb[None], resize_hw=(4, 6), center_crop_hw=(3, 3),
                    keep_uint8=keep_uint8, float_spatial_before_uint8=float_spatial,
                    return_transform=True,
                )
                target = (expected * 255).round().to(torch.uint8) if keep_uint8 else expected
                torch.testing.assert_close(output, target.expand(1, 2, 3, 3, 3))
                torch.testing.assert_close(transform, expected_transform)
                self.assertTrue(output.is_contiguous())
        np.testing.assert_array_equal(self.rgb, original)

    def test_training_shares_random_crop_and_quantizes_after_color(self):
        outputs = []
        transforms = []
        original = self.rgb.copy()
        for keep_uint8 in (False, True):
            torch.manual_seed(11)
            output, transform = preprocess_training_rgb(
                self.rgb, resize_hw=(6, 8), random_crop_hw=(3, 4), keep_uint8=keep_uint8,
                color_augmentation=lambda value: value * 4 - 1, return_transform=True,
            )
            torch.testing.assert_close(output[0], output[1])
            self.assertEqual(output.shape, (2, 3, 3, 4))
            outputs.append(output)
            transforms.append(transform)
        self.assertEqual(outputs[0].min().item(), 0)
        self.assertEqual(outputs[0].max().item(), 1)
        torch.testing.assert_close(outputs[1], (outputs[0] * 255).round().to(torch.uint8))
        torch.testing.assert_close(transforms[0], transforms[1])
        np.testing.assert_array_equal(self.rgb, original)


class AffineNormalizationTest(unittest.TestCase):
    def test_streaming_statistics_preserve_source_storage(self):
        for dtype in (np.float32, np.float64):
            values = np.arange(210, dtype=dtype).reshape(14, 5, 3)[::2]
            original = values.copy()
            for source in (values, torch.from_numpy(values)):
                for mode in ("limits", "gaussian"):
                    with self.subTest(dtype=dtype, source=type(source), mode=mode):
                        normalizer = AffineNormalizer()
                        normalizer.fit_field(
                            "fingertip_xyz", (source[:1], source[1:4], source[4:]), mode=mode,
                        )
                        np.testing.assert_array_equal(values, original)
                        stats = normalizer["fingertip_xyz"].input_stats
                        torch.testing.assert_close(
                            stats["mean"], torch.from_numpy(original.mean(0)).float(),
                        )
                        torch.testing.assert_close(
                            stats["std"], torch.from_numpy(original.std(0, ddof=1)).float(),
                        )

    def test_streaming_statistics_and_inverse_with_constant_features(self):
        values = torch.tensor([[1, 2, 7], [3, 2, 9], [5, 2, 11]], dtype=torch.float64)
        for mode in ("limits", "gaussian"):
            with self.subTest(mode=mode):
                normalizer = AffineNormalizer()
                normalizer.fit_field("action", (values[:1], values[1:]), mode=mode)
                field = normalizer["action"]
                for key, expected in (
                    ("min", [1, 2, 7]), ("max", [5, 2, 11]),
                    ("mean", [3, 2, 9]), ("std", [2, 0, 2]),
                ):
                    torch.testing.assert_close(field.input_stats[key], torch.tensor(expected).float())
                normalized = field.normalize(values)
                torch.testing.assert_close(normalized[:, 1], torch.zeros(3))
                torch.testing.assert_close(field.unnormalize(normalized), values.float())

    def test_single_sample_and_nonfinite_statistics(self):
        normalizer = AffineNormalizer()
        values = torch.tensor([[2.0, -3.0]])
        normalizer.fit_field("action", [values])
        torch.testing.assert_close(normalizer["action"].normalize(values), torch.zeros_like(values))
        with self.assertRaisesRegex(ValueError, "Insufficient samples"):
            normalizer.fit_field("action", [values], mode="gaussian")
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Nonfinite"):
                normalizer.fit_field("action", [values, torch.tensor([[invalid, 1]])])


class InferenceSettingsTest(unittest.TestCase):
    def test_selection_record_requires_explicit_action_steps(self):
        record = {
            "inference": {
                "use_ema": False, "inference_steps": 1,
                "policy_seed_mode": "episode_seed", "action_steps": None,
            },
            "evaluation": {"effective_config": {"agent": {
                "horizon": 4, "n_obs_steps": 2, "n_action_steps": 2,
            }}},
        }
        with self.assertRaisesRegex(ValueError, "action_steps"):
            validate_inference_settings(record)
        record["inference"]["action_steps"] = 2
        validate_inference_settings(record)


if __name__ == "__main__":
    unittest.main()
