import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

from dexmani_policy.datasets.common.augmentation import PointColorNoise, PointCoordinateNoise
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset
from dexmani_policy.datasets.hand_prior.lamp_dataset import LAMPDataset
from dexmani_policy.datasets.hand_prior.temporal_codec_dataset import TemporalCodecDataset
from dexmani_policy.datasets.multitask_dataset import MultiTaskDataset
from dexmani_policy.training.common.hand_representation import shuffled_batches
from dexmani_policy.training.data import build_normalizer


class DataPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "episodes.zarr"
        self.root = zarr.open_group(str(self.path), mode="w")
        self.actions = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None], (12, 19)).copy()
        rng = np.random.default_rng(0)
        fields = {
            "action": self.actions,
            "joint_state": self.actions + 100,
            "point_cloud": rng.random((12, 8, 6), dtype=np.float32),
            "rgb": rng.integers(0, 256, (12, 6, 8, 3), dtype=np.uint8),
        }
        for key, values in fields.items():
            self.root.create_dataset(f"data/{key}", data=values, chunks=(3, *values.shape[1:]))
        self.root.create_dataset("meta/episode_ends", data=np.array([6, 12]))
        self.root.attrs.update(episode_ids=["train", "val"], dt=0.05)
        self.options = dict(
            zarr_path=str(self.path), sensor_modalities=["joint_state", "point_cloud"],
            horizon=4, obs_horizon=2, pad_before=1, pad_after=2,
            saved_split=dict(
                episode_ids=["train", "val"], trial_ids={"train": "a", "val": "b"},
                train_ids=["train"], val_ids=["val"], exclusions=[],
            ),
        )

    def test_padding_and_full_buffer_normalization(self):
        dataset = DemonstrationDataset(**self.options)
        np.testing.assert_array_equal(dataset[0]["action"][:, 0], [0, 0, 1, 2])
        np.testing.assert_array_equal(dataset[-1]["action"][:, 0], [4, 5, 5, 5])
        validation = dataset.get_validation_dataset()
        np.testing.assert_array_equal(validation[0]["action"][:, 0], [6, 6, 7, 8])
        self.assertIs(validation, dataset.get_validation_dataset())
        normalizer = build_normalizer(dataset, {"action": "limits"}, "action")
        torch.testing.assert_close(normalizer["action"].input_stats["max"], torch.full((19,), 11.0))
        with self.assertRaisesRegex(ValueError, "reuse"):
            next(validation.iter_normalization_data("action"))

    def test_buffer_reopens_after_serialization(self):
        dataset = DemonstrationDataset(**self.options)
        buffer = dataset.replay_buffer
        expected = buffer.read("action", slice(1, 5))
        restored = pickle.loads(pickle.dumps(buffer))
        self.assertIsNone(restored.group)
        self.assertIsNone(restored.data)
        self.assertEqual(restored.read_cache, {})
        np.testing.assert_array_equal(restored.read("action", slice(1, 5)), expected)
        self.assertIsNot(restored.group, buffer.group)
        self.assertEqual(restored.attrs, buffer.attrs)

    def test_normalizer_rejects_nonfinite_holdout_rows(self):
        self.root["data/action"][8, 0] = np.inf
        dataset = DemonstrationDataset(**self.options)
        self.assertEqual(len(dataset), 6)
        with self.assertRaisesRegex(ValueError, "Nonfinite normalization"):
            build_normalizer(dataset, {"action": "limits"}, "action")

    def test_window_eligibility_uses_observation_and_action_horizons(self):
        self.root["data/joint_state"][1, 0] = np.nan
        self.root["data/action"][4, 0] = np.nan
        dataset = DemonstrationDataset(**self.options)
        self.assertEqual(len(dataset), 1)
        np.testing.assert_array_equal(dataset[0]["action"][:, 0], [0, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Nonfinite normalization"):
            build_normalizer(dataset, {"action": "limits"}, "action")

    def test_augmentation_does_not_mutate_storage_or_validation(self):
        dataset = DemonstrationDataset(
            **self.options,
            augmentation_cfg={"pc": {"coord_noise": {"noise_std": 0.1}, "random_keep_points": 4}},
        )
        original = dataset.replay_buffer.read("point_cloud", slice(0, 2))
        self.assertEqual(dataset[0]["obs"]["point_cloud"].shape, (2, 4, 6))
        readback = dataset.replay_buffer.read("point_cloud", slice(0, 2))
        np.testing.assert_array_equal(readback, original)
        readback[:] = 0
        np.testing.assert_array_equal(
            dataset.replay_buffer.read("point_cloud", slice(0, 2)), original,
        )
        validation = dataset.get_validation_dataset()
        np.testing.assert_array_equal(
            validation[0]["obs"]["point_cloud"], self.root["data/point_cloud"][[6, 6]],
        )

    def test_hand_context_stays_inside_episode(self):
        lamp = LAMPDataset(**self.options, history_steps=3)
        np.testing.assert_array_equal(
            lamp[0]["action_context"]["hand_history"][:, 0], [100, 100, 100],
        )
        np.testing.assert_array_equal(
            lamp[1]["action_context"]["hand_history"][:, 0], [100, 100, 0],
        )
        np.testing.assert_array_equal(
            lamp.get_validation_dataset()[0]["action_context"]["hand_history"][:, 0],
            [106, 106, 106],
        )
        codec = TemporalCodecDataset(**self.options)
        commands, reference = codec.raw_hand_batch(np.array([0, 1]))
        np.testing.assert_array_equal(commands[:, :, 0], [[0, 0, 1, 2], [0, 1, 2, 3]])
        np.testing.assert_array_equal(reference[:, 0], [100, 101])
        torch.testing.assert_close(
            codec[1]["action_context"]["hand_reference"], torch.full((12,), 101.0),
        )
        for method in (lamp.history_for_rows, codec.raw_hand_batch):
            with self.assertRaises(ValueError):
                method(np.array([-1]))

    def test_rgb_multitask_and_layout_mismatch(self):
        options = {**self.options, "sensor_modalities": ["joint_state", "rgb"]}
        first = DemonstrationDataset(**options)
        second = DemonstrationDataset(**options)
        dataset = MultiTaskDataset([first, second], ["a", "b"])
        self.assertEqual(dataset[len(first)]["obs"]["task_text"], "b")
        self.assertEqual(dataset[0]["obs"]["rgb"].dtype, torch.uint8)
        self.assertEqual(len(dataset.get_validation_dataset()), 12)
        other = DemonstrationDataset(**self.options)
        with self.assertRaisesRegex(ValueError, "modality"):
            MultiTaskDataset([first, other], ["a", "b"])

    def test_point_cloud_shape_is_checked_without_pc_dim(self):
        del self.root["data/point_cloud"]
        self.root.create_dataset("data/point_cloud", data=np.zeros((12, 6), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"\[T,N,C\]"):
            DemonstrationDataset(**self.options)


class AugmentationAndBatchesTest(unittest.TestCase):
    def test_noise_channels_and_clip(self):
        for transform, affected in (
            (PointCoordinateNoise, slice(0, 3)), (PointColorNoise, slice(3, 6)),
        ):
            with self.subTest(transform=transform.__name__):
                values = np.full((2, 1024, 6), 0.5, dtype=np.float32)
                transform(noise_std=0.05, clip_range=0.1).augment(values)
                delta = values - 0.5
                self.assertLessEqual(np.abs(delta).max(), 0.100001)
                self.assertGreater(delta[..., affected].std(), 0.03)
                untouched = slice(3, 6) if affected.start == 0 else slice(0, 3)
                np.testing.assert_array_equal(delta[..., untouched], 0)

    def test_shuffled_passes_keep_tail_and_rng_sequence(self):
        for population in (7, np.array([2, 5, 8, 11, 15, 20, 25])):
            with self.subTest(population=population):
                expected_rng = np.random.default_rng(42)
                actual_rng = np.random.default_rng(42)
                batches = shuffled_batches(population, 3, actual_rng)
                for _ in range(3):
                    expected = expected_rng.permutation(population)
                    parts = [next(batches) for _ in range(3)]
                    self.assertEqual([len(part) for part in parts], [3, 3, 1])
                    np.testing.assert_array_equal(np.concatenate(parts), expected)
                self.assertEqual(actual_rng.random(), expected_rng.random())


if __name__ == "__main__":
    unittest.main()
