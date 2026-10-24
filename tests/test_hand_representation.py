import unittest

import numpy as np

from dexmani_policy.training.common.hand_representation import shuffled_batches


class ShuffledBatchTests(unittest.TestCase):
    def test_drop_last_complete_passes_and_reproducibility(self):
        batch_size = 256
        for count in (257, 256, 512, 519):
            for indices in (count, np.arange(count) * 3 + 7):
                with self.subTest(count=count, integer_input=isinstance(indices, int)):
                    batches = shuffled_batches(
                        indices, batch_size, np.random.default_rng(42), drop_last=True
                    )
                    repeated = shuffled_batches(
                        indices, batch_size, np.random.default_rng(42), drop_last=True
                    )
                    reference_rng = np.random.default_rng(42)
                    passes = []
                    for _ in range(3):
                        expected = reference_rng.permutation(indices)
                        actual = [next(batches) for _ in range(count // batch_size)]
                        for batch in actual:
                            self.assertEqual(len(batch), batch_size)
                            np.testing.assert_array_equal(batch, next(repeated))
                        combined = np.concatenate(actual)
                        self.assertEqual(len(np.unique(combined)), len(combined))
                        np.testing.assert_array_equal(combined, expected[:len(combined)])
                        passes.append(combined)
                    self.assertFalse(np.array_equal(passes[0], passes[1]))

    def test_default_keeps_partial_batches(self):
        for count, sizes in ((257, [256, 1]), (519, [256, 256, 7]), (3, [3])):
            with self.subTest(count=count):
                batches = shuffled_batches(count, 256, np.random.default_rng(42))
                for _ in range(2):
                    actual = [next(batches) for _ in sizes]
                    self.assertEqual([len(batch) for batch in actual], sizes)
                    np.testing.assert_array_equal(np.sort(np.concatenate(actual)), np.arange(count))

    def test_empty_or_impossible_complete_batch_fails(self):
        for indices in (0, np.array([], dtype=np.int64), 3, np.arange(3)):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "empty|at least batch_size"):
                    next(shuffled_batches(indices, 256, np.random.default_rng(42), drop_last=True))
        with self.assertRaisesRegex(ValueError, "empty"):
            next(shuffled_batches(0, 256, np.random.default_rng(42)))


if __name__ == "__main__":
    unittest.main()
