"""Real warmup preserves the optional tactile validity field's boolean contract."""

from types import SimpleNamespace
import unittest

import numpy as np

from dexmani_policy.deployment.runtime import LoadedPolicy


class TactileValidityWarmupTest(unittest.TestCase):
    def test_warmup_builds_boolean_valid_fingers(self):
        policy = object.__new__(LoadedPolicy)
        policy.info = SimpleNamespace(
            horizon=4, n_obs_steps=2, observation_fields=("tactile_valid",),
        )
        policy.rtc_guidance_cap = 0
        policy.reset_episode = lambda: None
        observed = []

        def predict(observation):
            observed.append(observation["tactile_valid"])
            return np.zeros((3, 19))

        policy.predict = predict
        timing = policy.warmup(samples=1)
        self.assertEqual(len(timing["bootstrap"]), 1)
        self.assertEqual(len(observed), 2)
        for valid in observed:
            self.assertEqual(valid.dtype, np.bool_)
            self.assertEqual(valid.shape, (2, 5))
            self.assertTrue(valid.all())


if __name__ == "__main__":
    unittest.main()
