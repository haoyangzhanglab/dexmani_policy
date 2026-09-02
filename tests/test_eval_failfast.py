"""Regression tests for A3 (eval exception taxonomy + fail-fast) and A4 (fixed-seed selector).

Covers the eval-runner contract without requiring dexmani_sim: a fake
``BaseRunner`` subclass exercises the fail-fast vs. genuine-failure split,
and the pure validators are tested directly.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dexmani_policy.env_runner.base_runner import (
    BaseRunner,
    EvalEpisodeError,
    _classify_eval_exception,
)
from dexmani_policy.training.eval_utils import read_best_ckpt_json, validate_denoise_steps


class _FakeEnv:
    video_fps = 15
    action_cnt = 0

    def get_video(self):
        return None

    def close(self):
        pass


class _FakeRunner(BaseRunner):
    """Runner whose episodes either genuinely fail or raise, controlled by fail_with."""

    def __init__(self, fail_with=None):
        super().__init__(n_obs_steps=2, default_eval_episodes=2, clear_cache_freq=0)
        self.fail_with = fail_with
        self.call_seeds = []

    def make_env(self):
        return _FakeEnv()

    def get_seed_list(self):
        return [0, 1, 2]

    def run_one_episode(self, agent, env, episode_seed, denoise_timesteps=None, **kwargs):
        self.call_seeds.append(episode_seed)
        if self.fail_with is not None:
            raise self.fail_with
        return False, None  # genuine task failure (no exception)


class TestClassifyEvalException(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(_classify_eval_exception(ValueError("x")), "value_error")
        self.assertEqual(_classify_eval_exception(RuntimeError("x")), "runtime_error")
        self.assertEqual(_classify_eval_exception(KeyError("x")), "KeyError")
        if hasattr(torch.cuda, "OutOfMemoryError"):
            self.assertEqual(_classify_eval_exception(torch.cuda.OutOfMemoryError("oom")), "oom")


class TestValidateDenoiseSteps(unittest.TestCase):
    def test_midpoint_even_only(self):
        validate_denoise_steps([4], "midpoint")  # ok
        with self.assertRaises(ValueError):
            validate_denoise_steps([3], "midpoint")

    def test_euler_any_positive(self):
        validate_denoise_steps([1], "euler")
        validate_denoise_steps([10], "euler")

    def test_non_midpoint_solver_no_even_constraint(self):
        validate_denoise_steps([3], None)  # non-ActionFlow decoder: no solver constraint

    def test_rejects_invalid(self):
        for bad in ([0], [-1], [True], [1.5], []):
            with self.assertRaises(ValueError):
                validate_denoise_steps(bad, "euler")


class TestReadBestCkptJson(unittest.TestCase):
    def test_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_best_ckpt_json(Path(d)))

    def test_malformed_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "best_ckpt.json").write_text("{not valid json")
            with self.assertRaises(ValueError):
                read_best_ckpt_json(Path(d))

    def test_valid_parses(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "best_ckpt.json").write_text(
                json.dumps({"ckpt_path": "checkpoints/best.pt", "pct": 100})
            )
            info = read_best_ckpt_json(Path(d))
            self.assertEqual(info["pct"], 100)


class TestRunFailFast(unittest.TestCase):
    def test_model_forward_error_aborts(self):
        runner = _FakeRunner(fail_with=ValueError("bad nfe"))
        with self.assertRaises(EvalEpisodeError) as ctx:
            runner.run(agent=None)
        self.assertEqual(ctx.exception.category, "value_error")
        self.assertEqual(ctx.exception.seed, 0)

    def test_env_normal_failure_records_zero_no_raise(self):
        runner = _FakeRunner()
        result = runner.run(agent=None)
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["episodes_collected"], 2)
        # Genuine failures carry no error_category (that's for fatal exceptions).
        for d in result["episode_details"]:
            self.assertNotIn("error_category", d)


if __name__ == "__main__":
    unittest.main()
