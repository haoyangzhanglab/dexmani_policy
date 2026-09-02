"""Regression tests for A2 (MultiTask eval stats rewrite).

The multi-task evaluation unit is ``(task, seed)``; the micro success rate must
divide by the actual number of completed episode units (not the per-task seed
count), and the macro average must never silently drop a failed task.  These
tests exercise the pure stats helpers and the runner's ``get_seed_list`` /
fail-fast contract without requiring a live dexmani_sim environment.
"""

from __future__ import annotations

import unittest
from unittest import mock

from dexmani_policy.env_runner.multi_task_sim_runner import (
    MultiTaskSimRunner,
    TaskTextSimRunner,
)
from dexmani_policy.training.eval_utils import (
    collect_episode_details,
    compute_eval_stats,
)


def _task_result(successes: list[bool]) -> dict:
    """Build a child-runner result dict mimicking ``BaseRunner.run()`` output."""
    details = [
        {"seed": i, "success": s, "steps": 5 if s else None, "total_steps": 10}
        for i, s in enumerate(successes)
    ]
    n_success = sum(successes)
    return {
        "success_rate": (n_success / len(successes)) if successes else None,
        "avg_steps": 5 if n_success else None,
        "avg_steps_all": 10,
        "videos": [],
        "episode_details": details,
        "episodes_collected": len(successes),
        "episodes_requested": len(successes),
    }


class TestComputeEvalStats(unittest.TestCase):
    def test_single_task_uses_top_level_episode_details(self):
        result = {
            "success_rate": 0.5,
            "avg_steps": 5,
            "episode_details": [
                {"seed": 0, "success": True, "steps": 5},
                {"seed": 1, "success": False, "steps": None},
            ],
        }
        stats = compute_eval_stats(result)
        self.assertEqual(stats["n_success"], 1)
        self.assertEqual(stats["n_valid_episodes"], 2)
        self.assertAlmostEqual(stats["micro_success_rate"], 0.5)
        self.assertIsNone(stats["macro_success_rate"])
        self.assertEqual(stats["n_tasks"], 1)
        self.assertIsNone(stats["per_task"])

    def test_two_tasks_three_seeds_six_units(self):
        # 2 tasks × 3 seeds = 6 episode units; micro denominator must be 6.
        result = {
            "per_task": {
                "pour": _task_result([True, False, True]),   # 2/3
                "pick": _task_result([True, True, False]),   # 2/3
            },
        }
        stats = compute_eval_stats(result)
        self.assertEqual(stats["n_valid_episodes"], 6)
        self.assertEqual(stats["n_success"], 4)
        self.assertAlmostEqual(stats["micro_success_rate"], 4 / 6)
        self.assertAlmostEqual(stats["macro_success_rate"], 2 / 3)
        self.assertEqual(stats["n_tasks"], 2)
        self.assertEqual(set(stats["per_task"].keys()), {"pour", "pick"})
        self.assertAlmostEqual(stats["per_task"]["pour"]["success_rate"], 2 / 3)
        self.assertEqual(stats["per_task"]["pour"]["n_valid"], 3)

    def test_all_success_cannot_exceed_one(self):
        # The old bug divided the cross-task success count by the seed count,
        # yielding 6/3 = 2.0.  micro must stay within [0, 1].
        result = {
            "per_task": {
                "pour": _task_result([True, True, True]),
                "pick": _task_result([True, True, True]),
            },
        }
        stats = compute_eval_stats(result)
        self.assertEqual(stats["n_valid_episodes"], 6)
        self.assertEqual(stats["n_success"], 6)
        self.assertEqual(stats["micro_success_rate"], 1.0)
        self.assertEqual(stats["macro_success_rate"], 1.0)

    def test_all_success_rates_within_unit_interval(self):
        # Exhaustive over 2 tasks × 3 seeds success combos: every SR ∈ [0, 1].
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        result = {
                            "per_task": {
                                "t1": _task_result([bool(a & 1), bool(a & 2)]),
                                "t2": _task_result([bool(c & 1), bool(d & 2)]),
                            },
                        }
                        stats = compute_eval_stats(result)
                        self.assertGreaterEqual(stats["micro_success_rate"], 0.0)
                        self.assertLessEqual(stats["micro_success_rate"], 1.0)
                        self.assertGreaterEqual(stats["macro_success_rate"], 0.0)
                        self.assertLessEqual(stats["macro_success_rate"], 1.0)


class TestCollectEpisodeDetails(unittest.TestCase):
    def test_multi_task_flatten_tags_task_name(self):
        result = {
            "per_task": {
                "pour": _task_result([True]),
                "pick": _task_result([False]),
            },
        }
        details = collect_episode_details(result)
        self.assertEqual(len(details), 2)
        self.assertEqual({d["task_name"] for d in details}, {"pour", "pick"})
        # The original nested dicts must not be mutated.
        self.assertNotIn("task_name", result["per_task"]["pour"]["episode_details"][0])


class TestMultiTaskRunnerGetSeedList(unittest.TestCase):
    def test_returns_explicit_pool_when_set(self):
        runner = MultiTaskSimRunner(
            task_configs=[{"task_name": "pour"}, {"task_name": "pick"}],
            n_obs_steps=2,
            default_eval_episodes=3,
        )
        runner.eval_seeds = [7, 8, 9]
        self.assertEqual(runner.get_seed_list(), [7, 8, 9])

    def test_delegates_to_first_child_when_unset(self):
        runner = MultiTaskSimRunner(
            task_configs=[{"task_name": "pour"}, {"task_name": "pick"}],
            n_obs_steps=2,
            default_eval_episodes=3,
        )
        self.assertEqual(runner.get_seed_list(), runner.runners["pour"].get_seed_list())


class TestMultiTaskFailFast(unittest.TestCase):
    def test_single_task_infra_failure_raises_nonzero(self):
        """A child task raising an infra error must abort the whole run."""
        runner = MultiTaskSimRunner(
            task_configs=[{"task_name": "pour"}, {"task_name": "pick"}],
            n_obs_steps=2,
            default_eval_episodes=2,
        )

        def _fake_run(self, agent, denoise_timesteps=None, eval_episodes=None, video_save_dir=None):
            if self.task_name == "pour":
                raise RuntimeError("env construction failed")
            return _task_result([True, True])

        with mock.patch.object(TaskTextSimRunner, "run", _fake_run):
            with self.assertRaises(RuntimeError) as ctx:
                runner.run(agent=None)
            self.assertIn("pour", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
