import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_policy.evaluation import artifacts
from dexmani_policy.evaluation.metrics import summarize_episodes, wilson_interval


class MetricsTests(unittest.TestCase):
    def test_final_success_controls_step_average(self):
        episodes = [
            {"success": True, "steps": 2, "total_steps": 5},
            {"success": False, "steps": 1, "total_steps": 9},
            {"success": True, "steps": 6, "total_steps": 7},
        ]
        stats = summarize_episodes(episodes)
        self.assertEqual(stats["n_success"], 2)
        self.assertEqual(stats["n_valid_episodes"], 3)
        self.assertEqual(stats["success_rate"], 2 / 3)
        self.assertEqual(stats["avg_steps"], 4)
        self.assertEqual(stats["avg_steps_all"], 7)

    def test_all_failures_are_a_valid_result(self):
        stats = summarize_episodes([{"success": False, "steps": None, "total_steps": 10}])
        self.assertEqual(stats["success_rate"], 0)
        self.assertIsNone(stats["avg_steps"])
        self.assertEqual(stats["wilson_95"][0], 0)

    def test_wilson_interval_known_binomial_result(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038315303659956)
        self.assertAlmostEqual(high, 0.5961684696340044)


class ArtifactTests(unittest.TestCase):
    def test_existing_results_are_not_overwritten_on_run_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = artifacts.create_run_directory(temporary)
            path = directory / "result_details.json"
            artifacts.write_json(path, {"status": "completed"}, overwrite=False)
            with self.assertRaises(FileExistsError):
                artifacts.create_run_directory(temporary, directory)
            self.assertEqual(json.loads(path.read_text()), {"status": "completed"})

    def test_failure_save_preserves_the_rollout_exception(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), patch.object(
            artifacts, "write_json", side_effect=OSError("disk full")
        ), self.assertRaisesRegex(RuntimeError, "simulation failed"):
            try:
                raise RuntimeError("simulation failed")
            except RuntimeError:
                artifacts.write_failure(Path("unused.json"), {"status": "failed"})
                raise
        self.assertIn("disk full", stderr.getvalue())


class RunnerTests(unittest.TestCase):
    def make_runner(self):
        from dexmani_policy.env_runner.sim_runner import SimRunner

        return SimRunner("task", 1, [], {}, env_rebuild_interval=2)

    def test_runner_emits_each_seed_in_order_and_closes_environments(self):
        runner = self.make_runner()
        environments = [SimpleNamespace(action_cnt=0, close=Mock()) for _ in range(2)]
        agent = SimpleNamespace(consumed_observation_fields=[], tactile_source=None)

        def episode(agent, env, seed, inference_steps, *, action_steps):
            env.action_cnt = seed + 10
            return seed != 1, seed

        with patch.object(runner, "make_env", side_effect=environments), patch.object(
            runner, "_run_episode", side_effect=episode
        ):
            rows = list(runner.run(agent, seeds=[3, 1, 2], inference_steps=2, action_steps=1))

        self.assertEqual([row["seed"] for row in rows], [3, 1, 2])
        self.assertEqual([row["task_name"] for row in rows], ["task"] * 3)
        self.assertEqual([row["total_steps"] for row in rows], [13, 11, 12])
        for env in environments:
            env.close.assert_called_once_with()

    def test_runner_closes_environment_when_iteration_stops(self):
        runner = self.make_runner()
        env = SimpleNamespace(action_cnt=1, close=Mock())
        agent = SimpleNamespace(consumed_observation_fields=[], tactile_source=None)
        with patch.object(runner, "make_env", return_value=env), patch.object(
            runner, "_run_episode", return_value=(True, 1)
        ):
            episodes = runner.run(agent, seeds=[3, 1], inference_steps=2, action_steps=1)
            next(episodes)
            episodes.close()
        env.close.assert_called_once_with()


class RolloutTests(unittest.TestCase):
    def setUp(self):
        from dexmani_policy.evaluation import rollout
        from dexmani_policy.evaluation.protocol import EvalProtocol

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "run"
        self.rollout = rollout
        self.request = EvalProtocol(
            config={"eval": {"seed": 1024, "video": {"enabled": False}}},
            checkpoint="checkpoint.pt", use_ema=False, inference_steps=2, action_steps=1,
            device="cpu", runner_options={"task_name": "task"}, seeds=[9, 4], source={},
        )
        parameter = SimpleNamespace(dtype="torch.float32", device="cpu")
        agent = SimpleNamespace(checkpoint_step=10, parameters=lambda: iter([parameter]))
        self.completed = {
            "task_name": "task", "seed": 9, "success": True,
            "steps": 1, "total_steps": 3, "video": None,
        }
        self.runner = Mock(task_name="task")
        self.runner_factory = Mock(return_value=self.runner)
        for name, replacement in (
            ("SimRunner", self.runner_factory),
            ("restore_policy", Mock(return_value=agent)),
            ("set_seed", Mock()),
            ("runtime_provenance", Mock(return_value={})),
        ):
            patcher = patch.object(rollout, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def evaluate(self):
        return self.rollout.evaluate_policy(
            self.request, output_root=self.directory.parent, output_dir=self.directory
        )

    def read_record(self):
        return json.loads((self.directory / "result_details.json").read_text())

    def test_completed_rollout_saves_statistics(self):
        def episodes():
            yield self.completed
            yield {**self.completed, "seed": 4, "success": False, "steps": None, "total_steps": 5}

        self.runner.run.return_value = episodes()
        with redirect_stdout(io.StringIO()):
            record, path = self.evaluate()

        self.assertEqual(path, self.directory / "result_details.json")
        self.assertEqual(self.read_record(), record)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["statistics"]["success_rate"], 0.5)
        self.assertEqual(record["statistics"]["avg_steps"], 1)
        self.assertEqual(record["statistics"]["avg_steps_all"], 4)

    def test_checkpoint_step_mismatch_fails_before_runner_creation(self):
        self.request.source["global_step"] = 11
        with self.assertRaisesRegex(ValueError, "global_step"):
            self.evaluate()

        self.runner_factory.assert_not_called()
        record = self.read_record()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["episodes"], [])
        self.assertIsNone(record["statistics"])

    def test_failed_rollout_saves_completed_prefix_and_closes_iterator(self):
        closed = Mock()

        def episodes():
            try:
                yield self.completed
                raise RuntimeError("simulation failed")
            finally:
                closed()

        self.runner.run.return_value = episodes()
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            self.evaluate()
        record = self.read_record()

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["episodes"], [self.completed])
        self.assertIsNone(record["statistics"])
        self.assertIn("simulation failed", record["error"])
        closed.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
