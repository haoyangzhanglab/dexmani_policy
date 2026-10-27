import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.experiments import milestone_steps
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation.config import (
    evaluation_seed_pool,
    parse_evaluation_overrides,
    resolve_evaluation_request,
)
from dexmani_policy.evaluation.metrics import (
    compute_evaluation_statistics,
    validate_episodes,
    wilson_interval,
)
from dexmani_policy.evaluation.rollout import evaluate_policy
from dexmani_policy.evaluation.seeds import split_evaluation_seeds
from dexmani_policy.evaluation.selection import (
    rank_selection_results,
    run_pipeline,
    select_best_checkpoint,
)


class FakePolicy(torch.nn.Module):
    def __init__(self, step):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.checkpoint_step = step
        self.calls = []

    def predict_action(self, *, obs_dict, inference_steps, generator):
        draw = torch.rand((), generator=generator)
        self.calls.append((generator.initial_seed(), inference_steps, draw.item()))
        return {"future_actions": torch.full((1, 4, 19), draw)}


class FakeEnvironment:
    def __init__(self, fail_seed=None):
        self.fail_seed = fail_seed
        self.closed = False
        self.seeds = []
        self.actions = []

    def observation(self):
        return {"joint_state": np.full(19, self.action_cnt, dtype=np.float32)}

    def reset(self, *, seed):
        self.seed = seed
        self.seeds.append(seed)
        self.action_cnt = 0
        # Consuming a global stream must not affect the policy's per-episode generator.
        torch.rand(seed % 5 + 1)
        return self.observation(), {}

    def step(self, action):
        if self.seed == self.fail_seed:
            raise RuntimeError("fake environment failure")
        self.actions.append(action.copy())
        self.action_cnt += 1
        done = self.action_cnt == 3
        return self.observation(), 0, done, False, {
            "success_condition": self.action_cnt == 1,
            "success": done and self.seed % 2 == 0,
        }

    def close(self):
        self.closed = True


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "experiment"
        (self.root / "checkpoints").mkdir(parents=True)
        self.cfg = dict(
            task_name="fake", action_key="action",
            dataset={"action_key": "action", "sensor_modalities": ["joint_state"]},
            agent={"n_obs_steps": 2, "n_action_steps": 2, "horizon": 4},
            policy_metadata={"action_start_index": 1},
            training={"device": "cpu", "loop": {"total_train_steps": 10}},
            env_runner={"task_name": "fake", "env_rebuild_interval": 2,
                        "env_kwargs": {"control_mode": "joint"}, "eval_seeds": list(range(20))},
            eval={"seed": 17, "episodes": 3, "selection_episodes": 4, "seeds": None,
                  "inference_steps": 7, "use_ema": False, "action_steps": None,
                  "video": {"enabled": True}},
        )
        OmegaConf.save(OmegaConf.create(self.cfg), self.root / "config.yaml")
        self.milestones = []
        for step, pct in milestone_steps(10).items():
            path = self.root / f"checkpoints/epoch=1-step={step}-milestone={pct}pct.pt"
            torch.save({"global_step": step}, path)
            self.milestones.append(path)
        self.latest = self.root / "checkpoints/latest.pt"
        torch.save({"global_step": 10}, self.latest)
        self.environments, self.agents, self.restores, self.videos = [], [], [], []
        self.fail_seed = None
        self.fail_step = None

        def restore(config, checkpoint, *, use_ema, device):
            step = torch.load(checkpoint, weights_only=True)["global_step"]
            agent = FakePolicy(step)
            self.agents.append(agent)
            self.restores.append((config, checkpoint, use_ema, device))
            return agent

        def make_env(runner, *, record_video):
            fail = self.fail_seed if self.fail_step in (None, self.agents[-1].checkpoint_step) else None
            env = FakeEnvironment(fail)
            self.environments.append(env)
            self.videos.append(record_video)
            return env

        for target, replacement in (
            ("dexmani_policy.evaluation.rollout.restore_policy_agent", restore),
            ("dexmani_policy.env_runner.sim_runner.SimRunner.make_env", make_env),
        ):
            mock = patch(target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def test_seed_split_and_boundaries(self):
        pool = [9, 2, 0, 13, 8, 5, 1, 7]
        original, rng_state = pool.copy(), random.getstate()
        expected = sorted(pool)
        random.Random(17).shuffle(expected)
        test, selection = split_evaluation_seeds(pool, 17, 3, 2)
        self.assertEqual(test, expected[:3])
        self.assertEqual(selection, expected[3:5])
        self.assertEqual(test, split_evaluation_seeds(pool, 17, 3, 5)[0])
        self.assertFalse(set(test) & set(selection))
        self.assertEqual(pool, original)
        self.assertEqual(random.getstate(), rng_state)
        for bad in ([1, 1], [-1], [2**32], [True], [1.0], {"fake": [1]}):
            with self.subTest(pool=bad), self.assertRaises(ValueError):
                split_evaluation_seeds(bad, 0, 1)
        for seed, episodes, selection_count in ((-1, 1, 0), (True, 1, 0), (0, 0, 0), (0, 1, -1), (0, 8, 1)):
            with self.assertRaises(ValueError):
                split_evaluation_seeds(pool, seed, episodes, selection_count)
        data_dir = self.parent / "sim_data"
        (data_dir / "eval_seeds").mkdir(parents=True)
        (data_dir / "eval_seeds/fake.txt").write_text("9 3 17")
        cfg = copy.deepcopy(self.cfg)
        del cfg["env_runner"]["eval_seeds"]
        with patch.dict("sys.modules", {"dexmani_sim": SimpleNamespace(DATA_DIR=data_dir)}):
            self.assertEqual(evaluation_seed_pool(cfg), [9, 3, 17])

    def test_ordinary_selectors_explicit_seeds_and_overrides(self):
        overrides = parse_evaluation_overrides(["eval.seeds=[9,3,17]", "eval.action_steps=1"])
        for selector, path in (("latest", self.latest), ("20pct", self.milestones[0]), (self.milestones[1].name, self.milestones[1])):
            request = resolve_evaluation_request(self.root, selector, overrides)
            self.assertEqual(request.seeds, [9, 3, 17])
            self.assertEqual(request.checkpoint, str(path))
            self.assertEqual(request.action_steps, 1)
            self.assertEqual(request.runner_options["task_name"], "fake")
            self.assertNotIn("eval_seeds", request.runner_options)
            self.assertFalse(request.source["heldout_from_selection"])
        request = resolve_evaluation_request(self.root, "latest")
        self.assertEqual(request.seeds, split_evaluation_seeds(list(range(20)), 17, 3)[0])
        for seeds in ([], [20], [2, 2], [False], [2**32], {"fake": [2]}):
            with self.assertRaises(ValueError):
                resolve_evaluation_request(self.root, "latest", {"eval": {"seeds": seeds}})
        for override in ({"eval": {"action_steps": 4}}, {"env_runner": {"sensor_modalities": []}}):
            with self.assertRaises(ValueError):
                resolve_evaluation_request(self.root, "latest", override)

    def test_metrics_and_ranking(self):
        rows = [
            dict(task_name="fake", seed=i, success=success, steps=steps, total_steps=total)
            for i, (success, steps, total) in enumerate(((True, 2, 5), (False, 1, 9), (True, None, 4)))
        ]
        validate_episodes(rows, "fake", [0, 1, 2])
        stats = compute_evaluation_statistics(rows)
        self.assertEqual(set(stats), {"success_rate", "n_success", "n_valid_episodes", "avg_steps", "avg_steps_all", "wilson_95"})
        self.assertEqual((stats["success_rate"], stats["n_success"], stats["n_valid_episodes"]), (2 / 3, 2, 3))
        self.assertEqual((stats["avg_steps"], stats["avg_steps_all"]), (2, 6))
        np.testing.assert_allclose(wilson_interval(5, 10), [0.236593090512564, 0.763406909487436])
        for success in (False, True):
            result = compute_evaluation_statistics([dict(row, success=success) for row in rows])
            self.assertEqual(result["success_rate"], int(success))
            self.assertEqual(result["avg_steps"], 1.5 if success else None)
            self.assertAlmostEqual(result["wilson_95"][int(success)], int(success))
        for invalid in (rows[:-1], rows + rows[:1], [dict(rows[0], task_name="wrong"), *rows[1:]], [dict(rows[0], steps=6), *rows[1:]], [dict(rows[0], total_steps=-1), *rows[1:]]):
            with self.assertRaises(RuntimeError):
                validate_episodes(invalid, "fake", [0, 1, 2])
        results = [dict(statistics={"success_rate": rate, "avg_steps": steps}, global_step=step)
                   for rate, steps, step in ((0, None, 100), (0.5, 2, 10), (0.5, 1, 5), (0.5, 1, 7))]
        self.assertEqual(rank_selection_results(results), 3)
        self.assertEqual(rank_selection_results([results[0], dict(results[0], global_step=101)]), 1)

    def test_rollout_single_runner_rng_chunks_and_cleanup(self):
        request = resolve_evaluation_request(self.root, "latest", {"eval": {"seeds": [2, 3, 4], "video": {"enabled": False}}})
        with patch("dexmani_policy.evaluation.rollout.SimRunner", wraps=SimRunner) as runner:
            record, path = evaluate_policy(request, output_root=self.root / "eval_dexsim")
        runner.assert_called_once_with(**request.runner_options)
        self.assertEqual(len(self.restores), 1)
        self.assertEqual(record, json.loads(path.read_text()))
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["statistics"]["success_rate"], 2 / 3)
        self.assertEqual(record["statistics"]["avg_steps"], 1)
        self.assertEqual(record["statistics"]["avg_steps_all"], 3)
        self.assertEqual([env.seeds for env in self.environments], [[2, 3], [4]])
        self.assertTrue(all(env.closed for env in self.environments))
        self.assertEqual([len(env.actions) for env in self.environments], [6, 3])
        calls = self.agents[0].calls
        for i, seed in enumerate(request.seeds):
            generator = torch.Generator().manual_seed(seed)
            self.assertEqual(calls[2 * i : 2 * i + 2], [(seed, 7, torch.rand((), generator=generator).item()) for _ in range(2)])
        self.assertEqual(set(record["episodes"][0]), {"task_name", "seed", "success", "steps", "total_steps", "video"})
        with self.assertRaises(FileExistsError):
            evaluate_policy(request, output_root=path.parent.parent, output_dir=path.parent)

    def test_selection_frozen_relocation_and_best(self):
        directory = self.root / "eval_ckpt_selector/run"
        record, checkpoint = select_best_checkpoint(
            self.root, self.cfg, output_dir=directory,
            overrides={"eval": {"video": {"enabled": True}, "use_ema": True,
                                "inference_steps": 5, "action_steps": 1}},
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(checkpoint, self.milestones[-1])
        test, selection = split_evaluation_seeds(list(range(20)), 17, 3, 4)
        self.assertEqual(record["selection_seeds"], selection)
        self.assertEqual(record["test_request"]["seeds"], test)
        self.assertTrue(record["test_request"]["config"]["eval"]["video"]["enabled"])
        self.assertFalse(any(self.videos))
        for path in record["candidate_results"]:
            candidate = json.loads((self.root / path).read_text())
            self.assertEqual(candidate["request"]["seeds"], selection)
            self.assertEqual([ep["seed"] for ep in candidate["episodes"]], selection)
            self.assertEqual(candidate["status"], "completed")
        best = json.loads((self.root / "best_ckpt.json").read_text())
        self.assertEqual(set(best), {"ckpt_relpath", "global_step", "inference", "selection_result"})
        ordinary = resolve_evaluation_request(self.root, "best", {"eval": {"inference_steps": 9}})
        self.assertEqual(ordinary.inference_steps, 9)
        self.assertEqual((ordinary.use_ema, ordinary.action_steps), (True, 1))
        self.assertFalse(ordinary.source["heldout_from_selection"])
        self.assertEqual(ordinary.checkpoint, str(checkpoint))
        selection_path = directory / "selection_result.json"
        for override in ({"eval": {"seeds": [1]}}, {"eval": {"use_ema": True}}, {"eval": {"inference_steps": 8}}, {"eval": {"action_steps": 1}}, {"env_runner": {"env_rebuild_interval": 1}}, {"training": {"seed": 1}}):
            with self.assertRaisesRegex(ValueError, "Frozen test permits"):
                resolve_evaluation_request(self.root, overrides=override, selection_record=selection_path)
        frozen = resolve_evaluation_request(self.root, selection_record=selection_path)
        self.assertEqual(frozen.seeds, test)
        self.assertTrue(frozen.source["heldout_from_selection"])
        self.assertEqual((frozen.inference_steps, frozen.action_steps, frozen.use_ema), (5, 1, True))
        moved = self.parent / "moved_experiment"
        self.root.rename(moved)
        (moved / "config.yaml").unlink()
        moved_request = resolve_evaluation_request(moved, overrides={"training": {"device": "cpu:0"}}, selection_record=moved / record["selection_result"])
        self.assertEqual(moved_request.device, "cpu:0")
        self.assertEqual(moved_request.checkpoint, str(moved / record["ckpt_relpath"]))
        self.assertEqual(moved_request.runner_options, frozen.runner_options)
        self.assertEqual(moved_request.config, frozen.config)
        self.assertEqual(moved_request.seeds, frozen.seeds)
        with patch("dexmani_policy.env_runner.sim_runner.save_episode_video", side_effect=lambda env, directory, name: str(directory / f"{name}.mp4")):
            result, _ = evaluate_policy(moved_request, output_root=moved / "eval_dexsim", output_dir=moved / "eval_dexsim/test")
        self.assertEqual(result["global_step"], 10)
        self.assertTrue(all("videos/fake/episode_" in ep["video"] for ep in result["episodes"]))
        torch.save({"global_step": 99}, moved_request.checkpoint)
        with self.assertRaisesRegex(ValueError, "global_step differs"):
            evaluate_policy(moved_request, output_root=moved / "eval_dexsim")

    def test_selection_failure_keeps_best_and_records_partial_rollout(self):
        best_path = self.root / "best_ckpt.json"
        best_path.write_text('{"original": true}\n')
        _, selection = split_evaluation_seeds(list(range(20)), 17, 3, 4)
        self.fail_step, self.fail_seed = 4, selection[1]
        directory = self.root / "eval_ckpt_selector/failure"
        with self.assertRaisesRegex(RuntimeError, "fake environment failure"):
            select_best_checkpoint(self.root, self.cfg, output_dir=directory)
        self.assertEqual(best_path.read_text(), '{"original": true}\n')
        record = json.loads((directory / "selection_result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("fake environment failure", record["error"])
        self.assertEqual(len(record["candidate_results"]), 1)
        candidates = [json.loads(path.read_text()) for path in directory.glob("candidates/*/result_details.json")]
        failed = next(row for row in candidates if row["status"] == "failed")
        self.assertEqual(failed["episodes"][0]["seed"], selection[0])
        self.assertEqual(len(failed["episodes"]), 1)
        self.assertIsNone(failed["statistics"])
        self.assertTrue(all(env.closed for env in self.environments))
        with self.assertRaisesRegex(ValueError, "has not completed"):
            resolve_evaluation_request(self.root, selection_record=directory / "selection_result.json")
        self.milestones[0].unlink()
        with self.assertRaisesRegex(ValueError, "requires milestones"):
            select_best_checkpoint(self.root, self.cfg)
        self.assertEqual(best_path.read_text(), '{"original": true}\n')

    def test_pipeline_uses_saved_test_request(self):
        result_path = run_pipeline(self.root, self.cfg)
        result = json.loads(result_path.read_text())
        best = json.loads((self.root / "best_ckpt.json").read_text())
        selected = json.loads((self.root / best["selection_result"]).read_text())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["request"]["seeds"], selected["test_request"]["seeds"])
        self.assertEqual(result["request"]["runner_options"], selected["test_request"]["runner_options"])
        self.assertEqual(result["request"]["source"]["selection_result"], best["selection_result"])
        self.assertTrue(result["request"]["source"]["heldout_from_selection"])
        self.assertEqual(result["global_step"], best["global_step"])
        self.assertEqual(len(self.restores), len(self.milestones) + 1)
        self.assertFalse(any(self.videos))
        self.assertTrue(self.cfg["eval"]["video"]["enabled"])


if __name__ == "__main__":
    unittest.main()
