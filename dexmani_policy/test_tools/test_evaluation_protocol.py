import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from dexmani_policy.evaluation.protocol import (
    build_eval_protocol,
    load_eval_protocol,
    merge_eval_config,
    parse_overrides,
)
from dexmani_policy.evaluation.seeds import split_evaluation_seeds
from dexmani_policy.evaluation.selection import candidate_score, select_checkpoint


def experiment_config():
    return {
        "action_key": "action",
        "agent": {"n_obs_steps": 2, "n_action_steps": 4, "horizon": 8},
        "dataset": {"action_key": "action", "sensor_modalities": ["joint_state"]},
        "policy_metadata": {"action_start_index": 1},
        "env_runner": {"task_name": "task", "eval_seeds": list(range(10))},
        "training": {"device": "cpu", "loop": {"total_train_steps": 10}},
        "eval": {
            "seed": 1024, "episodes": 3, "selection_episodes": 2, "seeds": None,
            "inference_steps": 10, "action_steps": None, "use_ema": True,
            "video": {"enabled": True},
        },
    }


class SeedTests(unittest.TestCase):
    def test_test_prefix_is_independent_of_selection_budget_and_pool_order(self):
        state = random.getstate()
        test, selection = split_evaluation_seeds(list(range(10)), 1024, 3, 2)
        larger_test, larger_selection = split_evaluation_seeds(list(reversed(range(10))), 1024, 3, 5)
        self.assertEqual(test, larger_test)
        self.assertEqual(selection, larger_selection[:2])
        self.assertTrue(set(test).isdisjoint(larger_selection))
        self.assertEqual(state, random.getstate())

    def test_duplicate_seeds_and_insufficient_budget_are_rejected(self):
        for pool in ([1, 1, 2, 3, 4], [1, 2, 3, 4]):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                split_evaluation_seeds(pool, 1024, 3, 2)


class ProtocolTests(unittest.TestCase):
    def test_action_window_uses_saved_recipe_and_rejects_silent_truncation(self):
        cfg = experiment_config()
        protocol = build_eval_protocol(cfg, "model.pt", [1], source={})
        self.assertEqual(protocol.action_steps, 4)
        self.assertEqual(protocol.runner_options["n_obs_steps"], 2)
        self.assertNotIn("eval_seeds", protocol.runner_options)
        cfg["eval"]["action_steps"] = 8
        with self.assertRaisesRegex(ValueError, "future_steps"):
            build_eval_protocol(cfg, "model.pt", [1], source={})

    def test_overrides_cannot_change_model_inputs_or_sim_tactile_source(self):
        cfg = experiment_config()
        with self.assertRaisesRegex(ValueError, "Model inputs"):
            merge_eval_config(cfg, parse_overrides(["env_runner.n_obs_steps=3"]))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_overrides(["agent.n_obs_steps=3"])
        cfg["dataset"]["sensor_modalities"].append("finger_tactile_force")
        cfg["tactile_source"] = "real"
        with self.assertRaisesRegex(ValueError, "sim-source"):
            merge_eval_config(cfg)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = experiment_config()
        (self.root / "checkpoints").mkdir()
        for pct in (20, 40, 60, 80, 100):
            (self.root / "checkpoints" / f"epoch=1-step={pct // 10}-milestone={pct}pct.pt").touch()
        OmegaConf.save(self.cfg, self.root / "config.yaml")

    def rollout_result(self, protocol, *, output_root):
        step = protocol.source["global_step"]
        stats = {"success_rate": 0.5, "avg_steps": 2 if step in (4, 6) else 5}
        return {"statistics": stats, "global_step": step}, output_root / f"{step}/result_details.json"

    def test_candidates_share_a_budget_and_freeze_separate_test_settings(self):
        with patch("dexmani_policy.evaluation.rollout.evaluate_policy", side_effect=self.rollout_result) as evaluate:
            record, checkpoint = select_checkpoint(
                self.root, self.cfg, overrides=parse_overrides(["eval.video.enabled=true"])
            )

        protocols = [call.args[0] for call in evaluate.call_args_list]
        self.assertEqual(len(protocols), 5)
        self.assertEqual(record["global_step"], 6)
        self.assertEqual(checkpoint, self.root / record["ckpt_relpath"])
        self.assertTrue(all(protocol.seeds == record["selection_seeds"] for protocol in protocols))
        self.assertTrue(all(not protocol.config["eval"]["video"]["enabled"] for protocol in protocols))
        self.assertTrue(record["test_request"]["config"]["eval"]["video"]["enabled"])
        self.assertTrue(set(record["selection_seeds"]).isdisjoint(record["test_request"]["seeds"]))
        self.assertTrue(all("result" in row and "statistics" in row for row in record["candidates"]))
        best = json.loads((self.root / "best_ckpt.json").read_text())
        self.assertEqual(best["selection_result"], record["selection_result"])

        ordinary = load_eval_protocol(self.root, overrides=parse_overrides(["eval.inference_steps=3"]))
        self.assertEqual(ordinary.inference_steps, 3)
        self.assertFalse(ordinary.source["heldout_from_selection"])

    def test_frozen_test_survives_relocation_without_experiment_config(self):
        with patch("dexmani_policy.evaluation.rollout.evaluate_policy", side_effect=self.rollout_result):
            record, _ = select_checkpoint(self.root, self.cfg)
        (self.root / "config.yaml").unlink()
        moved = self.root / "relocated"
        moved.mkdir()
        (self.root / "checkpoints").rename(moved / "checkpoints")
        (self.root / "eval_ckpt_selector").rename(moved / "eval_ckpt_selector")
        selection = moved / record["selection_result"]
        protocol = load_eval_protocol(
            moved, selection_record=selection, overrides=parse_overrides(["training.device=cuda:1"])
        )
        self.assertEqual(protocol.checkpoint, str(moved / record["ckpt_relpath"]))
        self.assertEqual(protocol.device, "cuda:1")
        self.assertEqual(protocol.seeds, record["test_request"]["seeds"])
        self.assertTrue(protocol.source["heldout_from_selection"])
        self.assertFalse(protocol.config["eval"]["video"]["enabled"])
        with self.assertRaisesRegex(ValueError, "Frozen test"):
            load_eval_protocol(moved, selection_record=selection, overrides={"eval": {"action_steps": 1}})

    def test_technical_failure_preserves_old_best_and_completed_candidates(self):
        best = self.root / "best_ckpt.json"
        best.write_text('{"previous": true}')
        old_best = best.read_bytes()
        directory = self.root / "eval_ckpt_selector" / "failed"
        completed = self.rollout_result(
            build_eval_protocol(self.cfg, "model.pt", [1], source={"global_step": 2}),
            output_root=directory / "candidates",
        )
        with patch(
            "dexmani_policy.evaluation.rollout.evaluate_policy",
            side_effect=[completed, RuntimeError("simulation failed")],
        ), self.assertRaisesRegex(RuntimeError, "simulation failed"):
            select_checkpoint(self.root, self.cfg, output_dir=directory)

        record = json.loads((directory / "selection_result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(best.read_bytes(), old_best)
        self.assertIn("statistics", record["candidates"][0])
        self.assertNotIn("statistics", record["candidates"][1])

    def test_missing_milestones_fail_before_any_rollout(self):
        next((self.root / "checkpoints").glob("*20pct.pt")).unlink()
        with patch("dexmani_policy.evaluation.rollout.evaluate_policy") as evaluate:
            with self.assertRaisesRegex(ValueError, "milestones"):
                select_checkpoint(self.root, self.cfg)
        evaluate.assert_not_called()

    def test_rank_uses_success_then_steps_then_training_budget_then_order(self):
        candidates = [
            {"statistics": {"success_rate": 0, "avg_steps": None}, "global_step": 10},
            {"statistics": {"success_rate": 0.5, "avg_steps": 9}, "global_step": 10},
            {"statistics": {"success_rate": 0.5, "avg_steps": 2}, "global_step": 4},
            {"statistics": {"success_rate": 0.5, "avg_steps": 2}, "global_step": 6},
        ]
        self.assertIs(max(candidates, key=candidate_score), candidates[3])
        tied = copy.deepcopy(candidates[3])
        self.assertIs(max([candidates[3], tied], key=candidate_score), candidates[3])
        failed = [dict(candidates[0], global_step=2), candidates[0]]
        self.assertIs(max(failed, key=candidate_score), candidates[0])


if __name__ == "__main__":
    unittest.main()
