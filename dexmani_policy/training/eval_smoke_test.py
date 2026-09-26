"""Synthetic eval/config/runner integration, without a simulator or video output.

Run: python -m dexmani_policy.training.eval_smoke_test
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.core.inference_smoke_test import small_policy
from dexmani_policy.common.config import normalize_eval_config
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy import eval_best_ckpt, select_best_ckpt, record_demo
from dexmani_policy.training.eval_utils import (
    add_inference_steps_argument, parse_eval_overrides, read_best_ckpt_json,
)


class OneStepEnv:
    action_cnt = 0

    def reset(self, **kwargs):
        return {"point_cloud": np.ones((64, 6), dtype=np.float32),
                "joint_state": np.zeros(19, dtype=np.float32)}, {}

    def step(self, action):
        self.action_cnt = 1
        obs, _ = self.reset()
        return obs, 0, True, False, {"success": True, "success_condition": True}

    def get_video(self):
        return None

    def close(self):
        pass


class SyntheticRunner(BaseRunner):
    def __init__(self):
        super().__init__(n_obs_steps=2, default_eval_episodes=1, clear_cache_freq=0)
        self.eval_seeds = None
        self.task_text = "synthetic"

    def get_seed_list(self):
        return self.eval_seeds if self.eval_seeds is not None else [1, 2, 3, 4]

    def make_env(self):
        return OneStepEnv()


def selection_record(directory, *, legacy=False):
    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "checkpoints/model.pt").touch()
    record = dict(
        ckpt_relpath="checkpoints/model.pt", pct=100, global_step=10,
        success_rate=1, avg_steps=1, n_episodes=1,
        inference={"use_ema": True, "denoise_steps" if legacy else "inference_steps": 4,
                   "policy_seed_mode": "episode_seed"},
        selection=dict(shuffle_seed=1, seeds=[1], initial_episodes=1, tie_break_used=False),
    )
    (directory / "best_ckpt.json").write_text(json.dumps(record))
    return record


class EvalIngressTest(unittest.TestCase):
    def test_cli_alias_and_conflict(self):
        parser = argparse.ArgumentParser()
        add_inference_steps_argument(parser)
        self.assertIsNone(parser.parse_args([]).inference_steps)
        for flag in ("--inference-steps", "--denoise-steps"):
            self.assertEqual(vars(parser.parse_args([flag, "4"])), {"inference_steps": 4})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--inference-steps", "4", "--denoise-steps", "5"])

    def test_config_normalization_before_merge(self):
        cfg = OmegaConf.create({"eval": {"denoise_steps": 10,
                                        "denoise_timesteps_list": None,
                                        "demo": {"denoise_steps": 3}}})
        normalized = normalize_eval_config(cfg)
        self.assertIn("denoise_steps", cfg.eval)  # caller is unchanged
        self.assertEqual(normalized.eval.demo.inference_steps, 3)
        merged = OmegaConf.merge(normalized, parse_eval_overrides(["eval.denoise_steps=4"]))
        self.assertEqual(merged.eval.inference_steps, 4)
        self.assertNotIn("denoise_steps", merged.eval)
        equal = normalize_eval_config({"eval": {"denoise_steps": 4, "inference_steps": 4}})
        self.assertEqual(dict(equal.eval), {"inference_steps": 4})
        for legacy, canonical, left, right in (
            ("denoise_steps", "inference_steps", 3, 4),
            ("denoise_timesteps_list", "inference_steps_list", [2], [4]),
            ("denoise_steps", "inference_steps", True, 1),
        ):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                normalize_eval_config({"eval": {legacy: left, canonical: right}})
        with self.assertRaisesRegex(ValueError, "agent"):
            parse_eval_overrides(["agent.num_inference_steps=4"])

    def test_selection_record_and_override_precedence(self):
        cfg = OmegaConf.create({"env_runner": {}, "eval": {"denoise_steps": 10}})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for legacy in (False, True):
                record = selection_record(directory, legacy=legacy)
                normalized = read_best_ckpt_json(directory)
                self.assertEqual(normalized["inference"]["inference_steps"], 4)
                self.assertNotIn("denoise_steps", normalized["inference"])
                for overrides, cli, expected in (([], None, [4]),
                                                (["eval.denoise_steps=2"], None, [2]),
                                                (["eval.inference_steps_list=[1,2]"], None, [1, 2]),
                                                (["eval.inference_steps=2"], 3, [3])):
                    _, _, steps, _ = eval_best_ckpt._resolve_final_eval_request(
                        cfg, directory, "best", overrides, cli_inference_steps=cli,
                    )
                    self.assertEqual(steps, expected)
                _, steps = record_demo._resolve_demo_inference(
                    cfg, directory, "best", cli_use_ema=None, cli_inference_steps=None,
                )
                self.assertEqual(steps, [4])
            record["inference"]["inference_steps"] = 8
            (directory / "best_ckpt.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_best_ckpt_json(directory)
            for invalid in (0, -1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "inference_steps"):
                    eval_best_ckpt._resolve_final_eval_request(
                        cfg, directory, "latest", [], cli_inference_steps=invalid,
                    )


class EvalRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg, cls.agent, _ = small_policy("maniflow")
        cls.cfg.training.device = "cpu"

    def test_config_eval_runner_agent_decoder_and_sweep(self):
        runner = SyntheticRunner()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg, use_ema, steps, _ = eval_best_ckpt._resolve_final_eval_request(
                self.cfg, directory, "latest", ["eval.inference_steps_list=[1,4]"],
            )
            setup = (self.agent, runner, directory / "model.pt", "synthetic", 0)
            with (
                patch.object(eval_best_ckpt, "_setup_eval", return_value=setup),
                patch.object(self.agent.action_decoder.model, "forward", wraps=self.agent.action_decoder.model.forward) as forward,
                patch.object(self.agent, "predict_action", wraps=self.agent.predict_action) as predict,
            ):
                results = eval_best_ckpt.evaluate_checkpoint_sweep(
                    directory, cfg, ckpt_tag_or_path="latest", episodes=1,
                    inference_steps_list=steps, use_ema=use_ema, result_save_dir=directory / "results",
                )
                self.assertEqual(forward.call_count, 5)
                self.assertEqual([call.kwargs["inference_steps"] for call in predict.call_args_list], [1, 4])
            self.assertEqual([r["inference_steps"] for r in results], [1, 4])
            for nfe in steps:
                details = json.loads((directory / f"results/inference_steps{nfe}/result_details.json").read_text())
                self.assertEqual(details["inference_steps"], nfe)
            self.assertEqual(self.agent.action_decoder.time_sampler.num_steps, 10)
            self.assertEqual(self.agent.action_decoder.num_inference_steps, 10)

    def test_selection_writes_canonical_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoints = directory / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "epoch=0-step=10-milestone=100pct.pt").touch()
            with (
                patch.object(select_best_ckpt, "build_eval_components", return_value=(SyntheticRunner(), None)),
                patch.object(select_best_ckpt, "load_ckpt_for_inference", return_value=self.agent),
            ):
                select_best_ckpt.select_best_checkpoint(
                    directory, self.cfg, initial_episodes=1, batch_size=0,
                    max_episodes=1, inference_steps=2,
                )
            record = read_best_ckpt_json(directory)
            self.assertEqual(record["inference"]["inference_steps"], 2)
            self.assertNotIn("denoise_steps", json.loads((directory / "best_ckpt.json").read_text())["inference"])

    def test_invalid_override_before_env_construction(self):
        runner = SyntheticRunner()
        for invalid in (0, -1, 1.5, True):
            with patch.object(runner, "make_env") as make_env, self.assertRaises(ValueError):
                runner.run(self.agent, inference_steps=invalid)
            make_env.assert_not_called()
        # A removed keyword must fail, including the former **kwargs ingress.
        with self.assertRaises(TypeError):
            runner.run_one_episode(self.agent, OneStepEnv(), 1, denoise_timesteps=4)

    def test_multitask_parameter_routing_without_simulator(self):
        # Stub only the unavailable simulator import; execute the real runner methods.
        with patch.dict(sys.modules, {"dexmani_sim": SimpleNamespace(DATA_DIR=Path("/tmp"))}):
            from dexmani_policy.env_runner.multi_task_sim_runner import MultiTaskSimRunner
        runner = MultiTaskSimRunner.__new__(MultiTaskSimRunner)
        runner.runners = {"first": SyntheticRunner(), "second": SyntheticRunner()}
        runner.eval_seeds = None
        with patch.object(self.agent.action_decoder.model, "forward", wraps=self.agent.action_decoder.model.forward) as forward:
            result = runner.run(self.agent, inference_steps=2, eval_episodes=1)
        self.assertEqual(forward.call_count, 4)
        self.assertEqual(set(result["per_task"]), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
