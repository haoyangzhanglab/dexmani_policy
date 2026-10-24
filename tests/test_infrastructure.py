"""Infrastructure regressions with real request/rollout code and synthetic environments."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch import nn

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.common.experiments import rank_selection_results, resolve_best_checkpoint
from dexmani_policy.deployment.metadata import PolicyInfo
from dexmani_policy.deployment.runtime import PolicyRuntime
from dexmani_policy.env_runner.common.metrics import (
    compute_evaluation_statistics,
    validate_episodes,
)
from dexmani_policy.env_runner.common.seeds import split_evaluation_seeds, validate_seed_pool
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation.config import parse_evaluation_overrides, resolve_evaluation_request
from dexmani_policy.evaluation.rollout import evaluate_policy
from dexmani_policy.evaluation.selection import select_best_checkpoint


class WindowPolicy(BasePolicy):
    def __init__(self):
        super().__init__(nn.Identity(), nn.Identity(), 16, 2, 8, 28)
        self.anchor = nn.Parameter(torch.zeros(()))
        self.checkpoint_step = 0

    @property
    def control_action_dim(self):
        return 19

    @property
    def consumed_observation_fields(self):
        return ("joint_state", "point_cloud")

    def predict_action(self, obs_dict, inference_steps=None):
        values = torch.arange(16 * 28, dtype=torch.float32).reshape(1, 16, 28)
        return self.format_action_output(values + self.anchor)


class SyntheticEnvironment:
    def __init__(self, failing_seed=None):
        self.failing_seed = failing_seed
        self.closed = False
        self.seeds = []

    def observation(self):
        return {
            "joint_state": np.zeros(19, dtype=np.float32),
            "point_cloud": np.zeros((8, 6), dtype=np.float32),
        }

    def reset(self, seed):
        self.seeds.append(seed)
        if seed == self.failing_seed:
            raise RuntimeError("synthetic environment error")
        self.action_cnt = 0
        return self.observation(), {}

    def step(self, action):
        self.action_cnt += 1
        return (
            self.observation(),
            0,
            self.action_cnt == 2,
            False,
            {"success_condition": True, "success": True},
        )

    def close(self):
        self.closed = True


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "checkpoints").mkdir()
        self.path = self.root / "checkpoints/epoch=0001-step=00000001-milestone=20pct.pt"
        self.path.touch()
        self.cfg = load_policy_config("dp3")
        with open_dict(self.cfg):
            self.cfg.workspace.output_dir = str(self.root)
            self.cfg.training.device = "cpu"
            self.cfg.env_runner.eval_seeds = list(range(100))
            self.cfg.eval.video.enabled = False
        self.cfg = OmegaConf.to_container(self.cfg, resolve=True)
        self.cfg["policy_metadata"] = WindowPolicy().inference_metadata

    def load_model(self, cfg, path, **kwargs):
        model = WindowPolicy()
        model.checkpoint_step = int(Path(path).name.split("-step=")[1].split("-")[0])
        return model

    def test_complete_future_and_real_dtype_without_second_crop(self):
        model = WindowPolicy()
        cfg = copy.deepcopy(self.cfg)
        cfg["policy_metadata"] = model.inference_metadata
        cfg["agent"]["action_dim"] = 28
        info = PolicyInfo(self.root, self.path, OmegaConf.create(cfg), "raw", 10)
        runtime = PolicyRuntime(model, info, device="cpu", seed=0)
        observation = {
            key: np.stack([value, value])
            for key, value in SyntheticEnvironment().observation().items()
        }
        future = model.predict_action({})["future_actions"]
        self.assertEqual(future.shape, (1, 15, 19))
        expected = torch.arange(16 * 28).reshape(16, 28)[1:, :19].numpy()
        np.testing.assert_array_equal(runtime.predict(observation), expected)
        self.assertEqual(runtime.predict(observation).dtype, np.float64)
        runner = SimRunner("synthetic", 2, list(observation))
        actions = runner.predict_action_chunk({}, model, action_steps=4)
        np.testing.assert_array_equal(actions, expected[:4])
        self.assertEqual(actions.dtype, np.float32)
        self.assertFalse(hasattr(runtime, "configure_execution"))
        episode_rng = torch.get_rng_state()

        def stochastic_prediction(*args, **kwargs):
            return np.full((15, 19), torch.rand(()).item())

        with (
            patch(
                "dexmani_policy.deployment.runtime.make_warmup_observation",
                return_value=observation,
            ),
            patch.object(runtime, "predict", side_effect=stochastic_prediction),
        ):
            runtime.warmup(samples=2)
            self.assertTrue(torch.equal(torch.get_rng_state(), episode_rng))
        with (
            patch(
                "dexmani_policy.deployment.runtime.make_warmup_observation",
                return_value=observation,
            ),
            patch.object(runtime, "predict", side_effect=RuntimeError("warmup failed")),
            self.assertRaisesRegex(RuntimeError, "warmup failed"),
        ):
            runtime.warmup(samples=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), episode_rng))
        full = torch.zeros(1, 16, 28)
        full[0, 0, -1] = float("nan")
        with self.assertRaises(ValueError):
            model.format_action_output(full)

    def test_statistics_weight_successes_and_preserve_fraction(self):
        rows = [
            dict(task_name=t, seed=i, success=success, steps=steps, total_steps=80)
            for i, (t, success, steps) in enumerate(
                [
                    ("a", True, 10),
                    ("a", False, 2),
                    ("b", True, 20),
                    ("b", True, 40),
                    ("b", True, 60),
                ]
            )
        ]
        stats = compute_evaluation_statistics(rows)
        self.assertEqual(stats["avg_steps"], 32.5)
        self.assertEqual(stats["micro_success_rate"], 0.8)
        self.assertEqual(stats["macro_success_rate"], 0.75)
        for row in rows:
            row["success"] = False
        self.assertIsNone(compute_evaluation_statistics(rows)["avg_steps"])
        rows[0]["success"], rows[0]["steps"] = True, None
        self.assertIsNone(compute_evaluation_statistics(rows)["avg_steps"])

    def test_ordinary_evaluation_ignores_selection_capacity(self):
        self.cfg["eval"].update(episodes=2, selection_episodes=10000)
        self.cfg["env_runner"]["eval_seeds"] = [11, 12]
        request = resolve_evaluation_request(self.cfg, self.root, str(self.path))
        self.assertFalse(request.source["heldout_from_selection"])
        self.assertEqual(set(next(iter(request.task_seeds.values()))), {11, 12})
        self.assertIsInstance(request.inference_steps, int)

    def test_seed_prefix_stable_and_invalid_pools_rejected(self):
        pools = {"a": list(range(100))}
        a, _ = split_evaluation_seeds(pools, 1024, 70, 25)
        b, _ = split_evaluation_seeds(pools, 1024, 70, 30)
        self.assertEqual(a, b)
        for seeds in ([1, 1], [-1], [2**32], [True]):
            with self.assertRaises(ValueError):
                validate_seed_pool("a", seeds)
        with (
            self.assertRaises(FileNotFoundError),
            patch(
                "dexmani_policy.evaluation.config.task_seed_pools",
                side_effect=FileNotFoundError("missing source"),
            ),
        ):
            resolve_evaluation_request(self.cfg, self.root, str(self.path))

    def test_failure_keeps_prefix_and_closes_environment(self):
        self.cfg["eval"]["task_seeds"] = {self.cfg["task_name"]: [1, 2, 3]}
        request = resolve_evaluation_request(self.cfg, self.root, str(self.path))
        environment = SyntheticEnvironment(failing_seed=2)
        output = self.root / "failed"
        with (
            patch.object(SimRunner, "make_env", return_value=environment),
            patch(
                "dexmani_policy.evaluation.rollout.restore_policy_agent",
                side_effect=self.load_model,
            ),
            self.assertRaisesRegex(RuntimeError, "synthetic environment error"),
        ):
            evaluate_policy(request, output_root=self.root, output_dir=output)
        result = json.loads((output / "result_details.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["seed"] for r in result["episodes"]], [1])
        self.assertIsNone(result["statistics"])
        self.assertTrue(environment.closed)
        self.assertEqual(environment.seeds, [1, 2])
        with self.assertRaises(FileExistsError):
            evaluate_policy(request, output_root=self.root, output_dir=output)

    def test_selection_fixed_budget_frozen_test_and_evidence_validation(self):
        second = self.root / "checkpoints/epoch=0002-step=00000002-milestone=40pct.pt"
        second.touch()
        environments = []

        def make_env():
            env = SyntheticEnvironment()
            environments.append(env)
            return env

        with (
            patch.object(SimRunner, "make_env", side_effect=make_env),
            patch(
                "dexmani_policy.evaluation.rollout.restore_policy_agent",
                side_effect=self.load_model,
            ) as loader,
        ):
            path = select_best_checkpoint(self.root, self.cfg)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual([len(env.seeds) for env in environments], [25, 25])
        self.assertEqual(environments[0].seeds, environments[1].seeds)
        record, checkpoint = resolve_best_checkpoint(self.root)
        self.assertEqual(checkpoint, second)
        self.assertEqual(record["status"], "completed")
        results = [json.loads((self.root / ref).read_text()) for ref in record["candidate_results"]]
        for result in results:
            for episode in result["episodes"]:
                episode["success"] = False
                episode["steps"] = None
            result["statistics"] = compute_evaluation_statistics(result["episodes"])
        self.assertEqual(rank_selection_results(record, results), 1)
        results[0]["status"] = "failed"
        with self.assertRaises(ValueError):
            rank_selection_results(record, results)
        test = resolve_evaluation_request(self.cfg, self.root, selection_record=path)
        self.assertEqual(len(next(iter(test.task_seeds.values()))), 70)
        self.assertFalse(set(environments[0].seeds) & set(next(iter(test.task_seeds.values()))))
        for override in (
            "eval.inference_steps=2",
            "eval.action_steps=1",
            "eval.use_ema=false",
            "eval.seed=1",
            "env_runner.env_kwargs.table_random=false",
        ):
            with self.assertRaises(ValueError):
                resolve_evaluation_request(
                    self.cfg,
                    self.root,
                    selection_record=path,
                    overrides=parse_evaluation_overrides([override]),
                )
        moved_device = resolve_evaluation_request(
            self.cfg,
            self.root,
            selection_record=path,
            overrides=parse_evaluation_overrides(["training.device=cuda:1"]),
        )
        self.assertEqual(moved_device.device, "cuda:1")
        result_path = self.root / record["candidate_results"][0]
        result = json.loads(result_path.read_text())
        result["episodes"].pop()
        result_path.write_text(json.dumps(result))
        with self.assertRaises(RuntimeError):
            resolve_best_checkpoint(self.root)

    def test_load_and_cleanup_failures_never_publish_success(self):
        self.cfg["eval"]["task_seeds"] = {self.cfg["task_name"]: [1]}
        request = resolve_evaluation_request(self.cfg, self.root, str(self.path))
        output = self.root / "load_failed"
        with (
            patch(
                "dexmani_policy.evaluation.rollout.restore_policy_agent",
                side_effect=RuntimeError("load failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "load failed"),
        ):
            evaluate_policy(request, output_root=self.root, output_dir=output)
        result = json.loads((output / "result_details.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["episodes"], [])
        model = self.load_model(self.cfg, self.path)
        output = self.root / "cleanup_failed"
        with (
            patch.object(SimRunner, "make_env", return_value=SyntheticEnvironment()),
            patch("dexmani_policy.evaluation.rollout.restore_policy_agent", return_value=model),
            patch.object(model, "to", side_effect=RuntimeError("cleanup failed")),
            self.assertRaisesRegex(RuntimeError, "cleanup failed"),
        ):
            evaluate_policy(request, output_root=self.root, output_dir=output)
        result = json.loads((output / "result_details.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["episodes"]), 1)
        self.assertIsNone(result["statistics"])
        self.assertIn("cleanup failed", result["cleanup_error"])

    def test_exact_coverage_rejects_duplicate_missing_and_extra(self):
        row = dict(task_name="a", seed=1, success=True, steps=1, total_steps=2)
        for rows in ([], [row, row], [row, {**row, "seed": 3}]):
            with self.assertRaises(RuntimeError):
                validate_episodes(rows, {"a": [1, 2]})

    def test_inspection_has_no_torch_and_runner_has_no_sim_import(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import dexmani_policy.deployment; assert 'torch' not in sys.modules",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from dexmani_policy.env_runner.sim_runner import SimRunner; assert 'dexmani_sim' not in sys.modules",
            ],
            check=True,
        )

    def test_rtc_keeps_local_autograd_and_zero_guidance_numerics(self):
        from dexmani_policy.agents.action_decoders.diffusion import Diffusion

        class Denoiser(nn.Module):
            def forward(self, x, timestep, context):
                return 0.25 * x + context[:, None, :]

        decoder = Diffusion(Denoiser(), num_inference_steps=2)
        cond, template, prefix = torch.zeros(1, 2), torch.zeros(1, 4, 2), torch.ones(1, 3, 2)
        with torch.inference_mode():
            torch.manual_seed(4)
            ordinary = decoder.predict_action(cond, template)
            torch.manual_seed(4)
            unguided = decoder.predict_action(
                cond,
                template,
                rtc_prefix=prefix,
                history_offset=1,
                delay_steps=1,
                rtc_guidance_cap=0,
            )
            torch.testing.assert_close(ordinary, unguided, rtol=0, atol=0)
            torch.manual_seed(4)
            guided = decoder.predict_action(
                cond,
                template,
                rtc_prefix=prefix,
                history_offset=1,
                delay_steps=1,
                rtc_guidance_cap=1,
            )
            self.assertTrue(torch.isfinite(guided).all())
            self.assertFalse(torch.equal(guided, ordinary))
        self.assertEqual(decoder.noise_scheduler.alphas_cumprod.dtype, torch.float32)
        with self.assertRaises(ValueError):
            decoder.predict_action(cond, template, rtc_prefix=prefix, rtc_guidance_cap=float("nan"))

    def test_grouped_manifest_keeps_trials_whole_and_checks_sources(self):
        from dexmani_policy.datasets.common.episode_split import (
            build_index_manifest,
            validate_split_manifest,
        )

        attrs = {
            "episode_ids": ["a1", "a2", "b1", "b2", "c1", "c2"],
            "trial_ids": {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "c1": "c", "c2": "c"},
        }
        manifest = build_index_manifest(
            "/source", [10, 20, 30, 40, 50, 60], attrs, val_ratio=0.3, max_train_episodes=3
        )
        train = {attrs["trial_ids"][key] for key in manifest["train_ids"]}
        val = {attrs["trial_ids"][key] for key in manifest["val_ids"]}
        self.assertFalse(train & val)
        self.assertEqual(len(manifest["train_ids"]), 2)
        indexed = build_index_manifest("/source", [10, 20], {"source_id": "corpus"}, val_ratio=0)
        validate_split_manifest(
            indexed, {"source_id": "corpus"}, 2, episode_ends=[10, 20], zarr_path="/moved"
        )
        with self.assertRaises(ValueError):
            validate_split_manifest(
                indexed, {"source_id": "different"}, 2, episode_ends=[10, 20], zarr_path="/source"
            )

    def test_training_seed_does_not_change_child_split_seed_or_padding(self):
        for name in ("dp3", "dqrise", "lamp", "latent_flow", "multitask_dit"):
            a = load_policy_config(name, ["training.seed=1"])
            b = load_policy_config(name, ["training.seed=9"])
            children_a = a.dataset.get("datasets", [a.dataset])
            children_b = b.dataset.get("datasets", [b.dataset])
            self.assertEqual([c.seed for c in children_a], [c.seed for c in children_b])
            before = [c.pad_after for c in children_b]
            b.n_action_steps = 2
            self.assertEqual(before, [c.pad_after for c in children_b])
            if name == "multitask_dit":
                self.assertNotEqual(a.dataset.seed, b.dataset.seed)


if __name__ == "__main__":
    unittest.main()
