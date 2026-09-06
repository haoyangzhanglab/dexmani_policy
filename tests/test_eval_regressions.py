import contextlib
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy import eval_best_ckpt, record_demo
from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import validate_action_key_consistency
from dexmani_policy.datasets.base_dataset import preprocess_validation_rgb
from dexmani_policy.env_runner.base_runner import BaseRunner, EvalEpisodeError
from dexmani_policy.eval_best_ckpt import (
    _resolve_final_eval_request,
    _run_one_timestep,
    _select_eval_seeds,
)
from dexmani_policy.select_best_ckpt import evaluate_checkpoint, select_best_checkpoint
from dexmani_policy.training.eval_utils import (
    MilestoneCheckpoint,
    load_ckpt_for_inference,
    read_best_ckpt_json,
    resolve_checkpoint_path,
)

# Import the multi-task runner without loading or running dexmani_sim.  Its
# child runners are replaced with fakes in the test below.
_dexmani_sim_stub = types.ModuleType("dexmani_sim")
_dexmani_sim_stub.DATA_DIR = Path("/nonexistent")
with mock.patch.dict(sys.modules, {"dexmani_sim": _dexmani_sim_stub}):
    from dexmani_policy.env_runner.multi_task_sim_runner import MultiTaskSimRunner


class _ControlActionAgent:
    def predict_action(self, **kwargs):
        control = torch.full((1, 8, 19), 17.0, dtype=torch.float32)
        return {
            "pred_action": torch.arange(16 * 28, dtype=torch.float32).reshape(
                1, 16, 28
            ),
            "control_action": control,
        }


def _valid_record(ckpt_relpath="checkpoints/selected.pt"):
    return {
        "record_version": 2,
        "ckpt_relpath": ckpt_relpath,
        "pct": 80,
        "global_step": 80000,
        "success_rate": 0.75,
        "avg_steps": 12.5,
        "n_episodes": 4,
        "inference": {
            "use_ema": True,
            "denoise_steps": 8,
            "temporal_ensemble_coeff": 0.1,
            "policy_seed_mode": "episode_seed",
        },
        "selection": {
            "shuffle_seed": 17,
            "seeds": [1, 3],
            "initial_episodes": 2,
            "tie_break_used": False,
        },
    }


class EvalRegressionTests(unittest.TestCase):
    def test_temporal_ensemble_coefficient_validation(self):
        BaseRunner(
            n_obs_steps=2,
            default_eval_episodes=1,
            temporal_ensemble_coeff=None,
        )
        for coeff in (0, 0.01):
            with self.subTest(coeff=coeff):
                with self.assertRaisesRegex(ValueError, "Re-export this experiment"):
                    BaseRunner(
                        n_obs_steps=2,
                        default_eval_episodes=1,
                        temporal_ensemble_coeff=coeff,
                    )

    def test_periodic_refresh_closes_before_constructing_next_env(self):
        events = []

        class FakeEnv:
            def __init__(self, index):
                self.index = index
                self.action_cnt = 0
                self.closed = False

            def get_video(self):
                return None

            def close(self):
                self.closed = True
                events.append(f"close {self.index}")

        class FakeRunner(BaseRunner):
            def __init__(self):
                super().__init__(
                    n_obs_steps=2,
                    default_eval_episodes=1,
                    clear_cache_freq=1,
                    env_video_fps=15,
                )
                self.next_env = 0
                self.created_envs = []

            def make_env(self):
                if self.created_envs and not self.created_envs[-1].closed:
                    raise AssertionError("old environment was still live during refresh")
                index = self.next_env
                self.next_env += 1
                events.append(f"make {index}")
                env = FakeEnv(index)
                self.created_envs.append(env)
                return env

            def get_seed_list(self):
                return [7]

            def run_one_episode(self, agent, env, episode_seed, denoise_timesteps=None):
                return False, None

        FakeRunner().run(agent=None, eval_episodes=1)

        self.assertEqual(events[:3], ["make 0", "close 0", "make 1"])

    def test_runner_rgb_matches_validation_preprocessing_and_keeps_raw_history(self):
        runner = BaseRunner(
            n_obs_steps=2,
            default_eval_episodes=1,
            sensor_modalities=["rgb", "joint_state"],
            rgb_preprocess_size=(6, 8),
            rgb_random_crop_size=(4, 4),
        )
        first = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        second = np.flip(first, axis=0).copy()
        runner.update_obs({"rgb": first, "joint_state": np.array([1.0, 2.0])})
        runner.update_obs({"rgb": second, "joint_state": np.array([3.0, 4.0])})

        raw_rgb = runner.get_stacked_obs()["rgb"]
        expected = preprocess_validation_rgb(
            raw_rgb,
            resize_hw=(6, 8),
            center_crop_hw=(4, 4),
            keep_uint8=False,
        )
        batch = runner.get_obs_batch(device="cpu")

        torch.testing.assert_close(batch["rgb"], expected.unsqueeze(0))
        self.assertEqual(batch["rgb"].dtype, torch.float32)
        np.testing.assert_array_equal(runner._obs_buffer["rgb"][0], first)
        np.testing.assert_array_equal(runner._obs_buffer["rgb"][1], second)

    def test_runner_leaves_non_rgb_modalities_unchanged(self):
        runner = BaseRunner(
            n_obs_steps=2,
            default_eval_episodes=1,
            sensor_modalities=["joint_state"],
        )
        first = np.array([1.0, 2.0], dtype=np.float32)
        second = np.array([3.0, 4.0], dtype=np.float32)
        runner.update_obs({"joint_state": first})
        runner.update_obs({"joint_state": second})

        batch = runner.get_obs_batch(device="cpu")

        torch.testing.assert_close(
            batch["joint_state"],
            torch.from_numpy(np.stack([first, second])).unsqueeze(0),
        )

    def test_runner_uses_canonical_control_action(self):
        runner = BaseRunner(
            n_obs_steps=2,
            default_eval_episodes=1,
        )

        action_chunk = runner.get_action_chunk({}, _ControlActionAgent())

        self.assertEqual(action_chunk.shape, (8, 19))
        np.testing.assert_array_equal(action_chunk, np.full((8, 19), 17.0))

    def test_single_task_control_validation_is_preserved(self):
        validate_action_key_consistency(
            {
                "action_key": "action_ee",
                "env_runner": {"env_kwargs": {"control_mode": "ee"}},
            }
        )
        with self.assertRaisesRegex(
            ValueError, "env_runner.env_kwargs.control_mode='joint'"
        ):
            validate_action_key_consistency(
                {
                    "action_key": "action_ee",
                    "env_runner": {"env_kwargs": {"control_mode": "joint"}},
                }
            )

    def test_multi_task_control_validation_identifies_mismatched_task(self):
        with self.assertRaisesRegex(ValueError, "task 'open_box'.*control_mode='ee'"):
            validate_action_key_consistency(
                {
                    "action_key": "action",
                    "env_runner": {
                        "task_configs": [
                            {
                                "task_name": "pick_bottle",
                                "env_kwargs": {"control_mode": "joint"},
                            },
                            {
                                "task_name": "open_box",
                                "env_kwargs": {"control_mode": "ee"},
                            },
                        ]
                    },
                }
            )

    def test_multi_task_stops_after_fatal_episode_error(self):
        calls = []

        class FakeRunner:
            def __init__(self, task_text, error=None):
                self.task_text = task_text
                self.error = error

            def run(self, *args, **kwargs):
                calls.append(self.task_text)
                if self.error is not None:
                    raise self.error
                return {
                    "success_rate": 1.0,
                    "avg_steps": 1,
                    "videos": [],
                    "episode_details": [],
                }

        runner = object.__new__(MultiTaskSimRunner)
        runner.eval_seeds = None
        runner.runners = {
            "first": FakeRunner("first", EvalEpisodeError("runtime_error", 7, "boom")),
            "second": FakeRunner("second"),
        }

        with self.assertRaises(EvalEpisodeError):
            runner.run(agent=None)

        self.assertEqual(calls, ["first"])


class SelectionContractTests(unittest.TestCase):
    def test_missing_ema_fails_closed_and_raw_requires_explicit_false(self):
        checkpoint = SimpleNamespace(
            train_params=None,
            model_state={"weight": torch.tensor([1.0])},
            ema_model_state=None,
        )
        store = mock.Mock()
        store.load.return_value = checkpoint
        agent = mock.Mock()
        agent.normalizer.is_fitted.return_value = True

        with self.assertRaisesRegex(RuntimeError, "no EMA state"):
            load_ckpt_for_inference(agent, store, Path("missing-ema.pt"), True)
        agent.load_state_dict.assert_not_called()

        load_ckpt_for_inference(agent, store, Path("raw.pt"), False)
        loaded_state = agent.load_state_dict.call_args.args[0]
        torch.testing.assert_close(loaded_state["weight"], torch.tensor([1.0]))

    def test_strict_v2_record_resolves_relative_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir()
            selected = checkpoint_dir / "selected.pt"
            selected.touch()
            (checkpoint_dir / "latest.pt").touch()
            record = _valid_record()
            (exp_dir / "best_ckpt.json").write_text(json.dumps(record))

            parsed = read_best_ckpt_json(exp_dir)
            resolved, label = resolve_checkpoint_path(
                exp_dir, "best", CheckpointStore(checkpoint_dir)
            )

            self.assertEqual(parsed["ckpt_relpath"], "checkpoints/selected.pt")
            self.assertEqual(resolved, selected.resolve())
            self.assertIn("80%", label)

    def test_strict_v2_record_validates_coefficient_and_seed_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            checkpoint = exp_dir / "checkpoints" / "selected.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()

            for coeff in (None, 0, 0.01):
                with self.subTest(coeff=coeff):
                    record = _valid_record()
                    record["inference"]["temporal_ensemble_coeff"] = coeff
                    (exp_dir / "best_ckpt.json").write_text(json.dumps(record))
                    self.assertEqual(
                        read_best_ckpt_json(exp_dir)["inference"][
                            "temporal_ensemble_coeff"
                        ],
                        coeff,
                    )

            for coeff in (
                -0.01,
                float("nan"),
                float("inf"),
                float("-inf"),
                True,
                "0.01",
            ):
                with self.subTest(coeff=coeff):
                    record = _valid_record()
                    record["inference"]["temporal_ensemble_coeff"] = coeff
                    (exp_dir / "best_ckpt.json").write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, "temporal_ensemble_coeff"):
                        read_best_ckpt_json(exp_dir)

            for seed_mode in ("per_episode", None, True):
                with self.subTest(seed_mode=seed_mode):
                    record = _valid_record()
                    record["inference"]["policy_seed_mode"] = seed_mode
                    (exp_dir / "best_ckpt.json").write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, "policy_seed_mode"):
                        read_best_ckpt_json(exp_dir)

    def test_best_requires_a_v2_selection_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "latest.pt").touch()
            store = CheckpointStore(checkpoint_dir)

            with self.assertRaisesRegex(
                FileNotFoundError, "Selection record not found"
            ):
                resolve_checkpoint_path(exp_dir, "best", store)

            record = _valid_record()
            record["record_version"] = 1
            (exp_dir / "best_ckpt.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "record_version=2"):
                resolve_checkpoint_path(exp_dir, "best", store)

    def test_recorded_checkpoint_missing_never_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "latest.pt").touch()
            (exp_dir / "best_ckpt.json").write_text(json.dumps(_valid_record()))

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                resolve_checkpoint_path(
                    exp_dir, "best", CheckpointStore(checkpoint_dir)
                )

    def _run_selector(self, exp_dir: Path, tie: bool):
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        milestones = []
        for pct, step in ((20, 20), (40, 40)):
            path = checkpoint_dir / f"epoch=0-step={step}-milestone={pct}pct.pt"
            path.touch()
            milestones.append(MilestoneCheckpoint(path=path, pct=pct, global_step=step))

        cfg = OmegaConf.create(
            {
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "horizon": 16,
                "action_key": "action",
                "training": {"device": "cpu", "seed": 42},
                "env_runner": {
                    "env_kwargs": {"control_mode": "joint"},
                    "temporal_ensemble_coeff": 0.27,
                },
            }
        )

        class FakeRunner:
            def get_seed_list(self):
                return [10, 11, 12, 13]

        def fake_evaluate(
            agent,
            env_runner,
            checkpoint_store,
            ckpt,
            seeds,
            use_ema,
            denoise_steps,
            device,
            video_save_dir=None,
        ):
            if len(seeds) == 2:
                if tie:
                    successes = [True, False]
                else:
                    successes = [True, True] if ckpt.pct == 40 else [True, False]
            else:
                successes = [ckpt.pct == 40]
            return {
                "episode_details": [
                    {"success": success, "steps": index + 1}
                    for index, success in enumerate(successes)
                ]
            }

        with (
            mock.patch(
                "dexmani_policy.select_best_ckpt.discover_milestone_checkpoints",
                return_value=milestones,
            ),
            mock.patch(
                "dexmani_policy.select_best_ckpt.build_eval_components",
                return_value=(object(), FakeRunner(), object()),
            ),
            mock.patch(
                "dexmani_policy.select_best_ckpt.evaluate_checkpoint",
                side_effect=fake_evaluate,
            ),
        ):
            select_best_checkpoint(
                exp_dir,
                cfg,
                initial_episodes=2,
                batch_size=1,
                max_episodes=3,
                denoise_steps=4,
                use_ema=False,
                eval_seed=17,
            )
        return json.loads((exp_dir / "best_ckpt.json").read_text())

    def test_selector_records_actual_settings_and_no_tie_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._run_selector(Path(tmp), tie=False)

        expected = [10, 11, 12, 13]
        random.Random(17).shuffle(expected)
        self.assertEqual(record["record_version"], 2)
        self.assertNotIn("ckpt_path", record)
        self.assertEqual(record["inference"]["use_ema"], False)
        self.assertEqual(record["inference"]["denoise_steps"], 4)
        self.assertEqual(record["inference"]["temporal_ensemble_coeff"], 0.27)
        self.assertEqual(record["inference"]["policy_seed_mode"], "episode_seed")
        self.assertEqual(record["selection"]["shuffle_seed"], 17)
        self.assertEqual(record["selection"]["seeds"], expected[:2])
        self.assertEqual(record["selection"]["initial_episodes"], 2)
        self.assertFalse(record["selection"]["tie_break_used"])

    def test_selector_records_only_executed_tie_break_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._run_selector(Path(tmp), tie=True)

        expected = [10, 11, 12, 13]
        random.Random(17).shuffle(expected)
        self.assertEqual(record["selection"]["seeds"], expected[:3])
        self.assertTrue(record["selection"]["tie_break_used"])
        self.assertEqual(record["n_episodes"], 3)

    def test_selector_propagates_video_setting_to_multitask_child_runners(self):
        ckpt = MilestoneCheckpoint(path=Path("/tmp/checkpoint.pt"), pct=20, global_step=1)
        seeds = [3, 7]
        agent = mock.Mock()
        checkpoint_store = mock.sentinel.checkpoint_store

        for video_save_dir, expected_record_video in (
            (None, False),
            (Path("/tmp/videos"), True),
        ):
            with self.subTest(video_save_dir=video_save_dir):
                child_a = SimpleNamespace(record_video=not expected_record_video)
                child_b = SimpleNamespace(record_video=not expected_record_video)
                parent = SimpleNamespace(
                    runners={"task_a": child_a, "task_b": child_b},
                    run=mock.Mock(return_value={"episode_details": []}),
                )

                with mock.patch(
                    "dexmani_policy.select_best_ckpt.load_ckpt_for_inference"
                ) as load_ckpt:
                    evaluate_checkpoint(
                        agent,
                        parent,
                        checkpoint_store,
                        ckpt,
                        seeds,
                        use_ema=False,
                        denoise_steps=4,
                        device=torch.device("cpu"),
                        video_save_dir=video_save_dir,
                    )

                load_ckpt.assert_called_once_with(
                    agent, checkpoint_store, ckpt.path, False
                )
                agent.to.assert_called_with(torch.device("cpu"))
                agent.eval.assert_called_once_with()
                self.assertEqual(parent.eval_seeds, seeds)
                self.assertEqual(child_a.record_video, expected_record_video)
                self.assertEqual(child_b.record_video, expected_record_video)
                parent.run.assert_called_once_with(
                    agent,
                    denoise_timesteps=4,
                    eval_episodes=len(seeds),
                    video_save_dir=video_save_dir,
                )

            agent.reset_mock()

    def test_best_inference_precedence_handles_sections_lists_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "checkpoints").mkdir()
            (exp_dir / "checkpoints" / "selected.pt").touch()
            (exp_dir / "best_ckpt.json").write_text(json.dumps(_valid_record()))
            cfg = OmegaConf.create(
                {
                    "eval": {
                        "use_ema": False,
                        "denoise_steps": 2,
                        "denoise_timesteps_list": [2, 4],
                        "offline": {
                            "use_ema": False,
                            "denoise_timesteps_list": [6, 12],
                        },
                    },
                    "env_runner": {"temporal_ensemble_coeff": 0.9},
                }
            )

            resolved_cfg, use_ema, nfe, coeff, _ = _resolve_final_eval_request(
                cfg, exp_dir, "best", []
            )
            self.assertTrue(use_ema)
            self.assertEqual(nfe, [8])
            self.assertEqual(coeff, 0.1)
            self.assertEqual(resolved_cfg.env_runner.temporal_ensemble_coeff, 0.1)

            resolved_cfg, use_ema, nfe, coeff, _ = _resolve_final_eval_request(
                cfg,
                exp_dir,
                "best",
                [
                    "eval.use_ema=false",
                    "eval.offline.denoise_timesteps_list=[4,10]",
                    "env_runner.temporal_ensemble_coeff=0.2",
                ],
            )
            self.assertFalse(use_ema)
            self.assertEqual(nfe, [4, 10])
            self.assertEqual(coeff, 0.2)

            _, use_ema, nfe, coeff, _ = _resolve_final_eval_request(
                cfg,
                exp_dir,
                "best",
                [
                    "eval.offline.use_ema=false",
                    "eval.offline.denoise_timesteps_list=[4,10]",
                    "env_runner.temporal_ensemble_coeff=0.2",
                ],
                cli_use_ema=True,
                cli_denoise_steps=14,
            )
            self.assertTrue(use_ema)
            self.assertEqual(nfe, [14])
            self.assertEqual(coeff, 0.2)

    def test_non_best_does_not_inherit_selection_record_inference(self):
        cfg = OmegaConf.create(
            {
                "eval": {
                    "use_ema": False,
                    "denoise_steps": 3,
                    "denoise_timesteps_list": [3, 5],
                },
                "env_runner": {"temporal_ensemble_coeff": 0.6},
            }
        )
        _, use_ema, nfe, coeff, record = _resolve_final_eval_request(
            cfg, Path("/unused"), "latest", []
        )
        self.assertFalse(use_ema)
        self.assertEqual(nfe, [3, 5])
        self.assertEqual(coeff, 0.6)
        self.assertIsNone(record)

    def test_heldout_seed_selection_is_disjoint_caps_and_errors_at_zero(self):
        runner = SimpleNamespace(get_seed_list=lambda: [0, 1, 2, 3, 4])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            selected = _select_eval_seeds(
                runner, eval_seed=7, episodes=10, excluded_seeds=[1, 3]
            )
        self.assertEqual(set(selected), {0, 2, 4})
        self.assertTrue(set(selected).isdisjoint({1, 3}))
        self.assertIn("only 3 disjoint held-out seeds remain", output.getvalue())

        with self.assertRaisesRegex(RuntimeError, "No evaluation seeds remain"):
            _select_eval_seeds(
                runner, eval_seed=7, episodes=1, excluded_seeds=[0, 1, 2, 3, 4]
            )

    def test_result_details_include_selection_and_inference_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)

            class FakeRunner:
                def run(self, *args, **kwargs):
                    return {
                        "episode_details": [
                            {"seed": 2, "success": True, "steps": 9},
                            {"seed": 4, "success": False, "steps": 12},
                        ]
                    }

            info = _run_one_timestep(
                object(),
                FakeRunner(),
                [2, 4],
                8,
                None,
                result_save_dir=exp_dir / "eval_dexsim",
                ckpt_tag_or_path="best",
                ckpt_path=exp_dir / "checkpoints" / "selected.pt",
                eval_seed=17,
                selection_seeds_excluded=[1, 3],
                heldout_from_selection=True,
                use_ema=True,
                temporal_ensemble_coeff=None,
            )
            details = json.loads(
                (exp_dir / "eval_dexsim" / "result_details.json").read_text()
            )

        self.assertEqual(details["evaluation_seeds"], [2, 4])
        self.assertEqual(details["selection_seeds_excluded"], [1, 3])
        self.assertTrue(details["heldout_from_selection"])
        self.assertTrue(details["use_ema"])
        self.assertEqual(details["denoise_steps"], 8)
        self.assertIsNone(details["temporal_ensemble_coeff"])
        self.assertEqual(info["n_total"], 2)

    def test_eval_main_passes_resolved_coefficient_to_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "experiments" / "dp3" / "pour" / "test-exp"
            checkpoint = exp_dir / "checkpoints" / "selected.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            cfg = OmegaConf.create(
                {
                    "training": {"device": "cpu", "seed": 42},
                    "env_runner": {"temporal_ensemble_coeff": 0.5},
                    "eval": {"use_ema": False, "denoise_steps": 4},
                }
            )
            OmegaConf.save(cfg, exp_dir / "config.yaml")
            (exp_dir / "best_ckpt.json").write_text(json.dumps(_valid_record()))
            argv = [
                "eval_best_ckpt.py",
                "--policy-name",
                "dp3",
                "--task-name",
                "pour",
                "--exp-name",
                "test-exp",
                "--no-videos",
            ]

            with (
                mock.patch.object(eval_best_ckpt, "ROOT_DIR", str(root)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(eval_best_ckpt, "evaluate_checkpoint_robotwin") as run,
            ):
                eval_best_ckpt.main()

        self.assertEqual(run.call_args.kwargs["temporal_ensemble_coeff"], 0.1)


class EvalVideoOutputContractTests(unittest.TestCase):
    def _config(self):
        return OmegaConf.create(
            {
                "n_obs_steps": 2,
                "n_action_steps": 8,
                "horizon": 16,
                "action_key": "action",
                "training": {"device": "cpu", "seed": 42},
                "env_runner": {"env_kwargs": {"control_mode": "joint"}},
            }
        )

    def _runner(self):
        class FakeRunner:
            def __init__(self):
                self.runners = {
                    "first": SimpleNamespace(),
                    "second": SimpleNamespace(),
                }
                self.calls = []

            def get_seed_list(self):
                return [2]

            def run(self, *args, **kwargs):
                self.calls.append(kwargs)
                return {"episode_details": [{"seed": 2, "success": True, "steps": 9}]}

        return FakeRunner()

    def _agent(self):
        class FakeAgent:
            action_decoder = SimpleNamespace(solver=None)

            def to(self, device):
                return self

            def eval(self):
                return self

        return FakeAgent()

    def _patch_eval_setup(self, agent, runner, ckpt_path):
        return (
            mock.patch.object(
                eval_best_ckpt,
                "build_eval_components",
                return_value=(agent, runner, object()),
            ),
            mock.patch.object(
                eval_best_ckpt,
                "resolve_checkpoint_path",
                return_value=(ckpt_path, "latest"),
            ),
            mock.patch.object(eval_best_ckpt, "load_ckpt_for_inference"),
        )

    def test_single_video_separates_result_and_video_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self._runner()
            result_dir = root / "eval_dexsim"
            video_dir = result_dir / "timestamp"
            with contextlib.ExitStack() as stack:
                for patch in self._patch_eval_setup(
                    self._agent(), runner, root / "checkpoints" / "latest.pt"
                ):
                    stack.enter_context(patch)
                eval_best_ckpt.evaluate_checkpoint_robotwin(
                    root,
                    self._config(),
                    ckpt_tag_or_path="latest",
                    episodes=1,
                    denoise_steps=4,
                    use_ema=False,
                    video_save_dir=video_dir,
                    result_save_dir=result_dir,
                )

            self.assertEqual(runner.calls[0]["video_save_dir"], video_dir)
            self.assertTrue(all(child.record_video for child in runner.runners.values()))
            self.assertTrue((result_dir / "_result.txt").is_file())
            self.assertFalse((video_dir / "_result.txt").exists())

    def test_single_no_video_keeps_runner_video_path_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self._runner()
            result_dir = root / "eval_dexsim"
            cfg = self._config()
            cfg.env_runner.temporal_ensemble_coeff = 0.03
            with contextlib.ExitStack() as stack:
                for patch in self._patch_eval_setup(
                    self._agent(), runner, root / "checkpoints" / "latest.pt"
                ):
                    stack.enter_context(patch)
                eval_best_ckpt.evaluate_checkpoint_robotwin(
                    root,
                    cfg,
                    ckpt_tag_or_path="latest",
                    episodes=1,
                    denoise_steps=4,
                    use_ema=False,
                    video_save_dir=None,
                    result_save_dir=result_dir,
                )

            self.assertIsNone(runner.calls[0]["video_save_dir"])
            self.assertTrue(
                all(not child.record_video for child in runner.runners.values())
            )
            self.assertTrue((result_dir / "_result.txt").is_file())
            details = json.loads((result_dir / "result_details.json").read_text())
            self.assertEqual(details["temporal_ensemble_coeff"], 0.03)

    def test_sweep_video_shares_per_timestep_result_and_video_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self._runner()
            result_dir = root / "eval_dexsim" / "timestamp"
            with contextlib.ExitStack() as stack:
                for patch in self._patch_eval_setup(
                    self._agent(), runner, root / "checkpoints" / "latest.pt"
                ):
                    stack.enter_context(patch)
                eval_best_ckpt.evaluate_checkpoint_sweep(
                    root,
                    self._config(),
                    ckpt_tag_or_path="latest",
                    episodes=1,
                    denoise_timesteps_list=[2, 4],
                    use_ema=False,
                    video_save_dir=result_dir,
                    result_save_dir=result_dir,
                )

            self.assertEqual(
                [call["video_save_dir"] for call in runner.calls],
                [
                    result_dir / "denoise_timesteps2",
                    result_dir / "denoise_timesteps4",
                ],
            )
            self.assertTrue(all(child.record_video for child in runner.runners.values()))
            self.assertTrue((result_dir / "denoise_timesteps2" / "_result.txt").is_file())
            self.assertTrue((result_dir / "denoise_timesteps4" / "_result.txt").is_file())
            self.assertTrue((result_dir / "eval_summary.json").is_file())

    def test_sweep_no_video_keeps_result_root_out_of_runner_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self._runner()
            result_dir = root / "eval_dexsim" / "timestamp"
            with contextlib.ExitStack() as stack:
                for patch in self._patch_eval_setup(
                    self._agent(), runner, root / "checkpoints" / "latest.pt"
                ):
                    stack.enter_context(patch)
                eval_best_ckpt.evaluate_checkpoint_sweep(
                    root,
                    self._config(),
                    ckpt_tag_or_path="latest",
                    episodes=1,
                    denoise_timesteps_list=[2, 4],
                    use_ema=False,
                    video_save_dir=None,
                    result_save_dir=result_dir,
                )

            self.assertEqual(
                [call["video_save_dir"] for call in runner.calls], [None, None]
            )
            self.assertTrue(
                all(not child.record_video for child in runner.runners.values())
            )
            self.assertTrue((result_dir / "denoise_timesteps2" / "_result.txt").is_file())
            self.assertTrue((result_dir / "denoise_timesteps4" / "_result.txt").is_file())
            self.assertTrue((result_dir / "eval_summary.json").is_file())

    def test_sweep_creates_no_video_result_root_before_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "eval_dexsim" / "timestamp"
            runner = self._runner()

            def build_components(cfg, device):
                self.assertTrue(result_dir.is_dir())
                return self._agent(), runner, object()

            with (
                mock.patch.object(
                    eval_best_ckpt,
                    "build_eval_components",
                    side_effect=build_components,
                ),
                mock.patch.object(
                    eval_best_ckpt,
                    "resolve_checkpoint_path",
                    return_value=(root / "checkpoints" / "latest.pt", "latest"),
                ),
                mock.patch.object(eval_best_ckpt, "load_ckpt_for_inference"),
            ):
                eval_best_ckpt.evaluate_checkpoint_sweep(
                    root,
                    self._config(),
                    ckpt_tag_or_path="latest",
                    episodes=1,
                    denoise_timesteps_list=[2, 4],
                    use_ema=False,
                    video_save_dir=None,
                    result_save_dir=result_dir,
                )


class DemoInferenceContractTests(unittest.TestCase):
    def _config(self):
        return OmegaConf.create(
            {
                "eval": {
                    "use_ema": False,
                    "denoise_steps": 10,
                    "denoise_timesteps_list": None,
                },
                "env_runner": {"temporal_ensemble_coeff": 0.5},
            }
        )

    def _write_record(self, exp_dir: Path):
        checkpoint = exp_dir / "checkpoints" / "selected.pt"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        record = _valid_record()
        record["inference"].update(
            {
                "use_ema": True,
                "denoise_steps": 4,
                "temporal_ensemble_coeff": 0.02,
            }
        )
        (exp_dir / "best_ckpt.json").write_text(json.dumps(record))
        return record

    def test_best_uses_recorded_inference_without_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            self._write_record(exp_dir)

            use_ema, nfe, coeff = record_demo._resolve_demo_inference(
                self._config(),
                exp_dir,
                "best",
                cli_use_ema=None,
                cli_denoise_steps=None,
            )

        self.assertTrue(use_ema)
        self.assertEqual(nfe, [4])
        self.assertEqual(coeff, 0.02)

    def test_best_cli_overrides_ema_and_denoise_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            self._write_record(exp_dir)

            use_ema, nfe, coeff = record_demo._resolve_demo_inference(
                self._config(),
                exp_dir,
                "best",
                cli_use_ema=False,
                cli_denoise_steps=6,
            )

        self.assertFalse(use_ema)
        self.assertEqual(nfe, [6])
        self.assertEqual(coeff, 0.02)

    def test_non_best_retains_demo_config_resolution_and_sweep(self):
        cfg = self._config()
        cfg.eval.demo = {
            "use_ema": True,
            "denoise_timesteps_list": [6, 10],
        }

        use_ema, nfe, coeff = record_demo._resolve_demo_inference(
            cfg,
            Path("/unused"),
            "latest",
            cli_use_ema=None,
            cli_denoise_steps=None,
        )

        self.assertTrue(use_ema)
        self.assertEqual(nfe, [6, 10])
        self.assertEqual(coeff, 0.5)

    def test_main_injects_best_coefficient_before_component_build(self):
        class BuildStopped(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "experiments" / "dp3" / "pour" / "test-exp"
            exp_dir.mkdir(parents=True)
            cfg = OmegaConf.create(
                {
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "horizon": 16,
                    "action_key": "action",
                    "training": {"device": "cpu", "seed": 42},
                    "env_runner": {
                        "env_kwargs": {"control_mode": "joint"},
                        "temporal_ensemble_coeff": 0.5,
                    },
                    "eval": {"use_ema": False, "denoise_steps": 10},
                }
            )
            OmegaConf.save(cfg, exp_dir / "config.yaml")
            self._write_record(exp_dir)

            def stop_at_build(resolved_cfg, device):
                self.assertEqual(device, torch.device("cpu"))
                self.assertEqual(
                    resolved_cfg.env_runner.temporal_ensemble_coeff, 0.02
                )
                raise BuildStopped

            argv = [
                "record_demo.py",
                "--policy-name",
                "dp3",
                "--task-name",
                "pour",
                "--exp-name",
                "test-exp",
            ]
            with (
                mock.patch.object(record_demo, "ROOT_DIR", str(root)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(record_demo, "set_seed"),
                mock.patch.object(
                    record_demo,
                    "build_eval_components",
                    side_effect=stop_at_build,
                ) as build,
                self.assertRaises(BuildStopped),
            ):
                record_demo.main()

        build.assert_called_once()

    def test_demo_configures_every_leaf_runner_for_viewer_capture(self):
        class FakeAgent:
            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeRunner:
            def __init__(self):
                self.runners = {
                    "first": SimpleNamespace(),
                    "second": SimpleNamespace(),
                }

            def get_seed_list(self):
                return [2]

            def run(self, *args, **kwargs):
                return {"episode_details": [{"seed": 2, "success": True, "steps": 9}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "experiments" / "dp3" / "pour" / "test-exp"
            exp_dir.mkdir(parents=True)
            cfg = OmegaConf.create(
                {
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "horizon": 16,
                    "action_key": "action",
                    "training": {"device": "cpu", "seed": 42},
                    "env_runner": {"env_kwargs": {"control_mode": "joint"}},
                    "eval": {"use_ema": False, "denoise_steps": 4},
                }
            )
            OmegaConf.save(cfg, exp_dir / "config.yaml")
            runner = FakeRunner()
            argv = [
                "record_demo.py",
                "--policy-name",
                "dp3",
                "--task-name",
                "pour",
                "--exp-name",
                "test-exp",
                "--ckpt-tag",
                "latest",
                "--episodes",
                "1",
                "--resolution",
                "640",
                "480",
                "--fps",
                "24",
            ]
            with (
                mock.patch.object(record_demo, "ROOT_DIR", str(root)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(record_demo, "set_seed"),
                mock.patch.object(
                    record_demo,
                    "build_eval_components",
                    return_value=(FakeAgent(), runner, object()),
                ),
                mock.patch.object(
                    record_demo,
                    "resolve_checkpoint_path",
                    return_value=(root / "checkpoints" / "latest.pt", "latest"),
                ),
                mock.patch.object(record_demo, "load_ckpt_for_inference"),
            ):
                record_demo.main()

        for child in runner.runners.values():
            self.assertEqual(child.render_mode, "human")
            self.assertTrue(child.record_video)
            self.assertEqual(child.viewer_resolution, (640, 480))
            self.assertEqual(child.env_video_fps, 24)

    def test_demo_defaults_to_1280x960_without_resolution_override(self):
        class FakeAgent:
            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeRunner:
            def __init__(self):
                self.runners = {
                    "first": SimpleNamespace(),
                    "second": SimpleNamespace(),
                }

            def get_seed_list(self):
                return [2]

            def run(self, *args, **kwargs):
                return {"episode_details": [{"seed": 2, "success": True, "steps": 9}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "experiments" / "dp3" / "pour" / "test-exp"
            exp_dir.mkdir(parents=True)
            cfg = OmegaConf.create(
                {
                    "n_obs_steps": 2,
                    "n_action_steps": 8,
                    "horizon": 16,
                    "action_key": "action",
                    "training": {"device": "cpu", "seed": 42},
                    "env_runner": {"env_kwargs": {"control_mode": "joint"}},
                    "eval": {"use_ema": False, "denoise_steps": 4},
                }
            )
            OmegaConf.save(cfg, exp_dir / "config.yaml")
            runner = FakeRunner()
            argv = [
                "record_demo.py",
                "--policy-name",
                "dp3",
                "--task-name",
                "pour",
                "--exp-name",
                "test-exp",
                "--ckpt-tag",
                "latest",
                "--episodes",
                "1",
            ]
            with (
                mock.patch.object(record_demo, "ROOT_DIR", str(root)),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(record_demo, "set_seed"),
                mock.patch.object(
                    record_demo,
                    "build_eval_components",
                    return_value=(FakeAgent(), runner, object()),
                ),
                mock.patch.object(
                    record_demo,
                    "resolve_checkpoint_path",
                    return_value=(root / "checkpoints" / "latest.pt", "latest"),
                ),
                mock.patch.object(record_demo, "load_ckpt_for_inference"),
            ):
                record_demo.main()

        for child in runner.runners.values():
            self.assertEqual(child.viewer_resolution, (1280, 960))


if __name__ == "__main__":
    unittest.main()
