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

from dexmani_policy.common.checkpoint_io import CheckpointStore
from dexmani_policy.common.config import validate_action_key_consistency
from dexmani_policy.datasets.base_dataset import preprocess_validation_rgb
from dexmani_policy.env_runner.base_runner import BaseRunner, EvalEpisodeError
from dexmani_policy.eval_best_ckpt import (
    _resolve_final_eval_request,
    _run_one_timestep,
    _select_eval_seeds,
)
from dexmani_policy.select_best_ckpt import select_best_checkpoint
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


class _BlenderAgent:
    control_action_dim = 19
    n_action_steps = 8

    def predict_action(self, **kwargs):
        return {
            "pred_action": torch.arange(16 * 28, dtype=torch.float32).reshape(1, 16, 28)
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

    def test_blender_receives_control_action_dimensions_only(self):
        runner = BaseRunner(
            n_obs_steps=2,
            default_eval_episodes=1,
            temporal_ensemble_coeff=0.01,
        )

        action_chunk = runner.get_action_chunk({}, _BlenderAgent())

        self.assertEqual(action_chunk.shape, (8, 19))
        self.assertEqual(runner._blender._prev_tail.shape[-1], 19)

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
                exp_dir=exp_dir,
                ckpt_tag_or_path="best",
                ckpt_path=exp_dir / "checkpoints" / "selected.pt",
                ckpt_label="best",
                eval_seed=17,
                selection_seeds_excluded=[1, 3],
                heldout_from_selection=True,
                use_ema=True,
            )
            details = json.loads(
                (exp_dir / "eval_dexsim" / "result_details.json").read_text()
            )

        self.assertEqual(details["evaluation_seeds"], [2, 4])
        self.assertEqual(details["selection_seeds_excluded"], [1, 3])
        self.assertTrue(details["heldout_from_selection"])
        self.assertTrue(details["use_ema"])
        self.assertEqual(details["denoise_steps"], 8)
        self.assertEqual(info["n_total"], 2)


if __name__ == "__main__":
    unittest.main()
