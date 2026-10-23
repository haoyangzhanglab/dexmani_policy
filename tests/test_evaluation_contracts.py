import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.experiments import MilestoneCheckpoint, resolve_selection_record
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.env_runner.common import video
from dexmani_policy.env_runner.common.metrics import summarize_episodes
from dexmani_policy.evaluation.artifacts import validate_selection_evidence
from dexmani_policy.evaluation.config import (
    parse_evaluation_overrides,
    resolve_evaluation_request,
)
from dexmani_policy.evaluation.selection import CheckpointEvaluation, select_best_checkpoint


class EvaluationRequestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "latest.pt").touch()
        checkpoint = checkpoints / "epoch=1-step=8-milestone=50pct.pt"
        checkpoint.touch()
        self.cfg = OmegaConf.create({
            "agent": {"horizon": 6, "n_obs_steps": 2, "n_action_steps": 3},
            "eval": {
                "seed": 71,
                "inference_steps_list": [20, 30],
                "use_ema": False,
                "offline": {
                    "episodes": 4, "inference_steps": 10, "use_ema": True, "action_steps": 3,
                },
            },
        })
        self.record = self.directory / "selection_result.json"
        self.record.write_text(json.dumps({
            "status": "success",
            "ckpt_relpath": str(checkpoint.relative_to(self.directory)),
            "inference": {
                "use_ema": False, "inference_steps": 6, "action_steps": 2,
                "policy_seed_mode": "episode_seed",
            },
            "evaluation": {
                "effective_config": {
                    "agent": OmegaConf.to_container(self.cfg.agent),
                    "eval": {"offline": {"episodes": 2}},
                },
                "request": {"shuffle_seed": 91},
            },
        }))
        (self.directory / "best_ckpt.json").write_text(
            json.dumps({"selection_result": self.record.name})
        )

    def test_inference_parameter_precedence(self):
        cases = [
            ("latest", [], {}, (True, [10], 3)),
            ("best", [], {}, (False, [6], 2)),
            ("best", ["eval.use_ema=true", "eval.inference_steps_list=[4,8]",
                      "eval.action_steps=1"], {}, (True, [4, 8], 1)),
            ("latest", ["eval.inference_steps_list=[4,8]", "eval.offline.inference_steps=5",
                        "eval.action_steps=1", "eval.offline.action_steps=2"],
             {}, (True, [5], 2)),
            ("latest", ["eval.offline.inference_steps=5",
                        "eval.offline.inference_steps_list=[3,9]"],
             {}, (True, [3, 9], 3)),
            ("latest", ["eval.use_ema=false", "eval.inference_steps=5", "eval.action_steps=1"],
             {"use_ema": True, "inference_steps_list": [7], "action_steps": 4},
             (True, [7], 4)),
            ("latest", ["eval.use_ema=null", "eval.offline.inference_steps=null",
                        "eval.action_steps=null"], {}, (True, [10], 3)),
        ]
        for checkpoint, overrides, arguments, expected in cases:
            with self.subTest(checkpoint=checkpoint, overrides=overrides, arguments=arguments):
                request = resolve_evaluation_request(
                    self.cfg, self.directory, checkpoint,
                    parse_evaluation_overrides(overrides), **arguments,
                )
                self.assertEqual(
                    (request.use_ema, request.inference_steps, request.action_steps), expected
                )
                self.assertEqual(request.config.eval.seed, 91 if checkpoint == "best" else 71)
                self.assertEqual(
                    request.config.eval.offline.episodes, 2 if checkpoint == "best" else 4
                )

    def test_invalid_nfe_lists(self):
        for steps in ([], [2, 2], [True], [0], [2.0]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                resolve_evaluation_request(
                    self.cfg, self.directory, "latest", inference_steps_list=steps
                )

    def test_action_window_is_validated_when_resolving_request(self):
        for steps in (0, True, 2.0, 6):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                resolve_evaluation_request(self.cfg, self.directory, "latest", action_steps=steps)

    def test_selection_checkpoint_stays_immutable_and_inside_experiment(self):
        record = json.loads(self.record.read_text())
        target = self.directory / record["ckpt_relpath"]
        alias = self.directory / "checkpoints/alias.pt"
        alias.symlink_to(target)
        for path in ("checkpoints/latest.pt", "checkpoints/alias.pt", "../outside.pt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                record["ckpt_relpath"] = path
                self.record.write_text(json.dumps(record))
                resolve_selection_record(self.directory, self.record)

    def test_selection_uses_only_the_reserved_optional_tie_batch(self):
        (self.directory / "checkpoints/epoch=2-step=16-milestone=100pct.pt").touch()
        cfg = OmegaConf.create(self.cfg)
        cfg.training = {"device": "cpu"}
        cfg.eval.inference_steps = 8
        cfg.eval.offline.episodes = 2
        pools = {"task": list(range(100))}

        def snapshot(config, runners, **request):
            return {"effective_config": OmegaConf.to_container(config), "request": request}

        def restore(config, path, **kwargs):
            return SimpleNamespace(checkpoint_step=8 if "step=8-" in path.name else 16)

        for extra_count in (0, 1):
            with self.subTest(extra_count=extra_count):
                cfg.eval.select_best = {
                    "initial_episodes": 1, "batch_size": extra_count, "max_episodes": 2,
                }
                runner = Mock(task_runners={"task": SimpleNamespace(
                    get_seed_list=lambda: pools["task"],
                )})

                def run(agent, *, task_seeds, **kwargs):
                    success = runner.run.call_count <= 2 or agent.checkpoint_step == 8
                    return {"episode_details": [
                        {"task_name": task, "seed": seed, "success": success,
                         "steps": 10 if success else None, "total_steps": 20}
                        for task, seeds in task_seeds.items() for seed in seeds
                    ]}

                runner.run.side_effect = run
                with patch("dexmani_policy.evaluation.selection.build_evaluation_runner", return_value=runner), \
                     patch("dexmani_policy.evaluation.selection.restore_policy_agent", side_effect=restore), \
                     patch("dexmani_policy.evaluation.selection.evaluation_snapshot", side_effect=snapshot), \
                     patch("dexmani_policy.evaluation.selection.cprint"):
                    path = select_best_checkpoint(self.directory, cfg)
                record = json.loads(path.read_text())
                self.assertEqual(record["global_step"], 8 if extra_count else 16)
                self.assertEqual(runner.run.call_count, 4 if extra_count else 2)
                test_plan = validate_selection_evidence(record, pools, episodes=2)
                self.assertEqual(test_plan, record["evaluation"]["request"]["test_task_seeds"])

    def test_missing_settings_require_an_explicit_source(self):
        cases = (
            (["eval.inference_steps_list", "eval.offline.inference_steps"],
             "inference_steps", {"inference_steps_list": [5]}),
            (["eval.use_ema", "eval.offline.use_ema"], "use_ema", {"use_ema": False}),
            (["eval.offline.episodes"], "episodes", {"episodes": 2}),
        )
        for paths, setting, override in cases:
            with self.subTest(setting=setting):
                cfg = OmegaConf.create(self.cfg)
                for path in paths:
                    parent, key = path.rsplit(".", 1)
                    del OmegaConf.select(cfg, parent)[key]
                with self.assertRaisesRegex(ValueError, setting):
                    resolve_evaluation_request(cfg, self.directory, "latest")
                resolve_evaluation_request(cfg, self.directory, "latest", **override)
                resolve_evaluation_request(cfg, self.directory, "best")

    def test_selection_does_not_fill_missing_budgets(self):
        runner = Mock(task_runners={"task": Mock(get_seed_list=lambda: list(range(100)))})
        for field in ("initial_episodes", "batch_size", "max_episodes", "episodes"):
            with self.subTest(field=field):
                cfg = OmegaConf.create(self.cfg)
                cfg.eval.inference_steps = 8
                cfg.eval.select_best = {
                    "initial_episodes": 2, "batch_size": 1, "max_episodes": 3,
                }
                section = "offline" if field == "episodes" else "select_best"
                del cfg.eval[section][field]
                with patch(
                    "dexmani_policy.evaluation.selection.build_evaluation_runner",
                    return_value=runner,
                ), patch("dexmani_policy.evaluation.selection.restore_policy_agent") as restore:
                    with self.assertRaisesRegex(ValueError, field):
                        select_best_checkpoint(self.directory, cfg)
                    restore.assert_not_called()

    def test_selection_record_rejects_conflicting_requests(self):
        for arguments in (
            {"checkpoint": "latest"}, {"use_ema": True}, {"inference_steps_list": [7]},
            {"action_steps": 3}, {"eval_seed": 90}, {"episodes": 1}, {"episodes": 2.0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resolve_evaluation_request(
                    self.cfg, self.directory, selection_record=self.record, **arguments
                )
        request = resolve_evaluation_request(
            self.cfg, self.directory, selection_record=self.record, episodes=2, eval_seed=91
        )
        self.assertEqual(request.checkpoint.name, "epoch=1-step=8-milestone=50pct.pt")

    def test_dotlist_preserves_saved_inputs_and_seed_protocol(self):
        for overrides in (
            ["eval.seed=1", "eval.seed=2"], ["agent.horizon=8"],
            ["env_runner.task_configs=[{task_name:test,sensor_modalities:[rgb]}]"],
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                parse_evaluation_overrides(overrides)
        for overrides in (["eval.seed=90"], ["eval.offline.episodes=3"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_evaluation_request(
                    self.cfg, self.directory, "best", parse_evaluation_overrides(overrides)
                )


class CheckpointStatisticsTest(unittest.TestCase):
    def test_batch_statistics_keep_episode_weighting_and_success_steps(self):
        candidate = CheckpointEvaluation(MilestoneCheckpoint(Path("checkpoint.pt"), 50, 8))
        batches = (
            ([(False, None), (False, 10)], 2, 0, None),
            ([(True, None)], 3, 1 / 3, None),
            ([(True, 0), (True, 6), (False, 1)], 6, 1 / 2, 3),
            ([(True, 12)], 7, 4 / 7, 6),
        )
        for episodes, count, rate, steps in batches:
            with self.subTest(episodes=episodes):
                candidate.add_episodes([
                    {"success": success, "steps": elapsed} for success, elapsed in episodes
                ])
                self.assertEqual(candidate.n_episodes, count)
                self.assertEqual(candidate.success_rate, rate)
                self.assertEqual(candidate.avg_steps, steps)
                self.assertEqual(candidate.rank, (
                    rate, -(steps if steps is not None else float("inf")), 8,
                ))


class ObservationWindowTest(unittest.TestCase):
    def test_episode_summary_preserves_success_and_step_statistics(self):
        details = [
            {"success": True, "steps": 1, "total_steps": 5},
            {"success": False, "steps": None, "total_steps": 8},
            {"success": True, "steps": 4, "total_steps": 10},
        ]
        self.assertEqual(summarize_episodes(details), {
            "success_rate": 2 / 3, "avg_steps": 2, "avg_steps_all": 8,
        })
        self.assertIsNone(summarize_episodes([details[1]])["avg_steps"])

    def test_rollout_requires_simulator_result_fields(self):
        for missing in (
            None, "success", "success_condition", "action_cnt", "executed_joint_action",
        ):
            with self.subTest(missing=missing):
                runner = BaseRunner(2, ["joint_state"])
                observation = {"joint_state": np.zeros(2, dtype=np.float32)}
                agent = Mock(
                    history_steps=1, hand_dim=2, horizon=3, n_obs_steps=2,
                    action_dim=2, control_action_dim=2,
                )
                agent.parameters.return_value = iter([torch.zeros(1)])
                agent.predict_action.return_value = {"pred_action": torch.zeros(1, 3, 2)}
                env = SimpleNamespace(action_cnt=0, reset=lambda **kwargs: (observation, {}))
                completed_steps = []

                def step(action):
                    completed_steps.append(action)
                    env.action_cnt = len(completed_steps)
                    terminal = env.action_cnt == 2
                    info = {
                        "success": terminal,
                        "success_condition": True,
                        "executed_joint_action": np.ones(2, dtype=np.float32),
                    }
                    if missing == "action_cnt":
                        del env.action_cnt
                    elif missing is not None:
                        del info[missing]
                    return observation, 0, terminal, False, info

                env.step = step
                if missing is None:
                    self.assertEqual(runner.run_episode(agent, env, 0, action_steps=1), (True, 1))
                    self.assertEqual(env.action_cnt, 2)
                    np.testing.assert_array_equal(runner.hand_history, [[1, 1]])
                else:
                    error = AttributeError if missing == "action_cnt" else KeyError
                    with self.assertRaises(error):
                        runner.run_episode(agent, env, 0, action_steps=1)

    def test_padding_rollover_and_storage_isolation(self):
        runner = BaseRunner(3, ["joint_state", "eef_pose"])
        for value in range(1, 6):
            numpy_frame = np.array([value], dtype=np.float32)
            torch_frame = torch.tensor([value], dtype=torch.float32)
            runner.update_obs({"joint_state": numpy_frame, "eef_pose": torch_frame})
            numpy_frame.fill(-1)
            torch_frame.fill_(-1)
            expected = [[max(1, index)] for index in range(value - 2, value + 1)]
            window = runner.get_stacked_obs()
            np.testing.assert_array_equal(window["joint_state"], expected)
            expected_tensor = torch.tensor(expected, dtype=torch.float32)
            torch.testing.assert_close(window["eef_pose"], expected_tensor)
            window["joint_state"].fill(-2)
            window["eef_pose"].fill_(-2)
            fresh = runner.get_stacked_obs()
            np.testing.assert_array_equal(fresh["joint_state"], expected)
            torch.testing.assert_close(fresh["eef_pose"], expected_tensor)

        runner.reset()
        with self.assertRaisesRegex(RuntimeError, "No observation"):
            runner.get_stacked_obs()
        runner.update_obs({"joint_state": np.array([9.0]), "eef_pose": torch.tensor([9.0])})
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"], [[9], [9], [9]])

    def test_invalid_frame_cannot_partially_update_history(self):
        runner = BaseRunner(2, ["joint_state", "eef_pose"])
        frame = {"joint_state": np.array([1.0]), "eef_pose": torch.tensor([2.0])}
        runner.update_obs(frame)
        for invalid in (torch.ones(2), torch.ones(1, dtype=torch.int64), np.ones(1)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    runner.update_obs({"joint_state": np.array([99.0]), "eef_pose": invalid})
                self.assertEqual(runner.obs_cursor, 1)
                np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"], [[1], [1]])
                torch.testing.assert_close(
                    runner.get_stacked_obs()["eef_pose"], torch.tensor([[2.0], [2.0]])
                )

    def test_action_chunk_uses_future_control_dimensions(self):
        runner = BaseRunner(2, ["joint_state"])
        agent = Mock(history_steps=0, horizon=6, n_obs_steps=2, action_dim=4, control_action_dim=2)
        trajectory = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        agent.predict_action.return_value = {"pred_action": trajectory}
        actions = runner.predict_action_chunk({}, agent, inference_steps=3, action_steps=2)
        np.testing.assert_array_equal(actions, [[4, 5], [8, 9]])
        self.assertTrue(actions.flags.c_contiguous)


class VideoEncodingTest(unittest.TestCase):
    def setUp(self):
        executable = patch.object(video, "get_ffmpeg_exe", return_value="ffmpeg")
        executable.start()
        self.addCleanup(executable.stop)

    def test_buffer_writes_match_rgb_bytes(self):
        values = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)
        for frames in (values.astype(np.uint8), values.astype(np.float32), values[:, ::-1, ::2]):
            with self.subTest(dtype=frames.dtype, strides=frames.strides):
                stream = io.BytesIO()
                process = Mock()
                process.stdin.write.side_effect = stream.write
                process.wait.return_value = 0
                process.poll.return_value = 0
                with patch.object(video.subprocess, "Popen", return_value=process):
                    video.encode_video(Path("unused.mp4"), frames, 15)
                expected = b"".join(
                    frame.astype(np.uint8, copy=False).tobytes() for frame in frames
                )
                self.assertEqual(stream.getvalue(), expected)
                self.assertTrue(all(
                    isinstance(args[0], memoryview)
                    for args, _ in process.stdin.write.call_args_list
                ))
                process.terminate.assert_not_called()

    def test_broken_pipe_reaps_encoder(self):
        process = Mock()
        process.stdin.write.side_effect = BrokenPipeError("encoder closed input")
        process.poll.return_value = None
        with patch.object(video.subprocess, "Popen", return_value=process):
            with self.assertRaises(BrokenPipeError):
                video.encode_video(Path("unused.mp4"), np.zeros((1, 2, 2, 3), dtype=np.uint8), 15)
        process.stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_encoder_timeout_terminates_then_kills(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("ffmpeg", 60), subprocess.TimeoutExpired("ffmpeg", 5), -9,
        ]
        with patch.object(video.subprocess, "Popen", return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                video.encode_video(Path("unused.mp4"), np.zeros((1, 2, 2, 3), dtype=np.uint8), 15)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_args_list, [call(timeout=60), call(timeout=5), call()])

    def test_failed_export_does_not_register_video(self):
        env = Mock()
        env.get_video.return_value = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(video, "encode_video", side_effect=RuntimeError("encoder failed")):
                with patch.object(video, "cprint"):
                    self.assertIsNone(video.save_episode_video(env, directory, "episode", 15))


if __name__ == "__main__":
    unittest.main()
