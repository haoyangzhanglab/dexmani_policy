"""CPU rollout, seed provenance and result-format regression checks."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_policy.env_runner.base_runner import BaseRunner, EvaluationError
from dexmani_policy.env_runner.common.metrics import compute_eval_stats, validate_episode_details
from dexmani_policy.env_runner.common.seeds import select_eval_seeds
from dexmani_policy.env_runner.common.video import encode_video, save_episode_video
from dexmani_policy.env_runner.evaluation import (
    build_eval_runner,
    evaluate_checkpoint,
    evaluation_snapshot,
    get_eval_param,
    parse_eval_overrides,
    resolve_eval_request,
    run_eval_plan,
    save_eval_snapshot,
    selection_task_seeds,
    task_seed_pools,
)
from dexmani_policy.env_runner.multi_task_sim_runner import MultiTaskSimRunner
from dexmani_policy.env_runner.selection import select_best_checkpoint
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.training.common.checkpoint import (
    resolve_best_checkpoint,
    resolve_selection_record,
)


class ToyEnv:
    video_fps = 15

    def __init__(self, *, fail=False):
        self.fail = fail
        self.closed = 0
        self.seeds = []
        self.action_cnt = 0
        self.observation = np.zeros(19, dtype=np.float32)
        self.executed = []

    def reset(self, seed):
        self.seed = seed
        self.seeds.append(seed)
        self.action_cnt = 0
        return {"joint_state": self.observation.copy()}, {}

    def step(self, action):
        if self.fail:
            raise RuntimeError("step failed")
        self.action_cnt += 1
        executed = np.asarray(action, dtype=np.float32) + 100
        self.executed.append(executed.copy())
        success = self.seed % 2 == 0 and self.action_cnt >= 3
        return (
            {"joint_state": self.observation.copy()},
            0,
            success,
            self.action_cnt >= 4,
            {
                "success": success,
                "success_condition": self.action_cnt >= 2,
                "executed_joint_action": executed,
            },
        )

    def get_video(self):
        raise AssertionError("Video extraction is disabled")

    def close(self):
        self.closed += 1


class ToyAgent(torch.nn.Module):
    def __init__(self, history_steps=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.history_steps = history_steps
        self.hand_dim = 12
        self.action_dim = 19
        self.histories = []
        self.noise = []

    def predict_action(self, obs_dict, inference_steps=None, action_context=None):
        self.noise.append(torch.rand(()).item())
        if action_context is not None:
            self.histories.append(action_context["hand_history"].clone())
        return {"control_action": torch.arange(3 * 19).float().reshape(1, 3, 19)}


class ToyRunner(BaseRunner):
    def __init__(self, task="task", *, fail=False, seeds=None, clear_cache_freq=0):
        super().__init__(2, 2, sensor_modalities=["joint_state"], clear_cache_freq=clear_cache_freq)
        self.task_name = task
        self.task_text = task
        self.eval_seeds = [0, 1] if seeds is None else seeds
        self.environments = []
        self.fail = fail

    def make_env(self):
        env = ToyEnv(fail=self.fail)
        self.environments.append(env)
        return env

    def get_seed_list(self):
        return self.eval_seeds


def episode(seed, success=False, steps=None, task=None):
    result = {"seed": seed, "success": success, "steps": steps}
    if task is not None:
        result["task_name"] = task
    return result


class RunnerTests(unittest.TestCase):
    def test_observation_padding_wrap_and_independent_storage(self):
        runner = BaseRunner(3, 1, sensor_modalities=["array", "tensor", "text"])
        with self.assertRaises(RuntimeError):
            runner.get_stacked_obs()
        frames = []
        for index in range(7):
            array = np.full((4, 2), index, dtype=np.float32)
            frames.append(array.copy())
            runner.update_obs(
                {"array": array, "tensor": torch.from_numpy(array.copy()), "text": str(index)}
            )
            array[:] = -1
            expected = [frames[0]] * max(0, 3 - len(frames)) + frames[-3:]
            stacked = runner.get_stacked_obs()
            np.testing.assert_equal(stacked["array"], np.stack(expected))
            np.testing.assert_equal(stacked["tensor"], np.stack(expected))
            self.assertEqual(stacked["text"], [str(index)] * 3)
            stacked["array"][:] = -100
            stacked["tensor"][:] = -100
            np.testing.assert_equal(runner.get_stacked_obs()["array"], np.stack(expected))
        batch = runner.get_obs_batch("cpu")
        self.assertEqual(tuple(batch["array"].shape), (1, 3, 4, 2))
        self.assertEqual(batch["text"], ["6"] * 3)
        runner.reset()
        self.assertEqual(runner.obs_count, 0)

    def test_invalid_frame_does_not_partially_write(self):
        runner = BaseRunner(2, 1, sensor_modalities=["a", "b"])
        runner.update_obs({"a": np.ones(3), "b": np.ones(3)})
        baseline = runner.get_stacked_obs()
        for bad in (np.ones(1), np.ones(3, np.float32), "text", torch.ones(3)):
            with self.assertRaises((TypeError, ValueError)):
                runner.update_obs({"a": np.zeros(3), "b": bad})
            self.assertEqual(runner.obs_count, 1)
            for key in baseline:
                np.testing.assert_equal(runner.get_stacked_obs()[key], baseline[key])

    def test_rollout_metrics_rng_refresh_and_hand_commands(self):
        runner = ToyRunner(seeds=[0, 1, 2], clear_cache_freq=2)
        agent = ToyAgent(history_steps=2)
        result = runner.run(agent, inference_steps=1, eval_episodes=3)
        self.assertEqual(result["success_rate"], 2 / 3)
        self.assertEqual(result["avg_steps"], 2)
        self.assertEqual(result["avg_steps_all"], 3)
        self.assertEqual([d["total_steps"] for d in result["episode_details"]], [3, 4, 3])
        self.assertEqual([env.seeds for env in runner.environments], [[0, 1], [2]])
        self.assertEqual([env.closed for env in runner.environments], [1, 1])
        torch.testing.assert_close(agent.histories[0], torch.zeros(1, 2, 12))
        torch.testing.assert_close(agent.histories[1], torch.zeros(1, 2, 12))
        expected = torch.from_numpy(np.stack(runner.environments[0].executed[4:6]))[:, -12:]
        torch.testing.assert_close(agent.histories[2][0], expected)
        other = ToyAgent()
        ToyRunner(seeds=[0]).run(other, inference_steps=1, eval_episodes=1)
        self.assertEqual(agent.noise[0], other.noise[0])

    def test_errors_abort_and_close_environment(self):
        runner = ToyRunner(fail=True)
        with self.assertRaises(EvaluationError) as raised:
            runner.run(ToyAgent())
        self.assertEqual((raised.exception.task, raised.exception.seed), ("task", 0))
        self.assertIn("RuntimeError: step failed", str(raised.exception))
        self.assertEqual(runner.environments[0].closed, 1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        with patch.object(runner, "make_env", side_effect=ValueError("construction failed")):
            with self.assertRaises(EvaluationError) as raised:
                runner.run(ToyAgent())
            self.assertIsInstance(raised.exception.__cause__, ValueError)
            self.assertIn("ValueError: construction failed", str(raised.exception))
        with patch.object(runner, "run_one_episode", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(ToyAgent())
            self.assertEqual(runner.environments[-1].closed, 1)

    def test_explicit_plan_preserves_seed_pool_and_default_budget(self):
        runner = ToyRunner(seeds=list(range(8)))
        expected = select_eval_seeds(task_seed_pools(runner), 4, 4)
        plan = {"task": [4, 5, 6]}
        result = run_eval_plan(runner, ToyAgent(), plan, inference_steps=1)
        self.assertEqual([d["seed"] for d in result["episode_details"]], plan["task"])
        self.assertEqual(select_eval_seeds(task_seed_pools(runner), 4, 4), expected)
        self.assertEqual(runner.eval_seeds, list(range(8)))
        self.assertEqual(len(runner.run(ToyAgent())["episode_details"]), 2)

        runner.fail = True
        with self.assertRaises(EvaluationError):
            run_eval_plan(runner, ToyAgent(), plan, inference_steps=1)
        self.assertEqual(runner.eval_seeds, list(range(8)))
        self.assertEqual(plan, {"task": [4, 5, 6]})

    def test_multi_task_results_and_fail_fast(self):
        runner = MultiTaskSimRunner(
            [{"task_name": name, "task_text": name, "env_kwargs": {}} for name in ("a", "b")], 2, 2
        )
        runner.runners = {"a": ToyRunner("a"), "b": ToyRunner("b")}
        pools = task_seed_pools(runner)
        plan = {"a": [0, 2], "b": [1, 3]}
        result = run_eval_plan(runner, ToyAgent(), plan, inference_steps=1)
        self.assertEqual(result["macro_success_rate"], 0.5)
        self.assertEqual(result["micro_success_rate"], 0.5)
        self.assertEqual(result["n_valid_episodes"], 4)
        self.assertEqual(task_seed_pools(runner), pools)
        runner.runners = {"a": ToyRunner("a", fail=True), "b": ToyRunner("b")}
        with self.assertRaises(EvaluationError):
            run_eval_plan(runner, ToyAgent(), plan, inference_steps=1)
        self.assertEqual(runner.runners["b"].environments, [])
        self.assertEqual(task_seed_pools(runner), pools)
        with self.assertRaises(ValueError):
            run_eval_plan(runner, ToyAgent(), {"a": [0], "b": [1, 3]}, inference_steps=1)

    def test_video_failures_never_mask_rollout(self):
        env = Mock()
        self.assertIsNone(save_episode_video(env, None, "episode", 15))
        env.get_video.assert_not_called()
        env.get_video.return_value = np.zeros((2, 8, 8, 3), np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "dexmani_policy.env_runner.common.video.encode_video",
                side_effect=OSError("encoder failed"),
            ):
                self.assertIsNone(save_episode_video(env, temporary, "episode", 15))
            with patch("dexmani_policy.env_runner.common.video.encode_video"):
                self.assertEqual(
                    save_episode_video(env, temporary, "episode", 15),
                    {"episode": str(Path(temporary) / "episode.mp4")},
                )
        process = Mock()
        process.stdin.write.side_effect = BrokenPipeError
        process.poll.return_value = None
        with patch("dexmani_policy.env_runner.common.video.shutil.which", return_value="ffmpeg"):
            with patch(
                "dexmani_policy.env_runner.common.video.subprocess.Popen", return_value=process
            ):
                with self.assertRaises(BrokenPipeError):
                    encode_video(Path("unused.mp4"), env.get_video.return_value, 15)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_video_fps_tracks_current_environment_without_changing_defaults(self):
        for configured_fps in (None, 12):
            runner = ToyRunner(clear_cache_freq=1)
            runner.env_video_fps = configured_fps
            environments = [ToyEnv(), ToyEnv(), ToyEnv()]
            for env, fps in zip(environments, (20, 30, 40)):
                env.video_fps = fps
            with (
                patch.object(runner, "make_env", side_effect=environments),
                patch(
                    "dexmani_policy.env_runner.base_runner.save_episode_video", return_value=None
                ) as save_video,
            ):
                runner.run(ToyAgent())
                runner.run(ToyAgent(), seeds=[0])
            expected = [20, 30, 40] if configured_fps is None else [12, 12, 12]
            self.assertEqual([call.args[-1] for call in save_video.call_args_list], expected)
            self.assertEqual(runner.env_video_fps, configured_fps)


class EvaluationTests(unittest.TestCase):
    def test_selection_final_sweep_and_demo_entrypoints(self):
        from test_deployment import make_experiment
        from dexmani_policy import record_demo

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiments/toy/task/run"
            config, checkpoint = make_experiment(experiment)
            config["training"] = {"seed": 4, "device": "cpu"}
            config["eval"]["video"] = {"fps": 20}
            config["env_runner"] = {
                "_target_": f"{__name__}.ToyRunner",
                "task": "task",
                "seeds": list(range(10)),
            }
            OmegaConf.save(OmegaConf.create(config), experiment / "config.yaml")
            payload = torch.load(checkpoint, weights_only=False)
            payload["effective_config"] = config
            torch.save(payload, checkpoint)
            cfg = OmegaConf.create({**config, "experiment_dir": str(experiment)})
            with contextlib.redirect_stdout(io.StringIO()):
                select_best_checkpoint(
                    experiment,
                    cfg,
                    initial_episodes=2,
                    batch_size=0,
                    max_episodes=2,
                    inference_steps=3,
                    use_ema=False,
                )
                selected, selected_path = resolve_best_checkpoint(experiment)
                self.assertEqual(selected_path, checkpoint)
                evaluation = evaluate_checkpoint(
                    experiment,
                    cfg,
                    episodes=2,
                    output_dir=experiment / "final",
                )
                from dexmani_policy.env_runner import evaluation as evaluation_module

                with patch.object(
                    evaluation_module,
                    "restore_policy_agent",
                    wraps=evaluation_module.restore_policy_agent,
                ) as load:
                    sweep = evaluate_checkpoint(
                        experiment,
                        cfg,
                        episodes=2,
                        inference_steps_list=[1, 3],
                        output_dir=experiment / "sweep",
                    )
                    load.assert_called_once()
                arguments = [
                    "record_demo.py",
                    "--policy-name",
                    "toy",
                    "--task-name",
                    "task",
                    "--exp-name",
                    "run",
                    "--episodes",
                    "1",
                    "--fps",
                    "30",
                    "--seeds",
                    "0",
                ]
                with (
                    patch.object(record_demo, "ROOT_DIR", str(root)),
                    patch.object(sys, "argv", arguments),
                    patch.object(OmegaConf, "load", wraps=OmegaConf.load) as read_config,
                ):
                    record_demo.main()
                    read_config.assert_called_once()
                outputs = set((experiment / "demo_videos").iterdir())
                with (
                    patch.object(record_demo, "ROOT_DIR", str(root)),
                    patch.object(sys, "argv", [*arguments[:-1], "999"]),
                    patch.object(record_demo, "evaluate_policy") as load,
                ):
                    with self.assertRaisesRegex(ValueError, "Unavailable demo seeds"):
                        record_demo.main()
                    load.assert_not_called()
                self.assertEqual(set((experiment / "demo_videos").iterdir()), outputs)
            self.assertEqual(evaluation[0]["statistics"]["n_valid_episodes"], 2)
            final = json.loads((experiment / "final/result_details.json").read_text())
            self.assertFalse(
                set(final["evaluation_seeds"]["task"]) & set(selection_task_seeds(selected)["task"])
            )
            self.assertEqual(len(sweep), 2)
            self.assertEqual(sweep[0]["evaluation_seeds"], sweep[1]["evaluation_seeds"])
            demo_file = next((experiment / "demo_videos").glob("*/result_details.json"))
            demo = json.loads(demo_file.read_text())
            self.assertFalse(demo["heldout_from_selection"])
            snapshot = OmegaConf.load(experiment / demo["eval_config"])
            self.assertEqual(snapshot.runner_inputs.task.env_video_fps, 30)

    def test_cli_selection_and_evaluation_share_the_concrete_record(self):
        from test_deployment import make_experiment
        from dexmani_policy import eval_best_ckpt, select_best_ckpt

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            experiment = root / "experiments/toy/task/run"
            config, checkpoint = make_experiment(experiment)
            config["training"] = {"seed": 4, "device": "cpu"}
            config["eval"]["video"] = {"fps": 20}
            config["env_runner"] = {
                "_target_": f"{__name__}.ToyRunner",
                "task": "task",
                "seeds": list(range(6)),
            }
            OmegaConf.save(OmegaConf.create(config), experiment / "config.yaml")
            payload = torch.load(checkpoint, weights_only=False)
            payload["effective_config"] = config
            torch.save(payload, checkpoint)
            directory = experiment / "eval_ckpt_selector/pipeline"
            common = ["--policy-name", "toy", "--task-name", "task", "--exp-name", "run"]
            with (
                patch.object(select_best_ckpt, "ROOT_DIR", str(root)),
                patch.object(OmegaConf, "load", wraps=OmegaConf.load) as read_config,
                patch.object(
                    sys,
                    "argv",
                    [
                        "select_best_ckpt.py",
                        *common,
                        "--initial-episodes",
                        "1",
                        "--batch-size",
                        "0",
                        "--no-ema",
                        "--no-videos",
                        "--output-dir",
                        str(directory),
                    ],
                ),
            ):
                select_best_ckpt.main()
                read_config.assert_called_once()
            record = directory / "selection_result.json"
            selection = json.loads(record.read_text())
            self.assertEqual(selection["evaluation"]["runner_inputs"]["task"]["env_video_fps"], 20)
            with (
                patch.object(eval_best_ckpt, "ROOT_DIR", str(root)),
                patch.object(OmegaConf, "load", wraps=OmegaConf.load) as read_config,
                patch.object(
                    sys,
                    "argv",
                    [
                        "eval_best_ckpt.py",
                        *common,
                        "--selection-record",
                        str(record),
                        "--episodes",
                        "1",
                        "eval.video.enabled=false",
                        "eval.video.fps=24",
                    ],
                ),
            ):
                eval_best_ckpt.main()
                read_config.assert_called_once()
            result = next((experiment / "eval_dexsim").glob("*/result_details.json"))
            info = json.loads(result.read_text())
            snapshot = OmegaConf.load(experiment / info["eval_config"])
            self.assertEqual(snapshot.request.selection_result, str(record.relative_to(experiment)))
            self.assertFalse(snapshot.runner_inputs.task.record_video)
            self.assertEqual(snapshot.runner_inputs.task.env_video_fps, 24)
            self.assertFalse(info["use_ema"])
            self.assertTrue(info["heldout_from_selection"])
            self.assertTrue(result.with_name("result.txt").is_file())

    def test_selection_ties_failures_and_pinned_requests(self):
        from test_deployment import make_experiment

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config, first = make_experiment(root)
            config["training"] = {"seed": 4, "device": "cpu"}
            config["env_runner"] = {
                "_target_": f"{__name__}.ToyRunner",
                "task": "task",
                "seeds": [1, 3, 5, 7],
            }
            OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
            payload = torch.load(first, weights_only=False)
            payload["effective_config"] = config
            torch.save(payload, first)
            second = first.with_name("epoch=0002-step=4-milestone=40pct.pt")
            payload["state"]["global_step"] = 4
            torch.save(payload, second)
            cfg = OmegaConf.create(config)
            selected = select_best_checkpoint(
                root,
                cfg,
                initial_episodes=1,
                batch_size=1,
                max_episodes=2,
                inference_steps=3,
                use_ema=False,
            )
            self.assertEqual(OmegaConf.to_container(cfg), config)
            info, checkpoint = resolve_selection_record(root, selected)
            self.assertEqual(checkpoint, second)
            self.assertEqual(len(info["stages"]), 4)
            self.assertEqual(len(selection_task_seeds(info)["task"]), 2)
            self.assertTrue(
                all(not d["success"] for stage in info["stages"] for d in stage["episode_details"])
            )
            self.assertEqual({p.name for p in selected.parent.iterdir()}, {"selection_result.json"})
            pointer = root / "best_ckpt.json"
            before = pointer.read_bytes()
            self.assertEqual(set(json.loads(before)), {"selection_result"})
            for kwargs in (
                {"use_ema": True},
                {"inference_steps_list": [1]},
                {"checkpoint": "20pct"},
            ):
                with self.assertRaisesRegex(ValueError, "Selection record|selection record"):
                    resolve_eval_request(cfg, root, selection_record=selected, **kwargs)
            failure_dir = root / "eval_ckpt_selector/failed"
            with patch(
                "dexmani_policy.env_runner.selection.restore_policy_agent",
                side_effect=RuntimeError("inference failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    select_best_checkpoint(root, cfg, output_dir=failure_dir)
            failure = json.loads((failure_dir / "selection_result.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(pointer.read_bytes(), before)
            with self.assertRaises(ValueError):
                resolve_selection_record(root, failure_dir / "selection_result.json")
            with self.assertRaises(FileExistsError):
                select_best_checkpoint(root, cfg, output_dir=selected.parent)
            self.assertEqual(pointer.read_bytes(), before)
            with patch(
                "dexmani_policy.env_runner.selection.cprint", side_effect=OSError("stdout closed")
            ):
                with self.assertRaisesRegex(OSError, "stdout closed"):
                    select_best_checkpoint(
                        root, cfg, initial_episodes=1, batch_size=0, use_ema=False
                    )
            self.assertEqual(resolve_best_checkpoint(root)[0]["status"], "success")

    def test_seeds_overrides_and_heldout(self):
        pools = {"a": [1, 1, 2, 3, 4], "b": [1, 2, 3]}
        original = copy.deepcopy(pools)
        excluded = {"a": [1], "b": [2]}
        plan = select_eval_seeds(pools, 45, 10, excluded)
        self.assertEqual(plan, select_eval_seeds(pools, 45, 10, excluded))
        self.assertEqual(pools, original)
        self.assertEqual([len(values) for values in plan.values()], [2, 2])
        for task in plan:
            self.assertFalse(set(plan[task]) & set(excluded[task]))
        with self.assertRaises(RuntimeError):
            select_eval_seeds({"task": [1]}, 0, 1, {"task": [1]})
        for override in (
            "agent.horizon=1",
            "env_runner.sensor_modalities=[rgb]",
            "env_runner.task_configs.0.task_text=wrong",
            "dataset.pc_dim=3",
            "+eval.video.fps=24",
            "~eval.video.fps",
            "eval.video.fps",
        ):
            with self.assertRaises(ValueError):
                parse_eval_overrides([override])
        cfg = OmegaConf.create({"eval": {"use_ema": True, "offline": {"use_ema": False}}})
        self.assertFalse(get_eval_param(cfg, "use_ema", "offline"))
        self.assertEqual(parse_eval_overrides(["eval.video.fps=24"]).eval.video.fps, 24)
        self.assertEqual(
            SimRunner.expand_env_kwargs({"texture_random": True, "randomize_light": False})[
                "randomize_light"
            ],
            False,
        )

    def test_saved_task_inputs_and_per_task_seeds(self):
        config = {
            "action_key": "action",
            "agent": {"horizon": 4, "n_obs_steps": 2, "n_action_steps": 2},
            "dataset": {
                "task_names": ["a", "b"],
                "task_texts": ["open the box", "pick up the bottle"],
                "datasets": [
                    {"sensor_modalities": ["joint_state"], "rgb_preprocess_size": [6, 8]},
                    {"sensor_modalities": ["joint_state"], "rgb_preprocess_size": [8, 10]},
                ],
            },
            "env_runner": {
                "_target_": "dexmani_policy.env_runner.multi_task_sim_runner.MultiTaskSimRunner",
                "n_obs_steps": 1,
                "default_eval_episodes": 1,
                "sensor_modalities": ["rgb"],
                "task_configs": [
                    {
                        "task_name": "b",
                        "task_text": "open the box",
                        "eval_seeds": [11, 12],
                        "render_mode": "human",
                        "viewer_resolution": [640, 480],
                        "record_video": False,
                        "env_video_fps": 24,
                    },
                    {"task_name": "a", "task_text": "pick up the bottle", "eval_seeds": [7, 8]},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
            cfg = OmegaConf.create({**config, "experiment_dir": str(root)})
            runner = build_eval_runner(cfg)
            json.dumps(evaluation_snapshot(cfg, runner))
            child = runner.runners["b"]
            self.assertEqual(child.render_mode, "human")
            self.assertEqual(list(child.viewer_resolution), [640, 480])
            self.assertFalse(child.record_video)
            self.assertEqual(child.env_video_fps, 24)
            self.assertEqual(task_seed_pools(runner), {"b": [11, 12], "a": [7, 8]})
            for task, text, height in (("a", "open the box", 6), ("b", "pick up the bottle", 8)):
                child = runner.runners[task]
                self.assertEqual(child.task_text, text)
                self.assertEqual(child.n_obs_steps, 2)
                self.assertEqual(child.sensor_modalities, ["joint_state"])
                self.assertEqual(child.rgb_preprocessing["resize_hw"][0], height)
                agent = ToyAgent()
                with patch.object(agent, "predict_action", wraps=agent.predict_action) as predict:
                    child.get_action_chunk({"joint_state": torch.zeros(1, 2, 19)}, agent)
                    self.assertEqual(predict.call_args.kwargs["obs_dict"]["task_text"], [text])
            reference = save_eval_snapshot(root / "eval", cfg, runner)
            snapshot = OmegaConf.load(root / reference)
            self.assertEqual(snapshot.runner_inputs.a.task_text, "open the box")
            cfg.eval = {"video": {"fps": 30}}
            configured = build_eval_runner(cfg)
            self.assertEqual(
                [child.env_video_fps for child in configured.runners.values()], [30, 30]
            )
            config["dataset"]["datasets"].pop()
            OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
            self.assertEqual(task_seed_pools(build_eval_runner(cfg)), task_seed_pools(runner))
            with self.assertRaises(ValueError):
                build_eval_runner(OmegaConf.create(config))

    def test_eval_overrides_preserve_saved_model_inputs(self):
        from test_deployment import make_experiment

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, checkpoint = make_experiment(root)
            config["training"] = {"seed": 4, "device": "cuda:0"}
            config["agent"]["horizon"] = "${training.seed}"
            cfg = OmegaConf.create(config)
            merged, _, _, _, path = resolve_eval_request(
                cfg,
                root,
                str(checkpoint),
                overrides=["training.seed=9", "training.device=cpu"],
            )
            self.assertEqual(path, checkpoint)
            self.assertEqual(merged.training.seed, 9)
            self.assertEqual(merged.training.device, "cpu")
            self.assertEqual(merged.agent.horizon, 4)
            self.assertEqual(OmegaConf.to_container(cfg), config)

    def test_inference_requests_reject_duplicate_outputs_and_unknown_seed_modes(self):
        from dexmani_policy.training.common.checkpoint import validate_inference_settings

        config = OmegaConf.create({"eval": {"inference_steps_list": [2, 2]}})
        for section in ("offline", "demo"):
            with self.assertRaisesRegex(ValueError, "distinct"):
                resolve_eval_request(config, Path("unused"), "latest", section=section)
        with self.assertRaisesRegex(ValueError, "policy_seed_mode=episode_seed"):
            validate_inference_settings(
                {"inference": {"use_ema": False, "inference_steps": 2, "policy_seed_mode": "fixed"}}
            )

    def test_complete_results_and_statistics(self):
        plan = {"a": [0], "b": [1, 2, 3]}
        result = {
            "per_task": {
                "a": {"episode_details": [episode(0, np.bool_(True), 2, task="a")]},
                "b": {"episode_details": [episode(seed, task="b") for seed in (1, 2, 3)]},
            }
        }
        for task, child in result["per_task"].items():
            validate_episode_details(child["episode_details"], {task: plan[task]})
        stats = compute_eval_stats(result)
        self.assertEqual(stats["micro_success_rate"], 0.25)
        self.assertEqual(stats["macro_success_rate"], 0.5)
        self.assertEqual(stats["n_valid_episodes"], 4)
        self.assertIs(type(result["per_task"]["a"]["episode_details"][0]["success"]), bool)
        json.dumps(result)
        invalid = [
            None,
            [],
            [None],
            [episode(0)],
            [episode(0, task="a"), episode(0, task="a")],
            [episode(2, task="a")],
            [episode(False, task="a")],
            [episode(0, success=1, task="a")],
            [episode(0, steps=-1, task="a")],
            [episode(0, steps=float("nan"), task="a")],
            [{"seed": 0, "success": False, "task_name": "a"}],
            [{**episode(0, task="a"), "error": "failed"}],
            [{**episode(0, task="a"), "error_category": "runtime_error"}],
            [episode(0, task="b")],
        ]
        for value in invalid:
            with self.assertRaises(RuntimeError):
                validate_episode_details(value, {"a": [0]})
        bad = copy.deepcopy(result)
        bad["per_task"]["b"]["episode_details"][0]["task_name"] = "a"
        with self.assertRaises(RuntimeError):
            validate_episode_details(bad["per_task"]["b"]["episode_details"], {"b": plan["b"]})

    def test_selection_record_and_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints/epoch=0001-step=1-milestone=20pct.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            record = {
                "ckpt_relpath": str(checkpoint.relative_to(root)),
                "global_step": 1,
                "status": "success",
                "selection_id": "test",
                "inference": {
                    "use_ema": False,
                    "inference_steps": 3,
                    "policy_seed_mode": "episode_seed",
                },
                "stages": [
                    {
                        "checkpoint": str(checkpoint.relative_to(root)),
                        "status": "completed",
                        "requested_task_seeds": {"task": [0]},
                        "episode_details": [episode(0, True, 2, "task")],
                    }
                ],
            }
            path = root / "selection_result.json"
            path.write_text(json.dumps(record))
            (root / "best_ckpt.json").write_text(json.dumps({"selection_result": path.name}))
            info, resolved = resolve_best_checkpoint(root)
            self.assertEqual(resolved, checkpoint)
            self.assertEqual(selection_task_seeds(info), {"task": [0]})
            incomplete = copy.deepcopy(record)
            incomplete["stages"][0]["episode_details"] = []
            with self.assertRaisesRegex(RuntimeError, "episodes"):
                selection_task_seeds(incomplete)
            for status in ("running", "failed"):
                record["status"] = status
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "completed"):
                    resolve_best_checkpoint(root)
            runner = SimRunner("task", 2, 1, env_kwargs={"texture_random": True})
            cfg = OmegaConf.create({"experiment_dir": str(root)})
            output = root / "result"
            reference = save_eval_snapshot(output, cfg, runner, inference_steps=3)
            snapshot = OmegaConf.load(root / reference)
            self.assertTrue(snapshot.runner_inputs.task.env_kwargs.randomize_camera_pose)
            with self.assertRaises(FileExistsError):
                save_eval_snapshot(output, cfg, runner, inference_steps=3)


if __name__ == "__main__":
    unittest.main()
