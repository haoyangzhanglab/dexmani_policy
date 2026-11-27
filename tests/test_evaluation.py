import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.env_runner.dexjoco_runner import DexJoCoRunner
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation import rollout, selection

from tests.helpers import tiny_policy_config


class SmallPolicy(torch.nn.Module):
    n_obs_steps = 3
    n_action_steps = 2
    future_steps = 3
    control_action_dim = 19
    normalization_dims = {"joint_state": 19, "action": 19}
    uses_ddim = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def predict_action(self, obs, *, generator, inference_steps=None):
        self.calls.append((obs["joint_state"].clone(), generator.initial_seed()))
        return {"future_actions": torch.tensor([10., 20., 30.])[None, :, None].expand(1, 3, 19)}


class SmallEnv:
    dt = 0.05

    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)
        self.close_count = 0
        self.step_error = False

    def reset(self, *, seed):
        self.action_cnt = 0
        self.state.fill(0)
        return {"joint_state": self.state}, {}

    def step(self, action):
        if self.step_error:
            raise ValueError("step failed")
        self.action_cnt += 1
        self.state.fill(self.action_cnt)
        done = self.action_cnt == 3
        return {"joint_state": self.state}, 0, done, False, {
            "success_condition": self.action_cnt >= 2,
            "success": done,
        }

    def get_video(self):
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def close(self):
        self.close_count += 1


class CustomDexJoCoRunner(DexJoCoRunner):
    pass


@pytest.fixture
def environments(monkeypatch):
    created = []

    def make_env(self, *, record_video):
        env = SmallEnv()
        created.append(env)
        return env

    monkeypatch.setattr(SimRunner, "make_env", make_env)
    return created


def make_runner(**kwargs):
    return SimRunner("test", 3, ["joint_state"], {}, 0.05, **kwargs)


def test_observation_history_owns_snapshots():
    runner = make_runner()
    value = np.zeros(19, dtype=np.float32)
    for step, expected in enumerate(([0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3])):
        value.fill(step)
        runner.update_obs({"joint_state": value, "unused": np.zeros(1)})
        value.fill(-1)
        history = runner.get_stacked_obs()
        assert set(history) == {"joint_state"}
        np.testing.assert_array_equal(history["joint_state"][:, 0], expected)
        history["joint_state"][:] = -2
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"][:, 0], expected)
    runner.reset()
    value.fill(9)
    runner.update_obs({"joint_state": value})
    np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"][:, 0], [9, 9, 9])


def test_rollout_preserves_history_and_stops_mid_chunk(environments):
    agent = SmallPolicy()
    results = list(make_runner(env_rebuild_interval=1).run(
        agent, seeds=[7, 8], inference_steps=2, action_steps=2,
    ))
    assert [(r["seed"], r["success"], r["steps"], r["total_steps"]) for r in results] == [
        (7, True, 2, 3), (8, True, 2, 3),
    ]
    assert [env.close_count for env in environments] == [1, 1]
    for offset, seed in ((0, 7), (2, 8)):
        np.testing.assert_array_equal(agent.calls[offset][0][0, :, 0], [0, 0, 0])
        np.testing.assert_array_equal(agent.calls[offset + 1][0][0, :, 0], [0, 1, 2])
        assert agent.calls[offset][1] == agent.calls[offset + 1][1] == seed


def test_stopping_rollout_iteration_closes_env(environments):
    results = make_runner().run(SmallPolicy(), seeds=[7], inference_steps=2, action_steps=2)
    assert next(results)["success"]
    results.close()
    assert environments[0].close_count == 1


def test_fast_rollout_preserves_history_and_stops_in_skipped_steps(monkeypatch):
    monkeypatch.setitem(sys.modules, "dexmani_sim", SimpleNamespace(TACTILE_OBSERVATION_KEYS=("finger_tactile_force",)))

    class CountingEnv(SmallEnv):
        def __init__(self):
            super().__init__()
            self.captured_steps = []

        def step(self, action):
            self.captured_steps.append(self.action_cnt + 1)
            return super().step(action)

        def step_physics(self, action, *, collect_tactile):
            assert not collect_tactile
            _, reward, done, truncated, info = super().step(action)
            return None, reward, done, truncated, info

    calls = []
    for skip in (False, True):
        env = CountingEnv()
        runner = SimRunner("test", 1, ["joint_state"], {}, 0.05, skip_intermediate_observations=skip)
        monkeypatch.setattr(runner, "make_env", lambda *, record_video: env)
        agent = SmallPolicy()
        result, = runner.run(agent, seeds=[7], inference_steps=2, action_steps=2)
        assert result["success"] and result["steps"] == 2 and result["total_steps"] == 3
        assert env.captured_steps == ([2] if skip else [1, 2, 3])
        assert env.close_count == 1
        calls.append(agent.calls)
    torch.testing.assert_close(calls[0], calls[1], rtol=0, atol=0)


def test_step_error_closes_env(environments):
    results = make_runner().run(SmallPolicy(), seeds=[7, 8], inference_steps=2, action_steps=2)
    next(results)
    environments[0].step_error = True
    with pytest.raises(ValueError, match="step failed"):
        next(results)
    assert environments[0].close_count == 1


def test_control_period_mismatch_closes_env(monkeypatch):
    created = []

    class WrongPeriodEnv(SmallEnv):
        def __init__(self, **kwargs):
            assert kwargs["observation_keys"] == ("joint_state",)
            super().__init__()
            created.append(self)
            self.dt = 0.1

    monkeypatch.setitem(sys.modules, "dexmani_sim", SimpleNamespace(resolve_env_class=lambda task: WrongPeriodEnv))
    with pytest.raises(ValueError, match="Control period mismatch"):
        make_runner().make_env(record_video=False)
    assert created[0].close_count == 1


@pytest.mark.parametrize("runner_class,attribute", [(SimRunner, "dt"), (DexJoCoRunner, "data_processing")])
def test_unreadable_env_metadata_releases_env(monkeypatch, caplog, runner_class, attribute):
    kwargs = {"data_processing": {"joint_state": {}}} if runner_class is DexJoCoRunner else {}
    runner = runner_class("test", 1, ["joint_state"], {}, 0.05, **kwargs)
    env = SmallEnv()
    failure = ValueError(f"unreadable {attribute}")
    monkeypatch.setattr(SmallEnv, attribute, property(Mock(side_effect=failure)), raising=False)
    env.close = Mock(side_effect=OSError("env cleanup failed"))
    monkeypatch.setattr(runner, "_make_env", lambda **kwargs: env)
    with pytest.raises(ValueError) as caught:
        runner.make_env(record_video=False)
    assert caught.value is failure
    env.close.assert_called_once_with()
    assert "env cleanup failed" in caplog.text


def test_early_rollout_close_propagates_close_failure(monkeypatch):
    env, runner = SmallEnv(), make_runner()
    env.close = Mock(side_effect=OSError("env cleanup failed"))
    monkeypatch.setattr(runner, "make_env", lambda **kwargs: env)
    results = runner.run(SmallPolicy(), seeds=[7, 8], inference_steps=2, action_steps=2)
    next(results)
    with pytest.raises(OSError, match="env cleanup failed"):
        results.close()
    env.close.assert_called_once_with()


@pytest.mark.parametrize("change", [None, "num_points", "size"])
def test_dexjoco_point_cloud_observation_processing(monkeypatch, change):
    processing = {"proprio": {"layout": "wxyz"}, "rgb": {"size": [224, 224]}, "point_cloud": {"num_points": 4096}}
    actual = copy.deepcopy(processing)
    if change == "num_points":
        actual["point_cloud"]["num_points"] = 1024
    elif change == "size":
        actual["rgb"]["size"] = [128, 128]
    env = SmallEnv()
    env.data_processing = actual
    monkeypatch.setitem(sys.modules, "dexjoco.policy_env", SimpleNamespace(DexJoCoEnv=lambda *args, **kwargs: env))
    runner = DexJoCoRunner("test", 1, ["proprio", "point_cloud"], {}, env.dt, data_processing=processing)
    if change is None:
        assert runner.make_env(record_video=False) is env
        env.close()
    else:
        with pytest.raises(ValueError, match=f"DexJoCo observation processing mismatch.*{change}"):
            runner.make_env(record_video=False)
    assert env.close_count == 1


@pytest.fixture
def checkpoints(tmp_path, monkeypatch, environments):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    def restore(*args):
        assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
        assert torch.get_float32_matmul_precision() == "highest"
        return SmallPolicy()

    monkeypatch.setattr(rollout, "restore_policy", restore)
    monkeypatch.setattr(SimRunner, "load_seed_pool", staticmethod(lambda task: list(range(100, 110))))
    cfg = {
        "agent": {"_target_": f"{__name__}.SmallPolicy", "n_obs_steps": 3, "n_action_steps": 2, "horizon": 5},
        "eval": {"use_ema": False, "inference_steps": 2},
        "env_runner": {"_target_": "dexmani_policy.env_runner.sim_runner.SimRunner", "task_name": "test", "env_kwargs": {}},
        "control_dt_s": 0.05, "normalization_dims": SmallPolicy.normalization_dims, "dataset": {}, "action_key": "action",
    }
    return [save_checkpoint(tmp_path / f"{step}.pt", Checkpoint(step, {}, {}, {}, cfg)) for step in (1, 2)]


@pytest.mark.parametrize("skip_intermediate_observations", [False, True])
@pytest.mark.parametrize("default_kwargs_index", [0, 1])
@pytest.mark.parametrize("omit_env_kwargs", [False, True])
def test_selection_and_independent_test(
    checkpoints, tmp_path, skip_intermediate_observations, default_kwargs_index, omit_env_kwargs,
):
    checkpoint_path = checkpoints[default_kwargs_index]
    checkpoint = load_checkpoint(checkpoint_path)
    env_runner = checkpoint.effective_config["env_runner"]
    if omit_env_kwargs:
        del env_runner["env_kwargs"]
    else:
        env_runner["env_kwargs"] = None
    save_checkpoint(checkpoint_path, checkpoint)
    path = selection.run_pipeline(
        checkpoints, selection_episodes=2, test_episodes=3, seed=5, device="cpu", output_dir=tmp_path / "eval",
        skip_intermediate_observations=skip_intermediate_observations,
    )
    record = json.loads(path.read_text())
    assert record["global_step"] == 2
    assert record["skip_intermediate_observations"] is skip_intermediate_observations
    assert set(record["selection_seeds"]).isdisjoint(record["test_seeds"])
    for candidate in record["candidates"]:
        result = json.loads((path.parent / candidate["result"]).read_text())
        assert result["seeds"] == record["selection_seeds"]
        assert result["skip_intermediate_observations"] is skip_intermediate_observations
        assert result["completed"] and result["statistics"]["n_success"] == 2
    result = json.loads((path.parent / record["test_result"]).read_text())
    assert result["seeds"] == record["test_seeds"] and result["completed"]
    assert result["skip_intermediate_observations"] is skip_intermediate_observations
    assert (result["weights"], result["inference_steps"], result["action_steps"]) == ("raw", 2, 2)
    candidates = [
        {"global_step": step, "statistics": {"success_rate": rate, "avg_steps": steps}}
        for rate, steps, step in [(0, None, 100), (0.5, 1, 100), (1, 5, 100), (1, 2, 1), (1, 2, 2)]
    ]
    assert sorted(candidates, key=selection.candidate_score) == candidates


def test_selection_rejects_different_env_kwargs(checkpoints, tmp_path):
    checkpoint = load_checkpoint(checkpoints[1])
    checkpoint.effective_config["env_runner"]["env_kwargs"] = {"randomize": True}
    save_checkpoint(checkpoints[1], checkpoint)
    output = tmp_path / "different_env"
    with pytest.raises(ValueError, match="Candidates must use the same env_kwargs"):
        selection.run_pipeline(checkpoints, device="cpu", output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("steps,missing_ema,message", [
    (1, False, "leading DDIM"), (9, False, "num_training_steps"), (2, True, "EMA weights are absent"),
])
def test_selection_checks_later_candidate_before_rollout(checkpoints, environments, tmp_path, steps, missing_ema, message):
    checkpoint = load_checkpoint(checkpoints[1])
    agent = tiny_policy_config("unused", "unused",
        "horizon=4", "n_obs_steps=3", "n_action_steps=2", "agent.action_dim=19", "agent.num_training_steps=8",
    ).agent
    checkpoint.effective_config["agent"] = OmegaConf.to_container(agent, resolve=True)
    checkpoint.effective_config["eval"]["use_ema"] = missing_ema
    save_checkpoint(checkpoints[1], checkpoint)
    output = tmp_path / "eval"
    with pytest.raises(ValueError, match=message) as caught:
        selection.run_pipeline(checkpoints, inference_steps=steps, selection_episodes=2, test_episodes=2,
                               device="cpu", output_dir=output)
    assert str(checkpoints[1]) in str(caught.value)
    assert not environments and not output.exists()


def test_selection_checks_execution_window_before_rollout(checkpoints, environments, tmp_path):
    for path, horizon in zip(checkpoints, (8, 4)):
        checkpoint = load_checkpoint(path)
        agent = tiny_policy_config("unused", "unused", f"horizon={horizon}", "n_obs_steps=2", "n_action_steps=3").agent
        checkpoint.effective_config["agent"] = OmegaConf.to_container(agent, resolve=True)
        save_checkpoint(path, checkpoint)
    output = tmp_path / "eval"
    with pytest.raises(ValueError, match="future_steps") as caught:
        selection.run_pipeline(checkpoints, action_steps=4, output_dir=output)
    assert str(checkpoints[1]) in str(caught.value)
    assert not environments and not output.exists()


def test_fixed_checkpoint_uses_only_test_seeds(checkpoints, tmp_path):
    output = tmp_path / "fixed_eval"
    with pytest.raises(ValueError, match="at least two distinct checkpoints"):
        selection.run_pipeline(checkpoints[:1], output_dir=output)
    _, path = rollout.evaluate_policy(checkpoints[0], episodes=3, seed=5, device="cpu", output_dir=output)
    assert path == output / "result_details.json"
    result = json.loads(path.read_text())
    test_seeds, _ = selection.split_evaluation_seeds(range(100, 110), 5, 3, 2)
    assert [row["seed"] for row in result["episodes"]] == test_seeds
    assert result["control_dt_s"] == SmallEnv.dt
    assert result["completed"] and not (output / "candidates").exists()


@pytest.mark.parametrize("runner,action_key", [
    (SimRunner, "action_rotvec"), (DexJoCoRunner, "action"), (CustomDexJoCoRunner, "action"),
])
def test_eval_rejects_unsupported_action_before_loading_environment(checkpoints, tmp_path, runner, action_key):
    checkpoint = load_checkpoint(checkpoints[0])
    checkpoint.effective_config["env_runner"]["_target_"] = f"{runner.__module__}.{runner.__name__}"
    checkpoint.effective_config["action_key"] = "action_rotvec" if issubclass(runner, DexJoCoRunner) else "action"
    save_checkpoint(checkpoints[0], checkpoint)
    checkpoint.effective_config["action_key"] = action_key
    source = save_checkpoint(tmp_path / "wrong_action.pt", checkpoint)
    output = tmp_path / "eval"
    error = f"{runner.__name__} does not support {action_key}"
    with pytest.raises(ValueError, match=error):
        rollout.evaluate_policy(source, device="cpu", output_dir=output)
    with pytest.raises(ValueError, match=error):
        selection.run_pipeline([checkpoints[0], source], device="cpu", output_dir=output)
    assert not output.exists()


def test_dexjoco_subclass_uses_checkpoint_preprocessing(checkpoints, tmp_path):
    source = checkpoints[0]
    checkpoint = load_checkpoint(source)
    cfg = checkpoint.effective_config
    cfg["env_runner"]["_target_"] = f"{__name__}.CustomDexJoCoRunner"
    cfg["action_key"] = "action_rotvec"
    cfg["data_processing"] = {"joint_state": {}}
    save_checkpoint(source, checkpoint)
    result, _ = rollout.evaluate_policy(source, seeds=[7], device="cpu", output_dir=tmp_path / "eval")
    assert result["completed"] and result["episodes"][0]["success"]


@pytest.mark.parametrize("failure", ["video", "write"])
def test_evaluation_failure_closes_env_and_keeps_published_episodes(checkpoints, environments, tmp_path, monkeypatch, caplog, failure):
    output = tmp_path / "failed_eval"
    if failure == "video":
        def fail_video(frames, fps, path):
            assert fps == 1 / SmallEnv.dt
            if path.stem == "episode_8":
                raise OSError("video failed")
            return str(path)

        monkeypatch.setattr("dexmani_policy.env_runner.video.save_episode_video", fail_video)
    else:
        write_json = rollout.write_json
        close = SmallEnv.close

        def fail_close(env):
            close(env)
            raise OSError("env cleanup failed")

        monkeypatch.setattr(SmallEnv, "close", fail_close)

        def fail_write(path, record):
            if len(record["episodes"]) == 2:
                raise OSError("write failed")
            write_json(path, record)

        monkeypatch.setattr(rollout, "write_json", fail_write)
    with pytest.raises(OSError, match=f"{failure} failed"):
        rollout.evaluate_policy(checkpoints[0], seeds=[7, 8], device="cpu", video=failure == "video", output_dir=output)
    assert environments[0].close_count == 1
    if failure == "write":
        assert "env cleanup failed" in caplog.text
    record = json.loads((output / "result_details.json").read_text())
    assert not record["completed"]
    assert record["statistics"]["n_success"] == 1
    assert [row["seed"] for row in record["episodes"]] == [7]
    expected_video = str(Path("videos/test/episode_7.mp4")) if failure == "video" else None
    assert record["episodes"][0]["video"] == expected_video
