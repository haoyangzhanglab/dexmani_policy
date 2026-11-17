import json

import numpy as np
import pytest
import torch

from dexmani_policy.common.checkpoint import Checkpoint, save_checkpoint
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation import rollout, selection


class SmallPolicy(torch.nn.Module):
    n_obs_steps = 3
    n_action_steps = 2
    future_steps = 3
    control_action_dim = 19
    normalization_dims = {"joint_state": 19, "action": 19}
    requires_prev_action = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def predict_action(self, obs, *, generator, prev_action=None, **kwargs):
        self.calls.append((obs["joint_state"].clone(), prev_action, generator.initial_seed()))
        return {"future_actions": torch.tensor([10., 20., 30.])[None, :, None].expand(1, 3, 19)}


class SmallEnv:
    dt = 0.05
    video_fps = 20

    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)
        self.close_count = 0
        self.close_error = False
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
            "executed_joint_action": action + 1,
        }

    def get_video(self):
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def close(self):
        self.close_count += 1
        if self.close_error:
            raise RuntimeError("close failed")


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


def test_rollout_uses_executed_action_and_stops_mid_chunk(environments):
    agent = SmallPolicy()
    results = list(make_runner(env_rebuild_interval=1).run(
        agent, seeds=[7, 8], inference_steps=2, action_steps=2,
    ))
    assert [(r["seed"], r["success"], r["steps"], r["total_steps"]) for r in results] == [
        (7, True, 2, 3), (8, True, 2, 3),
    ]
    assert [env.close_count for env in environments] == [1, 1]
    for offset, seed in ((0, 7), (2, 8)):
        assert agent.calls[offset][1] is None
        torch.testing.assert_close(agent.calls[offset + 1][1], torch.full((1, 19), 21.))
        np.testing.assert_array_equal(agent.calls[offset + 1][0][0, :, 0], [0, 1, 2])
        assert agent.calls[offset][2] == agent.calls[offset + 1][2] == seed


@pytest.mark.parametrize("early_close", [False, True])
def test_normal_shutdown_propagates_close_failure(environments, early_close):
    results = make_runner().run(SmallPolicy(), seeds=[7], inference_steps=2, action_steps=2)
    assert next(results)["success"]
    environments[0].close_error = True
    with pytest.raises(RuntimeError, match="close failed"):
        results.close() if early_close else next(results)
    assert environments[0].close_count == 1


def test_step_error_preserved_during_cleanup(environments, capsys):
    results = make_runner().run(SmallPolicy(), seeds=[7, 8], inference_steps=2, action_steps=2)
    next(results)
    environments[0].step_error = environments[0].close_error = True
    with pytest.raises(ValueError, match="step failed"):
        next(results)
    assert environments[0].close_count == 1
    assert "close failed" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["init", "dt"])
def test_partial_environment_cleanup(monkeypatch, capsys, failure):
    import dexmani_sim

    created = []

    class FailingEnv(SmallEnv):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(self)
            self.close_error = True
            if failure == "init":
                raise ValueError("init failed")
            self.dt = 0.1

    monkeypatch.setattr(dexmani_sim, "resolve_env_class", lambda task: FailingEnv)
    with pytest.raises(ValueError, match="init failed|Control period mismatch"):
        make_runner().make_env(record_video=False)
    assert created[0].close_count == 1
    assert "close failed" in capsys.readouterr().err


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
        "eval": {"use_ema": False, "inference_steps": 2},
        "env_runner": {"_target_": "dexmani_policy.env_runner.sim_runner.SimRunner", "task_name": "test", "env_kwargs": {}},
        "control_dt_s": 0.05, "normalization_dims": SmallPolicy.normalization_dims, "dataset": {},
    }
    return [save_checkpoint(tmp_path / f"{step}.pt", Checkpoint(step, {}, {}, {}, cfg)) for step in (1, 2)]


def test_selection_and_independent_test(checkpoints, tmp_path):
    path = selection.run_pipeline(
        checkpoints, selection_episodes=2, test_episodes=3, seed=5, device="cpu", output_dir=tmp_path / "eval",
    )
    record = json.loads(path.read_text())
    assert record["global_step"] == 2
    assert set(record["selection_seeds"]).isdisjoint(record["test_seeds"])
    for candidate in record["candidates"]:
        result = json.loads((path.parent / candidate["result"]).read_text())
        assert result["seeds"] == record["selection_seeds"]
        assert result["completed"] and result["statistics"]["n_success"] == 2
    result = json.loads((path.parent / record["test_result"]).read_text())
    assert result["seeds"] == record["test_seeds"] and result["completed"]
    assert (result["weights"], result["inference_steps"], result["action_steps"]) == ("raw", 2, 2)
    candidates = [
        {"global_step": step, "statistics": {"success_rate": rate, "avg_steps": steps}}
        for rate, steps, step in [(0, None, 100), (0.5, 1, 100), (1, 5, 100), (1, 2, 1), (1, 2, 2)]
    ]
    assert sorted(candidates, key=selection.candidate_score) == candidates


@pytest.mark.parametrize("failure", ["video", "write"])
def test_evaluation_failure_closes_env_and_preserves_primary(checkpoints, environments, tmp_path, monkeypatch, capsys, failure):
    output = tmp_path / "failed_eval"
    if failure == "video":
        def fail_video(*args):
            environments[0].close_error = True
            raise OSError("video failed")

        monkeypatch.setattr("dexmani_policy.env_runner.video.save_episode_video", fail_video)
    else:
        write_json = rollout.write_json

        def fail_write(path, record):
            if record["episodes"]:
                environments[0].close_error = True
                raise OSError("write failed")
            write_json(path, record)

        monkeypatch.setattr(rollout, "write_json", fail_write)
    with pytest.raises(OSError, match=f"{failure} failed"):
        rollout.evaluate_policy(checkpoints[0], seeds=[7], device="cpu", video=failure == "video", output_dir=output)
    assert environments[0].close_count == 1
    assert "close failed" in capsys.readouterr().err
    record = json.loads((output / "result_details.json").read_text())
    assert not record["completed"]
    if failure == "video":
        assert record["statistics"]["n_success"] == 1
        assert record["episodes"][0]["video"] is None
