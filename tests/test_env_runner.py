import subprocess
import sys
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from imageio_ffmpeg import read_frames

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.rgb_preprocessing import preprocess_validation_rgb
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.env_runner.video import save_episode_video


class Policy(torch.nn.Module):
    hand_dim = 2

    def __init__(self, history_steps=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.history_steps = history_steps
        self.calls = []

    def predict_action(self, obs_dict, inference_steps, generator, action_context=None):
        assert not torch.is_grad_enabled()
        actions = torch.arange(12).reshape(1, 4, 3) + torch.rand((1, 4, 3), generator=generator)
        self.calls.append({
            "obs": obs_dict["joint_state"].clone(),
            "steps": inference_steps,
            "actions": actions.clone(),
            "history": None if action_context is None else action_context["hand_history"].clone(),
        })
        return {"future_actions": actions}


class Environment:
    video_fps = 17

    def __init__(self, **kwargs):
        self.options = kwargs
        self.success = True
        self.closed = False
        self.video_reads = 0
        self.frames = np.stack([np.full((16, 24, 3), i * 50, np.uint8) for i in range(4)])

    def reset(self, seed):
        self.action_cnt = 0
        self.actions = []
        return {"joint_state": np.full(3, seed, np.float32)}, {}

    def step(self, action):
        self.action_cnt += 1
        self.actions.append(action.copy())
        executed = np.clip(action, -0.25, 0.25)
        finished = self.action_cnt == 3
        return (
            {"joint_state": np.full(3, self.action_cnt, np.float32)},
            0, finished and self.success, finished and not self.success,
            {"success_condition": self.action_cnt in (1, 3),
             "success": finished and self.success, "executed_joint_action": executed},
        )

    def get_video(self):
        self.video_reads += 1
        return self.frames

    def close(self):
        self.closed = True


@pytest.fixture
def environments(monkeypatch):
    created = []

    def create(**kwargs):
        env = Environment(**kwargs)
        created.append(env)
        return env

    monkeypatch.setitem(sys.modules, "dexmani_sim", SimpleNamespace(resolve_env_class=lambda task: create))
    return created


@pytest.mark.parametrize("window", [1, 2, 3])
def test_observation_history(window):
    runner = SimRunner("task", window, ["joint_state"], {})
    for step in range(5):
        value = np.array([step], np.float32)
        runner.update_obs({"joint_state": value})
        value[:] = -1
        expected = np.maximum(np.arange(step - window + 1, step + 1), 0)
        stacked = runner.get_stacked_obs()["joint_state"]
        np.testing.assert_array_equal(stacked[:, 0], expected)
        stacked[:] = -2
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"][:, 0], expected)

    runner.reset()
    runner.update_obs({"joint_state": np.array([9], np.float32)})
    np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"], np.full((window, 1), 9))


@pytest.mark.parametrize("resize, keep_uint8", [(None, True), ((4, 6), True), ((4, 6), False)])
def test_rgb_batch_matches_dataset_recipe(resize, keep_uint8):
    recipe = dict(resize_hw=resize, center_crop_hw=(2, 4) if resize else None,
                  keep_uint8=keep_uint8, float_spatial_before_uint8=True)
    runner = SimRunner("task", 2, ["joint_state", "rgb"], recipe)
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    runner.update_obs({"joint_state": np.ones(3, np.float64), "rgb": rgb})
    batch = runner.get_obs_batch("cpu")
    expected = preprocess_validation_rgb(np.stack([rgb, rgb]), **recipe)
    torch.testing.assert_close(batch["rgb"], expected.unsqueeze(0))
    assert batch["joint_state"].shape == (1, 2, 3)
    assert batch["joint_state"].dtype == torch.float32
    assert batch["rgb"].dtype == (torch.uint8 if keep_uint8 else torch.float32)


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("history_steps", [0, 3])
def test_episode_chunks_history_and_random_stream(success, history_steps):
    runner = SimRunner("task", 2, ["joint_state"], {})
    agent = Policy(history_steps)
    env = Environment()
    env.success = success
    rng_state = torch.get_rng_state()
    assert runner.run_episode(agent, env, 17, 4, action_steps=2) == (success, 1)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert len(env.actions) == 3 and len(agent.calls) == 2
    assert all(call["steps"] == 4 for call in agent.calls)
    np.testing.assert_array_equal(env.actions[:2], agent.calls[0]["actions"][0, :2].numpy())
    np.testing.assert_array_equal(env.actions[2], agent.calls[1]["actions"][0, 0].numpy())
    torch.testing.assert_close(agent.calls[0]["obs"], torch.full((1, 2, 3), 17.))
    torch.testing.assert_close(agent.calls[1]["obs"][0, :, 0], torch.tensor([1., 2.]))
    if history_steps:
        torch.testing.assert_close(agent.calls[0]["history"], torch.full((1, 3, 2), 17.))
        torch.testing.assert_close(
            agent.calls[1]["history"], torch.tensor([[[17., 17.], [0.25, 0.25], [0.25, 0.25]]])
        )

    torch.rand(19)
    runner.run_episode(agent, env, 17, 4, action_steps=2)
    torch.testing.assert_close(agent.calls[0]["actions"], agent.calls[2]["actions"])
    torch.testing.assert_close(agent.calls[1]["actions"], agent.calls[3]["actions"])
    torch.testing.assert_close(agent.calls[0]["obs"], agent.calls[2]["obs"])
    if history_steps:
        torch.testing.assert_close(agent.calls[0]["history"], agent.calls[2]["history"])


@pytest.mark.parametrize("frequency, count", [(0, 1), (1, 26), (25, 2)])
def test_environment_lifecycle_without_video(environments, frequency, count):
    runner = SimRunner("task", 2, ["joint_state"], {}, env_rebuild_interval=frequency,
                       env_kwargs={"randomize_model_id": True, "randomize_table_height": True})
    rows = list(runner.run(Policy(), seeds=range(26), inference_steps=4, action_steps=2))
    assert len(rows) == 26 and len(environments) == count
    assert all(row["video"] is None and row["total_steps"] == 3 for row in rows)
    for env in environments:
        assert env.closed and env.video_reads == 0
        assert env.options == {"render_mode": "rgb_array", "record_video": False,
                               "randomize_model_id": True, "randomize_table_height": True}


def test_episode_video_roundtrip(environments, tmp_path):
    runner = SimRunner("task", 2, ["joint_state"], {}, env_kwargs={"render_mode": "human"})
    rows = list(runner.run(Policy(), seeds=[7], inference_steps=4, action_steps=2, video_save_dir=tmp_path))
    env, = environments
    assert env.options == {"render_mode": "human", "record_video": True}
    assert env.closed and env.video_reads == 1
    path = rows[0]["video"]
    assert path == str(tmp_path / "episode_7.mp4")
    reader = read_frames(path)
    metadata = next(reader)
    frames = list(reader)
    assert metadata["size"] == (24, 16) and metadata["fps"] == 17
    assert len(frames) == 4
    for index, frame in enumerate(frames):
        decoded = np.frombuffer(frame, np.uint8).astype(int)
        assert np.max(np.abs(decoded - index * 50)) <= 3


def test_video_failure_closes_environment(environments, monkeypatch, tmp_path, capfd):
    monkeypatch.setattr(Environment, "get_video", lambda self: np.zeros((4, 15, 24, 3), np.uint8))
    runner = SimRunner("task", 2, ["joint_state"], {})
    rows = []
    with pytest.raises(subprocess.CalledProcessError):
        rows.extend(runner.run(Policy(), seeds=[7], inference_steps=4, action_steps=2, video_save_dir=tmp_path))
    assert rows == [] and environments[0].closed
    assert "not divisible by 2" in capfd.readouterr().err


def test_empty_video_fails(tmp_path):
    env = Environment()
    env.frames = None
    with pytest.raises(ValueError, match="No video frames"):
        save_episode_video(env, tmp_path, "episode")


def test_early_consumer_exit_closes_environment(environments):
    runner = SimRunner("task", 2, ["joint_state"], {})
    with closing(runner.run(Policy(), seeds=[1, 2], inference_steps=4, action_steps=2)) as episodes:
        assert next(episodes)["seed"] == 1
    assert len(environments) == 1 and environments[0].closed


def test_conflicting_video_option_fails(environments):
    runner = SimRunner("task", 2, ["joint_state"], {}, env_kwargs={"record_video": True})
    with pytest.raises(TypeError, match="record_video"):
        list(runner.run(Policy(), seeds=[1], inference_steps=4, action_steps=2))
    assert environments == []


@pytest.mark.parametrize("nonfinite", [False, True])
def test_physical_policy_output_boundary(monkeypatch, nonfinite):
    prediction = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    if nonfinite:
        prediction[0, -1, 0] = float("nan")
    policy = Policy()
    metadata = SimpleNamespace(action_start_index=1, control_action_dim=3)
    monkeypatch.setattr(policy, "predict_action", lambda **kwargs: BasePolicy.format_action_output(metadata, prediction))
    runner = SimRunner("task", 2, ["joint_state"], {})
    if nonfinite:
        with pytest.raises(ValueError, match="Nonfinite physical policy prediction"):
            runner.predict_action_chunk({}, policy, action_steps=2)
    else:
        actions = runner.predict_action_chunk({}, policy, action_steps=2)
        np.testing.assert_array_equal(actions, [[4., 5., 6.], [8., 9., 10.]])
        assert actions.flags.c_contiguous
