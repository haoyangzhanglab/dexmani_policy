import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy import eval_pipeline as pipeline, evaluate
from dexmani_policy.common.experiments import (
    MILESTONE_PATTERN,
    milestone_steps,
    resolve_milestone_path,
)
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation import artifacts, config, rollout, selection


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    (root / "checkpoints").mkdir(parents=True)
    for step, pct in milestone_steps(100).items():
        (root / "checkpoints" / f"epoch=1-step={step}-milestone={pct}pct.pt").touch()
    (root / "checkpoints/latest.pt").touch()
    cfg = {
        "action_key": "action",
        "dataset": {"sensor_modalities": ["joint_state"], "action_key": "action"},
        "agent": {"n_obs_steps": 2, "n_action_steps": 3, "horizon": 8},
        "policy_metadata": {"action_start_index": 1},
        "training": {"device": "cpu", "loop": {"total_train_steps": 100}},
        "env_runner": {"task_name": "test_task", "eval_seeds": list(range(64))},
        "eval": {
            "seed": 1024, "episodes": 3, "selection_episodes": 2,
            "inference_steps": 4, "action_steps": None, "use_ema": True,
            "task_seeds": None, "video": {"enabled": False},
        },
    }
    OmegaConf.save(cfg, root / "config.yaml")
    return root, cfg


@pytest.fixture
def simulation(monkeypatch):
    state = {"envs": [], "episodes": [], "loads": [], "fault": None, "success": True}

    class Env:
        action_cnt = 5
        video_fps = 15

        def __init__(self):
            self.closed = False
            state["envs"].append(self)

        def close(self):
            self.closed = True
            if state["fault"] == "cleanup":
                raise OSError("cleanup failure")

    def restore(cfg, checkpoint, *, use_ema, device):
        state["loads"].append((checkpoint, use_ema, device))
        if state["fault"] == "load":
            raise RuntimeError("load failure")
        if state["fault"] == "second_load" and len(state["loads"]) == 2:
            raise RuntimeError("second load failure")
        agent = torch.nn.Linear(1, 1)
        match = MILESTONE_PATTERN.fullmatch(Path(checkpoint).name)
        agent.checkpoint_step = int(match["step"]) if match else 100
        return agent

    def episode(self, agent, env, seed, inference_steps, *, action_steps):
        if state["episodes"]:
            if state["fault"] in {"rollout", "cleanup"}:
                raise RuntimeError("rollout failure")
            if state["fault"] == "interrupt":
                raise KeyboardInterrupt("interrupted")
        state["episodes"].append((agent.checkpoint_step, seed, inference_steps, action_steps))
        return state["success"], 2 if state["success"] else None

    def video(env, directory, name, fps):
        if state["fault"] == "video" and len(state["episodes"]) == 2:
            raise OSError("video failure")
        return None

    monkeypatch.setattr(rollout, "restore_policy_agent", restore)
    monkeypatch.setattr(SimRunner, "make_env", lambda self: Env())
    monkeypatch.setattr(SimRunner, "run_episode", episode)
    monkeypatch.setattr("dexmani_policy.env_runner.sim_runner.save_episode_video", video)
    return state


@pytest.mark.parametrize("episodes", [1, 25])
def test_rollout_saves_twice(experiment, simulation, monkeypatch, episodes):
    root, cfg = experiment
    cfg["eval"]["episodes"] = episodes
    OmegaConf.save(cfg, root / "config.yaml")
    request = config.resolve_evaluation_request(root, "latest")
    writes = []
    original = rollout.write_json

    def write(path, record, **kwargs):
        writes.append((record["status"], len(record["episodes"])))
        return original(path, record, **kwargs)

    monkeypatch.setattr(rollout, "write_json", write)
    result, path = rollout.evaluate_policy(request, output_root=root / "eval")
    assert writes == [("running", 0), ("completed", episodes)]
    assert json.loads(path.read_text()) == result
    assert result["statistics"]["n_valid_episodes"] == episodes
    assert all(env.closed for env in simulation["envs"])
    assert all(row[2:] == (4, 3) for row in simulation["episodes"])
    assert "numerics" not in result["request"]


@pytest.mark.parametrize("fault, error, count", [
    ("load", RuntimeError, 0), ("rollout", RuntimeError, 1),
    ("video", OSError, 1), ("interrupt", KeyboardInterrupt, 1),
    ("cleanup", RuntimeError, 1),
])
def test_failure_keeps_prefix_and_closes(experiment, simulation, fault, error, count, capsys):
    root, cfg = experiment
    simulation["fault"] = fault
    cfg["eval"]["video"]["enabled"] = fault == "video"
    OmegaConf.save(cfg, root / "config.yaml")
    request = config.resolve_evaluation_request(root, "latest")
    with pytest.raises(error):
        rollout.evaluate_policy(request, output_root=root / "eval", output_dir=root / "failed")
    result = json.loads((root / "failed/result_details.json").read_text())
    assert result["status"] == "failed" and result["statistics"] is None
    assert len(result["episodes"]) == count
    assert "Traceback" in result["error"]
    assert all(env.closed for env in simulation["envs"])
    if fault == "cleanup":
        assert "cleanup failure" in capsys.readouterr().err
        assert "rollout failure" in result["error"]


def test_failure_record_io_preserves_original(experiment, simulation, monkeypatch, capsys):
    root, cfg = experiment
    simulation["fault"] = "rollout"
    request = config.resolve_evaluation_request(root, "latest")

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(artifacts, "write_json", fail)
    with pytest.raises(RuntimeError, match="rollout failure"):
        rollout.evaluate_policy(request, output_root=root / "eval")
    assert "disk failure" in capsys.readouterr().err
    assert all(env.closed for env in simulation["envs"])


def test_final_write_failure_is_failed(experiment, simulation, monkeypatch):
    root, cfg = experiment
    request = config.resolve_evaluation_request(root, "latest")
    original = rollout.write_json

    def write(path, record, **kwargs):
        if record["status"] == "completed":
            raise OSError("final write failure")
        return original(path, record, **kwargs)

    monkeypatch.setattr(rollout, "write_json", write)
    with pytest.raises(OSError, match="final write failure"):
        rollout.evaluate_policy(request, output_root=root / "eval", output_dir=root / "failed")
    record = json.loads((root / "failed/result_details.json").read_text())
    assert record["status"] == "failed" and record["statistics"] is None
    assert len(record["episodes"]) == 3


@pytest.mark.parametrize("rows, expected", [
    ([(1., 5, 20), (.5, 1, 100)], 0),
    ([(1., 5, 100), (1., 2, 20)], 1),
    ([(1., 2, 20), (1., 2, 100)], 1),
    ([(0., None, 20), (0., None, 100)], 1),
    ([(1., 2, 20), (1., 2, 20)], 0),
])
def test_rank(rows, expected):
    results = [{"statistics": {"micro_success_rate": rate, "avg_steps": avg}, "global_step": step}
               for rate, avg, step in rows]
    assert selection.rank_selection_results(results) == expected


@pytest.mark.parametrize("success", [True, False])
def test_selection_and_frozen_test(experiment, simulation, monkeypatch, success):
    root, cfg = experiment
    simulation["success"] = success
    original_cfg = copy.deepcopy(cfg)
    writes = []
    original_write = selection.write_json

    def write(path, record, **kwargs):
        writes.append((Path(path).name, record.get("status")))
        return original_write(path, record, **kwargs)

    monkeypatch.setattr(selection, "write_json", write)
    with monkeypatch.context() as context:
        context.setattr(Path, "read_text", lambda *a, **k: pytest.fail("selection read a result"))
        record, checkpoint = selection.select_best_checkpoint(root, cfg)
    assert cfg == original_cfg
    assert record["global_step"] == 100 and checkpoint.is_file()
    assert writes == [("selection_result.json", "running"),
                      ("selection_result.json", "completed"), ("best_ckpt.json", None)]
    chosen_seeds = record["selection_task_seeds"]["test_task"]
    test_seeds = record["test_request"]["task_seeds"]["test_task"]
    assert len(chosen_seeds) == 2 and len(test_seeds) == 3
    assert not set(chosen_seeds) & set(test_seeds)
    for step in (20, 40, 60, 80, 100):
        assert [row[1] for row in simulation["episodes"] if row[0] == step] == chosen_seeds
    for reference in record["candidate_results"]:
        (root / reference).unlink()
    frozen = config.resolve_evaluation_request(
        root, selection_record=root / record["selection_result"],
        overrides={"training": {"device": "cpu"}},
    )
    assert frozen.task_seeds["test_task"] == test_seeds
    assert frozen.source["heldout_from_selection"] and frozen.checkpoint == str(checkpoint)
    result, _ = rollout.evaluate_policy(frozen, output_root=root / "test")
    assert result["status"] == "completed"
    assert result["statistics"]["micro_success_rate"] == float(success)
    assert json.loads((root / "best_ckpt.json").read_text())["global_step"] == 100


def test_selection_failure_preserves_best(experiment, simulation):
    root, cfg = experiment
    best = root / "best_ckpt.json"
    best.write_text("old best")
    simulation["fault"] = "second_load"
    with pytest.raises(RuntimeError, match="second load failure"):
        selection.select_best_checkpoint(root, cfg)
    assert best.read_text() == "old best"
    path, = root.glob("eval_ckpt_selector/*/selection_result.json")
    record = json.loads(path.read_text())
    assert record["status"] == "failed" and len(record["candidate_results"]) == 1
    with pytest.raises(ValueError, match="not completed"):
        config.resolve_evaluation_request(root, selection_record=path)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_pct"])
def test_milestones_checked_before_load(experiment, simulation, mode):
    root, cfg = experiment
    path = root / "checkpoints/epoch=1-step=20-milestone=20pct.pt"
    if mode == "missing":
        path.unlink()
    elif mode == "duplicate":
        (path.parent / "epoch=2-step=20-milestone=20pct.pt").touch()
    else:
        path.rename(path.with_name("epoch=1-step=20-milestone=30pct.pt"))
    with pytest.raises(ValueError, match="requires milestones"):
        selection.select_best_checkpoint(root, cfg)
    assert not simulation["loads"]


def test_pipeline_and_ordinary_best_overrides(experiment, simulation):
    root, cfg = experiment
    path = pipeline.run_pipeline(root, cfg, overrides={"eval": {"inference_steps": 6}})
    result = json.loads(path.read_text())
    assert result["request"]["source"]["heldout_from_selection"]
    assert result["request"]["inference_steps"] == 6
    assert len(simulation["loads"]) == 6
    request = config.resolve_evaluation_request(root)
    assert request.inference_steps == 6
    request = config.resolve_evaluation_request(root, overrides=config.parse_evaluation_overrides([
        "eval.inference_steps=2", "eval.action_steps=1", "eval.use_ema=false",
    ]))
    assert (request.inference_steps, request.action_steps, request.use_ema) == (2, 1, False)
    assert not request.source["heldout_from_selection"]


@pytest.mark.parametrize("override", [
    {"eval": {"action_steps": 8}}, {"eval": {"inference_steps": 0}},
    {"eval": {"use_ema": "false"}}, {"eval": {"task_seeds": {"test_task": [90]}}},
    {"env_runner": {"sensor_modalities": ["rgb"]}},
    {"env_runner": {"env_kwargs": {"control_mode": "ee"}}},
])
def test_input_boundaries(experiment, override):
    root, cfg = experiment
    with pytest.raises(ValueError):
        config.resolve_evaluation_request(root, "latest", overrides=override)


@pytest.mark.parametrize("override", [{"eval": {"inference_steps": 1}}, {"env_runner": {"task_name": "other"}}])
def test_frozen_rejects_research_overrides(experiment, simulation, override):
    root, cfg = experiment
    record, _ = selection.select_best_checkpoint(root, cfg)
    with pytest.raises(ValueError, match="Frozen test permits"):
        config.resolve_evaluation_request(root, overrides=override,
                                          selection_record=root / record["selection_result"])


def test_atomic_write_preserves_previous_json(tmp_path):
    path = tmp_path / "best_ckpt.json"
    artifacts.write_json(path, {"old": True})
    with pytest.raises(ValueError):
        artifacts.write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_frozen_cli_after_move_without_config(experiment, simulation, monkeypatch, tmp_path):
    root, cfg = experiment
    record, checkpoint = selection.select_best_checkpoint(root, cfg)
    (root / "config.yaml").unlink()
    for reference in record["candidate_results"]:
        (root / reference).unlink()
    destination = tmp_path / "experiments/policy/task/run"
    destination.parent.mkdir(parents=True)
    root.rename(destination)
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "set_project_root", lambda: None)

    output = tmp_path / "frozen"
    evaluate.main([
        "--policy-name", "policy", "--task-name", "task", "--exp-name", "run",
        "--selection-record", str(destination / record["selection_result"]),
        "--output-dir", str(output), "training.device=cpu",
    ])
    result = json.loads((output / "result_details.json").read_text())
    request = result["request"]
    assert result["status"] == "completed"
    assert request["checkpoint"] == str(destination / checkpoint.relative_to(root))
    assert request["source"]["selection_result"] == record["selection_result"]
    assert request["task_seeds"] == record["test_request"]["task_seeds"]


def test_best_publication_failure_preserves_pointer(experiment, simulation, monkeypatch):
    root, cfg = experiment
    best = root / "best_ckpt.json"
    best.write_text("old best")
    original = selection.write_json

    def write(path, record, **kwargs):
        if Path(path) == best:
            raise OSError("best write failure")
        return original(path, record, **kwargs)

    monkeypatch.setattr(selection, "write_json", write)
    with pytest.raises(OSError, match="best write failure"):
        selection.select_best_checkpoint(root, cfg)
    assert best.read_text() == "old best"
    path, = root.glob("eval_ckpt_selector/*/selection_result.json")
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "negative_steps", "excess_steps"])
def test_invalid_runner_results_fail(experiment, simulation, monkeypatch, mode):
    root, _ = experiment
    request = config.resolve_evaluation_request(root, "latest")

    def run(self, agent, *, seeds, **kwargs):
        if mode == "missing":
            seeds = seeds[:-1]
        elif mode == "duplicate":
            seeds = seeds + seeds[:1]
        for seed in seeds:
            yield {
                "task_name": self.task_name, "seed": seed, "success": True,
                "steps": 6 if mode == "excess_steps" else 2,
                "total_steps": -1 if mode == "negative_steps" else 5, "video": None,
            }

    monkeypatch.setattr(SimRunner, "run", run)
    with pytest.raises(RuntimeError):
        rollout.evaluate_policy(request, output_root=root / "eval", output_dir=root / "failed")
    result = json.loads((root / "failed/result_details.json").read_text())
    assert result["status"] == "failed" and result["statistics"] is None


@pytest.mark.parametrize("relative", ["../outside.pt", "checkpoints/latest.pt", "config.yaml"])
def test_selected_checkpoint_boundary(experiment, relative):
    root, _ = experiment
    with pytest.raises(ValueError):
        resolve_milestone_path(root, relative)


def test_selected_checkpoint_rejects_symlink(experiment):
    root, _ = experiment
    path = root / "checkpoints/link.pt"
    path.symlink_to(root / "checkpoints/latest.pt")
    with pytest.raises(ValueError, match="immutable milestone"):
        resolve_milestone_path(root, str(path.relative_to(root)))


@pytest.mark.parametrize("total", [1, 2, 3])
def test_small_training_budget(experiment, simulation, total):
    root, cfg = experiment
    for path in (root / "checkpoints").iterdir():
        path.unlink()
    cfg["training"]["loop"]["total_train_steps"] = total
    for step, pct in milestone_steps(total).items():
        (root / "checkpoints" / f"epoch=1-step={step}-milestone={pct}pct.pt").touch()
    record, _ = selection.select_best_checkpoint(root, cfg)
    assert record["global_step"] == total
    assert len(record["candidates"]) == total
