"""Evaluation provenance, atomic artifacts and selection evidence."""

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.env_runner.common.metrics import validate_episode_details
from dexmani_policy.env_runner.common.seeds import (
    selection_seed_counts,
    validate_seed_pool,
    validate_task_seeds,
)


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    root = Path(exp_dir).resolve()
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def code_version(directory):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}


def evaluation_snapshot(cfg, runners, **request):
    """Capture the actual environment, inputs and request once per run."""
    sim = {"commit": "unknown", "dirty": "unknown", "version": "unknown"}
    try:
        spec = importlib.util.find_spec("dexmani_sim")
        if spec is not None and spec.origin:
            sim.update(code_version(Path(spec.origin).parent))
        sim["version"] = importlib.metadata.version("dexmani_sim")
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        pass
    inputs = {
        leaf.task_name: {
            "n_obs_steps": leaf.n_obs_steps,
            "sensor_modalities": list(leaf.sensor_modalities),
            "task_text": leaf.task_text,
            "rgb_preprocessing": leaf.rgb_preprocessing,
            "record_video": leaf.record_video,
            "render_mode": leaf.render_mode,
            "viewer_resolution": leaf.viewer_resolution,
            "env_video_fps": leaf.env_video_fps,
            "env_kwargs": leaf.expand_env_kwargs(leaf.env_kwargs),
        }
        for leaf in runners
    }
    snapshot = {
        "model_config_source": "config.yaml",
        "effective_config": cfg,
        "runner_inputs": inputs,
        "request": request,
        "argv": list(sys.argv),
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": sim,
    }
    return OmegaConf.to_container(OmegaConf.create(snapshot), resolve=True)


def save_evaluation_snapshot(directory, cfg, runners, **request):
    snapshot = evaluation_snapshot(cfg, runners, **request)
    directory = Path(directory)
    with atomic_path(directory / "eval_config.yaml", overwrite=False) as temporary:
        temporary.write_text(OmegaConf.to_yaml(OmegaConf.create(snapshot), resolve=True))
    return artifact_reference(directory / "eval_config.yaml", cfg.experiment_dir)


def write_json(path, record, *, overwrite=True):
    with atomic_path(path, overwrite=overwrite) as temporary:
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)


def create_run_directory(root, directory=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    return directory


def validate_selection_evidence(record, pools, episodes):
    """Validate the frozen test/reserved plans and every candidate's stage evidence."""
    request = record["evaluation"]["request"]
    stages = record["stages"]
    if not any(stage["checkpoint"] == record["ckpt_relpath"] for stage in stages):
        raise ValueError("Selected checkpoint has no evaluation evidence")

    test = request["test_task_seeds"]
    initial = request["initial_task_seeds"]
    extra = request["possible_tie_task_seeds"]
    if validate_task_seeds(test, pools) != episodes:
        raise ValueError("Fixed test seed counts disagree with saved eval.offline.episodes")
    initial_count, extra_count = selection_seed_counts(
        request["initial_episodes"], request["batch_size"], request["max_episodes"]
    )
    if (
        validate_task_seeds(initial, pools) != initial_count
        or validate_task_seeds(extra, pools, allow_empty=True) != extra_count
    ):
        raise ValueError("Reserved selection seed counts disagree with its budgets")
    reserved = {}
    for task in initial:
        test_seeds, initial_seeds, extra_seeds = map(set, (test[task], initial[task], extra[task]))
        if initial_seeds & extra_seeds:
            raise ValueError(f"{task}: initial and possible tie seeds overlap")
        reserved[task] = initial_seeds | extra_seeds
        pool = set(validate_seed_pool(task, pools[task]))
        if not reserved[task] <= pool:
            raise ValueError(f"{task}: reserved selection seeds are outside the task seed pool")
        if not test_seeds <= pool:
            raise ValueError(f"{task}: fixed test seeds are outside the task seed pool")
        if test_seeds & reserved[task]:
            raise ValueError(f"{task}: fixed test and reserved selection seeds overlap")

    for stage in stages:
        if stage.get("status") != "completed":
            raise ValueError("Selection contains an incomplete stage")
        plan = stage.get("requested_task_seeds")
        validate_task_seeds(plan, pools)
        validate_episode_details(stage.get("episode_details"), plan)
        for task, seeds in plan.items():
            if not set(seeds) <= reserved[task]:
                raise ValueError(f"{task}: selection stage seeds are outside the reserved plan")
    return test
