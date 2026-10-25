"""Atomic evaluation records and concrete code/runtime provenance."""

import json
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.common.experiments import resolve_record_path, validate_best_info
from dexmani_policy.env_runner.common.seeds import validate_task_seeds, validate_seed_pool


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    root = Path(exp_dir).resolve()
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def code_version(directory):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def runtime_provenance():
    import importlib.util
    import torch

    sim = importlib.util.find_spec("dexmani_sim")
    return {
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": (
            code_version(Path(sim.origin).parent) if sim is not None and sim.origin else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def write_json(path, record, *, overwrite=True):
    with atomic_path(Path(path), overwrite=overwrite) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


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


def validate_selection_plan(record):
    protocol = record["test_request"]
    tasks = [task["task_name"] for task in protocol["tasks"]]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Selection requires distinct tasks")
    selection, test = record["selection_task_seeds"], protocol["task_seeds"]
    if validate_task_seeds(selection, tasks) != protocol["config"]["eval"]["selection_episodes"]:
        raise ValueError("Selection plan differs from declared fixed budget")
    if validate_task_seeds(test, tasks) != protocol["config"]["eval"]["episodes"]:
        raise ValueError("Frozen test plan differs from declared budget")
    if set(record["seed_pools"]) != set(tasks):
        raise ValueError("Selection source pools differ from tasks")
    for task in tasks:
        pool = set(validate_seed_pool(task, record["seed_pools"][task]))
        if set(selection[task]) & set(test[task]) or not set(selection[task] + test[task]) <= pool:
            raise ValueError("Test and selection seeds must be disjoint and inside the source pool")


def resolve_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    path = Path(record_path).expanduser().resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Selection record must be a file inside the experiment")
    info = json.loads(path.read_text())
    if info["status"] != "completed":
        raise ValueError("Selection has not completed successfully")
    validate_best_info(info)
    validate_selection_plan(info)
    candidates, references = info["candidates"], info["candidate_results"]
    if not candidates or len(references) != len(candidates) or len(set(references)) != len(references):
        raise ValueError("Selection requires distinct references for every candidate")
    if len({candidate["checkpoint"] for candidate in candidates}) != len(candidates):
        raise ValueError("Duplicate selection candidate")
    selected = {"checkpoint": info["ckpt_relpath"], "global_step": info["global_step"]}
    if selected not in candidates:
        raise ValueError("Selected checkpoint is not a declared candidate")
    request = info["test_request"]
    if request["checkpoint"] != str(Path(info["experiment_dir"]) / info["ckpt_relpath"]):
        raise ValueError("Frozen test checkpoint differs from selection")
    if info["inference"] != {
        key: request[key] for key in ("use_ema", "inference_steps", "action_steps")
    }:
        raise ValueError("Selected inference settings differ from frozen test")
    config = request["config"]
    action_steps = config["eval"].get("action_steps")
    expected = {
        "use_ema": config["eval"]["use_ema"],
        "inference_steps": config["eval"]["inference_steps"],
        "action_steps": config["agent"]["n_action_steps"] if action_steps is None else action_steps,
    }
    if info["inference"] != expected:
        raise ValueError("Frozen test inference differs from its saved config")
    if request["action_steps"] > config["agent"]["horizon"] - config["policy_metadata"]["action_start_index"]:
        raise ValueError("Frozen test action_steps exceeds the prediction window")
    if request["source"].get("phase") != "test" or request["source"].get("heldout_from_selection") is not True:
        raise ValueError("Selection requires a frozen held-out test request")
    checkpoint = resolve_record_path(root, info["ckpt_relpath"], checkpoint=True)
    info["selection_result"] = str(path.relative_to(root))
    return info, checkpoint
