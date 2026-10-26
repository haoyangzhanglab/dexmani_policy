"""Atomic evaluation records and concrete code/runtime provenance."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.experiments import file_sha256, resolve_record_path, validate_best_info
from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.env_runner.common.seeds import validate_seed_pool, validate_task_seeds


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    root = Path(exp_dir).resolve()
    return str(path.relative_to(root))


def code_version(directory):
    root = Path(directory).resolve(strict=True)
    # Hash research sources, not data/assets, generated runs or symlinked storage.
    suffixes = {".py", ".yaml", ".yml", ".json", ".sh", ".toml", ".txt"}
    files = [p for p in root.iterdir() if p.is_file() and not p.is_symlink()
             and (p.suffix in {".py", ".sh"} or p.name == "pyproject.toml"
                  or p.name.startswith("requirements") and p.suffix == ".txt")]
    for name in ("dexmani_policy", "dexmani_sim", "configs", "scripts"):
        base = root / name
        if not base.is_dir() or base.is_symlink():
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                             and d not in {"__pycache__", "assets", "data", "robot_data",
                                           "logs", "outputs", "experiments"}
                             and not (Path(directory) / d).is_symlink())
            files.extend(Path(directory) / n for n in names
                         if Path(n).suffix in suffixes and not (Path(directory) / n).is_symlink())
    if not files:
        raise ValueError(f"No research sources found for code identity: {root}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    version: dict[str, str | bool | None] = {
        "commit": None, "dirty": None, "content_sha256": digest.hexdigest(),
    }
    try:
        git_root = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if Path(git_root).resolve() == root:
            version["commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
            version["dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
            ))
    except (OSError, subprocess.CalledProcessError):
        pass  # Source snapshots legitimately have no Git metadata; content is still identified.
    return version


def runtime_contract(runtime):
    identity = {}
    for key in ("policy_code", "simulator_code"):
        content = (runtime.get(key) or {}).get("content_sha256")
        if not isinstance(content, str) or len(content) != 64:
            raise ValueError(f"Formal evaluation requires {key} content identity; rerun selection")
        identity[key] = content
    return {**identity, **{key: runtime[key] for key in
                          ("torch", "cuda", "parameter_dtypes", "numerics")}}


def validate_runtime(expected, actual):
    if runtime_contract(expected) != runtime_contract(actual):
        raise ValueError("Runtime code, Torch/CUDA, parameter dtype or numerics differ from selection")


def training_source_evidence(config, plans):
    """Report comparable task/seed facts without upgrading them to scene identities."""
    overlaps: dict[str, dict[str, set[int]]] = {}
    for recipe in config.get('data_recipe', ()):
        evidence = recipe.get('source_evidence', {})
        task, seeds = evidence.get('task_name'), evidence.get('episode_seeds')
        if task is None or seeds is None:
            continue
        train = {seeds[i] for i in recipe['split_manifest']['train_indices']}
        for phase, plan in plans.items():
            if task in plan:
                overlaps.setdefault(phase, {}).setdefault(task, set()).update(train & set(plan[task]))
    return {
        'training_test_independence': 'unverified',
        'reason': 'No comparable generation/source/trial identity links training to evaluation. '
                  'Equal or different integer seeds alone do not prove scene overlap or independence.',
        'same_task_seed_overlap': {
            phase: {task: sorted(seeds) for task, seeds in tasks.items()}
            for phase, tasks in overlaps.items()
        },
    }


def runtime_provenance():
    import importlib.util
    import torch

    sim = importlib.util.find_spec("dexmani_sim")
    return {
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": (
            code_version(Path(sim.origin).parent.parent) if sim is not None and sim.origin else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numerics": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def write_json(path, record, *, overwrite=True):
    with atomic_path(Path(path), overwrite=overwrite) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def write_failure(path, record):
    """Record an active failure without replacing it with a secondary I/O error."""
    try:
        write_json(path, record)
    except BaseException:
        traceback.print_exc()


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


def _read_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    path = Path(record_path).expanduser().resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Selection record must be a file inside the experiment")
    info = json.loads(path.read_text())
    if info["status"] != "completed":
        raise ValueError("Selection has not completed successfully")
    validate_best_info(info)
    candidates, references = info["candidates"], info["candidate_results"]
    if not candidates or len(references) != len(candidates) or len(set(references)) != len(references):
        raise ValueError("Selection requires distinct references for every candidate")
    selected = {"checkpoint": info["ckpt_relpath"], "global_step": info["global_step"],
                "checkpoint_sha256": info["checkpoint_sha256"]}
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


def rank_selection_results(record, results):
    """Validate complete candidate evidence, then rank without file or environment I/O."""
    from dexmani_policy.env_runner.common.metrics import (
        compute_evaluation_statistics,
        validate_episodes,
    )

    candidates = record["candidates"]
    if not candidates or len(results) != len(candidates):
        raise ValueError("Selection requires complete results for every declared candidate")
    if len({c["checkpoint"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate selection candidate")
    validate_selection_plan(record)
    protocol = record["test_request"]
    selection = record["selection_task_seeds"]
    ranks = []
    for candidate, result in zip(candidates, results, strict=True):
        if result["status"] != "completed" or result["global_step"] != candidate["global_step"]:
            raise ValueError("Candidate is incomplete or global_step differs")
        request = result["request"]
        expected_checkpoint = str(Path(record["experiment_dir"]) / candidate["checkpoint"])
        if request["checkpoint"] != expected_checkpoint or request["task_seeds"] != selection:
            raise ValueError("Candidate checkpoint or fixed selection plan differs")
        for key in ("config", "tasks", "use_ema", "inference_steps", "action_steps", "numerics"):
            if request[key] != protocol[key]:
                raise ValueError(f"Candidate research conditions differ: {key}")
        if result["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
            raise ValueError("Candidate checkpoint content differs from frozen identity")
        if request["source"].get("phase") != "selection":
            raise ValueError("Candidate must contain selection episode facts")
        validate_runtime(results[0]["runtime"], result["runtime"])
        validate_episodes(result["episodes"], selection)
        stats = compute_evaluation_statistics(result["episodes"])
        if result["statistics"] != stats:
            raise ValueError("Candidate statistics disagree with episode facts")
        avg = stats["avg_steps"]
        ranks.append(
            (
                stats["micro_success_rate"],
                -(avg if avg is not None else float("inf")),
                candidate["global_step"],
            )
        )
    return max(range(len(candidates)), key=ranks.__getitem__)


def verify_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    record, checkpoint = _read_selection_record(root, record_path)
    results = [json.loads(resolve_record_path(root, ref).read_text()) for ref in record["candidate_results"]]
    for candidate in record["candidates"]:
        candidate_path = resolve_record_path(root, candidate["checkpoint"], checkpoint=True)
        if file_sha256(candidate_path) != candidate["checkpoint_sha256"]:
            raise ValueError("Checkpoint content differs from selection identity")
    best = record["candidates"][rank_selection_results(record, results)]
    if (record["ckpt_relpath"], record["global_step"]) != (best["checkpoint"], best["global_step"]):
        raise ValueError("Selected checkpoint disagrees with complete candidate ranking")
    validate_runtime(record["runtime"], results[0]["runtime"])
    return record, checkpoint
