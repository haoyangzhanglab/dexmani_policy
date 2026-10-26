"""Evaluation records, runtime settings and selection verification."""

import json
import platform
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.experiments import (
    MILESTONE_PATTERN, milestone_steps, resolve_record_path, validate_best_info,
)
from dexmani_policy.common.runtime import atomic_path
from dexmani_policy.env_runner.common.seeds import validate_seed_pool, validate_task_seeds


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    root = Path(exp_dir).resolve()
    return str(path.relative_to(root))


def validate_runtime(expected, actual):
    for key in ("torch", "cuda", "parameter_dtypes", "numerics"):
        if expected[key] != actual[key]:
            raise ValueError(f"Evaluation runtime differs from selection: {key}")


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
    import torch

    return {
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
    candidates = record["candidates"]
    expected = milestone_steps(protocol["config"]["training"]["loop"]["total_train_steps"])
    steps = [candidate["global_step"] for candidate in candidates]
    if any(type(step) is not int for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("Selection requires distinct integer milestone steps")
    missing = expected.keys() - set(steps)
    if missing:
        raise ValueError(f"Missing selection milestone steps: {sorted(missing)}")
    for candidate in candidates:
        match = MILESTONE_PATTERN.fullmatch(Path(candidate["checkpoint"]).name)
        step = candidate["global_step"]
        if (match is None or int(match["step"]) != step
                or expected.get(step) != int(match["pct"])):
            raise ValueError("Selection milestone filename/step differs from training budget")
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


def read_selection_record(experiment_dir, record_path):
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


def rank_selection_results(record, results):
    """Validate complete candidate evidence, then rank without file or environment I/O."""
    from dexmani_policy.env_runner.common.metrics import (
        compute_evaluation_statistics,
        validate_episodes,
    )

    candidates = record["candidates"]
    if not candidates or len(results) != len(candidates):
        raise ValueError("Selection requires complete results for every declared candidate")
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
    record, checkpoint = read_selection_record(root, record_path)
    results = [json.loads(resolve_record_path(root, ref).read_text()) for ref in record["candidate_results"]]
    best = rank_selection_results(record, results)
    candidate = record["candidates"][best]
    if record["ckpt_relpath"] != candidate["checkpoint"] or record["global_step"] != candidate["global_step"]:
        raise ValueError("Selected checkpoint disagrees with the winning candidate result")
    validate_runtime(record["runtime"], results[0]["runtime"])
    return record, checkpoint
