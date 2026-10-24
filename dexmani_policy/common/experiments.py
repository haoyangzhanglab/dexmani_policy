"""Experiment checkpoints and selection records without loading policy weights."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


def resolve_record_path(root, relative, *, checkpoint=False):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Record requires a nonempty relative file path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Record path must be relative to experiment")
    if checkpoint and (path.name == "latest.pt" or (root / path).is_symlink()):
        raise ValueError("Selection must name an immutable milestone checkpoint")
    resolved = (root / path).resolve(strict=True)
    directory = root / "checkpoints" if checkpoint else root
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError(f"Record path must be a file inside {directory}")
    return resolved


def rank_selection_results(record, results):
    """Validate complete candidate evidence, then rank without file or environment I/O."""
    from dexmani_policy.env_runner.common.metrics import (
        compute_evaluation_statistics,
        validate_episodes,
    )
    from dexmani_policy.env_runner.common.seeds import validate_task_seeds, validate_seed_pool

    candidates = record["candidates"]
    if not candidates or len(results) != len(candidates):
        raise ValueError("Selection requires complete results for every declared candidate")
    if len({c["checkpoint"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate selection candidate")
    protocol = record["test_request"]
    tasks = [task["task_name"] for task in protocol["tasks"]]
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
        for key in ("policy_code", "simulator_code", "torch", "cuda", "parameter_dtypes"):
            if result["runtime"][key] != results[0]["runtime"][key]:
                raise ValueError(f"Candidate runtime conditions differ: {key}")
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


def resolve_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    record_path = Path(record_path).expanduser().resolve(strict=True)
    if not record_path.is_relative_to(root):
        raise ValueError("Selection record must be inside the experiment")
    info = json.loads(record_path.read_text())
    if info["status"] != "completed":
        raise ValueError("Selection has not completed successfully")
    references = info["candidate_results"]
    if len(set(references)) != len(references):
        raise ValueError("Duplicate candidate result reference")
    results = [
        json.loads(resolve_record_path(root, reference).read_text()) for reference in references
    ]
    best = info["candidates"][rank_selection_results(info, results)]
    if info["ckpt_relpath"] != best["checkpoint"] or info["global_step"] != best["global_step"]:
        raise ValueError("Selected checkpoint disagrees with complete candidate ranking")
    request = info["test_request"]
    if request["checkpoint"] != str(Path(info["experiment_dir"]) / best["checkpoint"]):
        raise ValueError("Frozen test checkpoint differs from selection")
    if info["inference"] != {
        key: request[key] for key in ("use_ema", "inference_steps", "action_steps")
    }:
        raise ValueError("Selected inference settings differ from frozen test")
    checkpoint = resolve_record_path(root, info["ckpt_relpath"], checkpoint=True)
    info["selection_result"] = str(record_path.relative_to(root))
    return info, checkpoint


def resolve_best_checkpoint(experiment_dir):
    root = Path(experiment_dir).resolve()
    pointer = json.loads((root / "best_ckpt.json").read_text())
    record = resolve_record_path(root, pointer["selection_result"])
    return resolve_selection_record(root, record)


def resolve_checkpoint_path(exp_dir, selector="best"):
    """Resolve a tag or path to a file inside the experiment/checkpoints directory."""
    root = Path(exp_dir).resolve()
    if selector == "best":
        return resolve_best_checkpoint(root)[1]
    if selector.endswith("pct"):
        milestones = discover_milestone_checkpoints(root)
        target_pct = int(selector.removesuffix("pct"))
        for milestone in milestones:
            if milestone.pct == target_pct:
                return milestone.path
        available = sorted(m.pct for m in milestones)
        raise FileNotFoundError(f"No {target_pct}% milestone checkpoint. Available: {available}")
    if selector == "latest":
        path = root / "checkpoints/latest.pt"
    else:
        path = Path(selector)
        if not path.is_absolute():
            path = root / "checkpoints" / path
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root / "checkpoints"):
        raise ValueError("Checkpoint must be inside the selected experiment/checkpoints")
    return path


MILESTONE_PATTERN = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    path: Path
    pct: int
    global_step: int

    @property
    def label(self) -> str:
        return f"{self.pct}% (step={self.global_step})"


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    """Find regular milestone files inside this experiment, sorted by pct."""
    ckpt_dir = exp_dir / "checkpoints"
    found: list[MilestoneCheckpoint] = []
    for pt_file in ckpt_dir.glob("epoch=*.pt"):
        match = MILESTONE_PATTERN.fullmatch(pt_file.name)
        if not match:
            continue
        if pt_file.is_symlink():
            raise ValueError(f"Milestone checkpoint must not be a symlink: {pt_file}")
        found.append(
            MilestoneCheckpoint(
                path=resolve_checkpoint_path(exp_dir, pt_file.name),
                pct=int(match.group("pct")),
                global_step=int(match.group("step")),
            )
        )

    if not found:
        raise FileNotFoundError(f"No milestone checkpoints found in {ckpt_dir}")

    return sorted(found, key=lambda checkpoint: (checkpoint.pct, checkpoint.path.name))
