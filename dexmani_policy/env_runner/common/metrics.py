"""Episode completeness and success-rate statistics."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def collect_episode_details(result: dict) -> list[dict]:
    """Flatten validated results without copying episode records."""
    if "episode_details" in result:
        return result["episode_details"]
    return [detail for child in result["per_task"].values() for detail in child["episode_details"]]


def mean_success_steps(details):
    steps = [d["steps"] for d in details if d["success"] and d["steps"] is not None]
    return sum(steps) / len(steps) if steps else None


def summarize_episodes(details: list[dict]) -> dict:
    avg_steps = mean_success_steps(details)
    return {
        "success_rate": sum(detail["success"] for detail in details) / len(details),
        "avg_steps": round(avg_steps) if avg_steps is not None else None,
        "avg_steps_all": round(sum(detail["total_steps"] for detail in details) / len(details)),
    }


def compute_evaluation_statistics(result: dict) -> dict:
    """Micro averages episodes; macro averages tasks; Wilson describes micro uncertainty."""
    per_task = None
    macro = None
    if result.get("per_task"):
        per_task = {}
        for task, child in result["per_task"].items():
            episodes = child["episode_details"]
            n_success = sum(d["success"] for d in episodes)
            n_valid = len(episodes)
            per_task[task] = {
                "success_rate": n_success / n_valid if n_valid else None,
                "n_success": n_success,
                "n_valid": n_valid,
                "wilson_95": wilson_interval(n_success, n_valid),
            }
        rates = [
            task["success_rate"] for task in per_task.values() if task["success_rate"] is not None
        ]
        macro = sum(rates) / len(rates) if rates else None

    if "episode_details" in result or not per_task:
        details = collect_episode_details(result)
        successes = sum(bool(detail["success"]) for detail in details)
        count = len(details)
    else:
        successes = sum(task["n_success"] for task in per_task.values())
        count = sum(task["n_valid"] for task in per_task.values())

    return {
        "micro_success_rate": successes / count if count else 0.0,
        "macro_success_rate": macro,
        "n_success": successes,
        "n_valid_episodes": count,
        "wilson_95": wilson_interval(successes, count),
        "n_tasks": len(per_task) if per_task else 1,
        "per_task": per_task,
    }


def wilson_interval(successes, n):
    """95% binomial Wilson interval; not training-seed or macro uncertainty."""
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def validate_episode_details(details, plan):
    """Require complete task/seed evidence and normalize NumPy booleans for JSON."""
    if not isinstance(details, list):
        raise RuntimeError("Evaluation episode_details must be a list of objects")
    actual = Counter()
    for detail in details:
        if (
            not isinstance(detail, dict)
            or not isinstance(detail.get("task_name"), str)
            or type(detail.get("seed")) is not int
            or not isinstance(detail.get("success"), (bool, np.bool_))
            or "steps" not in detail
            or (
                detail["steps"] is not None
                and (type(detail["steps"]) is not int or detail["steps"] < 0)
            )
            or "error" in detail
            or "error_category" in detail
        ):
            raise RuntimeError(f"Invalid or exceptional evaluation episode: {detail}")
        detail["success"] = bool(detail["success"])
        actual[detail["task_name"], detail["seed"]] += 1
    expected = Counter((task, seed) for task, seeds in plan.items() for seed in seeds)
    if actual != expected:
        raise RuntimeError("Evaluation episodes do not match the requested task/seed plan")
