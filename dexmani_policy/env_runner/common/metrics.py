"""Flat episode completeness and shared evaluation statistics."""

import math
from collections import Counter

import numpy as np


def compute_evaluation_statistics(episodes):
    tasks = {}
    for episode in episodes:
        tasks.setdefault(episode["task_name"], []).append(episode)
    per_task = {}
    for task, rows in tasks.items():
        successes = sum(row["success"] for row in rows)
        steps = [row["steps"] for row in rows if row["success"] and row["steps"] is not None]
        per_task[task] = {
            "n_success": successes,
            "n_valid": len(rows),
            "success_rate": successes / len(rows),
            "avg_steps": sum(steps) / len(steps) if steps else None,
            "wilson_95": wilson_interval(successes, len(rows)),
        }
    count = len(episodes)
    successes = sum(row["success"] for row in episodes)
    steps = [row["steps"] for row in episodes if row["success"] and row["steps"] is not None]
    return {
        "micro_success_rate": successes / count if count else None,
        "macro_success_rate": (
            sum(t["success_rate"] for t in per_task.values()) / len(tasks) if tasks else None
        ),
        "avg_steps": sum(steps) / len(steps) if steps else None,
        "avg_steps_all": sum(row["total_steps"] for row in episodes) / count if count else None,
        "n_success": successes,
        "n_valid_episodes": count,
        "n_tasks": len(tasks),
        "wilson_95": wilson_interval(successes, count),
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


def validate_episodes(details, plan):
    """Require complete task/seed evidence and normalize NumPy booleans for JSON."""
    if not isinstance(details, list):
        raise RuntimeError("Evaluation episodes must be a list of objects")
    actual = Counter()
    for detail in details:
        if (
            not isinstance(detail, dict)
            or not isinstance(detail.get("task_name"), str)
            or type(detail.get("seed")) is not int
            or not isinstance(detail.get("success"), (bool, np.bool_))
            or type(detail.get("total_steps")) is not int
            or detail["total_steps"] < 0
            or "steps" not in detail
            or (
                detail["steps"] is not None
                and (
                    type(detail["steps"]) is not int
                    or not 0 <= detail["steps"] <= detail["total_steps"]
                )
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
