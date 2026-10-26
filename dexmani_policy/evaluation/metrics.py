import math
from collections import Counter


def compute_evaluation_statistics(episodes):
    tasks: dict[str, list[dict]] = {}
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


def validate_episodes(episodes, plan):
    actual = Counter((row["task_name"], row["seed"]) for row in episodes)
    expected = Counter((task, seed) for task, seeds in plan.items() for seed in seeds)
    if actual != expected:
        raise RuntimeError("Evaluation episodes do not match the requested task/seed plan")
    for row in episodes:
        steps, total = row["steps"], row["total_steps"]
        if total < 0 or (steps is not None and not 0 <= steps <= total):
            raise RuntimeError(f"Invalid evaluation step counts: {row}")
