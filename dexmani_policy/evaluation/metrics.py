import math
from collections import Counter


def compute_evaluation_statistics(episodes):
    count = len(episodes)
    successes = sum(row["success"] for row in episodes)
    steps = [row["steps"] for row in episodes if row["success"] and row["steps"] is not None]
    return {
        "success_rate": successes / count if count else None,
        "avg_steps": sum(steps) / len(steps) if steps else None,
        "avg_steps_all": sum(row["total_steps"] for row in episodes) / count if count else None,
        "n_success": successes,
        "n_valid_episodes": count,
        "wilson_95": wilson_interval(successes, count),
    }


def wilson_interval(successes, n):
    """95% binomial Wilson interval; not training-seed uncertainty."""
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def validate_episodes(episodes, task_name, seeds):
    actual = Counter((row["task_name"], row["seed"]) for row in episodes)
    expected = Counter((task_name, seed) for seed in seeds)
    if actual != expected:
        raise RuntimeError("Evaluation episodes do not match the requested task/seed plan")
    for row in episodes:
        steps, total = row["steps"], row["total_steps"]
        if total < 0 or (steps is not None and not 0 <= steps <= total):
            raise RuntimeError(f"Invalid evaluation step counts: {row}")
