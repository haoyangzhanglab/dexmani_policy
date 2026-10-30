import math


def summarize_episodes(episodes):
    count = len(episodes)
    successes = sum(row["success"] for row in episodes)
    steps = [row["steps"] for row in episodes if row["success"] and row["steps"] is not None]
    return {
        "success_rate": successes / count,
        "avg_steps": sum(steps) / len(steps) if steps else None,
        "avg_steps_all": sum(row["total_steps"] for row in episodes) / count,
        "n_success": successes,
        "n_valid_episodes": count,
        "wilson_95": wilson_interval(successes, count),
    }


def wilson_interval(successes, n):
    """95% binomial Wilson interval; not training-seed uncertainty."""
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]
