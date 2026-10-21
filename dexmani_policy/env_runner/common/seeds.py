"""Deterministic sampling and validation of physical task seed plans."""

import random

from dexmani_policy.utils.config import positive_int


def select_eval_seeds(pools, eval_seed, episodes, excluded_seeds=None):
    """Shuffle each task's pool and take equal budgets, excluding selection seeds."""
    positive_int(episodes, "episodes")
    if not pools:
        raise ValueError("Evaluation requires at least one task seed pool")
    excluded = {} if excluded_seeds is None else excluded_seeds
    if not isinstance(excluded, dict):
        raise ValueError("Excluded selection seeds must be a task-to-seeds mapping")
    if excluded:
        validate_task_seeds(excluded, pools, equal_budget=False)
    plan = {}
    for task, pool in pools.items():
        if any(type(seed) is not int or seed < 0 for seed in pool):
            raise ValueError(f"{task}: seed pool requires nonnegative integer seeds")
        removed = set(excluded.get(task, []))
        pool = [seed for seed in dict.fromkeys(pool) if seed not in removed]
        if len(pool) < episodes:
            raise ValueError(
                f"{task}: requested {episodes} evaluation seeds, only {len(pool)} remain "
                "after excluding checkpoint-selection seeds"
            )
        random.Random(eval_seed).shuffle(pool)
        plan[task] = pool[:episodes]
    return plan


def plan_size(plan):
    """Equal per-task budgets retain the existing macro/micro comparison."""
    sizes = {len(seeds) for seeds in plan.values()}
    if len(sizes) != 1:
        raise ValueError("Task seed plans require equal per-task episode counts")
    return sizes.pop()


def validate_task_seeds(plan, tasks, *, equal_budget=True):
    if not isinstance(plan, dict) or set(plan) != set(tasks):
        raise ValueError("Task seed plan does not match expected tasks")
    for task, seeds in plan.items():
        if (
            not isinstance(seeds, list)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"{task}: invalid or duplicate seeds")
    if equal_budget and plan_size(plan) == 0:
        raise ValueError("Task seed plan is empty")
