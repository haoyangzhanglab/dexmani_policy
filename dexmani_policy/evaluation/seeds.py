import random

from dexmani_policy.common.config import positive_int


def validate_evaluation_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Evaluation seed must be an integer in [0, 2**32)")
    return seed


def validate_seed_pool(task, pool):
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in pool):
        raise ValueError(f"{task}: seed pool requires nonnegative integer seeds")
    if len(set(pool)) != len(pool):
        raise ValueError(f"{task}: duplicate seeds in source pool")
    return pool


def split_evaluation_seeds(pools, eval_seed, episodes, selection_episodes=0):
    """Fix each task's test prefix before reserving any checkpoint selection seeds."""
    positive_int(episodes, "episodes")
    if type(selection_episodes) is not int or selection_episodes < 0:
        raise ValueError("selection_episodes must be a non-negative integer")
    validate_evaluation_seed(eval_seed)
    if not pools:
        raise ValueError("Evaluation requires at least one task seed pool")

    test, selection = {}, {}
    for task, pool in pools.items():
        pool = sorted(validate_seed_pool(task, pool))
        if len(pool) < episodes + selection_episodes:
            raise ValueError(
                f"{task}: seed pool size={len(pool)}, reserved selection={selection_episodes}, "
                f"available={max(0, len(pool) - selection_episodes)}, requested episodes={episodes}. "
                "Set eval.episodes explicitly or provide a larger seed pool."
            )
        random.Random(eval_seed).shuffle(pool)
        test[task] = pool[:episodes]
        selection[task] = pool[episodes : episodes + selection_episodes]
    return test, selection


def validate_task_seeds(plan, tasks) -> None:
    if not isinstance(plan, dict) or set(plan) != set(tasks):
        raise ValueError("Task seed plan does not match expected tasks")
    for task, seeds in plan.items():
        if not isinstance(seeds, list):
            raise ValueError(f"{task}: seeds must be a list")
        validate_seed_pool(task, seeds)

    sizes = {len(seeds) for seeds in plan.values()}
    if len(sizes) != 1:
        raise ValueError("Task seed plans require equal per-task episode counts")
    if sizes == {0}:
        raise ValueError("Task seed plan is empty")
