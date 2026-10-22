"""Deterministic sampling and validation of physical task seed plans."""

import random

from dexmani_policy.utils.config import positive_int


def validate_eval_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Evaluation seed must be an integer in [0, 2**32)")
    return seed


def resolve_eval_seed(cfg, cli_seed=None):
    seed = cli_seed if cli_seed is not None else (cfg.get("eval") or {}).get("seed")
    return validate_eval_seed(1024 if seed is None else seed)


def validate_seed_pool(task, pool):
    if any(type(seed) is not int or seed < 0 for seed in pool):
        raise ValueError(f"{task}: seed pool requires nonnegative integer seeds")
    return sorted(set(pool))


def split_eval_seeds(pools, eval_seed, episodes, selection_episodes=0):
    """Fix each task's test prefix before reserving any checkpoint selection seeds."""
    positive_int(episodes, "episodes")
    if type(selection_episodes) is not int or selection_episodes < 0:
        raise ValueError("selection_episodes must be a non-negative integer")
    validate_eval_seed(eval_seed)
    if not pools:
        raise ValueError("Evaluation requires at least one task seed pool")
    test, selection = {}, {}
    for task, pool in pools.items():
        pool = validate_seed_pool(task, pool)
        if len(pool) < episodes + selection_episodes:
            raise ValueError(
                f"{task}: seed pool size={len(pool)}, reserved selection={selection_episodes}, "
                f"available={max(0, len(pool) - selection_episodes)}, requested episodes={episodes}. "
                "Use an explicit --episodes override or provide a larger seed pool."
            )
        random.Random(eval_seed).shuffle(pool)
        test[task] = pool[:episodes]
        selection[task] = pool[episodes : episodes + selection_episodes]
    return test, selection


def selection_seed_counts(initial_episodes, batch_size, max_episodes):
    positive_int(initial_episodes, "initial_episodes")
    positive_int(max_episodes, "max_episodes")
    if type(batch_size) is not int or batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer")
    reserved_count = min(initial_episodes + batch_size, max_episodes)
    initial_count = min(initial_episodes, reserved_count)
    return initial_count, reserved_count - initial_count


def plan_selection_seeds(pools, eval_seed, *, episodes, initial_episodes, batch_size, max_episodes):
    """Fix test, then reserve the initial and possible tie batch before any rollout."""
    initial_count, extra_count = selection_seed_counts(initial_episodes, batch_size, max_episodes)
    test, reserved = split_eval_seeds(pools, eval_seed, episodes, initial_count + extra_count)
    initial = {task: seeds[:initial_count] for task, seeds in reserved.items()}
    extra = {task: seeds[initial_count:] for task, seeds in reserved.items()}
    return test, initial, extra


def plan_size(plan):
    """Equal per-task budgets retain the existing macro/micro comparison."""
    sizes = {len(seeds) for seeds in plan.values()}
    if len(sizes) != 1:
        raise ValueError("Task seed plans require equal per-task episode counts")
    return sizes.pop()


def validate_task_seeds(plan, tasks, *, allow_empty=False):
    """Validate equal per-task budgets and return the episode count."""
    if not isinstance(plan, dict) or set(plan) != set(tasks):
        raise ValueError("Task seed plan does not match expected tasks")
    for task, seeds in plan.items():
        if (
            not isinstance(seeds, list)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"{task}: invalid or duplicate seeds")
    size = plan_size(plan)
    if not allow_empty and size == 0:
        raise ValueError("Task seed plan is empty")
    return size
