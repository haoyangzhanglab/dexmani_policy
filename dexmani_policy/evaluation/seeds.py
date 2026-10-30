import random

from dexmani_policy.common.config import positive_int


def validate_evaluation_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Evaluation seed must be an integer in [0, 2**32)")
    return seed


def validate_seed_pool(pool):
    if not isinstance(pool, list):
        raise ValueError("Seeds must be a list")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in pool):
        raise ValueError("Seed pool requires integer seeds in [0, 2**32)")
    if len(set(pool)) != len(pool):
        raise ValueError("Duplicate seeds in source pool")
    return pool


def split_evaluation_seeds(pool, eval_seed, episodes, selection_episodes=0):
    """Fix the test prefix before reserving any checkpoint selection seeds."""
    positive_int(episodes, "episodes")
    if type(selection_episodes) is not int or selection_episodes < 0:
        raise ValueError("selection_episodes must be a non-negative integer")
    validate_evaluation_seed(eval_seed)
    pool = sorted(validate_seed_pool(pool))
    if len(pool) < episodes + selection_episodes:
        raise ValueError(
            f"Seed pool has {len(pool)} seeds; need {episodes} test + "
            f"{selection_episodes} selection seeds"
        )
    random.Random(eval_seed).shuffle(pool)
    return pool[:episodes], pool[episodes : episodes + selection_episodes]
