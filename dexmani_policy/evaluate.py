"""Evaluate one explicit policy checkpoint."""

import argparse

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.evaluation.rollout import evaluate_policy, load_seed_pool
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"))
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    episodes = options.pop("episodes")
    if args.seeds is None:
        cfg = load_checkpoint(args.checkpoint).effective_config
        options["seeds"], _ = split_evaluation_seeds(load_seed_pool(cfg), args.seed, episodes)
    evaluate_policy(**options)


if __name__ == "__main__":
    main()
