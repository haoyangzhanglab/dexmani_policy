"""Evaluate one explicit policy checkpoint."""

import argparse

from dexmani_policy.evaluation.rollout import evaluate_policy


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
    evaluate_policy(**vars(args))


if __name__ == "__main__":
    main()
