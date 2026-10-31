"""Select explicit checkpoint candidates, then test the winner on independent seeds."""

import argparse

from dexmani_policy.evaluation.selection import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"))
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--selection-episodes", type=int, default=25)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
