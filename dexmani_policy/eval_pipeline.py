"""Select among multiple checkpoints before independent testing."""

import argparse

from dexmani_policy.evaluation.selection import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Two or more distinct checkpoint paths")
    parser.add_argument("--weights", choices=("raw", "ema"))
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--selection-episodes", type=int, default=25)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-intermediate-observations", action="store_true",
                        help="Only read frames used for inference (built-in simulation, without video)")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
