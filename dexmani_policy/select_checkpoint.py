"""Selection-only CLI retained until external evaluation consumers migrate."""

from dexmani_policy.eval_pipeline import build_parser
from dexmani_policy.evaluation.selection import run_pipeline


def main(argv=None):
    run_pipeline(**vars(build_parser().parse_args(argv)), selection_only=True)


if __name__ == "__main__":
    main()
