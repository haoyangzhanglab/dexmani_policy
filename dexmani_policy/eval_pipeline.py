"""Select a checkpoint and execute its frozen test protocol."""

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluate import build_parser
from dexmani_policy.evaluation.config import parse_evaluation_overrides
from dexmani_policy.evaluation.pipeline import run_pipeline


def main(argv=None):
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        parser.error(
            "Pipeline uses independent selection/test run directories; use the individual entries for --output-dir"
        )
    set_project_root()
    register_resolvers()
    exp_dir = PROJECT_ROOT / "experiments" / args.policy_name / args.task_name / args.exp_name
    run_pipeline(
        exp_dir,
        load_experiment_config(exp_dir),
        overrides=parse_evaluation_overrides(args.overrides),
    )


if __name__ == "__main__":
    main()
