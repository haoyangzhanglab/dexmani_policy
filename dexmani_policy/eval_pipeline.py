"""Select a checkpoint and execute its frozen test protocol."""

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluation.config import (
    build_parser,
    parse_evaluation_overrides,
)
from dexmani_policy.evaluation.selection import run_pipeline


def main(argv=None):
    parser = build_parser(__doc__)
    args = parser.parse_args(argv)
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
