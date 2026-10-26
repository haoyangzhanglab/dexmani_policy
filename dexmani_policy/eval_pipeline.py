"""Select a checkpoint and execute its frozen test protocol."""

from pathlib import Path

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.evaluation.config import (
    build_parser,
    frozen_test_request,
    parse_evaluation_overrides,
)
from dexmani_policy.evaluation.rollout import evaluate_policy
from dexmani_policy.evaluation.selection import select_best_checkpoint


def run_pipeline(exp_dir, cfg, *, overrides=None) -> Path:
    record, checkpoint = select_best_checkpoint(exp_dir, cfg, overrides=overrides)
    request = frozen_test_request(record, checkpoint)
    _, result_path = evaluate_policy(request, output_root=Path(exp_dir) / "eval_dexsim")
    return result_path


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
