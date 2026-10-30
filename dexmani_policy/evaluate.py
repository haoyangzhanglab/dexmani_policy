"""Evaluate one resolved checkpoint or a frozen selection record."""

from dexmani_policy.common.config import (
    register_resolvers,
    set_project_root,
)
from dexmani_policy.common.experiments import resolve_experiment
from dexmani_policy.evaluation.protocol import (
    build_parser,
    load_eval_protocol,
    parse_overrides,
)
from dexmani_policy.evaluation.rollout import evaluate_policy


def main(argv=None):
    parser = build_parser(__doc__)
    parser.add_argument("--output-dir")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint",
        default="best",
        help="best, latest, percentage tag or concrete checkpoint path",
    )
    source.add_argument(
        "--selection-record", help="Completed selection result; freezes all research settings"
    )
    args = parser.parse_args(argv)
    set_project_root()
    register_resolvers()
    exp_dir = resolve_experiment(args.experiment)
    protocol = load_eval_protocol(
        exp_dir,
        args.checkpoint,
        parse_overrides(args.overrides),
        selection_record=args.selection_record,
    )
    evaluate_policy(protocol, output_root=exp_dir / "eval_dexsim", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
