"""Select a checkpoint using the same fixed budget for every candidate."""

from dexmani_policy.common.config import (
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.common.experiments import resolve_experiment
from dexmani_policy.evaluation.protocol import build_parser, parse_overrides
from dexmani_policy.evaluation.selection import select_checkpoint


def main(argv=None):
    parser = build_parser(__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    set_project_root()
    register_resolvers()
    exp_dir = resolve_experiment(args.experiment)
    select_checkpoint(
        exp_dir,
        load_experiment_config(exp_dir),
        overrides=parse_overrides(args.overrides),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
