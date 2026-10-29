"""Validate configs/data or exercise training and checkpoint restoration."""

import argparse
import json
from pathlib import Path

import hydra
import torch

from dexmani_policy.common.config import set_project_root
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.test_tools.smoke_checks import (
    load_smoke_config, smoke_output, validate_data_batch,
    validate_dataset, validate_target_imports, validate_training,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--config-only",
        action="store_true",
        help="Check config interfaces and target imports; model constraints are checked at construction/binding.",
    )
    modes.add_argument(
        "--data-only", action="store_true", help="Check CPU data batches without building a model."
    )
    modes.add_argument(
        "--contracts-only", action="store_true",
        help="Run shared and policy contracts on synthetic data, including bounded training/restoration.",
    )
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra override.")
    parser.add_argument("--device", help="Override training.device for this smoke run.")
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile policy backbones during bounded training; requires training mode.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Keep smoke artifacts in a new directory instead of a temporary directory.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=2,
        help="Data-only batch budget per split (default: 2); does not limit statistics scans.",
    )
    parser.add_argument(
        "config_names", nargs="+", help="Hydra config names or explicit test YAML paths."
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=4,
        help="Initial training-update budget; does not change the saved recipe (default: 4).",
    )
    parser.add_argument(
        "--resume-updates", type=int, default=2,
        help="Additional updates after checkpoint resume; 0 disables this phase (default: 2).",
    )
    parser.add_argument(
        "--rollout", action="store_true", help="Also run one simulation episode using the final checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.compile and (args.config_only or args.data_only or args.contracts_only):
        parser.error("--compile requires bounded training mode")

    set_project_root()
    for config_name in args.config_names:
        print(f"\nValidating {config_name}", flush=True)
        cfg = load_smoke_config(config_name, args.override)
        if args.device is not None:
            cfg.training.device = args.device
        if args.compile:
            cfg.training.use_compile = True
        output_dir = args.output_dir
        if output_dir is not None and len(args.config_names) > 1:
            output_dir = output_dir / cfg.policy_name
        if args.config_only:
            print(f"✓ {config_name}: config and {validate_target_imports(cfg)} Hydra targets PASSED")
            continue

        from dexmani_policy.agents.policies.adaptac import AdapTacPolicy

        policy_class = hydra.utils.get_class(cfg.agent._target_)
        is_adaptac = issubclass(policy_class, AdapTacPolicy)
        if is_adaptac:
            from dexmani_policy.test_tools.adaptac_contracts import (
                validate_adaptac_batch, validate_adaptac_contracts, validate_adaptac_deployment,
            )
        batch_check = validate_adaptac_batch if is_adaptac else validate_data_batch
        status = "PASSED"
        if args.data_only:
            validate_dataset(cfg, max_batches=args.max_batches, batch_check=batch_check)
            summary = "CPU data batches"
        elif args.contracts_only:
            from dexmani_policy.test_tools.common_contracts import validate_common_contracts

            set_seed(cfg.training.seed)
            with smoke_output(output_dir) as directory:
                report = {"common": validate_common_contracts(torch.device(cfg.training.device))}
                if is_adaptac:
                    report.update(validate_adaptac_contracts(cfg))
                from dexmani_policy.agents.policies.afro import AFROPolicy
                from dexmani_policy.agents.policies.cordvip import CordViPPolicy
                from dexmani_policy.agents.policies.sat import SATPolicy

                if issubclass(policy_class, (AFROPolicy, CordViPPolicy)):
                    from dexmani_policy.test_tools.policy_contracts import validate_pretrained_observation_transfer

                    report["pretrained_observations"] = validate_pretrained_observation_transfer(cfg)
                if issubclass(policy_class, SATPolicy):
                    from dexmani_policy.test_tools.policy_contracts import validate_sat_training

                    report["sat"] = validate_sat_training(cfg)
                not_run = [
                    f"{name}: {result['reason']}" for name, result in report["common"].items()
                    if result["status"] == "not_run"
                ]
                if not_run:
                    status = f"COMPLETED (not_run: {', '.join(not_run)})"
                (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            summary = "shared and policy contracts"
        else:
            validate_training(
                cfg, max_updates=args.max_updates, resume_updates=args.resume_updates,
                rollout=args.rollout, output_dir=output_dir, batch_check=batch_check,
                deployment_check=validate_adaptac_deployment if is_adaptac else None,
            )
            summary = "bounded training and checkpoint checks"
        print(f"✓ {config_name}: {summary} {status}")


if __name__ == "__main__":
    main()
