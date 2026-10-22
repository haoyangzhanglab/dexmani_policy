"""Freeze test seeds, select a checkpoint, then evaluate the saved plan without video."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import (
    load_experiment_config,
    register_resolvers,
    set_project_root,
)
from dexmani_policy.env_runner.evaluation import (
    add_evaluation_arguments,
    evaluate_checkpoint,
    parse_eval_overrides,
)
from dexmani_policy.env_runner.selection import select_best_checkpoint

ROOT_DIR = set_project_root()
register_resolvers()


def run_pipeline(
    exp_dir,
    cfg,
    *,
    episodes=None,
    eval_seed=None,
    inference_steps=None,
    action_steps=None,
    use_ema=None,
    overrides=(),
):
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(cfg), resolve=True))
    selection_cfg = OmegaConf.merge(cfg, parse_eval_overrides(list(overrides)))
    if episodes is not None:
        OmegaConf.update(selection_cfg, "eval.offline.episodes", episodes)
    record = select_best_checkpoint(
        exp_dir,
        selection_cfg,
        inference_steps=inference_steps,
        action_steps=action_steps,
        use_ema=use_ema,
        eval_seed=eval_seed,
        record_video=False,
    )
    return evaluate_checkpoint(
        exp_dir,
        cfg,
        selection_record=record,
        action_steps=action_steps,
        inference_steps_list=[inference_steps] if inference_steps is not None else None,
        use_ema=use_ema,
        overrides=overrides,
        episodes=episodes,
        eval_seed=eval_seed,
        record_video=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        help="Test episodes per task, reserved before selection; otherwise saved config.",
    )
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    args = parser.parse_args(argv)
    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    cfg = load_experiment_config(exp_dir)
    run_pipeline(
        exp_dir,
        cfg,
        episodes=args.episodes,
        eval_seed=args.seed,
        inference_steps=args.inference_steps,
        action_steps=args.action_steps,
        overrides=args.overrides,
        use_ema=args.use_ema,
    )


if __name__ == "__main__":
    main()
