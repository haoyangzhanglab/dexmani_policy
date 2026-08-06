import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import set_project_root, set_seed
from dexmani_policy.training.eval_utils import (
    build_eval_components,
    read_best_ckpt_json,
    resolve_best_checkpoint,
    validate_eval_config,
)
from dexmani_policy.training.sim_evaluator import SimEvaluator

ROOT_DIR = set_project_root()

register_resolvers()


def run_eval(exp_dir: Path, overrides: list[str]):
    exp_dir = exp_dir.expanduser().resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Can't find config.yaml: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Backward compat: legacy configs use action_mode → action_key
    validate_eval_config(cfg)
    # Stash exp_dir so build_eval_components can infer checkpoint paths
    cfg._exp_dir = str(exp_dir)

    if not hasattr(cfg, "eval") or not hasattr(cfg.eval, "offline"):
        raise KeyError(
            "Config is missing 'eval.offline' section. "
            "Please add eval.offline with keys: ckpt_tag_or_path, eval_episodes, "
            "denoise_timesteps_list, use_ema_for_eval."
        )

    # Standardized seed source (C6 fix): eval.seed → training.seed fallback
    eval_seed = (
        cfg.eval.get("seed")
        if hasattr(cfg, "eval") and cfg.eval.get("seed") is not None
        else cfg.training.get("seed", 0)
    )
    set_seed(eval_seed)

    device = torch.device(cfg.training.device)
    agent, env_runner, checkpoint_store = build_eval_components(cfg, device)
    # Explicit seed control for reproducibility (I1 fix)
    env_runner.get_seed_list()
    eval_root_dir = exp_dir / "eval"

    evaluator = SimEvaluator(device, agent, env_runner, checkpoint_store, eval_root_dir)

    eval_metadata = {
        "experiment_dir": str(exp_dir),
        "eval": OmegaConf.to_container(cfg.eval, resolve=True),
    }

    # Resolve "best" tag via best_ckpt.json (handoff from select_best_ckpt.py).
    # C5 fix: proper fallback chain best_ckpt.json → best.pt → latest.pt.
    ckpt_tag_or_path = cfg.eval.offline.ckpt_tag_or_path
    if ckpt_tag_or_path == "best":
        best_info = read_best_ckpt_json(exp_dir)
        if best_info:
            ckpt_tag_or_path = best_info["ckpt_path"]
            cprint(
                f"  Auto-loaded best checkpoint from best_ckpt.json: "
                f"{best_info['pct']}% (step={best_info['global_step']})",
                "cyan",
            )
        else:
            ckpt_tag_or_path = str(resolve_best_checkpoint(exp_dir, checkpoint_store))
            cprint(
                f"  ⚠ best_ckpt.json not found — using fallback: {ckpt_tag_or_path}",
                "yellow",
            )

    evaluator.run(
        eval_episodes=int(cfg.eval.offline.eval_episodes),
        denoise_timesteps_list=list(cfg.eval.offline.denoise_timesteps_list),
        ckpt_tag_or_path=ckpt_tag_or_path,
        use_ema_for_eval=bool(cfg.eval.offline.use_ema_for_eval),
        eval_metadata=eval_metadata,
    )

    cprint(f"Evaluation completed, results saved to {evaluator.eval_root_dir}", "green")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-name", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    run_eval(exp_dir, args.overrides)


if __name__ == "__main__":
    main()
