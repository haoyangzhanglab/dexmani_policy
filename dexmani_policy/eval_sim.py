import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)


import hydra
import torch
import warnings
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.sim_evaluator import SimEvaluator
from dexmani_policy.training.common.workspace import ReadOnlyWorkspace

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


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

    set_seed(cfg.eval.seed)

    device = torch.device(cfg.training.device)
    agent = hydra.utils.instantiate(cfg.agent)
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    workspace = ReadOnlyWorkspace(output_dir=str(exp_dir))

    evaluator = SimEvaluator(device, agent, env_runner, workspace)

    eval_config = {
        "experiment_dir": str(exp_dir),
        "eval": OmegaConf.to_container(cfg.eval, resolve=True),
    }

    # cfg.eval.sim 仅用于独立 eval，训练期 eval 见 trainer.py:evaluate()。
    summary = evaluator.run(
        eval_episodes=int(cfg.eval.sim.eval_episodes),
        denoise_timesteps_list=list(cfg.eval.sim.denoise_timesteps_list),
        ckpt_tag_or_path=cfg.eval.sim.ckpt_tag_or_path,
        use_ema_for_eval=bool(cfg.eval.sim.use_ema_for_eval),
        eval_config=eval_config,
    )

    print(f"Evaluation completed, results saved to {evaluator.eval_root_dir}")


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