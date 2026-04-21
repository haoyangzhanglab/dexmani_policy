import os
import sys
import pathlib

# 设置项目根目录，以便在训练脚本中正确导入模块，并且在运行训练脚本时保持当前工作目录为项目根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


import hydra
import torch
import warnings
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Sequence
from dataclasses import dataclass

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.sim_evaluator import SimEvaluator

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class EvalComponents:
    agent: Any
    env_runner: Any
    workspace: Any
    device: torch.device


class SimEvalBuilder:
    @staticmethod
    def load_cfg_from_experiment(exp_dir: Path):
        cfg_path = exp_dir / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Can't find config.yaml: {cfg_path}")
        return OmegaConf.load(cfg_path)

    @classmethod
    def build_cfg(cls, exp_dir: Path, overrides: Sequence[str]):
        cfg = cls.load_cfg_from_experiment(exp_dir)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        OmegaConf.update(cfg, "workspace.output_dir", str(exp_dir), force_add=True)
        return cfg

    @staticmethod
    def build_components(cfg) -> EvalComponents:
        cfg.workspace.wandb_cfg.mode = "disabled"   # 禁用wandb日志记录，因为评估过程中不需要记录训练指标
        return EvalComponents(
            device=torch.device(cfg.training.device),
            agent=hydra.utils.instantiate(cfg.agent),
            env_runner=hydra.utils.instantiate(cfg.env_runner),
            workspace=hydra.utils.instantiate(cfg.workspace),
        )

    @staticmethod
    def build_eval_record(cfg, exp_dir: Path):
        return {
            "experiment_dir": str(exp_dir),
            "eval": OmegaConf.to_container(cfg.eval, resolve=True),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-name", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. eval.sim.eval_episodes=200",
    )
    return parser.parse_args()


def run_eval(exp_dir: Path, overrides: Sequence[str]):
    exp_dir = exp_dir.expanduser().resolve()        # 把exp_dir变成一个标准化后的绝对路径
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg = SimEvalBuilder.build_cfg(exp_dir, overrides)
    set_seed(cfg.eval.seed)
    components = SimEvalBuilder.build_components(cfg)

    evaluator = SimEvaluator(
        device=components.device,
        agent=components.agent,
        env_runner=components.env_runner,
        workspace=components.workspace,
    )

    evaluator.run(
        eval_episodes=int(cfg.eval.sim.eval_episodes),
        denoise_timesteps_list=list(cfg.eval.sim.denoise_timesteps_list),
        ckpt_tag_or_path=cfg.eval.sim.ckpt_tag_or_path,
        use_ema_for_eval=bool(cfg.eval.sim.use_ema_for_eval),
        eval_config=SimEvalBuilder.build_eval_record(cfg, exp_dir),
    )


def main() -> None:
    args = parse_args()
    exp_dir = Path(ROOT_DIR) / "experiments" / args.policy_name / args.task_name / args.exp_name
    run_eval(exp_dir, args.overrides)


if __name__ == "__main__":
    main()