import argparse
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.common.resolver import register_resolvers
from dexmani_policy.training.common.checkpoint_io import CheckpointStore
from dexmani_policy.training.sim_evaluator import SimEvaluator

ROOT_DIR = str(Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

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

    # backward compat: 历史 checkpoint 使用 action_mode → action_key
    if not hasattr(cfg, 'action_key'):
        if hasattr(cfg, 'action_mode'):
            cfg.action_key = 'action_ee' if cfg.action_mode == 'eef_hand' else 'action'
        else:
            cfg.action_key = 'action'

    # 校验 action_key 与 control_mode 一致性（防 CLI override 误配）
    env_kwargs = cfg.get('env_runner', {}).get('env_kwargs', {})
    if isinstance(env_kwargs, dict):
        actual_control = env_kwargs.get('control_mode', 'joint')
    else:
        actual_control = 'joint'
    expected_control = 'ee' if cfg.action_key == 'action_ee' else 'joint'
    if actual_control != expected_control:
        raise ValueError(
            f"action_key='{cfg.action_key}' requires control_mode='{expected_control}', "
            f"but env_runner.env_kwargs.control_mode='{actual_control}'. "
            f"Check CLI overrides for env_runner.env_kwargs.control_mode."
        )

    if not hasattr(cfg, 'eval') or not hasattr(cfg.eval, "offline"):
        raise KeyError(
            "Config is missing 'eval.offline' section. "
            "Please add eval.offline with keys: ckpt_tag_or_path, eval_episodes, "
            "denoise_timesteps_list, use_ema_for_eval."
        )

    assert cfg.n_obs_steps >= 1 and cfg.n_action_steps >= 1, \
        f"n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps} must be >= 1"
    assert cfg.n_obs_steps - 1 + cfg.n_action_steps <= cfg.horizon, \
        f"n_obs_steps-1+n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) " \
        f"exceeds horizon ({cfg.horizon}). The control_action slice " \
        f"pred[:, {cfg.n_obs_steps - 1}:{cfg.n_obs_steps - 1 + cfg.n_action_steps}] would be out of bounds."

    set_seed(cfg.eval.seed)

    device = torch.device(cfg.training.device)
    agent = hydra.utils.instantiate(cfg.agent)
    agent.action_key = cfg.action_key
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    eval_root_dir = exp_dir / "eval"
    checkpoint_store = CheckpointStore(exp_dir / "checkpoints")

    evaluator = SimEvaluator(device, agent, env_runner, checkpoint_store, eval_root_dir)

    eval_config = {
        "experiment_dir": str(exp_dir),
        "eval": OmegaConf.to_container(cfg.eval, resolve=True),
    }

    evaluator.run(
        eval_episodes=int(cfg.eval.offline.eval_episodes),
        denoise_timesteps_list=list(cfg.eval.offline.denoise_timesteps_list),
        ckpt_tag_or_path=cfg.eval.offline.ckpt_tag_or_path,
        use_ema_for_eval=bool(cfg.eval.offline.use_ema_for_eval),
        eval_config=eval_config,
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