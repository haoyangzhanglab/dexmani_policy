import argparse
import sys
import hydra
import torch
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Dict, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.sim_evaluator import SimEvaluator

warnings.filterwarnings("ignore")
OmegaConf.register_new_resolver("eval", eval, replace=True)


class SimEvalApp:
    def __init__(self, exp_dir: Path, overrides: Sequence[str]):
        self.exp_dir = Path(exp_dir).expanduser().resolve()
        if not self.exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {self.exp_dir}")

        self.overrides = list(overrides)
        self.cfg = self._build_cfg()
        self.components = None

    @classmethod
    def load_cfg_from_experiment(cls, exp_dir: Path):
        cfg_path = Path(exp_dir).expanduser().resolve() / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Can't find config.yaml: {cfg_path}")
        return OmegaConf.load(cfg_path)

    def _build_cfg(self):
        cfg = self.load_cfg_from_experiment(self.exp_dir)
        if self.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(self.overrides))

        OmegaConf.update(cfg, "workspace.output_dir", str(self.exp_dir), force_add=True)
        self._validate_cfg(cfg)
        return cfg

    @staticmethod
    def _validate_cfg(cfg):
        required_paths = [
            "training.device",
            "workspace",
            "agent",
            "env_runner",
            "eval",
            "eval.seed",
            "eval.sim",
            "eval.sim.ckpt_tag_or_path",
            "eval.sim.eval_episodes",
            "eval.sim.denoise_timesteps_list",
            "eval.sim.use_ema_for_eval",
        ]
        missing_paths = [path for path in required_paths if OmegaConf.select(cfg, path) is None]
        if missing_paths:
            raise KeyError(
                "Experiment config is missing required eval fields: " + ", ".join(missing_paths)
            )

        if int(cfg.eval.sim.eval_episodes) <= 0:
            raise ValueError("cfg.eval.sim.eval_episodes must be > 0")
        if len(cfg.eval.sim.denoise_timesteps_list) == 0:
            raise ValueError("cfg.eval.sim.denoise_timesteps_list must not be empty")

    def build_components(self) -> Dict[str, Any]:
        if self.components is None:
            self.components = {
                "device": torch.device(self.cfg.training.device),
                "agent": hydra.utils.instantiate(self.cfg.agent),
                "env_runner": hydra.utils.instantiate(self.cfg.env_runner),
                "workspace": hydra.utils.instantiate(self.cfg.workspace),
            }
        return self.components

    def build_eval_record(self) -> Dict[str, Any]:
        return {
            "experiment_dir": str(self.exp_dir),
            "eval": OmegaConf.to_container(self.cfg.eval, resolve=True),
        }

    def run(self):
        set_seed(self.cfg.eval.seed)
        components = self.build_components()
        evaluator = SimEvaluator(
            device=components["device"],
            agent=components["agent"],
            env_runner=components["env_runner"],
            workspace=components["workspace"],
        )
        return evaluator.run(
            eval_episodes=int(self.cfg.eval.sim.eval_episodes),
            denoise_timesteps_list=list(self.cfg.eval.sim.denoise_timesteps_list),
            ckpt_tag_or_path=self.cfg.eval.sim.ckpt_tag_or_path,
            use_ema_for_eval=bool(self.cfg.eval.sim.use_ema_for_eval),
            eval_config=self.build_eval_record(),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. eval.sim.eval_episodes=200",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = SimEvalApp(exp_dir=Path(args.exp_dir), overrides=args.overrides)
    app.run()


if __name__ == "__main__":
    main()
