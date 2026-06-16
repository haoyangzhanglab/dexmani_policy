import re
import importlib
from typing import List, Optional, Dict, Any

from dexmani_sim import DATA_DIR
from dexmani_policy.env_runner.base_runner import BaseRunner

ENV_PREFIX = "dexmani_sim.envs"
DEFAULT_EVAL_SEED_COUNT = 100
"""Default number of evaluation seeds when no seed file is available."""


class SimRunner(BaseRunner):
    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: List[str] | None = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        eval_seeds: Optional[List[int]] = None,
        clear_cache_freq: int = 25,
    ):
        super().__init__(
            n_obs_steps=n_obs_steps,
            env_video_fps=env_video_fps,
            default_eval_episodes=default_eval_episodes,
            sensor_modalities=sensor_modalities,
            clear_cache_freq=clear_cache_freq,
        )
        self.task_name = task_name
        self.env_kwargs = env_kwargs or {}
        self.eval_seeds = eval_seeds


    @staticmethod
    def name_to_pascal_case(name: str) -> str:
        return ''.join(part.capitalize() for part in re.split(r'[_\s-]+', name) if part)


    def make_env(self):
        env_module = importlib.import_module(f"{ENV_PREFIX}.{self.task_name}")
        class_name = self.name_to_pascal_case(self.task_name)
        env_class = getattr(env_module, class_name, None)
        if env_class is None:
            available = [name for name in dir(env_module) if not name.startswith('_')]
            raise AttributeError(
                f"Environment class '{class_name}' not found in {ENV_PREFIX}.{self.task_name}. "
                f"Task name '{self.task_name}' maps to PascalCase '{class_name}'. "
                f"Available names in module: {available}. "
                f"If the class name doesn't follow the snake_case→PascalCase convention, "
                f"add an explicit mapping to name_to_pascal_case."
            )
        return env_class(render_mode="rgb_array", record_video=True, **self.env_kwargs)


    def get_seed_list(self) -> List[int]:
        if self.eval_seeds is not None:
            return self.eval_seeds
        seed_file = DATA_DIR / "eval_seeds" / f"{self.task_name}.txt"
        if seed_file.exists():
            return [int(x) for x in seed_file.read_text().split()]
        default_seeds = list(range(DEFAULT_EVAL_SEED_COUNT))
        if len(default_seeds) < self.default_eval_episodes:
            import warnings
            warnings.warn(
                f"eval_episodes={self.default_eval_episodes} exceeds default seed count "
                f"({DEFAULT_EVAL_SEED_COUNT}). Only {DEFAULT_EVAL_SEED_COUNT} "
                f"episodes will be evaluated unless eval_seeds is provided or "
                f"a seed file exists at {seed_file}.",
                UserWarning,
            )
        return default_seeds

