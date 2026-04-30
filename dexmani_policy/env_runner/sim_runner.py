import re
import importlib
from typing import List, Optional, Dict, Any

from dexmani_sim import DATA_DIR
from dexmani_policy.env_runner.base_runner import BaseRunner

ENV_PREFIX = "dexmani_sim.envs"


class SimRunner(BaseRunner):
    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: List[str] | None = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            n_obs_steps=n_obs_steps,
            env_video_fps=env_video_fps,
            default_eval_episodes=default_eval_episodes,
            sensor_modalities=sensor_modalities
        )
        self.task_name = task_name
        self.env_kwargs = env_kwargs or {}


    @staticmethod
    def name_to_pascal_case(name: str) -> str:
        return ''.join(part.capitalize() for part in re.split(r'[_\s-]+', name) if part)


    def make_env(self):
        full_module_path = f"{ENV_PREFIX}.{self.task_name}"
        try:
            env_module = importlib.import_module(full_module_path)
        except Exception as e:
            raise ImportError(f"Failed to import module {full_module_path}") from e
        env_name = self.name_to_pascal_case(self.task_name)
        try:
            env_class = getattr(env_module, env_name)
        except AttributeError as e:
            raise ImportError(f"Class {env_name} not found in module {full_module_path}") from e
        env_kwargs = {"render_mode": "rgb_array", **self.env_kwargs}
        return env_class(**env_kwargs)


    def get_seed_list(self) -> List[int]:
        seed_file = DATA_DIR / "eval_seeds" / f"{self.task_name}.txt"
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")

        try:
            content = seed_file.read_text().split()
            seeds = [int(x) for x in content]
            if not seeds:
                raise ValueError("Seed file is empty")
            return seeds
        except ValueError as e:
            raise ValueError(f"Invalid seed format in {seed_file}: {e}") from e

