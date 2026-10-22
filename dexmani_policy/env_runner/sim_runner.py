from __future__ import annotations

from typing import Any

from dexmani_sim import DATA_DIR, resolve_env_class

from dexmani_policy.env_runner.base_runner import BaseRunner

DEFAULT_EVAL_SEED_COUNT = 100


class SimRunner(BaseRunner):
    def __init__(
        self,
        task_name: str,
        n_obs_steps: int,
        sensor_modalities: list[str] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        eval_seeds: list[int] | None = None,
        clear_cache_freq: int = 25,
        env_video_fps: int | None = None,
        record_video: bool = True,
        render_mode: str = "rgb_array",
        viewer_resolution: tuple[int, int] | None = None,
    ):
        super().__init__(
            n_obs_steps=n_obs_steps,
            env_video_fps=env_video_fps,
            sensor_modalities=sensor_modalities,
            clear_cache_freq=clear_cache_freq,
        )
        self.task_name = task_name
        self.env_kwargs = env_kwargs or {}
        self.eval_seeds = eval_seeds
        self.record_video = record_video
        self.render_mode = render_mode
        self.viewer_resolution = viewer_resolution

    @staticmethod
    def expand_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Expand episode randomization switches; explicit kwargs take precedence."""
        expanded = dict(env_kwargs)

        if expanded.pop("texture_random", False):
            expanded.setdefault("use_texture_randomization", True)
            expanded.setdefault("per_episode_skybox", True)
            expanded.setdefault("randomize_light", True)
            expanded.setdefault("randomize_object_texture", True)
            expanded.setdefault("randomize_camera_pose", True)

        if expanded.pop("instance_random", False):
            expanded.setdefault("randomize_model_id", True)

        if expanded.pop("table_random", False):
            expanded.setdefault("randomize_table_height", True)

        return expanded

    def make_env(self):
        env_class = resolve_env_class(self.task_name)
        env_kwargs = self.expand_env_kwargs(self.env_kwargs)
        env = env_class(render_mode=self.render_mode, record_video=self.record_video, **env_kwargs)

        if self.viewer_resolution is not None and env.viewer is not None:
            env.viewer.resolution = self.viewer_resolution

        return env

    def get_seed_list(self) -> list[int]:
        if self.eval_seeds is not None:
            return self.eval_seeds
        seed_file = DATA_DIR / "eval_seeds" / f"{self.task_name}.txt"
        if seed_file.exists():
            return [int(x) for x in seed_file.read_text().split()]
        return list(range(DEFAULT_EVAL_SEED_COUNT))
