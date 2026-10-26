"""Inspect saved policy inputs without importing Torch or constructing a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from dexmani_policy.common.config import (
    load_experiment_config,
    positive_int,
    validate_action_window,
    validate_depth_recipe,
)
from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"


def resolve_experiment(selector):
    candidate = Path(selector).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        project_candidate = Path(__file__).resolve().parents[2] / candidate
        candidate = (
            project_candidate if project_candidate.is_dir() else EXPERIMENTS_ROOT / candidate
        )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(EXPERIMENTS_ROOT.resolve()):
            raise ValueError("Relative experiment selector must remain inside experiments")
    if not resolved.is_dir() or not (resolved / "config.yaml").is_file():
        raise ValueError(f"Experiment requires config.yaml: {resolved}")
    if not (resolved / "checkpoints").is_dir():
        raise ValueError(f"Experiment requires checkpoints: {resolved}")
    return resolved


@dataclass(frozen=True)
class PolicyInfo:
    """Resolved checkpoint and inference settings bound to one saved configuration."""

    experiment_dir: Path
    checkpoint_path: Path
    saved_config: DictConfig = field(repr=False)
    weights: str
    inference_steps: int

    def __post_init__(self):
        config = OmegaConf.create(self.saved_config)
        OmegaConf.resolve(config)
        OmegaConf.set_readonly(config, True)
        object.__setattr__(self, "saved_config", config)
        if "depth" in self.observation_fields:
            validate_depth_recipe(self.depth_config)
        if "hand_pointcloud" in self.observation_fields:
            recipe = self.hand_pointcloud_config
            if not isinstance(recipe, dict):
                raise ValueError("Hand-pointcloud input requires the saved template recipe")
            positive_int(recipe.get("num_points"), "hand_pointcloud.num_points")

    def to_dict(self):
        """Serialize the deployment summary without duplicating the saved configuration."""
        values = {
            "experiment_dir": str(self.experiment_dir),
            "checkpoint_path": str(self.checkpoint_path),
        }
        values.update(
            (name, getattr(self, name))
            for name in (
                "policy_name",
                "task_name",
                "observation_fields",
                "n_obs_steps",
                "n_action_steps",
                "horizon",
                "action_start_index",
                "future_steps",
                "action_mode",
                "control_dt_s",
                "pointcloud_config",
                "hand_pointcloud_config",
                "depth_config",
                "fingertip_link_names",
                "weights",
                "inference_steps",
                "action_dim",
                "control_action_dim",
                "rgb_preprocessing",
                "normalization",
                "pc_dim",
                "hand_history_steps",
                "supported_execution_modes",
                "action_context_shapes",
            )
        )
        return OmegaConf.to_container(OmegaConf.create(values), resolve=True)

    @cached_property
    def policy_name(self):
        return self.saved_config.policy_name

    @cached_property
    def task_name(self):
        return self.saved_config.task_name

    @cached_property
    def observation_fields(self):
        return tuple(self.saved_config.dataset.sensor_modalities)

    @cached_property
    def n_obs_steps(self):
        return self.saved_config.agent.n_obs_steps

    @cached_property
    def n_action_steps(self):
        return self.saved_config.agent.n_action_steps

    @cached_property
    def horizon(self):
        return self.saved_config.agent.horizon

    @cached_property
    def action_start_index(self):
        return self.saved_config.policy_metadata.action_start_index

    @cached_property
    def future_steps(self):
        return self.horizon - self.action_start_index

    @cached_property
    def action_dim(self):
        return self.saved_config.agent.action_dim

    @cached_property
    def action_mode(self):
        return {"action": "joint", "action_ee": "eef"}[self.saved_config.action_key]

    @cached_property
    def control_action_dim(self):
        return self.saved_config.policy_metadata.control_action_dim

    @cached_property
    def control_dt_s(self):
        runtime = self.saved_config.get("real_runtime") or {}
        return runtime.get("dt")

    @property
    def pointcloud_config(self):
        recipe = (self.saved_config.get("real_runtime") or {}).get("pointcloud")
        return OmegaConf.to_container(recipe, resolve=True) if recipe is not None else None

    @property
    def hand_pointcloud_config(self):
        recipe = (self.saved_config.get("real_runtime") or {}).get("hand_pointcloud")
        return OmegaConf.to_container(recipe, resolve=True) if recipe is not None else None

    @property
    def depth_config(self):
        recipe = (self.saved_config.get("real_runtime") or {}).get("depth")
        return OmegaConf.to_container(recipe, resolve=True) if recipe is not None else None

    @cached_property
    def fingertip_link_names(self):
        links = (self.saved_config.get("real_runtime") or {}).get("fingertip_link_names")
        return tuple(links) if links is not None else None

    @property
    def rgb_preprocessing(self):
        return rgb_preprocessing_kwargs(self.saved_config.dataset)

    @property
    def normalization(self):
        return dict(self.saved_config.normalization)

    @cached_property
    def pc_dim(self):
        return self.saved_config.dataset.get("pc_dim")

    @property
    def supported_execution_modes(self):
        return tuple(self.saved_config.policy_metadata.supported_execution_modes)

    @property
    def action_context_shapes(self):
        return OmegaConf.to_container(self.saved_config.policy_metadata.action_context_shapes)

    @cached_property
    def hand_history_steps(self):
        shape = self.action_context_shapes.get("hand_history")
        return shape[0] if shape is not None else None


def inspect_policy(experiment, *, checkpoint, weights, inference_steps):
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory)
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(directory / "checkpoints"):
        raise ValueError("Provide a resolved checkpoint inside experiment/checkpoints")
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(inference_steps, "inference_steps")
    shapes = cfg["policy_metadata"]["action_context_shapes"]
    if "hand_history" in shapes:
        if steps != 1 or cfg["dataset"]["history_steps"] != shapes["hand_history"][0]:
            raise ValueError("LAMP requires NFE=1 and matching dataset/policy history")
    if cfg["action_key"] not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    recipe = cfg.get("real_runtime")
    fields = tuple(cfg["dataset"]["sensor_modalities"])
    if len(set(fields)) != len(fields):
        raise ValueError("Observation fields must be unique")
    links = recipe.get("fingertip_link_names") if recipe is not None else None
    if links is not None and not isinstance(links, (list, tuple)):
        raise ValueError("Saved fingertip_link_names must be a list of link names")
    positive_int(cfg["agent"]["action_dim"], "action_dim")
    validate_action_window(
        cfg["agent"]["horizon"], cfg["agent"]["n_obs_steps"], cfg["agent"]["n_action_steps"],
        action_start_index=cfg["policy_metadata"]["action_start_index"],
    )
    rgb_preprocessing_kwargs(cfg["dataset"])
    return PolicyInfo(directory, path, cfg, weights, steps)
