"""Inspect saved policy inputs without importing Torch or constructing a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from dexmani_policy.datasets.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.training.common.checkpoint import resolve_best_checkpoint, resolve_checkpoint
from dexmani_policy.utils.config import load_experiment_config, positive_int, validate_action_window

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
                "action_mode",
                "control_dt_s",
                "pointcloud_config",
                "fingertip_link_names",
                "weights",
                "inference_steps",
                "action_dim",
                "control_action_dim",
                "rgb_preprocessing",
                "normalization",
                "pc_dim",
                "hand_history_steps",
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
    def action_dim(self):
        return self.saved_config.agent.action_dim

    @cached_property
    def action_mode(self):
        return {"action": "joint", "action_ee": "eef"}[self.saved_config.action_key]

    @cached_property
    def control_action_dim(self):
        return 19 if self.action_mode == "joint" else 21

    @cached_property
    def control_dt_s(self):
        runtime = self.saved_config.get("real_runtime") or {}
        return runtime.get("dt", self.saved_config.get("hand_data", {}).get("dt"))

    @property
    def pointcloud_config(self):
        recipe = (self.saved_config.get("real_runtime") or {}).get("pointcloud")
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

    @cached_property
    def hand_history_steps(self):
        if (
            self.saved_config.agent.get("_target_")
            == "dexmani_policy.agents.core.lamp_policy.LAMPPolicy"
        ):
            return self.saved_config.agent.history_steps
        return None


def inspect_policy(experiment, *, checkpoint="best", weights=None, inference_steps=None):
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory)
    if (
        cfg["agent"].get("_target_")
        == "dexmani_policy.agents.core.multitask_policy.MultiTaskPolicy"
        or cfg["dataset"].get("_target_")
        == "dexmani_policy.datasets.multitask_dataset.MultiTaskDataset"
    ):
        raise NotImplementedError(
            "MultiTask Real deployment lacks task selection and child numerical recipe restoration"
        )
    resolved_best = resolve_best_checkpoint(directory) if checkpoint == "best" else None
    path = (
        resolved_best[1] if resolved_best is not None else resolve_checkpoint(directory, checkpoint)
    )
    defaults = dict(cfg.get("eval", {})) if weights is None or inference_steps is None else {}
    if resolved_best is not None:
        defaults.update(resolved_best[0]["inference"])
    if weights is None:
        if type(defaults.get("use_ema")) is not bool:
            raise ValueError("Inference defaults require boolean use_ema")
        weights = "ema" if defaults["use_ema"] else "raw"
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(
        defaults.get("inference_steps") if inference_steps is None else inference_steps,
        "inference_steps",
    )
    if cfg["agent"].get("_target_") == "dexmani_policy.agents.core.lamp_policy.LAMPPolicy":
        if steps != 1:
            raise ValueError("LAMP requires inference_steps=1")
        history = positive_int(cfg["agent"]["history_steps"], "agent.history_steps")
        if cfg["dataset"]["history_steps"] != history:
            raise ValueError("LAMP dataset/policy history_steps must match")
    if cfg["action_key"] not in {"action", "action_ee"}:
        raise ValueError("action_key must be action or action_ee")
    recipe = cfg.get("real_runtime")
    fields = tuple(cfg["dataset"].get("sensor_modalities", ()))
    if len(set(fields)) != len(fields):
        raise ValueError("Observation fields must be unique")
    links = recipe.get("fingertip_link_names") if recipe is not None else None
    if links is not None and not isinstance(links, (list, tuple)):
        raise ValueError("Saved fingertip_link_names must be a list of link names")
    positive_int(cfg["agent"]["action_dim"], "action_dim")
    validate_action_window(
        cfg["agent"]["horizon"], cfg["agent"]["n_obs_steps"], cfg["agent"]["n_action_steps"]
    )
    rgb_preprocessing_kwargs(cfg["dataset"])
    return PolicyInfo(directory, path, cfg, weights, steps)
