from dataclasses import dataclass, fields
from pathlib import Path

from dexmani_policy.common.config import POINT_CLOUD_FIELDS, PROJECT_ROOT, positive_int
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs


def _resolve_checkpoint(experiment, selector):
    directory = PROJECT_ROOT / Path(experiment).expanduser()
    if not directory.is_dir():
        directory = PROJECT_ROOT / "experiments" / experiment
    directory = directory.resolve(strict=True)
    selector = str(selector)
    if selector == "latest":
        path = directory / "checkpoints/latest.pt"
    else:
        path = Path(selector).expanduser()
        if not path.is_absolute():
            path = directory / "checkpoints" / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Checkpoint must be a file: {path}")
    return directory, path


@dataclass(frozen=True)
class PolicyInfo:
    experiment_dir: Path
    checkpoint_path: Path
    weights: str
    inference_steps: int
    policy_name: str
    task_name: str
    observation_fields: tuple[str, ...]
    n_obs_steps: int
    n_action_steps: int
    horizon: int
    action_start_index: int
    action_dim: int
    control_action_dim: int
    action_mode: str
    rgb_preprocessing: dict
    pc_dim: int | None
    supported_execution_modes: tuple[str, ...]
    control_dt_s: float | None
    depth_config: dict | None
    pointcloud_config: dict | None
    hand_pointcloud_config: dict | None
    fingertip_link_names: tuple[str, ...] | None
    tactile_source: str | None

    @property
    def future_steps(self) -> int:
        return self.horizon - self.action_start_index

    def to_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["experiment_dir"] = str(self.experiment_dir)
        result["checkpoint_path"] = str(self.checkpoint_path)
        result["future_steps"] = self.future_steps
        return result


def inspect_policy(experiment, checkpoint="latest", weights=None, inference_steps=None) -> PolicyInfo:
    # Read the checkpoint recipe without constructing a model.
    directory, path = _resolve_checkpoint(experiment, checkpoint)
    cfg = load_checkpoint(path).effective_config
    defaults = cfg["eval"]
    if weights is None:
        if type(defaults["use_ema"]) is not bool:
            raise ValueError("Saved use_ema must be boolean")
        weights = "ema" if defaults["use_ema"] else "raw"
    if inference_steps is None:
        inference_steps = defaults["inference_steps"]
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(inference_steps, "inference_steps")
    agent, dataset, metadata = cfg["agent"], cfg["dataset"], cfg["policy_metadata"]
    deployment = cfg["deployment_metadata"]
    fingertip_links = deployment.get("fingertip_link_names")
    return PolicyInfo(
        experiment_dir=directory,
        checkpoint_path=path,
        weights=weights,
        inference_steps=steps,
        policy_name=cfg["policy_name"],
        task_name=cfg["task_name"],
        observation_fields=tuple(key for key in cfg["normalization_dims"] if key != "action"),
        n_obs_steps=agent["n_obs_steps"],
        n_action_steps=agent["n_action_steps"],
        horizon=agent["horizon"],
        action_start_index=metadata["action_start_index"],
        action_dim=agent["action_dim"],
        control_action_dim=metadata["control_action_dim"],
        action_mode={"action": "joint", "action_ee": "eef"}[cfg["action_key"]],
        rgb_preprocessing=rgb_preprocessing_kwargs(dataset),
        pc_dim=next((width for key, width in cfg["normalization_dims"].items()
                     if key in POINT_CLOUD_FIELDS), None),
        supported_execution_modes=tuple(metadata["supported_execution_modes"]),
        control_dt_s=deployment.get("control_dt_s"),
        depth_config=deployment.get("depth_config"),
        pointcloud_config=deployment.get("pointcloud_config"),
        hand_pointcloud_config=deployment.get("hand_pointcloud_config"),
        fingertip_link_names=tuple(fingertip_links) if fingertip_links is not None else None,
        tactile_source=cfg.get("tactile_source"),
    )
