from dataclasses import dataclass, fields
from pathlib import Path

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    positive_int,
)
from dexmani_policy.common.experiments import resolve_policy_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs


def resolve_experiment(selector):
    path = PROJECT_ROOT / Path(selector).expanduser()
    if not path.is_dir():
        path = PROJECT_ROOT / "experiments" / selector
    return path.resolve(strict=True)


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
    future_steps: int
    action_dim: int
    control_action_dim: int
    action_mode: str
    control_dt_s: float | None
    pointcloud_config: dict | None
    pointcloud_num_points: int | None
    hand_pointcloud_config: dict | None
    depth_config: dict | None
    fingertip_link_names: tuple[str, ...] | None
    rgb_preprocessing: dict
    normalization: dict
    pc_dim: int | None
    supported_execution_modes: tuple[str, ...]
    action_context_shapes: dict
    hand_history_steps: int | None

    def to_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["experiment_dir"] = str(self.experiment_dir)
        result["checkpoint_path"] = str(self.checkpoint_path)
        return result


def inspect_policy(experiment, checkpoint="best", weights=None, inference_steps=None):
    # Inspect sensor requirements without importing Torch or constructing a model.
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory)
    path, selection = resolve_policy_checkpoint(directory, checkpoint)
    defaults = cfg["eval"] if selection is None else selection["inference"]
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
    observation_fields = tuple(dataset["sensor_modalities"])
    shapes = metadata["action_context_shapes"]
    history_steps = shapes["hand_history"][0] if "hand_history" in shapes else None
    if history_steps is not None and (steps != 1 or dataset["history_steps"] != history_steps):
        raise ValueError("LAMP requires NFE=1 and matching dataset/policy history")

    runtime = cfg.get("real_runtime") or {}
    pointcloud_config = runtime.get("pointcloud")
    # Real warmup must keep the raw input size, before the encoder's FPS.
    pointcloud_num_points = (
        pointcloud_config["num_points"]
        if pointcloud_config is not None else agent.get("num_points")
    )
    if "hand_pointcloud" in observation_fields:
        positive_int(runtime["hand_pointcloud"]["num_points"], "hand_pointcloud.num_points")
    links = runtime.get("fingertip_link_names")
    return PolicyInfo(
        experiment_dir=directory,
        checkpoint_path=path,
        weights=weights,
        inference_steps=steps,
        policy_name=cfg["policy_name"],
        task_name=cfg["task_name"],
        observation_fields=observation_fields,
        n_obs_steps=agent["n_obs_steps"],
        n_action_steps=agent["n_action_steps"],
        horizon=agent["horizon"],
        action_start_index=metadata["action_start_index"],
        future_steps=agent["horizon"] - metadata["action_start_index"],
        action_dim=agent["action_dim"],
        control_action_dim=metadata["control_action_dim"],
        action_mode={"action": "joint", "action_ee": "eef"}[cfg["action_key"]],
        control_dt_s=runtime.get("dt"),
        pointcloud_config=pointcloud_config,
        pointcloud_num_points=pointcloud_num_points,
        hand_pointcloud_config=runtime.get("hand_pointcloud"),
        depth_config=runtime.get("depth"),
        fingertip_link_names=tuple(links) if links is not None else None,
        rgb_preprocessing=rgb_preprocessing_kwargs(dataset),
        normalization=cfg["normalization"],
        pc_dim=dataset.get("pc_dim"),
        supported_execution_modes=tuple(metadata["supported_execution_modes"]),
        action_context_shapes=shapes,
        hand_history_steps=history_steps,
    )
