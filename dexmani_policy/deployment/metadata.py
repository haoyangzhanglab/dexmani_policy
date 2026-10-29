from dataclasses import dataclass, fields
from pathlib import Path

from dexmani_policy.common.config import load_experiment_config, positive_int
from dexmani_policy.common.experiments import resolve_experiment, resolve_policy_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs


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
    rgb_preprocessing: dict
    pc_dim: int | None
    supported_execution_modes: tuple[str, ...]
    action_context_shapes: dict
    control_dt_s: float | None
    depth_config: dict | None
    pointcloud_config: dict | None
    hand_pointcloud_config: dict | None
    fingertip_link_names: tuple[str, ...] | None
    tactile_source: str | None

    @property
    def hand_history_steps(self):
        shape = self.action_context_shapes.get("hand_history")
        return shape[0] if shape is not None else None

    def to_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["experiment_dir"] = str(self.experiment_dir)
        result["checkpoint_path"] = str(self.checkpoint_path)
        result["hand_history_steps"] = self.hand_history_steps
        return result


def inspect_policy(experiment, checkpoint="best", weights=None, inference_steps=None) -> PolicyInfo:
    # Inspect model inputs without importing Torch or constructing a model.
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
    deployment = cfg.get("deployment_metadata", {})
    fingertip_links = deployment.get("fingertip_link_names")
    return PolicyInfo(
        experiment_dir=directory,
        checkpoint_path=path,
        weights=weights,
        inference_steps=steps,
        policy_name=cfg["policy_name"],
        task_name=cfg["task_name"],
        observation_fields=tuple(dataset["sensor_modalities"]),
        n_obs_steps=agent["n_obs_steps"],
        n_action_steps=agent["n_action_steps"],
        horizon=agent["horizon"],
        action_start_index=metadata["action_start_index"],
        future_steps=agent["horizon"] - metadata["action_start_index"],
        action_dim=agent["action_dim"],
        control_action_dim=metadata["control_action_dim"],
        action_mode={"action": "joint", "action_ee": "eef"}[cfg["action_key"]],
        rgb_preprocessing=rgb_preprocessing_kwargs(dataset),
        pc_dim=dataset.get("pc_dim"),
        supported_execution_modes=tuple(metadata["supported_execution_modes"]),
        action_context_shapes=metadata["action_context_shapes"],
        control_dt_s=deployment.get("control_dt_s"),
        depth_config=deployment.get("depth_config"),
        pointcloud_config=deployment.get("pointcloud_config"),
        hand_pointcloud_config=deployment.get("hand_pointcloud_config"),
        fingertip_link_names=tuple(fingertip_links) if fingertip_links is not None else None,
        tactile_source=cfg.get("tactile_source"),
    )
