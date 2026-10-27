import math
from collections.abc import Mapping
from typing import Any

from dataclasses import dataclass, field, fields
from pathlib import Path

from dexmani_policy.common.config import (
    PROJECT_ROOT,
    load_experiment_config,
    positive_int,
    validate_depth_recipe,
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
    saved_config: dict = field(repr=False)
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
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self) if item.name != "saved_config"
        }
        result["experiment_dir"] = str(self.experiment_dir)
        result["checkpoint_path"] = str(self.checkpoint_path)
        return result


def inspect_policy(experiment, checkpoint="best", weights=None, inference_steps=None):
    # Inspect sensor requirements without importing Torch or constructing a model.
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory)
    path, defaults, _ = resolve_policy_checkpoint(directory, checkpoint, cfg)
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
    if "depth" in observation_fields:
        validate_depth_recipe(runtime["depth"])
    if "hand_pointcloud" in observation_fields:
        positive_int(runtime["hand_pointcloud"]["num_points"], "hand_pointcloud.num_points")
    links = runtime.get("fingertip_link_names")
    return PolicyInfo(
        experiment_dir=directory,
        checkpoint_path=path,
        saved_config=cfg,
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
        pointcloud_config=runtime.get("pointcloud"),
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


def describe_real_runtime(buffer, sensor_modalities, task_name) -> dict | None:
    """Capture numerical deployment inputs from the buffer actually used to train."""
    attrs = buffer.attrs
    if attrs.get("format") != "dexmani.real.canonical":
        return None
    if attrs.get("task_name") != task_name:
        raise ValueError("Real canonical task_name differs from the training task")
    dt = attrs.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Real canonical dt must be finite and positive")
    runtime: dict[str, Any] = {"dt": float(dt)}
    if "point_cloud" in sensor_modalities:
        cloud = attrs.get("pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real point_cloud requires root pointcloud_config")
        count = positive_int(cloud["num_points"], "pointcloud_config.num_points")
        shape = buffer["point_cloud"].shape
        if shape[1] != count:
            raise ValueError("Stored point count disagrees with pointcloud_config")
        # Preserve the stored point count; encoder FPS selects its own subset.
        runtime["pointcloud"] = dict(cloud)
    if "hand_pointcloud" in sensor_modalities:
        cloud = attrs.get("hand_pointcloud_config")
        if not isinstance(cloud, Mapping):
            raise ValueError("Real hand_pointcloud requires root hand_pointcloud_config")
        count = positive_int(cloud["num_points"], "hand_pointcloud_config.num_points")
        data = buffer["hand_pointcloud"]
        if data.shape[1:] != (count, 6) or str(data.dtype) != "float32":
            raise ValueError("Stored hand_pointcloud must match the saved XYZRGB template")
        runtime["hand_pointcloud"] = dict(cloud)
    if "depth" in sensor_modalities:
        data = buffer["depth"]
        runtime["depth"] = validate_depth_recipe(
            {
                "shape": list(data.shape[1:]),
                "dtype": str(data.dtype),
                "unit": "raw_counts",
                "depth_scale_m": attrs["depth_scale_m_per_unit"],
            }
        )
    if "fingertip_xyz" in sensor_modalities:
        links = attrs.get("fingertip_link_names")
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Real fingertip_xyz requires five distinct non-empty link names")
        runtime["fingertip_link_names"] = list(links)
    return runtime
