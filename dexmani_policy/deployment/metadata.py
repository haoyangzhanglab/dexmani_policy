from dataclasses import dataclass

from dexmani_policy.common.image_util import rgb_preprocessing_kwargs


@dataclass(frozen=True)
class PolicyInfo:
    checkpoint_path: str
    weights: str
    inference_steps: int
    task_name: str
    observation_fields: tuple[str, ...]
    n_obs_steps: int
    n_action_steps: int
    future_steps: int
    control_action_dim: int
    action_mode: str
    rgb_preprocessing: dict
    supports_rtc: bool
    requires_prev_action: bool
    control_dt_s: float | None
    depth_config: dict | None
    pointcloud_config: dict | None
    fingertip_link_names: tuple[str, ...] | None


def policy_info(path, cfg, agent, *, weights, inference_steps) -> PolicyInfo:
    deployment = cfg["deployment_metadata"]
    fingertip_links = deployment.get("fingertip_link_names")
    fields = tuple(key for key in agent.normalization_dims if key != "action")
    return PolicyInfo(
        checkpoint_path=str(path),
        weights=weights,
        inference_steps=inference_steps,
        task_name=cfg["task_name"],
        observation_fields=fields,
        n_obs_steps=agent.n_obs_steps,
        n_action_steps=agent.n_action_steps,
        future_steps=agent.future_steps,
        control_action_dim=agent.control_action_dim,
        action_mode={"action": "joint", "action_ee": "eef"}[cfg["action_key"]],
        rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]) if "rgb" in fields else {},
        supports_rtc=agent.supports_rtc,
        requires_prev_action=agent.requires_prev_action,
        control_dt_s=deployment.get("control_dt_s"),
        depth_config=deployment.get("depth_config"),
        pointcloud_config=deployment.get("pointcloud_config"),
        fingertip_link_names=tuple(fingertip_links) if fingertip_links is not None else None,
    )
