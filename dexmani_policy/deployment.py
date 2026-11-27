"""仿真与真机共用的策略恢复、观测准备及物理动作输出。"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import (
    CALIBRATED_POINT_CLOUD_FIELDS,
    POINT_CLOUD_FIELDS,
    positive_int,
    validate_ddim_steps,
    validate_nonempty_point_cloud,
    validate_observation_shape,
)
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs

if TYPE_CHECKING:
    import torch

    from dexmani_policy.agents.policies.base import Policy
    from dexmani_policy.common.checkpoint import Checkpoint


__all__ = ["PolicyInfo", "PolicyRuntime", "load_policy"]


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
    control_dt_s: float
    data_processing: dict | None


def load_policy(path, *, weights=None, inference_steps=None, device="cuda:0", seed=0):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = (path / "checkpoints/latest.pt").resolve()
    checkpoint = load_checkpoint(path)
    cfg = checkpoint.effective_config
    weights, steps, state = select_inference_weights(checkpoint, weights=weights, inference_steps=inference_steps)
    del checkpoint
    agent = restore_policy(cfg, state, device)
    del state
    fields = tuple(key for key in agent.normalization_dims if key != "action")
    info = PolicyInfo(
        checkpoint_path=str(path),
        weights=weights,
        inference_steps=steps,
        task_name=cfg["task_name"],
        observation_fields=fields,
        n_obs_steps=agent.n_obs_steps,
        n_action_steps=agent.n_action_steps,
        future_steps=agent.future_steps,
        control_action_dim=agent.control_action_dim,
        action_mode={"action": "joint", "action_ee": "eef", "action_rotvec": "eef_rotvec"}[cfg["action_key"]],
        rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]) if "rgb" in fields else {},
        supports_rtc=agent.supports_rtc,
        control_dt_s=cfg["control_dt_s"],
        data_processing=cfg["data_processing"],
    )
    return PolicyRuntime(agent, info, seed=seed)


class PolicyRuntime:
    def __init__(self, agent, info, *, seed):
        import torch

        self.agent = agent
        self.info = info
        self.generator = torch.Generator(device=next(agent.parameters()).device).manual_seed(seed)

    def reset_episode(self):
        self.generator.manual_seed(self.generator.initial_seed())

    def predict(self, observation, *, rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        device = self.generator.device
        if type(delay_steps) is not int or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        if rtc_prefix is None:
            if delay_steps:
                raise ValueError("No prefix requires delay_steps=0")
        else:
            prefix = np.asarray(rtc_prefix)
            if (
                prefix.ndim != 2
                or prefix.shape[1] != self.info.control_action_dim
                or prefix.dtype.kind not in "fiu"
                or not delay_steps <= len(prefix) <= self.info.future_steps
            ):
                raise ValueError("Invalid physical RTC prefix")
            rtc_prefix = prepare_numeric_tensor(prefix, device=device, label="physical RTC prefix")
        tensors = prepare_observations(
            observation, normalization_dims=self.agent.normalization_dims,
            n_obs_steps=self.info.n_obs_steps, rgb_preprocessing=self.info.rgb_preprocessing,
            device=device,
        )
        return predict_actions(
            self.agent, tensors, inference_steps=self.info.inference_steps,
            generator=self.generator, rtc_prefix=rtc_prefix,
            delay_steps=delay_steps, rtc_guidance_cap=rtc_guidance_cap,
        )

    def warmup(self, observation, *, rtc_delay=0, rtc_guidance_cap=None):
        generator_state = self.generator.get_state()
        try:
            prefix = self.predict(observation)
            if rtc_guidance_cap is not None:
                self.predict(observation, rtc_prefix=prefix, delay_steps=rtc_delay,
                             rtc_guidance_cap=rtc_guidance_cap)
        finally:
            self.generator.set_state(generator_state)

    def close(self):
        self.agent = None


def select_inference_weights(
    checkpoint: "Checkpoint", *, weights: str | None = None, inference_steps: int | None = None,
) -> tuple[str, int, dict[str, Any]]:
    from hydra.utils import get_class

    config = checkpoint.effective_config
    defaults = config["eval"]
    if weights is None:
        weights = "ema" if defaults["use_ema"] else "raw"
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = defaults["inference_steps"] if inference_steps is None else inference_steps
    agent_config = config["agent"]
    policy_class = cast("type[Policy]", get_class(agent_config["_target_"]))
    if policy_class.uses_ddim:
        steps = validate_ddim_steps(steps, "inference_steps", num_training_steps=agent_config["num_training_steps"])
    else:
        steps = positive_int(steps, "inference_steps")
    state = checkpoint.ema_model_state if weights == "ema" else checkpoint.model_state
    if state is None:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")
    return weights, steps, state


def contiguous_numpy(value, dtype=None):
    with np.errstate(over="ignore"):
        value = np.asarray(value, dtype=dtype, order="C")
    # Singleton 轴的负 stride 也可能被 NumPy 视为连续，Torch 不接受它。
    if any(stride < 0 for stride in value.strides):
        value = value.copy(order="C")
    return value


def prepare_numeric_tensor(value, *, device, label):
    import torch

    value = contiguous_numpy(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {label}")
    return torch.from_numpy(value).unsqueeze(0).to(device)


def prepare_observations(observation, *, normalization_dims, n_obs_steps, rgb_preprocessing, device):
    from dexmani_policy.common.image_util import preprocess_rgb_eval

    tensors = {}
    for name, width in normalization_dims.items():
        if name == "action":
            continue
        if name not in observation:
            raise ValueError(f"Missing policy observation field: {name}")
        value = np.asarray(observation[name])
        if value.ndim < 2 or value.dtype.kind not in "buif":
            raise ValueError(f"Policy history {name} requires numeric (T, ...) input")
        if len(value) < n_obs_steps:
            raise ValueError(f"Policy history {name} requires at least {n_obs_steps} frames")
        value = value[-n_obs_steps:]
        validate_observation_shape(name, value.shape[1:], width=width)
        if name in POINT_CLOUD_FIELDS:
            value = value[..., :width]
        if name == "point_cloud" and CALIBRATED_POINT_CLOUD_FIELDS.issubset(normalization_dims):
            validate_nonempty_point_cloud(value, source="policy observation history")
        if name == "rgb":
            if value.dtype != np.uint8:
                raise ValueError("RGB input requires dtype uint8")
            tensor = preprocess_rgb_eval(contiguous_numpy(value), **rgb_preprocessing)
            tensors[name] = tensor.unsqueeze(0).to(device)
        else:
            tensors[name] = prepare_numeric_tensor(value, device=device, label=f"policy observation: {name}")
    return tensors


def predict_actions(agent, observations, *, inference_steps=None, generator=None, **kwargs):
    """返回完整未来物理动作窗口，模型负责辅助维度裁剪及反归一化。"""
    import torch

    device = next(agent.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        result = agent.predict_action(
            observations, inference_steps=inference_steps, generator=generator, **kwargs,
        )
    prediction = result["future_actions"]
    expected = (1, agent.future_steps, agent.control_action_dim)
    if tuple(prediction.shape) != expected:
        raise ValueError(f"Physical policy prediction requires shape {expected}, got {tuple(prediction.shape)}")
    actions = prediction[0].to(device="cpu", dtype=torch.float32).contiguous().numpy()
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite physical policy prediction")
    return actions


def restore_policy(
    effective_config: dict[str, Any], state_dict: dict[str, Any],
    device: "str | torch.device",
) -> "Policy":
    import hydra

    from dexmani_policy.common.normalizer import normalization_feature_shapes
    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    feature_dims = effective_config["normalization_dims"]
    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(effective_config["agent"], _convert_="all")
    finally:
        set_rng_state(rng_state)

    feature_shapes = normalization_feature_shapes(
        feature_dims, effective_config["normalization"], required_identity_fields=agent.required_identity_fields,
        required_limits_fields=agent.required_limits_fields,
    )
    agent.configure_normalizer(feature_dims, feature_shapes)
    agent.load_weights(state_dict, device)
    agent.eval()
    rgb_encoder = getattr(agent.obs_encoder, "rgb_encoder", None)
    if rgb_encoder is not None and rgb_encoder.tune_mode == "lora":
        rgb_encoder.backbone = rgb_encoder.backbone.merge_and_unload()
    return agent
