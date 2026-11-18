"""仿真与真机共用的策略恢复、观测准备及物理动作输出。"""

from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_policy.common.config import (
    POINT_CLOUD_FIELDS,
    positive_int,
    validate_observation_shape,
)

if TYPE_CHECKING:
    import torch

    from dexmani_policy.agents.policies.base import Policy


def resolve_inference_settings(config, *, weights=None, inference_steps=None):
    defaults = config["eval"]
    if weights is None:
        weights = "ema" if defaults["use_ema"] else "raw"
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(defaults["inference_steps"] if inference_steps is None else inference_steps, "inference_steps")
    return weights, steps


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
    effective_config: dict[str, Any], state_dict: dict[str, Any] | None,
    device: "str | torch.device",
) -> "Policy":
    import hydra

    from dexmani_policy.common.normalizer import normalization_feature_shapes
    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    if state_dict is None:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")

    feature_dims = effective_config["normalization_dims"]
    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(effective_config["agent"], _convert_="all")
    finally:
        set_rng_state(rng_state)

    feature_shapes = normalization_feature_shapes(
        feature_dims, effective_config["normalization"], required_identity_fields=agent.required_identity_fields,
    )
    agent.configure_normalizer(feature_dims, feature_shapes)
    agent.load_weights(state_dict, device)
    agent.eval()
    rgb_encoder = getattr(agent.obs_encoder, "rgb_encoder", None)
    if rgb_encoder is not None and rgb_encoder.tune_mode == "lora":
        rgb_encoder.backbone = rgb_encoder.backbone.merge_and_unload()
    return agent
