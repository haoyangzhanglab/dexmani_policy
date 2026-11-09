"""仿真与真机共用的观测准备及物理动作输出边界。"""

import numpy as np

from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, TACTILE_FEATURE_SHAPES


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
        shape = FIELD_FEATURE_SHAPES.get(name, TACTILE_FEATURE_SHAPES.get(name))
        if shape is not None and value.shape[1:] != shape:
            raise ValueError(f"Policy history {name} requires trailing shape {shape}")
        if name in POINT_CLOUD_FIELDS:
            if value.ndim != 3 or value.shape[1] == 0 or value.shape[-1] < width:
                raise ValueError(f"{name} requires shape (T, N>0, C>={width})")
            value = value[..., :width]
        if name == "rgb":
            if value.ndim != 4 or value.shape[-1] != 3 or min(value.shape[1:3]) == 0 or value.dtype != np.uint8:
                raise ValueError("RGB input requires shape (T, H, W, 3) and dtype uint8")
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
