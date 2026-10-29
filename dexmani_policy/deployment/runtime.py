import time
from typing import Any

import numpy as np

from dexmani_policy.common.checkpoint import restore_policy
from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, positive_int


_OBSERVATION_SHAPES = {
    "joint_state": (19,),
    "eef_pose": (9,),
    **FIELD_FEATURE_SHAPES,
    "camera_intrinsic": (9,),
    "camera_extrinsic": (12,),
    "tactile_valid": (5,),
}


def load_policy(info, *, device="cuda:0", seed=0):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy(
        info.saved_config, info.checkpoint_path, use_ema=info.weights == "ema", device=device
    )
    return PolicyRuntime(agent, info, device=device, seed=seed)


def make_warmup_observation(info, *, rgb_hw=None):
    observation = {}
    dtype: type[np.generic]
    for name in info.observation_fields:
        if name == "tactile_valid":
            observation[name] = np.ones((info.n_obs_steps, 5), dtype=np.bool_)
            continue
        if name in {"camera_intrinsic", "camera_extrinsic"}:
            # Synthetic pinhole K / base-to-camera [R|t] is only used for warmup.
            columns = 3 if name == "camera_intrinsic" else 4
            matrix = np.eye(3, columns, dtype=np.float32)
            observation[name] = np.tile(matrix.reshape(1, -1), (info.n_obs_steps, 1))
            continue
        if name == "rgb":
            if rgb_hw is None:
                raise ValueError("RGB warmup requires the current raw camera height/width")
            shape, dtype = (*rgb_hw, 3), np.uint8
        elif name == "depth":
            recipe = info.depth_config
            shape, dtype = tuple(recipe["shape"]), np.uint16
            if rgb_hw is not None and tuple(rgb_hw) != shape:
                raise ValueError("Depth warmup camera shape differs from saved raw counts")
        elif name == "hand_pointcloud":
            shape, dtype = (info.hand_pointcloud_config["num_points"], 6), np.float32
        elif name == "point_cloud":
            shape, dtype = (info.pointcloud_num_points, 6), np.float32
        else:
            shape, dtype = _OBSERVATION_SHAPES[name], np.float32
        shape = (info.n_obs_steps, *shape)
        pattern = (
            np.arange(1, 252, dtype=dtype)
            if dtype in {np.uint8, np.uint16}
            else (np.arange(101) / 100.0 - 0.5).astype(dtype)
        )
        observation[name] = np.resize(pattern, shape)

        if name == "hand_pointcloud":
            observation[name][..., 3:] = 0.5

    return observation


class PolicyRuntime:
    def __init__(self, agent, info, *, device, seed):
        import torch

        self.agent = agent
        self.info = info
        self.rgb_preprocessing = info.rgb_preprocessing
        self.depth_shape = None
        self.hand_pointcloud_shape = None
        if "depth" in info.observation_fields:
            self.depth_shape = tuple(info.depth_config["shape"])
        if "hand_pointcloud" in info.observation_fields:
            self.hand_pointcloud_shape = (info.hand_pointcloud_config["num_points"], 6)

        self.generator = torch.Generator(device=device)
        self.device = device
        self.seed = seed
        self.reset_episode()

    def reset_episode(self):
        self.generator.manual_seed(self.seed)

    def prepare_observations(self, observation):
        import torch

        from dexmani_policy.common.image_util import preprocess_rgb_eval

        tensors = {}
        history_length = None
        for name in self.info.observation_fields:
            value = np.asarray(observation[name])
            if value.ndim < 2 or value.dtype.kind not in "buif":
                raise ValueError(f"Policy history {name} requires numeric (T, ...) input")
            if history_length is None:
                history_length = len(value)
            if len(value) != history_length or len(value) < self.info.n_obs_steps:
                raise ValueError("Policy histories must have equal raw lengths >= n_obs_steps")
            value = value[-self.info.n_obs_steps :]
            if name in _OBSERVATION_SHAPES and value.shape[1:] != _OBSERVATION_SHAPES[name]:
                raise ValueError(f"Policy history {name} requires trailing shape {_OBSERVATION_SHAPES[name]}")
            if name == "depth":
                if value.shape[1:] != self.depth_shape or value.dtype != np.uint16:
                    raise ValueError(
                        "Depth input must be uint16 raw counts in the saved image shape"
                    )
            elif name == "hand_pointcloud":
                if value.shape[1:] != self.hand_pointcloud_shape or value.dtype != np.float32:
                    raise ValueError(
                        "Hand-pointcloud input must be float32 XYZRGB in the saved template shape"
                    )
            if name == "point_cloud":
                if value.ndim != 3:
                    raise ValueError("Point cloud requires shape (T, N, C)")
                if self.info.pc_dim is not None and value.shape[-1] < self.info.pc_dim:
                    raise ValueError("Point cloud has fewer channels than the saved input recipe")
                value = value[..., : self.info.pc_dim]
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite policy observation: {name}")
            if name == "rgb":
                if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError("RGB input requires shape (T, H, W, 3) and dtype uint8")
                tensor = preprocess_rgb_eval(value, **self.rgb_preprocessing)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(value))
            dtype = None if name in {"rgb", "tactile_valid"} else torch.float32
            tensors[name] = tensor.unsqueeze(0).to(device=self.device, dtype=dtype)

        return tensors

    def predict(self, observation, *, action_context=None, rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        import torch

        tensors = self.prepare_observations(observation)
        dimensions = self.agent.control_action_dim
        kwargs: dict[str, Any] = {}

        if action_context is not None:
            shapes = self.info.action_context_shapes
            if not isinstance(action_context, dict) or set(action_context) - shapes.keys():
                raise ValueError("Unsupported action_context fields")
            context = {}
            for name, value in action_context.items():
                value = np.asarray(value)
                if (
                    value.shape != tuple(shapes[name])
                    or value.dtype.kind not in "fiu"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid physical action_context.{name}")
                context[name] = torch.as_tensor(value, device=self.device, dtype=torch.float32)[None]
            if context:
                kwargs["action_context"] = context

        if rtc_prefix is not None:
            prefix = np.asarray(rtc_prefix)
            future_steps = self.info.future_steps
            if (
                prefix.ndim != 2
                or prefix.shape[1] != dimensions
                or not np.isfinite(prefix).all()
                or not 0 <= delay_steps <= len(prefix) <= future_steps
            ):
                raise ValueError("Invalid physical RTC prefix")
            kwargs.update(
                rtc_prefix=torch.as_tensor(
                    prefix, device=self.device, dtype=torch.float32
                ).unsqueeze(0),
                delay_steps=delay_steps,
                rtc_guidance_cap=rtc_guidance_cap,
            )
        elif delay_steps:
            raise ValueError("No prefix requires delay_steps=0")

        with torch.inference_mode():
            result = self.agent.predict_action(
                tensors, inference_steps=self.info.inference_steps, generator=self.generator, **kwargs
            )

        # Real consumes the full future after the query, without auxiliary dimensions.
        prediction = result["future_actions"][0].to(device="cpu", dtype=torch.float64).numpy()
        return np.ascontiguousarray(prediction)

    def warmup(self, *, samples=5, rgb_hw=None, action_context=None, rtc_delay=0, rtc_guidance_cap=None):
        positive_int(samples, "samples")
        future_steps = self.info.future_steps
        if type(rtc_delay) is not int or not 0 <= rtc_delay <= future_steps:
            raise ValueError(f"rtc_delay must be an integer in [0, {future_steps}]")
        observation = make_warmup_observation(self.info, rgb_hw=rgb_hw)

        def measure(**kwargs):
            durations = []
            for _ in range(samples):
                start = time.perf_counter_ns()
                self.predict(observation, action_context=action_context, **kwargs)
                durations.append((time.perf_counter_ns() - start) / 1e9)
            return tuple(durations)

        generator_state = self.generator.get_state()
        try:
            # Exclude initialization and reuse its physical prediction for RTC warmup.
            prefix = self.predict(observation, action_context=action_context)
            bootstrap = measure()
            steady = bootstrap
            if rtc_guidance_cap is not None:
                kwargs = dict(rtc_prefix=prefix, delay_steps=rtc_delay, rtc_guidance_cap=rtc_guidance_cap)
                self.predict(observation, action_context=action_context, **kwargs)
                steady = measure(**kwargs)
            return {"bootstrap": bootstrap, "steady": steady}
        finally:
            self.generator.set_state(generator_state)

    def close(self):
        self.agent = None
