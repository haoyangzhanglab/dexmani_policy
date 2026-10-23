"""Load saved policies and predict physical actions from NumPy observations."""

from __future__ import annotations

import random
import time
from contextlib import suppress

import numpy as np

from dexmani_policy.common.config import positive_int


def load_policy(info, *, device="cuda:0", seed=0):
    from dexmani_policy.common.checkpoint import restore_policy_agent

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy_agent(
        info.saved_config, info.checkpoint_path, use_ema=info.weights == "ema", device=device
    )
    try:
        return PolicyRuntime(agent, info, device=device, seed=seed)
    except BaseException:
        with suppress(Exception):
            agent.to("cpu")
        raise


def make_warmup_observation(info, *, rgb_hw=None):
    """Synthetic inputs match the saved layouts and never substitute for live observations."""
    shapes = {
        "joint_state": (19,),
        "eef_pose": (9,),
        "finger_tactile_force": (5, 3),
        "fingertip_xyz": (5, 3),
        "taxel_tactile_force": (5, 120, 3),
    }
    observation = {}
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
            recipe = info.pointcloud_config
            if not isinstance(recipe, dict):
                raise ValueError("Point-cloud warmup requires the saved production recipe")
            num_points = positive_int(recipe.get("num_points"), "pointcloud.num_points")
            shape, dtype = (num_points, 6), np.float32
        else:
            shape, dtype = shapes[name], np.float32
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
    """Bridge raw observation histories to physical future action chunks."""

    def __init__(self, agent, info, *, device, seed):
        from dexmani_policy.common.config import validate_observation_fields

        if getattr(agent, "history_steps", None) != info.hand_history_steps:
            raise ValueError("Agent hand history disagrees with saved configuration")
        if info.hand_history_steps is not None:
            agent.validate_inference_steps(info.inference_steps)
            if agent.hand_dim != 12:
                raise ValueError("Agent hand history disagrees with saved configuration")
        for name in ("horizon", "n_obs_steps", "n_action_steps", "action_dim"):
            if getattr(agent, name) != getattr(info, name):
                raise ValueError(f"Restored policy differs from saved configuration: {name}")
        aux = bool(getattr(agent, "use_aux_ee", False))
        if (
            agent.control_action_dim != info.control_action_dim
            or agent.action_dim != info.control_action_dim + (9 if aux else 0)
            or (aux and (info.action_mode != "joint" or agent.ee_dim != 9))
        ):
            raise ValueError("Invalid control action layout or auxiliary tail")
        encoder = agent.obs_encoder
        if (
            hasattr(encoder, "use_coord_only")
            and info.pc_dim is not None
            and info.pc_dim != (3 if encoder.use_coord_only else 6)
        ):
            raise ValueError("Dataset and encoder point-cloud channels disagree")
        validate_observation_fields(agent, info.observation_fields)

        self.agent = agent
        self.info = info
        self.rgb_preprocessing = info.rgb_preprocessing
        self.depth_shape = None
        self.hand_pointcloud_shape = None
        if "depth" in info.observation_fields:
            self.depth_shape = tuple(info.depth_config["shape"])
        if "hand_pointcloud" in info.observation_fields:
            self.hand_pointcloud_shape = (info.hand_pointcloud_config["num_points"], 6)

        self.device = device
        self.seed = seed
        self.rtc_guidance_cap = 0.0
        self.reset_episode()

    def reset_episode(self):
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def configure_execution(self, mode, guidance_cap=None):
        from dexmani_policy.agents.action_decoders.sampling.rtc import prepare_schedule

        if mode not in {"sync", "async", "rtc"}:
            raise ValueError("Unsupported execution mode")
        if mode not in self.agent.supported_execution_modes:
            raise NotImplementedError(
                f"{type(self.agent).__name__} does not support {mode} execution"
            )
        if mode == "rtc":
            prepare_schedule(
                self.agent.action_decoder.noise_scheduler, self.info.inference_steps, guidance_cap
            )
            self.rtc_guidance_cap = guidance_cap
        else:
            self.rtc_guidance_cap = 0.0

    def prepare_observations(self, observation):
        import torch

        from dexmani_policy.common.rgb_preprocessing import preprocess_validation_rgb

        missing = set(self.info.observation_fields) - observation.keys()
        if missing:
            raise ValueError(f"Missing policy observations: {sorted(missing)}")
        arrays = {name: np.asarray(observation[name]) for name in self.info.observation_fields}
        lengths = set()
        for name, value in arrays.items():
            if value.ndim < 2 or value.dtype.kind not in "buif":
                raise ValueError(f"Policy history {name} requires numeric (T, ...) input")
            lengths.add(len(value))
        if len(lengths) != 1 or next(iter(lengths), 0) < self.info.n_obs_steps:
            raise ValueError("Policy histories must have equal raw lengths >= n_obs_steps")

        tensors = {}
        for name, value in arrays.items():
            value = value[-self.info.n_obs_steps :]
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
            if name == "point_cloud" and self.info.pc_dim is not None:
                if value.shape[-1] < self.info.pc_dim:
                    raise ValueError("Point cloud has fewer channels than the saved input recipe")
                value = value[..., : self.info.pc_dim]
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite policy observation: {name}")
            if name == "rgb":
                tensor = preprocess_validation_rgb(value, **self.rgb_preprocessing)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(value))
            dtype = None if name in {"rgb", "tactile_valid"} else torch.float32
            tensors[name] = tensor.unsqueeze(0).to(device=self.device, dtype=dtype)

        return tensors

    def predict(self, observation, *, action_context=None, rtc_prefix=None, delay_steps=0):
        import torch

        if self.agent is None:
            raise RuntimeError("Policy is closed")
        tensors = self.prepare_observations(observation)
        dimensions = self.agent.control_action_dim
        kwargs = {}

        if self.info.hand_history_steps is not None:
            if not isinstance(action_context, dict) or set(action_context) != {"hand_history"}:
                raise ValueError(
                    "LAMP requires only action_context.hand_history from past dispatched targets"
                )
            history = action_context["hand_history"]
            if (
                not isinstance(history, np.ndarray)
                or history.dtype.kind not in "fiu"
                or history.shape != (self.agent.history_steps, self.agent.hand_dim)
                or not np.isfinite(history).all()
            ):
                raise ValueError("hand_history must be finite physical NumPy [K,12]")
            kwargs["action_context"] = {
                "hand_history": torch.tensor(
                    history, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            }
        elif action_context is not None and (
            not isinstance(action_context, dict) or action_context
        ):
            raise ValueError("This runtime policy does not accept external action_context keys")

        if rtc_prefix is not None:
            prefix = np.asarray(rtc_prefix)
            future_steps = self.info.horizon - self.info.n_obs_steps + 1
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
                rtc_guidance_cap=self.rtc_guidance_cap,
            )
        elif delay_steps:
            raise ValueError("No prefix requires delay_steps=0")

        with torch.inference_mode():
            result = self.agent.predict_action(
                tensors, inference_steps=self.info.inference_steps, **kwargs
            )

        # Real consumes the full future after the query, without auxiliary dimensions.
        prediction = result.get("pred_action") if isinstance(result, dict) else None
        expected = (1, self.info.horizon, self.agent.action_dim)
        if (
            not torch.is_tensor(prediction)
            or prediction.shape != expected
            or not prediction.is_floating_point()
        ):
            raise ValueError(f"Policy future requires floating pred_action {expected}")
        prediction = prediction.detach().squeeze(0).to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite policy prediction")
        return np.ascontiguousarray(prediction[self.info.n_obs_steps - 1 :, :dimensions])

    def warmup(self, *, samples=5, rgb_hw=None, rtc_delay=0):
        positive_int(samples, "samples")
        future_steps = self.info.horizon - self.info.n_obs_steps + 1
        if type(rtc_delay) is not int or not 0 <= rtc_delay <= future_steps:
            raise ValueError(f"rtc_delay must be an integer in [0, {future_steps}]")
        observation = make_warmup_observation(self.info, rgb_hw=rgb_hw)

        context = None
        if self.info.hand_history_steps is not None:
            context = {
                "hand_history": np.repeat(
                    observation["joint_state"][-1:, -12:], self.agent.history_steps, axis=0
                )
            }

        def measure(**kwargs):
            durations = []
            for _ in range(samples):
                start = time.perf_counter_ns()
                self.predict(observation, action_context=context, **kwargs)
                durations.append((time.perf_counter_ns() - start) / 1e9)
            return tuple(durations)

        self.reset_episode()
        try:
            # Exclude initialization and reuse its physical prediction for RTC warmup.
            prefix = self.predict(observation, action_context=context)
            bootstrap = measure()
            steady = bootstrap
            if self.rtc_guidance_cap > 0:
                kwargs = {"rtc_prefix": prefix, "delay_steps": rtc_delay}
                self.predict(observation, action_context=context, **kwargs)
                steady = measure(**kwargs)
            return {"bootstrap": bootstrap, "steady": steady}
        finally:
            self.reset_episode()

    def close(self):
        if self.agent is None:
            return
        agent, self.agent = self.agent, None
        try:
            agent.to("cpu")
        finally:
            if str(self.device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
