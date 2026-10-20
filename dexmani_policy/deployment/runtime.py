"""Load saved policies and predict physical actions from NumPy observations."""

from __future__ import annotations

import random
import time
from contextlib import suppress

import numpy as np

from dexmani_policy.utils.config import positive_int


def load_policy(info, *, device="cuda:0", seed=0):
    from dexmani_policy.training.common.checkpoint import restore_policy_agent

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy_agent(
        info.saved_config, info.checkpoint_path, use_ema=info.weights == "ema", device=device
    )
    try:
        return PolicyRuntime(agent, info, device=device, seed=seed)
    except BaseException:
        with suppress(Exception):
            close = getattr(agent, "close", None)
            if close is not None:
                close()
        with suppress(Exception):
            agent.to("cpu")
        raise


class PolicyRuntime:
    """Bridge raw observation histories to physical future action chunks."""

    def __init__(self, agent, info, *, device, seed):
        from dexmani_policy.agents.core.lamp_policy import LAMPPolicy
        from dexmani_policy.utils.config import validate_observation_fields

        if isinstance(agent, LAMPPolicy):
            agent.validate_inference_steps(info.inference_steps)
            if agent.history_steps != info.hand_history_steps or agent.hand_dim != 12:
                raise ValueError("Agent hand history disagrees with saved configuration")
        elif info.hand_history_steps is not None:
            raise ValueError("Only LAMP accepts external hand history")
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
        validate_observation_fields(agent, info.observation_fields, require_declaration=True)

        self.agent = agent
        self.info = info
        self.rgb_preprocessing = info.rgb_preprocessing
        self.device = device
        self.seed = seed
        self.rtc_guidance_cap = 0.0
        self.reset_episode()

    def reset_episode(self):
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        reset = getattr(self.agent, "reset_episode", None)
        if reset is not None:
            reset()

    def configure_execution(self, mode, guidance_cap=None):
        from dexmani_policy.agents.core.lamp_policy import LAMPPolicy
        from dexmani_policy.agents.core.temporal_latent_flow_policy import TemporalLatentFlowPolicy

        if isinstance(self.agent, (LAMPPolicy, TemporalLatentFlowPolicy)) and mode in {
            "async",
            "rtc",
        }:
            raise NotImplementedError("LAMP and Temporal support sync execution only")
        if mode == "rtc":
            self.configure_rtc(guidance_cap)
        elif mode in {"sync", "async"}:
            self.rtc_guidance_cap = 0.0
        else:
            raise ValueError("Unsupported execution mode")

    def configure_rtc(self, guidance_cap):
        from dexmani_policy.agents.action_decoders.diffusion import Diffusion
        from dexmani_policy.agents.action_decoders.sampling.rtc import prepare_schedule
        from dexmani_policy.agents.core.base_policy import BasePolicy

        if (
            type(self.agent).predict_action is not BasePolicy.predict_action
            or type(self.agent).predict_action_from_cond is not BasePolicy.predict_action_from_cond
            or not isinstance(self.agent.action_decoder, Diffusion)
        ):
            raise NotImplementedError("RTC supports continuous BasePolicy DDIM agents only")
        prepare_schedule(
            self.agent.action_decoder.noise_scheduler, self.info.inference_steps, guidance_cap
        )
        self.rtc_guidance_cap = guidance_cap

    def predict(self, observation, *, action_context=None, rtc_prefix=None, delay_steps=0):
        import torch

        from dexmani_policy.datasets.common.rgb_preprocessing import preprocess_validation_rgb

        if self.agent is None:
            raise RuntimeError("Policy is closed")
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
        for name in self.info.observation_fields:
            value = arrays[name][-self.info.n_obs_steps :]
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
            tensors[name] = tensor.unsqueeze(0).to(self.device)
        dimensions = self.agent.control_action_dim
        kwargs = {}
        from dexmani_policy.agents.core.lamp_policy import LAMPPolicy

        if isinstance(self.agent, LAMPPolicy):
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
            if (
                prefix.ndim != 2
                or prefix.shape[1] != dimensions
                or not np.isfinite(prefix).all()
                or not 0
                <= delay_steps
                <= len(prefix)
                <= self.info.horizon - self.info.n_obs_steps + 1
            ):
                raise ValueError("Invalid physical RTC prefix")
            kwargs.update(
                rtc_prefix=torch.as_tensor(prefix, device=self.device).unsqueeze(0),
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
        shapes = {
            "joint_state": (19,),
            "eef_pose": (9,),
            "finger_tactile_force": (5, 3),
            "fingertip_xyz": (5, 3),
            "taxel_tactile_force": (5, 120, 3),
        }
        observation = {}
        for name in self.info.observation_fields:
            if name == "tactile_valid":
                observation[name] = np.ones((self.info.n_obs_steps, 5), dtype=np.bool_)
                continue
            if name in {"camera_intrinsic", "camera_extrinsic"}:
                # Synthetic pinhole K / base-to-camera [R|t], not a live calibration fallback.
                columns = 3 if name == "camera_intrinsic" else 4
                matrix = np.eye(3, columns, dtype=np.float32)
                observation[name] = np.tile(matrix.reshape(1, -1), (self.info.n_obs_steps, 1))
                continue
            if name == "rgb":
                if rgb_hw is None:
                    raise ValueError("RGB warmup requires the current raw camera height/width")
                shape, dtype = (*rgb_hw, 3), np.uint8
            elif name == "point_cloud":
                recipe = self.info.pointcloud_config or {}
                if "num_points" in recipe:
                    num_points = positive_int(recipe["num_points"], "pointcloud.num_points")
                else:
                    from dexmani_policy.agents.core.dp3_policy import DP3ObservationEncoder
                    from dexmani_policy.agents.core.sat_policy import SATObservationEncoder

                    encoder = self.agent.obs_encoder
                    if not isinstance(encoder, (DP3ObservationEncoder, SATObservationEncoder)):
                        raise ValueError("Point-cloud warmup requires saved pointcloud.num_points")
                    # These encoders can warm up at their sampling size when no raw recipe was saved.
                    num_points = positive_int(encoder.num_points, "encoder.num_points")
                shape, dtype = (num_points, 6), np.float32
            else:
                shape, dtype = shapes[name], np.float32
            shape = (self.info.n_obs_steps, *shape)
            values = np.arange(np.prod(shape)).reshape(shape)
            observation[name] = (
                (values % 251 + 1).astype(dtype)
                if dtype == np.uint8
                else ((values % 101) / 100.0 - 0.5).astype(dtype)
            )

        from dexmani_policy.agents.core.lamp_policy import LAMPPolicy

        context = None
        if isinstance(self.agent, LAMPPolicy):
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
            close = getattr(agent, "close", None)
            if close is not None:
                close()
        finally:
            agent.to("cpu")
            if str(self.device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
