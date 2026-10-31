import time

import numpy as np

from dexmani_policy.common.checkpoint import restore_policy
from dexmani_policy.common.config import FIELD_FEATURE_SHAPES, POINT_CLOUD_FIELDS, positive_int


_OBSERVATION_SHAPES = {
    "joint_state": (19,),
    "eef_pose": (9,),
    **FIELD_FEATURE_SHAPES,
}


def _contiguous_numpy(value, dtype=None):
    with np.errstate(over="ignore"):
        value = np.asarray(value, dtype=dtype, order="C")
    # NumPy considers negative strides on singleton axes contiguous; Torch rejects them.
    if any(stride < 0 for stride in value.strides):
        value = value.copy(order="C")
    return value


def load_policy(info, *, device="cuda:0", seed=0):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy(
        info.checkpoint_path, use_ema=info.weights == "ema", device=device
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

    def _prepare_numeric_tensor(self, value, label):
        import torch

        value = _contiguous_numpy(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {label}")
        return torch.from_numpy(value).unsqueeze(0).to(self.generator.device)

    def prepare_observations(self, observation):
        from dexmani_policy.common.image_util import preprocess_rgb_eval

        tensors = {}
        for name in self.info.observation_fields:
            value = np.asarray(observation[name])
            if value.ndim < 2 or value.dtype.kind not in "buif":
                raise ValueError(f"Policy history {name} requires numeric (T, ...) input")
            if len(value) < self.info.n_obs_steps:
                raise ValueError(f"Policy history {name} requires at least {self.info.n_obs_steps} frames")
            value = value[-self.info.n_obs_steps :]
            if name in _OBSERVATION_SHAPES and value.shape[1:] != _OBSERVATION_SHAPES[name]:
                raise ValueError(f"Policy history {name} requires trailing shape {_OBSERVATION_SHAPES[name]}")
            if name in POINT_CLOUD_FIELDS:
                if value.ndim != 3 or value.shape[1] == 0:
                    raise ValueError(f"{name} requires shape (T, N>0, C)")
                if self.info.pc_dim is not None and value.shape[-1] < self.info.pc_dim:
                    raise ValueError(f"{name} has fewer channels than the saved input recipe")
                value = value[..., : self.info.pc_dim]
            if name == "rgb":
                if value.ndim != 4 or value.shape[-1] != 3 or min(value.shape[1:3]) == 0 or value.dtype != np.uint8:
                    raise ValueError("RGB input requires shape (T, H, W, 3) and dtype uint8")
                tensor = preprocess_rgb_eval(_contiguous_numpy(value), **self.info.rgb_preprocessing)
                tensors[name] = tensor.unsqueeze(0).to(self.generator.device)
            else:
                tensors[name] = self._prepare_numeric_tensor(
                    value, f"policy observation after preprocessing: {name}",
                )

        return tensors

    def _prepare_rtc(self, rtc_prefix, delay_steps, rtc_guidance_cap):
        if rtc_prefix is None:
            if delay_steps:
                raise ValueError("No prefix requires delay_steps=0")
            return {}
        prefix = np.asarray(rtc_prefix)
        if (
            prefix.ndim != 2
            or prefix.shape[1] != self.info.control_action_dim
            or prefix.dtype.kind not in "fiu"
            or not 0 <= delay_steps <= len(prefix) <= self.info.future_steps
        ):
            raise ValueError("Invalid physical RTC prefix")
        return dict(
            rtc_prefix=self._prepare_numeric_tensor(prefix, "physical RTC prefix"),
            delay_steps=delay_steps,
            rtc_guidance_cap=rtc_guidance_cap,
        )

    def predict(self, observation, *, rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        import torch

        tensors = self.prepare_observations(observation)
        kwargs = self._prepare_rtc(rtc_prefix, delay_steps, rtc_guidance_cap)

        with torch.inference_mode():
            result = self.agent.predict_action(
                tensors, inference_steps=self.info.inference_steps, generator=self.generator, **kwargs
            )

        # Return the full physical future window, excluding auxiliary predictions.
        prediction = result["future_actions"][0].to(device="cpu", dtype=torch.float64).numpy()
        return np.ascontiguousarray(prediction)

    def warmup(self, observation, *, samples=5, rtc_delay=0, rtc_guidance_cap=None):
        positive_int(samples, "samples")
        future_steps = self.info.future_steps
        if type(rtc_delay) is not int or not 0 <= rtc_delay <= future_steps:
            raise ValueError(f"rtc_delay must be an integer in [0, {future_steps}]")

        def measure(**kwargs):
            durations = []
            for _ in range(samples):
                start = time.perf_counter_ns()
                self.predict(observation, **kwargs)
                durations.append((time.perf_counter_ns() - start) / 1e9)
            return tuple(durations)

        generator_state = self.generator.get_state()
        try:
            # Exclude initialization and reuse its physical prediction for RTC warmup.
            prefix = self.predict(observation)
            bootstrap = measure()
            steady = bootstrap
            if rtc_guidance_cap is not None:
                kwargs = dict(rtc_prefix=prefix, delay_steps=rtc_delay, rtc_guidance_cap=rtc_guidance_cap)
                self.predict(observation, **kwargs)
                steady = measure(**kwargs)
            return {"bootstrap": bootstrap, "steady": steady}
        finally:
            self.generator.set_state(generator_state)

    def close(self):
        self.agent = None
