import numpy as np

from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.inference import prepare_numeric_tensor, prepare_observations, predict_actions
from dexmani_policy.deployment.metadata import _resolve_checkpoint, policy_info


def load_policy(path, *, weights=None, inference_steps=None, device="cuda:0", seed=0):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    path = _resolve_checkpoint(path)
    checkpoint = load_checkpoint(path)
    cfg = checkpoint.effective_config
    defaults = cfg["eval"]
    if weights is None:
        if type(defaults["use_ema"]) is not bool:
            raise ValueError("Saved use_ema must be boolean")
        weights = "ema" if defaults["use_ema"] else "raw"
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(defaults["inference_steps"] if inference_steps is None else inference_steps, "inference_steps")
    state = checkpoint.ema_model_state if weights == "ema" else checkpoint.model_state
    del checkpoint
    agent = restore_policy(cfg, state, device)
    del state
    info = policy_info(path, cfg, agent, weights=weights, inference_steps=steps)
    return PolicyRuntime(agent, info, seed=seed)


class PolicyRuntime:
    def __init__(self, agent, info, *, seed):
        import torch

        self.agent = agent
        self.info = info
        self.generator = torch.Generator(device=next(agent.parameters()).device).manual_seed(seed)

    def reset_episode(self):
        self.generator.manual_seed(self.generator.initial_seed())

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
        return {
            "rtc_prefix": prepare_numeric_tensor(prefix, device=self.generator.device, label="physical RTC prefix"),
            "delay_steps": delay_steps,
            "rtc_guidance_cap": rtc_guidance_cap,
        }

    def predict(self, observation, *, rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        tensors = prepare_observations(
            observation, normalization_dims=self.agent.normalization_dims,
            n_obs_steps=self.agent.n_obs_steps, rgb_preprocessing=self.info.rgb_preprocessing,
            device=self.generator.device,
        )
        kwargs = self._prepare_rtc(rtc_prefix, delay_steps, rtc_guidance_cap)
        return predict_actions(
            self.agent, tensors, inference_steps=self.info.inference_steps,
            generator=self.generator, **kwargs,
        )

    def warmup(self, observation, *, rtc_delay=0, rtc_guidance_cap=None):
        if rtc_guidance_cap is not None and (
            type(rtc_delay) is not int or not 0 <= rtc_delay <= self.info.future_steps
        ):
            raise ValueError("rtc_delay must be an integer within the future window")
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
