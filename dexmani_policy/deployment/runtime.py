"""Config-only inspection and a thin NumPy bridge to normal checkpoint inference."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_policy.agents.loader import (
    load_experiment_config,
    resolve_best_checkpoint,
    resolve_checkpoint,
)
from dexmani_policy.datasets.preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.utils.validation import positive_int

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"


def resolve_experiment(selector):
    candidate = Path(selector).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        project_candidate = Path(__file__).resolve().parents[2] / candidate
        candidate = (
            project_candidate if project_candidate.is_dir() else _EXPERIMENTS_ROOT / candidate
        )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(_EXPERIMENTS_ROOT.resolve()):
            raise ValueError("Relative experiment selector must remain inside experiments")
    if not resolved.is_dir() or not (resolved / "config.yaml").is_file():
        raise ValueError(f"Experiment requires config.yaml: {resolved}")
    if not (resolved / "checkpoints").is_dir():
        raise ValueError(f"Experiment requires checkpoints: {resolved}")
    return resolved


def list_experiments(filter=None):
    return tuple(
        sorted(
            str(path.parent.relative_to(_EXPERIMENTS_ROOT))
            for path in _EXPERIMENTS_ROOT.rglob("config.yaml")
            if any(p.is_file() for p in (path.parent / "checkpoints").glob("*.pt"))
            and (filter is None or filter.casefold() in str(path.parent).casefold())
        )
    )


@dataclass(frozen=True)
class PolicyInfo:
    experiment_dir: Path
    checkpoint_path: Path
    policy_name: str
    task_name: str
    observation_fields: tuple[str, ...]
    n_obs_steps: int
    n_action_steps: int
    horizon: int
    action_mode: str
    control_dt_s: float | None
    pointcloud_config: dict | None
    fingertip_link_names: tuple[str, ...] | None
    weights: str
    inference_steps: int


def inspect_policy(
    experiment, *, config=None, checkpoint="best", weights=None, inference_steps=None
):
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory) if config is None else config
    if (
        cfg["agent"].get("_target_") == "dexmani_policy.agents.core.multi_task.MultiTaskAgent"
        or cfg["dataset"].get("_target_") == "dexmani_policy.datasets.multi_task_dataset.MultiTaskDataset"
    ):
        raise NotImplementedError("MultiTask Real deployment lacks task selection and child numerical recipe restoration")
    resolved_best = resolve_best_checkpoint(directory) if checkpoint == "best" else None
    path = (
        resolved_best[1] if resolved_best is not None else resolve_checkpoint(directory, checkpoint)
    )
    defaults = dict(cfg.get("eval", {})) if weights is None or inference_steps is None else {}
    if resolved_best is not None:
        defaults.update(resolved_best[0]["inference"])
    if weights is None:
        if type(defaults.get("use_ema")) is not bool:
            raise ValueError("Inference defaults require boolean use_ema")
        weights = "ema" if defaults["use_ema"] else "raw"
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw")
    steps = positive_int(
        defaults.get("inference_steps") if inference_steps is None else inference_steps,
        "inference_steps",
    )
    action_modes = {"action": "joint", "action_ee": "eef"}
    if cfg["action_key"] not in action_modes:
        raise ValueError("action_key must be action or action_ee")
    recipe = cfg.get("real_runtime")
    if recipe is not None and not isinstance(recipe, dict):
        raise ValueError("real_runtime must be a mapping or null")
    fields = tuple(cfg["dataset"].get("sensor_modalities", ()))
    if len(set(fields)) != len(fields):
        raise ValueError("Observation fields must be unique")
    links = recipe.get("fingertip_link_names") if recipe is not None else None
    if links is not None and not isinstance(links, (list, tuple)):
        raise ValueError("Saved fingertip_link_names must be a list of link names")
    return PolicyInfo(
        directory,
        path,
        cfg["policy_name"],
        cfg["task_name"],
        fields,
        positive_int(cfg["agent"]["n_obs_steps"], "n_obs_steps"),
        positive_int(cfg["agent"]["n_action_steps"], "n_action_steps"),
        positive_int(cfg["agent"]["horizon"], "horizon"),
        action_modes[cfg["action_key"]],
        recipe.get("dt") if recipe is not None else None,
        recipe.get("pointcloud") if recipe is not None else None,
        tuple(links) if links is not None else None,
        weights,
        steps,
    )


def load_policy(config, info, *, device="cuda:0", seed=0):
    from dexmani_policy.agents.loader import restore_policy_agent

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy_agent(
        config, info.checkpoint_path, use_ema=info.weights == "ema", device=device
    )
    try:
        return LoadedPolicy(agent, config, info, device=device, seed=seed)
    except BaseException:
        close = getattr(agent, "close", None)
        if close is not None:
            close()
        agent.to("cpu")
        raise


class LoadedPolicy:
    def __init__(self, agent, config, info, *, device, seed):
        from dexmani_policy.utils.validation import validate_observation_fields
        consumed = getattr(agent, "consumed_observation_fields", ())
        if "task_text" in consumed:
            raise NotImplementedError("MultiTask Real deployment lacks task selection and child numerical recipe restoration")
        validate_observation_fields(agent, info.observation_fields, require_declaration=True)
        self.agent = agent
        self.info = info
        self._rgb = rgb_preprocessing_kwargs(config["dataset"])
        self._device = device
        self._seed = seed
        self.rtc_guidance_cap = 0.0
        self.reset_episode()

    def reset_episode(self):
        import torch

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        reset = getattr(self.agent, "reset_episode", None)
        if reset is not None:
            reset()

    def configure_execution(self, mode, guidance_cap=None):
        if mode == "rtc":
            self.configure_rtc(guidance_cap)
        elif mode in {"sync", "async"}:
            self.rtc_guidance_cap = 0.0
        else:
            raise ValueError("Unsupported execution mode")

    def configure_rtc(self, guidance_cap):
        from dexmani_policy.agents.action_decoders.diffusion import Diffusion
        from dexmani_policy.agents.action_decoders.rtc import validate_scheduler
        from dexmani_policy.agents.core.base import BaseAgent

        if (
            type(self.agent).predict_action is not BaseAgent.predict_action
            or type(self.agent).predict_action_from_cond is not BaseAgent.predict_action_from_cond
            or not isinstance(self.agent.action_decoder, Diffusion)
        ):
            raise NotImplementedError("RTC supports continuous BaseAgent DDIM agents only")
        validate_scheduler(
            self.agent.action_decoder.noise_scheduler, self.info.inference_steps, guidance_cap
        )
        self.rtc_guidance_cap = guidance_cap

    def predict(self, observation, *, rtc_prefix=None, delay_steps=0):
        import torch

        from dexmani_policy.datasets.preprocessing import preprocess_validation_rgb

        if self.agent is None:
            raise RuntimeError("Policy is closed")
        missing = set(self.info.observation_fields) - observation.keys()
        if missing:
            raise ValueError(f"Missing policy observations: {sorted(missing)}")
        tensors = {}
        for name in self.info.observation_fields:
            value = np.asarray(observation[name])
            if name == "rgb":
                tensor = preprocess_validation_rgb(value, **self._rgb)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(value))
            tensors[name] = tensor.unsqueeze(0).to(self._device)
        dimensions = 19 if self.info.action_mode == "joint" else 21
        kwargs = {}
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
            kwargs = dict(
                rtc_prefix=torch.as_tensor(prefix, device=self._device).unsqueeze(0),
                delay_steps=delay_steps,
                rtc_guidance_cap=self.rtc_guidance_cap,
            )
        elif delay_steps:
            raise ValueError("No prefix requires delay_steps=0")
        with torch.inference_mode():
            result = self.agent.predict_action(
                tensors, inference_steps=self.info.inference_steps, **kwargs
            )
        # Core callers keep their A-step control_action convention. Real needs P.
        prediction = result.get("pred_action") if isinstance(result, dict) else None
        if (
            not torch.is_tensor(prediction) or prediction.ndim != 3
            or tuple(prediction.shape[:2]) != (1, self.info.horizon)
            or prediction.shape[2] < dimensions or not prediction.is_floating_point()
        ):
            raise ValueError(f"Policy future requires floating pred_action (1, {self.info.horizon}, >= {dimensions})")
        control = prediction[:, self.info.n_obs_steps - 1 :, :dimensions]
        expected = (1, self.info.horizon - self.info.n_obs_steps + 1, dimensions)
        if tuple(control.shape) != expected:
            raise ValueError(f"Policy future must be floating point {expected}")
        future = control.detach().squeeze(0).to(device="cpu", dtype=torch.float64).numpy()
        return future

    def warmup(self, *, samples=5, rgb_hw=None, rtc_delay=0):
        positive_int(samples, "samples")
        future_steps = self.info.horizon - self.info.n_obs_steps + 1
        if type(rtc_delay) is not int or not 0 <= rtc_delay <= future_steps:
            raise ValueError(f"rtc_delay must be an integer in [0, {future_steps}]")
        shapes = {
            "joint_state": (19,),
            "eef_pose": (9,),
            "contact_force": (5, 3),
            "fingertip_points": (5, 3),
            "tactile_force": (5, 120, 3),
        }
        observation = {}
        for name in self.info.observation_fields:
            if name == "tactile_valid":
                observation[name] = np.ones((self.info.n_obs_steps, 5), dtype=np.bool_)
                continue
            if name == "rgb":
                if rgb_hw is None:
                    raise ValueError("RGB warmup requires the current raw camera height/width")
                shape, dtype = (*rgb_hw, 3), np.uint8
            elif name == "point_cloud":
                shape, dtype = (
                    (
                        self.info.pointcloud_config["num_points"],
                        6,
                    ),
                    np.float32,
                )
            else:
                shape, dtype = shapes[name], np.float32
            shape = (self.info.n_obs_steps, *shape)
            values = np.arange(np.prod(shape)).reshape(shape)
            observation[name] = (
                (values % 251 + 1).astype(dtype)
                if dtype == np.uint8
                else ((values % 101) / 100.0 - 0.5).astype(dtype)
            )
        def checked_predict(**kwargs):
            output = self.predict(observation, **kwargs)
            if not np.isfinite(output).all():
                raise ValueError("Nonfinite policy warmup future")
            return output

        def measure(**kwargs):
            durations = []
            for _ in range(samples):
                start = time.perf_counter_ns()
                checked_predict(**kwargs)
                durations.append((time.perf_counter_ns() - start) / 1e9)
            return tuple(durations)

        self.reset_episode()
        try:
            # Initialization is outside the measured set; its finite result is the
            # reusable synthetic prefix in physical units for the guided path.
            prefix = checked_predict()
            bootstrap = measure()
            steady = bootstrap
            if self.rtc_guidance_cap > 0:
                kwargs = {"rtc_prefix": prefix, "delay_steps": rtc_delay}
                checked_predict(**kwargs)
                steady = measure(**kwargs)
            return {"bootstrap": bootstrap, "steady": steady}
        finally:
            self.reset_episode()

    def close(self):
        if self.agent is None:
            return
        agent, self.agent = self.agent, None
        close = getattr(agent, "close", None)
        if close is not None:
            close()
        agent.to("cpu")
        if str(self._device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
