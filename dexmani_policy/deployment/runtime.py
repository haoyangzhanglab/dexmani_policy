"""Config-only inspection and a thin NumPy bridge to normal checkpoint inference."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_policy.common.inference import (
    load_experiment_config,
    positive_int,
    read_best_ckpt_json,
    resolve_checkpoint,
    rgb_preprocessing_kwargs,
)

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"


def resolve_experiment(selector):
    candidate = Path(selector).expanduser()
    if not candidate.is_dir():
        parts = os.fspath(selector).split("/")
        if (
            candidate.is_absolute()
            or len(parts) != 3
            or any(p in {"", ".", ".."} for p in parts)
        ):
            raise ValueError(
                "Expected an experiment directory or policy/task/run selector"
            )
        candidate = _EXPERIMENTS_ROOT.joinpath(*parts)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Experiment must be a directory")
    return resolved


def list_experiments(filter=None):
    return tuple(
        sorted(
            str(path.parent.relative_to(_EXPERIMENTS_ROOT))
            for path in _EXPERIMENTS_ROOT.glob("*/*/*/config.yaml")
            if (path.parent / "checkpoints/latest.pt").is_file()
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
    action_mode: str
    control_dt_s: float | None
    pointcloud_config: dict | None
    weights: str
    inference_steps: int


def inspect_policy(
    experiment, *, config=None, checkpoint="best", weights=None, inference_steps=None
):
    directory = resolve_experiment(experiment)
    cfg = load_experiment_config(directory) if config is None else config
    path = resolve_checkpoint(directory, checkpoint)
    defaults = dict(cfg["eval"])
    if checkpoint == "best":
        inference = read_best_ckpt_json(directory).get("inference", {})
        if not isinstance(inference, dict):
            raise ValueError("Best inference settings must be an object")
        defaults.update(inference)
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
    recipe = cfg["real_runtime"]
    if recipe is not None and not isinstance(recipe, dict):
        raise ValueError("real_runtime must be a mapping or null")
    fields = tuple(cfg["dataset"].get("sensor_modalities", ()))
    if len(set(fields)) != len(fields):
        raise ValueError("Observation fields must be unique")
    return PolicyInfo(
        directory,
        path,
        cfg["policy_name"],
        cfg["task_name"],
        fields,
        positive_int(cfg["agent"]["n_obs_steps"], "n_obs_steps"),
        positive_int(cfg["agent"]["n_action_steps"], "n_action_steps"),
        action_modes[cfg["action_key"]],
        recipe.get("control_dt_s") if recipe is not None else None,
        recipe.get("pointcloud") if recipe is not None else None,
        weights,
        steps,
    )


def load_policy(config, info, *, device="cuda:0", seed=0):
    from dexmani_policy.common.inference import restore_policy_agent

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    agent = restore_policy_agent(
        config, info.checkpoint_path, use_ema=info.weights == "ema", device=device
    )
    return LoadedPolicy(agent, config, info, device=device, seed=seed)


class LoadedPolicy:
    def __init__(self, agent, config, info, *, device, seed):
        self.agent = agent
        self.info = info
        self._rgb = rgb_preprocessing_kwargs(config["dataset"])
        self._device = device
        self._seed = seed
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

    def predict(self, observation):
        import torch

        from dexmani_policy.datasets.base_dataset import preprocess_validation_rgb

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
                tensor = torch.from_numpy(np.array(value, copy=True, order="C"))
            tensors[name] = tensor.unsqueeze(0).to(self._device)
        with torch.inference_mode():
            result = self.agent.predict_action(
                tensors, inference_steps=self.info.inference_steps
            )
        control = result.get("control_action")
        dimensions = 19 if self.info.action_mode == "joint" else 21
        expected = (1, self.info.n_action_steps, dimensions)
        if (
            not torch.is_tensor(control)
            or tuple(control.shape) != expected
            or not control.is_floating_point()
            or not torch.isfinite(control).all()
        ):
            raise ValueError(
                f"Policy control_action must be finite floating point {expected}"
            )
        return (
            control.detach()
            .squeeze(0)
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .copy()
        )

    def warmup(self, *, samples=5, rgb_hw=None):
        positive_int(samples, "samples")
        shapes = {
            "joint_state": (19,),
            "eef_pose": (9,),
            "contact_force": (5, 3),
            "fingertip_points": (5, 3),
            "tactile_force": (5, 120, 3),
        }
        observation = {}
        for name in self.info.observation_fields:
            if name == "rgb":
                if rgb_hw is None:
                    raise ValueError(
                        "RGB warmup requires the current raw camera height/width"
                    )
                shape, dtype = (*rgb_hw, 3), np.uint8
            elif name == "point_cloud":
                shape, dtype = (
                    (self.info.pointcloud_config["num_points"], 6),
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
        durations = []
        self.reset_episode()
        try:
            for _ in range(samples):
                start = time.perf_counter()
                self.predict(observation)
                durations.append(time.perf_counter() - start)
        finally:
            self.reset_episode()
        return tuple(durations)

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
