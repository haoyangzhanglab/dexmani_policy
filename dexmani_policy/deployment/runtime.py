"""Config-only inspection and a thin NumPy bridge to normal checkpoint inference."""

from __future__ import annotations

import random
import copy
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
    action_dim: int
    control_action_dim: int
    rgb_preprocessing: dict
    normalization: dict
    pc_dim: int | None


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
        positive_int(cfg['agent']['action_dim'], 'action_dim'),
        19 if cfg['action_key'] == 'action' else 21,
        rgb_preprocessing_kwargs(cfg['dataset']),
        dict(cfg['normalization']),
        cfg['dataset'].get('pc_dim'),
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
        from dexmani_policy.utils.config import validate_source_recipe, validate_window_contract
        from dexmani_policy.agents.normalization import resolve_normalization_spec, normalization_dims_from_config
        if info.weights not in {'raw', 'ema'}:
            raise ValueError('weights must be raw or ema')
        positive_int(info.inference_steps, 'inference_steps')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a non-negative integer')
        modes = {'action': 'joint', 'action_ee': 'eef'}
        key = config.get('action_key')
        if key not in modes or info.action_mode not in modes.values() or info.action_mode != modes[key]:
            raise ValueError('Policy action mode disagrees with saved config')
        if getattr(agent, 'action_key', None) != key:
            raise ValueError('Agent action_key disagrees with PolicyInfo/config')
        validate_window_contract(info.horizon, info.n_obs_steps, info.n_action_steps)
        for name in ('horizon', 'n_obs_steps', 'n_action_steps', 'action_dim'):
            if getattr(agent, name) != getattr(info, name) or config['agent'][name] != getattr(info, name):
                raise ValueError(f'Agent/config/PolicyInfo mismatch: {name}')
        control_dim = 19 if key == 'action' else 21
        aux = bool(getattr(agent, 'use_aux_ee', False))
        if (aux != bool(config['agent'].get('use_aux_ee', False))
                or agent.control_action_dim != control_dim or info.control_action_dim != control_dim
                or agent.action_dim != control_dim + (9 if aux else 0)
                or (aux and (key != 'action' or getattr(agent, 'ee_dim', None) != 9))):
            raise ValueError('Invalid complete/control action layout or auxiliary tail')
        if tuple(config['dataset'].get('sensor_modalities', ())) != info.observation_fields:
            raise ValueError('PolicyInfo observation fields disagree with config')
        if (info.rgb_preprocessing != rgb_preprocessing_kwargs(config['dataset'])
                or info.pc_dim != config['dataset'].get('pc_dim')
                or info.normalization != resolve_normalization_spec(config)
                or getattr(agent, 'normalization_spec', None) != info.normalization):
            raise ValueError('Policy numerical preprocessing/normalization mismatch')
        feature_dims = normalization_dims_from_config(config)
        if any(agent.normalization_dims.get(key) != feature_dims.get(key)
               for key, mode in info.normalization.items() if mode != 'identity'):
            raise ValueError('Agent normalization feature widths disagree with config')
        runtime = config.get('real_runtime') or {}
        if (info.control_dt_s != runtime.get('dt') or info.pointcloud_config != runtime.get('pointcloud')
                or info.fingertip_link_names != (tuple(runtime['fingertip_link_names'])
                                               if runtime.get('fingertip_link_names') is not None else None)):
            raise ValueError('PolicyInfo real numerical recipe disagrees with config')
        if hasattr(agent, '_inference_config'):
            validate_source_recipe(agent._inference_config, config)
        encoder = agent.obs_encoder
        if hasattr(encoder, 'use_coord_only') and encoder.use_coord_only != (config['agent'].get('pc_dim') == 3):
            raise ValueError('Agent XYZ/RGB channel selection disagrees with config')
        if (hasattr(encoder, 'use_coord_only') and info.pc_dim is not None
                and info.pc_dim != (3 if encoder.use_coord_only else 6)):
            raise ValueError('Dataset/Agent point-cloud channel selection disagrees')
        for name in ('num_points', 'fps_random_config'):
            if hasattr(encoder, name) and getattr(encoder, name) != (config['agent'].get(name) or ({} if name == 'fps_random_config' else None)):
                raise ValueError(f'Agent point-cloud preprocessing disagrees with config: {name}')
        if hasattr(encoder, 'image_processor'):
            import torch
            from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
            from dexmani_policy.agents.obs_encoder.rgb.utils import to_hw, get_interpolation
            expected = ImageProcessor.from_preset(config['agent']['rgb_backbone_name'])
            rgb = config['agent'].get('rgb_backbone_config') or {}
            if rgb.get('image_size') is not None:
                expected.image_size = to_hw(rgb['image_size'])
            if rgb.get('interpolation') is not None:
                expected.interpolation = get_interpolation(rgb['interpolation'])
            processor = encoder.image_processor
            if (processor.image_size != expected.image_size or processor.interpolation != expected.interpolation
                    or any(not torch.equal(getattr(processor, name).cpu(), torch.as_tensor(
                        rgb[name] if rgb.get(name) is not None else getattr(expected, name), dtype=torch.float32))
                        for name in ('image_mean', 'image_std'))):
                raise ValueError('Agent RGB numerical preprocessing disagrees with config')
        agent.validate_normalizer()
        consumed = getattr(agent, "consumed_observation_fields", ())
        if "task_text" in consumed:
            raise NotImplementedError("MultiTask Real deployment lacks task selection and child numerical recipe restoration")
        validate_observation_fields(agent, info.observation_fields, require_declaration=True)
        self.agent = agent
        self.info = copy.deepcopy(info)
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
        arrays = {name: np.asarray(observation[name]) for name in self.info.observation_fields}
        lengths = set()
        for name, value in arrays.items():
            if value.ndim < 2 or value.dtype.kind not in 'buif':
                raise ValueError(f'Policy history {name} requires numeric (T, ...) input')
            lengths.add(len(value))
        if len(lengths) != 1 or next(iter(lengths), 0) < self.info.n_obs_steps:
            raise ValueError('Policy histories must have equal raw lengths >= n_obs_steps')
        tensors = {}
        for name in self.info.observation_fields:
            value = arrays[name][-self.info.n_obs_steps:]
            if name == 'point_cloud' and self.info.pc_dim is not None:
                if value.shape[-1] < self.info.pc_dim:
                    raise ValueError('Point cloud has fewer channels than the saved input recipe')
                value = value[..., :self.info.pc_dim]
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite policy observation: {name}')
            if name == "rgb":
                tensor = preprocess_validation_rgb(value, **self._rgb)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(value))
            tensors[name] = tensor.unsqueeze(0).to(self._device)
        dimensions = self.agent.control_action_dim
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
        expected = (1, self.info.horizon, self.agent.action_dim)
        if (
            not torch.is_tensor(prediction) or prediction.shape != expected
            or not prediction.is_floating_point()
        ):
            raise ValueError(f"Policy future requires floating pred_action {expected}")
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite policy prediction')
        control = prediction[:, self.info.n_obs_steps - 1 :, :dimensions]
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
            "finger_tactile_force": (5, 3),
            "fingertip_xyz": (5, 3),
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
