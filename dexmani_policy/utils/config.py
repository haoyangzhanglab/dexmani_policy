"""Lightweight input recipes, Hydra resolvers and config validation."""

import copy
import re
import warnings
from datetime import datetime
from numbers import Integral
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


_LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TASK_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]+(?:\+[a-zA-Z0-9_-]+)*")
FINGER_VECTOR_FIELDS = frozenset({"fingertip_xyz", "finger_tactile_force"})
NON_NUMERIC_OBSERVATION_FIELDS = frozenset({"task_text", "task_name"})
ALLOWED_NORMALIZATION_MODES = frozenset({"identity", "limits", "gaussian", "auto"})


RESUME_OPERATIONAL_FIELDS = {
    'resume_from', 'max_updates', 'workspace.output_dir', 'workspace.wandb_cfg',
    'training.device', 'training.gpu_ids', 'training.use_compile', 'training.compile_mode',
    'training.loop.log_interval_steps', 'dataloader.num_workers',
    'dataloader.persistent_workers', 'dataloader.prefetch_factor',
}


def recipe_content(config):
    """Compare recipe values, excluding only supported run/location changes."""
    plain = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config

    def visit(value, path=''):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = f'{path}.{key}' if path else key
                if (child_path in RESUME_OPERATIONAL_FIELDS
                        or child_path == 'normalizer_source'
                        or re.fullmatch(r'dataset\.(?:datasets\.\d+\.)?zarr_path', child_path)):
                    continue
                result[key] = visit(child, child_path)
            return result
        if isinstance(value, (list, tuple)):
            return [visit(child, f'{path}.{i}') for i, child in enumerate(value)]
        return value

    return visit(plain)


def validate_source_recipe(saved, current):
    """Reject a sidecar or programmatic resume config with a different recipe."""
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    differences.append(f'{path}.{key}: missing field')
                else:
                    compare(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f'{path}: different lengths')
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f'{path}[{i}]')
        elif type(left) is not type(right) or left != right:
            differences.append(f'{path}: checkpoint={left!r}, config={right!r}')

    compare(recipe_content(saved), recipe_content(current), 'config')
    if differences:
        raise ValueError('Checkpoint source recipe mismatch:\n' + '\n'.join(differences))


def checkpoint_source_config(payload, checkpoint_path, *, supplied=None):
    """Compare sidecar and supplied recipes against a validated checkpoint's bound config."""
    bound = payload['effective_config']
    source = Path(checkpoint_path).resolve().parent.parent / 'config.yaml'
    sidecar = OmegaConf.to_container(OmegaConf.load(source), resolve=True) if source.is_file() else None
    for candidate in (sidecar, supplied):
        if candidate is not None:
            if not isinstance(candidate, (dict, DictConfig)) or not candidate:
                raise ValueError('Source config must be a nonempty mapping')
            validate_source_recipe(bound, candidate)
    return copy.deepcopy(bound)


def load_resume_source_config(checkpoint_path):
    from dexmani_policy.training.checkpoint import CheckpointStore
    payload = CheckpointStore(Path(checkpoint_path).parent).load_payload(checkpoint_path)
    return checkpoint_source_config(payload, checkpoint_path)


def resolve_input_recipe(cfg, *, overrides, resume_checkpoint=None):
    """Select the authoritative recipe without binding or creating a training run.

    Explicit task overrides assert the saved identity; only operational and
    dataset-location overrides can change a resumed recipe. Unrelated incoming
    interpolations (notably Hydra's runtime output directory) stay unevaluated.
    ``resume_checkpoint``, when supplied, must be the source already validated
    by CheckpointStore.load in this invocation; reuse only its bound config.
    """
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from dexmani_policy.utils.path import resolve_resume_source

    source = resolve_resume_source(cfg.get("resume_from"))
    if source is None:
        return copy.deepcopy(cfg)
    source_config = (load_resume_source_config(source) if resume_checkpoint is None
                     else resume_checkpoint.effective_config)
    result = OmegaConf.create(source_config)
    with open_dict(result):
        # The execution limit belongs to this invocation.
        result.pop('max_updates', None)
        for override in OverridesParser.create().parse_overrides(list(overrides)):
            key = override.key_or_group
            if key.startswith('hydra.'):
                continue
            if override.is_delete() or override.is_sweep_override():
                raise ValueError(f'Unsupported resume override: {override.input_line}')
            if key == 'task_name':
                requested, identity = override.value(), result.get('task_name')
                if (
                    not isinstance(requested, str)
                    or not TASK_NAME_PATTERN.fullmatch(requested)
                    or requested != identity
                ):
                    raise ValueError(f'Resume task identity mismatch: requested={requested!r}, saved={identity!r}')
                continue
            data_path = re.fullmatch(r'dataset\.(?:datasets\.\d+\.)?zarr_path', key) is not None
            if key not in RESUME_OPERATIONAL_FIELDS and not key.startswith('workspace.wandb_cfg.') and not data_path:
                raise ValueError(f'Resume recipe is owned by saved config; forbidden override: {key}')
            if key == 'resume_from':
                continue
            OmegaConf.update(result, key, override.value(), merge=False, force_add=True)
        result.resume_from = source
    return result


def register_resolvers():
    OmegaConf.register_new_resolver("run_id", lambda: _LAUNCH_ID, replace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OmegaConf.register_new_resolver(
            "eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True
        )
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)


def validate_window_contract(horizon, n_obs_steps, n_action_steps) -> None:
    """Shared training and checkpoint-evaluation action-window grammar."""
    for name, value in (
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
        ("n_action_steps", n_action_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if n_obs_steps - 1 + n_action_steps > horizon:
        raise ValueError(
            f"n_obs_steps - 1 + n_action_steps ({n_obs_steps - 1 + n_action_steps}) "
            f"exceeds horizon ({horizon})"
        )


def validate_action_key_consistency(cfg) -> None:
    """Validate that ``action_key`` matches ``env_runner.env_kwargs.control_mode``.

    Raises ValueError if the configuration is contradictory (e.g. joint-space
    ``action_key`` with ``control_mode='ee'`` in the env runner).  This
    prevents silent misconfiguration from CLI overrides.
    """
    action_key = cfg.get("action_key")
    if action_key not in {"action", "action_ee"}:
        raise ValueError("action_key must be explicitly set to 'action' or 'action_ee'")

    expected_control = "ee" if action_key == "action_ee" else "joint"
    env_runner = cfg.get("env_runner", {})
    task_configs = env_runner.get("task_configs")
    if task_configs is not None:
        for task_index, task_config in enumerate(task_configs):
            env_kwargs = task_config.get("env_kwargs", {})
            actual_control = env_kwargs.get("control_mode", "joint")
            if actual_control != expected_control:
                task_name = task_config.get("task_name", f"index {task_index}")
                raise ValueError(
                    f"action_key='{action_key}' requires control_mode='{expected_control}', "
                    f"but env_runner.task_configs task '{task_name}' has "
                    f"control_mode='{actual_control}'."
                )
    else:
        env_kwargs = env_runner.get("env_kwargs", {})
        if isinstance(env_kwargs, (dict, DictConfig)):
            actual_control = env_kwargs.get("control_mode", "joint")
        else:
            actual_control = "joint"
        if actual_control != expected_control:
            raise ValueError(
                f"action_key='{action_key}' requires control_mode='{expected_control}', "
                f"but env_runner.env_kwargs.control_mode='{actual_control}'. "
                f"Check CLI overrides for env_runner.env_kwargs.control_mode."
            )

    # Guard against CLI overrides that desync dataset.action_key from the
    # top-level action_key (e.g. --dataset.action_key=action while
    # --action_key=action_ee, which would cause a dimension mismatch).
    ds_action_key = cfg.get("dataset", {}).get("action_key")
    if ds_action_key is not None and ds_action_key != action_key:
        raise ValueError(
            f"dataset.action_key='{ds_action_key}' != action_key='{action_key}'. "
            f"They must be the same. Check CLI overrides for dataset.action_key "
            f"and/or action_key."
        )


def load_experiment_config(experiment_dir):
    """Read the resolved snapshot without composing current repository YAML."""
    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(experiment_dir) / "config.yaml"), resolve=True
    )
    if not isinstance(cfg, dict):
        raise ValueError("Expected a saved experiment config mapping")
    return cfg


def validate_normalization_spec(spec, *, observation_fields=None):
    """Validate declared modes and optional exact numeric-field coverage."""
    if not isinstance(spec, dict):
        raise ValueError("Normalization must map field names to modes")
    for key, mode in spec.items():
        if not isinstance(key, str) or not key or key in NON_NUMERIC_OBSERVATION_FIELDS:
            raise ValueError(f"Invalid numeric normalization field: {key!r}")
        if not isinstance(mode, str) or mode not in ALLOWED_NORMALIZATION_MODES:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        if mode == "auto" and key != "action":
            raise ValueError("Normalization auto is only supported for action")
    if not {"joint_state", "action"} <= spec.keys():
        raise ValueError("Normalization must declare joint_state and action")
    if spec["action"] == "identity":
        raise ValueError("Action normalization requires fitted affine parameters")
    if spec.get("rgb", "identity") != "identity":
        raise ValueError("RGB preprocessing requires normalization.rgb=identity")
    if observation_fields is not None:
        expected = set(observation_fields) | {"action"}
        if set(spec) != expected:
            raise ValueError(
                f"Normalization field mismatch: missing={sorted(expected - spec.keys())}, "
                f"extra={sorted(spec.keys() - expected)}"
            )
    return dict(spec)


def resolve_normalization_spec(cfg, *, observation_fields=None):
    spec = cfg.get("normalization")
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    return validate_normalization_spec(spec, observation_fields=observation_fields)


def normalization_dims_from_config(cfg):
    """Number of fitted statistics per field, checked against observation layouts."""
    dims = dict(cfg.get('normalization_dims', {}))
    agent = cfg['agent']
    action_dim = int(agent['action_dim'])
    if 'action' in dims and dims['action'] != action_dim:
        raise ValueError('Saved normalization action dimension disagrees with Agent')
    dims['action'] = action_dim
    # Finger vectors have 15 independent statistics; dense taxels share three axis statistics.
    defaults = {'joint_state': agent.get('state_dim', 19), 'eef_pose': 9,
                'fingertip_xyz': 15, 'finger_tactile_force': 15, 'taxel_tactile_force': 3,
                'camera_intrinsic': 9, 'camera_extrinsic': 12,
                'tactile_valid': 5,
                'point_cloud': cfg.get('dataset', {}).get('pc_dim') or dims.get('point_cloud', 6)}
    for key, width in defaults.items():
        dims.setdefault(key, width)
        if dims[key] != width and (
            key in FINGER_VECTOR_FIELDS or key == 'eef_pose'
            or cfg.get('normalization', {}).get(key, 'identity') != 'identity'
        ):
            raise ValueError(f'Saved {key} feature width must be {width}')
    return dims


def validate_dq_normalization(spec):
    """DQ/VQ hand prototypes require the existing bounded action coordinates."""
    if spec.get("action") not in {"auto", "limits"}:
        raise ValueError("DQ/VQ hand prototypes require action normalization auto or limits")


def validate_action_clipping(spec, clip_sample):
    if type(clip_sample) is not bool:
        raise TypeError("agent.clip_sample must be a bool")
    if spec.get("action") == "gaussian" and clip_sample:
        raise ValueError(
            "Gaussian action normalization requires agent.clip_sample=false."
        )
