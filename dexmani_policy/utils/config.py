"""Lightweight input recipes, Hydra resolvers and config validation."""

import copy
import re
import warnings
import uuid
from datetime import datetime
from numbers import Integral
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


_LAUNCH_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + uuid.uuid4().hex[:8]
TASK_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]+(?:\+[a-zA-Z0-9_-]+)*")


def load_resume_source_config(checkpoint_path):
    """Read the required resolved recipe belonging to the source checkpoint."""
    source = Path(checkpoint_path).resolve().parent.parent / "config.yaml"
    if not source.is_file():
        raise ValueError(f"Full resume requires the source experiment config.yaml: {source}")
    saved = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(saved, dict) or not saved:
        raise ValueError(f"Full resume requires a non-empty config mapping: {source}")
    return saved


def resolve_input_recipe(cfg, *, overrides):
    """Select the authoritative recipe without binding or creating a training run.

    Explicit task overrides assert the saved identity; only operational and
    dataset-location overrides can change a resumed recipe. Unrelated incoming
    interpolations (notably Hydra's runtime output directory) stay unevaluated.
    """
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from dexmani_policy.training.run_identity import resolve_resume_source

    source = resolve_resume_source(cfg.get("resume_from"))
    if source is None:
        return copy.deepcopy(cfg)
    result = OmegaConf.create(load_resume_source_config(source))
    operational = {
        'resume_from', 'max_updates', 'workspace.output_dir',
        'training.device', 'training.gpu_ids', 'training.use_compile', 'training.compile_mode',
        'training.loop.log_interval_steps', 'dataloader.num_workers',
        'dataloader.persistent_workers', 'dataloader.prefetch_factor',
    }
    with open_dict(result):
        # These belong to an invocation, not the saved learning recipe.
        if 'workspace' in result:
            result.workspace.pop('claim_token', None)
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
            if key not in operational and not key.startswith('workspace.wandb_cfg.') and not data_path:
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
