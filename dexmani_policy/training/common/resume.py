import copy
import errno
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.config import PROJECT_ROOT
from dexmani_policy.common.experiments import discover_milestone_checkpoints
from dexmani_policy.common.io_util import atomic_path

RESUME_OPERATIONAL_FIELDS = {
    "resume_from",
    "max_updates",
    "workspace.output_dir",
    "workspace.wandb_cfg",
    "training.device",
    "training.use_compile",
    "training.compile_mode",
    "training.loop.log_interval_steps",
    "training.save_every_updates",
    "training.save_every_epochs",
    "training.val_every_epochs",
    "dataloader.num_workers",
    "dataloader.persistent_workers",
    "dataloader.prefetch_factor",
}


def resolve_resume_source(source):
    if source is None:
        return None
    path = PROJECT_ROOT / Path(source).expanduser()
    if path.is_dir():
        path = path / "checkpoints" / "latest.pt"
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Resume source must be a checkpoint file: {path}")
    return str(path)


def resolve_input_recipe(cfg, *, overrides, saved_config=None):
    """Use the saved recipe on resume, applying only explicit operational/location overrides."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    if saved_config is None:
        return copy.deepcopy(cfg)
    result = OmegaConf.create(saved_config)
    if not OmegaConf.is_dict(result.dataset.get("split_manifest")):
        raise ValueError("Resume requires an embedded dataset.split_manifest")
    with open_dict(result):
        # The execution limit belongs to this invocation.
        result.pop("max_updates", None)
        for override in OverridesParser.create().parse_overrides(list(overrides)):
            key = override.key_or_group
            if key.startswith("hydra."):
                continue
            if override.is_delete() or override.is_sweep_override():
                raise ValueError(f"Unsupported resume override: {override.input_line}")
            if key == "task_name":
                requested, identity = override.value(), result.get("task_name")
                if requested != identity:
                    raise ValueError(
                        f"Resume task identity mismatch: requested={requested!r}, saved={identity!r}"
                    )
                continue
            if (
                key not in RESUME_OPERATIONAL_FIELDS
                and not key.startswith("workspace.wandb_cfg.")
                and key != "dataset.zarr_path"
            ):
                raise ValueError(
                    f"Resume recipe is owned by saved config; forbidden override: {key}"
                )
            if key == "resume_from":
                continue
            OmegaConf.update(result, key, override.value(), merge=False, force_add=True)
        result.resume_from = cfg.get("resume_from")
    return result


def archive_resume_milestones(source, destination, *, max_step):
    """Preserve this experiment's milestone files up to the resume point."""
    directory = Path(source).resolve(strict=True).parent.parent
    target = Path(destination) / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    for candidate in discover_milestone_checkpoints(directory):
        if candidate.global_step > max_step:
            continue
        path = target / candidate.path.name
        with atomic_path(path, overwrite=False) as temporary:
            temporary.unlink()
            try:
                os.link(candidate.path, temporary)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(candidate.path, temporary)
