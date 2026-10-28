import copy
import errno
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import PROJECT_ROOT
from dexmani_policy.common.experiments import discover_milestone_checkpoints, milestone_steps
from dexmani_policy.common.io_util import atomic_path

ARTIFACT_LOCATION_FIELDS = {
    "agent.pretrained_path",
    "lamp.prior_path",
    "codec.path",
    "codebook_path",
    "agent.prior_path",
    "agent.codec_path",
    "agent.codebook_path",
}

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


def validate_source_recipe(saved, current):
    """Require the same experiment recipe, allowing supported run/location changes."""
    if OmegaConf.is_config(saved):
        saved = OmegaConf.to_container(saved, resolve=True)
    if OmegaConf.is_config(current):
        current = OmegaConf.to_container(current, resolve=True)
    ignored = RESUME_OPERATIONAL_FIELDS | ARTIFACT_LOCATION_FIELDS | {"hand_data.provenance.zarr_path"}
    differences = []

    def compare(left, right, path=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child_path = f"{path}.{key}" if path else key
                if child_path in ignored or child_path == "dataset.zarr_path":
                    continue
                if key not in left or key not in right:
                    differences.append(f"config.{child_path}: missing field")
                else:
                    compare(left[key], right[key], child_path)
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                differences.append(f"config.{path}: different lengths")
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}.{i}")
        elif type(left) is not type(right) or left != right:
            differences.append(f"config.{path}: checkpoint={left!r}, config={right!r}")

    compare(saved, current)
    if differences:
        raise ValueError("Checkpoint source recipe mismatch:\n" + "\n".join(differences))


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
                and key not in ARTIFACT_LOCATION_FIELDS
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


def archive_resume_milestones(source, destination, checkpoint):
    """Preserve available valid milestones up to the resume point."""
    total = checkpoint.effective_config["training"]["loop"]["total_train_steps"]
    expected = {step: pct for step, pct in milestone_steps(total).items() if step <= checkpoint.global_step}
    if not expected:
        return

    selected = {}
    directory = Path(source).resolve(strict=True).parent.parent
    for candidate in discover_milestone_checkpoints(directory):
        step = candidate.global_step
        if step > checkpoint.global_step:
            continue
        if expected.get(step) != candidate.pct:
            raise ValueError(f"Milestone step/percentage differs from training budget: {candidate.path}")
        if step in selected:
            raise ValueError(f"Conflicting milestones at step {step}")
        candidate_state = load_checkpoint(candidate.path, mmap=True)
        validate_source_recipe(checkpoint.effective_config, candidate_state.effective_config)
        if candidate_state.global_step != step:
            raise ValueError(f"Milestone filename/state step mismatch: {candidate.path}")
        del candidate_state
        selected[step] = candidate
    if not selected:
        return

    target = Path(destination) / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    for step in sorted(selected):
        candidate = selected[step]
        path = target / candidate.path.name
        with atomic_path(path, overwrite=False) as temporary:
            temporary.unlink()
            try:
                os.link(candidate.path, temporary)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(candidate.path, temporary)
