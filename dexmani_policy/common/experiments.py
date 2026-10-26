"""Experiment checkpoint paths and direct best pointers without loading weights."""

import errno
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


def resolve_record_path(root, relative, *, checkpoint=False):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Record requires a nonempty relative file path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Record path must be relative to experiment")
    if checkpoint and (path.name == "latest.pt" or (root / path).is_symlink()):
        raise ValueError("Selection must name an immutable milestone checkpoint")
    resolved = (root / path).resolve(strict=True)
    directory = root / "checkpoints" if checkpoint else root
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError(f"Record path must be a file inside {directory}")
    return resolved


def validate_best_info(info):
    from dexmani_policy.common.config import positive_int

    if type(info["global_step"]) is not int or info["global_step"] < 0:
        raise ValueError("Selected global_step must be an int >= 0")
    inference = info["inference"]
    if set(inference) != {"use_ema", "inference_steps", "action_steps"}:
        raise ValueError("Best requires explicit use_ema, inference_steps and action_steps")
    if type(inference["use_ema"]) is not bool:
        raise ValueError("Best use_ema must be boolean")
    for key in ("inference_steps", "action_steps"):
        positive_int(inference[key], key)


def resolve_best_checkpoint(experiment_dir):
    root = Path(experiment_dir).resolve()
    info = json.loads((root / "best_ckpt.json").read_text())
    validate_best_info(info)
    source = info["selection_result"]
    if not isinstance(source, str) or not source or Path(source).is_absolute() or ".." in Path(source).parts:
        raise ValueError("selection_result must be a relative source path")
    return info, resolve_record_path(root, info["ckpt_relpath"], checkpoint=True)


def resolve_checkpoint_path(exp_dir, selector="best"):
    """Resolve a tag or path to a file inside the experiment/checkpoints directory."""
    root = Path(exp_dir).resolve()
    if selector == "best":
        return resolve_best_checkpoint(root)[1]
    if selector.endswith("pct"):
        milestones = discover_milestone_checkpoints(root)
        target_pct = int(selector.removesuffix("pct"))
        for milestone in milestones:
            if milestone.pct == target_pct:
                return milestone.path
        available = sorted(m.pct for m in milestones)
        raise FileNotFoundError(f"No {target_pct}% milestone checkpoint. Available: {available}")
    if selector == "latest":
        path = root / "checkpoints/latest.pt"
    else:
        path = Path(selector)
        if not path.is_absolute():
            path = root / "checkpoints" / path
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root / "checkpoints"):
        raise ValueError("Checkpoint must be inside the selected experiment/checkpoints")
    return path


MILESTONE_PATTERN = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    path: Path
    pct: int
    global_step: int


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    """Find regular milestone files inside this experiment, sorted by pct."""
    ckpt_dir = exp_dir / "checkpoints"
    found: list[MilestoneCheckpoint] = []
    for pt_file in ckpt_dir.glob("epoch=*.pt"):
        match = MILESTONE_PATTERN.fullmatch(pt_file.name)
        if not match:
            continue
        if pt_file.is_symlink():
            raise ValueError(f"Milestone checkpoint must not be a symlink: {pt_file}")
        found.append(
            MilestoneCheckpoint(
                path=resolve_checkpoint_path(exp_dir, pt_file.name),
                pct=int(match.group("pct")),
                global_step=int(match.group("step")),
            )
        )

    if not found:
        raise FileNotFoundError(f"No milestone checkpoints found in {ckpt_dir}")

    return sorted(found, key=lambda checkpoint: (checkpoint.pct, checkpoint.path.name))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def milestone_steps(total):
    return {(pct * total + 99) // 100: pct for pct in (20, 40, 60, 80, 100)}


def archive_resume_milestones(source, destination, checkpoint):
    """Make the resumed run self-contained before its first optimizer update."""
    from dexmani_policy.common.checkpoint import read_checkpoint
    from dexmani_policy.common.config import validate_source_recipe
    from dexmani_policy.common.runtime import atomic_path

    expected = {step: pct for step, pct in milestone_steps(
        checkpoint.effective_config["training"]["loop"]["total_train_steps"]
    ).items() if step <= checkpoint.global_step}
    if not expected:
        return
    selected = {}
    directory = Path(source).resolve(strict=True).parent.parent
    for candidate in discover_milestone_checkpoints(directory):
        step = candidate.global_step
        if expected.get(step) != candidate.pct:
            continue
        if step in selected:
            raise ValueError(f"Conflicting milestones at step {step}")
        payload = read_checkpoint(candidate.path, mmap=True)
        validate_source_recipe(checkpoint.effective_config, payload["effective_config"])
        if payload["state"]["global_step"] != step:
            raise ValueError(f"Milestone filename/state step mismatch: {candidate.path}")
        del payload
        selected[step] = candidate
    missing = expected.keys() - selected.keys()
    if missing:
        raise FileNotFoundError(f"Missing source-run milestones: {sorted(missing)}")

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
