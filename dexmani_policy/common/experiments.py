"""Milestone discovery, checkpoint paths and resume archives."""

import errno
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


def resolve_milestone_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Checkpoint path must be relative to experiment")
    if path.name == "latest.pt" or (root / path).is_symlink():
        raise ValueError("Selection must name an immutable milestone checkpoint")
    resolved = (root / path).resolve()
    directory = root / "checkpoints"
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise ValueError(f"Checkpoint must be a file inside {directory}")
    return resolved


def resolve_checkpoint_path(root: Path, selector: str) -> Path:
    """Resolve a tag or path to a file inside the experiment/checkpoints directory."""
    if selector.endswith("pct"):
        milestones = discover_milestone_checkpoints(root)
        target_pct = int(selector.removesuffix("pct"))
        for milestone in milestones:
            if milestone.pct == target_pct:
                return milestone.path
        available = [m.pct for m in milestones]
        raise FileNotFoundError(f"No {target_pct}% milestone checkpoint. Available: {available}")
    if selector == "latest":
        path = root / "checkpoints/latest.pt"
    else:
        path = Path(selector)
        if not path.is_absolute():
            path = root / "checkpoints" / path
    path = path.resolve()
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
        found.append(
            MilestoneCheckpoint(
                path=resolve_milestone_path(exp_dir, str(pt_file.relative_to(exp_dir))),
                pct=int(match.group("pct")),
                global_step=int(match.group("step")),
            )
        )

    return sorted(found, key=lambda checkpoint: (checkpoint.pct, checkpoint.path.name))


def milestone_steps(total):
    return {(pct * total + 99) // 100: pct for pct in (20, 40, 60, 80, 100)}


def archive_resume_milestones(source, destination, checkpoint):
    """Preserve available valid milestones up to the resume point."""
    from dexmani_policy.common.checkpoint import read_checkpoint
    from dexmani_policy.common.config import validate_source_recipe
    from dexmani_policy.common.runtime import atomic_path

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
        payload = read_checkpoint(candidate.path, mmap=True)
        validate_source_recipe(checkpoint.effective_config, payload["effective_config"])
        if payload["state"]["global_step"] != step:
            raise ValueError(f"Milestone filename/state step mismatch: {candidate.path}")
        del payload
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
