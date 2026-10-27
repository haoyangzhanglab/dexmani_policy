import re
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


MILESTONE_PATTERN = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    path: Path
    pct: int
    global_step: int


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
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
