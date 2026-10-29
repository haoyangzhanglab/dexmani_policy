import json
import re
from dataclasses import dataclass
from pathlib import Path

from dexmani_policy.common.config import PROJECT_ROOT


def resolve_experiment(selector: str | Path) -> Path:
    path = PROJECT_ROOT / Path(selector).expanduser()
    if not path.is_dir():
        path = PROJECT_ROOT / "experiments" / selector
    return path.resolve(strict=True)


def resolve_checkpoint_path(root: Path, path: Path) -> Path:
    path = path.resolve(strict=True)
    directory = (root / "checkpoints").resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Checkpoint must be a file inside {directory}")
    return path


MILESTONE_PATTERN = re.compile(r"^epoch=\d+-step=(?P<step>\d+)-milestone=(?P<pct>\d+)pct\.pt$")


@dataclass
class MilestoneCheckpoint:
    path: Path
    pct: int
    global_step: int


def discover_milestone_checkpoints(exp_dir: Path) -> list[MilestoneCheckpoint]:
    ckpt_dir = exp_dir.resolve() / "checkpoints"
    found: list[MilestoneCheckpoint] = []
    for pt_file in ckpt_dir.glob("epoch=*.pt"):
        match = MILESTONE_PATTERN.fullmatch(pt_file.name)
        if not match:
            continue
        found.append(
            MilestoneCheckpoint(
                path=pt_file,
                pct=int(match.group("pct")),
                global_step=int(match.group("step")),
            )
        )

    return sorted(found, key=lambda checkpoint: (checkpoint.pct, checkpoint.path.name))


def milestone_steps(total):
    return {(pct * total + 99) // 100: pct for pct in (20, 40, 60, 80, 100)}


def resolve_policy_checkpoint(root: Path, selector: str | Path) -> tuple[Path, dict | None]:
    """Resolve selectors and filenames within the experiment's checkpoints directory."""
    selector = str(selector)
    selected = None
    if selector == "best":
        selected = json.loads((root / "best_ckpt.json").read_text())
        path = root / selected["ckpt_relpath"]
    elif selector == "latest":
        path = root / "checkpoints/latest.pt"
    elif selector.endswith("pct") and selector[:-3].isdigit():
        milestones = discover_milestone_checkpoints(root)
        target = int(selector.removesuffix("pct"))
        found = next((item for item in milestones if item.pct == target), None)
        if found is None:
            raise FileNotFoundError(f"No {target}% milestone; available: {[m.pct for m in milestones]}")
        path = found.path
    else:
        path = Path(selector).expanduser()
        if not path.is_absolute():
            path = root / "checkpoints" / path
    return resolve_checkpoint_path(root, path), selected
