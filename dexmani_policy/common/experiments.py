"""Experiment checkpoints and selection records without loading policy weights."""

import json
import re
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


def validate_inference_settings(info):
    from omegaconf import OmegaConf

    from dexmani_policy.common.config import positive_int, resolve_action_steps

    inference = info["inference"]
    if type(inference["use_ema"]) is not bool:
        raise ValueError("inference.use_ema must be a bool")
    positive_int(inference["inference_steps"], "inference.inference_steps")
    if inference["policy_seed_mode"] != "episode_seed":
        raise ValueError("Record requires inference.policy_seed_mode=episode_seed")
    agent = OmegaConf.create(info["evaluation"]["effective_config"]["agent"])
    if inference["action_steps"] is None:
        raise ValueError("Record requires explicit inference.action_steps")
    resolve_action_steps(agent, inference["action_steps"])


def resolve_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    record_path = Path(record_path).expanduser().resolve(strict=True)
    if not record_path.is_relative_to(root):
        raise ValueError("Selection record must be inside the experiment")
    info = json.loads(record_path.read_text())
    if info["status"] != "success":
        raise ValueError("Selection has not completed successfully")
    validate_inference_settings(info)
    checkpoint = resolve_record_path(root, info["ckpt_relpath"], checkpoint=True)
    info["selection_result"] = str(record_path.relative_to(root))
    return info, checkpoint


def resolve_best_checkpoint(experiment_dir):
    root = Path(experiment_dir).resolve()
    pointer = json.loads((root / "best_ckpt.json").read_text())
    record = resolve_record_path(root, pointer["selection_result"])
    return resolve_selection_record(root, record)


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

    @property
    def label(self) -> str:
        return f"{self.pct}% (step={self.global_step})"


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
