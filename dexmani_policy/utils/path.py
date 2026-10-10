"""Project path helpers for executable entry points."""

import os
from pathlib import Path


def set_project_root() -> str:
    """Change cwd to the project root and return the path.

    Call once at the top of entry-point scripts so that relative paths
    (Hydra config dir, data dir) resolve correctly regardless of the
    current working directory at launch time.
    """
    root = str(Path(__file__).parent.parent.parent)
    os.chdir(root)
    return root


def resolve_resume_source(source):
    """Resolve a checkpoint or experiment path relative to the project root."""
    if source is None:
        return None
    root = Path(__file__).resolve().parents[2]
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        path = path / "checkpoints" / "latest.pt"
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Resume source must be a checkpoint file: {path}")
    return str(path)
