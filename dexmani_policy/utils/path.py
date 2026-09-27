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
