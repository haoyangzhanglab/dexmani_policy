"""Atomic evaluation records and concrete code/runtime provenance."""

import json
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.runtime import atomic_path


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    root = Path(exp_dir).resolve()
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def code_version(directory):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def runtime_provenance():
    import importlib.util
    import torch

    sim = importlib.util.find_spec("dexmani_sim")
    return {
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": (
            code_version(Path(sim.origin).parent) if sim is not None and sim.origin else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def write_json(path, record, *, overwrite=True):
    with atomic_path(Path(path), overwrite=overwrite) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def create_run_directory(root, directory=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    return directory
