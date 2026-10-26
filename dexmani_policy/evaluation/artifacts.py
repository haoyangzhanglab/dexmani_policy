"""Evaluation records and runtime metadata."""

import json
import platform
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.io_util import atomic_path


def runtime_provenance():
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "numerics": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def write_json(path, record, *, overwrite=True):
    with atomic_path(Path(path), overwrite=overwrite) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def write_failure(path, record):
    # Preserve the original failure if saving its details also fails.
    try:
        write_json(path, record)
    except BaseException:
        traceback.print_exc()


def create_run_directory(root, directory=None):
    if directory is None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    return directory
