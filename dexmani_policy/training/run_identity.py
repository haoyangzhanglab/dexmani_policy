"""Claim training output directories and resolve external resume paths."""

import json
import os
import uuid
from pathlib import Path


def claim_run(output_dir, *, resume_from=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        if (root / name).exists():
            raise FileExistsError(f"Training output already exists: {root / name}; use a new output directory")
    token = uuid.uuid4().hex
    marker = root / ".training_run.json"
    try:
        with marker.open("x") as stream:
            json.dump({"token": token, "pid": os.getpid(), "resume_from": resume_from}, stream)
    except FileExistsError:
        raise FileExistsError(f"Training directory already claimed: {root}; use a new output directory") from None
    return token


def check_run_claim(output_dir, token):
    record = json.loads((Path(output_dir) / ".training_run.json").read_text())
    if record["token"] != token:
        raise ValueError("Training directory claim does not belong to this launch")


def resolve_resume_source(source):
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
