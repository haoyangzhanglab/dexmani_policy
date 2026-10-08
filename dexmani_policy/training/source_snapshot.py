"""Small, content-addressed snapshot of the code actually present at launch."""

import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import zipfile
from pathlib import Path
from dexmani_policy.utils.atomic import atomic_path


def save_source_snapshot(output_dir, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir)
    files = []
    for folder in ("dexmani_policy", "scripts"):
        for path in (root / folder).rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in {".py", ".yaml", ".yml", ".sh"}
                    and not any(part.startswith(".") or part == "__pycache__"
                                for part in path.relative_to(root).parts)):
                files.append(path)
    for pattern in ("pyproject.toml", "setup.py", "setup.cfg", "requirements*.txt", "environment*.y*ml", "*.lock"):
        files.extend(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
    hashes = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    def git(*args):
        # Launch copies may live inside the main checkout's ignored outputs/.
        # Parent-repository discovery would falsely describe another source tree.
        if not (root / ".git").exists():
            return None
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    dependencies = {}
    for name in ("torch", "numpy", "hydra-core", "zarr", "diffusers", "transformers", "peft", "timm"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    record = {"commit": git("rev-parse", "HEAD") or "unknown",
              "dirty": status != "" if status is not None else "unknown",
              "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
              "files": hashes, "python": platform.python_version(), "dependencies": dependencies}
    with atomic_path(output_dir / "source.zip", overwrite=False) as temporary:
        temporary.write_bytes(buffer.getvalue())
    with atomic_path(output_dir / "source_manifest.json", overwrite=False) as temporary:
        temporary.write_text(json.dumps(record, indent=2))
    return record
