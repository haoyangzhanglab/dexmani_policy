"""Publish complete files from private temporaries in the destination directory."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path, *, overwrite=True, suffix=".tmp"):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".dexmani-publish-", suffix=suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_symlink():
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # link is atomic and refuses existing targets, unlike replace.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
