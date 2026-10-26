import os
import sys
import tempfile
import traceback
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
            # A hard link publishes atomically without replacing an existing target.
            os.link(temporary, path)
    finally:
        close_resources(lambda: temporary.unlink(missing_ok=True))


def close_resources(*callbacks):
    """Close owned resources; report secondary errors and propagate the first failure."""
    primary = sys.exc_info()[1]
    failure = None
    for close in callbacks:
        try:
            close()
        except BaseException as error:
            if primary is None and failure is None:
                failure = error
            else:
                traceback.print_exc()
    if failure is not None:
        raise failure
