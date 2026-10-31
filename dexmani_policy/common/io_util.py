import json
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(
    path: str | os.PathLike[str], *, overwrite: bool = True, suffix: str = ".tmp"
) -> Iterator[Path]:
    """Publish a temporary sibling path atomically after the writer succeeds."""
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
            # Hard links fail atomically when the destination already exists.
            os.link(temporary, path)
    finally:
        close_resources(lambda: temporary.unlink(missing_ok=True))


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


def close_resources(*callbacks: Callable[[], object]) -> None:
    """Run all callbacks; preserve an active exception or raise the first failure."""
    active_error = sys.exc_info()[1]
    first_error = None

    for callback in callbacks:
        try:
            callback()
        except BaseException as error:
            if active_error is None and first_error is None:
                first_error = error
            else:
                traceback.print_exc()

    if first_error is not None:
        raise first_error
