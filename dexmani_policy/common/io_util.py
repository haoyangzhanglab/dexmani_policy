import json
import logging
import os
import sys
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Publish a temporary sibling path atomically after the writer succeeds."""
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".dexmani-publish-", suffix=".tmp")
    os.close(fd)
    temporary = Path(name)

    try:
        yield temporary
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning(
                "Could not remove temporary file %s", temporary, exc_info=True,
            )
        raise


def write_json(path, record):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def close_resources(*closers):
    """Run every closer; preserve an active error and report secondary failures."""
    active_error = sys.exc_info()[1]
    # Generator.close() is a normal shutdown, so cleanup failures must propagate.
    if isinstance(active_error, GeneratorExit):
        active_error = None
    first_error = None
    for close in closers:
        try:
            close()
        except BaseException as error:
            if active_error is None and first_error is None:
                first_error = error
            else:
                print("Error while closing resources:", file=sys.stderr)
                traceback.print_exception(type(error), error, error.__traceback__, chain=False)
    if first_error is not None:
        raise first_error
