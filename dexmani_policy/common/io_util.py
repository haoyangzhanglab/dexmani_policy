import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def cleanup_after_error(cleanup, error):
    try:
        cleanup()
    except BaseException:
        # Closing a paused generator is a normal exit for its resource owner.
        if isinstance(error, GeneratorExit):
            raise
        logging.getLogger(__name__).exception("Secondary error during cleanup")


@contextmanager
def cleanup_on_exit(cleanup):
    try:
        yield
    except BaseException as error:
        cleanup_after_error(cleanup, error)
        raise
    else:
        cleanup()


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
    except BaseException as error:
        cleanup_after_error(lambda: temporary.unlink(missing_ok=True), error)
        raise


def write_json(path, record):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
