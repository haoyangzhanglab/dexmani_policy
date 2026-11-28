import fcntl
import signal
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def lock_run_directory(path, *, create=False):
    """Hold exclusive write access for the whole run, including checkpoint loading."""
    directory = Path(path).expanduser().resolve()
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    # Keep the same inode across runs so competing processes lock the same file.
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training run already has an active writer: {directory}") from error
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class StopRequest:
    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        if not self.requested:
            name = signal.Signals(signum).name
            print(f"\n{name} — stopping and saving at the next update boundary.", flush=True)
            self.requested = True


@contextmanager
def training_signals():
    stop = StopRequest()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        yield stop
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
