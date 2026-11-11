import signal
from contextlib import contextmanager


@contextmanager
def training_signal_handlers(handler):
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handler)
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)
