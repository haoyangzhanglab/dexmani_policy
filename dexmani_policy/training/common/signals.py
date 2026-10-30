import signal
from contextlib import contextmanager
from functools import partial

from dexmani_policy.common.io_util import close_resources


@contextmanager
def training_signal_handlers(handler):
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handler)
        yield
    finally:
        close_resources(*(
            partial(signal.signal, signum, old_handler)
            for signum, old_handler in previous.items()
        ))
