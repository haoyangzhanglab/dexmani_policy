import signal
from contextlib import contextmanager


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
