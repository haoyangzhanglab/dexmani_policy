import os
import signal
import sys
import traceback
from contextlib import contextmanager


class StopRequest:
    def __init__(self):
        self.requested = False

    def __call__(self, signum, frame):
        name = signal.Signals(signum).name
        if self.requested:
            print(f"\nSecond {name} — forcing exit.", flush=True)
            os._exit(1)
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


def close_resources(*closers):
    """Run every closer; report secondary failures without replacing an active error."""
    active_error = sys.exc_info()[1]
    first_error = None
    for close in closers:
        try:
            close()
        except BaseException as error:
            if active_error is None and first_error is None:
                first_error = error
            else:
                print("Error while closing training resources:", file=sys.stderr)
                traceback.print_exception(type(error), error, error.__traceback__, chain=False)
    if first_error is not None:
        raise first_error
