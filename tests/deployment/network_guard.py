"""Test-only fail-fast guard for deployment restore network access."""

from __future__ import annotations

import importlib
import socket
import urllib.request
from contextlib import ExitStack, contextmanager
from typing import Iterator
from unittest.mock import patch

import torch


class NetworkAccessForbidden(AssertionError):
    """Raised when deployment qualification attempts a network entry point."""


def _forbidden(*args, **kwargs):
    del args, kwargs
    raise NetworkAccessForbidden(
        "network access is forbidden during deployment restore"
    )


@contextmanager
def network_forbidden() -> Iterator[None]:
    """Block common Python download paths for the duration of a restore.

    Optional downloader packages are patched only when installed.  Socket and
    urllib are always guarded, so an unlisted client still fails at connection.
    """

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", _forbidden))
        stack.enter_context(patch.object(socket, "create_connection", _forbidden))
        stack.enter_context(patch.object(urllib.request, "urlopen", _forbidden))
        stack.enter_context(patch.object(urllib.request, "urlretrieve", _forbidden))
        stack.enter_context(patch.object(torch.hub, "download_url_to_file", _forbidden))
        stack.enter_context(
            patch.object(torch.hub, "load_state_dict_from_url", _forbidden)
        )

        optional_targets = (
            ("requests.sessions", "Session", "request"),
            ("huggingface_hub", None, "hf_hub_download"),
            ("gdown", None, "download"),
        )
        for module_name, owner_name, attribute in optional_targets:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            owner = getattr(module, owner_name) if owner_name else module
            if hasattr(owner, attribute):
                stack.enter_context(patch.object(owner, attribute, _forbidden))
        yield
