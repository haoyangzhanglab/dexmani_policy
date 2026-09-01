from __future__ import annotations

import socket
import unittest
import urllib.request

try:
    from .network_guard import NetworkAccessForbidden, network_forbidden
except ImportError:
    from network_guard import NetworkAccessForbidden, network_forbidden


class NetworkGuardTest(unittest.TestCase):
    def test_socket_and_urllib_attempts_fail_immediately(self) -> None:
        with network_forbidden():
            with self.assertRaises(NetworkAccessForbidden):
                socket.create_connection(("127.0.0.1", 9))
            with self.assertRaises(NetworkAccessForbidden):
                urllib.request.urlopen("https://example.invalid")

    def test_guard_restores_entry_points(self) -> None:
        original = socket.create_connection
        with network_forbidden():
            self.assertIsNot(socket.create_connection, original)
        self.assertIs(socket.create_connection, original)


if __name__ == "__main__":
    unittest.main()
