"""Make `src/` importable without installing the package (dev laptop has no build deps)."""
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def loopback_ok() -> bool:
    """True if this process may bind AND connect to 127.0.0.1 (false inside network sandboxes)."""
    try:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        cli = socket.create_connection(srv.getsockname(), timeout=2)
        cli.close()
        srv.close()
        return True
    except OSError:
        return False


LOOPBACK_OK = loopback_ok()
