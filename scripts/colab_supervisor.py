#!/usr/bin/env python3
"""CLI entrypoint for `qloop.colab_supervisor` runnable with stdlib macOS `/usr/bin/python3`."""
from __future__ import annotations

import os
import sys

_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from qloop.colab_supervisor import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
