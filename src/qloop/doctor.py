"""Top-level alias for :mod:`qloop.env.doctor` (`python -m qloop.doctor`)."""
from __future__ import annotations

import sys

from .env.doctor import (
    DEFAULT_MODEL,
    Check,
    build_parser,
    check_hf_access,
    check_packages,
    check_python,
    check_system,
    check_torch_gpu,
    freeze,
    main,
    run_doctor,
)

__all__ = [
    "DEFAULT_MODEL",
    "Check",
    "build_parser",
    "check_hf_access",
    "check_packages",
    "check_python",
    "check_system",
    "check_torch_gpu",
    "freeze",
    "main",
    "run_doctor",
]


if __name__ == "__main__":
    sys.exit(main())
