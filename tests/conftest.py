"""Make `src/` importable without installing the package (dev laptop has no build deps)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
