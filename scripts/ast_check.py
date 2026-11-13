#!/usr/bin/env python3
"""AST check: parse every tracked Python file (+ code cells of notebooks).

Catches syntax errors that only show up on the target interpreter, e.g. code written
on Colab (3.12) that breaks local dev (3.9) or vice versa. Exit code 1 on any failure.

    python scripts/ast_check.py [--target 3.9]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "runs", "__pycache__", ".ruff_cache"}


def iter_files(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if f.endswith((".py", ".ipynb")):
                yield os.path.join(d, f)


def check_source(src: str, name: str, feature_version) -> str:
    try:
        ast.parse(src, filename=name, feature_version=feature_version)
    except SyntaxError as e:
        return f"{name}:{e.lineno}:{e.offset}: {e.msg}"
    return ""


def notebook_sources(path: str):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # drop IPython magics / shell escapes before parsing
        lines = ["pass  # magic" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines()]
        yield f"{path}#cell{i}", "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--target", default="3.9", help="minimum python grammar version")
    args = ap.parse_args()
    major, minor = (int(x) for x in args.target.split("."))
    errors, n = [], 0
    for path in iter_files(args.root):
        if path.endswith(".ipynb"):
            for name, src in notebook_sources(path):
                n += 1
                err = check_source(src, name, (major, minor))
                if err:
                    errors.append(err)
        else:
            n += 1
            with open(path, encoding="utf-8") as f:
                err = check_source(f.read(), path, (major, minor))
            if err:
                errors.append(err)
    for e in errors:
        print(e)
    print(f"ast-check: {n} sources, {len(errors)} errors (grammar {args.target})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
