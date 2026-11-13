"""Sandboxed-ish Python test execution for LiveCodeBench-style problems.

Each test runs the candidate program in a fresh `python` subprocess (own temp dir, timeout,
process-group kill, address-space limit on Linux). This is the same isolation level as the
official LCB harness (it is NOT a security sandbox: only run on disposable machines like Colab).

Test case format (LCB):  {"input": str, "output": str, "testtype": "stdin" | "functional"}
  * stdin:      program reads stdin, stdout compared token-wise (floats with tolerance)
  * functional: `input` holds one JSON value per line = positional args of
                `Solution().<func_name>(...)` (or a top-level function), output is JSON.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Sequence
from typing import Any

from .scoring import Scorer, strip_thinking

log = logging.getLogger("qloop")

LCB_PRELUDE = (
    "import sys, os, math, heapq, bisect, itertools, functools, collections, string, re, random, operator\n"
    "from typing import *\nfrom collections import *\nfrom functools import lru_cache, reduce, cache\n"
    "from itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom math import *\n"
    "sys.setrecursionlimit(1 << 20)\n"
)

_FUNC_HARNESS = r'''
import json, sys
src = open(sys.argv[1], encoding="utf-8").read()
fn_name = sys.argv[2]
g = {"__name__": "__qloop_solution__"}
exec(compile(src, "solution.py", "exec"), g)
args = [json.loads(line) for line in sys.stdin.read().split("\n") if line.strip()]
if "Solution" in g:
    fn = getattr(g["Solution"](), fn_name)
else:
    fn = g[fn_name]
res = fn(*args)
sys.stdout.write("\n__QLOOP_RESULT__" + json.dumps(res) + "\n")
'''


def extract_code(text: str, lang: str = "python", must_contain: str | None = None) -> str | None:
    """Last fenced code block (preferring `lang` / unlabeled blocks, optionally containing a marker)."""
    t = strip_thinking(text)
    blocks = [(m.group(1).strip().lower(), m.group(2)) for m in
              re.finditer(r"```([\w+.-]*)[ \t]*\n(.*?)```", t, flags=re.S)]
    if not blocks:  # unterminated final block (truncated generation)
        m = re.search(r"```([\w+.-]*)[ \t]*\n(.*)$", t, flags=re.S)
        if m:
            blocks = [(m.group(1).strip().lower(), m.group(2))]
    good = [b for lab, b in blocks if lab in ("", lang, "py", "python3")] or [b for _, b in blocks]
    if must_contain:
        good = [b for b in good if must_contain in b]
    return good[-1].strip("\n") if good else None


def decode_lcb_tests(raw: Any) -> list[dict[str, Any]]:
    """LCB public tests are JSON; private tests are base64(zlib(pickle(json-string)))."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    data = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))  # official LCB format
    return json.loads(data) if isinstance(data, (str, bytes)) else data


def _limit_code(memory_mb: int) -> str:
    """Prepended to the solution: address-space limit inside the child (Linux only; no preexec_fn)."""
    if not memory_mb:
        return ""
    return ("try:\n    import resource as _r, sys as _s\n    if _s.platform.startswith('linux'):\n"
            f"        _r.setrlimit(_r.RLIMIT_AS, ({memory_mb} << 20, {memory_mb} << 20))\n"
            "except Exception:\n    pass\n")


def _run(cmd: list[str], stdin: str, timeout: float, cwd: str) -> tuple[int | None, str, str]:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                            text=True, start_new_session=True,
                            env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        return None, out or "", err or ""


def _tok_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def outputs_match(got: str, expected: str) -> bool:
    gl = [ln.strip() for ln in got.strip().splitlines()]
    el = [ln.strip() for ln in expected.strip().splitlines()]
    if gl == el:
        return True
    gt, et = " ".join(gl).split(), " ".join(el).split()
    return len(gt) == len(et) and all(_tok_equal(x, y) for x, y in zip(gt, et))


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return _tok_equal(str(a), str(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return a == b


def run_python_tests(code: str, tests: Sequence[dict[str, Any]], fn_name: str | None = None, timeout: float = 6.0,
                     stop_on_fail: bool = True, memory_mb: int = 4096, python: str = sys.executable,
                     prelude: str = LCB_PRELUDE) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qloop_code_") as tmp:
        sol = os.path.join(tmp, "solution.py")
        with open(sol, "w", encoding="utf-8") as f:
            f.write(_limit_code(memory_mb) + prelude + "\n" + code + "\n")
        harness = os.path.join(tmp, "harness.py")
        with open(harness, "w", encoding="utf-8") as f:
            f.write(_FUNC_HARNESS)
        for i, tc in enumerate(tests):
            functional = tc.get("testtype") == "functional" or (fn_name and tc.get("testtype") != "stdin")
            if functional:
                if not fn_name:
                    results.append({"i": i, "ok": False, "error": "functional test without func_name"})
                    break
                rc, out, err = _run([python, harness, sol, fn_name], tc["input"], timeout, tmp)
            else:
                rc, out, err = _run([python, sol], tc["input"], timeout, tmp)
            res: dict[str, Any] = {"i": i, "ok": False}
            if rc is None:
                res["error"] = f"timeout after {timeout}s"
            elif rc != 0:
                res["error"] = f"exit code {rc}: {err.strip()[-800:]}"
            elif functional:
                marker = out.rfind("__QLOOP_RESULT__")
                if marker < 0:
                    res["error"] = "no result printed"
                else:
                    got = json.loads(out[marker + len("__QLOOP_RESULT__"):].strip().split("\n")[0])
                    try:
                        exp = json.loads(tc["output"])
                    except ValueError:
                        exp = tc["output"]
                    res["ok"] = _json_equal(got, exp)
                    if not res["ok"]:
                        res["error"] = f"wrong answer: got {str(got)[:200]!r} expected {str(exp)[:200]!r}"
            else:
                res["ok"] = outputs_match(out, tc["output"])
                if not res["ok"]:
                    res["error"] = f"wrong answer: got {out.strip()[:200]!r} expected {tc['output'].strip()[:200]!r}"
            results.append(res)
            if stop_on_fail and not res["ok"]:
                break
    passed = sum(r["ok"] for r in results)
    return {"passed": passed, "total": len(tests), "all_passed": passed == len(tests) and len(tests) > 0,
            "first_error": next((r.get("error") for r in results if not r["ok"]), None)}


def _lcb_eval(text: str, item: dict[str, Any]) -> dict[str, Any]:
    code = extract_code(text)
    if code is None:
        return {"score": 0.0, "pred": None, "error": "no code block found"}
    tests = list(item.get("public_tests") or []) + list(item.get("private_tests") or [])
    if not tests:
        return {"score": 0.0, "pred": None, "error": "item has no test cases"}
    res = run_python_tests(code, tests, item.get("fn_name") or None, timeout=float(item.get("timeout", 6.0)))
    return {"score": float(res["all_passed"]), "pred": None, "passed": res["passed"], "total": res["total"],
            "error": res["first_error"]}


LCB_SCORER = Scorer("lcb", lambda t, it: None, lambda p, it: False, evaluate_fn=_lcb_eval)
