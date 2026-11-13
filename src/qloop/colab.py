"""Notebook-side helpers (no torch import): keep the Colab notebook to a handful of lines.

    from qloop.colab import setup_env, q
    setup_env(run_root="/content/drive/MyDrive/qloop_runs")   # secrets -> env, drive paths
    q("doctor")
    q("train -c configs/stages/s1_loop_heal.yaml --debug 2")

`q(...)` runs `python -m qloop ...` in a *subprocess* (fresh CUDA context per stage, debug env flags
applied, a crash can never wedge the kernel), streams its output live, and on failure prints the
hints + traceback section of the crash report it wrote, so the error is visible in the cell output.
"""
from __future__ import annotations

import glob
import os
import shlex
import subprocess
import sys
import time

SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "TEACHER_API_KEY", "DAYTONA_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
           "OPENAI_API_KEY")


def setup_env(run_root: str | None = None, hf_home: str | None = None) -> dict[str, str]:
    """Copy Colab secrets into os.environ (missing ones are reported, not fatal) and set paths."""
    found = {}
    try:
        from google.colab import userdata

        for k in SECRETS:
            try:
                v = userdata.get(k)
            except Exception:
                v = None
            if v:
                os.environ[k] = v
                found[k] = "set"
    except ImportError:
        pass
    for k in SECRETS:
        found.setdefault(k, "set" if os.environ.get(k) else "missing")
    if run_root:
        os.makedirs(run_root, exist_ok=True)
        os.environ["QLOOP_RUN_ROOT"] = run_root
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("secrets:", ", ".join(f"{k}={v}" for k, v in found.items()))
    print("QLOOP_RUN_ROOT =", os.environ.get("QLOOP_RUN_ROOT", "runs (local)"))
    return found


def _newest_crash(since: float) -> str | None:
    root = os.environ.get("QLOOP_RUN_ROOT", "runs")
    files = [f for f in glob.glob(os.path.join(root, "**", "crash", "crash_*.md"), recursive=True)
             if os.path.getmtime(f) >= since]
    return max(files, key=os.path.getmtime) if files else None


def _sections(path: str, names: tuple[str, ...], max_lines: int = 80) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line[3:].strip() in names
        if keep or line.startswith(("# qloop crash", "**stage**", "**error**")):
            out.append(line)
    return "\n".join(out[:max_lines])


def q(args: str, check: bool = True) -> int:
    """Run `python -m qloop <args>` with live output; surface the crash report on failure."""
    cmd = [sys.executable, "-u", "-m", "qloop", *shlex.split(args)]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(2)
        rc = proc.wait()
        print("\ninterrupted - the stage checkpointed before exiting; re-run the cell to resume")
    dt = time.time() - t0
    if rc != 0:
        crash = _newest_crash(t0 - 1)
        print(f"\n✗ exit code {rc} after {dt / 60:.1f} min")
        if crash:
            print(f"crash report: {crash}\n" + "-" * 80)
            print(_sections(crash, ("Hints", "Traceback")))
        if check:
            raise RuntimeError(f"qloop {args!r} failed (rc={rc})" + (f"; see {crash}" if crash else ""))
    else:
        print(f"\n✓ done in {dt / 60:.1f} min")
    return rc


def tail_metrics(run_name: str, keys: tuple[str, ...] = ("loss/lm", "eval/loss_d1", "eval/loss_d2",
                                                          "eval/depth_gain", "tok_per_s"), n: int = 10) -> None:
    import json

    path = os.path.join(os.environ.get("QLOOP_RUN_ROOT", "runs"), run_name, "metrics.jsonl")
    if not os.path.exists(path):
        print("no metrics yet:", path)
        return
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows[-n:]:
        print(r["step"], {k: round(r[k], 4) for k in keys if isinstance(r.get(k), (int, float))})
