"""Notebook-side helpers (no torch import): keep the single Colab notebook concise and debuggable.

    from qloop.colab import setup_env, q, hub_status, run_pipeline, tail_metrics
    setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64")
    q("doctor --freeze requirements/colab.lock.txt --model Qwen/Qwen3.6-35B-A3B")
    hub_status()
    run_pipeline(debug=1, time_budget_hours=21.5)

Design:
* Zero Google Drive dependency: every stage commits its trainable checkpoint + `manifest.json` +
  auto-rendered `README.md` directly to `TassP0lar/Qwen3.6-35B-A3B-Loop64` via `HF_TOKEN`, then
  prunes superseded local checkpoints and HF upload caches to stay comfortably within Colab's
  230 GB SSD limit.
* `q(...)` executes `python -u -m qloop ...` in a *fresh subprocess* (clean CUDA context per stage,
  `CUDA_LAUNCH_BLOCKING` / `TORCH_SHOW_CPP_STACKTRACES` applied before CUDA init), streams stdout
  live, and on failure prints the structured crash report (`Hints`, `Traceback`, `Local Tensors`,
  `CUDA Memory`) inline in the notebook output.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence

DEFAULT_HF_REPO = "TassP0lar/Qwen3.6-35B-A3B-Loop64"
SECRETS = (
    "HF_TOKEN",
    "WANDB_API_KEY",
    "TEACHER_API_KEY",
    "DAYTONA_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "OPENAI_API_KEY",
)
DEFAULT_METRIC_KEYS = (
    "loss/lm",
    "eval/loss_d1",
    "eval/loss_d2",
    "eval/depth_gain",
    "loop/reroute_rate",
    "loop/unique_experts_core",
    "loop/effective_expert_mult",
    "lsa/distill_kl",
    "reward/mean",
    "gate/score",
    "tok_per_s",
    "mem/peak_gib",
)


def setup_env(
    hf_repo: str = DEFAULT_HF_REPO,
    run_root: str = "/content/runs",
    hf_home: str = "/content/hf_cache",
    *,
    require_hf_token: bool = True,
) -> dict[str, str]:
    """Load Colab Secrets into `os.environ`, configure HF Hub sync + local SSD paths, and print disk budget."""
    found: dict[str, str] = {}
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        for k in SECRETS:
            try:
                v = userdata.get(k)
            except Exception:
                v = None
            if v and str(v).strip():
                os.environ[k] = str(v).strip()
                found[k] = "set (Colab Secret)"
    except ImportError:
        pass
    for k in SECRETS:
        found.setdefault(k, "set" if os.environ.get(k) else "missing")

    if require_hf_token and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is missing! In Google Colab, click the left-sidebar '🔑 Secrets' icon, add a secret named "
            f"'HF_TOKEN' with Write permission for https://huggingface.co/{hf_repo}, toggle 'Notebook access' ON, "
            "and re-run this cell."
        )

    os.makedirs(run_root, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_pp = os.environ.get("PYTHONPATH", "")
    if src_dir not in cur_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{src_dir}{os.pathsep}{cur_pp}" if cur_pp else src_dir
    os.environ["QLOOP_HF_REPO"] = hf_repo
    os.environ["QLOOP_RUN_ROOT"] = run_root
    os.environ["HF_HOME"] = hf_home
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from .hub import disk_report

    rep = disk_report(run_root, hf_home=hf_home)
    print(f"HF Sync Repo   : https://huggingface.co/{hf_repo} (public commit-history sync)")
    print(f"Secrets        : {', '.join(f'{k}={v}' for k, v in found.items())}")
    print(f"Run Root (SSD) : {run_root}  |  HF_HOME : {hf_home}")
    print(
        f"Disk (230G cap): {rep['free_gib']:.1f} GiB free / {rep['total_gib']:.1f} GiB total "
        f"(HF base weights={rep['hf_hub_weights_gib']:.1f} GiB, runs={rep['runs_gib']:.2f} GiB)"
    )
    return found


def _newest_crash(since: float) -> str | None:
    root = os.environ.get("QLOOP_RUN_ROOT", "runs")
    files = [
        f
        for f in glob.glob(os.path.join(root, "**", "crash", "crash_*.md"), recursive=True)
        if os.path.getmtime(f) >= since
    ]
    return max(files, key=os.path.getmtime) if files else None


def _sections(path: str, names: tuple[str, ...], max_lines: int = 120) -> str:
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
    """Run `python -u -m qloop <args>` in a subprocess; surface verbose crash diagnostics on failure."""
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
        repo = os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
        print(f"\ninterrupted - checkpoint saved & synced to https://huggingface.co/{repo}; re-run cell to resume")
    dt = time.time() - t0
    if rc != 0:
        crash = _newest_crash(t0 - 1)
        print(f"\n✗ exit code {rc} after {dt / 60:.1f} min")
        if crash:
            print(f"crash report: {crash}\n" + "=" * 88)
            print(_sections(crash, ("Hints", "Traceback", "Local Tensors", "CUDA Memory")))
            print("=" * 88)
            print("Tip: re-run with `--debug 2` (NaN/Inf module hooks + C++ traces) or `--debug 3` "
                  "(CUDA_LAUNCH_BLOCKING=1 + autograd anomaly + memory timeline).")
        if check:
            raise RuntimeError(f"qloop {args!r} failed (rc={rc})" + (f"; see {crash}" if crash else ""))
    else:
        print(f"\n✓ done in {dt / 60:.1f} min")
    return rc


def hub_status(hf_repo: str | None = None) -> int:
    """Print live stage status, recent Hub commits, and 230 GB Colab disk headroom."""
    repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    return q(f"sync status --repo {shlex.quote(repo)}")


def reclaim_disk(keep: str | None = None, pip: bool = False) -> int:
    """Reclaim superseded local stage checkpoints & HF upload staging caches."""
    parts = ["sync", "disk", "--clean"]
    if keep:
        parts += ["--keep", shlex.quote(keep)]
    if pip:
        parts.append("--pip")
    return q(" ".join(parts))


def run_pipeline(
    *,
    hf_repo: str | None = None,
    debug: int = 1,
    time_budget_hours: float = 21.5,
    stages: Sequence[str] | None = None,
    until: str | None = None,
    force: bool = False,
    overrides: Sequence[str] = (),
) -> int:
    """Run/resume the full end-to-end pipeline synced via HF Hub commit history."""
    repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    parts = [
        "sync",
        "pipeline",
        "--repo",
        shlex.quote(repo),
        "--debug",
        str(int(debug)),
        "--time-budget-hours",
        str(float(time_budget_hours)),
    ]
    if stages:
        parts += ["--stages", shlex.quote(",".join(stages))]
    if until:
        parts += ["--until", shlex.quote(until)]
    if force:
        parts.append("--force")
    for ov in overrides:
        parts += ["-s", shlex.quote(ov)]
    return q(" ".join(parts))


def tail_metrics(
    run_name: str,
    keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    n: int = 12,
) -> None:
    path = os.path.join(os.environ.get("QLOOP_RUN_ROOT", "runs"), run_name, "metrics.jsonl")
    if not os.path.exists(path):
        print("no local metrics file:", path)
        return
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows[-n:]:
        filtered = {k: round(r[k], 4) for k in keys if isinstance(r.get(k), (int, float))}
        print(f"step {r.get('step', '?'):>6}:", filtered)
