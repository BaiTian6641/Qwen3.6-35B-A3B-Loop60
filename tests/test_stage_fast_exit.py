"""Regression tests for fast CLI stage exit (`flush_and_hard_exit` / `entrypoint`).

Verifies that lingering non-daemon background threads (such as `hf_xet` / `datasets`
streaming pools on Colab) cannot hang `python -m qloop ...` or `qloop.cli:entrypoint`
inside `threading._shutdown()`, while in-process `main(argv)` callers continue to receive
plain `int` return codes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from qloop.cli import entrypoint, flush_and_hard_exit, main

ROOT = Path(__file__).resolve().parent.parent


class _InterceptedHardExit(BaseException):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def test_pyproject_console_script_points_to_entrypoint() -> None:
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'qloop = "qloop.cli:entrypoint"' in raw
    assert 'qloop = "qloop.cli:main"' not in raw


def test_in_process_main_returns_int_without_hard_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(AssertionError(f"main() must not call os._exit({code})")),
    )

    assert main(["sync", "disk"]) == 0
    lm_cfg = str(ROOT / "configs" / "local" / "toy_lm.yaml")
    assert main(["train", "-c", lm_cfg, "-s", "stage_type=invalid", "-s", "run.name=bad_inproc"]) == 1


def test_flush_and_hard_exit_flushes_cuda_logs_stdio_and_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        synchronize=lambda: events.append("cuda.synchronize"),
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_stdout = SimpleNamespace(flush=lambda: events.append("stdout.flush"))
    fake_stderr = SimpleNamespace(flush=lambda: events.append("stderr.flush"))
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)
    monkeypatch.setattr("qloop.cli.logging.shutdown", lambda: events.append("logging.shutdown"))
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(_InterceptedHardExit(code)))
    monkeypatch.delenv("QLOOP_NO_FAST_EXIT", raising=False)

    with pytest.raises(_InterceptedHardExit) as exc_info:
        flush_and_hard_exit(0)
    assert exc_info.value.code == 0
    assert events[0] == "cuda.synchronize"
    assert "logging.shutdown" in events
    assert "stdout.flush" in events
    assert "stderr.flush" in events

    # Uninitialized CUDA must not call synchronize()
    events.clear()
    fake_cuda.is_initialized = lambda: False
    with pytest.raises(_InterceptedHardExit) as exc_uninit:
        flush_and_hard_exit(3)
    assert exc_uninit.value.code == 3
    assert "cuda.synchronize" not in events
    assert "logging.shutdown" in events

    # Exceptions inside synchronize() / shutdown() must be swallowed so os._exit(rc) still runs
    events.clear()
    fake_cuda.is_initialized = lambda: True

    def _boom() -> None:
        events.append("cuda.boom")
        raise RuntimeError("simulated CUDA error")

    fake_cuda.synchronize = _boom
    with pytest.raises(_InterceptedHardExit) as exc_boom:
        flush_and_hard_exit(5)
    assert exc_boom.value.code == 5
    assert "cuda.boom" in events
    assert "logging.shutdown" in events

    # QLOOP_NO_FAST_EXIT=1 / true / yes falls back to SystemExit(rc)
    for truthy in ("1", "true", "TRUE", "yes", " 1 "):
        monkeypatch.setenv("QLOOP_NO_FAST_EXIT", truthy)
        with pytest.raises(SystemExit) as sysexit:
            flush_and_hard_exit(4)
        assert sysexit.value.code == 4


def test_entrypoint_normalizes_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    monkeypatch.delenv("QLOOP_NO_FAST_EXIT", raising=False)
    monkeypatch.setattr("qloop.cli.logging.shutdown", lambda: None)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(_InterceptedHardExit(code)))

    with pytest.raises(_InterceptedHardExit) as ok_exit:
        entrypoint(["sync", "disk"])
    assert ok_exit.value.code == 0

    with pytest.raises(_InterceptedHardExit) as help_exit:
        entrypoint(["--help"])
    assert help_exit.value.code == 0

    with pytest.raises(_InterceptedHardExit) as bad_arg_exit:
        entrypoint(["--unknown-cli-flag"])
    assert bad_arg_exit.value.code == 2

    lm_cfg = str(ROOT / "configs" / "local" / "toy_lm.yaml")
    with pytest.raises(_InterceptedHardExit) as fail_exit:
        entrypoint(["train", "-c", lm_cfg, "-s", "stage_type=invalid", "-s", "run.name=ep_bad"])
    assert fail_exit.value.code == 1


def _make_nondaemon_site_dir(base: Path) -> Path:
    site_dir = base / "site_inject"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        "import os, pathlib, threading, time\n"
        "sleep_s = float(os.environ.get('QLOOP_TEST_NONDAEMON_SLEEP', '60'))\n"
        "th = threading.Thread(target=time.sleep, args=(sleep_s,), name='fake-hf-xet-pool', daemon=False)\n"
        "th.start()\n"
        "marker = os.environ.get('QLOOP_TEST_THREAD_MARKER')\n"
        "if marker:\n"
        "    pathlib.Path(marker).write_text(f'{th.ident}:{th.daemon}:{th.is_alive()}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    return site_dir


def _subproc_env(site_dir: Path, run_root: Path, marker: Path, *, sleep_s: float = 60.0) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("QLOOP_NO_FAST_EXIT", None)
    src_dir = str(ROOT / "src")
    existing_pp = env.get("PYTHONPATH", "")
    parts = [str(site_dir), src_dir, *([existing_pp] if existing_pp else [])]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["QLOOP_RUN_ROOT"] = str(run_root)
    env["QLOOP_TEST_NONDAEMON_SLEEP"] = str(sleep_s)
    env["QLOOP_TEST_THREAD_MARKER"] = str(marker)
    return env


def _assert_nondaemon_thread_started(marker: Path) -> None:
    assert marker.is_file(), "sitecustomize non-daemon thread marker was not written"
    ident_s, daemon_s, alive_s = marker.read_text(encoding="utf-8").strip().split(":")
    assert int(ident_s) > 0
    assert daemon_s == "False"
    assert alive_s == "True"


def test_subprocess_fast_exit_with_live_non_daemon_thread(tmp_path: Path) -> None:
    site_dir = _make_nondaemon_site_dir(tmp_path)
    run_root = tmp_path / "runs"
    lm_cfg = str(ROOT / "configs" / "local" / "toy_lm.yaml")

    # 1a. `python -m qloop sync disk` with live 60s non-daemon thread -> rc=0 in < 3s + flushed stdout & run.log
    m_sync = tmp_path / "marker_sync.txt"
    t0 = time.monotonic()
    p_sync = subprocess.run(
        [sys.executable, "-m", "qloop", "sync", "disk"],
        env=_subproc_env(site_dir, run_root, m_sync, sleep_s=60.0),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_sync = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_sync)
    assert p_sync.returncode == 0, f"stdout={p_sync.stdout}\nstderr={p_sync.stderr}"
    assert dt_sync < 3.0, f"expected fast exit < 3s despite 60s non-daemon thread, took {dt_sync:.2f}s"
    assert json.loads(p_sync.stdout[p_sync.stdout.index("{") :])["total_gib"] > 0
    assert (run_root / "sync" / "run.log").is_file()

    # 1b. `python -m qloop train ...` (exact production stage command) with 60s non-daemon thread -> rc=0 in < 3s
    m_train = tmp_path / "marker_train.txt"
    t0 = time.monotonic()
    p_train = subprocess.run(
        [
            sys.executable,
            "-m",
            "qloop",
            "train",
            "-c",
            lm_cfg,
            "-s",
            "train.steps=0",
            "-s",
            "train.grad_accum=1",
            "-s",
            "train.eval_every=0",
            "-s",
            "eval_quick.n_batches=0",
            "-s",
            "data.decontaminate.enabled=false",
            "-s",
            "data.seq_len=16",
            "-s",
            "run.name=fast_exit_train",
        ],
        env=_subproc_env(site_dir, run_root, m_train, sleep_s=60.0),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_train = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_train)
    assert p_train.returncode == 0, f"stdout={p_train.stdout}\nstderr={p_train.stderr}"
    assert dt_train < 3.0, f"expected train fast exit < 3s despite 60s non-daemon thread, took {dt_train:.2f}s"
    assert 'done: {"step": 0' in p_train.stdout
    assert (run_root / "fast_exit_train" / "run.log").is_file()
    assert 'done: {"step": 0' in (run_root / "fast_exit_train" / "run.log").read_text(encoding="utf-8")

    # 2. `qloop.cli:entrypoint` console-script target with 60s non-daemon thread -> rc=0 in < 3s + flushed stdout
    m_ep = tmp_path / "marker_ep.txt"
    t0 = time.monotonic()
    p_ep = subprocess.run(
        [sys.executable, "-c", "from qloop.cli import entrypoint; entrypoint(['sync', 'disk'])"],
        env=_subproc_env(site_dir, run_root, m_ep, sleep_s=60.0),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_ep = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_ep)
    assert p_ep.returncode == 0, p_ep.stderr
    assert dt_ep < 3.0, f"entrypoint() took {dt_ep:.2f}s"
    assert json.loads(p_ep.stdout[p_ep.stdout.index("{") :])["total_gib"] > 0

    # 3. Runtime failure (`rc=1`) with 60s non-daemon thread -> propagates rc=1 in < 3s + crash report + stderr
    m_err1 = tmp_path / "marker_rc1.txt"
    t0 = time.monotonic()
    p_err1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "qloop",
            "train",
            "-c",
            lm_cfg,
            "-s",
            "stage_type=invalid",
            "-s",
            "run.name=fast_exit_rc1",
        ],
        env=_subproc_env(site_dir, run_root, m_err1, sleep_s=60.0),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_err1 = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_err1)
    assert p_err1.returncode == 1
    assert dt_err1 < 3.0, f"rc=1 fast exit took {dt_err1:.2f}s"
    assert "FAILED in stage" in p_err1.stderr
    assert list((run_root / "fast_exit_rc1" / "crash").glob("crash_*.md"))

    # 4. Argparse error (`SystemExit(2)`) with 60s non-daemon thread -> propagates rc=2 in < 3s + stderr flushed
    m_err2 = tmp_path / "marker_rc2.txt"
    t0 = time.monotonic()
    p_err2 = subprocess.run(
        [sys.executable, "-m", "qloop", "--definitely-not-a-valid-flag"],
        env=_subproc_env(site_dir, run_root, m_err2, sleep_s=60.0),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_err2 = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_err2)
    assert p_err2.returncode == 2
    assert dt_err2 < 3.0, f"rc=2 fast exit took {dt_err2:.2f}s"
    assert "qloop: error:" in p_err2.stderr

    # 5. `QLOOP_NO_FAST_EXIT=1` falls back to normal SystemExit (joins non-daemon thread before exiting)
    m_slow = tmp_path / "marker_slow.txt"
    slow_env = _subproc_env(site_dir, run_root, m_slow, sleep_s=0.55)
    slow_env["QLOOP_NO_FAST_EXIT"] = "1"
    t0 = time.monotonic()
    p_slow = subprocess.run(
        [sys.executable, "-m", "qloop", "--help"],
        env=slow_env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dt_slow = time.monotonic() - t0
    _assert_nondaemon_thread_started(m_slow)
    assert p_slow.returncode == 0
    assert "usage: qloop" in p_slow.stdout
    assert dt_slow >= 0.45, f"QLOOP_NO_FAST_EXIT=1 should wait for non-daemon thread (~0.55s), took {dt_slow:.2f}s"
