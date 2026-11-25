"""Unit & integration tests for `qloop.colab_supervisor` and 25-step stage checkpoint configs."""
from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import stat
import subprocess
import urllib.error
import urllib.request
from typing import Any

import pytest

from qloop.colab_supervisor import (
    BOOTSTRAP_SENTINEL,
    DEFAULT_MAX_ROLLOVERS,
    PROBE_SENTINEL,
    TAIL_SENTINEL,
    ColabCommandError,
    ColabSupervisor,
    ColabSupervisorError,
    ProbeResult,
    SupervisorConfig,
    build_arg_parser,
    build_bootstrap_python_script,
    build_probe_python_script,
    build_remote_run_pipeline_script,
    build_tail_python_script,
    config_from_args,
    fetch_hub_manifest_dict,
    is_session_missing_output,
    main,
    normalize_git_https_url,
    parse_active_sessions,
    redact_argv,
    redact_secret,
    resolve_local_hf_token,
)
from qloop.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee zero live HTTP requests occur during supervisor unit tests."""

    def _fail_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"unexpected live HTTP call during unit test: {args!r}")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)


def _cp(
    args: list[str],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _probe_stdout(
    *,
    pid_alive: bool,
    exit_code: int | None = None,
    gpu_summary: str = "NVIDIA RTX PRO 6000 Blackwell util=98% mem=64555/97887MiB pwr=418W temp=61C",
    last_log_lines: list[str] | None = None,
    disk_free_gib: float = 168.4,
    with_sentinel: bool = True,
) -> str:
    payload = {
        "pid_alive": pid_alive,
        "exit_code": exit_code,
        "gpu_summary": gpu_summary,
        "last_log_lines": last_log_lines if last_log_lines is not None else ["step 25/1000 loss/lm=1.0412"],
        "disk_free_gib": disk_free_gib,
    }
    raw = json.dumps(payload)
    return f"{PROBE_SENTINEL}{raw}\n" if with_sentinel else f"{raw}\n"


def _make_cfg(tmp_path: Any, **overrides: Any) -> SupervisorConfig:
    defaults: dict[str, Any] = {
        "colab_bin": "/fake/bin/colab",
        "session": "loop64",
        "gpu": "G4",
        "repo": "TassP0lar/Qwen3.6-35B-A3B-Loop64",
        "stages": ("s0_probe", "s1_loop_a6b", "s1b_hssd_distill"),
        "interval_sec": 300.0,
        "exec_timeout_sec": 180,
        "bootstrap_timeout_sec": 600,
        "max_respawns": 3,
        "respawn_backoff_sec": 15.0,
        "log_file": str(tmp_path / "runs" / "colab_supervisor.log"),
        "tail_file": str(tmp_path / "runs" / "colab_last_tail.log"),
    }
    defaults.update(overrides)
    return SupervisorConfig(**defaults)


# --------------------------------------------------------------------------- (a) Secret hygiene & redaction


def test_secret_resolution_and_total_redaction(tmp_path, capsys):
    secret = "hf_SuperSecretWriteTokenValue987654321"
    custom_secret = "custom_non_hf_secret_value_xyz123"

    # 1. Env resolution takes precedence; launchctl fallback works when env is unset.
    assert resolve_local_hf_token(env={"HF_TOKEN": f"  {secret} \n"}) == secret

    def fake_launchctl(cmd, **kwargs):
        assert cmd == ["launchctl", "getenv", "HF_TOKEN"]
        assert kwargs.get("stdin") == subprocess.DEVNULL
        return _cp(cmd, 0, stdout=f"{secret}\n")

    assert resolve_local_hf_token(env={}, runner=fake_launchctl) == secret

    # 2. Inline, URL-encoded, Bearer, and argv redaction.
    leaked = (
        f"Authorization: Bearer {secret} and HF_TOKEN={custom_secret} "
        f"and url={secret.replace('_', '%5F')}"
    )
    scrubbed = redact_secret(leaked, secrets=(secret, custom_secret))
    assert secret not in scrubbed
    assert custom_secret not in scrubbed
    assert "<REDACTED>" in scrubbed

    argv = ["/fake/bin/colab", "exec", "-s", "loop64", "--env", f"HF_TOKEN={secret}"]
    safe_argv = redact_argv(argv, secrets=(secret,))
    assert secret not in " ".join(safe_argv)
    assert "HF_TOKEN=<REDACTED>" in safe_argv

    # 3. Remote scripts never embed HF_TOKEN in source or command lines (`ps aux` safe).
    cfg = _make_cfg(tmp_path)
    wrapper_sh = build_remote_run_pipeline_script(cfg)
    boot_py = build_bootstrap_python_script(cfg)
    assert secret not in wrapper_sh
    assert secret not in boot_py
    assert 'export HF_TOKEN="$(cat /content/.hf_token)"' in wrapper_sh

    # 4. Even when bootstrap `colab exec --env HF_TOKEN=...` fails or times out,
    #    neither logs, stdout, stderr, nor exception attributes/messages ever contain `secret`.
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def failing_runner(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4-vm | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            return _cp(cmd, 1, stdout=f"fatal error with {secret}", stderr=f"HF_TOKEN={secret}")
        return _cp(cmd, 0)

    sup = ColabSupervisor(cfg, hf_token=secret, runner=failing_runner, sleep_fn=lambda _: None)
    with pytest.raises(ColabCommandError) as exc_info:
        sup.ensure_session_and_launch()

    err = exc_info.value
    assert secret not in str(err)
    assert secret not in repr(err)
    assert secret not in " ".join(err.cmd)
    assert secret not in err.stdout
    assert secret not in err.stderr

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    log_contents = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    assert secret not in log_contents


def test_timeout_and_hub_errors_never_leak_token(tmp_path, monkeypatch):
    secret = "hf_TimeoutSecretTokenValue1122334455"
    cfg = _make_cfg(tmp_path)

    def timeout_runner(cmd, **kwargs):
        if cmd[1] == "sessions":
            return _cp(cmd, 0, stdout="[loop64] ep | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if cmd[1] == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=10, output=f"partial {secret}")

    sup = ColabSupervisor(cfg, hf_token=secret, runner=timeout_runner, sleep_fn=lambda _: None)
    with pytest.raises(ColabCommandError) as exc_info:
        sup.ensure_session_and_launch()
    assert secret not in str(exc_info.value)
    assert secret not in " ".join(exc_info.value.cmd)
    assert secret not in exc_info.value.stdout

    def fake_urlopen(req, timeout=20.0):
        assert req.get_header("Authorization") == f"Bearer {secret}"
        raise urllib.error.HTTPError(
            req.full_url, 500, f"Internal Error token={secret}", hdrs=None, fp=io.BytesIO(b"")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ColabSupervisorError) as hub_exc:
        fetch_hub_manifest_dict(cfg.repo, hf_token=secret)
    assert secret not in str(hub_exc.value)


# --------------------------------------------------------------------------- (b) Keepalive probe & stdin safety


def test_keepalive_probe_parsing_stdin_hygiene_and_once_mode(tmp_path):
    secret = "hf_KeepaliveSecretTokenabcdef123456"
    cfg_once = _make_cfg(tmp_path, once=True)
    recorded: list[tuple[list[str], dict[str, Any]]] = []
    sleeps: list[float] = []

    def fake_runner(cmd, **kwargs):
        recorded.append((list(cmd), dict(kwargs)))
        sub = cmd[1]
        if sub == "sessions":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            assert cmd[2:] == ["-s", "loop64", "--gpu", "G4"]
            return _cp(cmd, 0, stdout="[colab] Creating session 'loop64'...\n[colab] Session READY.\n")
        if sub == "url":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            return _cp(
                cmd,
                0,
                stdout="https://colab.research.google.com/notebooks/empty.ipynb?dbu=%2Ftun%2Fm%2Fgpu-g4\n",
            )
        if sub == "exec":
            # Every `colab exec` MUST pass piped Python code via `input=` (never inheriting TTY stdin).
            assert isinstance(kwargs.get("input"), str) and kwargs["input"].strip()
            assert "stdin" not in kwargs
            if "--env" in cmd:
                # Bootstrap call only
                return _cp(
                    cmd,
                    0,
                    stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 5904, "head": "350ca77"}}\n',
                )
            # Keepalive probe call: must NOT pass --env HF_TOKEN and must set --timeout 180
            assert "--env" not in cmd
            assert "--timeout" in cmd and cmd[cmd.index("--timeout") + 1] == "180"
            return _cp(
                cmd,
                0,
                stdout=_probe_stdout(
                    pid_alive=True,
                    exit_code=None,
                    last_log_lines=["step 15/1000 loss/lm=1.0521 tok_per_s=1931.4"],
                    with_sentinel=False,  # verify bare JSON probe parsing works too
                ),
            )
        raise AssertionError(f"unexpected subcommand: {cmd}")

    sup = ColabSupervisor(cfg_once, hf_token=secret, runner=fake_runner, sleep_fn=sleeps.append)
    rc = sup.run()
    assert rc == 0
    assert sleeps == []
    # Live PID in --once mode must NOT call `colab stop`
    assert not any(c[0][1] == "stop" for c in recorded)

    log_text = pathlib.Path(cfg_once.log_file).read_text(encoding="utf-8")
    assert "[KEEPALIVE] session=loop64 pid_alive=True exit_code=None" in log_text
    assert "step 15/1000 loss/lm=1.0521" in log_text
    assert secret not in log_text


# --------------------------------------------------------------------------- (c) Clean completion (exit_code == 0)


def test_exit_code_zero_stops_session_and_exits_zero(tmp_path):
    cfg = _make_cfg(tmp_path, interval_sec=120.0)
    sleeps: list[float] = []
    stop_calls: list[list[str]] = []
    probe_idx = 0

    def fake_runner(cmd, **kwargs):
        nonlocal probe_idx
        sub = cmd[1]
        if sub == "sessions":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            return _cp(cmd, 0, stdout="[loop64] gpu-g4-1 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "url":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "stop":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            stop_calls.append(list(cmd))
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 5904}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "[pipeline] ALL STAGES COMPLETED\\n"}}\n')
            probe_idx += 1
            if probe_idx == 1:
                return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=True, exit_code=None))
            return _cp(
                cmd,
                0,
                stdout=_probe_stdout(
                    pid_alive=False,
                    exit_code=0,
                    last_log_lines=["[pipeline] ALL STAGES COMPLETED"],
                ),
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    def fake_manifest(_repo_id: str) -> dict[str, Any]:
        return {
            "stages": {
                "s0_probe": {"status": "completed", "step": 0},
                "s1_loop_a6b": {"status": "completed", "step": 1000},
                "s1b_hssd_distill": {"status": "completed", "step": 1500},
            }
        }

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_CleanToken123456",
        runner=fake_runner,
        manifest_fetcher=fake_manifest,
        sleep_fn=sleeps.append,
    )
    assert sup.run() == 0
    assert sleeps == [120.0]
    assert stop_calls == [["/fake/bin/colab", "stop", "-s", "loop64"]]
    assert "ALL STAGES COMPLETED" in pathlib.Path(cfg.tail_file).read_text(encoding="utf-8")
    assert "[DONE] pipeline exited 0; stopped session" in pathlib.Path(cfg.log_file).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- (d) Crash auto-stop (exit_code == 1)


def test_nonzero_exit_code_stops_session_captures_tail_and_exits_nonzero(tmp_path):
    # Even if keep_on_complete=True, a crashed run MUST stop the VM immediately to prevent idle GPU burn.
    secret = "hf_CrashSecretToken999888777"
    cfg = _make_cfg(tmp_path, keep_on_complete=True)
    stop_calls: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4-1 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "stop":
            assert kwargs.get("stdin") == subprocess.DEVNULL
            stop_calls.append(list(cmd))
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 7777}}\n')
            if TAIL_SENTINEL in code:
                tb = f"RuntimeError: CUDA error (HF_TOKEN={secret})\n"
                return _cp(cmd, 0, stdout=f"{TAIL_SENTINEL}{json.dumps({'tail': tb})}\n")
            return _cp(
                cmd,
                0,
                stdout=_probe_stdout(
                    pid_alive=False,
                    exit_code=1,
                    last_log_lines=["RuntimeError: CUDA error"],
                ),
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(cfg, hf_token=secret, runner=fake_runner, sleep_fn=lambda _: None)
    rc = sup.run()
    assert rc == 1
    assert stop_calls == [["/fake/bin/colab", "stop", "-s", "loop64"]]

    tail_text = pathlib.Path(cfg.tail_file).read_text(encoding="utf-8")
    assert "RuntimeError: CUDA error" in tail_text
    assert secret not in tail_text
    assert "<REDACTED>" in tail_text

    log_text = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    assert "[ERROR] pipeline exited 1; stopped session" in log_text
    assert secret not in log_text


# --------------------------------------------------------------------------- (e) Preemption auto-respawn


def test_preemption_auto_respawns_while_stages_incomplete_then_completes(tmp_path):
    cfg = _make_cfg(
        tmp_path,
        stages=("s0_probe", "s1_loop_a6b"),
        max_respawns=2,
        respawn_backoff_sec=15.0,
        interval_sec=60.0,
    )
    sleeps: list[float] = []
    new_calls = 0
    probe_calls = 0

    def fake_manifest(_repo_id: str) -> dict[str, Any]:
        # s0_probe done; s1_loop_a6b is incomplete during the 2 preemptions, then completed on final exit_code=0.
        if probe_calls >= 4:
            return {
                "stages": {
                    "s0_probe": {"status": "completed", "step": 0},
                    "s1_loop_a6b": {"status": "completed", "step": 1000},
                }
            }
        return {
            "stages": {
                "s0_probe": {"status": "completed", "step": 0},
                "s1_loop_a6b": {"status": "running", "step": 75},
            }
        }

    session_alive_on_server = False

    def fake_runner(cmd, **kwargs):
        nonlocal new_calls, probe_calls, session_alive_on_server
        sub = cmd[1]
        if sub == "sessions":
            if session_alive_on_server:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4-vm | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            new_calls += 1
            session_alive_on_server = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "stop":
            session_alive_on_server = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8000}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "done\\n"}}\n')
            probe_calls += 1
            if probe_calls in (1, 2):
                # Simulate Colab 90-min reaper / preemption twice!
                session_alive_on_server = False
                msg = (
                    "[colab] Session 'loop64' not found.\n"
                    if probe_calls == 1
                    else "[colab] Session 'loop64' appears to be lost (404/401). Cleaning up.\n"
                )
                return _cp(cmd, 1, stdout=msg)
            if probe_calls == 3:
                return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=True, exit_code=None))
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_RespawnToken123456",
        runner=fake_runner,
        manifest_fetcher=fake_manifest,
        sleep_fn=sleeps.append,
    )
    assert sup.run() == 0
    # 1 initial `colab new` + 2 respawns = 3 total provisions
    assert new_calls == 3
    assert sleeps == [15.0, 15.0, 60.0]


def test_preemption_stops_when_max_respawns_exceeded_or_stages_already_completed(tmp_path):
    # Case 1: max_respawns=1 -> 1 respawn allowed, 2nd preemption exits 1.
    cfg_cap = _make_cfg(tmp_path, stages=("s0_probe", "s1_loop_a6b"), max_respawns=1, respawn_backoff_sec=15.0)
    new_count = 0

    def always_preempted_runner(cmd, **kwargs):
        nonlocal new_count
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            new_count += 1
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            if BOOTSTRAP_SENTINEL in kwargs["input"]:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 9001}}\n')
            return _cp(cmd, 1, stdout="[colab] Session 'loop64' not found.\n")
        return _cp(cmd, 0)

    sup_cap = ColabSupervisor(
        cfg_cap,
        hf_token="hf_CapToken123456",
        runner=always_preempted_runner,
        manifest_fetcher=lambda _: {"stages": {"s0_probe": {"status": "completed"}}},
        sleep_fn=lambda _: None,
    )
    assert sup_cap.run() == 1
    assert new_count == 2  # initial + 1 respawn

    # Case 2: All requested stages ALREADY completed on Hub when session disappears -> exit 0, 0 respawns!
    cfg_done = _make_cfg(
        tmp_path,
        stages=("s0_probe", "s1_loop_a6b"),
        max_respawns=3,
        log_file=str(tmp_path / "runs" / "done.log"),
    )
    done_new_count = 0

    def done_on_hub_runner(cmd, **kwargs):
        nonlocal done_new_count
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] ep | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "new":
            done_new_count += 1
            return _cp(cmd, 0)
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            if BOOTSTRAP_SENTINEL in kwargs["input"]:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 9002}}\n')
            return _cp(cmd, 1, stdout="[colab] Session 'loop64' not found.\n")
        return _cp(cmd, 0)

    sup_done = ColabSupervisor(
        cfg_done,
        hf_token="hf_DoneToken123456",
        runner=done_on_hub_runner,
        manifest_fetcher=lambda _: {
            "stages": {
                "s0_probe": {"status": "completed"},
                "s1_loop_a6b": {"status": "completed"},
            }
        },
        sleep_fn=lambda _: None,
    )
    assert sup_done.run() == 0
    assert done_new_count == 0


# --------------------------------------------------------------------------- Remote script AST & functional smoke


def test_generated_remote_scripts_parse_py39_and_execute_cleanly(tmp_path):
    cfg = _make_cfg(tmp_path, remote_run_root=str(tmp_path / "remote_runs"))
    for src in (
        build_bootstrap_python_script(cfg),
        build_probe_python_script(cfg.remote_run_root),
        build_tail_python_script(cfg.remote_run_root),
    ):
        ast.parse(src, feature_version=(3, 9))

    # Execute probe & tail scripts locally against a simulated finished run_root!
    run_root = tmp_path / "remote_runs"
    run_root.mkdir(parents=True)
    (run_root / "pipeline.pid").write_text("999999\n", encoding="utf-8")
    (run_root / "pipeline.exit_code").write_text("0\n", encoding="utf-8")
    (run_root / "pipeline_stdout.log").write_text("line 1\nstep 25/1000 pushed to Hub\n", encoding="utf-8")

    probe_proc = subprocess.run(
        ["/usr/bin/python3", "-c", build_probe_python_script(str(run_root))],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe_proc.stdout.startswith(PROBE_SENTINEL)
    parsed = ProbeResult.from_dict(json.loads(probe_proc.stdout[len(PROBE_SENTINEL) :]))
    assert parsed.pid_alive is False
    assert parsed.exit_code == 0
    assert parsed.last_log_lines[-1] == "step 25/1000 pushed to Hub"
    assert parsed.disk_free_gib > 0.0

    # Verify helpers (`normalize_git_https_url`, `parse_active_sessions`, `is_session_missing_output`, `main`)
    assert (
        normalize_git_https_url("git@github.com:BaiTian6641/Qwen3.6-35B-A3B-Loop60.git")
        == "https://github.com/BaiTian6641/Qwen3.6-35B-A3B-Loop60.git"
    )
    sessions_out = (
        "[colab] Pruned 1 stale local session(s).\n"
        "[loop64] gpu-g4-s-abc | Hardware: G4 | Shape: Standard | Variant: GPU\n"
        "[?] gpu-t4-s-xyz | Hardware: T4 | Shape: Standard | Variant: GPU\n"
    )
    assert parse_active_sessions(sessions_out) == {"loop64"}
    assert is_session_missing_output("[colab] Session 'loop64' not found.", "loop64")
    assert not is_session_missing_output("[colab] Session 'other' not found.", "loop64")

    # Verify `scripts/colab_supervisor.py` has valid mode/syntax and `--help` exits 0 under stdlib python3.
    script_path = os.path.join(ROOT, "scripts", "colab_supervisor.py")
    help_proc = subprocess.run(
        ["/usr/bin/python3", script_path, "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--interval-sec" in help_proc.stdout
    assert "--max-respawns" in help_proc.stdout
    assert "--max-rollovers" in help_proc.stdout
    parsed_default = config_from_args(build_arg_parser().parse_args([]))
    assert parsed_default.max_rollovers == DEFAULT_MAX_ROLLOVERS == 10
    parsed_zero = config_from_args(build_arg_parser().parse_args(["--max-rollovers", "0"]))
    assert parsed_zero.max_rollovers == 0
    assert stat.S_IMODE(os.stat(script_path).st_mode) & 0o111 or os.path.isfile(script_path)
    with pytest.raises(SystemExit) as sysexit:
        main(["--help"])
    assert sysexit.value.code == 0


# --------------------------------------------------------------------------- (f) Stage YAML `ckpt_every == 25`


def test_all_v3_stage_configs_resolve_ckpt_every_25_and_hub_sync():
    base_cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))
    assert base_cfg.train["ckpt_every"] == 50

    expected_stages = (
        "s1_loop_a6b",
        "s1b_hssd_distill",
        "s3_long_1m_lsa",
        "s2_joint_bptt",
        "s4_rl",
    )
    for stage_name in expected_stages:
        cfg_path = os.path.join(ROOT, "configs", "stages", f"{stage_name}.yaml")
        cfg = load_config(cfg_path)
        assert cfg.train["ckpt_every"] == 25, f"{stage_name}: ckpt_every={cfg.train.get('ckpt_every')}"
        sync_cfg = cfg.checkpoint["sync"]
        assert sync_cfg["enabled"] is True, stage_name
        assert sync_cfg["push_every_ckpt"] is True, stage_name
        assert sync_cfg["push_optim"] is False, stage_name


# --------------------------------------------------------------------------- (g) Clean time-budget VM rollover


def test_time_budget_rollover_stops_old_vm_before_new_then_completes(tmp_path):
    """exit_code == 0 with incomplete Hub stages must stop the expiring VM BEFORE `colab new`."""
    cfg = _make_cfg(
        tmp_path,
        stages=("s0_probe", "s3_long_1m_lsa", "s2_joint_bptt"),
        interval_sec=60.0,
        respawn_backoff_sec=15.0,
        max_rollovers=5,
    )
    events: list[str] = []
    sleeps: list[float] = []
    vm_generation = 0
    session_alive = True  # Starts attached to an existing expiring VM (gen 0)
    probe_in_vm = 0

    def fake_manifest(_repo_id: str) -> dict[str, Any]:
        if vm_generation == 0:
            return {
                "stages": {
                    "s0_probe": {"status": "completed", "step": 0},
                    "s3_long_1m_lsa": {"status": "in_progress", "step": 425},
                }
            }
        return {
            "stages": {
                "s0_probe": {"status": "completed", "step": 0},
                "s3_long_1m_lsa": {"status": "completed", "step": 600},
                "s2_joint_bptt": {"status": "completed", "step": 300},
            }
        }

    def fake_runner(cmd, **kwargs):
        nonlocal vm_generation, session_alive, probe_in_vm
        sub = cmd[1]
        if sub == "sessions":
            events.append(f"sessions(alive={session_alive})")
            if session_alive:
                return _cp(
                    cmd,
                    0,
                    stdout=f"[loop64] gpu-g4-vm{vm_generation} | Hardware: G4 | Shape: Standard | Variant: GPU\n",
                )
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "stop":
            events.append(f"stop(gen={vm_generation})")
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "new":
            assert not session_alive, "colab new must NEVER be called while old VM is still active"
            vm_generation += 1
            session_alive = True
            probe_in_vm = 0
            events.append(f"new(gen={vm_generation})")
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout=f"https://colab.research.google.com/notebooks/vm{vm_generation}.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                events.append(f"bootstrap(gen={vm_generation})")
                return _cp(
                    cmd,
                    0,
                    stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": {5000 + vm_generation}}}\n',
                )
            if TAIL_SENTINEL in code:
                msg = (
                    "[pipeline] s3_long_1m_lsa paused cleanly before completion (step=425)\n"
                    if vm_generation == 0
                    else "[pipeline] requested pipeline stages finished\n"
                )
                return _cp(cmd, 0, stdout=f"{TAIL_SENTINEL}{json.dumps({'tail': msg})}\n")
            probe_in_vm += 1
            events.append(f"probe(gen={vm_generation},n={probe_in_vm})")
            if vm_generation == 0:
                return _cp(
                    cmd,
                    0,
                    stdout=_probe_stdout(
                        pid_alive=False,
                        exit_code=0,
                        last_log_lines=["[pipeline] s3_long_1m_lsa paused cleanly before completion (step=425)"],
                    ),
                )
            if probe_in_vm == 1:
                return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=True, exit_code=None))
            return _cp(
                cmd,
                0,
                stdout=_probe_stdout(
                    pid_alive=False,
                    exit_code=0,
                    last_log_lines=["[pipeline] requested pipeline stages finished"],
                ),
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_RolloverToken123456",
        runner=fake_runner,
        manifest_fetcher=fake_manifest,
        sleep_fn=sleeps.append,
    )
    assert sup.run() == 0
    assert vm_generation == 1
    # Verify exact ordering: old VM stopped and verified gone BEFORE fresh `colab new`, then final VM stopped.
    stop0_idx = events.index("stop(gen=0)")
    gone_idx = events.index("sessions(alive=False)", stop0_idx)
    new1_idx = events.index("new(gen=1)")
    boot1_idx = events.index("bootstrap(gen=1)")
    stop1_idx = events.index("stop(gen=1)")
    assert stop0_idx < gone_idx < new1_idx < boot1_idx < stop1_idx
    assert sleeps == [15.0, 60.0]

    log_text = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    assert "[ROLLOVER]" in log_text
    assert "rollover 1/5" in log_text
    assert "s3_long_1m_lsa=in_progress@425" in log_text
    assert "[DONE] pipeline exited 0; stopped session" in log_text


def test_multi_rollover_progression_when_hub_step_advances(tmp_path):
    """Multiple consecutive clean time-budget pauses advance across rollovers until all stages complete."""
    cfg = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa", "s2_joint_bptt"),
        interval_sec=30.0,
        respawn_backoff_sec=10.0,
        max_rollovers=5,
        keep_on_complete=True,  # Must still stop expiring VMs during rollovers, and keep ONLY the final VM!
    )
    manifests = [
        {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 250}}},
        {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 500}}},
        {
            "stages": {
                "s3_long_1m_lsa": {"status": "completed", "step": 600},
                "s2_joint_bptt": {"status": "in_progress", "step": 150},
            }
        },
        {
            "stages": {
                "s3_long_1m_lsa": {"status": "completed", "step": 600},
                "s2_joint_bptt": {"status": "completed", "step": 300},
            }
        },
    ]
    manifest_idx = 0
    session_alive = False
    new_count = 0
    stop_count = 0

    def fake_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal manifest_idx
        res = manifests[min(manifest_idx, len(manifests) - 1)]
        manifest_idx += 1
        return res

    def fake_runner(cmd, **kwargs):
        nonlocal session_alive, new_count, stop_count
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            assert not session_alive
            new_count += 1
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "stop":
            stop_count += 1
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 7000}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_MultiRolloverToken123",
        runner=fake_runner,
        manifest_fetcher=fake_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup.run() == 0
    # 1 initial VM + 3 rollovers = 4 VMs provisioned; 3 expiring VMs stopped; final VM kept (--keep-on-complete).
    assert new_count == 4
    assert stop_count == 3
    log_text = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    assert "rollover 1/5" in log_text
    assert "rollover 2/5" in log_text
    assert "rollover 3/5" in log_text
    assert "[DONE] pipeline exited 0; keeping session (--keep-on-complete)" in log_text


def test_rollover_no_progress_guard_stops_vm_and_exits_one(tmp_path):
    """2 consecutive exit_code == 0 exits with identical Hub (status, step) must stop VM and return 1."""
    cfg = _make_cfg(
        tmp_path,
        stages=("s0_probe", "s3_long_1m_lsa"),
        max_rollovers=10,
        keep_on_complete=True,  # Error guard MUST still stop VM to avoid idle burn!
    )
    session_alive = True
    new_count = 0
    stop_count = 0

    def stuck_manifest(_repo_id: str) -> dict[str, Any]:
        return {
            "stages": {
                "s0_probe": {"status": "completed", "step": 0},
                "s3_long_1m_lsa": {"status": "in_progress", "step": 230},
            }
        }

    def fake_runner(cmd, **kwargs):
        nonlocal session_alive, new_count, stop_count
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            new_count += 1
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "stop":
            stop_count += 1
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8100}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_NoProgressToken123456",
        runner=fake_runner,
        manifest_fetcher=stuck_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup.run() == 1
    # 1st exit_code=0 rolls over (1 `new`, 1 `stop`); 2nd identical Hub progress stops VM and aborts (0 extra `new`).
    assert new_count == 1
    assert stop_count == 2
    assert not session_alive
    log_text = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    assert "[ERROR]" in log_text
    assert "zero Hub progress since prior rollover" in log_text


def test_rollover_cap_and_zero_disables_rollover(tmp_path):
    # Case 1: max_rollovers=2 -> allows 2 rollovers as Hub advances, stops VM and exits 1 on 3rd incomplete exit 0.
    cfg_cap = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=2,
        log_file=str(tmp_path / "runs" / "cap.log"),
    )
    step_counter = 100
    session_alive = True
    new_count = 0
    stop_count = 0

    def advancing_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal step_counter
        step_counter += 100
        return {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": step_counter}}}

    def cap_runner(cmd, **kwargs):
        nonlocal session_alive, new_count, stop_count
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            new_count += 1
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "stop":
            stop_count += 1
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8200}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_cap = ColabSupervisor(
        cfg_cap,
        hf_token="hf_RolloverCapToken123",
        runner=cap_runner,
        manifest_fetcher=advancing_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup_cap.run() == 1
    assert new_count == 2
    assert stop_count == 3  # 2 rollover stops + 1 final stop on cap reached
    assert not session_alive
    cap_log = pathlib.Path(cfg_cap.log_file).read_text(encoding="utf-8")
    assert "max_rollovers (2) reached" in cap_log

    # Case 2: max_rollovers=0 disables rollover completely -> stops VM and exits 0 without querying Hub.
    cfg_zero = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=0,
        log_file=str(tmp_path / "runs" / "zero.log"),
    )
    manifest_calls = 0
    zero_new = 0
    zero_stop = 0

    def forbid_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal manifest_calls
        manifest_calls += 1
        return {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 230}}}

    def zero_runner(cmd, **kwargs):
        nonlocal zero_new, zero_stop
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "new":
            zero_new += 1
            return _cp(cmd, 0)
        if sub == "stop":
            zero_stop += 1
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8300}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_zero = ColabSupervisor(
        cfg_zero,
        hf_token="hf_ZeroRolloverToken123",
        runner=zero_runner,
        manifest_fetcher=forbid_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup_zero.run() == 0
    assert manifest_calls == 0
    assert zero_new == 0
    assert zero_stop == 1


def test_rollover_hub_unreachable_fallback_on_exit_zero(tmp_path):
    # Case 1: Hub unreachable on all 3 retries, tail HAS "paused cleanly before completion" -> rolls over,
    #         then Hub recovers on VM #2 showing completion -> stops VM and returns 0.
    cfg_pause = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=3,
        log_file=str(tmp_path / "runs" / "hub_pause.log"),
    )
    vm_gen = 0
    session_alive = True
    manifest_attempts_gen0 = 0

    def flaky_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal manifest_attempts_gen0
        if vm_gen == 0:
            manifest_attempts_gen0 += 1
            raise ColabSupervisorError("HF Hub manifest HTTP 503: Service Unavailable")
        return {"stages": {"s3_long_1m_lsa": {"status": "completed", "step": 600}}}

    def pause_runner(cmd, **kwargs):
        nonlocal vm_gen, session_alive
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "stop":
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "new":
            vm_gen += 1
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8400}}\n')
            if TAIL_SENTINEL in code:
                tail = (
                    "[pipeline] s3_long_1m_lsa paused cleanly before completion (step=400)\n"
                    if vm_gen == 0
                    else "[pipeline] requested pipeline stages finished\n"
                )
                return _cp(cmd, 0, stdout=f"{TAIL_SENTINEL}{json.dumps({'tail': tail})}\n")
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0, last_log_lines=["ok"]))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_pause = ColabSupervisor(
        cfg_pause,
        hf_token="hf_HubFallbackToken123",
        runner=pause_runner,
        manifest_fetcher=flaky_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup_pause.run() == 0
    assert manifest_attempts_gen0 == 3
    assert vm_gen == 1
    assert not session_alive

    # Case 2: Hub unreachable on all 3 retries, tail has NO clean-pause marker -> treat as complete,
    #         stop VM, return 0, ZERO extra `colab new` (never burns VMs on Hub outage at real completion!).
    cfg_done = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=3,
        log_file=str(tmp_path / "runs" / "hub_done.log"),
    )
    done_new = 0
    done_stop = 0
    done_manifest_attempts = 0

    def always_down_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal done_manifest_attempts
        done_manifest_attempts += 1
        raise ColabSupervisorError("HF Hub manifest fetch failed: connection timed out")

    def done_runner(cmd, **kwargs):
        nonlocal done_new, done_stop
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "new":
            done_new += 1
            return _cp(cmd, 0)
        if sub == "stop":
            done_stop += 1
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8500}}\n')
            if TAIL_SENTINEL in code:
                return _cp(
                    cmd,
                    0,
                    stdout=f'{TAIL_SENTINEL}{{"tail": "[pipeline] requested pipeline stages finished\\n"}}\n',
                )
            return _cp(
                cmd,
                0,
                stdout=_probe_stdout(
                    pid_alive=False,
                    exit_code=0,
                    last_log_lines=["[pipeline] requested pipeline stages finished"],
                ),
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_done = ColabSupervisor(
        cfg_done,
        hf_token="hf_HubDoneToken123456",
        runner=done_runner,
        manifest_fetcher=always_down_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup_done.run() == 0
    assert done_manifest_attempts == 3
    assert done_new == 0
    assert done_stop == 1

    # Case 3: Hub fails twice then succeeds on 3rd retry -> uses real Hub response without fallback!
    cfg_retry = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=3,
        log_file=str(tmp_path / "runs" / "hub_retry.log"),
    )
    retry_attempts = 0
    retry_new = 0

    def recover_on_third(_repo_id: str) -> dict[str, Any]:
        nonlocal retry_attempts
        retry_attempts += 1
        if retry_attempts < 3:
            raise ColabSupervisorError("HF Hub HTTP 502 Bad Gateway")
        return {"stages": {"s3_long_1m_lsa": {"status": "completed", "step": 600}}}

    def retry_runner(cmd, **kwargs):
        nonlocal retry_new
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "new":
            retry_new += 1
            return _cp(cmd, 0)
        if sub == "stop":
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8550}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_retry = ColabSupervisor(
        cfg_retry,
        hf_token="hf_HubRetryToken123456",
        runner=retry_runner,
        manifest_fetcher=recover_on_third,
        sleep_fn=lambda _: None,
    )
    assert sup_retry.run() == 0
    assert retry_attempts == 3
    assert retry_new == 0


def test_once_mode_never_triggers_rollover_on_exit_zero(tmp_path):
    """--once mode must preserve existing exit_code == 0 stop behavior and never trigger rollover."""
    cfg = _make_cfg(tmp_path, once=True, max_rollovers=10, stages=("s3_long_1m_lsa",))
    manifest_calls = 0
    new_calls = 0
    stop_calls = 0

    def forbid_manifest(_repo_id: str) -> dict[str, Any]:
        nonlocal manifest_calls
        manifest_calls += 1
        return {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 230}}}

    def fake_runner(cmd, **kwargs):
        nonlocal new_calls, stop_calls
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "new":
            new_calls += 1
            return _cp(cmd, 0)
        if sub == "stop":
            stop_calls += 1
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8600}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_OnceModeToken123456",
        runner=fake_runner,
        manifest_fetcher=forbid_manifest,
        sleep_fn=lambda _: None,
    )
    assert sup.run() == 0
    assert manifest_calls == 0
    assert new_calls == 0
    assert stop_calls == 1


def test_rollover_stop_session_retries_when_old_session_lingers_and_aborts_if_stuck(tmp_path):
    # Case 1: Old session lingers in `colab sessions` for 2 checks after initial stop, disappears on 2nd retry -> OK!
    cfg_ok = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=3,
        respawn_backoff_sec=15.0,
        log_file=str(tmp_path / "runs" / "linger_ok.log"),
    )
    vm_gen = 0
    linger_checks_left = 0
    stop_calls_gen0 = 0
    new_calls = 0
    sleeps: list[float] = []

    def ok_manifest(_repo_id: str) -> dict[str, Any]:
        if vm_gen == 0:
            return {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 230}}}
        return {"stages": {"s3_long_1m_lsa": {"status": "completed", "step": 600}}}

    def linger_ok_runner(cmd, **kwargs):
        nonlocal vm_gen, linger_checks_left, stop_calls_gen0, new_calls
        sub = cmd[1]
        if sub == "sessions":
            if vm_gen == 0 and stop_calls_gen0 == 0:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4-old | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            if linger_checks_left > 0:
                linger_checks_left -= 1
                return _cp(cmd, 0, stdout="[loop64] gpu-g4-old | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "stop":
            if vm_gen == 0:
                stop_calls_gen0 += 1
                if stop_calls_gen0 == 1:
                    linger_checks_left = 2  # Still listed for check 1 & check 2; gone after 2nd retry stop!
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "new":
            assert stop_calls_gen0 == 3, "must retry stop twice before provisioning new VM"
            new_calls += 1
            vm_gen += 1
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8700}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "ok\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_ok = ColabSupervisor(
        cfg_ok,
        hf_token="hf_LingerOkToken123456",
        runner=linger_ok_runner,
        manifest_fetcher=ok_manifest,
        sleep_fn=sleeps.append,
    )
    assert sup_ok.run() == 0
    assert stop_calls_gen0 == 3
    assert new_calls == 1
    assert sleeps == [15.0, 5.0, 5.0]

    # Case 2: Old session NEVER disappears from `colab sessions` -> abort cleanly with rc=1, NEVER re-attach!
    cfg_stuck = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=3,
        respawn_backoff_sec=15.0,
        log_file=str(tmp_path / "runs" / "linger_stuck.log"),
    )
    stuck_stops = 0
    stuck_news = 0
    stuck_boots = 0

    def stuck_runner(cmd, **kwargs):
        nonlocal stuck_stops, stuck_news, stuck_boots
        sub = cmd[1]
        if sub == "sessions":
            return _cp(cmd, 0, stdout="[loop64] gpu-g4-old | Hardware: G4 | Shape: Standard | Variant: GPU\n")
        if sub == "stop":
            stuck_stops += 1
            return _cp(cmd, 0, stdout="[colab] Stop accepted.\n")
        if sub == "new":
            stuck_news += 1
            return _cp(cmd, 0)
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                stuck_boots += 1
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8800}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_stuck = ColabSupervisor(
        cfg_stuck,
        hf_token="hf_LingerStuckToken123",
        runner=stuck_runner,
        manifest_fetcher=lambda _: {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 230}}},
        sleep_fn=lambda _: None,
    )
    assert sup_stuck.run() == 1
    assert stuck_stops == 3  # initial stop + 2 retries
    assert stuck_news == 0
    assert stuck_boots == 1  # only the initial attach; never re-attached to expiring VM!
    stuck_log = pathlib.Path(cfg_stuck.log_file).read_text(encoding="utf-8")
    assert "refusing to re-attach to expiring VM" in stuck_log


def test_rollover_transient_launch_retry_and_zero_hf_token_leakage(tmp_path, capsys):
    """Transient provision/bootstrap failures during rollover retry cleanly with ZERO HF_TOKEN leakage."""
    secret = "hf_RolloverSuperSecretTokenXYZ987654"
    cfg = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=5,
        max_respawns=2,
        respawn_backoff_sec=10.0,
    )
    session_alive = True
    vm_gen = 0
    rollover_new_attempts = 0
    rollover_boot_failures = 0
    stop_calls = 0
    manifest_calls = 0

    def manifest_with_transient_secret_err(_repo_id: str) -> dict[str, Any]:
        nonlocal manifest_calls
        manifest_calls += 1
        if manifest_calls == 1:
            # Even if an upstream error message accidentally contains a secret, rollover logs must redact it!
            raise ColabSupervisorError(f"transient Hub error with Authorization: Bearer {secret}")
        if vm_gen == 0:
            return {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 350}}}
        return {"stages": {"s3_long_1m_lsa": {"status": "completed", "step": 600}}}

    def fake_runner(cmd, **kwargs):
        nonlocal session_alive, vm_gen, rollover_new_attempts, rollover_boot_failures, stop_calls
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "stop":
            stop_calls += 1
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "new":
            rollover_new_attempts += 1
            if rollover_new_attempts == 1:
                # 1st `colab new` fails transiently with token-like text in stderr
                return _cp(cmd, 1, stderr=f"temporary GPU quota error HF_TOKEN={secret}")
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                if rollover_new_attempts == 2 and rollover_boot_failures == 0:
                    # 2nd attempt provisions VM, then bootstrap fails -> ensure_session_and_launch MUST stop idle VM!
                    rollover_boot_failures += 1
                    return _cp(cmd, 1, stdout=f"bootstrap error {secret}", stderr=f"Bearer {secret}")
                if rollover_new_attempts >= 3:
                    vm_gen = 1
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 8900}}\n')
            if TAIL_SENTINEL in code:
                tail = f"paused cleanly before completion HF_TOKEN={secret}\n"
                return _cp(cmd, 0, stdout=f"{TAIL_SENTINEL}{json.dumps({'tail': tail})}\n")
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token=secret,
        runner=fake_runner,
        manifest_fetcher=manifest_with_transient_secret_err,
        sleep_fn=lambda _: None,
    )
    assert sup.run() == 0
    assert rollover_new_attempts == 3  # attempt 1 (new fail), attempt 2 (boot fail -> stop), attempt 3 (success)
    assert stop_calls == 3  # 1 expiring VM stop + 1 failed-bootstrap VM cleanup stop + 1 final completion stop
    assert not session_alive

    captured = capsys.readouterr()
    log_text = pathlib.Path(cfg.log_file).read_text(encoding="utf-8")
    tail_text = pathlib.Path(cfg.tail_file).read_text(encoding="utf-8")
    for text in (captured.out, captured.err, log_text, tail_text):
        assert secret not in text
    assert "<REDACTED>" in log_text
    assert "<REDACTED>" in tail_text

    # Exhausted rollover provision attempts (with `colab new` timeout rc=124 on final try) -> returns 1,
    # performs best-effort stop cleanup, and still never leaks `secret`.
    cfg_fail = _make_cfg(
        tmp_path,
        stages=("s3_long_1m_lsa",),
        max_rollovers=5,
        max_respawns=1,
        respawn_backoff_sec=5.0,
        log_file=str(tmp_path / "runs" / "rollover_fail.log"),
        tail_file=str(tmp_path / "runs" / "rollover_fail_tail.log"),
    )
    fail_session_alive = True
    fail_new_attempts = 0
    fail_stop_calls = 0

    def fail_launch_runner(cmd, **kwargs):
        nonlocal fail_session_alive, fail_new_attempts, fail_stop_calls
        sub = cmd[1]
        if sub == "sessions":
            if fail_session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "stop":
            fail_stop_calls += 1
            fail_session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "new":
            fail_new_attempts += 1
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=300, output=f"timeout {secret}")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": false, "pid": 8910}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "paused cleanly before completion\\n"}}\n')
            return _cp(cmd, 0, stdout=_probe_stdout(pid_alive=False, exit_code=0))
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_fail = ColabSupervisor(
        cfg_fail,
        hf_token=secret,
        runner=fail_launch_runner,
        manifest_fetcher=lambda _: {"stages": {"s3_long_1m_lsa": {"status": "in_progress", "step": 350}}},
        sleep_fn=lambda _: None,
    )
    assert sup_fail.run() == 1
    assert fail_new_attempts == 2  # 1 initial + 1 retry (`max_respawns=1`)
    assert fail_stop_calls == 2  # 1 expiring VM stop + 1 best-effort stop after final `colab new` timeout
    fail_captured = capsys.readouterr()
    fail_log = pathlib.Path(cfg_fail.log_file).read_text(encoding="utf-8")
    assert "failed to provision/bootstrap fresh session" in fail_log
    for text in (fail_captured.out, fail_captured.err, fail_log):
        assert secret not in text


def test_sync_pipeline_tracks_remaining_session_time_budget_across_stages(tmp_path, monkeypatch, caplog):
    """`qloop sync pipeline --time-budget-hours` must track cumulative session wall-clock time across stages."""
    import logging
    from types import SimpleNamespace

    from qloop.cli import _run_sync_action
    from qloop.hub import HubManifest

    # Case 1: Stage 1 completes after 18.0h -> Stage 2 receives remaining 3.5h (not a fresh 21.5h!).
    clock = [1000.0]
    monkeypatch.setattr("qloop.cli.time.monotonic", lambda: clock[0])
    stage_status: dict[str, dict[str, Any]] = {}
    recorded_cmds: list[list[str]] = []

    def fake_call_two_stages(cmd, *args, **kwargs):
        recorded_cmds.append(list(cmd))
        cfg_arg = cmd[cmd.index("-c") + 1]
        if cfg_arg.endswith("s3_long_1m_lsa.yaml"):
            clock[0] += 18.0 * 3600.0
            stage_status["s3_long_1m_lsa"] = {"status": "completed", "step": 600}
        elif cfg_arg.endswith("s2_joint_bptt.yaml"):
            clock[0] += 3.5 * 3600.0
            stage_status["s2_joint_bptt"] = {"status": "in_progress", "step": 125}
        return 0

    monkeypatch.setattr("qloop.cli.subprocess.call", fake_call_two_stages)

    class _FakeHub:
        def fetch_manifest(self) -> HubManifest:
            return HubManifest(stages=dict(stage_status))

    args_1 = SimpleNamespace(
        action="pipeline",
        stages="s3_long_1m_lsa,s2_joint_bptt",
        until=None,
        force=False,
        debug=1,
        set=[],
        time_budget_hours=21.5,
        scan_batches=2,
        scan_seq_len=64,
        config_root=os.path.join(ROOT, "configs"),
    )
    rc1 = _run_sync_action(
        args_1,
        hub=_FakeHub(),
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        run_root=str(tmp_path / "run1"),
        ckpt_cls=None,
    )
    assert rc1 == 0
    assert len(recorded_cmds) == 2
    assert "train.time_budget_hours=21.5" in recorded_cmds[0]
    assert "train.time_budget_hours=3.5" in recorded_cmds[1]

    # Case 2: Stage 1 completes after 22.0h (>= 21.5h budget) -> Stage 2 is NOT started; exits 0 cleanly for rollover.
    clock[0] = 5000.0
    stage_status.clear()
    recorded_cmds.clear()

    def fake_call_exhausted(cmd, *args, **kwargs):
        recorded_cmds.append(list(cmd))
        clock[0] += 22.0 * 3600.0
        stage_status["s3_long_1m_lsa"] = {"status": "completed", "step": 600}
        return 0

    monkeypatch.setattr("qloop.cli.subprocess.call", fake_call_exhausted)
    monkeypatch.setattr(logging.getLogger("qloop"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="qloop"):
        rc2 = _run_sync_action(
            args_1,
            hub=_FakeHub(),
            repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
            run_root=str(tmp_path / "run2"),
            ckpt_cls=None,
        )
    assert rc2 == 0
    assert len(recorded_cmds) == 1  # s2_joint_bptt did NOT launch on an already-expiring VM!
    assert any("paused cleanly before completion" in rec.getMessage() for rec in caplog.records)
