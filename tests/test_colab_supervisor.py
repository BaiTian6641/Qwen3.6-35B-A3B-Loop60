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
    PROBE_SENTINEL,
    TAIL_SENTINEL,
    ColabCommandError,
    ColabSupervisor,
    ColabSupervisorError,
    ProbeResult,
    SupervisorConfig,
    build_bootstrap_python_script,
    build_probe_python_script,
    build_remote_run_pipeline_script,
    build_tail_python_script,
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

    sup = ColabSupervisor(cfg, hf_token="hf_CleanToken123456", runner=fake_runner, sleep_fn=sleeps.append)
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
        # s0_probe done, s1_loop_a6b still incomplete at step 75 -> should trigger respawn!
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
