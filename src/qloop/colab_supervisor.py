"""Local macOS/Linux keepalive & auto-recovery supervisor for Google Colab CLI (`colab`).

Prevents two silent failure modes during long unattended Blackwell (`G4`) runs:
  1. **Colab 90-minute idle Jupyter kernel reaper + 60-minute proxy token expiry**:
     Launching `qloop sync pipeline` via `setsid nohup` leaves the Jupyter kernel `IDLE` with
     zero open WebSockets once `colab exec` exits. Executing a lightweight (~1s) Python probe
     inside the remote Jupyter kernel every `--interval-sec` (default 300s = 5 min) both keeps
     the kernel active and refreshes `colab-cli`'s `runtimeProxyInfo` token.
  2. **Zero-idle compute unit burn + automatic rollover & preemption recovery**:
     - Immediately runs `colab stop -s <session>` when the remote pipeline finishes all requested
       stages (`exit_code == 0`, unless `--keep-on-complete`) or crashes (`exit_code != 0`),
       capturing the remote log tail locally.
     - Automatically rolls over to a brand-new 24h VM (stopping the expiring VM FIRST and confirming
       termination before provisioning, up to `--max-rollovers`) when `qloop sync pipeline` exits `0`
       early due to `--time-budget-hours` while requested stages remain incomplete in `manifest.json`.
     - Automatically re-provisions, bootstraps, and resumes from the latest 25-step Hugging Face Hub
       checkpoint (up to `--max-respawns`) if Colab preempts the session while requested stages are
       still incomplete in `manifest.json`.

Stdlib-only and Python 3.9+ grammar compatible so `/usr/bin/python3` on macOS can run it directly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_COLAB_BIN = os.path.expanduser("~/.local/bin/colab")
DEFAULT_SESSION = "loop64"
DEFAULT_GPU = "G4"
DEFAULT_HF_REPO = "TassP0lar/Qwen3.6-35B-A3B-Loop64"
DEFAULT_GIT_URL = "https://github.com/BaiTian6641/Qwen3.6-35B-A3B-Loop60.git"
DEFAULT_GIT_REF = "origin/main"
DEFAULT_REMOTE_REPO_DIR = "/content/Qwen3.6-35B-A3B-Loop60"
DEFAULT_REMOTE_RUN_ROOT = "/content/runs"
DEFAULT_STAGES: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_a6b",
    "s1b_hssd_distill",
    "s3_long_1m_lsa",
    "s2_joint_bptt",
    "s4_rl",
)
DEFAULT_TIME_BUDGET_HOURS = 21.5
DEFAULT_DEBUG_LEVEL = 1
DEFAULT_INTERVAL_SEC = 300.0
DEFAULT_EXEC_TIMEOUT_SEC = 180
DEFAULT_BOOTSTRAP_TIMEOUT_SEC = 900
DEFAULT_MAX_RESPAWNS = 3
DEFAULT_MAX_ROLLOVERS = 10
DEFAULT_RESPAWN_BACKOFF_SEC = 15.0
DEFAULT_CAPACITY_BACKOFF_BASE_SEC = 60.0
DEFAULT_MAX_RESPAWN_BACKOFF_SEC = 300.0
DEFAULT_LOG_FILE = "runs/colab_supervisor.log"
DEFAULT_TAIL_FILE = "runs/colab_last_tail.log"

BOOTSTRAP_SENTINEL = "__QLOOP_BOOTSTRAP_JSON__:"
PROBE_SENTINEL = "__QLOOP_PROBE_JSON__:"
TAIL_SENTINEL = "__QLOOP_TAIL_JSON__:"
CLEAN_PAUSE_LOG_MARKER = "paused cleanly before completion"

_SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+\S+\s+\|\s+Hardware:")
_HF_TOKEN_INLINE_RE = re.compile(r"\bhf_[A-Za-z0-9]{6,}\b")
_ENV_ASSIGN_RE = re.compile(r"((?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*)[^\s'\"&;]+")
_AUTH_BEARER_RE = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SSH_GITHUB_RE = re.compile(r"^git@github\.com:([^/\s]+/[^\s]+?)(?:\.git)?/?$")
_SSH_PROTO_GITHUB_RE = re.compile(r"^ssh://git@github\.com/([^/\s]+/[^\s]+?)(?:\.git)?/?$")


class ColabSupervisorError(RuntimeError):
    """Base error for redacted supervisor failures."""


class ColabCommandError(ColabSupervisorError):
    """Raised when a `colab` CLI command fails (all stored strings are pre-redacted)."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int = 1,
        cmd: Sequence[str] = (),
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = int(returncode)
        self.cmd = tuple(cmd)
        self.stdout = stdout
        self.stderr = stderr


class ColabSessionLostError(ColabCommandError):
    """Raised when the remote Colab session was preempted, pruned, or not found."""


def redact_secret(text: str, secrets: Sequence[str] = ()) -> str:
    """Scrub explicit token values and common credential patterns from `text`."""
    if not text:
        return ""
    out = str(text)
    valid_secrets = sorted(
        {s.strip() for s in secrets if s and s.strip()},
        key=len,
        reverse=True,
    )
    for sec in valid_secrets:
        out = out.replace(sec, "<REDACTED>")
        url_enc = urllib.parse.quote(sec, safe="")
        if url_enc and url_enc != sec:
            out = out.replace(url_enc, "<REDACTED>")
    out = _ENV_ASSIGN_RE.sub(r"\1<REDACTED>", out)
    out = _AUTH_BEARER_RE.sub(r"\1<REDACTED>", out)
    out = _HF_TOKEN_INLINE_RE.sub("<REDACTED>", out)
    return out


def redact_argv(argv: Sequence[str], secrets: Sequence[str] = ()) -> list[str]:
    """Return a copy of `argv` safe for logging and exception messages."""
    cleaned: list[str] = []
    for arg in argv:
        if arg.startswith(("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
            key = arg.split("=", 1)[0]
            cleaned.append(f"{key}=<REDACTED>")
        else:
            cleaned.append(redact_secret(arg, secrets))
    return cleaned


def normalize_git_https_url(url: str) -> str:
    """Normalize GitHub SSH URLs (`git@github.com:owner/repo.git`) to HTTPS for keyless Colab clone."""
    raw = url.strip()
    for pat in (_SSH_GITHUB_RE, _SSH_PROTO_GITHUB_RE):
        m = pat.match(raw)
        if m:
            return f"https://github.com/{m.group(1)}.git"
    return raw


def resolve_local_hf_token(
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> str | None:
    """Resolve `HF_TOKEN` from environment or macOS `launchctl getenv HF_TOKEN`."""
    src_env = os.environ if env is None else env
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = src_env.get(key)
        if val and val.strip():
            return val.strip()
    run_fn = subprocess.run if runner is None else runner
    try:
        proc = run_fn(
            ["launchctl", "getenv", "HF_TOKEN"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
            return proc.stdout.strip()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return None


def parse_active_sessions(sessions_stdout: str) -> set[str]:
    """Parse active session names from `colab sessions` output."""
    names: set[str] = set()
    for line in (sessions_stdout or "").splitlines():
        m = _SESSION_LINE_RE.match(line.strip())
        if m:
            name = m.group(1).strip()
            if name and name != "?":
                names.add(name)
    return names


def is_session_missing_output(output: str, session: str) -> bool:
    """Detect `google-colab-cli` output indicating a missing, pruned, or preempted session."""
    if not output:
        return False
    low = output.lower()
    sess = session.lower()
    markers = (
        f"session '{sess}' not found",
        f'session "{sess}" not found',
        f"session '{sess}' appears to be lost",
        f'session "{sess}" appears to be lost',
        "appears to be lost (404/401)",
        "no active sessions found",
    )
    if any(m in low for m in markers):
        return True
    return "pruned" in low and "stale local session" in low


def is_transient_capacity_error(detail: str | Exception) -> bool:
    """Detect transient Colab HTTP 503 / GPU capacity / quota provisioning failures."""
    parts = [str(detail or "")]
    if isinstance(detail, Exception):
        for attr in ("stdout", "stderr"):
            val = getattr(detail, attr, None)
            if isinstance(val, str) and val:
                parts.append(val)
    text = "\n".join(parts).lower()
    if not text:
        return False
    markers = (
        "503",
        "service unavailable",
        "temporarily unavailable",
        "resource exhausted",
        "resource_exhausted",
        "no gpu",
        "gpu unavailable",
        "gpu capacity",
        "capacity",
        "cannot connect to gpu backend",
        "unable to assign",
        "usage limit",
        "quota",
        "too many requests",
        "429",
        "try again later",
        "backend is temporarily",
        "overloaded",
    )
    return any(m in text for m in markers)


def compute_respawn_backoff_sec(
    failure_count: int,
    *,
    base_backoff_sec: float = DEFAULT_RESPAWN_BACKOFF_SEC,
    max_backoff_sec: float = DEFAULT_MAX_RESPAWN_BACKOFF_SEC,
    transient_capacity: bool = False,
) -> float:
    """Compute bounded exponential retry delay (`60s -> 120s -> 240s -> 300s` on 503/capacity)."""
    cap = max(0.0, float(max_backoff_sec))
    base = max(0.0, float(base_backoff_sec))
    if transient_capacity:
        base = max(DEFAULT_CAPACITY_BACKOFF_BASE_SEC, base)
    exp = max(0, int(failure_count) - 1)
    delay = base * (2 ** min(exp, 10))
    return min(cap, delay) if cap > 0.0 else delay


@dataclass
class SupervisorConfig:
    colab_bin: str = DEFAULT_COLAB_BIN
    colab_config: str | None = None
    colab_auth: str | None = None
    session: str = DEFAULT_SESSION
    gpu: str = DEFAULT_GPU
    repo: str = DEFAULT_HF_REPO
    git_url: str = DEFAULT_GIT_URL
    git_ref: str = DEFAULT_GIT_REF
    remote_repo_dir: str = DEFAULT_REMOTE_REPO_DIR
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT
    stages: tuple[str, ...] = DEFAULT_STAGES
    time_budget_hours: float = DEFAULT_TIME_BUDGET_HOURS
    debug: int = DEFAULT_DEBUG_LEVEL
    interval_sec: float = DEFAULT_INTERVAL_SEC
    exec_timeout_sec: int = DEFAULT_EXEC_TIMEOUT_SEC
    bootstrap_timeout_sec: int = DEFAULT_BOOTSTRAP_TIMEOUT_SEC
    max_respawns: int = DEFAULT_MAX_RESPAWNS
    max_rollovers: int = DEFAULT_MAX_ROLLOVERS
    respawn_backoff_sec: float = DEFAULT_RESPAWN_BACKOFF_SEC
    max_respawn_backoff_sec: float = DEFAULT_MAX_RESPAWN_BACKOFF_SEC
    log_file: str = DEFAULT_LOG_FILE
    tail_file: str = DEFAULT_TAIL_FILE
    keep_on_complete: bool = False
    once: bool = False
    extra_overrides: tuple[str, ...] = ()

    @property
    def stages_csv(self) -> str:
        return ",".join(self.stages)


@dataclass
class ProbeResult:
    pid_alive: bool
    exit_code: int | None
    gpu_summary: str
    last_log_lines: list[str]
    disk_free_gib: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProbeResult:
        raw_exit = raw.get("exit_code")
        exit_code = int(raw_exit) if raw_exit is not None else None
        raw_lines = raw.get("last_log_lines") or []
        lines = [str(x) for x in raw_lines] if isinstance(raw_lines, list) else [str(raw_lines)]
        return cls(
            pid_alive=bool(raw.get("pid_alive", False)),
            exit_code=exit_code,
            gpu_summary=str(raw.get("gpu_summary") or "unknown"),
            last_log_lines=lines,
            disk_free_gib=float(raw.get("disk_free_gib") or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid_alive": self.pid_alive,
            "exit_code": self.exit_code,
            "gpu_summary": self.gpu_summary,
            "last_log_lines": list(self.last_log_lines),
            "disk_free_gib": self.disk_free_gib,
        }


def build_remote_run_pipeline_script(cfg: SupervisorConfig) -> str:
    """Build `/content/runs/run_pipeline.sh` (`ps aux` safe: loads `HF_TOKEN` from `0600` file, never argv)."""
    run_root_q = shlex.quote(cfg.remote_run_root)
    repo_dir_q = shlex.quote(cfg.remote_repo_dir)
    hf_repo_q = shlex.quote(cfg.repo)
    stages_q = shlex.quote(cfg.stages_csv)
    hours_q = shlex.quote(f"{cfg.time_budget_hours:g}")
    debug_q = shlex.quote(str(int(cfg.debug)))
    ov_parts = " ".join(f"-s {shlex.quote(ov)}" for ov in cfg.extra_overrides)
    ov_suffix = f" {ov_parts}" if ov_parts else ""

    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"RUN_ROOT={run_root_q}\n"
        f"REPO_DIR={repo_dir_q}\n"
        'mkdir -p "$RUN_ROOT"\n'
        'rm -f "$RUN_ROOT/pipeline.exit_code" "$RUN_ROOT/pipeline.exit_code.tmp"\n'
        'printf \'%s\\n\' "$$" > "$RUN_ROOT/pipeline.pid.tmp"\n'
        'mv -f "$RUN_ROOT/pipeline.pid.tmp" "$RUN_ROOT/pipeline.pid"\n'
        "_qloop_write_exit() {\n"
        "  local rc=$?\n"
        '  printf \'%s\\n\' "$rc" > "$RUN_ROOT/pipeline.exit_code.tmp"\n'
        '  mv -f "$RUN_ROOT/pipeline.exit_code.tmp" "$RUN_ROOT/pipeline.exit_code"\n'
        "}\n"
        "trap _qloop_write_exit EXIT\n"
        "if [ -f /content/.hf_token ]; then\n"
        '  export HF_TOKEN="$(cat /content/.hf_token)"\n'
        "fi\n"
        "unset HF_HUB_ENABLE_HF_TRANSFER || true\n"
        "export HF_XET_HIGH_PERFORMANCE=1\n"
        'export QLOOP_RUN_ROOT="$RUN_ROOT"\n'
        'export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"\n'
        'cd "$REPO_DIR" || exit 2\n'
        f"/usr/bin/python3 -u -m qloop sync pipeline --repo {hf_repo_q} "
        f"--debug {debug_q} --time-budget-hours {hours_q} --stages {stages_q}{ov_suffix}\n"
        "exit $?\n"
    )


def build_bootstrap_python_script(cfg: SupervisorConfig) -> str:
    """Build remote Python bootstrap code executed via `colab exec`."""
    payload = {
        "git_url": normalize_git_https_url(cfg.git_url),
        "git_ref": cfg.git_ref,
        "repo_dir": cfg.remote_repo_dir,
        "run_root": cfg.remote_run_root,
        "wrapper_sh": build_remote_run_pipeline_script(cfg),
        "sentinel": BOOTSTRAP_SENTINEL,
    }
    cfg_json = json.dumps(payload)
    return f"""import json
import os
import shlex
import subprocess
import time

_CFG = json.loads({cfg_json!r})
_TOK = os.environ.pop("HF_TOKEN", "").strip()
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


def _scrub(text):
    if not text:
        return ""
    out = str(text)
    if _TOK:
        out = out.replace(_TOK, "<REDACTED>")
    return out


def _write_private(path, content, mode=0o600):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    os.chmod(path, mode)


def _pid_alive(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = f"/proc/{{pid}}/stat"
    if os.path.exists(stat_path):
        try:
            with open(stat_path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            rparen = raw.rfind(")")
            if rparen != -1:
                parts = raw[rparen + 2 :].split()
                if parts and parts[0] == "Z":
                    return False
        except OSError:
            return False
    return True


def _read_pid(pid_file):
    if not os.path.isfile(pid_file):
        return None
    try:
        with open(pid_file, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _run_checked(cmd, cwd=None, extra_env=None):
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + "\\n" + proc.stderr).strip()[-1200:]
        raise RuntimeError(f"command {{cmd[0]}} exited {{proc.returncode}}: {{_scrub(tail)}}")
    return proc.stdout.strip()


try:
    if _TOK:
        _write_private("/root/.cache/huggingface/token", _TOK + "\\n", 0o600)
        _write_private("/content/.hf_token", _TOK + "\\n", 0o600)

    run_root = _CFG["run_root"]
    repo_dir = _CFG["repo_dir"]
    os.makedirs(run_root, exist_ok=True)
    pid_file = os.path.join(run_root, "pipeline.pid")
    exit_file = os.path.join(run_root, "pipeline.exit_code")
    wrapper_path = os.path.join(run_root, "run_pipeline.sh")
    stdout_log = os.path.join(run_root, "pipeline_stdout.log")

    cur_pid = _read_pid(pid_file)
    already_alive = _pid_alive(cur_pid)
    launched = False

    if not already_alive:
        git_dir = os.path.join(repo_dir, ".git")
        if not os.path.isdir(git_dir):
            os.makedirs(os.path.dirname(repo_dir) or "/content", exist_ok=True)
            _run_checked(["git", "clone", _CFG["git_url"], repo_dir])
            if _CFG["git_ref"] and _CFG["git_ref"] != "origin/main":
                _run_checked(["git", "-C", repo_dir, "checkout", "-f", _CFG["git_ref"]])
        else:
            _run_checked(["git", "-C", repo_dir, "remote", "set-url", "origin", _CFG["git_url"]])
            _run_checked(["git", "-C", repo_dir, "fetch", "origin"])
            _run_checked(["git", "-C", repo_dir, "checkout", "-f", _CFG["git_ref"]])

        chk = subprocess.run(
            [
                "/usr/bin/python3",
                "-c",
                "import qloop, transformers, huggingface_hub, datasets, compressed_tensors; "
                "assert int(transformers.__version__.split('.')[0]) >= 5",
            ],
            cwd=repo_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if chk.returncode != 0:
            _run_checked(
                ["bash", "scripts/colab_setup.sh"],
                cwd=repo_dir,
                extra_env={{"PYTHON": "/usr/bin/python3"}},
            )
        else:
            _run_checked(
                ["/usr/bin/python3", "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "."],
                cwd=repo_dir,
            )

        _write_private(wrapper_path, _CFG["wrapper_sh"], 0o700)
        for stale_path in (exit_file, pid_file):
            try:
                os.remove(stale_path)
            except OSError:
                pass

        launch_cmd = (
            f"setsid nohup bash {{shlex.quote(wrapper_path)}} "
            f"> {{shlex.quote(stdout_log)}} 2>&1 < /dev/null & echo $!"
        )
        out_pid = _run_checked(["bash", "-c", launch_cmd], cwd=repo_dir)
        fallback_pid = int(out_pid.splitlines()[-1].strip())
        cur_pid = None
        for _ in range(20):
            cur_pid = _read_pid(pid_file)
            if cur_pid is not None:
                break
            time.sleep(0.1)
        if cur_pid is None:
            cur_pid = fallback_pid
            _write_private(pid_file, f"{{cur_pid}}\\n", 0o644)
        launched = True

    head_sha = ""
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        p = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            head_sha = p.stdout.strip()

    print(
        _CFG["sentinel"]
        + json.dumps(
            {{
                "ok": True,
                "launched": launched,
                "pid": cur_pid,
                "pid_alive": _pid_alive(cur_pid),
                "head": head_sha,
            }}
        ),
        flush=True,
    )
except Exception as exc:
    print(
        _CFG["sentinel"] + json.dumps({{"ok": False, "error": _scrub(str(exc))}}),
        flush=True,
    )
"""


def build_probe_python_script(
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT,
    tail_lines: int = 15,
) -> str:
    """Build lightweight (~1s) remote Jupyter kernel probe returning JSON."""
    payload = {
        "run_root": remote_run_root,
        "tail_lines": int(tail_lines),
        "sentinel": PROBE_SENTINEL,
    }
    cfg_json = json.dumps(payload)
    return f"""import json
import os
import shutil
import subprocess

_CFG = json.loads({cfg_json!r})
run_root = _CFG["run_root"]
pid_file = os.path.join(run_root, "pipeline.pid")
exit_file = os.path.join(run_root, "pipeline.exit_code")
log_file = os.path.join(run_root, "pipeline_stdout.log")

pid = None
if os.path.isfile(pid_file):
    try:
        with open(pid_file, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pid = None

pid_alive = False
if pid is not None and pid > 0:
    try:
        os.kill(pid, 0)
        pid_alive = True
        stat_p = f"/proc/{{pid}}/stat"
        if os.path.exists(stat_p):
            with open(stat_p, encoding="utf-8", errors="replace") as sf:
                raw = sf.read()
            rparen = raw.rfind(")")
            if rparen != -1:
                parts = raw[rparen + 2 :].split()
                if parts and parts[0] == "Z":
                    pid_alive = False
    except OSError:
        pid_alive = False

exit_code = None
if not pid_alive and os.path.isfile(exit_file):
    try:
        with open(exit_file, encoding="utf-8") as f:
            exit_code = int(f.read().strip())
    except (OSError, ValueError):
        exit_code = None

gpu_summary = "unavailable"
try:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if smi.returncode == 0 and smi.stdout.strip():
        parts = [p.strip() for p in smi.stdout.splitlines()[0].split(",")]
        if len(parts) >= 6:
            gpu_summary = (
                f"{{parts[0]}} util={{parts[1]}}% mem={{parts[2]}}/{{parts[3]}}MiB "
                f"pwr={{parts[4]}}W temp={{parts[5]}}C"
            )
        else:
            gpu_summary = smi.stdout.splitlines()[0].strip()
except Exception:
    gpu_summary = "unavailable"

last_log_lines = []
if os.path.isfile(log_file):
    try:
        with open(log_file, "rb") as lf:
            lf.seek(0, os.SEEK_END)
            size = lf.tell()
            lf.seek(max(0, size - 32768), os.SEEK_SET)
            chunk = lf.read().decode("utf-8", errors="replace")
        non_empty = [ln.rstrip() for ln in chunk.splitlines() if ln.strip()]
        last_log_lines = non_empty[-int(_CFG["tail_lines"]) :]
    except OSError:
        last_log_lines = []

disk_free_gib = 0.0
try:
    target = "/content" if os.path.exists("/content") else "/"
    disk_free_gib = round(shutil.disk_usage(target).free / float(2**30), 2)
except OSError:
    disk_free_gib = 0.0

print(
    _CFG["sentinel"]
    + json.dumps(
        {{
            "pid_alive": pid_alive,
            "exit_code": exit_code,
            "gpu_summary": gpu_summary,
            "last_log_lines": last_log_lines,
            "disk_free_gib": disk_free_gib,
        }}
    ),
    flush=True,
)
"""


def build_tail_python_script(
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT,
    max_lines: int = 200,
) -> str:
    """Build remote Python snippet to read the tail of `/content/runs/pipeline_stdout.log`."""
    payload = {
        "log_file": os.path.join(remote_run_root, "pipeline_stdout.log"),
        "max_lines": int(max_lines),
        "sentinel": TAIL_SENTINEL,
    }
    cfg_json = json.dumps(payload)
    return f"""import json
import os

_CFG = json.loads({cfg_json!r})
tail_text = ""
if os.path.isfile(_CFG["log_file"]):
    try:
        with open(_CFG["log_file"], "rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - 65536), os.SEEK_SET)
            raw = f.read().decode("utf-8", errors="replace")
        tail_text = "\\n".join(raw.splitlines()[-int(_CFG["max_lines"]) :])
    except OSError:
        tail_text = ""
print(_CFG["sentinel"] + json.dumps({{"tail": tail_text}}), flush=True)
"""


def _extract_json_payload(stdout: str, sentinel: str) -> dict[str, Any] | None:
    """Extract JSON dict emitted after `sentinel` (falling back to bare JSON stdout)."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        idx = line.find(sentinel)
        if idx != -1:
            candidate = line[idx + len(sentinel) :].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    stripped = stdout.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    for line in reversed(stdout.splitlines()):
        ln = line.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                parsed = json.loads(ln)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def fetch_hub_manifest_dict(
    repo_id: str,
    *,
    hf_token: str | None = None,
    timeout_sec: float = 20.0,
) -> dict[str, Any] | None:
    """Fetch `manifest.json` from Hugging Face Hub using stdlib `urllib.request`.

    Returns:
      - Parsed `dict` when `manifest.json` exists (`200 OK`).
      - `None` when `manifest.json` does not exist yet (`404 Not Found`).
    Raises `ColabSupervisorError` (redacted) on network or server errors.
    """
    secrets = (hf_token,) if hf_token else ()
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.strip("/").split("/"))
    url = f"https://huggingface.co/{quoted_repo}/resolve/main/manifest.json"
    headers = {
        "Accept": "application/json",
        "User-Agent": "qloop-colab-supervisor/1.0",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ColabSupervisorError(
            f"HF Hub manifest HTTP {exc.code} for {repo_id}: {redact_secret(str(exc), secrets)}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ColabSupervisorError(
            f"HF Hub manifest fetch failed for {repo_id}: {redact_secret(str(exc), secrets)}"
        ) from None


def _tail_indicates_clean_pause(tail_text: str, last_log_lines: Sequence[str] = ()) -> bool:
    """Return True when remote logs indicate `qloop sync pipeline` paused cleanly before completion."""
    needle = CLEAN_PAUSE_LOG_MARKER.lower()
    if needle in (tail_text or "").lower():
        return True
    return any(needle in str(line).lower() for line in last_log_lines)


def _format_progress_summary(fingerprint: Sequence[tuple[str, str, int]]) -> str:
    """Render a compact `stage=status@step` summary for rollover and guard logs."""
    if not fingerprint:
        return "hub=unavailable"
    return ", ".join(f"{st}={status}@{step}" for st, status, step in fingerprint)


class ColabSupervisor:
    """Local keepalive, auto-stop, and preemption auto-respawn supervisor for `colab` CLI."""

    def __init__(
        self,
        cfg: SupervisorConfig,
        *,
        hf_token: str | None = None,
        env: Mapping[str, str] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        manifest_fetcher: Callable[[str], dict[str, Any] | None] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._runner = runner
        self._manifest_fetcher = manifest_fetcher
        self._sleep_fn = sleep_fn
        self.hf_token = hf_token if hf_token is not None else resolve_local_hf_token(env=env, runner=runner)

    @property
    def _secrets(self) -> tuple[str, ...]:
        return (self.hf_token,) if self.hf_token else ()

    def redact(self, text: str) -> str:
        return redact_secret(text, self._secrets)

    def _sleep(self, seconds: float) -> None:
        fn = time.sleep if self._sleep_fn is None else self._sleep_fn
        fn(max(0.0, float(seconds)))

    def log(self, message: str) -> None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        safe_line = f"{ts} {self.redact(message)}"
        print(safe_line, flush=True)
        if self.cfg.log_file:
            log_dir = os.path.dirname(os.path.abspath(self.cfg.log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.cfg.log_file, "a", encoding="utf-8") as lf:
                lf.write(safe_line + "\n")

    def _colab_prefix(self) -> list[str]:
        cmd = [self.cfg.colab_bin]
        if self.cfg.colab_config:
            cmd.extend(["--config", self.cfg.colab_config])
        if self.cfg.colab_auth:
            cmd.extend(["--auth", self.cfg.colab_auth])
        return cmd

    def _run_colab(
        self,
        subargs: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_sec: float = 120.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = [*self._colab_prefix(), *subargs]
        safe_argv = redact_argv(argv, self._secrets)
        run_fn = subprocess.run if self._runner is None else self._runner
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": timeout_sec,
            "check": False,
        }
        if input_text is not None:
            kwargs["input"] = input_text
        else:
            kwargs["stdin"] = subprocess.DEVNULL

        try:
            proc = run_fn(argv, **kwargs)
        except subprocess.TimeoutExpired as exc:
            raw_out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            out_s = self.redact(raw_out)
            err_s = self.redact(raw_err)
            raise ColabCommandError(
                f"command timed out after {timeout_sec:g}s: {' '.join(safe_argv)}",
                returncode=124,
                cmd=safe_argv,
                stdout=out_s,
                stderr=err_s,
            ) from None
        except OSError as exc:
            raise ColabCommandError(
                f"failed to execute {' '.join(safe_argv)}: {self.redact(str(exc))}",
                returncode=127,
                cmd=safe_argv,
            ) from None

        safe_out = self.redact(proc.stdout or "")
        safe_err = self.redact(proc.stderr or "")
        combined = f"{safe_out}\n{safe_err}".strip()

        if check and proc.returncode != 0:
            err_cls = (
                ColabSessionLostError
                if is_session_missing_output(combined, self.cfg.session)
                else ColabCommandError
            )
            detail = combined[-600:] if combined else f"exit code {proc.returncode}"
            raise err_cls(
                f"{' '.join(safe_argv[:4])} failed (rc={proc.returncode}): {detail}",
                returncode=proc.returncode,
                cmd=safe_argv,
                stdout=safe_out,
                stderr=safe_err,
            )

        return subprocess.CompletedProcess(
            args=safe_argv,
            returncode=proc.returncode,
            stdout=safe_out,
            stderr=safe_err,
        )

    def list_active_sessions(self) -> set[str]:
        proc = self._run_colab(["sessions"], timeout_sec=60.0, check=True)
        return parse_active_sessions(proc.stdout)

    def stop_session(self) -> bool:
        proc = self._run_colab(["stop", "-s", self.cfg.session], timeout_sec=90.0, check=False)
        if proc.returncode == 0:
            return True
        combined = f"{proc.stdout}\n{proc.stderr}".strip()
        if is_session_missing_output(combined, self.cfg.session):
            return True
        self.log(f"[WARN] colab stop -s {self.cfg.session} exited {proc.returncode}: {combined[-300:]}")
        return False

    def ensure_session_and_launch(self) -> dict[str, Any]:
        """Ensure `<session>` is active, attach URL is logged, repo/auth are bootstrapped, and pipeline is running."""
        active = self.list_active_sessions()
        created_new = False
        if self.cfg.session not in active:
            new_args = ["new", "-s", self.cfg.session]
            if self.cfg.gpu and self.cfg.gpu.upper() != "CPU":
                new_args.extend(["--gpu", self.cfg.gpu])
            self.log(f"[BOOT] provisioning session '{self.cfg.session}' (gpu={self.cfg.gpu or 'CPU'})...")
            self._run_colab(new_args, timeout_sec=300.0, check=True)
            created_new = True
            self.log(f"[BOOT] session '{self.cfg.session}' provisioned")
        else:
            self.log(f"[BOOT] attaching to existing active session '{self.cfg.session}'")

        try:
            url_proc = self._run_colab(["url", "-s", self.cfg.session], timeout_sec=30.0, check=False)
            if url_proc.returncode == 0 and url_proc.stdout.strip():
                url_line = url_proc.stdout.strip().splitlines()[-1].strip()
                self.log(f"[URL] session={self.cfg.session} {url_line}")

            exec_args = [
                "exec",
                "-s",
                self.cfg.session,
                "--timeout",
                str(int(self.cfg.bootstrap_timeout_sec)),
            ]
            if self.hf_token:
                exec_args.extend(["--env", f"HF_TOKEN={self.hf_token}"])

            boot_proc = self._run_colab(
                exec_args,
                input_text=build_bootstrap_python_script(self.cfg),
                timeout_sec=float(max(self.cfg.bootstrap_timeout_sec + 120, 300)),
                check=True,
            )
            payload = _extract_json_payload(boot_proc.stdout, BOOTSTRAP_SENTINEL)
            if payload is not None:
                if not payload.get("ok", True):
                    raise ColabCommandError(
                        f"remote bootstrap failed: {self.redact(str(payload.get('error') or 'unknown error'))}",
                        returncode=1,
                        cmd=boot_proc.args,
                        stdout=boot_proc.stdout,
                        stderr=boot_proc.stderr,
                    )
                self.log(
                    f"[BOOT] ready session={self.cfg.session} launched={payload.get('launched')} "
                    f"pid={payload.get('pid')} head={payload.get('head') or '-'}"
                )
                return payload

            if "Traceback (most recent call last):" in boot_proc.stderr:
                raise ColabCommandError(
                    f"remote bootstrap traceback: {boot_proc.stderr[-600:]}",
                    returncode=1,
                    cmd=boot_proc.args,
                    stdout=boot_proc.stdout,
                    stderr=boot_proc.stderr,
                )
            self.log(f"[BOOT] ready session={self.cfg.session}")
            return {"ok": True}
        except Exception:
            if created_new:
                self.log(
                    f"[WARN] bootstrap failed on newly provisioned session '{self.cfg.session}'; "
                    "stopping VM to avoid idle burn"
                )
                self.stop_session()
            raise

    def probe_once(self) -> ProbeResult:
        """Execute a ~1s probe inside the remote Jupyter kernel (keeping kernel alive & refreshing proxy token)."""
        exec_args = [
            "exec",
            "-s",
            self.cfg.session,
            "--timeout",
            str(int(self.cfg.exec_timeout_sec)),
        ]
        proc = self._run_colab(
            exec_args,
            input_text=build_probe_python_script(self.cfg.remote_run_root),
            timeout_sec=float(max(self.cfg.exec_timeout_sec + 60, 120)),
            check=False,
        )
        combined = f"{proc.stdout}\n{proc.stderr}".strip()
        if is_session_missing_output(combined, self.cfg.session):
            raise ColabSessionLostError(
                f"session '{self.cfg.session}' not found / lost",
                returncode=proc.returncode or 1,
                cmd=proc.args,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        if proc.returncode != 0:
            try:
                active = self.list_active_sessions()
            except ColabCommandError:
                active = {self.cfg.session}
            if self.cfg.session not in active:
                raise ColabSessionLostError(
                    f"session '{self.cfg.session}' disappeared from active sessions",
                    returncode=proc.returncode,
                    cmd=proc.args,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
            raise ColabCommandError(
                f"probe exec failed (rc={proc.returncode}): {combined[-400:]}",
                returncode=proc.returncode,
                cmd=proc.args,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        payload = _extract_json_payload(proc.stdout, PROBE_SENTINEL)
        if payload is None or "pid_alive" not in payload:
            raise ColabCommandError(
                f"probe returned unparseable output: {combined[-400:]}",
                returncode=1,
                cmd=proc.args,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        probe = ProbeResult.from_dict(payload)
        probe.gpu_summary = self.redact(probe.gpu_summary)
        probe.last_log_lines = [self.redact(ln) for ln in probe.last_log_lines]
        return probe

    def capture_last_tail(self, fallback_lines: Sequence[str] = ()) -> str:
        """Fetch tail of `/content/runs/pipeline_stdout.log` into local `cfg.tail_file` (best-effort, redacted)."""
        tail_text = ""
        try:
            proc = self._run_colab(
                ["exec", "-s", self.cfg.session, "--timeout", "60"],
                input_text=build_tail_python_script(self.cfg.remote_run_root),
                timeout_sec=90.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                payload = _extract_json_payload(proc.stdout, TAIL_SENTINEL)
                if payload is not None and "tail" in payload:
                    tail_text = str(payload.get("tail") or "")
                elif "pid_alive" not in (payload or {}):
                    tail_text = proc.stdout.strip()
        except Exception:
            tail_text = ""

        if not tail_text and fallback_lines:
            tail_text = "\n".join(str(x) for x in fallback_lines)

        safe_tail = self.redact(tail_text)
        if self.cfg.tail_file:
            tail_dir = os.path.dirname(os.path.abspath(self.cfg.tail_file))
            if tail_dir:
                os.makedirs(tail_dir, exist_ok=True)
            with open(self.cfg.tail_file, "w", encoding="utf-8") as tf:
                tf.write(safe_tail + ("\n" if safe_tail and not safe_tail.endswith("\n") else ""))
        return safe_tail

    def _fetch_hub_manifest_with_retries(
        self,
        retries: int = 3,
        retry_sleep_sec: float = 2.0,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Fetch HF Hub `manifest.json` with bounded retries.

        Returns `(manifest_dict_or_none, fetch_ok)`:
          - `(dict, True)` when `manifest.json` was fetched cleanly.
          - `(None, True)` when `manifest.json` does not exist yet (HTTP 404).
          - `(None, False)` when Hub remained unreachable after all retries.
        """
        attempts = max(1, int(retries))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if self._manifest_fetcher is not None:
                    manifest = self._manifest_fetcher(self.cfg.repo)
                else:
                    manifest = fetch_hub_manifest_dict(self.cfg.repo, hf_token=self.hf_token)
                return manifest, True
            except ColabSupervisorError as exc:
                last_exc = exc
                if attempt < attempts:
                    self.log(
                        f"[WARN] Hub manifest fetch failed (attempt {attempt}/{attempts}): "
                        f"{self.redact(str(exc))}; retrying"
                    )
                    delay = min(float(retry_sleep_sec) * attempt, float(self.cfg.respawn_backoff_sec))
                    self._sleep(delay)
        self.log(
            f"[WARN] Hub manifest unreachable after {attempts} attempt(s): "
            f"{self.redact(str(last_exc) if last_exc is not None else 'unknown error')}"
        )
        return None, False

    def _hub_stage_progress(
        self,
        retries: int = 3,
    ) -> tuple[list[str], tuple[tuple[str, str, int], ...], bool]:
        """Query Hub manifest and return `(incomplete_stages, progress_fingerprint, fetch_ok)`."""
        manifest, fetch_ok = self._fetch_hub_manifest_with_retries(retries=retries)
        if not fetch_ok:
            return [], (), False

        stages_map = manifest.get("stages") if isinstance(manifest, dict) else None
        if not isinstance(stages_map, dict):
            stages_map = {}

        fingerprint_items: list[tuple[str, str, int]] = []
        incomplete: list[str] = []
        for st in self.cfg.stages:
            raw_info = stages_map.get(st)
            info = raw_info if isinstance(raw_info, dict) else {}
            raw_status = info.get("status")
            status = str(raw_status) if raw_status is not None else "pending"
            raw_step = info.get("step")
            try:
                step = int(raw_step) if raw_step is not None else 0
            except (TypeError, ValueError):
                step = 0
            fingerprint_items.append((st, status, step))
            if status != "completed":
                incomplete.append(st)
        return incomplete, tuple(fingerprint_items), True

    def _try_stop_session(self) -> bool:
        try:
            return self.stop_session()
        except ColabSupervisorError as exc:
            self.log(f"[WARN] colab stop -s {self.cfg.session} raised error: {exc}")
            return False

    def _stop_and_confirm_session_gone(self, *, max_retries: int = 2) -> bool:
        """Stop the expiring VM and verify it is no longer listed in `colab sessions`."""
        self._try_stop_session()
        self._sleep(self.cfg.respawn_backoff_sec)
        short_sleep = min(5.0, max(0.0, float(self.cfg.respawn_backoff_sec)))
        retries = max(0, int(max_retries))
        for retry_idx in range(retries + 1):
            try:
                active = self.list_active_sessions()
            except ColabCommandError as exc:
                self.log(
                    f"[WARN] failed to verify session '{self.cfg.session}' termination "
                    f"(check {retry_idx + 1}/{retries + 1}): {exc}"
                )
                active = {self.cfg.session}
            if self.cfg.session not in active:
                return True
            if retry_idx < retries:
                self.log(
                    f"[WARN] session '{self.cfg.session}' still listed after stop "
                    f"(retry {retry_idx + 1}/{retries}); retrying colab stop"
                )
                self._try_stop_session()
                self._sleep(short_sleep)
        return False

    def _retry_backoff_sec(self, failure_count: int, exc: Exception | None = None) -> float:
        is_cap = exc is not None and is_transient_capacity_error(exc)
        return compute_respawn_backoff_sec(
            failure_count,
            base_backoff_sec=self.cfg.respawn_backoff_sec,
            max_backoff_sec=self.cfg.max_respawn_backoff_sec,
            transient_capacity=is_cap,
        )

    def _launch_fresh_rollover_session(self, rollover_count: int) -> bool:
        """Provision and bootstrap a fresh 24h VM after old session termination, retrying transient errors."""
        max_attempts = 1 + max(0, int(self.cfg.max_respawns))
        last_exc: ColabSupervisorError | None = None
        cap_failures = 0
        other_failures = 0
        for attempt in range(1, max_attempts + 1):
            try:
                self.ensure_session_and_launch()
                return True
            except ColabSupervisorError as exc:
                last_exc = exc
                is_cap = is_transient_capacity_error(exc)
                if is_cap:
                    cap_failures += 1
                    fail_idx = cap_failures
                else:
                    other_failures += 1
                    fail_idx = other_failures
                if attempt < max_attempts:
                    delay = self._retry_backoff_sec(fail_idx, exc)
                    kind = "transient capacity/503" if is_cap else "transient"
                    self.log(
                        f"[WARN] rollover {rollover_count}/{self.cfg.max_rollovers} "
                        f"provision/bootstrap attempt {attempt}/{max_attempts} failed ({kind}): {exc}; "
                        f"retrying after {delay:g}s"
                    )
                    self._sleep(delay)
                else:
                    self.log(
                        f"[WARN] rollover {rollover_count}/{self.cfg.max_rollovers} "
                        f"provision/bootstrap attempt {attempt}/{max_attempts} failed: {exc}"
                    )
        if isinstance(last_exc, ColabCommandError) and last_exc.returncode == 124:
            self._try_stop_session()
        self.log(
            f"[ERROR] rollover {rollover_count}/{self.cfg.max_rollovers} failed to provision/bootstrap "
            f"fresh session '{self.cfg.session}' after {max_attempts} attempt(s)"
        )
        return False

    def _initial_launch_with_retry(self) -> dict[str, Any]:
        """Run initial provision/bootstrap, retrying transient Colab 503 / GPU capacity errors with exp backoff."""
        max_retries = 0 if self.cfg.once else max(0, int(self.cfg.max_respawns))
        cap_failures = 0
        while True:
            try:
                return self.ensure_session_and_launch()
            except ColabSupervisorError as exc:
                if not is_transient_capacity_error(exc) or cap_failures >= max_retries:
                    raise
                cap_failures += 1
                delay = self._retry_backoff_sec(cap_failures, exc)
                self.log(
                    f"[WARN] initial provision/bootstrap hit transient capacity/503 "
                    f"(retry {cap_failures}/{max_retries}): {exc}; waiting {delay:g}s"
                )
                self._sleep(delay)

    def _finish_completed_exit_zero(self) -> int:
        if self.cfg.keep_on_complete:
            self.log("[DONE] pipeline exited 0; keeping session (--keep-on-complete)")
        else:
            self.stop_session()
            self.log("[DONE] pipeline exited 0; stopped session")
        return 0

    def incomplete_hub_stages(self) -> list[str]:
        """Return requested stages whose HF Hub `manifest.json` status is not `'completed'`."""
        try:
            if self._manifest_fetcher is not None:
                manifest = self._manifest_fetcher(self.cfg.repo)
            else:
                manifest = fetch_hub_manifest_dict(self.cfg.repo, hf_token=self.hf_token)
        except ColabSupervisorError as exc:
            self.log(f"[WARN] could not verify Hub manifest ({exc}); assuming requested stages remain incomplete")
            return list(self.cfg.stages)

        if not manifest:
            return list(self.cfg.stages)
        stages_map = manifest.get("stages")
        if not isinstance(stages_map, dict):
            return list(self.cfg.stages)
        return [
            st
            for st in self.cfg.stages
            if (stages_map.get(st) if isinstance(stages_map.get(st), dict) else {}).get("status") != "completed"
        ]

    def _respawn_or_fail(self, reason: str, respawn_count: int) -> tuple[bool, int, int]:
        """Handle preemption/lost session: return `(did_respawn, new_respawn_count, exit_code_if_stopping)`."""
        incomplete = self.incomplete_hub_stages()
        if not incomplete:
            self.log(
                f"[DONE] {reason}; all requested stages ({self.cfg.stages_csv}) are completed on Hub"
            )
            return False, respawn_count, 0

        cap_failures = 0
        other_failures = 0
        next_delay = float(self.cfg.respawn_backoff_sec)
        while respawn_count < self.cfg.max_respawns:
            respawn_count += 1
            self.log(
                f"[RESPAWN] {reason} while stages {incomplete} incomplete; "
                f"respawn {respawn_count}/{self.cfg.max_respawns} after {next_delay:g}s"
            )
            self._sleep(next_delay)
            try:
                self.ensure_session_and_launch()
                return True, respawn_count, 0
            except ColabSupervisorError as exc:
                is_cap = is_transient_capacity_error(exc)
                if is_cap:
                    cap_failures += 1
                    next_delay = self._retry_backoff_sec(cap_failures, exc)
                else:
                    other_failures += 1
                    next_delay = self._retry_backoff_sec(other_failures, None)
                kind = "transient capacity/503" if is_cap else "transient"
                self.log(
                    f"[WARN] respawn attempt {respawn_count}/{self.cfg.max_respawns} failed ({kind}): {exc}"
                )

        self.log(
            f"[ERROR] {reason} with incomplete stages {incomplete}; "
            f"max_respawns ({self.cfg.max_respawns}) reached"
        )
        return False, respawn_count, 1

    def run(self) -> int:
        """Main supervisor loop: bootstrap -> periodic Jupyter probe -> auto-stop, rollover, or respawn."""
        respawn_count = 0
        rollover_count = 0
        last_rollover_progress: tuple[tuple[str, str, int], ...] | None = None
        consecutive_probe_errors = 0

        self._initial_launch_with_retry()

        while True:
            try:
                probe = self.probe_once()
                consecutive_probe_errors = 0
            except ColabSessionLostError as exc:
                did_respawn, respawn_count, rc = self._respawn_or_fail(
                    f"session '{self.cfg.session}' preempted/lost ({exc})",
                    respawn_count,
                )
                if not did_respawn:
                    return rc
                continue
            except ColabCommandError as exc:
                consecutive_probe_errors += 1
                self.log(f"[WARN] probe error ({consecutive_probe_errors}/3): {exc}")
                if self.cfg.once:
                    return 1
                if consecutive_probe_errors >= 3:
                    self.log(
                        f"[ERROR] 3 consecutive probe errors on '{self.cfg.session}'; stopping session"
                    )
                    self.stop_session()
                    return 1
                self._sleep(min(30.0, self.cfg.interval_sec))
                continue

            last_line = probe.last_log_lines[-1] if probe.last_log_lines else "-"
            self.log(
                f"[KEEPALIVE] session={self.cfg.session} pid_alive={probe.pid_alive} "
                f"exit_code={probe.exit_code} disk_free={probe.disk_free_gib:.1f}GiB "
                f"gpu={probe.gpu_summary} | {last_line}"
            )

            if not probe.pid_alive and probe.exit_code is not None:
                tail_text = self.capture_last_tail(probe.last_log_lines)
                if probe.exit_code == 0:
                    if self.cfg.once or self.cfg.max_rollovers <= 0:
                        return self._finish_completed_exit_zero()

                    incomplete, progress_fp, fetch_ok = self._hub_stage_progress(retries=3)
                    if not fetch_ok:
                        if _tail_indicates_clean_pause(tail_text, probe.last_log_lines):
                            incomplete = list(self.cfg.stages)
                        else:
                            self.log(
                                "[WARN] Hub manifest unavailable on exit 0 and tail has no clean-pause marker; "
                                "treating run as completed to avoid burning VMs"
                            )
                            incomplete = []

                    if not incomplete:
                        return self._finish_completed_exit_zero()

                    prog_summary = (
                        _format_progress_summary(progress_fp)
                        if fetch_ok
                        else "hub=unavailable (tail matched clean pause)"
                    )

                    if (
                        fetch_ok
                        and last_rollover_progress is not None
                        and progress_fp == last_rollover_progress
                    ):
                        self.stop_session()
                        self.log(
                            f"[ERROR] pipeline exited 0 with zero Hub progress since prior rollover "
                            f"({prog_summary}); stopped session '{self.cfg.session}'"
                        )
                        return 1

                    if rollover_count >= self.cfg.max_rollovers:
                        self.stop_session()
                        self.log(
                            f"[ERROR] pipeline exited 0 with incomplete stages {incomplete} ({prog_summary}); "
                            f"max_rollovers ({self.cfg.max_rollovers}) reached; stopped session '{self.cfg.session}'"
                        )
                        return 1

                    rollover_count += 1
                    if fetch_ok:
                        last_rollover_progress = progress_fp
                    self.log(
                        f"[ROLLOVER] pipeline exited 0 with incomplete stages {incomplete} ({prog_summary}); "
                        f"rollover {rollover_count}/{self.cfg.max_rollovers} — stopping expiring session "
                        f"'{self.cfg.session}' before provisioning fresh VM"
                    )
                    if not self._stop_and_confirm_session_gone():
                        self.log(
                            f"[ERROR] rollover {rollover_count}/{self.cfg.max_rollovers} aborted: "
                            f"old session '{self.cfg.session}' still active after stop retries; "
                            "refusing to re-attach to expiring VM"
                        )
                        return 1
                    if not self._launch_fresh_rollover_session(rollover_count):
                        return 1
                    consecutive_probe_errors = 0
                    continue

                self.stop_session()
                self.log(f"[ERROR] pipeline exited {probe.exit_code}; stopped session")
                return probe.exit_code if 1 <= probe.exit_code <= 255 else 1

            if not probe.pid_alive and probe.exit_code is None:
                if self.cfg.once:
                    self.log("[WARN] remote PID is not alive and no exit_code file is present (--once)")
                    return 1
                did_respawn, respawn_count, rc = self._respawn_or_fail(
                    f"remote pipeline PID disappeared without exit_code on '{self.cfg.session}'",
                    respawn_count,
                )
                if not did_respawn:
                    if rc != 0 or not self.cfg.keep_on_complete:
                        self.stop_session()
                    return rc
                continue

            if self.cfg.once:
                return 0

            self._sleep(self.cfg.interval_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="colab_supervisor",
        description=(
            "Local anti-idle Jupyter keepalive, crash/completion auto-stop, time-budget VM rollover, "
            "and preemption auto-respawn supervisor for Google Colab CLI (`colab`)."
        ),
    )
    ap.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN, help="Path to `colab` CLI binary")
    ap.add_argument("--colab-config", default=None, help="Optional `--config` state file for `colab` CLI")
    ap.add_argument("--colab-auth", default=None, choices=["oauth2", "adc"], help="Optional `--auth` strategy")
    ap.add_argument("-s", "--session", default=DEFAULT_SESSION, help="Colab session name (default: loop64)")
    ap.add_argument("--gpu", default=DEFAULT_GPU, help="Colab GPU variant (default: G4; use CPU for none)")
    ap.add_argument("--repo", default=DEFAULT_HF_REPO, help="Hugging Face Hub checkpoint sync repo")
    ap.add_argument("--git-url", default=DEFAULT_GIT_URL, help="Git HTTPS clone URL for the VM")
    ap.add_argument("--git-ref", default=DEFAULT_GIT_REF, help="Git ref to checkout on VM (default: origin/main)")
    ap.add_argument("--remote-repo-dir", default=DEFAULT_REMOTE_REPO_DIR, help="Remote repo checkout path")
    ap.add_argument("--remote-run-root", default=DEFAULT_REMOTE_RUN_ROOT, help="Remote QLOOP_RUN_ROOT path")
    ap.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help="Comma-separated pipeline stages for `qloop sync pipeline`",
    )
    ap.add_argument(
        "--time-budget-hours",
        type=float,
        default=DEFAULT_TIME_BUDGET_HOURS,
        help="Per-session `--time-budget-hours` passed to `qloop sync pipeline`",
    )
    ap.add_argument("--debug", type=int, default=DEFAULT_DEBUG_LEVEL, help="`qloop` debug level (default: 1)")
    ap.add_argument(
        "--interval-sec",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help="Seconds between Jupyter kernel keepalive probes (default: 300)",
    )
    ap.add_argument(
        "--exec-timeout-sec",
        type=int,
        default=DEFAULT_EXEC_TIMEOUT_SEC,
        help="`colab exec --timeout` seconds for keepalive probes (default: 180)",
    )
    ap.add_argument(
        "--bootstrap-timeout-sec",
        type=int,
        default=DEFAULT_BOOTSTRAP_TIMEOUT_SEC,
        help="`colab exec --timeout` seconds for VM bootstrap/setup (default: 900)",
    )
    ap.add_argument(
        "--max-respawns",
        type=int,
        default=DEFAULT_MAX_RESPAWNS,
        help="Maximum automatic VM respawns on Colab preemption (default: 3)",
    )
    ap.add_argument(
        "--max-rollovers",
        type=int,
        default=DEFAULT_MAX_ROLLOVERS,
        help=(
            "Maximum automatic fresh-VM rollovers on clean exit_code=0 time-budget pauses "
            "(default: 10; 0 disables rollover)"
        ),
    )
    ap.add_argument(
        "--respawn-backoff-sec",
        type=float,
        default=DEFAULT_RESPAWN_BACKOFF_SEC,
        help="Initial backoff seconds before respawning/rolling over a session (default: 15)",
    )
    ap.add_argument(
        "--max-respawn-backoff-sec",
        type=float,
        default=DEFAULT_MAX_RESPAWN_BACKOFF_SEC,
        help="Cap in seconds for exponential respawn/503 capacity retry backoff (default: 300)",
    )
    ap.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="Local supervisor status log path")
    ap.add_argument("--tail-file", default=DEFAULT_TAIL_FILE, help="Local file written with remote stdout tail on exit")
    ap.add_argument(
        "--keep-on-complete",
        action="store_true",
        help="Keep the Colab VM running when pipeline exits 0 (crashes still stop the VM)",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run bootstrap + 1 probe and exit without stopping a live remote PID",
    )
    ap.add_argument(
        "--set",
        dest="extra_overrides",
        action="append",
        default=[],
        help="Optional `-s key=val` override forwarded to `qloop sync pipeline` (repeatable)",
    )
    return ap


def config_from_args(args: argparse.Namespace) -> SupervisorConfig:
    stages = tuple(s.strip() for s in str(args.stages).split(",") if s.strip())
    if not stages:
        raise ValueError("`--stages` must specify at least one pipeline stage")
    return SupervisorConfig(
        colab_bin=os.path.expanduser(str(args.colab_bin)),
        colab_config=os.path.expanduser(args.colab_config) if args.colab_config else None,
        colab_auth=args.colab_auth,
        session=str(args.session),
        gpu=str(args.gpu),
        repo=str(args.repo),
        git_url=normalize_git_https_url(str(args.git_url)),
        git_ref=str(args.git_ref),
        remote_repo_dir=str(args.remote_repo_dir),
        remote_run_root=str(args.remote_run_root),
        stages=stages,
        time_budget_hours=float(args.time_budget_hours),
        debug=int(args.debug),
        interval_sec=float(args.interval_sec),
        exec_timeout_sec=int(args.exec_timeout_sec),
        bootstrap_timeout_sec=int(args.bootstrap_timeout_sec),
        max_respawns=int(args.max_respawns),
        max_rollovers=int(args.max_rollovers),
        respawn_backoff_sec=float(args.respawn_backoff_sec),
        max_respawn_backoff_sec=float(args.max_respawn_backoff_sec),
        log_file=str(args.log_file),
        tail_file=str(args.tail_file),
        keep_on_complete=bool(args.keep_on_complete),
        once=bool(args.once),
        extra_overrides=tuple(args.extra_overrides or ()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = config_from_args(args)
    sup = ColabSupervisor(cfg)
    try:
        return sup.run()
    except KeyboardInterrupt:
        sup.log("[INFO] supervisor interrupted locally (remote session left untouched)")
        return 130
    except ColabSupervisorError as exc:
        sup.log(f"[FATAL] {exc}")
        return 1
    except Exception as exc:
        sup.log(f"[FATAL] {type(exc).__name__}: {sup.redact(str(exc))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
