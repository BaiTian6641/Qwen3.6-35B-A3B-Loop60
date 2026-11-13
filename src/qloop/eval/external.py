"""External harnesses for benchmarks that cannot be scored in-process on Colab.

Colab (RTX PRO 6000 runtime) has NO Docker daemon, so container benchmarks (Terminal-Bench 2.0,
SWE-bench Pro, AgentWorldBench, OpenThoughts-TB-dev) run through **Harbor** with a remote sandbox
provider (`--env daytona` default, or `modal`). The agent loop (e.g. terminus-2) runs locally
and talks to our looped model through the OpenAI-compatible `qloop.eval.server`; only the task
containers are remote. (Agents that are *installed inside* the container, e.g. claude-code,
cannot reach a localhost server - expose it via a tunnel or use a host-side agent.)

Provided here:
  * env checks that report precisely what is missing (binary, API keys, docker socket)
  * command builders (Harbor, tau-bench)
  * `run_logged`: subprocess with streaming logs to file + logger, timeout, process-group kill
  * result parsers that glob the job output directory and report what they found
  * `run_openai_benchmark`: score any registry benchmark against an OpenAI-compatible endpoint
    (GAIA / tau-style text tasks, or evaluating a vLLM-served checkpoint)

Harbor/tau-bench CLI flag names live in HARBOR_FLAGS / TAU_FLAGS so a CLI rename is a one-line
override; verify with `harbor run --help`.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from qloop.debug import require

log = logging.getLogger("qloop")

DOCKER_SOCK = "/var/run/docker.sock"
HARBOR_ENVS = ("daytona", "modal", "e2b", "runloop", "docker")
ENV_KEYS: dict[str, list[str]] = {
    "daytona": ["DAYTONA_API_KEY"],
    "modal": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    "e2b": ["E2B_API_KEY"],
    "runloop": ["RUNLOOP_API_KEY"],
    "docker": [],
}
PROVIDER_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}
HARBOR_FLAGS: dict[str, str] = {
    "dataset": "--dataset", "agent": "--agent", "model": "--model", "env": "--env",
    "n_concurrent": "--n-concurrent", "n_attempts": "--n-attempts", "jobs_dir": "--jobs-dir",
    "task_name": "--task-name", "agent_kwarg": "--agent-kwarg", "timeout_multiplier": "--timeout-multiplier",
}
TAU_FLAGS: dict[str, str] = {
    "agent_strategy": "--agent-strategy", "env": "--env", "model": "--model", "model_provider": "--model-provider",
    "user_model": "--user-model", "user_model_provider": "--user-model-provider", "user_strategy": "--user-strategy",
    "max_concurrency": "--max-concurrency", "num_trials": "--num-trials", "start_index": "--start-index",
    "end_index": "--end-index", "task_ids": "--task-ids", "log_dir": "--log-dir", "temperature": "--temperature",
}


# ============================================================================ env checks

@dataclass
class EnvCheck:
    ok: bool
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        s = "OK" if self.ok else "MISSING: " + "; ".join(self.missing)
        return s + ("" if not self.notes else " | notes: " + "; ".join(self.notes))

    def raise_if_failed(self, what: str) -> None:
        if not self.ok:
            raise RuntimeError(f"{what}: environment not ready -> " + "; ".join(self.missing)
                               + ("" if not self.notes else " (" + "; ".join(self.notes) + ")"))


def docker_available(sock: str = DOCKER_SOCK, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return os.path.exists(sock) or bool(env.get("DOCKER_HOST"))


def require_docker(sock: str = DOCKER_SOCK, environ: Mapping[str, str] | None = None) -> None:
    if not docker_available(sock, environ):
        raise RuntimeError(
            f"--env docker requested but no Docker daemon is reachable ({sock} does not exist and DOCKER_HOST is "
            "unset). Colab runtimes (incl. RTX PRO 6000 Blackwell) cannot run a Docker daemon. Use a remote sandbox "
            "instead: --env daytona (set DAYTONA_API_KEY) or --env modal (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or "
            "`modal token new`).")


def _provider(model: str) -> str:
    return model.split("/", 1)[0].lower() if "/" in model else "openai"


def check_harbor_env(env: str = "daytona", model: str | None = None, api_base: str | None = None,
                     binary: str = "harbor", environ: Mapping[str, str] | None = None,
                     which: Callable[[str], str | None] = shutil.which, docker_sock: str = DOCKER_SOCK) -> EnvCheck:
    environ = os.environ if environ is None else environ
    missing, notes = [], []
    if which(binary) is None:
        missing.append(f"`{binary}` binary not on PATH (pip install harbor  /  uv tool install harbor)")
    if env not in HARBOR_ENVS:
        missing.append(f"unknown --env {env!r}; choose from {HARBOR_ENVS}")
    elif env == "docker":
        if not docker_available(docker_sock, environ):
            missing.append(f"docker daemon ({docker_sock} absent; Colab has no Docker - use --env daytona or modal)")
    elif env == "modal":
        have_tokens = all(environ.get(k) for k in ENV_KEYS["modal"])
        if not have_tokens and not os.path.exists(os.path.expanduser("~/.modal.toml")):
            missing.append("MODAL_TOKEN_ID + MODAL_TOKEN_SECRET (or ~/.modal.toml from `modal token new`)")
    else:
        missing += [f"env var {k} (needed by --env {env})" for k in ENV_KEYS[env] if not environ.get(k)]
    if model:
        prov = _provider(model)
        key = PROVIDER_KEYS.get(prov)
        if api_base:
            notes.append(f"model {model} served at {api_base} (local server: API key not required)")
        elif key and not environ.get(key):
            missing.append(f"env var {key} (needed for model provider '{prov}')")
    return EnvCheck(not missing, missing, notes)


# ============================================================================ command builders

@dataclass
class HarborRun:
    dataset: str = "terminal-bench@2.0"
    agent: str = "terminus-2"
    model: str = "openai/qloop"
    env: str = "daytona"
    n_concurrent: int = 4
    n_attempts: int = 1
    jobs_dir: str | None = None
    task_names: list[str] = field(default_factory=list)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_multiplier: float | None = None
    api_base: str | None = None      # e.g. http://127.0.0.1:8000/v1 (qloop.eval.server)
    api_key: str | None = None
    extra_args: list[str] = field(default_factory=list)
    binary: str = "harbor"


def build_harbor_command(run: HarborRun, docker_sock: str = DOCKER_SOCK,
                         environ: Mapping[str, str] | None = None) -> list[str]:
    if run.env == "docker":
        require_docker(docker_sock, environ)
    require(run.env in HARBOR_ENVS, f"unknown harbor env {run.env!r}; choose from {HARBOR_ENVS}")
    F = HARBOR_FLAGS
    cmd = [run.binary, "run", F["dataset"], run.dataset, F["agent"], run.agent, F["model"], run.model,
           F["env"], run.env, F["n_concurrent"], str(int(run.n_concurrent))]
    if run.n_attempts and run.n_attempts != 1:
        cmd += [F["n_attempts"], str(int(run.n_attempts))]
    if run.jobs_dir:
        cmd += [F["jobs_dir"], run.jobs_dir]
    for t in run.task_names:
        cmd += [F["task_name"], t]
    for k, v in run.agent_kwargs.items():
        cmd += [F["agent_kwarg"], f"{k}={v}"]
    if run.timeout_multiplier:
        cmd += [F["timeout_multiplier"], str(run.timeout_multiplier)]
    return cmd + list(run.extra_args)


def harbor_env_vars(run: HarborRun) -> dict[str, str]:
    """Env overrides that point LiteLLM-based agents (terminus-2) at an OpenAI-compatible server."""
    env: dict[str, str] = {}
    if run.api_base:
        env["OPENAI_API_BASE"] = run.api_base
        env["OPENAI_BASE_URL"] = run.api_base
        env["OPENAI_API_KEY"] = run.api_key or os.environ.get("OPENAI_API_KEY") or "sk-qloop-local"
    return env


@dataclass
class TauBenchRun:
    env: str = "retail"
    model: str = "qloop"
    model_provider: str = "openai"
    user_model: str = "gpt-4.1"
    user_model_provider: str = "openai"
    agent_strategy: str = "tool-calling"
    user_strategy: str = "llm"
    max_concurrency: int = 4
    num_trials: int = 1
    start_index: int = 0
    end_index: int = -1
    task_ids: list[int] = field(default_factory=list)
    temperature: float = 0.0
    log_dir: str = "results"
    repo_dir: str = "tau-bench"
    api_base: str | None = None      # agent model endpoint (qloop.eval.server)
    user_api_base: str | None = None  # set if the user simulator is ALSO served locally


def build_tau_command(run: TauBenchRun, python: str = sys.executable) -> list[str]:
    F = TAU_FLAGS
    cmd = [python, "run.py", F["agent_strategy"], run.agent_strategy, F["env"], run.env, F["model"], run.model,
           F["model_provider"], run.model_provider, F["user_model"], run.user_model, F["user_model_provider"],
           run.user_model_provider, F["user_strategy"], run.user_strategy, F["max_concurrency"],
           str(run.max_concurrency), F["num_trials"], str(run.num_trials), F["temperature"], str(run.temperature),
           F["log_dir"], run.log_dir]
    if run.task_ids:
        cmd += [F["task_ids"], *[str(t) for t in run.task_ids]]
    else:
        cmd += [F["start_index"], str(run.start_index), F["end_index"], str(run.end_index)]
    return cmd


def check_tau_env(run: TauBenchRun, environ: Mapping[str, str] | None = None) -> EnvCheck:
    environ = os.environ if environ is None else environ
    missing, notes = [], []
    if not os.path.exists(os.path.join(run.repo_dir, "run.py")):
        missing.append(f"tau-bench checkout with run.py at {run.repo_dir!r} "
                       "(git clone https://github.com/sierra-research/tau-bench && pip install -e tau-bench)")
    for role, prov, base in (("agent", run.model_provider, run.api_base),
                             ("user simulator", run.user_model_provider, run.user_api_base)):
        key = PROVIDER_KEYS.get(prov.lower())
        if base:
            notes.append(f"{role} served at {base}")
        elif key and not environ.get(key):
            missing.append(f"env var {key} (needed for the {role} model provider '{prov}')")
    if run.api_base and run.user_api_base is None and run.user_model_provider == run.model_provider:
        notes.append("agent and user simulator share provider env vars; OPENAI_BASE_URL will point BOTH at the "
                     "local server unless user_api_base / a different user provider is set")
    return EnvCheck(not missing, missing, notes)


def tau_env_vars(run: TauBenchRun) -> dict[str, str]:
    env: dict[str, str] = {}
    base = run.api_base or run.user_api_base
    if base and run.model_provider == "openai":
        env["OPENAI_API_BASE"] = base
        env["OPENAI_BASE_URL"] = base
        env.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-qloop-local")
    return env


# ============================================================================ subprocess runner

def run_logged(cmd: Sequence[str], log_path: str, env: Mapping[str, str] | None = None, cwd: str | None = None,
               timeout: float | None = None, echo: bool = True, prefix: str = "") -> int:
    """Run `cmd`, streaming merged stdout/stderr line-by-line to `log_path` (and the logger).

    Returns the exit code (-9 style negative on kill; 124 on timeout, like coreutils `timeout`).
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    full_env = dict(os.environ)
    full_env.update(env or {})
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    log.info("exec: %s (cwd=%s, log=%s)", " ".join(cmd), cwd or os.getcwd(), log_path)
    try:
        proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=full_env,
                                text=True, bufsize=1, start_new_session=True, errors="replace")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"cannot execute {cmd[0]!r}: not found on PATH ({e})") from e
    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(5)
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    timer = threading.Timer(timeout, _kill) if timeout else None
    if timer:
        timer.daemon = True
        timer.start()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n")
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if echo:
                    log.info("%s%s", prefix, line.rstrip("\n"))
        rc = proc.wait()
    finally:
        if timer:
            timer.cancel()
    if timed_out.is_set():
        log.error("exec timed out after %ss: %s", timeout, cmd[0])
        return 124
    return rc


# ============================================================================ result parsing

def _find_reward(d: Any) -> float | None:
    if not isinstance(d, dict):
        return None
    for k in ("reward", "score"):
        v = d.get(k)
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
    for k in ("is_resolved", "resolved", "passed", "success"):
        if isinstance(d.get(k), bool):
            return float(d[k])
    rw = d.get("rewards")
    if isinstance(rw, dict):
        if isinstance(rw.get("reward"), (int, float)):
            return float(rw["reward"])
        nums = [float(v) for v in rw.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums:
            return sum(nums) / len(nums)
    for k in ("verifier_result", "verifier", "result", "evaluation"):
        r = _find_reward(d.get(k))
        if r is not None:
            return r
    return None


def parse_harbor_results(job_dir: str) -> dict[str, Any]:
    """Defensively summarise a Harbor job directory (any layout): glob JSONs, find per-trial rewards."""
    summary: dict[str, Any] = {"job_dir": job_dir, "exists": os.path.isdir(job_dir)}
    if not summary["exists"]:
        summary["error"] = f"job dir {job_dir!r} does not exist"
        return summary
    files = sorted(glob.glob(os.path.join(job_dir, "**", "*.json"), recursive=True))
    summary["n_json_files"] = len(files)
    summary["files_found"] = [os.path.relpath(f, job_dir) for f in files[:50]]
    per_task: dict[str, list[float]] = {}
    errors: list[dict[str, str]] = []
    unparsed: list[str] = []
    job_level: dict[str, Any] | None = None
    for fp in files:
        rel = os.path.relpath(fp, job_dir)
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            unparsed.append(f"{rel}: {type(e).__name__}")
            continue
        if os.path.dirname(rel) == "" and os.path.basename(fp) in ("result.json", "results.json"):
            job_level = data if isinstance(data, dict) else {"value": data}
            continue
        entries = data if isinstance(data, list) else [data]
        for d in entries:
            if not isinstance(d, dict):
                continue
            exc = d.get("exception_info") or d.get("error") or d.get("exception")
            r = _find_reward(d)
            task = str(d.get("task_name") or d.get("task_id") or d.get("instance_id")
                       or os.path.basename(os.path.dirname(fp)) or rel)
            if exc:
                errors.append({"file": rel, "task": task, "error": str(exc)[:500]})
            if r is not None:
                per_task.setdefault(task, []).append(r)
            elif exc:
                per_task.setdefault(task, []).append(0.0)  # failed trial counts as 0, like harbor
    rewards = [x for v in per_task.values() for x in v]
    summary.update({
        "n_trials": len(rewards), "n_tasks": len(per_task),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "n_solved": sum(x >= 1.0 for x in rewards),
        "per_task": per_task, "errors": errors[:100], "n_errors": len(errors), "unparsed": unparsed,
    })
    if job_level is not None:
        summary["job_result_keys"] = sorted(job_level)[:50]
        summary["job_result"] = job_level
    if not rewards:
        summary["warning"] = ("no per-trial rewards found; inspect files_found (layout may have changed) or the "
                              "harbor log")
    return summary


def newest_subdir(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return max(subs, key=os.path.getmtime) if subs else None


def run_harbor(run: HarborRun, log_dir: str, timeout: float | None = None, dry_run: bool = False) -> dict[str, Any]:
    check = check_harbor_env(run.env, run.model, run.api_base, run.binary)
    cmd = build_harbor_command(run)
    os.makedirs(log_dir, exist_ok=True)
    if dry_run:
        return {"cmd": cmd, "env_check": str(check), "ok": check.ok}
    check.raise_if_failed(f"harbor run ({run.dataset})")
    jobs_root = run.jobs_dir or os.path.join(os.getcwd(), "jobs")
    before = set(os.listdir(jobs_root)) if os.path.isdir(jobs_root) else set()
    t0 = time.time()
    rc = run_logged(cmd, os.path.join(log_dir, "harbor.log"), env=harbor_env_vars(run), timeout=timeout,
                    prefix="[harbor] ")
    after = set(os.listdir(jobs_root)) if os.path.isdir(jobs_root) else set()
    new = sorted(after - before)
    job_dir = os.path.join(jobs_root, new[-1]) if new else (newest_subdir(jobs_root) or jobs_root)
    summary = parse_harbor_results(job_dir)
    summary.update({"cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 1), "dataset": run.dataset,
                    "agent": run.agent, "model": run.model, "env": run.env})
    with open(os.path.join(log_dir, "harbor_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    if rc != 0:
        log.error("harbor exited with code %s; see %s", rc, os.path.join(log_dir, "harbor.log"))
    return summary


def parse_tau_results(path: str, ks: Iterable[int] = (1, 2, 4)) -> dict[str, Any]:
    from .scoring import pass_hat_k
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    per_task: dict[str, list[float]] = {}
    unparsed = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            unparsed.append(f"{fp}: {type(e).__name__}")
            continue
        for d in data if isinstance(data, list) else [data]:
            if isinstance(d, dict) and "task_id" in d and _find_reward(d) is not None:
                per_task.setdefault(str(d["task_id"]), []).append(_find_reward(d) or 0.0)
    rewards = [x for v in per_task.values() for x in v]
    out: dict[str, Any] = {"files": files[:50], "n_tasks": len(per_task), "n_trials": len(rewards),
                           "avg_reward": (sum(rewards) / len(rewards)) if rewards else None, "unparsed": unparsed}
    if per_task:
        n_min = min(len(v) for v in per_task.values())
        for k in ks:
            if k <= n_min:
                out[f"pass^{k}"] = sum(pass_hat_k(len(v), sum(x >= 1.0 for x in v), k)
                                       for v in per_task.values()) / len(per_task)
    return out


def run_tau_bench(run: TauBenchRun, log_dir: str, timeout: float | None = None,
                  dry_run: bool = False) -> dict[str, Any]:
    check = check_tau_env(run)
    cmd = build_tau_command(run)
    if dry_run:
        return {"cmd": cmd, "env_check": str(check), "ok": check.ok}
    check.raise_if_failed("tau-bench")
    rc = run_logged(cmd, os.path.join(log_dir, "tau_bench.log"), env=tau_env_vars(run), cwd=run.repo_dir,
                    timeout=timeout, prefix="[tau] ")
    res_dir = run.log_dir if os.path.isabs(run.log_dir) else os.path.join(run.repo_dir, run.log_dir)
    summary = parse_tau_results(res_dir)
    summary.update({"cmd": cmd, "returncode": rc})
    with open(os.path.join(log_dir, "tau_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ============================================================================ OpenAI-compatible client

def openai_chat(base_url: str, messages: list[dict[str, Any]], model: str = "qloop", api_key: str | None = None,
                timeout: float = 600.0, **params: Any) -> dict[str, Any]:
    """POST {base_url}/chat/completions (stdlib only). Raises RuntimeError with the server's error body."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, **params}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url}: {e.reason}") from e


def run_openai_benchmark(spec: Any, items: list[dict[str, Any]], base_url: str, out_dir: str, model: str = "qloop",
                         api_key: str | None = None, max_tokens: int | None = None, temperature: float = 0.0,
                         extra_params: dict[str, Any] | None = None, timeout: float = 1800.0) -> dict[str, Any]:
    """Score a registry benchmark through any OpenAI-compatible endpoint (resumable JSONL)."""
    from .registry import get_spec
    from .scoring import get_scorer
    spec = get_spec(spec) if isinstance(spec, str) else spec
    require(spec.scorer is not None, f"{spec.name} has no scorer")
    scorer = get_scorer(spec.scorer)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{spec.name}_openai.jsonl")
    done: dict[str, dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except ValueError:
                    continue
    with open(path, "a", encoding="utf-8") as fout:
        for it in items:
            if str(it["id"]) in done:
                continue
            try:
                resp = openai_chat(base_url, spec.build_prompt(it), model=model, api_key=api_key, timeout=timeout,
                                   max_tokens=int(max_tokens or spec.max_new_tokens), temperature=temperature,
                                   **(extra_params or {}))
                text = resp["choices"][0]["message"].get("content") or ""
                ev = scorer.evaluate(text, it)
                rec = {"id": str(it["id"]), "text": text, **ev, "usage": resp.get("usage")}
            except RuntimeError as e:
                rec = {"id": str(it["id"]), "text": "", "score": 0.0, "pred": None, "error": str(e)}
            fout.write(json.dumps(rec, default=str) + "\n")
            fout.flush()
            done[rec["id"]] = rec
    recs = [done[str(it["id"])] for it in items if str(it["id"]) in done]
    summary = {"benchmark": spec.name, "endpoint": base_url, "model": model, "n_items": len(recs),
               "mean_score": (sum(r["score"] for r in recs) / len(recs)) if recs else None,
               "n_errors": sum(1 for r in recs if r.get("error"))}
    with open(os.path.join(out_dir, f"{spec.name}_openai_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
