"""External harness integration: command building, env checks, subprocess logging, result parsing."""
from __future__ import annotations

import json
import os
import sys

import pytest

from qloop.eval import external as X


def _which_none(_name):
    return None


def _which_all(name):
    return f"/usr/bin/{name}"


def test_harbor_command_defaults_to_daytona():
    run = X.HarborRun(model="openai/qloop", api_base="http://127.0.0.1:8000/v1", task_names=["hello-world"],
                      agent_kwargs={"max_turns": 50}, n_attempts=2, jobs_dir="/tmp/jobs")
    cmd = X.build_harbor_command(run)
    assert cmd[:2] == ["harbor", "run"]
    assert cmd[cmd.index("--dataset") + 1] == "terminal-bench@2.0"
    assert cmd[cmd.index("--env") + 1] == "daytona"
    assert cmd[cmd.index("--task-name") + 1] == "hello-world"
    assert "max_turns=50" in cmd and cmd[cmd.index("--n-attempts") + 1] == "2"
    env = X.harbor_env_vars(run)
    assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert env["OPENAI_API_KEY"]


def test_harbor_refuses_docker_without_socket(tmp_path):
    sock = str(tmp_path / "docker.sock")
    with pytest.raises(RuntimeError, match="no Docker daemon"):
        X.build_harbor_command(X.HarborRun(env="docker"), docker_sock=sock, environ={})
    open(sock, "w").close()
    cmd = X.build_harbor_command(X.HarborRun(env="docker"), docker_sock=sock, environ={})
    assert cmd[cmd.index("--env") + 1] == "docker"
    with pytest.raises(AssertionError, match="unknown harbor env"):
        X.build_harbor_command(X.HarborRun(env="kubernetes"))


def test_check_harbor_env_reports_exactly_what_is_missing(tmp_path):
    c = X.check_harbor_env("daytona", "anthropic/claude-x", environ={}, which=_which_none)
    assert not c.ok
    assert any("harbor" in m and "PATH" in m for m in c.missing)
    assert any("DAYTONA_API_KEY" in m for m in c.missing)
    assert any("ANTHROPIC_API_KEY" in m for m in c.missing)
    with pytest.raises(RuntimeError, match="DAYTONA_API_KEY"):
        c.raise_if_failed("harbor")
    ok = X.check_harbor_env("daytona", "openai/qloop", api_base="http://localhost:8000/v1",
                            environ={"DAYTONA_API_KEY": "k"}, which=_which_all)
    assert ok.ok and ok.missing == [] and "local server" in ok.notes[0]
    d = X.check_harbor_env("docker", None, environ={}, which=_which_all, docker_sock=str(tmp_path / "nope"))
    assert not d.ok and "Colab has no Docker" in d.missing[0]
    m = X.check_harbor_env("modal", None, environ={"MODAL_TOKEN_ID": "a"}, which=_which_all)
    if not os.path.exists(os.path.expanduser("~/.modal.toml")):
        assert not m.ok and "MODAL_TOKEN_SECRET" in m.missing[0]
    assert not X.check_harbor_env("nowhere", None, environ={}, which=_which_all).ok


def test_tau_command_and_env(tmp_path):
    run = X.TauBenchRun(env="airline", model="qloop", api_base="http://127.0.0.1:8000/v1", repo_dir=str(tmp_path),
                        task_ids=[1, 2], num_trials=4)
    cmd = X.build_tau_command(run, python="python")
    assert cmd[:2] == ["python", "run.py"]
    assert cmd[cmd.index("--env") + 1] == "airline" and cmd[cmd.index("--num-trials") + 1] == "4"
    assert cmd[cmd.index("--task-ids") + 1:cmd.index("--task-ids") + 3] == ["1", "2"]
    c = X.check_tau_env(run, environ={})
    assert not c.ok and any("run.py" in m for m in c.missing) and any("OPENAI_API_KEY" in m for m in c.missing)
    (tmp_path / "run.py").write_text("print('ok')")
    c = X.check_tau_env(X.TauBenchRun(repo_dir=str(tmp_path), user_model_provider="anthropic"),
                        environ={"OPENAI_API_KEY": "x"})
    assert not c.ok and c.missing == ["env var ANTHROPIC_API_KEY (needed for the user simulator model provider "
                                      "'anthropic')"]
    assert X.run_tau_bench(run, str(tmp_path / "logs"), dry_run=True)["cmd"] == X.build_tau_command(run)


def test_run_logged_streams_and_times_out(tmp_path):
    log_path = str(tmp_path / "logs" / "x.log")
    rc = X.run_logged([sys.executable, "-c", "import sys; print('hello'); print('err', file=sys.stderr); sys.exit(3)"],
                      log_path, env={"FOO": "1"})
    assert rc == 3
    with open(log_path) as f:
        text = f.read()
    assert "hello" in text and "err" in text and text.startswith("$ ")
    rc = X.run_logged([sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path / "t.log"), timeout=1)
    assert rc == 124
    with pytest.raises(FileNotFoundError, match="not found"):
        X.run_logged(["definitely-not-a-binary-qloop"], str(tmp_path / "n.log"))


def test_parse_harbor_results(tmp_path):
    job = tmp_path / "2026-01-01__00-00-00"
    (job / "task-a__1").mkdir(parents=True)
    (job / "task-b__1").mkdir()
    (job / "task-c__1").mkdir()
    (job / "result.json").write_text(json.dumps({"stats": {"n_trials": 3}}))
    (job / "task-a__1" / "result.json").write_text(json.dumps(
        {"task_name": "task-a", "verifier_result": {"rewards": {"reward": 1.0}}}))
    (job / "task-b__1" / "result.json").write_text(json.dumps({"task_name": "task-b", "reward": 0.0}))
    (job / "task-c__1" / "result.json").write_text(json.dumps(
        {"task_name": "task-c", "exception_info": {"exception_type": "AgentTimeoutError"}}))
    (job / "task-c__1" / "broken.json").write_text("{oops")
    s = X.parse_harbor_results(str(job))
    assert s["n_trials"] == 3 and s["n_tasks"] == 3 and s["mean_reward"] == pytest.approx(1 / 3)
    assert s["n_solved"] == 1 and s["n_errors"] == 1 and "AgentTimeoutError" in s["errors"][0]["error"]
    assert s["job_result"] == {"stats": {"n_trials": 3}} and len(s["unparsed"]) == 1
    assert "task-a__1/result.json" in s["files_found"]
    empty = X.parse_harbor_results(str(tmp_path / "missing"))
    assert not empty["exists"] and "does not exist" in empty["error"]
    (tmp_path / "e").mkdir()
    assert "warning" in X.parse_harbor_results(str(tmp_path / "e"))


def test_parse_tau_results(tmp_path):
    data = [{"task_id": 0, "reward": 1.0}, {"task_id": 0, "reward": 0.0},
            {"task_id": 1, "reward": 1.0}, {"task_id": 1, "reward": 1.0}]
    (tmp_path / "r.json").write_text(json.dumps(data))
    s = X.parse_tau_results(str(tmp_path))
    assert s["n_tasks"] == 2 and s["avg_reward"] == 0.75
    assert s["pass^1"] == pytest.approx(0.75) and s["pass^2"] == pytest.approx(0.5)


@pytest.mark.skipif(not __import__("conftest").LOOPBACK_OK, reason="loopback sockets not permitted (sandbox)")
def test_run_openai_benchmark_against_local_server(tmp_path):
    from qloop.eval.generate import Generator
    from qloop.eval.server import start_background
    from qloop.eval.testing import ByteTokenizer, ScriptedLM

    gen = Generator(ScriptedLM.from_text("thinking\nFINAL ANSWER: Paris"), ByteTokenizer())
    srv, th = start_background(gen, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        items = [{"id": "g1", "question": "Capital of France?", "answer": "paris"},
                 {"id": "g2", "question": "Capital of Italy?", "answer": "Rome"}]
        s = X.run_openai_benchmark("gaia", items, base, str(tmp_path), max_tokens=64)
        assert s["n_items"] == 2 and s["mean_score"] == 0.5 and s["n_errors"] == 0
        s2 = X.run_openai_benchmark("gaia", items, "http://127.0.0.1:1/v1", str(tmp_path))  # resumed: no calls
        assert s2["mean_score"] == 0.5
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)
    with pytest.raises(RuntimeError, match="cannot reach"):
        X.openai_chat("http://127.0.0.1:1/v1", [{"role": "user", "content": "x"}], timeout=2)
