"""OpenAI-compatible server round trips (toy model + scripted model) over a real socket."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from conftest import LOOPBACK_OK

from qloop.eval.external import openai_chat
from qloop.eval.generate import Generator
from qloop.eval.server import start_background
from qloop.eval.testing import ByteTokenizer, ScriptedLM, build_toy_generator

pytestmark = pytest.mark.skipif(not LOOPBACK_OK, reason="loopback sockets not permitted (sandbox)")

TOY_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


@pytest.fixture()
def toy_server():
    gen = build_toy_generator(loop_spec=TOY_SPEC)
    srv, th = start_background(gen, "127.0.0.1", 0, model_name="qloop-toy", debug=True, default_max_tokens=4)
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield base, gen
    srv.shutdown()
    srv.server_close()
    th.join(timeout=5)


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _post_raw(url, data: bytes):
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_models_and_health(toy_server):
    base, _ = toy_server
    m = _get(base + "/models")
    assert m["object"] == "list" and m["data"][0]["id"] == "qloop-toy"
    h = _get(base.replace("/v1", "") + "/health")
    assert h["status"] == "ok" and h["max_depth"] == 2


def test_chat_completion_round_trip_with_depth(toy_server):
    base, gen = toy_server
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    r1 = openai_chat(base, msgs, model="qloop-toy", max_tokens=5, temperature=0, loop_depth=1)
    assert r1["object"] == "chat.completion" and r1["loop_depth"] == 1
    ch = r1["choices"][0]
    assert ch["message"]["role"] == "assistant" and isinstance(ch["message"]["content"], str)
    assert ch["finish_reason"] in ("stop", "length")
    assert r1["usage"]["completion_tokens"] <= 5 and r1["usage"]["prompt_tokens"] > 0
    # same request in-process gives the same text (temperature 0)
    direct = gen.generate([[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]],
                          max_new_tokens=5, depth=1)[0][0]
    assert ch["message"]["content"] == direct.text
    assert gen.loop_controller.active_passes == 2  # restored after the request
    r2 = openai_chat(base, msgs, max_tokens=3, temperature=0.8, n=2, seed=1, logprobs=True)
    assert len(r2["choices"]) == 2 and r2["usage"]["completion_tokens"] <= 6
    lp = r2["choices"][0]["logprobs"]["content"]
    assert all(x["logprob"] <= 0 for x in lp)


def test_errors_are_json(toy_server):
    base, _ = toy_server
    code, body = _post_raw(base + "/chat/completions", b"{not json")
    assert code == 400 and "invalid JSON" in json.loads(body)["error"]["message"]
    code, body = _post_raw(base + "/chat/completions", json.dumps({"messages": []}).encode())
    assert code == 400 and json.loads(body)["error"]["type"] == "invalid_request_error"
    code, body = _post_raw(base + "/chat/completions",
                           json.dumps({"messages": [{"role": "user", "content": "x"}], "loop_depth": 5}).encode())
    err = json.loads(body)["error"]
    assert code == 400 and "trained depth" in err["message"] and "traceback" in err  # debug=True
    code, body = _post_raw(base + "/nope", b"{}")
    assert code == 404
    with pytest.raises(RuntimeError, match="HTTP 400"):
        openai_chat(base, [{"role": "user", "content": "x"}], temperature=9)


def test_scripted_server_completions_stop_and_stream():
    gen = Generator(ScriptedLM.from_text("FINAL ANSWER: 42\nmore"), ByteTokenizer())
    srv, th = start_background(gen, "127.0.0.1", 0, api_key="secret")
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        r = openai_chat(base, [{"role": "user", "content": "q"}], api_key="secret", stop=["\n"], max_tokens=50)
        assert r["choices"][0]["message"]["content"] == "FINAL ANSWER: 42"
        assert r["choices"][0]["finish_reason"] == "stop"
        with pytest.raises(RuntimeError, match="HTTP 401"):
            openai_chat(base, [{"role": "user", "content": "q"}], api_key="wrong")
        with pytest.raises(RuntimeError, match="loop controller"):
            openai_chat(base, [{"role": "user", "content": "q"}], api_key="secret", loop_depth=1)
        req = urllib.request.Request(base + "/completions", method="POST",
                                     data=json.dumps({"prompt": ["a", "b"], "max_tokens": 5}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer secret"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            c = json.loads(resp.read())
        assert [x["text"] for x in c["choices"]] == ["FINAL", "FINAL"] and c["object"] == "text_completion"
        req = urllib.request.Request(base + "/chat/completions", method="POST",
                                     data=json.dumps({"messages": [{"role": "user", "content": "q"}], "stream": True,
                                                      "max_tokens": 5}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer secret"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            events = [ln for ln in resp.read().decode().split("\n\n") if ln.startswith("data: ")]
        assert events[-1] == "data: [DONE]"
        first = json.loads(events[0][len("data: "):])
        assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["content"] == "FINAL"
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)
