"""Minimal OpenAI-compatible HTTP server backed by `Generator` (stdlib only, no FastAPI).

Endpoints
    GET  /health, /v1/models
    POST /v1/chat/completions   messages, max_tokens|max_completion_tokens, temperature, top_p, top_k, n, stop,
                                seed, logprobs, tools (passed to the chat template), stream (single-chunk SSE),
                                and the custom `loop_depth` (looped model test-time depth)
    POST /v1/completions        prompt (str | list[str]), same sampling fields

GPU access is serialised with a lock (ThreadingHTTPServer accepts concurrent connections, e.g.
Harbor's n-concurrent agents, but generation runs one request at a time). Errors are JSON
`{"error": {"message", "type", "code"[, "traceback"]}}`; tracebacks only when debug=True.

    from qloop.eval.server import serve, start_background
    srv, th = start_background(generator, port=8000)     # notebook-friendly
    serve(generator, host="0.0.0.0", port=8000)           # blocking
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger("qloop")


class BadRequest(ValueError):
    pass


class _State:
    def __init__(self, generator: Any, model_name: str, debug: bool, default_max_tokens: int,
                 max_tokens_cap: int | None, api_key: str | None, default_temperature: float):
        self.generator = generator
        self.model_name = model_name
        self.debug = debug
        self.default_max_tokens = default_max_tokens
        self.max_tokens_cap = max_tokens_cap
        self.api_key = api_key
        self.default_temperature = default_temperature
        self.lock = threading.Lock()
        self.created = int(time.time())
        self.n_requests = 0


def _num(body: dict[str, Any], key: str, default: Any, typ: type, lo: float | None = None,
         hi: float | None = None) -> Any:
    v = body.get(key, default)
    if v is None:
        v = default
    try:
        v = typ(v)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be {typ.__name__}, got {body.get(key)!r}") from None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise BadRequest(f"'{key}'={v} out of range [{lo}, {hi}]")
    return v


def _sampling(state: _State, body: dict[str, Any]) -> dict[str, Any]:
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    max_tokens = _num({"max_tokens": mt}, "max_tokens", state.default_max_tokens, int, 1)
    if state.max_tokens_cap is not None:
        max_tokens = min(max_tokens, state.max_tokens_cap)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise BadRequest("'stop' must be a string or a list of strings")
    depth = body.get("loop_depth")
    if depth is not None:
        depth = _num(body, "loop_depth", None, int, 1)
        gen = state.generator
        if getattr(gen, "loop_controller", None) is None:
            raise BadRequest("'loop_depth' given but the served model has no loop controller")
        max_depth = getattr(gen, "max_depth", None)
        if max_depth is not None and depth > max_depth:
            raise BadRequest(f"'loop_depth'={depth} exceeds the model's trained depth {max_depth}")
    seed = body.get("seed")
    return {
        "max_new_tokens": max_tokens,
        "temperature": _num(body, "temperature", state.default_temperature, float, 0.0, 2.0),
        "top_p": _num(body, "top_p", 1.0, float, 1e-6, 1.0),
        "top_k": _num(body, "top_k", 0, int, 0),
        "n": _num(body, "n", 1, int, 1, 64),
        "stop": stop,
        "seed": int(seed) if seed is not None else int(time.time() * 1000) % (2**31),
        "depth": depth,
        "return_logprobs": bool(body.get("logprobs", False)),
    }


def _usage(outs: list[Any]) -> dict[str, int]:
    p = outs[0].prompt_tokens if outs else 0
    c = sum(o.completion_tokens for o in outs)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _logprobs_obj(o: Any, gen: Any) -> dict[str, Any] | None:
    if o.logprobs is None:
        return None
    return {"content": [{"token": gen.decode([t]), "logprob": lp, "token_id": t}
                        for t, lp in zip(o.token_ids, o.logprobs)]}


def make_handler(state: _State) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "qloop-openai/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("http %s " + fmt, self.address_string(), *args)

        # ------------------------------------------------------------ io helpers
        def _send(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str, etype: str, exc: BaseException | None = None) -> None:
            err: dict[str, Any] = {"message": message, "type": etype, "code": code}
            if state.debug and exc is not None:
                err["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._send(code, {"error": err})

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except ValueError as e:
                raise BadRequest(f"invalid JSON body: {e}") from None
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            return body

        def _authorized(self) -> bool:
            if not state.api_key:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {state.api_key}"

        # ------------------------------------------------------------ routes
        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                gen = state.generator
                self._send(200, {"status": "ok", "model": state.model_name, "requests": state.n_requests,
                                 "max_depth": getattr(gen, "max_depth", None)})
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": state.model_name, "object": "model",
                                                             "created": state.created, "owned_by": "qloop"}]})
            else:
                self._error(404, f"no route GET {path}", "not_found")

        def do_POST(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            t0 = time.time()
            if not self._authorized():
                self._error(401, "invalid or missing API key", "authentication_error")
                return
            try:
                body = self._body()
                if path == "/v1/chat/completions":
                    payload, stream = self._chat(body)
                elif path == "/v1/completions":
                    payload, stream = self._completions(body), False
                else:
                    self._error(404, f"no route POST {path}", "not_found")
                    return
            except BadRequest as e:
                log.warning("http 400 %s: %s", path, e)
                self._error(400, str(e), "invalid_request_error", e)
                return
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                log.warning("http 400 %s: %s: %s", path, type(e).__name__, e)
                self._error(400, f"{type(e).__name__}: {e}", "invalid_request_error", e)
                return
            except Exception as e:
                log.error("http 500 %s: %s: %s", path, type(e).__name__, e)
                self._error(500, f"{type(e).__name__}: {e}", "server_error", e)
                return
            usage = payload.get("usage", {})
            log.info("http POST %s 200 %.2fs prompt=%s completion=%s", path, time.time() - t0,
                     usage.get("prompt_tokens"), usage.get("completion_tokens"))
            if stream:
                self._send_stream(payload)
            else:
                self._send(200, payload)

        def _generate(self, prompt: Any, samp: dict[str, Any], tmpl_kw: dict[str, Any] | None) -> list[Any]:
            with state.lock:
                state.n_requests += 1
                return state.generator.generate([prompt], chat_template_kwargs=tmpl_kw, **samp)[0]

        def _chat(self, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise BadRequest("'messages' must be a non-empty list")
            for m in msgs:
                if not isinstance(m, dict) or "role" not in m:
                    raise BadRequest(f"each message needs a 'role': {str(m)[:200]}")
            samp = _sampling(state, body)
            tmpl_kw = {"tools": body["tools"]} if body.get("tools") else None
            outs = self._generate(msgs, samp, tmpl_kw)
            gen = state.generator
            choices = []
            for i, o in enumerate(outs):
                ch: dict[str, Any] = {"index": i, "message": {"role": "assistant", "content": o.text},
                                      "finish_reason": o.finish_reason, "logprobs": _logprobs_obj(o, gen)}
                choices.append(ch)
            payload = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
                       "created": int(time.time()), "model": body.get("model") or state.model_name,
                       "choices": choices, "usage": _usage(outs), "loop_depth": samp["depth"]}
            return payload, bool(body.get("stream", False))

        def _completions(self, body: dict[str, Any]) -> dict[str, Any]:
            prompt = body.get("prompt")
            prompts = [prompt] if isinstance(prompt, str) else prompt
            if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
                raise BadRequest("'prompt' must be a string or a non-empty list of strings")
            samp = _sampling(state, body)
            choices, all_outs = [], []
            for p in prompts:
                outs = self._generate(p, samp, None)
                all_outs += outs
                for o in outs:
                    choices.append({"index": len(choices), "text": o.text, "finish_reason": o.finish_reason,
                                    "logprobs": _logprobs_obj(o, state.generator)})
            usage = {"prompt_tokens": sum(o.prompt_tokens for o in all_outs) // max(1, samp["n"]),
                     "completion_tokens": sum(o.completion_tokens for o in all_outs)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return {"id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion", "created": int(time.time()),
                    "model": body.get("model") or state.model_name, "choices": choices, "usage": usage}

        def _send_stream(self, payload: dict[str, Any]) -> None:
            """Non-incremental SSE (whole completion in one delta) for clients that insist on stream=true."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            base = {"id": payload["id"], "object": "chat.completion.chunk", "created": payload["created"],
                    "model": payload["model"]}
            for ch in payload["choices"]:
                delta = {"role": "assistant", "content": ch["message"]["content"]}
                for d, fr in ((delta, None), ({}, ch["finish_reason"])):
                    chunk = dict(base, choices=[{"index": ch["index"], "delta": d, "finish_reason": fr}])
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def build_server(generator: Any, host: str = "127.0.0.1", port: int = 8000, *, model_name: str = "qloop",
                 debug: bool = False, default_max_tokens: int = 1024, max_tokens_cap: int | None = None,
                 api_key: str | None = None, default_temperature: float = 1.0) -> ThreadingHTTPServer:
    state = _State(generator, model_name, debug, default_max_tokens, max_tokens_cap, api_key, default_temperature)
    srv = ThreadingHTTPServer((host, port), make_handler(state))
    srv.daemon_threads = True
    srv.qloop_state = state  # type: ignore[attr-defined]
    return srv


def start_background(generator: Any, host: str = "127.0.0.1", port: int = 8000,
                     **kwargs: Any) -> tuple[ThreadingHTTPServer, threading.Thread]:
    srv = build_server(generator, host, port, **kwargs)
    th = threading.Thread(target=srv.serve_forever, name="qloop-openai-server", daemon=True)
    th.start()
    log.info("OpenAI-compatible server on http://%s:%d/v1 (model=%s)", srv.server_address[0], srv.server_address[1],
             srv.qloop_state.model_name)  # type: ignore[attr-defined]
    return srv, th


def serve(generator: Any, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
    srv = build_server(generator, host, port, **kwargs)
    log.info("serving OpenAI-compatible API on http://%s:%d/v1 (Ctrl-C to stop)", host, srv.server_address[1])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("server interrupted")
    finally:
        srv.server_close()
