"""Row adapters: heterogeneous HF rows -> `Example` (normalised conversation or plain text).

Conversation = list of {"role": system|user|assistant|tool, "content": str, "reasoning": str|None,
                        optional "tool_calls": [...], "name": str}

Adapters (DatasetSpec.format):
  messages     OpenAI style list (role/content [+reasoning_content|reasoning|thinking, tool_calls])
  sharegpt     list of {from, value} (human/gpt/system/tool/function_call)   - same code path as messages
  prompt_response  prompt/problem/question/query/instruction + response/solution/answer/output
               (+ reasoning/thinking/...) columns
  nemotron     Llama-Nemotron: input(messages) + output (+system_prompt)
  chatml_text  one pre-rendered ChatML string ("<|im_start|>role\\n...<|im_end|>")
  text         plain text (pretrain)
  arc_agi      ARC task json -> one example per test pair
  auto         detect any of the above from the column names
Reasoning embedded in content as <think>..</think> or <thought>..</thought> is moved to `reasoning`.

Two failure classes, both counted per dataset by `DropCounter`:
  RowFormatError  - the row does not match the expected schema (says which dataset/row/field)
  RowDrop         - the row parsed fine but fails validation (empty/encrypted reasoning, ...)
"""
from __future__ import annotations

import collections
import gzip
import io
import json
import logging
import re
import tarfile
from dataclasses import dataclass, field
from typing import Any

from qloop.debug import describe

from .registry import DatasetSpec

log = logging.getLogger("qloop")

Conversation = list  # list[dict[str, Any]]  (alias kept simple for py3.9 runtime use)

PROMPT_KEYS = ("prompt", "problem", "question", "query", "instruction", "input", "user")
RESPONSE_KEYS = ("response", "solution", "answer", "output", "completion", "assistant", "chosen")
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking", "thought", "think", "cot", "rationale",
                  "reasoning_trace")
SYSTEM_KEYS = ("system", "system_prompt")
MESSAGE_LIST_KEYS = ("messages", "conversations", "conversation", "dialog", "turns")
TEXT_KEYS = ("text", "content", "raw_content", "document")

ROLE_MAP = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant", "bot": "assistant", "chatgpt": "assistant",
    "system": "system", "developer": "system",
    "tool": "tool", "function": "tool", "observation": "tool", "ipython": "tool", "tool_response": "tool",
    "function_call": "function_call",
}

_LEAD_THINK_RE = re.compile(r"\s*<(think|thought|thinking)>(.*?)</\1>", re.DOTALL)
_LEAD_OPEN_RE = re.compile(r"\s*<(think|thought|thinking)>")
_THINK_OPEN_RE = re.compile(r"<(think|thought|thinking)>")
_CLOSE_ONLY_RE = re.compile(r"</(think|thought|thinking)>")
_CHATML_RE = re.compile(r"<\|im_start\|>([a-zA-Z_]+)\n(.*?)<\|im_end\|>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"\n?<tool_call>\n?(.*?)\n?</tool_call>", re.DOTALL)
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{160,}")
_REDACT_RE = re.compile(
    r"\[(redacted|encrypted|reasoning (omitted|redacted|hidden)|thinking (omitted|redacted|hidden)|hidden)\]"
    r"|redacted_thinking|encrypted_content|<encrypted|reasoning (is )?(encrypted|hidden|not available)"
    r"|thinking (is )?(encrypted|hidden|not available)",
    re.IGNORECASE,
)


class RowFormatError(ValueError):
    def __init__(self, dataset: str, row: int, field_name: str, msg: str, row_obj: Any = None):
        keys = sorted(row_obj.keys()) if isinstance(row_obj, dict) else type(row_obj).__name__
        super().__init__(f"[{dataset}] row {row}: field '{field_name}': {msg} | row keys={keys}")
        self.dataset, self.row, self.field, self.reason = dataset, row, field_name, f"format:{field_name}"


class RowDrop(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


@dataclass
class Example:
    source: str
    kind: str  # "chat" | "text"
    messages: list[dict[str, Any]] | None = None
    text: str | None = None
    tools: list[Any] | None = None
    row_index: int = -1   # raw row position within the (epoch's) stream
    sub_index: int = 0    # >0 when one raw row expands to several examples (ARC test pairs)
    meta: dict[str, Any] = field(default_factory=dict)

    def prompt_text(self) -> str:
        if self.kind == "text":
            return self.text or ""
        return "\n".join(str(m.get("content") or "") for m in (self.messages or []) if m["role"] in ("system", "user"))

    def plain_text(self, include_reasoning: bool = True) -> str:
        if self.kind == "text":
            return self.text or ""
        parts = []
        for m in self.messages or []:
            if include_reasoning and m.get("reasoning"):
                parts.append(str(m["reasoning"]))
            parts.append(str(m.get("content") or ""))
        return "\n".join(parts)

    def response_text(self) -> str:
        return "\n".join(str(m.get("content") or "") for m in (self.messages or []) if m["role"] == "assistant")


# ----------------------------------------------------------------------------- think handling

def extract_think(content: str | None) -> tuple[str | None, str]:
    """Split `<think>R</think>C` (also <thought>/<thinking>) -> (R, C). Returns (None, content) if absent.

    Only *leading* think blocks are extracted (one or several in a row, joined with blank lines), so a
    response that merely talks about "`<thinking>` tags" is left alone. Also handles a missing opening
    tag ("R</think>C", common when the template pre-fills <think>). A leading but *unclosed* tag is
    left in content and flagged by `has_unclosed_think` (= truncated generation).
    """
    if not content:
        return None, content or ""
    blocks: list[str] = []
    rest = content
    while True:
        m = _LEAD_THINK_RE.match(rest)
        if not m:
            break
        blocks.append(m.group(2))
        rest = rest[m.end():]
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip()), rest.lstrip()
    m = _CLOSE_ONLY_RE.search(content)
    if m and not _THINK_OPEN_RE.search(content[: m.start()]) and not content[: m.start()].endswith("`"):
        return content[: m.start()].strip(), content[m.end():].lstrip()
    return None, content


def has_unclosed_think(content: str | None) -> bool:
    if not content:
        return False
    return bool(_LEAD_OPEN_RE.match(content)) and not _LEAD_THINK_RE.match(content)


def looks_encrypted(reasoning: str | None) -> str | None:
    """Drop reason if reasoning is unusable as a training target, else None.

    reasoning_empty     None / whitespace
    reasoning_redacted  placeholder text ("[redacted]", "reasoning omitted", redacted_thinking ...)
    reasoning_encrypted opaque blob: a long base64-ish run that makes up most of the text, or an
                        OpenAI/Fernet-style "gAAAA" token
    """
    if reasoning is None or not str(reasoning).strip():
        return "reasoning_empty"
    r = str(reasoning).strip()
    if len(r) < 400 and _REDACT_RE.search(r):
        return "reasoning_redacted"
    if r.startswith("gAAAA") and " " not in r[:200]:
        return "reasoning_encrypted"
    runs = _B64_RUN_RE.findall(r)
    if runs:
        blob = sum(len(x) for x in runs)
        if blob >= 0.5 * len(r):
            return "reasoning_encrypted"
    return None


# ----------------------------------------------------------------------------- helpers

def _first_key(row: dict[str, Any], keys: tuple[str, ...] | list[str], nonempty: bool = True) -> str | None:
    for k in keys:
        if k in row and row[k] is not None and (not nonempty or row[k] != ""):
            return k
    return None


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):  # OpenAI content parts
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in x)
    return str(x)


def _maybe_json(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return x
    return x


def _norm_tool_calls(tcs: Any) -> list[dict[str, Any]] | None:
    tcs = _maybe_json(tcs)
    if not tcs:
        return None
    if isinstance(tcs, dict):
        tcs = [tcs]
    out = []
    for raw in tcs:
        tc = _maybe_json(raw)
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", tc)
        fn = _maybe_json(fn)
        if not isinstance(fn, dict) or "name" not in fn:
            continue
        out.append({"type": "function", "function": {"name": fn["name"], "arguments": fn.get("arguments", {})}})
    return out or None


def _norm_message(item: Any, spec: DatasetSpec, idx: int, pos: int, row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RowFormatError(spec.name, idx, f"messages[{pos}]", f"expected dict, got {describe(item)}", row)
    raw_role = item.get("role", item.get("from"))
    if raw_role is None:
        raise RowFormatError(spec.name, idx, f"messages[{pos}].role", f"no role/from key in {describe(item)}", row)
    role = ROLE_MAP.get(str(raw_role).strip().lower())
    if role is None:
        raise RowFormatError(spec.name, idx, f"messages[{pos}].role", f"unknown role {raw_role!r}", row)
    content = _as_text(item.get("content", item.get("value", item.get("text"))))
    msg: dict[str, Any] = {"role": role, "content": content, "reasoning": None}
    if role == "function_call":  # sharegpt tool call turn -> assistant with tool_calls
        tcs = _norm_tool_calls(content)
        if tcs:
            return {"role": "assistant", "content": "", "reasoning": None, "tool_calls": tcs}
        msg["role"] = "assistant"
        role = "assistant"
    if role == "assistant":
        rk = _first_key(item, REASONING_KEYS)
        reasoning = item.get(rk) if rk else None
        if isinstance(reasoning, str):
            r2, _ = extract_think(reasoning)  # some datasets keep the tags inside the reasoning field
            reasoning = r2 if r2 is not None else reasoning
        if has_unclosed_think(content):
            raise RowDrop("think_unclosed", f"[{spec.name}] row {idx} messages[{pos}] has <think> without close")
        r_in, content = extract_think(content)
        if reasoning is None or not str(reasoning).strip():
            reasoning = r_in if r_in is not None else reasoning
        elif r_in:
            reasoning = f"{reasoning}\n\n{r_in}"
        msg["content"], msg["reasoning"] = content, reasoning
        tcs = _norm_tool_calls(item.get("tool_calls") or item.get("function_call"))
        if tcs:
            msg["tool_calls"] = tcs
    if role == "tool" and item.get("name"):
        msg["name"] = item["name"]
    return msg


def _parse_tools(x: Any) -> list[Any] | None:
    x = _maybe_json(x)
    if not x:
        return None
    if isinstance(x, dict):
        x = [x]
    return [_maybe_json(t) for t in x] if isinstance(x, list) else None


# ----------------------------------------------------------------------------- adapters

def adapt_messages(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    key = spec.fields.get("messages") or _first_key(row, MESSAGE_LIST_KEYS)
    if key is None or key not in row:
        raise RowFormatError(spec.name, idx, spec.fields.get("messages", "messages"),
                             f"no message list (looked for {MESSAGE_LIST_KEYS})", row)
    items = _maybe_json(row[key])
    if not isinstance(items, list) or not items:
        raise RowFormatError(spec.name, idx, key, f"expected non-empty list, got {describe(items)}", row)
    msgs = [_norm_message(it, spec, idx, i, row) for i, it in enumerate(items)]
    sk = _first_key(row, SYSTEM_KEYS)
    if sk and isinstance(row[sk], str) and row[sk].strip() and msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": row[sk], "reasoning": None})
    tools = _parse_tools(row.get(spec.fields.get("tools", "tools")))
    return [Example(spec.name, "chat", messages=msgs, tools=tools, row_index=idx)]


def adapt_prompt_response(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    pk = spec.fields.get("prompt") or _first_key(row, PROMPT_KEYS)
    if pk is None or not _as_text(row.get(pk)).strip():
        raise RowFormatError(spec.name, idx, pk or "prompt", f"missing/empty prompt (looked for {PROMPT_KEYS})", row)
    rk = spec.fields.get("response") or _first_key(row, RESPONSE_KEYS)
    gk = spec.fields.get("reasoning") or _first_key(row, REASONING_KEYS)
    response = _as_text(row.get(rk)) if rk else ""
    reasoning = row.get(gk) if gk else None
    if reasoning is not None and not isinstance(reasoning, str):
        raise RowFormatError(spec.name, idx, gk or "reasoning", f"expected str, got {describe(reasoning)}", row)
    if reasoning:
        r2, _ = extract_think(reasoning)
        reasoning = r2 if r2 is not None else reasoning
    if has_unclosed_think(response):
        raise RowDrop("think_unclosed", f"[{spec.name}] row {idx} field '{rk}' has <think> without close")
    r_in, response = extract_think(response)
    if not (reasoning and reasoning.strip()):
        reasoning = r_in if r_in is not None else reasoning
    if rk is None and not reasoning:
        raise RowFormatError(spec.name, idx, "response", f"no response column (looked for {RESPONSE_KEYS})", row)
    msgs: list[dict[str, Any]] = []
    sk = spec.fields.get("system") or _first_key(row, SYSTEM_KEYS)
    if sk and _as_text(row.get(sk)).strip():
        msgs.append({"role": "system", "content": _as_text(row[sk]), "reasoning": None})
    msgs.append({"role": "user", "content": _as_text(row[pk]), "reasoning": None})
    msgs.append({"role": "assistant", "content": response, "reasoning": reasoning})
    return [Example(spec.name, "chat", messages=msgs, row_index=idx)]


def adapt_nemotron(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    inp = _maybe_json(row.get("input"))
    if not isinstance(inp, list):
        raise RowFormatError(spec.name, idx, "input", f"expected list of messages, got {describe(inp)}", row)
    if "output" not in row:
        raise RowFormatError(spec.name, idx, "output", "missing", row)
    msgs = [_norm_message(it, spec, idx, i, row) for i, it in enumerate(inp)]
    sysp = row.get("system_prompt")
    if isinstance(sysp, str) and sysp.strip() and not (msgs and msgs[0]["role"] == "system"):
        msgs.insert(0, {"role": "system", "content": sysp, "reasoning": None})
    msgs.append(_norm_message({"role": "assistant", "content": row["output"]}, spec, idx, len(inp), row))
    return [Example(spec.name, "chat", messages=msgs, row_index=idx, meta={"reasoning_flag": row.get("reasoning")})]


def _parse_chatml_assistant(body: str) -> dict[str, Any]:
    tcs = []
    for m in _TOOL_CALL_RE.finditer(body):
        call = _maybe_json(m.group(1).strip())
        if not isinstance(call, dict) or "name" not in call:
            return {"role": "assistant", "content": body}  # not parseable: keep verbatim
        tcs.append({"type": "function", "function": {"name": call["name"], "arguments": call.get("arguments", {})}})
    if not tcs:
        return {"role": "assistant", "content": body}
    return {"role": "assistant", "content": _TOOL_CALL_RE.sub("", body), "tool_calls": tcs}


def adapt_chatml_text(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    key = spec.fields.get("text") or _first_key(row, TEXT_KEYS)
    text = row.get(key) if key else None
    if not isinstance(text, str) or "<|im_start|>" not in text:
        raise RowFormatError(spec.name, idx, key or "text", "expected a ChatML string with <|im_start|>", row)
    raw: list[dict[str, Any]] = []
    for m in _CHATML_RE.finditer(text):
        role, body = m.group(1).strip().lower(), m.group(2)
        if role == "user":
            stripped = body.strip()
            resps = _TOOL_RESP_RE.findall(stripped)
            if resps and not _TOOL_RESP_RE.sub("", stripped).strip():
                raw.extend({"role": "tool", "content": r} for r in resps)
                continue
        if role == "assistant":
            raw.append(_parse_chatml_assistant(body))
        else:
            raw.append({"role": role, "content": body})
    if not raw:
        raise RowFormatError(spec.name, idx, key or "text", "no complete <|im_start|>..<|im_end|> turns", row)
    msgs = [_norm_message(it, spec, idx, i, row) for i, it in enumerate(raw)]
    return [Example(spec.name, "chat", messages=msgs, row_index=idx)]


def adapt_text(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    key = spec.fields.get("text") or _first_key(row, TEXT_KEYS, nonempty=False)
    if key is None:
        raise RowFormatError(spec.name, idx, "text", f"no text column (looked for {TEXT_KEYS})", row)
    val = row[key]
    if not isinstance(val, str):
        raise RowFormatError(spec.name, idx, key, f"expected str, got {describe(val)}", row)
    return [Example(spec.name, "text", text=val, row_index=idx)]


def arc_grid_text(grid: Any) -> str:
    return "\n".join("".join(str(int(c)) for c in r) for r in grid)


def arc_task_prompt(task: dict[str, Any], test_input: Any) -> str:
    lines = ["Find the common transformation rule in the examples and apply it to the test input. "
             "Grids are rows of digits 0-9 (colours).", ""]
    for i, pair in enumerate(task.get("train", [])):
        lines += [f"Example {i + 1} input:", arc_grid_text(pair["input"]),
                  f"Example {i + 1} output:", arc_grid_text(pair["output"]), ""]
    lines += ["Test input:", arc_grid_text(test_input), "Test output:"]
    return "\n".join(lines)


def adapt_arc(row: dict[str, Any], spec: DatasetSpec, idx: int) -> list[Example]:
    if not isinstance(row.get("train"), list) or not isinstance(row.get("test"), list):
        raise RowFormatError(spec.name, idx, "train/test", "ARC task needs list fields 'train' and 'test'", row)
    out = []
    for j, pair in enumerate(row["test"]):
        if "output" not in pair:
            raise RowFormatError(spec.name, idx, f"test[{j}].output", "missing (hidden test set?)", row)
        msgs = [{"role": "user", "content": arc_task_prompt(row, pair["input"]), "reasoning": None},
                {"role": "assistant", "content": arc_grid_text(pair["output"]), "reasoning": None}]
        out.append(Example(spec.name, "chat", messages=msgs, row_index=idx, sub_index=j,
                           meta={"task_id": row.get("task_id")}))
    return out


def detect_format(row: dict[str, Any]) -> str:
    if isinstance(row.get("train"), list) and isinstance(row.get("test"), list):
        return "arc_agi"
    if isinstance(_maybe_json(row.get("input")), list) and "output" in row:
        return "nemotron"
    if _first_key(row, MESSAGE_LIST_KEYS):
        return "messages"
    tk = _first_key(row, TEXT_KEYS)
    if tk and isinstance(row[tk], str) and "<|im_start|>" in row[tk]:
        return "chatml_text"
    if _first_key(row, PROMPT_KEYS) and (_first_key(row, RESPONSE_KEYS) or _first_key(row, REASONING_KEYS)):
        return "prompt_response"
    if tk:
        return "text"
    return "unknown"


ADAPTERS = {
    "messages": adapt_messages,
    "sharegpt": adapt_messages,
    "prompt_response": adapt_prompt_response,
    "nemotron": adapt_nemotron,
    "chatml_text": adapt_chatml_text,
    "text": adapt_text,
    "arc_agi": adapt_arc,
}


def normalize_row(row: dict[str, Any], spec: DatasetSpec, idx: int = -1) -> list[Example]:
    """Row -> list of Examples (usually one). Raises RowFormatError / RowDrop with dataset+row context."""
    if not isinstance(row, dict):
        raise RowFormatError(spec.name, idx, "<row>", f"expected dict row, got {describe(row)}", row)
    fmt = spec.format
    if fmt == "auto":
        fmt = detect_format(row)
        if fmt == "unknown":
            raise RowFormatError(spec.name, idx, "<auto>", "could not detect row format; set `format`/`fields` "
                                 "for this dataset", row)
    adapter = ADAPTERS.get(fmt)
    if adapter is None:
        raise RowFormatError(spec.name, idx, "<format>", f"format {fmt!r} has no SFT/pretrain adapter "
                             f"(known: {sorted(ADAPTERS)}); RL formats go through normalize_rl_task", row)
    exs = adapter(row, spec, idx)
    for ex in exs:
        if ex.kind == "chat" and ex.messages:
            msgs = ex.messages
            if spec.drop_system_prompts:
                drop = {s.strip() for s in spec.drop_system_prompts}
                msgs = [m for m in msgs if not (m["role"] == "system" and str(m["content"]).strip() in drop)]
            while msgs and msgs[-1]["role"] != "assistant":  # agent traces often end on a tool result
                msgs = msgs[:-1]
            ex.messages = msgs
    return exs


def validate_example(ex: Example, spec: DatasetSpec) -> str | None:
    """None if trainable, else the drop reason."""
    if ex.kind == "text":
        if len((ex.text or "").strip()) < max(1, spec.min_chars):
            return "text_too_short"
        return None
    msgs = ex.messages or []
    asst = [m for m in msgs if m["role"] == "assistant"]
    if not asst:
        return "no_assistant"
    if not any(m["role"] == "user" for m in msgs):
        return "no_user"
    last = asst[-1]
    has_reasoning = bool(last.get("reasoning") and str(last["reasoning"]).strip())
    if not str(last.get("content") or "").strip() and not last.get("tool_calls"):
        if not has_reasoning:
            return "empty_response"
        if not spec.allow_empty_content:
            return "empty_content"
    reasons = [m.get("reasoning") for m in asst if m.get("reasoning") and str(m["reasoning"]).strip()]
    for r in reasons:
        bad = looks_encrypted(r)
        if bad:
            return bad
    if spec.require_reasoning and not reasons:
        return "reasoning_empty"
    return None


# ----------------------------------------------------------------------------- drop accounting

class DropCounter:
    """Per-dataset kept/dropped counts by reason. Logs the first few drops of each (dataset, reason)
    with the row index, then only a running summary; raises when a dataset is mostly malformed."""

    def __init__(self, log_first: int = 3, max_format_error_frac: float = 0.5, min_rows_for_budget: int = 200):
        self.kept: collections.Counter[str] = collections.Counter()
        self.dropped: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        self.samples: dict[str, list[str]] = collections.defaultdict(list)
        self.log_first = log_first
        self.max_format_error_frac = max_format_error_frac
        self.min_rows_for_budget = min_rows_for_budget

    def keep(self, source: str) -> None:
        self.kept[source] += 1

    def drop(self, source: str, reason: str, row: int, detail: str = "") -> None:
        c = self.dropped[source]
        c[reason] += 1
        if c[reason] <= self.log_first:
            log.info("[data] drop %s row %d: %s %s", source, row, reason, detail[:300])
            self.samples[source].append(f"row {row}: {reason} {detail[:300]}")
        elif c[reason] in (10, 100, 1000) or c[reason] % 10000 == 0:
            log.info("[data] %s: %d rows dropped so far for '%s' (kept %d)", source, c[reason], reason,
                     self.kept[source])
        self._check_budget(source)

    def _check_budget(self, source: str) -> None:
        c = self.dropped[source]
        n_fmt = sum(v for k, v in c.items() if k.startswith("format:"))
        seen = self.kept[source] + sum(c.values())
        if seen >= self.min_rows_for_budget and n_fmt > self.max_format_error_frac * seen:
            raise RuntimeError(
                f"[{source}] {n_fmt}/{seen} rows failed to parse — the dataset schema probably changed or the "
                f"wrong `format` is configured. Drop reasons: {dict(c)}. Samples: {self.samples[source][:5]}"
            )

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for s in sorted(set(self.kept) | set(self.dropped)):
            out[s] = {"kept": self.kept[s], **dict(self.dropped[s])}
        return out

    def log_summary(self) -> None:
        for s, d in self.summary().items():
            n_drop = sum(v for k, v in d.items() if k != "kept")
            log.info("[data] %s: kept %d, dropped %d %s", s, d["kept"], n_drop,
                     {k: v for k, v in d.items() if k != "kept"})


# ----------------------------------------------------------------------------- RL tasks

def extract_task_archive(blob: bytes, max_file_bytes: int = 1 << 20) -> dict[str, str]:
    """Unpack a (gzipped) tar held in memory -> {path: text} for UTF-8 decodable files."""
    data = bytes(blob)
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or member.size > max_file_bytes:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                files[member.name] = fh.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
    return files


_KERNEL_PROMPT = (
    "You are given a PyTorch reference implementation. Write a faster drop-in replacement named `ModelNew` "
    "with the same interface, using custom CUDA/Triton kernels where they help. Keep numerics within "
    "tolerance and return the complete Python module.\n\n```python\n{code}\n```"
)


def normalize_rl_task(row: dict[str, Any], spec: DatasetSpec, idx: int = -1) -> dict[str, Any]:
    """RL row -> {"task_id", "prompt", "source", "meta"} (verifier-specific data lives in meta)."""
    fmt = spec.format
    if fmt == "rl_kernelbench":
        for k in ("code", "level", "name"):
            if k not in row:
                raise RowFormatError(spec.name, idx, k, "missing KernelBench column", row)
        return {"task_id": f"L{row['level']}_{row['name']}", "prompt": _KERNEL_PROMPT.format(code=row["code"]),
                "source": spec.name, "meta": {"level": row["level"], "problem_id": row.get("problem_id"),
                                              "code": row["code"]}}
    if fmt == "rl_swe":
        for k in ("instance_id", "problem_statement"):
            if not row.get(k):
                raise RowFormatError(spec.name, idx, k, "missing SWE task column", row)
        prompt = row["problem_statement"]
        if row.get("interface") and "No new interfaces" not in str(row["interface"]):
            prompt += "\n\nInterface:\n" + str(row["interface"])
        meta = {k: row.get(k) for k in ("repo", "base_commit", "image_name", "language", "FAIL_TO_PASS",
                                        "PASS_TO_PASS", "test_patch", "install_config", "patch")}
        return {"task_id": row["instance_id"], "prompt": prompt, "source": spec.name, "meta": meta}
    if fmt == "rl_ot_agent":
        if "task_binary" not in row or "path" not in row:
            raise RowFormatError(spec.name, idx, "task_binary", "needs 'path' and 'task_binary'", row)
        try:
            files = extract_task_archive(row["task_binary"])
        except Exception as e:
            raise RowFormatError(spec.name, idx, "task_binary", f"cannot unpack archive: {e!r}", row) from e
        prompt = ""
        for name in sorted(files, key=lambda p: (p.count("/"), p)):
            base = name.rsplit("/", 1)[-1].lower()
            if base.startswith(("instruction", "task", "prompt", "readme")) and base.endswith((".md", ".txt")):
                prompt = files[name]
                break
        return {"task_id": row["path"], "prompt": prompt, "source": spec.name, "meta": {"files": files}}
    # generic: prompt from messages or prompt-like column
    if isinstance(_maybe_json(row.get("input")), list):
        msgs = _maybe_json(row["input"])
        prompt = "\n".join(_as_text(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user")
    else:
        pk = spec.fields.get("prompt") or _first_key(row, PROMPT_KEYS)
        if pk is None:
            raise RowFormatError(spec.name, idx, "prompt", f"no prompt column (looked for {PROMPT_KEYS})", row)
        prompt = _as_text(row[pk])
    tid = row.get("id", row.get("task_id", f"{spec.name}-{idx}"))
    return {"task_id": str(tid), "prompt": prompt, "source": spec.name, "meta": dict(row)}
