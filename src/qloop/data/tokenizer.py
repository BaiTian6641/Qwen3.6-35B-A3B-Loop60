"""Tokenizers + the single source of truth for the Qwen chat format.

`render_conversation()` turns a normalised conversation into `(text, trainable)` segments in the
Qwen3 ChatML style:

    <|im_start|>system\\n...<|im_end|>\\n
    <|im_start|>user\\n...<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\nREASONING\\n</think>\\n\\nCONTENT<|im_end|>\\n

Segment boundaries are always placed next to special tokens, so tokenising segment-by-segment
gives exactly the same ids as tokenising the joined string (added/special tokens are never
merged by BPE). `packing.tokenize_conversation` relies on that to build loss masks.

`ToyTokenizer` is a byte-level tokenizer with the HF surface we use (offline tests); its
`apply_chat_template` *is* `render_conversation`, so the toy path and the training path cannot
drift. For the real tokenizer, `check_template_parity()` diffs our rendering against the model's
own jinja template (run by `qloop doctor`) so a template change upstream is caught loudly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from qloop.debug import describe, require

log = logging.getLogger("qloop")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def _text(x: Any) -> str:
    """Message content -> str. Accepts str, None, or OpenAI-style list of {type:text,text:..} parts."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for p in x:
            if isinstance(p, dict):
                parts.append(str(p.get("text", p.get("content", ""))))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(x)


def _split_reasoning(msg: dict[str, Any]) -> tuple[str, str]:
    """(reasoning, content) of an assistant message; falls back to <think> inside content."""
    from .formats import extract_think

    content = _text(msg.get("content"))
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    if reasoning is None:
        r, content = extract_think(content)
        reasoning = r
    return ("" if reasoning is None else str(reasoning)), content


def _render_tool_call(tc: Any) -> str:
    fn = tc.get("function", tc) if isinstance(tc, dict) else {"name": str(tc), "arguments": {}}
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    head = '<tool_call>\n{"name": ' + json.dumps(name, ensure_ascii=False)
    return head + ', "arguments": ' + args + "}\n</tool_call>"


def render_conversation(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    keep_history_reasoning: bool = False,
) -> list[tuple[str, bool]]:
    """Render to `(text, trainable)` segments. Trainable = assistant output (reasoning, content,
    tool calls, closing <|im_end|>). Everything else (system/user/tool turns, role headers, the
    newline after <|im_end|>) is scaffolding/context and gets label -100.

    Mirrors the Qwen3 template: reasoning of assistant turns *before the last real user query* is
    dropped (that is what the model sees at inference); pass keep_history_reasoning=True to keep it.
    """
    require(isinstance(messages, list), "render_conversation: messages must be a list", messages=messages)
    segs: list[tuple[str, bool]] = []
    msgs = list(messages)

    # ---- system (+ tools)
    sys_text = None
    if msgs and msgs[0].get("role") == "system":
        sys_text = _text(msgs[0].get("content"))
        msgs = msgs[1:]
    if tools:
        body = (sys_text + "\n\n") if sys_text else ""
        body += _TOOLS_HEADER
        for t in tools:
            body += "\n" + (t if isinstance(t, str) else json.dumps(t, ensure_ascii=False))
        body += _TOOLS_FOOTER
        segs.append((f"{IM_START}system\n{body}{IM_END}\n", False))
    elif sys_text is not None:
        segs.append((f"{IM_START}system\n{sys_text}{IM_END}\n", False))

    last_query = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            last_query = i

    n = len(msgs)
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role in ("user", "system"):
            segs.append((f"{IM_START}{role}\n{_text(m.get('content'))}{IM_END}\n", False))
        elif role == "assistant":
            reasoning, content = _split_reasoning(m)
            out = ""
            show_think = (i > last_query and (i == n - 1 or reasoning.strip())) or (
                keep_history_reasoning and reasoning.strip()
            )
            if show_think:
                out += f"{THINK_OPEN}\n{reasoning.strip(chr(10))}\n{THINK_CLOSE}\n\n{content.lstrip(chr(10))}"
            else:
                out += content
            for j, tc in enumerate(m.get("tool_calls") or []):
                if (j == 0 and content) or j > 0:
                    out += "\n"
                out += _render_tool_call(tc)
            segs.append((f"{IM_START}assistant\n", False))
            segs.append((out + IM_END, True))
            segs.append(("\n", False))
        elif role == "tool":
            first = i == 0 or msgs[i - 1].get("role") != "tool"
            last = i == n - 1 or msgs[i + 1].get("role") != "tool"
            s = (f"{IM_START}user" if first else "") + f"\n<tool_response>\n{_text(m.get('content'))}\n</tool_response>"
            if last:
                s += f"{IM_END}\n"
            segs.append((s, False))
        else:
            raise ValueError(f"render_conversation: unknown role {role!r} at message {i}: {describe(m)}")

    if add_generation_prompt:
        gp = f"{IM_START}assistant\n"
        if enable_thinking is False:
            gp += f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
        segs.append((gp, False))

    # merge adjacent segments with the same mask (fewer tokenizer calls, boundaries stay at specials)
    merged: list[tuple[str, bool]] = []
    for t, tr in segs:
        if merged and merged[-1][1] == tr:
            merged[-1] = (merged[-1][0] + t, tr)
        else:
            merged.append((t, tr))
    return merged


def render_text(messages: list[dict[str, Any]], **kw: Any) -> str:
    return "".join(t for t, _ in render_conversation(messages, **kw))


# ----------------------------------------------------------------------------- toy tokenizer

class ToyTokenizer:
    """Byte-level tokenizer (ids 0..255 = raw UTF-8 bytes) + Qwen-style special tokens above 255.

    Like Qwen3: eos = <|im_end|>, pad = <|endoftext|>, bos = None. `<think>`/`<tool_call>` tags are
    *added but not special* (kept by decode(skip_special_tokens=True)), same as the real tokenizer.
    """

    SPECIAL_TOKENS = (ENDOFTEXT, IM_START, IM_END)
    ADDED_TOKENS = (THINK_OPEN, THINK_CLOSE, "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")

    def __init__(self) -> None:
        self.added = list(self.SPECIAL_TOKENS) + list(self.ADDED_TOKENS)
        self.token_to_id = {t: 256 + i for i, t in enumerate(self.added)}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._special_ids = {self.token_to_id[t] for t in self.SPECIAL_TOKENS}
        alt = "|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True))
        self._split_re = re.compile(f"({alt})")
        self.name_or_path = "toy"
        self.eos_token = IM_END
        self.pad_token = ENDOFTEXT
        self.bos_token = None
        self.chat_template = "qloop.data.tokenizer.render_conversation"

    # ---- HF-compatible attributes
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.added)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[IM_END]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[ENDOFTEXT]

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(self._special_ids)

    def convert_tokens_to_ids(self, token: str | list[str]) -> Any:
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        if token in self.token_to_id:
            return self.token_to_id[token]
        b = token.encode("utf-8")
        require(len(b) == 1, "ToyTokenizer.convert_tokens_to_ids: unknown multi-byte token", token=token)
        return b[0]

    # ---- encode / decode
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        require(isinstance(text, str), "ToyTokenizer.encode expects str", text=text)
        ids: list[int] = []
        for piece in self._split_re.split(text):
            if not piece:
                continue
            tid = self.token_to_id.get(piece)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(piece.encode("utf-8"))
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out: list[str] = []
        buf = bytearray()
        for raw in ids:
            i = int(raw)
            if i < 0:
                continue  # e.g. -100 label padding
            if i < 256:
                buf.append(i)
                continue
            require(i < self.vocab_size, "ToyTokenizer.decode: id out of range", id=i, vocab=self.vocab_size)
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            if skip_special_tokens and i in self._special_ids:
                continue
            out.append(self.id_to_token[i])
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        **_: Any,
    ) -> Any:
        text = render_text(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                           enable_thinking=enable_thinking)
        return self.encode(text) if tokenize else text


# ----------------------------------------------------------------------------- loading / checks

def load_tokenizer(name_or_path: str, **kwargs: Any) -> Any:
    """'toy' -> ToyTokenizer, else transformers.AutoTokenizer(trust_remote_code=True)."""
    if name_or_path == "toy":
        return ToyTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - colab only
        raise ImportError(
            f"load_tokenizer({name_or_path!r}) needs `transformers` (pip install -e '.[colab]'); "
            "use name 'toy' for offline tests"
        ) from e
    kwargs.setdefault("trust_remote_code", True)
    tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    require(tok.eos_token_id is not None, f"tokenizer {name_or_path} has no eos_token_id", tok=tok)
    log.info("tokenizer %s: vocab=%d eos=%r(%s) pad=%r(%s)", name_or_path, len(tok), tok.eos_token,
             tok.eos_token_id, tok.pad_token, tok.pad_token_id)
    return tok


def special_id(tokenizer: Any, token: str) -> int | None:
    """Id of an added/special token string, or None if the tokenizer does not know it."""
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    ids = tokenizer.encode(token, add_special_tokens=False)
    return int(tid) if list(ids) == [tid] else None


def check_template_parity(tokenizer: Any, messages: list[dict[str, Any]], **kw: Any) -> tuple[bool, str]:
    """Compare render_conversation() with the tokenizer's own chat template. Returns (ok, report)."""
    ours = render_text(messages, **kw)
    hf_msgs = []
    for m in messages:
        mm = dict(m)
        if mm.get("reasoning") is not None:
            mm["reasoning_content"] = mm.pop("reasoning")
        else:
            mm.pop("reasoning", None)
        hf_msgs.append(mm)
    try:
        theirs = tokenizer.apply_chat_template(hf_msgs, tokenize=False, **kw)
    except Exception as e:
        return False, f"tokenizer.apply_chat_template failed: {type(e).__name__}: {e}"
    if ours == theirs:
        return True, "chat template parity OK"
    k = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
    rep = (f"chat template MISMATCH at char {k}:\n  ours  : {ours[max(0, k - 80):k + 120]!r}\n"
           f"  theirs: {theirs[max(0, k - 80):k + 120]!r}\n"
           "Loss masks are built from qloop's renderer; update qloop/data/tokenizer.py to match.")
    return False, rep
