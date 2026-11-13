"""Tiny HF-like stand-ins for offline tests and smoke runs (no transformers / downloads needed).

* `ByteTokenizer` - byte-level tokenizer with `encode / decode / apply_chat_template / eos_token_id`
* `ScriptedLM`    - cache-capable "model" that emits a fixed script (deterministic text for
                    testing stop strings, scoring, runners and the server end-to-end)
* `build_toy_generator` - `Generator` over `qloop.model.toy` (optionally loop-retrofitted)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .generate import Generator


class ByteTokenizer:
    """ids 0..255 = UTF-8 bytes, 256 = <eos>, 257 = <pad>."""

    eos_token_id = 256
    pad_token_id = 257
    vocab_size = 258
    eos_token = "<eos>"
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        out, buf = [], bytearray()
        for raw in (ids.tolist() if hasattr(ids, "tolist") else ids):
            i = int(raw)
            if i < 256:
                buf.append(i)
                continue
            if not skip_special_tokens:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append(self.eos_token if i == self.eos_token_id else self.pad_token)
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = True, **kwargs: Any) -> Any:
        text = "".join(f"<|{m['role']}|>{m.get('content', '')}\n" for m in messages)
        if kwargs.get("tools"):
            text = f"<|tools|>{kwargs['tools']}\n" + text
        if add_generation_prompt:
            text += "<|assistant|>"
        return self.encode(text) if tokenize else text


@dataclass
class _ScriptOut:
    logits: torch.Tensor
    past_key_values: Any


class ScriptedLM(nn.Module):
    """Emits `script` (token ids) one token per step regardless of the prompt, then `eos_id` forever.

    The "cache" is a dict counting decode steps, so the Generator's cache protocol is exercised.
    The scripted token gets logit +`margin`, all others 0 => greedy is exact, sampling mostly follows.
    """

    def __init__(self, script: list[int], vocab_size: int = 258, eos_id: int | None = 256, margin: float = 30.0):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.margin = margin
        self.dummy = nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    @classmethod
    def from_text(cls, text: str, eos: bool = True, **kw: Any) -> ScriptedLM:
        return cls(list(text.encode("utf-8")), eos_id=256 if eos else None, **kw)

    def forward(self, input_ids: torch.Tensor, past_key_values: Any = None, use_cache: bool = False,
                **kwargs: Any) -> _ScriptOut:
        self.forward_calls += 1
        b, t = input_ids.shape
        cache = past_key_values if past_key_values is not None else {"step": -1, "seen": 0}
        cache["step"] += 1
        cache["seen"] += t
        k = cache["step"]
        if k < len(self.script):
            tok = self.script[k]
        else:
            tok = self.eos_id if self.eos_id is not None else (self.script[-1] if self.script else 0)
        logits = torch.zeros(b, t, self.vocab_size, device=input_ids.device)
        logits[:, :, tok] = self.margin
        return _ScriptOut(logits, cache if use_cache else None)


def build_toy_generator(seed: int = 0, loop_spec: dict[str, Any] | None = None, **gen_kwargs: Any) -> Generator:
    from qloop.model.loop import apply_loop
    from qloop.model.toy import build_toy

    model = build_toy(seed=seed, vocab_size=ByteTokenizer.vocab_size)
    ctl = apply_loop(model, loop_spec) if loop_spec else None
    return Generator(model, ByteTokenizer(), "cpu", ctl, **gen_kwargs)
