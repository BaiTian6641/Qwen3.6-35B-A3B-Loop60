"""Tokenisation with loss masks + greedy sequence packing.

Label convention (HF): `labels` is aligned with `input_ids` (the model shifts internally), -100 =
ignored. Trained: assistant reasoning/content/tool calls and the closing <|im_end|> (the turn eos),
and every token of pretrain text plus its trailing document separator. Ignored: system/user/tool
turns, role headers, the newline after <|im_end|>, the separator after a chat document, padding.

Each document ends with a separator token (<|endoftext|> if the tokenizer has it, else eos).
`Packer` concatenates documents into fixed `seq_len` rows:
  * respect_doc_boundaries=False: one continuous stream cut every seq_len tokens (no padding)
  * respect_doc_boundaries=True : a document is never split unless it is longer than seq_len
    (then it fills the current row and continues in the next ones); otherwise the row is padded
    and the document starts a fresh row.
Output per row: input_ids/labels/doc_ids/position_ids LongTensor[T] (+ n_loss_tokens, n_pad, n_docs).
`doc_ids` (-1 = pad) and per-document `position_ids` allow block-diagonal attention / diagnostics.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from qloop.debug import require

from .formats import Example
from .tokenizer import ENDOFTEXT, render_conversation, special_id

log = logging.getLogger("qloop")

IGNORE = -100


def separator_id(tokenizer: Any) -> int:
    sid = special_id(tokenizer, ENDOFTEXT)
    if sid is None:
        sid = tokenizer.eos_token_id
    require(sid is not None, "tokenizer has neither <|endoftext|> nor eos_token_id", tokenizer=tokenizer)
    return int(sid)


def tokenize_conversation(messages: list[dict[str, Any]], tokenizer: Any, tools: list[Any] | None = None,
                          keep_history_reasoning: bool = False,
                          verify: bool = False) -> tuple[list[int], list[int]]:
    """Render with the Qwen template and tokenise segment-wise -> (ids, labels)."""
    segs = render_conversation(messages, tools=tools, keep_history_reasoning=keep_history_reasoning)
    ids: list[int] = []
    labels: list[int] = []
    for text, trainable in segs:
        t = list(tokenizer.encode(text, add_special_tokens=False))
        ids.extend(t)
        labels.extend(t if trainable else [IGNORE] * len(t))
    if verify:
        full = list(tokenizer.encode("".join(s for s, _ in segs), add_special_tokens=False))
        if full != ids:
            k = next((i for i, (a, b) in enumerate(zip(full, ids)) if a != b), min(len(full), len(ids)))
            log.warning("segment-wise tokenisation differs from full-string tokenisation at token %d "
                        "(full=%d seg=%d tokens): %r", k, len(full), len(ids),
                        tokenizer.decode(full[max(0, k - 8):k + 8], skip_special_tokens=False))
    return ids, labels


def tokenize_text(text: str, tokenizer: Any, sep_id: int | None = None) -> tuple[list[int], list[int]]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    ids.append(separator_id(tokenizer) if sep_id is None else sep_id)
    return ids, list(ids)


def tokenize_example(ex: Example, tokenizer: Any, sep_id: int | None = None, keep_history_reasoning: bool = False,
                     verify: bool = False) -> tuple[list[int], list[int]]:
    sep = separator_id(tokenizer) if sep_id is None else sep_id
    if ex.kind == "text":
        return tokenize_text(ex.text or "", tokenizer, sep)
    require(ex.kind == "chat" and ex.messages, f"[{ex.source}] row {ex.row_index}: cannot tokenize example",
            kind=ex.kind, messages=ex.messages)
    ids, labels = tokenize_conversation(ex.messages or [], tokenizer, tools=ex.tools,
                                        keep_history_reasoning=keep_history_reasoning, verify=verify)
    ids.append(sep)
    labels.append(IGNORE)
    return ids, labels


class Packer:
    """Greedy packer with serialisable state (so a resumed run continues mid-row exactly)."""

    def __init__(self, seq_len: int, pad_id: int, respect_doc_boundaries: bool = True, drop_no_loss: bool = True):
        require(seq_len >= 2, "Packer: seq_len must be >= 2", seq_len=seq_len)
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.respect = bool(respect_doc_boundaries)
        self.drop_no_loss = drop_no_loss
        self._ids: list[int] = []
        self._labels: list[int] = []
        self._docs: list[int] = []
        self.n_rows = 0
        self.n_dropped_no_loss = 0
        self.n_pad_tokens = 0
        self.n_tokens = 0

    def add(self, ids: list[int], labels: list[int], doc_id: int) -> list[dict[str, Any]]:
        require(len(ids) == len(labels), "Packer.add: ids/labels length mismatch", doc_id=doc_id, ids=len(ids),
                labels=len(labels))
        if not ids:
            return []
        out: list[dict[str, Any]] = []
        T = self.seq_len
        if self.respect and len(ids) <= T and self._ids and len(self._ids) + len(ids) > T:
            self._emit(out, pad=True)
        self._ids.extend(ids)
        self._labels.extend(labels)
        self._docs.extend([doc_id] * len(ids))
        while len(self._ids) >= T:
            self._emit(out, pad=False)
        return out

    def flush(self) -> dict[str, Any] | None:
        out: list[dict[str, Any]] = []
        if self._ids:
            self._emit(out, pad=True)
        return out[0] if out else None

    def _emit(self, out: list[dict[str, Any]], pad: bool) -> None:
        T = self.seq_len
        ids, labels, docs = self._ids[:T], self._labels[:T], self._docs[:T]
        self._ids, self._labels, self._docs = self._ids[T:], self._labels[T:], self._docs[T:]
        n_pad = T - len(ids)
        if n_pad:
            require(pad, "Packer internal error: short row without padding", n=len(ids), T=T)
            ids = ids + [self.pad_id] * n_pad
            labels = labels + [IGNORE] * n_pad
            docs = docs + [-1] * n_pad
        lab = torch.tensor(labels, dtype=torch.long)
        n_loss = int((lab[1:] != IGNORE).sum())  # position 0 is never a target after the causal shift
        if n_loss == 0 and self.drop_no_loss:
            self.n_dropped_no_loss += 1
            return
        docs_t = torch.tensor(docs, dtype=torch.long)
        n_valid = T - n_pad
        if n_valid == 0:
            pos_t = torch.zeros(T, dtype=torch.long)
            n_docs = 0
        elif bool((docs_t[:n_valid] == -1).any()):
            pos_list: list[int] = []
            prev: int | None = None
            p = 0
            for d in docs:
                if d == -1:
                    pos_list.append(0)
                    continue
                p = p + 1 if d == prev else 0
                pos_list.append(p)
                prev = d
            pos_t = torch.tensor(pos_list, dtype=torch.long)
            n_docs = len({d for d in docs if d != -1})
        else:
            v_docs = docs_t[:n_valid]
            change = torch.ones(n_valid, dtype=torch.bool)
            if n_valid > 1:
                change[1:] = v_docs[1:] != v_docs[:-1]
            idx = torch.arange(n_valid, dtype=torch.long)
            run_starts = torch.where(change, idx, torch.zeros_like(idx)).cummax(dim=0).values
            pos_valid = idx - run_starts
            pos_t = F.pad(pos_valid, (0, n_pad), value=0) if n_pad else pos_valid
            n_docs = int(torch.unique(v_docs).numel())
        self.n_rows += 1
        self.n_pad_tokens += n_pad
        self.n_tokens += T
        out.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": lab,
            "doc_ids": docs_t,
            "position_ids": pos_t,
            "n_loss_tokens": n_loss,
            "n_pad": n_pad,
            "n_docs": n_docs,
        })

    def state_dict(self) -> dict[str, Any]:
        return {"ids": list(self._ids), "labels": list(self._labels), "docs": list(self._docs),
                "n_rows": self.n_rows, "n_dropped_no_loss": self.n_dropped_no_loss,
                "n_pad_tokens": self.n_pad_tokens, "n_tokens": self.n_tokens, "seq_len": self.seq_len}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if int(sd.get("seq_len", self.seq_len)) != self.seq_len:
            log.warning("Packer: resuming with seq_len %d (checkpoint had %s); buffered tokens are kept",
                        self.seq_len, sd.get("seq_len"))
        self._ids, self._labels, self._docs = list(sd["ids"]), list(sd["labels"]), list(sd["docs"])
        self.n_rows = int(sd.get("n_rows", 0))
        self.n_dropped_no_loss = int(sd.get("n_dropped_no_loss", 0))
        self.n_pad_tokens = int(sd.get("n_pad_tokens", 0))
        self.n_tokens = int(sd.get("n_tokens", 0))

    @property
    def fill_ratio(self) -> float:
        return 1.0 - self.n_pad_tokens / max(1, self.n_tokens)
