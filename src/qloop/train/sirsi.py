"""Supervised Instructed Recursive Self-Improvement (SI-RSI) for Stage 4 RL.

Provides pure-Python pathology detection, vibrant anti-slop voice shaping, numerically bounded
token-level unlikelihood ("slap") loss, surgical parameter-group routing helpers, structured
failure logging, and Supervisor prompt/decontamination utilities:

  1. Tier 1 (Local, zero API dependency):
     - `detect_pathologies(text, *, reward, cfg)` tags `LOOP_DEGENERATION` (with exact
       `loop_onset_char`), `UNCLOSED_THINK`, `PREMATURE_EXIT`, and `AI_SLOP` (strictly on
       final-answer prose after `</think>`, bounded tie-breaker penalty).
     - `map_char_span_to_token_mask(...)` maps `loop_onset_char` to a boolean completion mask.
     - `unlikelihood_span_loss(...)` applies bounded `-log(1 - p)` strictly on masked tokens.
     - `FailureLedger` records `all_fail` and pathology-tagged rollouts to `sirsi_failures.jsonl`.
  2. Tier 2 (Guided On-Policy Recovery when Supervisor `rl.teacher` is configured):
     - `build_hint_messages(...)` requests a 1-sentence supervisory nudge before divergence so the
       student can retry on-policy and distill its own verified recovery without the hint.
  3. Tier 3 (Curriculum Sculpting + 13-Gram Decontamination):
     - `build_curriculum_sculpt_messages(...)` + `parse_and_decontaminate_sculpted_tasks(...)`
       synthesize targeted verifiable micro-tasks while blocking any held-out benchmark overlap.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..debug import require
from ..model.experts import param_group_of

log = logging.getLogger("qloop")

TAG_LOOP_DEGENERATION = "LOOP_DEGENERATION"
TAG_UNCLOSED_THINK = "UNCLOSED_THINK"
TAG_PREMATURE_EXIT = "PREMATURE_EXIT"
TAG_AI_SLOP = "AI_SLOP"

# Curated anti-slop patterns inspected strictly on final-answer prose (never inside <think>).
DEFAULT_SLOP_PATTERNS: tuple[str, ...] = (
    r"\bcertainly!\s*(?:let'?s\s+delve|here(?:'s|\s+is)|i(?:'d|\s+would)\s+be\s+happy|let\s+me)\b",
    r"\b(?:absolutely|of\s+course|sure\s+thing)!\s*(?:let'?s\s+delve|here(?:'s|\s+is)|i(?:'d|\s+would)\s+be\s+happy)\b",
    r"\bgreat\s+question!\b",
    r"\bi(?:'d|\s+would)\s+be\s+happy\s+to\s+(?:help|assist|explain|walk\s+you)\b",
    r"\blet'?s\s+delve\s+(?:into|deeper)\b",
    r"\bdelve\s+into\s+the\s+(?:intricacies|nuances|world|realm|depths)\b",
    r"\bin\s+today'?s\s+fast[- ]paced\b",
    r"\bin\s+the\s+ever[- ]evolving\s+(?:world|landscape|realm|field)\b",
    r"\bit\s+is\s+important\s+to\s+note\s+that\b",
    r"\bit'?s\s+worth\s+noting\s+that\b",
    r"\bas\s+an\s+ai(?:\s+language)?\s+model\b",
    r"\ba\s+testament\s+to\b",
    r"\bunlock(?:ing)?\s+the\s+(?:full\s+)?potential\b",
    r"\bnavigate\s+the\s+(?:complexities|nuances|landscape)\b",
    r"\b(?:rich|intricate)\s+tapestry\b",
    r"\bwithout\s+further\s+ado\b",
    r"\bgame[- ]changer\b",
    r"\bseamless(?:ly)?\s+integrat(?:e|es|ing|ion)\b",
    r"\bin\s+conclusion,\s+to\s+summarize\b",
    r"\bfeel\s+free\s+to\s+(?:ask|reach\s+out)\s+if\s+you\s+have\s+any\s+(?:more|further|other)\s+questions\b",
    r"\bi\s+hope\s+this\s+helps!\s*(?:let\s+me\s+know|feel\s+free)\b",
)

_TOKEN_SPAN_RE = re.compile(r"\S+")
_SINGLE_CHAR_LOOP_RE = re.compile(r"([^\s\-=_*#])\1{31,}")
_ANY_CHAR_LOOP_RE = re.compile(r"(\S)\1{63,}")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_GROUP_ALIASES: dict[str, str] = {
    "loop_adapter": "loop_adapters",
    "hssd_adapter": "hssd_adapters",
    "gdn2_adapter": "gdn2_adapters",
    "lsa_indexer": "lsa_indexers",
    "expert_delta": "expert_deltas",
}


# ============================================================================ config & pathology


@dataclass
class SIRSIConfig:
    enabled: bool = True
    slap_weight: float = 0.1
    slap_eps: float = 1e-4
    slap_max_seqs_per_step: int = 4
    slap_target_groups: list[str] = field(default_factory=list)
    slap_fallback_tail_frac: float = 0.5
    anti_slop_weight: float = 0.05
    max_slop_penalty: float | None = None
    slop_penalty_per_hit: float = 0.02
    slop_patterns: list[str] | None = None
    min_answer_chars: int = 8
    loop_ngram_sizes: tuple[int, ...] = (8, 12)
    loop_min_repeats: int = 3
    tail_ngram_size: int = 4
    tail_min_tokens: int = 24
    tail_window_tokens: int = 128
    tail_dup_threshold: float = 0.5
    guided_hint_enabled: bool = False
    max_hints_per_step: int = 2
    guided_retry_n: int = 1
    failure_ledger: bool = True
    failure_ledger_filename: str = "sirsi_failures.jsonl"
    curriculum_sculpt_n_tasks: int = 4

    @property
    def effective_max_slop_penalty(self) -> float:
        if self.max_slop_penalty is None:
            return max(0.0, float(self.anti_slop_weight))
        return max(0.0, min(float(self.anti_slop_weight), float(self.max_slop_penalty)))

    @classmethod
    def from_dict(cls, raw: SIRSIConfig | dict[str, Any] | None = None) -> SIRSIConfig:
        if isinstance(raw, cls):
            return raw
        if not raw:
            return cls()
        d = dict(raw)
        has_anti = "anti_slop_weight" in d and d["anti_slop_weight"] is not None
        has_max = "max_slop_penalty" in d and d["max_slop_penalty"] is not None
        anti_w = float(d["anti_slop_weight"]) if has_anti else (float(d["max_slop_penalty"]) if has_max else 0.05)
        max_pen = float(d["max_slop_penalty"]) if has_max else None
        raw_ngrams = d.get("loop_ngram_sizes", (8, 12))
        ngrams = tuple(int(x) for x in raw_ngrams) if raw_ngrams else (8, 12)
        raw_groups = d.get("slap_target_groups") or []
        groups = [str(g) for g in raw_groups] if not isinstance(raw_groups, str) else [raw_groups]
        raw_pats = d.get("slop_patterns")
        pats = [str(p) for p in raw_pats] if raw_pats is not None else None
        cfg = cls(
            enabled=bool(d.get("enabled", True)),
            slap_weight=float(d.get("slap_weight", 0.1)),
            slap_eps=float(d.get("slap_eps", 1e-4)),
            slap_max_seqs_per_step=int(d.get("slap_max_seqs_per_step", 4)),
            slap_target_groups=groups,
            slap_fallback_tail_frac=float(d.get("slap_fallback_tail_frac", 0.5)),
            anti_slop_weight=anti_w,
            max_slop_penalty=max_pen,
            slop_penalty_per_hit=float(d.get("slop_penalty_per_hit", 0.02)),
            slop_patterns=pats,
            min_answer_chars=int(d.get("min_answer_chars", 8)),
            loop_ngram_sizes=ngrams,
            loop_min_repeats=int(d.get("loop_min_repeats", 3)),
            tail_ngram_size=int(d.get("tail_ngram_size", 4)),
            tail_min_tokens=int(d.get("tail_min_tokens", 24)),
            tail_window_tokens=int(d.get("tail_window_tokens", 128)),
            tail_dup_threshold=float(d.get("tail_dup_threshold", 0.5)),
            guided_hint_enabled=bool(d.get("guided_hint_enabled", False)),
            max_hints_per_step=int(d.get("max_hints_per_step", 2)),
            guided_retry_n=max(1, int(d.get("guided_retry_n", 1))),
            failure_ledger=bool(d.get("failure_ledger", True)),
            failure_ledger_filename=str(d.get("failure_ledger_filename", "sirsi_failures.jsonl")),
            curriculum_sculpt_n_tasks=max(1, int(d.get("curriculum_sculpt_n_tasks", 4))),
        )
        require(0.0 < cfg.slap_eps < 0.5, "rl.sirsi.slap_eps must be in (0, 0.5)", slap_eps=cfg.slap_eps)
        require(cfg.slap_weight >= 0.0, "rl.sirsi.slap_weight must be >= 0", slap_weight=cfg.slap_weight)
        require(cfg.anti_slop_weight >= 0.0, "rl.sirsi.anti_slop_weight must be >= 0",
                anti_slop_weight=cfg.anti_slop_weight)
        return cfg


@dataclass
class SIRSIPathology:
    tags: list[str] = field(default_factory=list)
    loop_onset_char: int | None = None
    slop_hits: list[str] = field(default_factory=list)
    slop_penalty: float = 0.0


def extract_final_answer(text: str) -> tuple[str, bool]:
    """Split final-answer prose outside `<think>...</think>` and flag unclosed `<think>`.

    Returns `(final_answer, has_unclosed_think)`. When `<think>` is unclosed, returns `("", True)`
    so anti-slop inspection never touches internal reasoning traces.
    """
    last_open = text.rfind("<think>")
    last_close = text.rfind("</think>")
    if last_open != -1 and last_open > last_close:
        return "", True
    if last_close != -1:
        return text[last_close + len("</think>") :], False
    return text, False


def _detect_loop_onset(text: str, cfg: SIRSIConfig) -> int | None:
    """Locate the earliest char offset where degenerate n-gram, tail, or char-level repetition begins."""
    if not text:
        return None
    spans = [(m.start(), m.group(0).lower()) for m in _TOKEN_SPAN_RE.finditer(text)]
    words = [w for _, w in spans]
    n_tok = len(words)
    onset: int | None = None

    # 1. Exact n-gram cycle detector (e.g., 8-gram or 12-gram repeating >= 3x).
    min_rep = max(2, int(cfg.loop_min_repeats))
    for n in cfg.loop_ngram_sizes:
        if n <= 0 or n_tok < n + min_rep - 1:
            continue
        seen: dict[tuple[str, ...], list[int]] = {}
        for i in range(n_tok - n + 1):
            gram = tuple(words[i : i + n])
            pos_list = seen.get(gram)
            if pos_list is None:
                seen[gram] = [i]
            else:
                pos_list.append(i)
                if len(pos_list) == min_rep:
                    cand = spans[pos_list[1]][0]
                    if onset is None or cand < onset:
                        onset = cand
    if onset is not None:
        return onset

    # 2. High duplicate tail fraction (catches shorter or slightly varying tail cycles).
    tn = max(2, int(cfg.tail_ngram_size))
    if n_tok >= max(int(cfg.tail_min_tokens), tn + 4):
        w_start = max(0, n_tok - int(cfg.tail_window_tokens))
        tail_words = words[w_start:]
        total_grams = len(tail_words) - tn + 1
        if total_grams >= 12:
            tail_seen: dict[tuple[str, ...], int] = {}
            first_dup_idx: int | None = None
            for idx in range(total_grams):
                gram = tuple(tail_words[idx : idx + tn])
                if gram in tail_seen:
                    if first_dup_idx is None:
                        first_dup_idx = idx
                else:
                    tail_seen[gram] = idx
            dup_frac = 1.0 - (len(tail_seen) / float(total_grams))
            if dup_frac >= float(cfg.tail_dup_threshold) and first_dup_idx is not None:
                return spans[w_start + first_dup_idx][0]

    # 3. Unspaced / character-level repetition inside long single tokens (e.g. "abcd1234"*8 or "z"*40).
    for m in _TOKEN_SPAN_RE.finditer(text):
        tok = m.group(0)
        if len(tok) < 32:
            continue
        m_char = _SINGLE_CHAR_LOOP_RE.search(tok) or _ANY_CHAR_LOOP_RE.search(tok)
        if m_char is not None:
            return m.start() + m_char.start() + 1
        # Non-overlapping 8-char substring repeating >= 4x inside one unspaced token.
        c_n = 8
        c_pos: dict[str, list[int]] = {}
        for j in range(len(tok) - c_n + 1):
            sub = tok[j : j + c_n]
            lst = c_pos.get(sub)
            if lst is None:
                c_pos[sub] = [j]
            elif j - lst[-1] >= c_n:
                lst.append(j)
                if len(lst) == 4:
                    return m.start() + lst[1]
    return None


def detect_pathologies(
    text: str,
    *,
    reward: float = 0.0,
    cfg: SIRSIConfig | dict[str, Any] | None = None,
) -> SIRSIPathology:
    """Fast, pure-Python, deterministic pathology & anti-slop voice detector."""
    scfg = SIRSIConfig.from_dict(cfg)
    raw = text or ""
    tags: list[str] = []

    loop_onset = _detect_loop_onset(raw, scfg)
    if loop_onset is not None:
        tags.append(TAG_LOOP_DEGENERATION)

    final_answer, unclosed_think = extract_final_answer(raw)
    if unclosed_think:
        tags.append(TAG_UNCLOSED_THINK)

    if (not unclosed_think) and float(reward) <= 0.0 and len(final_answer.strip()) < scfg.min_answer_chars:
        tags.append(TAG_PREMATURE_EXIT)

    # Anti-slop rubric inspects STRICTLY final-answer prose (never inside <think>).
    slop_hits: list[str] = []
    slop_penalty = 0.0
    if final_answer.strip():
        patterns = scfg.slop_patterns if scfg.slop_patterns is not None else DEFAULT_SLOP_PATTERNS
        for pat in patterns:
            m = re.search(pat, final_answer, flags=re.IGNORECASE)
            if m is not None:
                slop_hits.append(m.group(0).strip())
        if slop_hits:
            tags.append(TAG_AI_SLOP)
            cap = scfg.effective_max_slop_penalty
            if cap > 0.0:
                per_hit = max(0.0, float(scfg.slop_penalty_per_hit))
                slop_penalty = min(cap, float(len(slop_hits)) * per_hit)

    return SIRSIPathology(
        tags=tags,
        loop_onset_char=loop_onset,
        slop_hits=slop_hits,
        slop_penalty=slop_penalty,
    )


# ============================================================================ span mapping & unlikelihood


def unlikelihood_span_loss(
    token_logprobs: torch.Tensor,
    span_mask: torch.Tensor | Sequence[Any],
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Bounded token-level unlikelihood loss `-log(1 - p)` strictly over `span_mask`.

    Computes in fp32 (or fp64 for fp64 inputs) so `1.0 - eps` never rounds to `1.0` in bf16/fp16.
    Strictly finite even when `token_logprobs == 0.0`; returns exact `0.0` with clean autograd
    graph when `span_mask` has no True entries; exact zero gradient outside `span_mask`.
    """
    require(0.0 < float(eps) < 0.5, "unlikelihood_span_loss: eps must be in (0, 0.5)", eps=eps)
    acc = torch.float64 if token_logprobs.dtype == torch.float64 else torch.float32
    lp = token_logprobs.to(acc)
    mask = torch.as_tensor(span_mask, dtype=torch.bool, device=lp.device)
    require(
        tuple(mask.shape) == tuple(lp.shape),
        "unlikelihood_span_loss: span_mask shape must match token_logprobs",
        span_mask=tuple(mask.shape),
        token_logprobs=tuple(lp.shape),
    )
    safe_lp = torch.where(mask, lp, torch.full_like(lp, -1.0))
    p = torch.exp(safe_lp.clamp(max=-float(eps))).clamp(max=1.0 - float(eps))
    ul = -torch.log1p(-p)
    masked_ul = torch.where(mask, ul, torch.zeros_like(ul))
    denom = mask.to(acc).sum().clamp_min(1.0)
    return masked_ul.sum() / denom


def _decode_safe(tokenizer: Any, ids: Sequence[int]) -> str | None:
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    try:
        return str(tokenizer.decode(list(ids), skip_special_tokens=True))
    except TypeError:
        try:
            return str(tokenizer.decode(list(ids)))
        except Exception:
            return None
    except Exception:
        return None


def map_char_span_to_token_mask(
    tokenizer: Any,
    completion_ids: Sequence[int],
    completion_text: str,
    onset_char: int | None,
    *,
    fallback_tail_frac: float = 0.5,
) -> list[bool]:
    """Map `onset_char` in `completion_text` to a boolean token mask aligned with `completion_ids`.

    Prefers HF fast-tokenizer `offset_mapping`, falls back to O(log T) binary search over
    `tokenizer.decode(prefix)` (exact for `ToyTokenizer`/`ByteTokenizer` and slow HF tokenizers),
    and finally to proportional char-ratio / tail-fraction fallback. Preserves trailing EOS as
    unpenalized (`False`) when `completion_ids` ends with EOS and body tokens are masked.
    """
    ids = [int(t) for t in completion_ids]
    n = len(ids)
    if n == 0:
        return []

    eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    has_trailing_eos = (
        eos_id is not None
        and n > 1
        and (ids[-1] in eos_id if isinstance(eos_id, (list, tuple, set)) else ids[-1] == int(eos_id))
    )
    body_n = n - 1 if has_trailing_eos else n

    def _finish(start_tok: int) -> list[bool]:
        st = max(0, min(start_tok, body_n - 1))
        mask = [i >= st for i in range(body_n)]
        if has_trailing_eos:
            mask.append(False)
        return mask

    if onset_char is None:
        frac = max(0.05, min(1.0, float(fallback_tail_frac)))
        start_tok = max(0, min(body_n - 1, int(body_n * (1.0 - frac))))
        return _finish(start_tok)

    oc = max(0, int(onset_char))
    if oc == 0:
        return _finish(0)
    if completion_text and oc >= len(completion_text):
        return _finish(body_n - 1)

    # Strategy 1: HF fast tokenizer offset_mapping.
    if tokenizer is not None and callable(tokenizer) and completion_text:
        try:
            enc = tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True)
            if isinstance(enc, dict) or (hasattr(enc, "keys") and "offset_mapping" in enc):
                offsets = enc.get("offset_mapping")
                enc_ids = enc.get("input_ids")
                if offsets is not None and enc_ids is not None:
                    enc_list = [int(t) for t in enc_ids]
                    if enc_list == ids[: len(enc_list)] and len(enc_list) in (n, body_n):
                        mask = [bool(end > oc or (start == end and start >= oc)) for start, end in offsets]
                        if len(mask) < n:
                            mask.extend([False] * (n - len(mask)))
                        if has_trailing_eos and any(mask[:body_n]):
                            mask[-1] = False
                        if any(mask):
                            return mask
        except Exception:
            pass

    # Strategy 2: O(log T) binary search on prefix decodes (exact for ToyTokenizer / slow tokenizers).
    body_ids = ids[:body_n]
    full_dec = _decode_safe(tokenizer, body_ids)
    if full_dec:
        lo, hi = 0, body_n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            pref = _decode_safe(tokenizer, body_ids[: mid + 1])
            if pref is not None and len(pref) > oc:
                hi = mid
            else:
                lo = mid + 1
        return _finish(lo)

    # Strategy 3: Proportional char-ratio fallback (or tail fallback when completion_text is empty).
    if completion_text:
        ratio = oc / float(max(len(completion_text), 1))
        return _finish(round(ratio * body_n))
    frac = max(0.05, min(1.0, float(fallback_tail_frac)))
    return _finish(int(body_n * (1.0 - frac)))


# ============================================================================ surgical param routing


def resolve_slap_target_params(
    named_trainable: dict[str, nn.Parameter],
    target_groups: Sequence[str] | None,
) -> list[nn.Parameter] | None:
    """Resolve `slap_target_groups` into the exact parameter list for surgical `slap_neg` grads.

    Returns `None` when `target_groups` is empty/None (meaning standard full-trainable backward).
    """
    if not target_groups:
        return None
    canon = {_GROUP_ALIASES.get(str(g).strip(), str(g).strip()) for g in target_groups if str(g).strip()}
    if not canon:
        return None
    return [p for name, p in named_trainable.items() if p.requires_grad and param_group_of(name) in canon]


# ============================================================================ failure ledger & supervisor prompts


def _render_prompt_preview(prompt: Any, max_chars: int = 1600) -> str:
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, Sequence):
        parts = []
        for m in prompt:
            if isinstance(m, dict):
                parts.append(f"[{m.get('role', 'user')}] {m.get('content', '')}")
            else:
                parts.append(str(m))
        text = "\n".join(parts)
    else:
        text = str(prompt)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _truncate_around_tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n... [middle truncated] ...\n" + text[-tail:]


class FailureLedger:
    """Append-only JSONL ledger for `all_fail` and pathology-tagged SI-RSI rollouts."""

    def __init__(self, path: str):
        self.path = path
        self.n_written = 0

    def record(
        self,
        *,
        step: int,
        task_id: str,
        tags: Sequence[str],
        reward: float,
        loop_onset_char: int | None,
        prompt: Any,
        bad_completion: str,
        **extra: Any,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step": int(step),
            "task_id": str(task_id),
            "tags": [str(t) for t in tags],
            "reward": float(reward),
            "loop_onset_char": None if loop_onset_char is None else int(loop_onset_char),
            "prompt": prompt,
            "bad_completion": str(bad_completion),
            **extra,
        }
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n_written += 1
        return row

    def read_recent(self, limit: int = 32) -> list[dict[str, Any]]:
        if not os.path.exists(self.path) or limit <= 0:
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]


def build_hint_messages(
    prompt: Any,
    bad_completion: str,
    tags: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build Supervisor chat messages requesting a 1-sentence supervisory nudge before divergence."""
    tag_str = ", ".join(str(t) for t in tags) if tags else "INCORRECT_SOLUTION"
    sys_msg = (
        "You are a concise mathematical and coding supervisor guiding a student model during "
        "reinforcement learning. Provide EXACTLY ONE crisp, direct sentence pointing out the "
        "student's point of divergence or pathology and the key conceptual pivot needed to recover. "
        "Do NOT give away the final answer, do NOT write full code, and avoid filler or clichés."
    )
    user_msg = (
        f"Task:\n{_render_prompt_preview(prompt)}\n\n"
        f"Detected failure tags: {tag_str}\n\n"
        f"Student attempt:\n{_truncate_around_tail(bad_completion)}\n\n"
        "Respond with exactly 1 sentence giving a surgical hint so the student can solve this cleanly."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_curriculum_sculpt_messages(
    failure_records: Sequence[dict[str, Any]],
    n_tasks: int = 4,
) -> list[dict[str, str]]:
    """Build Supervisor chat messages to synthesize targeted, verifiable micro-tasks."""
    n_tasks = max(1, int(n_tasks))
    tag_counts: dict[str, int] = {}
    examples: list[str] = []
    for idx, rec in enumerate(list(failure_records)[-8:]):
        rec_tags = [str(t) for t in (rec.get("tags") or ())]
        for tg in rec_tags:
            tag_counts[tg] = tag_counts.get(tg, 0) + 1
        examples.append(
            f"- Failure {idx + 1} (task_id={rec.get('task_id', '?')}, tags={rec_tags}, "
            f"reward={rec.get('reward', 0.0)}): {_render_prompt_preview(rec.get('prompt', ''), 300)}"
        )
    summary = ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items())) or "general reasoning failures"
    sys_msg = (
        "You are a curriculum sculptor designing novel, original, deterministically verifiable "
        "training micro-tasks for a looped reasoning model. Never reproduce problems from public "
        "benchmarks (AIME, HMMT, GPQA, HLE, LiveCodeBench, KernelBench, ARC-AGI, MRCR). "
        "Return ONLY a valid JSON array of task objects."
    )
    user_msg = (
        f"Observed student pathology summary: {summary}\n\n"
        f"Recent failure excerpts:\n" + ("\n".join(examples) if examples else "- (none)") + "\n\n"
        f"Synthesize exactly {n_tasks} original, self-contained, verifiable micro-tasks targeting "
        "these failure modes. Each object in the JSON array MUST contain:\n"
        '  - "id": short unique string id\n'
        '  - "prompt": clear problem statement asking for a direct final answer (e.g. in \\boxed{})\n'
        '  - "answer": exact verifiable ground-truth answer string\n'
        '  - "verifier": "scorer:math" (or "regex" with a "pattern" field)\n'
        '  - "target_tags": list of addressed pathology tags\n'
        "Return ONLY the JSON array with no surrounding prose."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def _extract_json_candidates(raw_text: str) -> Any:
    text = (raw_text or "").strip()
    if not text:
        return []
    m = _CODE_FENCE_RE.search(text)
    if m is not None:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        lo, hi = text.find(open_ch), text.rfind(close_ch)
        if lo != -1 and hi > lo:
            try:
                return json.loads(text[lo : hi + 1])
            except json.JSONDecodeError:
                pass
    # JSONL fallback
    items: list[Any] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return items


def parse_and_decontaminate_sculpted_tasks(
    raw_text: str,
    decontaminator: Any = None,
    *,
    source: str = "sirsi_sculpt",
) -> list[dict[str, Any]]:
    """Parse Supervisor-sculpted tasks and enforce 13-gram decontamination against held-out evals."""
    parsed = _extract_json_candidates(raw_text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tasks"), list):
            raw_list = parsed["tasks"]
        elif "prompt" in parsed:
            raw_list = [parsed]
        else:
            raw_list = []
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        raw_list = []

    clean: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for i, obj in enumerate(raw_list):
        if not isinstance(obj, dict):
            continue
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt = prompt.strip()
        if prompt in seen_prompts:
            continue

        ans_raw = obj.get("answer")
        answer = (
            str(ans_raw).strip()
            if (isinstance(ans_raw, (str, int, float)) and not isinstance(ans_raw, bool))
            else ""
        )
        pattern = obj.get("pattern")
        pattern_str = pattern.strip() if isinstance(pattern, str) else ""
        if not answer and not pattern_str:
            continue

        verifier = str(obj.get("verifier") or ("scorer:math" if answer else "regex")).strip()
        if not (verifier.startswith("scorer:") or verifier == "regex"):
            log.warning("[sirsi] dropping sculpted task with disallowed verifier %r", verifier)
            continue
        if verifier == "regex" and not pattern_str:
            continue

        if decontaminator is not None:
            combined = f"{prompt}\n{answer}" if answer else prompt
            hit, group = decontaminator.check(combined, source=source)
            if hit:
                log.warning(
                    "[sirsi] decontaminator blocked sculpted task %r (matched eval group %r)",
                    obj.get("id", f"{source}-{i}"),
                    group,
                )
                continue

        seen_prompts.add(prompt)
        task_dict: dict[str, Any] = {
            "id": str(obj.get("id") or f"{source}-{i}"),
            "prompt": prompt,
            "verifier": verifier,
        }
        if answer:
            task_dict["answer"] = answer
        if pattern_str:
            task_dict["pattern"] = pattern_str
        if isinstance(obj.get("target_tags"), list):
            task_dict["target_tags"] = [str(t) for t in obj["target_tags"]]
        clean.append(task_dict)
    return clean
