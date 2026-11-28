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

import ast
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
from ..eval.code_exec import extract_code
from ..eval.scoring import clean_answer, last_boxed, normalize_math, parse_int_answer, parse_number
from ..model.experts import param_group_of

log = logging.getLogger("qloop")

TAG_LOOP_DEGENERATION = "LOOP_DEGENERATION"
TAG_UNCLOSED_THINK = "UNCLOSED_THINK"
TAG_PREMATURE_EXIT = "PREMATURE_EXIT"
TAG_AI_SLOP = "AI_SLOP"

REGRET_NEWLY_HURT = "NEWLY_HURT"
REGRET_UNSOLVED_CHRONIC = "UNSOLVED_CHRONIC"
REGRET_UNSOLVED_NOVEL = "UNSOLVED_NOVEL"
REGRET_PINNED_PASS = "PINNED_PASS"

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
    regret_enabled: bool = True
    regret_drop_eps: float = 0.05
    regret_hurt_gamma: float = 2.0
    regret_max_weight: float = 3.0
    regret_chronic_weight: float = 0.5
    hurt_replay_prob: float = 0.15
    pinned_replay_prob: float = 0.10
    atomic_checks_enabled: bool = False
    atomic_check_alpha: float = 0.25
    atomic_check_min_spread: float = 0.08

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
            regret_enabled=bool(d.get("regret_enabled", True)),
            regret_drop_eps=float(d.get("regret_drop_eps", 0.05)),
            regret_hurt_gamma=float(d.get("regret_hurt_gamma", 2.0)),
            regret_max_weight=float(d.get("regret_max_weight", 3.0)),
            regret_chronic_weight=float(d.get("regret_chronic_weight", 0.5)),
            hurt_replay_prob=float(d.get("hurt_replay_prob", 0.15)),
            pinned_replay_prob=float(d.get("pinned_replay_prob", 0.10)),
            atomic_checks_enabled=bool(d.get("atomic_checks_enabled", False)),
            atomic_check_alpha=float(d.get("atomic_check_alpha", 0.25)),
            atomic_check_min_spread=float(d.get("atomic_check_min_spread", 0.08)),
        )
        require(0.0 < cfg.slap_eps < 0.5, "rl.sirsi.slap_eps must be in (0, 0.5)", slap_eps=cfg.slap_eps)
        require(cfg.slap_weight >= 0.0, "rl.sirsi.slap_weight must be >= 0", slap_weight=cfg.slap_weight)
        require(cfg.anti_slop_weight >= 0.0, "rl.sirsi.anti_slop_weight must be >= 0",
                anti_slop_weight=cfg.anti_slop_weight)
        require(cfg.regret_drop_eps >= 0.0, "rl.sirsi.regret_drop_eps must be >= 0",
                regret_drop_eps=cfg.regret_drop_eps)
        require(cfg.regret_hurt_gamma >= 0.0, "rl.sirsi.regret_hurt_gamma must be >= 0",
                regret_hurt_gamma=cfg.regret_hurt_gamma)
        require(cfg.regret_max_weight >= 1.0, "rl.sirsi.regret_max_weight must be >= 1",
                regret_max_weight=cfg.regret_max_weight)
        require(cfg.regret_chronic_weight >= 0.0, "rl.sirsi.regret_chronic_weight must be >= 0",
                regret_chronic_weight=cfg.regret_chronic_weight)
        require(0.0 <= cfg.hurt_replay_prob <= 1.0, "rl.sirsi.hurt_replay_prob must be in [0, 1]",
                hurt_replay_prob=cfg.hurt_replay_prob)
        require(0.0 <= cfg.pinned_replay_prob <= 1.0, "rl.sirsi.pinned_replay_prob must be in [0, 1]",
                pinned_replay_prob=cfg.pinned_replay_prob)
        require(0.0 <= cfg.atomic_check_alpha < 1.0, "rl.sirsi.atomic_check_alpha must be in [0, 1)",
                atomic_check_alpha=cfg.atomic_check_alpha)
        require(cfg.atomic_check_min_spread >= 0.0, "rl.sirsi.atomic_check_min_spread must be >= 0",
                atomic_check_min_spread=cfg.atomic_check_min_spread)
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
    seq_weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Bounded token-level unlikelihood loss `-log(1 - p)` strictly over `span_mask`.

    Computes in fp32 (or fp64 for fp64 inputs) so `1.0 - eps` never rounds to `1.0` in bf16/fp16.
    Strictly finite even when `token_logprobs == 0.0`; returns exact `0.0` with clean autograd
    graph when `span_mask` has no True entries; exact zero gradient outside `span_mask`.
    When `seq_weights` (`[B]` for 2D inputs or scalar/`[T]` for 1D inputs) is provided, scales
    each sequence's masked unlikelihood contribution proportionally while normalizing by total
    masked token count (`seq_weights=None` gives exact legacy parity).
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
    if seq_weights is not None:
        w = torch.as_tensor(seq_weights, dtype=acc, device=lp.device)
        require(bool(torch.all(w >= 0.0)), "unlikelihood_span_loss: seq_weights must be >= 0")
        if lp.ndim == 2:
            if w.ndim == 1 and w.shape[0] == lp.shape[0]:
                w_exp = w.unsqueeze(1)
            elif w.ndim == 2 and tuple(w.shape) == (lp.shape[0], 1):
                w_exp = w
            else:
                require(
                    False,
                    "unlikelihood_span_loss: seq_weights shape must be [B] or [B, 1] for 2D token_logprobs",
                    seq_weights=tuple(w.shape),
                    token_logprobs=tuple(lp.shape),
                )
        elif lp.ndim == 1:
            if w.ndim == 0 or (w.ndim == 1 and w.numel() == 1):
                w_exp = w.reshape(1)
            elif w.ndim == 1 and w.shape[0] == lp.shape[0]:
                w_exp = w
            else:
                require(
                    False,
                    "unlikelihood_span_loss: seq_weights shape must be scalar, [1], or [T] for 1D token_logprobs",
                    seq_weights=tuple(w.shape),
                    token_logprobs=tuple(lp.shape),
                )
        else:
            require(
                w.ndim == 1 and w.shape[0] == lp.shape[0],
                "unlikelihood_span_loss: seq_weights shape must match batch dim",
                seq_weights=tuple(w.shape),
                token_logprobs=tuple(lp.shape),
            )
            w_exp = w.reshape((lp.shape[0],) + (1,) * (lp.ndim - 1))
        ul = ul * w_exp
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


# ============================================================================ G2 directional regret tracking


@dataclass
class TaskMasteryState:
    seen: int = 0
    ever_passed: bool = False
    last_pass_rate: float = 0.0
    best_pass_rate: float = 0.0
    consec_fails: int = 0
    best_check_bits: tuple[int, ...] = ()
    last_check_bits: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": int(self.seen),
            "ever_passed": bool(self.ever_passed),
            "last_pass_rate": float(self.last_pass_rate),
            "best_pass_rate": float(self.best_pass_rate),
            "consec_fails": int(self.consec_fails),
            "best_check_bits": [int(b) for b in self.best_check_bits],
            "last_check_bits": [int(b) for b in self.last_check_bits],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskMasteryState:
        return cls(
            seen=int(d.get("seen", 0)),
            ever_passed=bool(d.get("ever_passed", False)),
            last_pass_rate=float(d.get("last_pass_rate", 0.0)),
            best_pass_rate=float(d.get("best_pass_rate", 0.0)),
            consec_fails=int(d.get("consec_fails", 0)),
            best_check_bits=tuple(int(b) for b in (d.get("best_check_bits") or ())),
            last_check_bits=tuple(int(b) for b in (d.get("last_check_bits") or ())),
        )


@dataclass(frozen=True)
class RegretDecision:
    kind: str
    weight: float
    prev_pass_rate: float
    regressed_checks: tuple[int | str, ...] = ()


class TaskRegretTracker:
    """Tracks per-task mastery & check-bit history to classify directional regret (GUARD G2)."""

    def __init__(self, cfg: SIRSIConfig | dict[str, Any] | None = None):
        self.cfg = SIRSIConfig.from_dict(cfg)
        self.states: dict[str, TaskMasteryState] = {}

    def classify_and_update(
        self,
        task_id: str,
        pass_rate: float,
        check_bits: Sequence[int] = (),
        *,
        check_names: Sequence[str] = (),
    ) -> RegretDecision:
        tid = str(task_id)
        pr = max(0.0, min(1.0, float(pass_rate)))
        bits = tuple(1 if int(b) else 0 for b in check_bits)
        st = self.states.get(tid)
        if st is None:
            st = TaskMasteryState()
        prev_pr = float(st.last_pass_rate)

        regressed_idx: list[int] = []
        check_hurt_frac = 0.0
        if bits and st.best_check_bits:
            n_cmp = min(len(bits), len(st.best_check_bits))
            prev_passed = sum(1 for i in range(n_cmp) if st.best_check_bits[i] == 1)
            for i in range(n_cmp):
                if st.best_check_bits[i] == 1 and bits[i] == 0:
                    regressed_idx.append(i)
            if prev_passed > 0 and regressed_idx:
                check_hurt_frac = float(len(regressed_idx)) / float(prev_passed)

        regressed_checks: tuple[int | str, ...] = tuple(
            str(check_names[i]) if i < len(check_names) else i for i in regressed_idx
        )

        drop_eps = float(self.cfg.regret_drop_eps)
        hurt_gamma = float(self.cfg.regret_hurt_gamma)
        max_w = float(self.cfg.regret_max_weight)
        chronic_w = float(self.cfg.regret_chronic_weight)

        is_hurt = (
            (st.ever_passed and pr < st.last_pass_rate - drop_eps)
            or (st.best_pass_rate >= 0.5 and pr == 0.0)
            or bool(regressed_idx)
        )
        new_consec_fails = 0 if pr > 0.0 else (st.consec_fails + 1)

        if is_hurt:
            drop_mag = max(0.0, st.best_pass_rate - pr)
            weight = min(max_w, 1.0 + hurt_gamma * max(drop_mag, check_hurt_frac))
            kind = REGRET_NEWLY_HURT
        elif pr > 0.0:
            kind = REGRET_PINNED_PASS
            weight = 1.0
        elif (not st.ever_passed) and new_consec_fails >= 2:
            kind = REGRET_UNSOLVED_CHRONIC
            weight = chronic_w
        elif not st.ever_passed:
            kind = REGRET_UNSOLVED_NOVEL
            weight = 1.0
        else:
            # Previously passed (<0.5) and already at 0.0 previously; avoid pinning a 0-pass step.
            kind = REGRET_UNSOLVED_CHRONIC if new_consec_fails >= 2 else REGRET_UNSOLVED_NOVEL
            weight = chronic_w if kind == REGRET_UNSOLVED_CHRONIC else 1.0

        st.seen += 1
        st.ever_passed = st.ever_passed or (pr > 0.0)
        st.last_pass_rate = pr
        st.best_pass_rate = max(st.best_pass_rate, pr)
        st.consec_fails = new_consec_fails
        if bits:
            st.last_check_bits = bits
            if len(st.best_check_bits) == len(bits):
                st.best_check_bits = tuple(max(a, b) for a, b in zip(st.best_check_bits, bits))
            else:
                st.best_check_bits = bits
        self.states[tid] = st

        return RegretDecision(
            kind=kind,
            weight=float(weight),
            prev_pass_rate=prev_pr,
            regressed_checks=regressed_checks,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"states": {tid: st.to_dict() for tid, st in self.states.items()}}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.states = {
            str(tid): TaskMasteryState.from_dict(val)
            for tid, val in (sd.get("states") or {}).items()
            if isinstance(val, dict)
        }


# ============================================================================ G3 Tier-A atomic self-checks


@dataclass(frozen=True)
class AtomicCheckResult:
    bits: tuple[int, ...]
    names: tuple[str, ...]
    credit: float
    has_exec_pass: bool


_KB_RUNTIME_ERR_MARKERS: tuple[str, ...] = (
    "cuda",
    "oom",
    "out of memory",
    "shape",
    "runtimeerror",
    "runtime",
    "traceback",
    "segfault",
    "crashed",
    "timeout",
    "nan",
    "inf",
    "typeerror",
    "attributeerror",
    "indexerror",
    "keyerror",
    "nameerror",
    "valueerror",
    "zerodivision",
    "illegal memory",
    "assert",
    "type mismatch",
    "output length mismatch",
)

_AIME_PROMPT_RE = re.compile(
    r"\baime\b|\b0\s*(?:to|through|-|\.\.)\s*999\b|between\s+0\s+and\s+999|"
    r"three[- ]digit\s+(?:positive\s+)?integer|remainder\s+when.*divided\s+by\s+1000|modulo\s+1000",
    re.IGNORECASE,
)
_PROB_PROMPT_RE = re.compile(r"\bprobabilit(?:y|ies)\b|\bchance\s+that\b", re.IGNORECASE)
_NONNEG_INT_PROMPT_RE = re.compile(
    r"\b(?:positive\s+integer|non[- ]?negative\s+integer|how\s+many|"
    r"number\s+of\s+(?:ways|pairs|tuples|triples|subsets|solutions|integers))\b",
    re.IGNORECASE,
)
_INT_PROMPT_RE = re.compile(r"\b(?:find\s+the\s+integer|is\s+an\s+integer|integer\s+value)\b", re.IGNORECASE)
_MATH_BAD_WORDS_RE = re.compile(
    r"\b(?:unknown|none|null|error|sorry|cannot|undefined|answer|todo|placeholder|wrong|invalid)\b",
    re.IGNORECASE,
)
_LATEX_CMD_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|pi|infty|sin|cos|tan|cot|sec|csc|log|ln|exp|left|right|"
    r"cdot|times|pm|mp|cup|cap|text|mathrm|mathbf|textbf|circ|le|ge|leq|geq)\b"
)
_MATH_CHARS_RE = re.compile(r"^[0-9a-zA-Z+\-*/^_()\[\]{}.,=<>!|\\: ]+$")


def _flatten_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence):
        parts: list[str] = []
        for m in prompt:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(prompt or "")


def _check_kernelbench_ast(src: str | None) -> bool:
    """AST verify `ModelNew` is defined and no direct reference `Model` import/bypass is present."""
    if not src or not src.strip():
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    has_model_new = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            has_model_new = True
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Model":
                    return False
                if isinstance(base, ast.Attribute) and base.attr == "Model":
                    return False
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in ("reference", "kb_reference"):
                    return False
        elif isinstance(node, ast.ImportFrom):
            mod_root = (node.module or "").split(".", 1)[0]
            if mod_root in ("reference", "kb_reference"):
                return False
            for alias in node.names:
                if alias.name == "Model":
                    return False
        elif isinstance(node, ast.Attribute):
            if node.attr in ("spec_from_file_location", "exec_module"):
                return False
            if (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                return False
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("__import__", "eval", "exec")
        ):
            return False
    return has_model_new


def _balanced_delimiters(s: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _is_valid_math_expr(raw_cand: str) -> bool:
    cleaned = clean_answer(raw_cand)
    norm = normalize_math(raw_cand)
    if not cleaned or not norm or len(norm) > 128:
        return False
    if parse_number(norm) is not None:
        return True
    if _MATH_BAD_WORDS_RE.search(cleaned):
        return False
    if not _balanced_delimiters(norm):
        return False
    stripped = _LATEX_CMD_RE.sub("", cleaned)
    if not _MATH_CHARS_RE.match(stripped):
        return False
    # Reject multi-word natural language sentences masquerading as answers.
    long_words = re.findall(r"[A-Za-z]{4,}", stripped)
    if len(long_words) >= 2:
        return False
    if len(long_words) == 1 and not re.search(r"[\d+\-*/^=]", stripped):
        return False
    return bool(re.search(r"[\d]|pi|oo|sqrt|frac|[a-zA-Z]", norm))


def _check_math_domain(verifier: str, prompt_str: str, cand: str | None) -> bool:
    if not cand or not normalize_math(cand):
        return False
    v_low = verifier.lower()
    if "aime" in v_low or _AIME_PROMPT_RE.search(prompt_str):
        return parse_int_answer(cand, lo=0, hi=999) is not None
    if _PROB_PROMPT_RE.search(prompt_str):
        val = parse_number(cand)
        return val is not None and 0 <= val <= 1
    if _NONNEG_INT_PROMPT_RE.search(prompt_str):
        val = parse_number(cand)
        return val is not None and val.denominator == 1 and val >= 0
    if _INT_PROMPT_RE.search(prompt_str):
        val = parse_number(cand)
        return val is not None and val.denominator == 1
    return _is_valid_math_expr(cand)


def _check_http_no_test_tamper(raw_comp: str, meta_d: dict[str, Any], exec_present: bool) -> bool:
    if "no_test_tamper" in meta_d:
        return bool(meta_d["no_test_tamper"])
    if "test_tampered" in meta_d:
        return not bool(meta_d["test_tampered"])
    if not exec_present:
        return False
    low = raw_comp.lower()
    if re.search(r"^\+\+\+\s+[ab]/.*(?:test_|/tests?/)", raw_comp, flags=re.MULTILINE):
        return False
    return not ("pytest.skip(" in low or "unittest.skip(" in low)


def evaluate_tier_a_checks(
    verifier: str,
    prompt_text: Any,
    completion_text: str,
    reward: float,
    meta: dict[str, Any] | None,
    pathology: SIRSIPathology | None,
    *,
    alpha_check: float = 0.25,
) -> AtomicCheckResult:
    """Deterministic Tier-A atomic self-checks with strict anti-hacking bounds (GUARD G3).

    Invariants:
      - Syntactic check weights sum to <= 25% of total weight (<= 0.0625 at alpha_check=0.25).
      - `TAG_LOOP_DEGENERATION` or `TAG_UNCLOSED_THINK` vetoes credit strictly to 0.0.
      - Passing rollouts (`reward >= 1.0`) receive 0.0 additional credit so oracle pass strictly dominates.
    """
    v = str(verifier or "").strip().lower()
    meta_d = meta if isinstance(meta, dict) else {}
    tags = set(pathology.tags) if pathology is not None else set()
    raw_comp = str(completion_text or "")
    final_ans, unclosed_think = extract_final_answer(raw_comp)
    alpha = max(0.0, float(alpha_check))
    vetoed = (TAG_LOOP_DEGENERATION in tags) or (TAG_UNCLOSED_THINK in tags)

    if "kernelbench" in v:
        src = extract_code(raw_comp, "python", must_contain="ModelNew") or extract_code(raw_comp, "python")
        c_extract = int(bool(src and src.strip()))
        c_no_cheat = int(c_extract and _check_kernelbench_ast(src))
        compiled = bool(meta_d.get("compiled"))
        correct = bool(meta_d.get("correct"))
        err_low = str(meta_d.get("error") or "").lower()
        no_rt_err = int(
            compiled
            and (
                correct
                or (not any(tok in err_low for tok in _KB_RUNTIME_ERR_MARKERS))
            )
        )
        speedup = float(meta_d.get("speedup") or 0.0)
        fast_1 = int(correct and speedup > 1.05)
        checks: list[tuple[str, float, bool, int]] = [
            ("c_extractable_code", 0.08, False, c_extract),
            ("c_no_cheat_ast", 0.12, True, c_no_cheat),
            ("c_compiles", 0.25, True, int(compiled)),
            ("c_no_runtime_err", 0.20, True, no_rt_err),
            ("c_correct", 0.25, True, int(correct)),
            ("c_fast_1", 0.10, True, fast_1),
        ]
    elif "math" in v or "aime" in v:
        prompt_str = _flatten_prompt_text(prompt_text)
        c_think_closed = int(bool(raw_comp.strip()) and (not unclosed_think) and (TAG_UNCLOSED_THINK not in tags))
        c_no_loop = int(bool(raw_comp.strip()) and (TAG_LOOP_DEGENERATION not in tags))
        boxed = last_boxed(final_ans) if not unclosed_think else None
        c_boxed = int(boxed is not None and bool(clean_answer(boxed)))
        cand = boxed if c_boxed else (str(meta_d["pred"]) if meta_d.get("pred") is not None else None)
        c_domain = int((not unclosed_think) and _check_math_domain(v, prompt_str, cand))
        checks = [
            ("c_think_closed", 0.10, False, c_think_closed),
            ("c_no_loop", 0.10, False, c_no_loop),
            ("c_valid_boxed_format", 0.05, False, c_boxed),
            ("c_answer_type_domain", 0.75, True, c_domain),
        ]
    else:
        has_diff = ("--- " in raw_comp and "+++ " in raw_comp) or ("@@ " in raw_comp) or ("diff --git" in raw_comp)
        has_tool = "<tool_call>" in raw_comp or bool(meta_d.get("valid_tool_or_diff"))
        c_valid_fmt = int(has_tool or has_diff or (extract_code(raw_comp) is not None))
        total_tests = int(meta_d.get("total") or 0)
        passed_tests = int(meta_d.get("passed") or 0)
        err_s = str(meta_d.get("error") or "")
        applies = bool(
            meta_d.get("applies_cleanly")
            or meta_d.get("patch_applied")
            or meta_d.get("compiled")
            or (total_tests > 0 and "SyntaxError" not in err_s)
        )
        exec_present = applies or any(
            k in meta_d
            for k in (
                "existing_tests_pass",
                "target_tests_pass",
                "passed",
                "applies_cleanly",
                "patch_applied",
            )
        )
        no_tamper = _check_http_no_test_tamper(raw_comp, meta_d, exec_present)
        existing_pass = bool(meta_d.get("existing_tests_pass") or passed_tests > 0)
        target_pass = bool(
            meta_d.get("target_tests_pass")
            or (total_tests > 0 and passed_tests >= total_tests)
            or float(reward) >= 1.0
        )
        checks = [
            ("c_valid_tool_or_diff", 0.10, False, c_valid_fmt),
            ("c_applies_cleanly", 0.25, True, int(applies)),
            ("c_no_test_tamper", 0.15, True, int(no_tamper)),
            ("c_existing_tests_pass", 0.25, True, int(existing_pass)),
            ("c_target_tests_pass", 0.25, True, int(target_pass)),
        ]

    names = tuple(name for name, _, _, _ in checks)
    bits = tuple(int(bit) for _, _, _, bit in checks)
    has_exec_pass = (not vetoed) and any(is_exec and bool(bit) for _, _, is_exec, bit in checks)

    if vetoed or float(reward) >= 1.0 or alpha <= 0.0:
        credit = 0.0
    else:
        weighted = sum(w * float(bit) for _, w, _, bit in checks)
        credit = min(alpha, alpha * weighted)

    return AtomicCheckResult(bits=bits, names=names, credit=float(credit), has_exec_pass=bool(has_exec_pass))


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
    *,
    regret_kind: str | None = None,
    prev_pass_rate: float | None = None,
) -> list[dict[str, str]]:
    """Build Supervisor chat messages requesting a 1-sentence supervisory nudge before divergence."""
    tag_str = ", ".join(str(t) for t in tags) if tags else "INCORRECT_SOLUTION"
    sys_msg = (
        "You are a concise mathematical and coding supervisor guiding a student model during "
        "reinforcement learning. Provide EXACTLY ONE crisp, direct sentence pointing out the "
        "student's point of divergence or pathology and the key conceptual pivot needed to recover. "
        "Do NOT give away the final answer, do NOT write full code, and avoid filler or clichés."
    )
    regret_block = ""
    if regret_kind == REGRET_NEWLY_HURT:
        prev_s = f" (previous pass_rate={float(prev_pass_rate):.2f})" if prev_pass_rate is not None else ""
        regret_block = (
            f"Regression status: {REGRET_NEWLY_HURT}{prev_s} — the student previously succeeded or passed "
            "verified sub-checks on this task and has now regressed. Focus the hint on restoring the broken step.\n\n"
        )
    user_msg = (
        f"Task:\n{_render_prompt_preview(prompt)}\n\n"
        f"{regret_block}"
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
    recs = list(failure_records)
    has_regret = any(isinstance(r, dict) and r.get("regret_kind") for r in recs)
    if has_regret:
        hurt_recs = [r for r in recs if isinstance(r, dict) and r.get("regret_kind") == REGRET_NEWLY_HURT][-5:]
        hurt_ids = {id(r) for r in hurt_recs}
        novel_recs = [
            r for r in recs
            if isinstance(r, dict) and id(r) not in hurt_ids and r.get("regret_kind") == REGRET_UNSOLVED_NOVEL
        ]
        other_recs = [
            r for r in recs
            if isinstance(r, dict) and id(r) not in hurt_ids and r.get("regret_kind") != REGRET_UNSOLVED_NOVEL
        ]
        slots_left = max(0, 8 - len(hurt_recs))
        tail_novel = novel_recs[-slots_left:] if slots_left > 0 else []
        rem_slots = max(0, slots_left - len(tail_novel))
        tail_other = other_recs[-rem_slots:] if rem_slots > 0 else []
        selected = hurt_recs + tail_novel + tail_other
    else:
        selected = recs[-8:]

    tag_counts: dict[str, int] = {}
    examples: list[str] = []
    for idx, rec in enumerate(selected):
        rec_tags = [str(t) for t in (rec.get("tags") or ())]
        for tg in rec_tags:
            tag_counts[tg] = tag_counts.get(tg, 0) + 1
        rk = rec.get("regret_kind")
        rk_str = f", regret={rk}" if rk else ""
        examples.append(
            f"- Failure {idx + 1} (task_id={rec.get('task_id', '?')}, tags={rec_tags}, "
            f"reward={rec.get('reward', 0.0)}{rk_str}): {_render_prompt_preview(rec.get('prompt', ''), 300)}"
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
