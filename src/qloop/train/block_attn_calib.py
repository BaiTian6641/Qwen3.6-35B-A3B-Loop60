"""Block-Attention Logit-Distillation Calibrator (`YELLOW_B -> GREEN` for Kill-Gate `K0`).

Calibrates attention projections (`q/k/v/o`, `q_norm/k_norm`), `GDN-2-1M` adapters, `D-LSA`
indexers, `shared_expert`, and layer norms so that independently prefilled Partition-B KV-Mem
blobs (`Pre-RoPE un-rotate + EPIC sink-strip + Block-64 Hadamard INT4 + virtual RoPE mount`,
with `GDN-2` recurrent states strictly isolated to Partition A) match the Full Contiguous
Prefill Oracle logits on multi-session suffixes.

Depth-agnostic by construction:
  * Option A (`loop.enabled: false`, `40L` `d=1` pruned `GDN-2-1M` + `D-LSA 1.0x`)
  * Option B (`loop.enabled: true`, `64L` `d=2` / `88L` `d=3` with `SharedKVView` & depth sampling)

Key invariants:
  * 100% frozen MoE experts (`sha256_frozen_experts: true`).
  * Zero-leakage episode construction: clean-room corpus stream (`holdout_mod=100`, 13-gram
    decontam) mixed with out-of-eval-seed synthetic multi-session QA / `temporal_update` /
    code-symbol episodes (`synthetic_seed_offset >= 10000`, strictly disjoint from `K0` `seed=42`).
  * Memory-efficient chunked FP32 KL + CE (`_ChunkedLogitsKLCE`): computes
    `kl_weight * T^2 * KL(softmax(oracle/T) || softmax(student/T)) + ce_weight * CE(student, y)`
    in `ce_chunk_size` token slices so peak extra FP32 VRAM stays tiny on `248,320`-vocab models
    (fits comfortably on 96 GB GDDR7 ECC VRAM).
  * Periodic contiguous LM anchor steps (`contiguous_lm_every`) to prevent general-LM / OOD drift.
  * Live Kill-Gate `K0` evaluation (`run_k0_suite`) at step 0, every `eval_every`, and final step.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import math
import os
import random
import signal
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import torch
import torch.nn.functional as F

from ..debug import cuda_mem_brief, require, stage
from ..eval.kill_gate_k0 import (
    HFCheckpointK0Evaluator,
    K0Config,
    K0Example,
    K0Report,
    _clone_linear_states_into,
    _resolve_text_configs,
    build_corrected_partition_b_kv,
    build_default_smoke_examples,
    extract_layer_kv_pairs,
    inject_layer_kv_pairs,
    load_examples_from_file,
    run_k0_suite,
)
from ..model._cache_compat import cache_kwarg_name, extract_output_cache
from ..utils.tf32 import configure_tf32
from .checkpoint import CheckpointManager, deserialize_py_rng_state, serialize_py_rng_state
from .loss import lm_loss
from .optim import FusedStepOptimizer
from .stack import Stack, build_stack
from .trainer import MetricsLogger

log = logging.getLogger("qloop")

MIN_SYNTHETIC_SEED_OFFSET: int = 10_000


# ==============================================================================
# Config & Episode Data Structures
# ==============================================================================


@dataclass
class BlockAttnCalibConfig:
    """Configuration for `stage_type: block_attn_calib`."""

    kl_weight: float = 0.7
    ce_weight: float = 0.3
    kl_temperature: float = 1.0
    ce_chunk_size: int = 128
    contiguous_lm_every: int = 5
    n_sessions: int = 3
    session_len: int = 512
    suffix_len: int = 512
    live_sink_len: int = 4
    sink_tokens: int = 4
    boundary_header_tokens: int = 4
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    virtual_offset: int = 10_000
    interleaved: bool = False
    synthetic_mix_prob: float = 0.25
    synthetic_seed_offset: int = 10_000
    eval_k0_examples: int = 20
    eval_k0_mid_examples: int = 8
    eval_k0_seed: int = 42
    eval_k0_dataset: str | None = None
    eval_depth: int | None = None
    k0_green_threshold: float = 0.92
    k0_yellow_threshold: float = 0.75
    k0_max_green_nll_gap: float | None = 0.05
    k0_refresh_frac: float = 0.05

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> BlockAttnCalibConfig:
        raw = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown block_attn.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def validate(self) -> None:
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0.0:
            raise ValueError(f"block_attn.kl_weight must be finite >= 0, got {self.kl_weight}")
        if not math.isfinite(self.ce_weight) or self.ce_weight < 0.0:
            raise ValueError(f"block_attn.ce_weight must be finite >= 0, got {self.ce_weight}")
        if self.kl_weight + self.ce_weight <= 0.0:
            raise ValueError("block_attn.kl_weight + block_attn.ce_weight must be > 0")
        if not math.isfinite(self.kl_temperature) or self.kl_temperature <= 0.0:
            raise ValueError(f"block_attn.kl_temperature must be finite > 0, got {self.kl_temperature}")
        if int(self.ce_chunk_size) < 1:
            raise ValueError(f"block_attn.ce_chunk_size must be >= 1, got {self.ce_chunk_size}")
        if int(self.contiguous_lm_every) < 0:
            raise ValueError(f"block_attn.contiguous_lm_every must be >= 0, got {self.contiguous_lm_every}")
        if int(self.n_sessions) < 1:
            raise ValueError(f"block_attn.n_sessions must be >= 1, got {self.n_sessions}")
        if int(self.session_len) < 2:
            raise ValueError(f"block_attn.session_len must be >= 2, got {self.session_len}")
        if int(self.suffix_len) < 1:
            raise ValueError(f"block_attn.suffix_len must be >= 1, got {self.suffix_len}")
        if int(self.live_sink_len) < 0 or int(self.sink_tokens) < 0 or int(self.boundary_header_tokens) < 0:
            raise ValueError("block_attn.live_sink_len, sink_tokens, and boundary_header_tokens must be >= 0")
        if int(self.rotary_dim) < 0 or int(self.rotary_dim) % 2 != 0:
            raise ValueError(f"block_attn.rotary_dim must be an even int >= 0, got {self.rotary_dim}")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            raise ValueError(f"block_attn.rope_theta must be finite > 0, got {self.rope_theta}")
        if int(self.virtual_offset) < 0:
            raise ValueError(f"block_attn.virtual_offset must be >= 0, got {self.virtual_offset}")
        if not math.isfinite(self.synthetic_mix_prob) or not (0.0 <= self.synthetic_mix_prob <= 1.0):
            raise ValueError(f"block_attn.synthetic_mix_prob must be in [0, 1], got {self.synthetic_mix_prob}")
        if int(self.synthetic_seed_offset) < MIN_SYNTHETIC_SEED_OFFSET:
            raise ValueError(
                f"block_attn.synthetic_seed_offset must be >= {MIN_SYNTHETIC_SEED_OFFSET} to stay strictly "
                f"disjoint from K0 eval seeds (got {self.synthetic_seed_offset})"
            )
        if int(self.eval_k0_examples) < 0 or int(self.eval_k0_mid_examples) < 0:
            raise ValueError("block_attn.eval_k0_examples and eval_k0_mid_examples must be >= 0")
        if self.eval_depth is not None and int(self.eval_depth) < 1:
            raise ValueError(f"block_attn.eval_depth must be None or >= 1, got {self.eval_depth}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockEpisode:
    """Tokenized multi-session episode for Block-Attention calibration."""

    live_sink_ids: list[int]
    sessions_ids: list[list[int]]
    live_rest_ids: list[int]
    suffix_ids: list[int]
    superseded_indices: tuple[int, ...] = ()
    source: str = "corpus"  # "corpus" | "synthetic"
    category: str = "corpus"


# ==============================================================================
# Memory-Efficient Chunked FP32 KL + CE Loss
# ==============================================================================


class _ChunkedLogitsKLCE(torch.autograd.Function):
    """Exact chunked FP32 `w_kl * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + w_ce * CE(z_s, y)`.

    Processes `chunk_size` tokens at a time in FP32 during both forward and backward so peak
    extra FP32 logit memory stays strictly bounded by `[chunk_size, V]` instead of allocating
    multiple full `[T_suf, V]` FP32 tensors simultaneously.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        student_logits: torch.Tensor,
        oracle_logits: torch.Tensor,
        targets: torch.Tensor,
        kl_weight: float,
        ce_weight: float,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tok = int(student_logits.shape[0])
        dev = student_logits.device
        inv_t = 1.0 / float(temperature)
        t_sq = float(temperature) * float(temperature)
        kl_sum = torch.zeros((), dtype=torch.float32, device=dev)
        ce_sum = torch.zeros((), dtype=torch.float32, device=dev)
        n_ce_valid = 0

        for s in range(0, n_tok, chunk_size):
            e = min(s + chunk_size, n_tok)
            zs_raw = student_logits[s:e].float()
            if kl_weight > 0.0:
                zt_scaled = oracle_logits[s:e].detach().float() * inv_t
                zs_scaled = zs_raw * inv_t
                log_pt = F.log_softmax(zt_scaled, dim=-1)
                log_ps = F.log_softmax(zs_scaled, dim=-1)
                pt = log_pt.exp()
                kl_sum = kl_sum + (pt * (log_pt - log_ps)).sum()
                del zt_scaled, zs_scaled, log_pt, log_ps, pt
            if ce_weight > 0.0:
                tgt_c = targets[s:e]
                valid = tgt_c != ignore_index
                n_v = int(valid.sum().item())
                if n_v > 0:
                    ce_sum = ce_sum + F.cross_entropy(
                        zs_raw, tgt_c, ignore_index=ignore_index, reduction="sum"
                    )
                    n_ce_valid += n_v
            del zs_raw

        kl_mean = kl_sum / float(max(n_tok, 1))
        ce_mean = (
            ce_sum / float(max(n_ce_valid, 1))
            if n_ce_valid > 0
            else torch.zeros((), dtype=torch.float32, device=dev)
        )
        total = float(kl_weight) * t_sq * kl_mean + float(ce_weight) * ce_mean

        ctx.save_for_backward(student_logits, oracle_logits.detach(), targets)
        ctx.kl_weight = float(kl_weight)
        ctx.ce_weight = float(ce_weight)
        ctx.temperature = float(temperature)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        ctx.n_tok = n_tok
        ctx.n_ce_valid = n_ce_valid
        return total, kl_mean.detach(), ce_mean.detach()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_total: torch.Tensor,
        _grad_kl: torch.Tensor | None,
        _grad_ce: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if not ctx.needs_input_grad[0]:
            return (None,) * 8
        student_logits, oracle_logits, targets = ctx.saved_tensors
        kl_w: float = ctx.kl_weight
        ce_w: float = ctx.ce_weight
        temp: float = ctx.temperature
        chunk_size: int = ctx.chunk_size
        ignore_index: int = ctx.ignore_index
        n_tok: int = ctx.n_tok
        n_ce_valid: int = ctx.n_ce_valid

        inv_t = 1.0 / temp
        g_out = grad_total.float()
        kl_scale = g_out * (kl_w * temp / float(max(n_tok, 1)))
        ce_scale = g_out * (ce_w / float(max(n_ce_valid, 1))) if (ce_w > 0.0 and n_ce_valid > 0) else None

        grad_s = torch.empty_like(student_logits)
        for s in range(0, n_tok, chunk_size):
            e = min(s + chunk_size, n_tok)
            zs_raw = student_logits[s:e].float()
            dz = torch.zeros_like(zs_raw)
            ps_t: torch.Tensor | None = None
            if kl_w > 0.0:
                ps_t = F.softmax(zs_raw * inv_t, dim=-1)
                pt_t = F.softmax(oracle_logits[s:e].float() * inv_t, dim=-1)
                dz.add_(ps_t - pt_t, alpha=float(kl_scale.item()))
                del pt_t
            if ce_scale is not None:
                tgt_c = targets[s:e]
                valid = tgt_c != ignore_index
                if bool(valid.any().item()):
                    ps_1 = ps_t if (ps_t is not None and temp == 1.0) else F.softmax(zs_raw, dim=-1)
                    rows = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    ps_1[rows, tgt_c[rows]] -= 1.0
                    if not bool(valid.all().item()):
                        ps_1 = ps_1 * valid.unsqueeze(-1).float()
                    dz.add_(ps_1, alpha=float(ce_scale.item()))
                    del ps_1
            grad_s[s:e] = dz.to(student_logits.dtype)
            del zs_raw, dz, ps_t

        return grad_s, None, None, None, None, None, None, None


def chunked_kl_ce_loss(
    student_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    kl_weight: float = 0.7,
    ce_weight: float = 0.3,
    temperature: float = 1.0,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float, float]:
    """Compute memory-lean chunked FP32 KL + CE loss over suffix logits `[..., T, V]`."""
    require(
        student_logits.shape == oracle_logits.shape,
        "student_logits and oracle_logits shapes must match",
        student=tuple(student_logits.shape),
        oracle=tuple(oracle_logits.shape),
    )
    vocab = int(student_logits.shape[-1])
    s_flat = student_logits.reshape(-1, vocab)
    o_flat = oracle_logits.detach().reshape(-1, vocab)
    t_flat = targets.reshape(-1).to(device=student_logits.device, dtype=torch.long)
    require(
        s_flat.shape[0] == t_flat.shape[0] and s_flat.shape[0] > 0,
        "suffix targets length must match suffix logits length and be > 0",
        logits_n=int(s_flat.shape[0]),
        targets_n=int(t_flat.shape[0]),
    )
    total, kl_m, ce_m = _ChunkedLogitsKLCE.apply(
        s_flat,
        o_flat,
        t_flat,
        float(kl_weight),
        float(ce_weight),
        float(temperature),
        max(1, int(chunk_size)),
        int(ignore_index),
    )
    return total, float(kl_m.item()), float(ce_m.item())


# ==============================================================================
# Zero-Leakage Episode Builders (Clean-Room Corpus + Out-Of-Eval Synthetic)
# ==============================================================================


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if callable(tokenizer):
        try:
            out = tokenizer(text, add_special_tokens=False)
        except TypeError:
            out = tokenizer(text)
        ids = out["input_ids"] if isinstance(out, dict) else getattr(out, "input_ids", out)
    elif hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
    else:
        raise TypeError(f"tokenizer {type(tokenizer).__name__} must be callable or implement .encode()")
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def slice_corpus_sequence_to_episode(
    token_ids: Sequence[int] | torch.Tensor,
    cfg: BlockAttnCalibConfig,
) -> BlockEpisode:
    """Slice a clean-room corpus sequence into `[live_sinks, session_1..K, live_rest, suffix]`."""
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else [int(x) for x in token_ids]
    total = len(ids)
    k_sess = int(cfg.n_sessions)
    min_needed = k_sess * 2 + 2
    require(
        total >= min_needed,
        "corpus sequence too short to form a multi-session block-attention episode",
        seq_len=total,
        min_needed=min_needed,
    )

    desired_total = int(cfg.live_sink_len) + k_sess * int(cfg.session_len) + int(cfg.suffix_len) + 1
    if total >= desired_total:
        live_sink_n = int(cfg.live_sink_len)
        sess_len = int(cfg.session_len)
        suf_len = int(cfg.suffix_len)
    else:
        # Proportionally fit shorter (e.g. toy) sequences while guaranteeing live_rest >= 1 & sess_len >= sink + 1
        live_sink_n = min(int(cfg.live_sink_len), max(0, total - (k_sess * 2 + 2)))
        rem = total - live_sink_n
        suf_len = max(1, min(int(cfg.suffix_len), max(1, rem // 4)))
        rem_for_sess = rem - suf_len - 1
        sess_len = max(2, min(int(cfg.session_len), rem_for_sess // k_sess))
        # Ensure we didn't overflow `rem`
        while live_sink_n + k_sess * sess_len + suf_len + 1 > total and suf_len > 1:
            suf_len -= 1
        while live_sink_n + k_sess * sess_len + suf_len + 1 > total and live_sink_n > 0:
            live_sink_n -= 1

    cursor = 0
    live_sink_ids = ids[cursor : cursor + live_sink_n]
    cursor += live_sink_n
    sessions_ids: list[list[int]] = []
    for _ in range(k_sess):
        sessions_ids.append(ids[cursor : cursor + sess_len])
        cursor += sess_len
    tail = ids[cursor:]
    require(len(tail) >= 2, "insufficient tail tokens after sessions", tail_len=len(tail))
    eff_suf = min(suf_len, len(tail) - 1)
    live_rest_ids = tail[:-eff_suf]
    suffix_ids = tail[-eff_suf:]
    return BlockEpisode(
        live_sink_ids=live_sink_ids,
        sessions_ids=sessions_ids,
        live_rest_ids=live_rest_ids,
        suffix_ids=suffix_ids,
        superseded_indices=(),
        source="corpus",
        category="corpus",
    )


# Strictly training-only synthetic vocabularies (100% disjoint from K0 eval examples)
_TRAIN_MODULES: tuple[str, ...] = (
    "calib_ingest.py",
    "calib_ledger.py",
    "calib_shard.py",
    "calib_worker.py",
    "calib_bridge.py",
    "calib_index.py",
    "calib_queue.py",
    "calib_codec.py",
)
_TRAIN_SYMBOLS: tuple[str, ...] = (
    "TR_BATCH_CAP",
    "TR_QUEUE_PORT",
    "TR_RETRY_LIMIT",
    "TR_SHARD_DEPTH",
    "TR_WINDOW_MS",
    "TR_FLUSH_KB",
    "TR_SLOT_QUOTA",
    "TR_RING_STRIDE",
)
_TRAIN_FILLER_CLAUSES: tuple[str, ...] = (
    "Background telemetry recorded stable throughput across worker pools.",
    "Checksum verification confirmed deterministic partition boundaries.",
    "Periodic compaction archived completed trace spans to cold storage.",
    "Scheduler heartbeat reported nominal queue depth and memory headroom.",
    "Streaming parser normalized metadata headers before segment commit.",
)


def _pack_session_tokens(
    tokenizer: Any,
    *,
    sink_tokens: int,
    fact_text: str,
    filler_text: str,
    target_len: int,
) -> list[int]:
    """Assemble `[sink_prefix_ids, fact_ids, filler_ids]` so sink-stripping never drops the fact."""
    prefix_ids: list[int] = []
    if sink_tokens > 0:
        unit = _encode_text(tokenizer, "Archive block header. ")
        if not unit:
            unit = [1]
        while len(prefix_ids) < sink_tokens:
            prefix_ids.extend(unit)
        prefix_ids = prefix_ids[:sink_tokens]
    fact_ids = _encode_text(tokenizer, fact_text)
    if not fact_ids:
        fact_ids = [1]
    out = prefix_ids + fact_ids
    if len(out) < target_len:
        filler_ids = _encode_text(tokenizer, filler_text) or [1]
        while len(out) < target_len:
            need = target_len - len(out)
            out.extend(filler_ids[:need])
    return out[: max(len(prefix_ids) + len(fact_ids), target_len)]


def build_synthetic_calib_episode(
    tokenizer: Any,
    cfg: BlockAttnCalibConfig,
    *,
    episode_idx: int,
    base_seed: int = 0,
) -> BlockEpisode:
    """Build a deterministic out-of-eval-seed multi-session QA / temporal_update / code episode.

    Uses `base_seed + synthetic_seed_offset + episode_idx * 7919` (`synthetic_seed_offset >= 10000`)
    and strictly training-only identifiers (`calib_*.py`, `TR_*`) so training episodes never
    collide with Kill-Gate `K0` (`seed=42` / `build_default_smoke_examples()`).
    """
    ep_seed = int(base_seed) + int(cfg.synthetic_seed_offset) + int(episode_idx) * 7919
    rng = random.Random(ep_seed)
    k_sess = int(cfg.n_sessions)
    kind_idx = int(episode_idx) % 3
    if kind_idx == 1 and k_sess < 2:
        kind_idx = 0

    filler = " ".join(rng.sample(_TRAIN_FILLER_CLAUSES, k=min(3, len(_TRAIN_FILLER_CLAUSES))))
    sessions_ids: list[list[int]] = []
    superseded: tuple[int, ...] = ()

    if kind_idx == 0:
        # Multi-session cross-block fact QA
        category = "synthetic_multi_session_qa"
        sym_pool = rng.sample(_TRAIN_SYMBOLS, k=min(k_sess, len(_TRAIN_SYMBOLS)))
        mod_pool = rng.sample(_TRAIN_MODULES, k=min(k_sess, len(_TRAIN_MODULES)))
        facts: list[tuple[str, str, str]] = []
        for si in range(k_sess):
            sym = f"{sym_pool[si % len(sym_pool)]}_{rng.randint(10, 99)}"
            mod = mod_pool[si % len(mod_pool)]
            val = str(rng.randint(100, 9999))
            facts.append((mod, sym, val))
            ftext = f"Record {si + 1}: module {mod} defines {sym} = {val}. "
            sessions_ids.append(
                _pack_session_tokens(
                    tokenizer,
                    sink_tokens=int(cfg.sink_tokens),
                    fact_text=ftext,
                    filler_text=filler + " ",
                    target_len=int(cfg.session_len),
                )
            )
        pick = rng.randrange(k_sess)
        t_mod, t_sym, t_val = facts[pick]
        live_prompt = f"Live query block: From module {t_mod}, report the exact integer for {t_sym}: "
        answer = t_val
    elif kind_idx == 1:
        # Temporal update / SleepGate supersession episode
        category = "synthetic_temporal_update"
        mod = rng.choice(_TRAIN_MODULES)
        sym = f"{rng.choice(_TRAIN_SYMBOLS)}_{rng.randint(10, 99)}"
        old_val = str(rng.randint(10, 499))
        new_val = str(rng.randint(500, 9999))
        update_idx = rng.randint(1, k_sess - 1)
        for si in range(k_sess):
            if si == 0:
                ftext = f"Initial record (stale): {mod} sets {sym} = {old_val}. "
            elif si == update_idx:
                ftext = f"Superseding initial record: {mod} updates {sym} = {new_val}. "
            else:
                aux_sym = f"{rng.choice(_TRAIN_SYMBOLS)}_{rng.randint(100, 999)}"
                ftext = f"Auxiliary record {si + 1}: {aux_sym} = {rng.randint(1, 99)}. "
            sessions_ids.append(
                _pack_session_tokens(
                    tokenizer,
                    sink_tokens=int(cfg.sink_tokens),
                    fact_text=ftext,
                    filler_text=filler + " ",
                    target_len=int(cfg.session_len),
                )
            )
        superseded = (0,)
        live_prompt = f"Live query block: After the latest update in {mod}, what is the active value of {sym}? "
        answer = new_val
    else:
        # Code symbol & hex tag retrieval across sessions
        category = "synthetic_code_retrieval"
        mod_pool = rng.sample(_TRAIN_MODULES, k=min(k_sess, len(_TRAIN_MODULES)))
        entries: list[tuple[str, str, str]] = []
        for si in range(k_sess):
            mod = mod_pool[si % len(mod_pool)]
            tag = f"tr_rev_{rng.randint(10, 99)}_{si}"
            hex_tok = f"{rng.randrange(16**6):06x}"
            entries.append((mod, tag, hex_tok))
            ftext = f"Code manifest {si + 1}: `{mod}` pins `{tag}` -> `{hex_tok}`. "
            sessions_ids.append(
                _pack_session_tokens(
                    tokenizer,
                    sink_tokens=int(cfg.sink_tokens),
                    fact_text=ftext,
                    filler_text=filler + " ",
                    target_len=int(cfg.session_len),
                )
            )
        pick = rng.randrange(k_sess)
        t_mod, t_tag, t_hex = entries[pick]
        live_prompt = f"Live query block: In `{t_mod}`, return the 6-hex token pinned for `{t_tag}`: "
        answer = t_hex

    live_ids = _encode_text(tokenizer, live_prompt)
    ans_ids = _encode_text(tokenizer, answer)
    if len(live_ids) < 2:
        live_ids = [*live_ids, 1, 1][:2]
    if not ans_ids:
        ans_ids = [1]

    live_sink_n = min(int(cfg.live_sink_len), max(0, len(live_ids) - 1))
    return BlockEpisode(
        live_sink_ids=live_ids[:live_sink_n],
        sessions_ids=sessions_ids,
        live_rest_ids=live_ids[live_sink_n:],
        suffix_ids=ans_ids,
        superseded_indices=superseded,
        source="synthetic",
        category=category,
    )


# Strictly eval-only vocabularies for K0 evaluation (disjoint from training synthetic identifiers)
_EVAL_SERVICES: tuple[str, ...] = (
    "eval_gateway.py",
    "eval_planner.py",
    "eval_vault.py",
    "eval_broker.py",
    "eval_sentinel.py",
    "eval_dispatch.py",
)
_EVAL_KEYS: tuple[str, ...] = (
    "EV_PORT",
    "EV_MAX_HOPS",
    "EV_QUORUM",
    "EV_EPOCH_TTL",
    "EV_POOL_CAP",
    "EV_SYNC_MS",
)


def build_disjoint_k0_eval_examples(
    n_examples: int,
    *,
    seed: int = 42,
    dataset_path: str | None = None,
) -> list[K0Example]:
    """Build deterministic Kill-Gate `K0` evaluation examples disjoint from training episodes."""
    if n_examples <= 0:
        return []
    if dataset_path:
        loaded = load_examples_from_file(dataset_path)
        return loaded[:n_examples]

    examples: list[K0Example] = list(build_default_smoke_examples())
    if len(examples) >= n_examples:
        return examples[:n_examples]

    rng = random.Random(int(seed))
    idx = 0
    while len(examples) < n_examples:
        kind = idx % 3
        svc = _EVAL_SERVICES[idx % len(_EVAL_SERVICES)]
        key = f"{_EVAL_KEYS[idx % len(_EVAL_KEYS)]}_{idx}"
        if kind == 0:
            v1 = str(rng.randint(100, 999))
            v2 = str(rng.randint(1000, 9999))
            examples.append(
                K0Example(
                    example_id=f"k0_eval_multihop_{idx:03d}",
                    sessions=[
                        f"Session 1: Service `{svc}` configures `{key} = {v1}` during boot.",
                        f"Session 2: Secondary monitor checks `{svc}` metric `{key}_AUX = {v2}`.",
                    ],
                    live_prompt=f"Session 3: What value does `{svc}` configure for `{key}`?",
                    answer=v1,
                    horizon_tokens=32_000,
                    category="multi_hop",
                    metadata={"seed": int(seed) + idx + 1},
                )
            )
        elif kind == 1:
            old_v = str(rng.randint(10, 99))
            new_v = str(rng.randint(200, 899))
            examples.append(
                K0Example(
                    example_id=f"k0_eval_update_{idx:03d}",
                    sessions=[
                        f"Session 1 (stale): `{svc}` sets `{key} = {old_v}`.",
                        f"Session 2 (updated): Superseding Session 1, `{svc}` updates `{key} = {new_v}`.",
                    ],
                    live_prompt=f"Session 3: What is the active value of `{key}` in `{svc}`?",
                    answer=new_v,
                    horizon_tokens=64_000,
                    category="contradictory_update",
                    superseded_session_indices=(0,),
                    metadata={"seed": int(seed) + idx + 1},
                )
            )
        else:
            tag = f"ev_rel_{idx}"
            sha = f"{rng.randrange(16**6):06x}"
            examples.append(
                K0Example(
                    example_id=f"k0_eval_code_{idx:03d}",
                    sessions=[
                        f"Session 1: Release `{tag}` in `{svc}` pins digest `{sha}`.",
                    ],
                    live_prompt=f"Session 2: Return the pinned digest for `{tag}` in `{svc}`.",
                    answer=sha,
                    horizon_tokens=128_000,
                    category="codebase_qa",
                    metadata={"seed": int(seed) + idx + 1},
                )
            )
        idx += 1
    return examples


# ==============================================================================
# Model Geometry & Depth-Agnostic K0 Evaluator
# ==============================================================================


def _resolve_model_k0_config(model: Any, cfg: BlockAttnCalibConfig) -> K0Config:
    """Build a `K0Config` aligned to `cfg` and safe for the model's physical `head_dim`."""
    rotary_dim = int(cfg.rotary_dim)
    rope_theta = float(cfg.rope_theta)
    text_cfgs = _resolve_text_configs(model)
    for tc in text_cfgs:
        head_dim = getattr(tc, "head_dim", None)
        if head_dim is not None and int(head_dim) > 0:
            hd = int(head_dim)
            if rotary_dim >= hd:
                prf = getattr(tc, "partial_rotary_factor", None)
                if prf is None and isinstance(getattr(tc, "rope_parameters", None), dict):
                    prf = tc.rope_parameters.get("partial_rotary_factor")
                inferred = round(hd * float(prf)) if prf is not None else (hd // 4)
                inferred = max(0, min(hd - 2, inferred - (inferred % 2)))
                rotary_dim = inferred
            break

    return K0Config(
        green_threshold=float(cfg.k0_green_threshold),
        yellow_threshold=float(cfg.k0_yellow_threshold),
        max_green_nll_gap=cfg.k0_max_green_nll_gap,
        refresh_frac=float(cfg.k0_refresh_frac),
        boundary_header_tokens=int(cfg.boundary_header_tokens),
        sink_tokens=int(cfg.sink_tokens),
        rope_theta=rope_theta,
        rotary_dim=rotary_dim,
        virtual_offset=int(cfg.virtual_offset),
        interleaved=bool(cfg.interleaved),
    )


def _detach_cache_states_(cache: Any) -> None:
    """Ensure any linear-attention state tensors cloned into `cache` are detached from autograd."""
    if cache is None:
        return
    for attr in ("conv_states", "recurrent_states"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            for idx, item in enumerate(lst):
                if isinstance(item, torch.Tensor):
                    lst[idx] = item.detach()
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)):
        for lyr in layers:
            for attr in ("conv_states", "recurrent_states"):
                val = getattr(lyr, attr, None)
                if isinstance(val, torch.Tensor):
                    try:
                        setattr(lyr, attr, val.detach())
                    except (AttributeError, TypeError):
                        pass
                elif isinstance(val, list):
                    for idx, item in enumerate(val):
                        if isinstance(item, torch.Tensor):
                            val[idx] = item.detach()


class ModelK0Evaluator(HFCheckpointK0Evaluator):
    """Depth-agnostic `K0` evaluator wrapping `HFCheckpointK0Evaluator` for live trainer stacks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str | None = None,
        live_sink_len: int = 4,
        loop_controller: Any = None,
        lsa_manager: Any = None,
        hssd_handle: Any = None,
        eval_depth: int | None = None,
    ) -> None:
        super().__init__(model, tokenizer, device=device, live_sink_len=live_sink_len)
        self.loop_controller = loop_controller
        self.lsa_manager = lsa_manager
        self.hssd_handle = hssd_handle
        self.eval_depth = eval_depth

    def _clear_aux(self) -> None:
        if self.loop_controller is not None and callable(getattr(self.loop_controller, "release_graph", None)):
            self.loop_controller.release_graph(clear_routing=True)
        if self.lsa_manager is not None:
            self.lsa_manager.aux_loss()
            self.lsa_manager.clear_cache()
        if self.hssd_handle is not None:
            for ad in self.hssd_handle.adapters.values():
                ad.clear_aux()

    def __call__(self, example: K0Example, cfg: K0Config) -> dict[str, Any]:
        was_training = bool(getattr(self.model, "training", False))
        prev_passes = self.loop_controller.active_passes if self.loop_controller is not None else None
        self.model.eval()
        if self.loop_controller is not None and self.eval_depth is not None:
            self.loop_controller.set_active_passes(int(self.eval_depth))
        self._clear_aux()
        try:
            return super().__call__(example, cfg)
        finally:
            self._clear_aux()
            if self.loop_controller is not None and prev_passes is not None:
                self.loop_controller.set_active_passes(prev_passes)
            self.model.train(was_training)


# ==============================================================================
# BlockAttentionCalibrator Trainer
# ==============================================================================


class BlockAttentionCalibrator:
    """Depth-agnostic Block-Attention Logit-Distillation Calibrator (`stage_type: block_attn_calib`)."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: str,
        stack: Stack | None = None,
        hub_api: Any = None,
    ) -> None:
        self.cfg, self.run_dir = cfg, run_dir
        self.tcfg = dict(cfg.get("train") or {})
        configure_tf32(bool(self.tcfg.get("allow_tf32", True)))
        self._hub_api = hub_api
        sync_cfg = (cfg.get("checkpoint") or {}).get("sync")
        if stack is None and sync_cfg:
            from ..hub import HubSyncManager

            hub = HubSyncManager.from_sync_cfg(sync_cfg, api=hub_api)
            if hub is not None:
                stage_id = str(cfg.get("stage") or os.path.basename(os.path.abspath(run_dir)))
                run_root = os.path.dirname(os.path.abspath(run_dir))
                hub.prepare_stage_workspace(stage_id, dict(cfg), run_root)

        if stack is None:
            # Keep `gradient_checkpointing=False` on the short cached suffix forward so HF never
            # silently suppresses `use_cache=True`.
            build_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else copy.deepcopy(dict(cfg))
            t_build = dict(build_cfg.get("train") or {})
            if t_build.get("gradient_checkpointing"):
                log.info(
                    "[block_attn_calib] forcing train.gradient_checkpointing=False for cached suffix forward"
                )
            t_build["gradient_checkpointing"] = False
            build_cfg["train"] = t_build
            self.stack = build_stack(build_cfg)
        else:
            self.stack = stack
            self._disable_hf_gradient_checkpointing()

        require(
            bool(self.stack.named_trainable),
            "nothing is trainable - check train.trainable / gdn2 / lsa / loop",
        )
        self.bcfg = BlockAttnCalibConfig.from_dict(cfg.get("block_attn"))
        self.k0_cfg = _resolve_model_k0_config(self.stack.model, self.bcfg)
        self._cache_kw = cache_kwarg_name(self.stack.model)
        self._accepts_pos_ids = HFCheckpointK0Evaluator._detect_position_ids_support(self.stack.model)

        self.seed = int(cfg.get("seed", 0))
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)

        ocfg = dict(self.tcfg.get("optim") or {})
        ocfg.setdefault("total_steps", int(self.tcfg.get("steps", 300)))
        ocfg.setdefault("seed", self.seed)
        self.opt = FusedStepOptimizer.from_config(
            list(self.stack.named_trainable.items()),
            ocfg,
            grad_accum=int(self.tcfg.get("grad_accum", 1)),
        )
        self.ckpt = CheckpointManager(
            run_dir,
            keep_last=int(self.tcfg.get("keep_last", 3)),
            sync=sync_cfg,
            hub_api=hub_api,
        )
        self.metrics = MetricsLogger(run_dir, cfg)
        self.step = 0
        self.tokens_seen = 0
        self.synthetic_episode_idx = 0
        self.resume_mode = "fresh"
        self._stop = False
        self.stream: Any = None
        self.last_k0: dict[str, Any] = {}
        self._fresh_cache_helper = HFCheckpointK0Evaluator(
            self.stack.model,
            self.stack.tokenizer,
            device=self.stack.device,
            live_sink_len=int(self.bcfg.live_sink_len),
        )

    def _disable_hf_gradient_checkpointing(self) -> None:
        for cand in (self.stack.model, self.stack.parts.text_model):
            disable_fn = getattr(cand, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                try:
                    disable_fn()
                except Exception:
                    pass

    @contextlib.contextmanager
    def _no_loop_checkpointing(self) -> Iterator[None]:
        """Temporarily disable loop layer checkpointing during `use_cache=True` suffix forward+backward."""
        lp = self.stack.loop
        if lp is None or not lp.spec.checkpoint:
            yield
            return
        lp.spec.checkpoint = False
        try:
            yield
        finally:
            lp.spec.checkpoint = True

    def _clear_transient_state(self) -> None:
        st = self.stack
        if st.loop is not None and callable(getattr(st.loop, "release_graph", None)):
            st.loop.release_graph(clear_routing=True)
        if st.lsa is not None:
            st.lsa.aux_loss()
            st.lsa.clear_cache()
        if st.hssd is not None:
            for ad in st.hssd.adapters.values():
                ad.clear_aux()

    # ------------------------------------------------------------------ data
    def _build_stream(self) -> Any:
        dcfg = dict(self.cfg.get("data") or {})
        if not dcfg.get("sources"):
            require(
                self.bcfg.synthetic_mix_prob >= 1.0,
                "data.sources is empty and block_attn.synthetic_mix_prob < 1.0",
            )
            return None

        from ..data.mixture import build_train_stream

        holdout_mod = int(dcfg.get("holdout_mod", 100))
        holdout_eval_buckets = int(dcfg.get("holdout_eval_buckets", 2))
        explicit_holdout = "holdout_mod" in dcfg or "holdout_eval_buckets" in dcfg
        active = holdout_mod > 0 and holdout_eval_buckets > 0

        patched: list[Any] = []
        any_part = False
        for src in list(dcfg.get("sources") or []):
            if not isinstance(src, dict):
                patched.append(src)
                continue
            s = dict(src)
            if "split_role" not in s:
                src_explicit = "holdout_mod" in s or "holdout_eval_buckets" in s or explicit_holdout
                src_mod = int(s.get("holdout_mod", holdout_mod))
                is_toy = s.get("name") in ("toy_text", "toy_cot")
                if not active or src_mod <= 0 or (is_toy and not src_explicit):
                    s["split_role"] = "all"
                else:
                    s["split_role"] = "train"
            if s.get("split_role") == "train":
                any_part = True
            patched.append(s)

        stream_cfg = dict(dcfg)
        stream_cfg["sources"] = patched
        stream_cfg["holdout_mod"] = holdout_mod
        stream_cfg["holdout_eval_buckets"] = holdout_eval_buckets
        eff_role = "train" if any_part else str(stream_cfg.get("split_role", "all"))
        return build_train_stream(
            stream_cfg,
            self.stack.tokenizer,
            seed=self.seed,
            split_role=eff_role,
            holdout_mod=holdout_mod,
            holdout_eval_buckets=holdout_eval_buckets,
        )

    def _sample_episode(self) -> BlockEpisode:
        use_synth = (self.stream is None) or (self.rng.random() < float(self.bcfg.synthetic_mix_prob))
        if use_synth:
            ep = build_synthetic_calib_episode(
                self.stack.tokenizer,
                self.bcfg,
                episode_idx=self.synthetic_episode_idx,
                base_seed=self.seed,
            )
            self.synthetic_episode_idx += 1
            return ep
        item = next(self.stream)
        return slice_corpus_sequence_to_episode(item["input_ids"], self.bcfg)

    # ------------------------------------------------------------------ resume / save
    def _state(self) -> dict[str, Any]:
        data_state = (
            self.stream.state_dict()
            if (self.stream is not None and callable(getattr(self.stream, "state_dict", None)))
            else None
        )
        return {
            "stage": self.cfg.get("stage"),
            "tokens_seen": self.tokens_seen,
            "synthetic_episode_idx": self.synthetic_episode_idx,
            "stack": self.stack.describe(),
            "block_attn": self.bcfg.to_dict(),
            "k0": dict(self.last_k0),
            "data": data_state,
            "trainer_rng": serialize_py_rng_state(self.rng),
            "resume_mode": self.resume_mode,
            "config": self.cfg.get("_meta_", {}),
        }

    def save(self, *, completed: bool = False, metrics: dict[str, Any] | None = None) -> str:
        raw_cfg = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg)
        return self.ckpt.save(
            self.step,
            self.stack.checkpoint_params(),
            self.opt.state_dict(),
            self._state(),
            completed=completed,
            target_steps=int(self.tcfg.get("steps", 300)),
            metrics=metrics,
            config_dict=raw_cfg,
            extra_rng={"trainer_rng": self.rng.getstate()},
        )

    def maybe_resume(self) -> None:
        from ..hub import resolve_run_path

        run_root = os.path.dirname(os.path.abspath(self.run_dir))
        init = resolve_run_path(self.tcfg.get("init_from"), run_root=run_root)
        if self.tcfg.get("resume", True) and self.ckpt.latest():
            ck = self.ckpt.load(map_location="cpu", restore_rng=True)
            assert ck is not None
            st = ck["state"]
            smeta = st.get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            self.stack.restore_checkpoint(ck["params"], strict=True, expert_delta_scale=sc, prior_stack_meta=smeta)
            self.step = int(st["step"])
            self.tokens_seen = int(st.get("tokens_seen", 0))
            self.synthetic_episode_idx = int(st.get("synthetic_episode_idx", 0))
            if isinstance(st.get("k0"), dict):
                self.last_k0 = dict(st["k0"])
            if ck["optim"] is not None:
                self.opt.load_state_dict(ck["optim"])
                self.resume_mode = "exact"
            else:
                self.resume_mode = "weights_only"
                if not bool(self.tcfg.get("reset_lr_on_weights_only", False)):
                    self.opt.sync_step(self.step)
            trainer_rng = st.get("trainer_rng")
            if trainer_rng is None and isinstance(ck.get("rng"), dict):
                trainer_rng = ck["rng"].get("trainer_rng")
            if trainer_rng is not None:
                self.rng.setstate(deserialize_py_rng_state(trainer_rng))
            if st.get("data") is not None and self.stream is not None:
                self.stream.load_state_dict(st["data"])
            log.info(
                "[block_attn_calib] resumed at step %d (%.2fM tokens, synth_idx=%d, mode=%s)",
                self.step,
                self.tokens_seen / 1e6,
                self.synthetic_episode_idx,
                self.resume_mode,
            )
        elif init:
            ck = (
                CheckpointManager(init, keep_last=0).load(map_location="cpu", restore_rng=False)
                if os.path.isdir(init)
                else None
            )
            require(ck is not None, "train.init_from: no checkpoint found", init_from=init)
            assert ck is not None
            smeta = (ck.get("state") or {}).get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            counts = self.stack.restore_checkpoint(
                ck["params"],
                strict=False,
                expert_delta_scale=sc,
                prior_stack_meta=smeta,
                drop_inactive_loop_adapters=bool(self.tcfg.get("drop_inactive_loop_adapters", False)),
            )
            self.resume_mode = "init_from"
            log.info("[block_attn_calib] initialised from %s (%s; optimizer fresh)", ck["path"], counts)

    # ------------------------------------------------------------------ forward helpers
    def _model_forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Any = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> Any:
        kw: dict[str, Any] = {"input_ids": input_ids, "use_cache": use_cache}
        if cache is not None:
            kw[self._cache_kw] = cache
        if position_ids is not None and self._accepts_pos_ids:
            kw["position_ids"] = position_ids
        return self.stack.model(**kw)

    def _block_attn_microstep(
        self,
        episode: BlockEpisode,
        *,
        accum: int,
        aux_w: dict[str, Any],
    ) -> tuple[dict[str, float], int]:
        st = self.stack
        dev = st.device
        k0_cfg = self.k0_cfg
        bcfg = self.bcfg
        superseded = set(episode.superseded_indices)

        live_sink_ids = list(episode.live_sink_ids)
        live_sink_n = len(live_sink_ids)
        active_sessions = [
            list(s) for idx, s in enumerate(episode.sessions_ids) if idx not in superseded and len(s) > 0
        ]
        live_rest_ids = list(episode.live_rest_ids)
        suffix_ids = list(episode.suffix_ids)
        require(len(live_rest_ids) >= 1 and len(suffix_ids) >= 1, "episode needs >=1 live_rest and >=1 suffix token")

        # 1. Teacher contiguous forward + independent block prefills (all under `torch.no_grad()`)
        with torch.no_grad():
            flat_active = [tid for seq in active_sessions for tid in seq]
            full_ids_a = torch.tensor(
                [live_sink_ids + flat_active + live_rest_ids + suffix_ids],
                dtype=torch.long,
                device=dev,
            )
            out_a = self._model_forward(full_ids_a, use_cache=True)
            prefix_a = live_sink_n + len(flat_active) + len(live_rest_ids)
            start_a = prefix_a - 1
            t_suf = len(suffix_ids)
            oracle_logits = out_a.logits[:, start_a : start_a + t_suf, :].detach().clone()
            oracle_k_ref, oracle_v_ref, layer_indices = extract_layer_kv_pairs(out_a, clone=False)
            ref_cache = extract_output_cache(out_a)
            empty_k_proto = oracle_k_ref[..., :0, :].detach()
            empty_v_proto = oracle_v_ref[..., :0, :].detach()
            del out_a, oracle_k_ref, oracle_v_ref

            # Prefill live Partition-A `[0:live_sink_n)` sinks at `pos = 0..live_sink_n-1`
            live_sink_k: torch.Tensor | None = None
            live_sink_v: torch.Tensor | None = None
            live_sink_cache: Any = None
            if live_sink_n > 0:
                sink_t = torch.tensor([live_sink_ids], dtype=torch.long, device=dev)
                sink_pos = torch.arange(0, live_sink_n, dtype=torch.long, device=dev).unsqueeze(0)
                sink_out = self._model_forward(sink_t, position_ids=sink_pos, use_cache=True)
                live_sink_k, live_sink_v, _ = extract_layer_kv_pairs(sink_out, layer_indices=layer_indices)
                live_sink_cache = extract_output_cache(sink_out)
                del sink_out

            # Independent session prefills -> Pre-RoPE un-rotate + sink-strip + H64-INT4 + virtual RoPE mount
            base_mount_offset = max(int(k0_cfg.virtual_offset), live_sink_n) if self._accepts_pos_ids else live_sink_n
            cursor = base_mount_offset
            mnt_k_list: list[torch.Tensor] = []
            mnt_v_list: list[torch.Tensor] = []
            for s_idx, s_ids in enumerate(active_sessions):
                s_len = len(s_ids)
                s_t = torch.tensor([s_ids], dtype=torch.long, device=dev)
                s_out = self._model_forward(s_t, use_cache=True)
                sk, sv, _ = extract_layer_kv_pairs(s_out, layer_indices=layer_indices)
                del s_out
                eff_sink = min(int(k0_cfg.sink_tokens), max(0, s_len - 1))
                local_k0_cfg = replace(k0_cfg, sink_tokens=eff_sink) if eff_sink != k0_cfg.sink_tokens else k0_cfg
                s_pos = torch.arange(0, s_len, dtype=torch.long, device=dev)
                mnt = build_corrected_partition_b_kv(
                    sk,
                    sv,
                    s_pos,
                    live_sink_n,
                    cursor,
                    local_k0_cfg,
                    blob_id=f"calib_s{self.step}_{s_idx}",
                )
                mnt_k_list.append(mnt.k.detach())
                mnt_v_list.append(mnt.v.detach())
                cursor += int(mnt.k.shape[-2])
                del sk, sv, mnt

            if mnt_k_list:
                k_c_partb = torch.cat(mnt_k_list, dim=-2)
                v_c_partb = torch.cat(mnt_v_list, dim=-2)
            else:
                k_c_partb = empty_k_proto
                v_c_partb = empty_v_proto

            if live_sink_k is not None and live_sink_v is not None:
                k_c_mount = torch.cat([live_sink_k.detach(), k_c_partb], dim=-2).detach().contiguous()
                v_c_mount = torch.cat([live_sink_v.detach(), v_c_partb], dim=-2).detach().contiguous()
            else:
                k_c_mount = k_c_partb.detach().contiguous()
                v_c_mount = v_c_partb.detach().contiguous()

            cache_c = self._fresh_cache_helper._fresh_cache_from(ref_cache)
            _clone_linear_states_into(live_sink_cache, cache_c)
            _detach_cache_states_(cache_c)
            if k_c_mount.shape[-2] > 0:
                inject_layer_kv_pairs(cache_c, k_c_mount, v_c_mount, layer_indices)

            live_rest_and_suf = torch.tensor([live_rest_ids + suffix_ids], dtype=torch.long, device=dev)
            live_pos_ids: torch.Tensor | None = (
                torch.arange(cursor, cursor + len(live_rest_ids) + t_suf, dtype=torch.long, device=dev).unsqueeze(0)
                if self._accepts_pos_ids
                else None
            )
            suffix_targets = torch.tensor([suffix_ids], dtype=torch.long, device=dev)
            del ref_cache, live_sink_cache, live_sink_k, live_sink_v, k_c_partb, v_c_partb, mnt_k_list, mnt_v_list

        # Drop any stale no-grad aux terms created during teacher / prefill forwards
        self._clear_transient_state()

        # 2. Differentiable student suffix forward with mounted Partition-B cache
        with torch.enable_grad(), self._no_loop_checkpointing():
            student_out = self._model_forward(
                live_rest_and_suf,
                cache=cache_c,
                position_ids=live_pos_ids,
                use_cache=True,
            )
            start_s = len(live_rest_ids) - 1
            student_logits = student_out.logits[:, start_s : start_s + t_suf, :]
            loss, kl_val, ce_val = chunked_kl_ce_loss(
                student_logits,
                oracle_logits,
                suffix_targets,
                kl_weight=float(bcfg.kl_weight),
                ce_weight=float(bcfg.ce_weight),
                temperature=float(bcfg.kl_temperature),
                chunk_size=int(bcfg.ce_chunk_size),
            )
            total = loss
            aux_total = 0.0
            aux_details: dict[str, float] = {}
            for a_name, a_loss in st.aux_losses().items():
                w_a = float(aux_w.get(a_name, 1.0))
                if w_a != 0.0:
                    total = total + w_a * a_loss
                a_val = float(a_loss.detach().item())
                aux_details[a_name] = a_val
                aux_total += w_a * a_val
            (total / float(max(accum, 1))).backward()

        self._clear_transient_state()
        del student_out, student_logits, oracle_logits, cache_c, k_c_mount, v_c_mount

        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val) or not math.isfinite(kl_val) or not math.isfinite(ce_val):
            raise FloatingPointError(
                f"non-finite block-attention calibration loss at step {self.step}: "
                f"loss={loss_val}, kl={kl_val}, ce={ce_val}"
            )

        out: dict[str, float] = {
            "total": loss_val + aux_total,
            "block_attn": loss_val,
            "kl": kl_val,
            "ce": ce_val,
            "aux": aux_total,
            "synthetic_frac": 1.0 if episode.source == "synthetic" else 0.0,
            "contiguous_anchor": 0.0,
            **aux_details,
        }
        return out, int(live_rest_and_suf.numel())

    def _contiguous_lm_microstep(
        self,
        *,
        accum: int,
        aux_w: dict[str, Any],
        impl: str,
        pass_pos: bool,
    ) -> tuple[dict[str, float], int]:
        st = self.stack
        item = next(self.stream)
        ids = item["input_ids"].unsqueeze(0).to(st.device)
        labels = item["labels"].unsqueeze(0).to(st.device)
        pos = item["position_ids"].unsqueeze(0).to(st.device) if (pass_pos and "position_ids" in item) else None

        h = st.hidden(ids, pos)
        loss, _ntok = lm_loss(h, st.lm_head_weight, labels, impl=impl)
        total = loss
        aux_total = 0.0
        aux_details: dict[str, float] = {}
        for a_name, a_loss in st.aux_losses().items():
            w_a = float(aux_w.get(a_name, 1.0))
            if w_a != 0.0:
                total = total + w_a * a_loss
            a_val = float(a_loss.detach().item())
            aux_details[a_name] = a_val
            aux_total += w_a * a_val
        (total / float(max(accum, 1))).backward()
        if st.loop is not None and callable(getattr(st.loop, "release_graph", None)):
            st.loop.release_graph()

        lm_val = float(loss.detach().item())
        if not math.isfinite(lm_val) or not math.isfinite(aux_total):
            raise FloatingPointError(f"non-finite contiguous LM anchor loss at step {self.step}: lm={lm_val}")
        out: dict[str, float] = {
            "total": lm_val + aux_total,
            "contiguous_lm": lm_val,
            "aux": aux_total,
            "contiguous_anchor": 1.0,
            **aux_details,
        }
        return out, int(ids.numel())

    # ------------------------------------------------------------------ live K0 eval
    @torch.no_grad()
    def eval_k0(self, *, n_examples: int | None = None) -> dict[str, Any]:
        """Run Kill-Gate `K0` evaluation (`run_k0_suite`) and return flat loggable metrics."""
        n_ex = int(self.bcfg.eval_k0_examples if n_examples is None else n_examples)
        if n_ex <= 0:
            return {}
        examples = build_disjoint_k0_eval_examples(
            n_ex,
            seed=int(self.bcfg.eval_k0_seed),
            dataset_path=self.bcfg.eval_k0_dataset,
        )
        if not examples:
            return {}

        eval_depth = self.bcfg.eval_depth
        if eval_depth is None and self.stack.loop is not None:
            eval_depth = int(self.stack.loop.spec.num_passes)

        evaluator = ModelK0Evaluator(
            self.stack.model,
            self.stack.tokenizer,
            device=self.stack.device,
            live_sink_len=int(self.bcfg.live_sink_len),
            loop_controller=self.stack.loop,
            lsa_manager=self.stack.lsa,
            hssd_handle=self.stack.hssd,
            eval_depth=eval_depth,
        )
        report: K0Report = run_k0_suite(
            examples,
            evaluator,
            self.k0_cfg,
            metadata={
                "step": self.step,
                "stage": self.cfg.get("stage"),
                "eval_depth": eval_depth if eval_depth is not None else 1,
            },
        )
        arm_a = report.arms["arm_a"]
        arm_b = report.arms["arm_b"]
        arm_c = report.arms["arm_c"]
        arm_d = report.arms["arm_d"]
        summary: dict[str, Any] = {
            "k0/arm_a_nll": float(arm_a.answer_nll),
            "k0/arm_b_nll_gap": float(arm_b.nll_gap),
            "k0/arm_c_nll": float(arm_c.answer_nll),
            "k0/arm_c_nll_gap": float(arm_c.nll_gap),
            "k0/arm_c_em": float(arm_c.exact_match),
            "k0/arm_c_em_retention": float(arm_c.oracle_retention_ratio),
            "k0/arm_c_tok_acc": float(arm_c.token_acc),
            "k0/arm_d_nll_gap": float(arm_d.nll_gap),
            "k0/arm_d_em_retention": float(arm_d.oracle_retention_ratio),
            "k0/verdict": str(report.verdict),
            "k0/sub_verdict": str(report.sub_verdict),
            "k0/n_examples": int(report.n_examples),
        }
        self.last_k0 = dict(summary)
        log.info(
            "[block_attn_calib] K0 eval step %d (n=%d): verdict=%s (%s) arm_c_nll_gap=%.4f "
            "arm_c_em_ret=%.3f arm_c_tok_acc=%.3f",
            self.step,
            report.n_examples,
            report.verdict,
            report.sub_verdict,
            arm_c.nll_gap,
            arm_c.oracle_retention_ratio,
            arm_c.token_acc,
        )
        return summary

    # ------------------------------------------------------------------ main loop
    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            log.warning("[block_attn_calib] signal %s received: will checkpoint and stop after this step", signum)
            self._stop = True

        try:
            signal.signal(signal.SIGTERM, handler)
        except Exception:
            pass

    def train(self) -> dict[str, Any]:
        st, t = self.stack, self.tcfg
        configure_tf32(bool(t.get("allow_tf32", True)))
        steps = int(t.get("steps", 300))
        accum = int(t.get("grad_accum", 1))
        log_every = int(t.get("log_every", 5))
        ckpt_every = int(t.get("ckpt_every", 50))
        eval_every = int(t.get("eval_every", 100))
        budget = float(t.get("time_budget_hours", 0)) * 3600.0
        aux_w = dict(t.get("aux_weights") or {})
        impl = str(t.get("loss_impl", "auto"))
        pass_pos = bool(t.get("pass_position_ids", False))

        with stage("block_attn:data"):
            self.stream = self._build_stream()
        self.maybe_resume()
        if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
            st.loop.set_step(self.step, steps)

        self._install_signal_handlers()
        st.model.train()
        t_start = time.time()

        mid_k0_n = min(int(self.bcfg.eval_k0_mid_examples), int(self.bcfg.eval_k0_examples))
        if mid_k0_n == 0 and int(self.bcfg.eval_k0_examples) > 0:
            mid_k0_n = int(self.bcfg.eval_k0_examples)

        if self.step == 0 and mid_k0_n > 0:
            k0_0 = self.eval_k0(n_examples=mid_k0_n)
            if k0_0:
                self.metrics.log(0, k0_0)

        t_log, tok_log = time.time(), 0
        last: dict[str, Any] = {}
        self.opt.zero_grad(set_to_none=True)
        try:
            while self.step < steps and not self._stop:
                if st.lsa is not None and callable(getattr(st.lsa, "set_step", None)):
                    st.lsa.set_step(self.step)
                if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
                    st.loop.set_step(self.step, steps)
                depth = st.loop.sample_active_passes(self.rng) if st.loop is not None else 1

                is_anchor = (
                    int(self.bcfg.contiguous_lm_every) > 0
                    and self.stream is not None
                    and ((self.step + 1) % int(self.bcfg.contiguous_lm_every) == 0)
                )
                losses: dict[str, float] = {}
                for k in range(accum):
                    self.opt.set_boundary(k == accum - 1)
                    if is_anchor:
                        m_stats, n_tok = self._contiguous_lm_microstep(
                            accum=accum,
                            aux_w=aux_w,
                            impl=impl,
                            pass_pos=pass_pos,
                        )
                    else:
                        ep = self._sample_episode()
                        m_stats, n_tok = self._block_attn_microstep(ep, accum=accum, aux_w=aux_w)
                    for mk, mv in m_stats.items():
                        losses[mk] = losses.get(mk, 0.0) + mv / float(max(accum, 1))
                    self.tokens_seen += n_tok
                    tok_log += n_tok

                ostats = self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1

                if self.step % log_every == 0 or self.step == 1:
                    dt = time.time() - t_log
                    last = {
                        **{f"loss/{k}": v for k, v in losses.items()},
                        "depth": depth,
                        "tok_per_s": tok_log / max(dt, 1e-6),
                        "tokens_seen": self.tokens_seen,
                        "lr_factor": self.opt.lr_factor,
                        **{f"optim/{k}": v for k, v in ostats.items()},
                        **st.stats(),
                    }
                    if torch.cuda.is_available():
                        last["mem/peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    log.info(
                        "[block_attn_calib] step %d/%d total=%.4f kl=%.4f ce=%.4f anchor=%.0f d=%d %.0f tok/s %s",
                        self.step,
                        steps,
                        losses.get("total", 0.0),
                        losses.get("kl", 0.0),
                        losses.get("ce", 0.0),
                        losses.get("contiguous_anchor", 0.0),
                        depth,
                        last["tok_per_s"],
                        cuda_mem_brief(),
                    )
                    self.metrics.log(self.step, last)
                    t_log, tok_log = time.time(), 0

                if eval_every and self.step % eval_every == 0 and mid_k0_n > 0:
                    k0_ev = self.eval_k0(n_examples=mid_k0_n)
                    if k0_ev:
                        last.update(k0_ev)
                        self.metrics.log(self.step, k0_ev)
                    if tok_log == 0:
                        t_log = time.time()

                if ckpt_every and self.step % ckpt_every == 0:
                    self.save(metrics=last)
                    if tok_log == 0:
                        t_log = time.time()

                if budget and (time.time() - t_start) > budget:
                    log.warning(
                        "[block_attn_calib] time budget %.1fh reached: checkpointing and stopping",
                        budget / 3600.0,
                    )
                    break
        except KeyboardInterrupt:
            log.warning("[block_attn_calib] interrupted: saving checkpoint before exit")
            self.save(metrics=last)
            raise

        if hasattr(self.stream, "log_stats"):
            self.stream.log_stats()
        if st.frozen_experts_sha256 is not None:
            st.verify_frozen_experts()
        if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
            st.loop.set_step(self.step, steps)

        n_final_k0 = int(self.bcfg.eval_k0_examples)
        final_k0 = self.eval_k0(n_examples=n_final_k0) if n_final_k0 > 0 else {}
        if final_k0:
            self.metrics.log(self.step, final_k0)
        path = self.save(completed=(self.step >= steps), metrics={**last, **final_k0})
        return {
            "step": self.step,
            "checkpoint": path,
            "last": last,
            "k0": final_k0,
            "resume_mode": self.resume_mode,
        }


__all__ = [
    "MIN_SYNTHETIC_SEED_OFFSET",
    "BlockAttentionCalibrator",
    "BlockAttnCalibConfig",
    "BlockEpisode",
    "ModelK0Evaluator",
    "build_disjoint_k0_eval_examples",
    "build_synthetic_calib_episode",
    "chunked_kl_ce_loss",
    "slice_corpus_sequence_to_episode",
]
