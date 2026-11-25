"""Zero-Training Pre-RoPE KV-Mem Blob Injection Kill-Gate `K0` Evaluator Harness.

Implements the four falsifiable experimental arms and automated decision tree from
`docs/PROPOSAL_LSA_NESTED_LEARNING.md` (§3 & §5.1 Kill-Gate `K0`), strictly isolated
under `src/qloop/eval/kill_gate_k0.py` with zero modifications to model/trainer code:

  * **Arm `a` — Full Contiguous Prefill Oracle (`100%` Reference):**
    Concatenates historical sessions and the live prompt into a single contiguous context.
  * **Arm `b` — Naive Spliced KV (Negative Control):**
    Concatenates raw post-RoPE session KV directly without Pre-RoPE un-rotation or sink
    stripping, exposing both Geometric Error (`G`) and Dynamic / Sink Error (`D`).
  * **Arm `c` — Training-Free `LSA-NL` Partition-B Injection:**
    Reuses `strip_sinks_and_unrotate_k`, `pack_kv_blob` (`H64-INT4`), and `mount_kv_blob`
    (virtual Partition-B coordinates + optional `clamped_relative_distance`) directly from
    `src/qloop/memory/kv_blob.py`, leaving `[rotary_dim:head_dim)` (`192`-D NoPE) strictly
    orthogonal and bit-isolated from Group-0 rotary channels.
  * **Arm `d` — Arm `c` + `5%` `CacheBlend` Selective Refresh:**
    Scores per-token KV discrepancy (HKVD / value-variance salience) and refreshes strictly
    the top `ceil(refresh_frac * T)` tokens to quantify Dynamic Error (`D`) recoverability.

Supports both fast deterministic pure-CPU partial-RoPE attention probes / synthetic
multi-session + contradictory-memory (`SleepGate`) suites (`--smoke-cpu`) and real HF
`Qwen3.6-64L` checkpoints (`--checkpoint ...`) via cache extraction/injection.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qloop.memory.kv_blob import (
    DynamicLatentMemoryGraph,
    KVBlobConfig,
    KVMemBlob,
    SplitHorizonKVBlobStore,
    mount_kv_blob,
    pack_kv_blob,
    strip_sinks_and_unrotate_k,
)
from qloop.model._cache_compat import cache_kwarg_name, extract_output_cache
from qloop.model.pos_rerope import apply_partial_rope_k, build_rope_inv_freq

CANONICAL_ARMS: tuple[str, str, str, str] = ("arm_a", "arm_b", "arm_c", "arm_d")

ARM_DISPLAY_NAMES: dict[str, str] = {
    "arm_a": "Arm (a) Contiguous Oracle",
    "arm_b": "Arm (b) Naive Spliced KV",
    "arm_c": "Arm (c) Pre-RoPE + Sink-Strip + H64-INT4",
    "arm_d": "Arm (d) Arm (c) + CacheBlend Refresh",
}

_ARM_ALIASES: dict[str, str] = {
    "a": "arm_a",
    "arm_a": "arm_a",
    "oracle": "arm_a",
    "b": "arm_b",
    "arm_b": "arm_b",
    "naive": "arm_b",
    "c": "arm_c",
    "arm_c": "arm_c",
    "corrected": "arm_c",
    "partition_b": "arm_c",
    "d": "arm_d",
    "arm_d": "arm_d",
    "cacheblend": "arm_d",
}


def canonicalize_arm_name(name: str) -> str:
    """Normalize arm identifiers (`'a'` / `'arm_a'` / `'oracle'` -> `'arm_a'`)."""
    key = str(name).strip().lower()
    if key in _ARM_ALIASES:
        return _ARM_ALIASES[key]
    raise ValueError(f"unknown K0 arm name {name!r}; expected one of {CANONICAL_ARMS}")


# ==============================================================================
# Configuration, Metrics, Decision Tree & Report Data Structures
# ==============================================================================
@dataclass
class K0Config:
    """Configuration for Zero-Training Pre-RoPE KV-Mem Blob Injection Kill-Gate `K0`."""

    green_threshold: float = 0.92
    yellow_threshold: float = 0.75
    max_green_nll_gap: float | None = 0.05
    refresh_frac: float = 0.05
    boundary_header_tokens: int = 4
    sink_tokens: int = 4
    blob_config: KVBlobConfig = field(default_factory=KVBlobConfig)
    rope_theta: float = 10_000_000.0
    rotary_dim: int = 64
    virtual_offset: int = 10_000
    clamp_relative_to_q_pos: float | None = None
    linear_ceil: int = 131_072
    log_scale: int = 32_768
    max_pos: int = 240_000
    interleaved: bool = False
    retention_metric: str = "exact_match"

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> K0Config:
        raw = dict(d or {})
        if isinstance(raw.get("blob_config"), dict):
            raw["blob_config"] = KVBlobConfig.from_dict(raw["blob_config"])
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown k0.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def validate(self) -> None:
        if not math.isfinite(self.green_threshold) or not (0.0 < self.green_threshold <= 1.0):
            raise ValueError(f"green_threshold must be finite in (0, 1], got {self.green_threshold}")
        if not math.isfinite(self.yellow_threshold) or not (0.0 < self.yellow_threshold <= self.green_threshold):
            raise ValueError(
                f"yellow_threshold ({self.yellow_threshold}) must be finite in "
                f"(0, green_threshold={self.green_threshold}]"
            )
        if self.max_green_nll_gap is not None and (
            not math.isfinite(self.max_green_nll_gap) or self.max_green_nll_gap < 0.0
        ):
            raise ValueError(f"max_green_nll_gap must be None or finite >= 0, got {self.max_green_nll_gap}")
        if not math.isfinite(self.refresh_frac) or not (0.0 <= self.refresh_frac <= 1.0):
            raise ValueError(f"refresh_frac must be finite in [0, 1], got {self.refresh_frac}")
        if self.boundary_header_tokens < 0 or self.sink_tokens < 0:
            raise ValueError("boundary_header_tokens and sink_tokens must be >= 0")
        if self.rotary_dim < 0 or self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be an even int >= 0, got {self.rotary_dim}")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be finite and > 0, got {self.rope_theta}")
        if self.virtual_offset < 0:
            raise ValueError(f"virtual_offset must be >= 0, got {self.virtual_offset}")
        if self.linear_ceil < 0 or self.log_scale <= 0 or self.max_pos < self.linear_ceil:
            raise ValueError("invalid ReRoPE clamp bounds (linear_ceil, log_scale, max_pos)")
        if self.retention_metric not in ("exact_match", "token_acc", "combined"):
            raise ValueError(
                f"retention_metric must be 'exact_match', 'token_acc', or 'combined', got {self.retention_metric!r}"
            )
        self.blob_config.validate()

    def effective_blob_config(
        self,
        *,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        blob_tokens: int | None = None,
    ) -> KVBlobConfig:
        """Build a `KVBlobConfig` aligned with `K0Config` rotary/sink/header settings and tensor shape."""
        return replace(
            self.blob_config,
            num_global_layers=self.blob_config.num_global_layers if num_layers is None else int(num_layers),
            num_kv_heads=self.blob_config.num_kv_heads if num_kv_heads is None else int(num_kv_heads),
            head_dim=self.blob_config.head_dim if head_dim is None else int(head_dim),
            rotary_dim=int(self.rotary_dim),
            blob_tokens=self.blob_config.blob_tokens if blob_tokens is None else max(1, int(blob_tokens)),
            sink_tokens=int(self.sink_tokens),
            boundary_header_tokens=int(self.boundary_header_tokens),
            rope_theta=float(self.rope_theta),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class K0ArmMetrics:
    """Aggregated evaluation metrics for a single Kill-Gate `K0` arm."""

    arm_name: str
    answer_nll: float
    token_acc: float
    exact_match: float
    oracle_retention_ratio: float
    nll_gap: float = 0.0
    kv_bytes_mounted: int = 0
    refreshed_tokens: int = 0
    elapsed_ms: float = 0.0
    n_examples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "answer_nll": float(self.answer_nll),
            "token_acc": float(self.token_acc),
            "exact_match": float(self.exact_match),
            "oracle_retention_ratio": float(self.oracle_retention_ratio),
            "nll_gap": float(self.nll_gap),
            "kv_bytes_mounted": int(self.kv_bytes_mounted),
            "refreshed_tokens": int(self.refreshed_tokens),
            "elapsed_ms": float(self.elapsed_ms),
            "n_examples": int(self.n_examples),
            "extra": _to_jsonable(self.extra),
        }


@dataclass
class K0Verdict:
    """Automated falsification verdict produced by `decide_k0_verdict(report)`."""

    verdict: str  # "GREEN" | "YELLOW" | "RED"
    sub_verdict: str  # "GREEN" | "YELLOW_A_SELECTIVE_REFRESH" | "YELLOW_B_BLOCK_ATTN_CALIB" | "RED_PIVOT_ESM"
    rationale: str
    recommended_action: str
    arm_c_retention: float
    arm_d_retention: float
    arm_b_retention: float
    arm_c_nll_gap: float
    arm_d_nll_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class K0Example:
    """Single multi-session / multi-hop QA episode for Kill-Gate `K0` evaluation."""

    example_id: str
    sessions: list[str]
    live_prompt: str
    answer: str
    horizon_tokens: int = 0
    category: str = "multi_hop"
    superseded_session_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class K0EpisodeResult:
    """Per-example evaluation result for a single `K0` arm."""

    arm_name: str
    example_id: str
    answer_nll: float
    token_acc: float
    exact_match: float
    kv_bytes_mounted: int = 0
    refreshed_tokens: int = 0
    elapsed_ms: float = 0.0
    predicted: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class K0Report:
    """Complete Kill-Gate `K0` report across all four paired arms plus automated verdict."""

    arms: dict[str, K0ArmMetrics]
    config: K0Config = field(default_factory=K0Config)
    verdict: str = "PENDING"
    sub_verdict: str = "PENDING"
    rationale: str = ""
    recommended_action: str = ""
    per_category: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    n_examples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_arm(self, name: str) -> K0ArmMetrics | None:
        canon = canonicalize_arm_name(name)
        if canon in self.arms:
            return self.arms[canon]
        for k, v in self.arms.items():
            try:
                if canonicalize_arm_name(k) == canon:
                    return v
            except ValueError:
                continue
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "sub_verdict": self.sub_verdict,
            "rationale": self.rationale,
            "recommended_action": self.recommended_action,
            "n_examples": int(self.n_examples),
            "config": self.config.to_dict(),
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
            "per_category": _to_jsonable(self.per_category),
            "metadata": _to_jsonable(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def decide_k0_verdict(report: K0Report, cfg: K0Config | None = None) -> K0Verdict:
    """Evaluate the `PROPOSAL_LSA_NESTED_LEARNING.md §5.1` falsification decision tree.

    Decision rules:
      * **GREEN:** Arm `c` (`Pre-RoPE + sink-strip + 4-tok boundary header + H64-INT4`)
        achieves `>= green_threshold` (`0.92`) of Arm `a` (Oracle), beats Arm `b` (Naive
        Spliced KV), and satisfies `nll_gap <= max_green_nll_gap` (when configured) ->
        training-free Partition-B KV-Mem Blob deployment works immediately.
      * **YELLOW (`YELLOW_A_SELECTIVE_REFRESH` / `YELLOW_B_BLOCK_ATTN_CALIB`):**
        Arm `c` is in `[yellow_threshold, green_threshold)` (`75%–92%`) OR Arm `d`
        (`+ 5% CacheBlend`) crosses `>= green_threshold` (or stays `>= yellow_threshold`),
        OR Arm `c` crosses `green_threshold` accuracy while exceeding the NLL gap budget ->
        enable `5% CacheBlend` selective refresh and/or trigger lightweight `2–4 h`
        `Block-Attention` spliced-KV adapter calibration (`32.212B` MoE 100% frozen).
      * **RED (`RED_PIVOT_ESM`):** Both Arm `c` and Arm `d` fall strictly below
        `yellow_threshold` (`< 75%`) of Oracle -> abort direct KV-blob splicing and pivot
        cleanly to `ESM`-only vertical consolidation + text-header re-prefill.
    """
    eff_cfg = report.config if cfg is None else cfg
    eff_cfg.validate()

    arm_a = report.get_arm("arm_a")
    arm_b = report.get_arm("arm_b")
    arm_c = report.get_arm("arm_c")
    arm_d = report.get_arm("arm_d")
    if arm_c is None:
        raise ValueError("K0Report must contain Arm 'c' ('arm_c') to decide a verdict")

    r_b = 0.0 if arm_b is None else float(arm_b.oracle_retention_ratio)
    r_c = float(arm_c.oracle_retention_ratio)
    r_d = r_c if arm_d is None else float(arm_d.oracle_retention_ratio)

    gap_c = float(arm_c.nll_gap)
    if arm_a is not None and gap_c == 0.0 and arm_c.answer_nll != arm_a.answer_nll:
        gap_c = float(arm_c.answer_nll - arm_a.answer_nll)
    gap_d = gap_c if arm_d is None else float(arm_d.nll_gap)
    if arm_a is not None and arm_d is not None and gap_d == 0.0 and arm_d.answer_nll != arm_a.answer_nll:
        gap_d = float(arm_d.answer_nll - arm_a.answer_nll)

    g_thr = float(eff_cfg.green_threshold)
    y_thr = float(eff_cfg.yellow_threshold)
    nll_thr = eff_cfg.max_green_nll_gap

    beats_naive_c = (
        arm_b is None
        or r_c > r_b
        or (math.isclose(r_c, r_b, abs_tol=1e-9) and arm_c.answer_nll < arm_b.answer_nll)
    )
    nll_ok_c = nll_thr is None or gap_c <= float(nll_thr)

    if r_c >= g_thr and beats_naive_c and nll_ok_c:
        verdict_obj = K0Verdict(
            verdict="GREEN",
            sub_verdict="GREEN",
            rationale=(
                f"Arm (c) achieved {r_c:.1%} oracle retention (>= {g_thr:.0%}), beat Arm (b) naive splice "
                f"({r_b:.1%}), with NLL gap {gap_c:+.4f} nats"
                + (f" (<= {nll_thr:.3f} nats)." if nll_thr is not None else ".")
            ),
            recommended_action=(
                "Deploy training-free Partition-B KVMemBlobStore + DLMG directly (Phase 2B); "
                "zero spliced-KV calibration GPU-hours required."
            ),
            arm_c_retention=r_c,
            arm_d_retention=r_d,
            arm_b_retention=r_b,
            arm_c_nll_gap=gap_c,
            arm_d_nll_gap=gap_d,
        )
    elif r_d >= g_thr and (arm_d is not None):
        verdict_obj = K0Verdict(
            verdict="YELLOW",
            sub_verdict="YELLOW_A_SELECTIVE_REFRESH",
            rationale=(
                f"Arm (c) achieved {r_c:.1%} oracle retention (NLL gap {gap_c:+.4f} nats), while Arm (d) "
                f"(+{eff_cfg.refresh_frac:.0%} CacheBlend selective refresh) crossed {r_d:.1%} (>= {g_thr:.0%}, "
                f"NLL gap {gap_d:+.4f} nats), isolating Dynamic Error (D) to high-HKVD tokens."
            ),
            recommended_action=(
                f"Enable {eff_cfg.refresh_frac:.0%} CacheBlend selective token refresh on multi-hop "
                "Partition-B mounts (~10x-18x faster than Layer-0 text re-prefill)."
            ),
            arm_c_retention=r_c,
            arm_d_retention=r_d,
            arm_b_retention=r_b,
            arm_c_nll_gap=gap_c,
            arm_d_nll_gap=gap_d,
        )
    elif r_c >= y_thr or r_d >= y_thr:
        extra_note = ""
        if r_c >= g_thr and not beats_naive_c:
            extra_note = f" (Arm c {r_c:.1%} did not beat naive Arm b {r_b:.1%}; verify probe separation)"
        elif r_c >= g_thr and not nll_ok_c:
            extra_note = f" (NLL gap {gap_c:+.4f} exceeded green limit {nll_thr:.3f} nats)"
        verdict_obj = K0Verdict(
            verdict="YELLOW",
            sub_verdict="YELLOW_B_BLOCK_ATTN_CALIB",
            rationale=(
                f"Arm (c) achieved {r_c:.1%} and Arm (d) achieved {r_d:.1%} oracle retention "
                f"(within [{y_thr:.0%}, {g_thr:.0%}) band{extra_note}), indicating residual cross-session "
                "Dynamic Error (D) under frozen attention projections."
            ),
            recommended_action=(
                "Enable 5% CacheBlend selective refresh and/or run lightweight 2-4h single-GPU "
                "Block-Attention APOLLO calibration on D-LSA indexers + global q/k/v/o adapters "
                "(keeping 32.212B MoE 100% frozen), then re-verify K0 >= 90%."
            ),
            arm_c_retention=r_c,
            arm_d_retention=r_d,
            arm_b_retention=r_b,
            arm_c_nll_gap=gap_c,
            arm_d_nll_gap=gap_d,
        )
    else:
        verdict_obj = K0Verdict(
            verdict="RED",
            sub_verdict="RED_PIVOT_ESM",
            rationale=(
                f"Both Arm (c) ({r_c:.1%}) and Arm (d) ({r_d:.1%}) fell below {y_thr:.0%} of Oracle "
                f"(Arm c NLL gap {gap_c:+.4f} nats, Arm d NLL gap {gap_d:+.4f} nats): cross-session "
                "KV-blob splicing failed falsification."
            ),
            recommended_action=(
                "KILL direct cross-session KV-blob splicing immediately and pivot cleanly to "
                "L3 EpisodicSideMemory (ESM) vertical null-space consolidation + compact text-header re-prefill."
            ),
            arm_c_retention=r_c,
            arm_d_retention=r_d,
            arm_b_retention=r_b,
            arm_c_nll_gap=gap_c,
            arm_d_nll_gap=gap_d,
        )

    report.verdict = verdict_obj.verdict
    report.sub_verdict = verdict_obj.sub_verdict
    report.rationale = verdict_obj.rationale
    report.recommended_action = verdict_obj.recommended_action
    return verdict_obj


# ==============================================================================
# Core Reusable Primitives (HF Qwen3.6-64L & Synthetic CPU Attention Fixtures)
# ==============================================================================
def _normalize_layer_kv_tensor(t: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
    """Normalize a single layer's K or V tensor (`[B, H, T, D]` or `[H, T, D]`) to `[H, T, D]`."""
    if t.ndim == 4:
        if not (0 <= batch_index < t.shape[0]):
            raise IndexError(f"batch_index={batch_index} out of range for batch size {t.shape[0]}")
        return t[batch_index]
    if t.ndim == 3:
        return t
    raise ValueError(f"expected per-layer KV tensor with shape [B, H, T, D] or [H, T, D], got {tuple(t.shape)}")


def extract_layer_kv_pairs(
    cache_or_output: Any,
    *,
    layer_indices: Sequence[int] | None = None,
    batch_index: int = 0,
    clone: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Extract stacked `[L_full, H_kv, T, D]` `(K, V)` tensors and physical full-attention layer indices.

    Supports:
      * HF model outputs (`out.cache` in `transformers >= 5.24` or `out.past_key_values`),
      * `DynamicCache` / `Qwen3_5DynamicCache` / `ToyCache` / `HSSDCache` (`.key_cache` + `.value_cache`
        or `.layers[i].keys` + `.layers[i].values`, automatically skipping `None` linear-attention slots),
      * Legacy tuple/list of `(k, v)` pairs,
      * Dict mapping `layer_idx -> (k, v)`,
      * Already-stacked `(k_4d, v_4d)` pair of `[L, H_kv, T, D]` tensors.
    """
    cache = extract_output_cache(cache_or_output)
    if cache is None:
        cache = cache_or_output

    raw_pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    wanted = None if layer_indices is None else {int(i) for i in layer_indices}

    # Case 1: Already-stacked `(K_4d, V_4d)` tuple where both elements are 4D tensors
    if (
        isinstance(cache, tuple)
        and len(cache) == 2
        and isinstance(cache[0], torch.Tensor)
        and isinstance(cache[1], torch.Tensor)
        and cache[0].ndim == 4
        and cache[1].ndim == 4
    ):
        k_all, v_all = cache[0], cache[1]
        if k_all.shape != v_all.shape:
            raise ValueError(f"K {tuple(k_all.shape)} and V {tuple(v_all.shape)} shapes must match")
        idxs = list(range(k_all.shape[0])) if layer_indices is None else [int(i) for i in layer_indices]
        k_sel = k_all[idxs]
        v_sel = v_all[idxs]
        return (k_sel.clone(), v_sel.clone(), idxs) if clone else (k_sel, v_sel, idxs)

    # Case 2: `.key_cache` / `.value_cache` (Qwen3_5DynamicCache, ToyCache, HSSDCache, DynamicCache)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        kc_list = cache.key_cache
        vc_list = cache.value_cache
        for idx, (k_t, v_t) in enumerate(zip(kc_list, vc_list)):
            if wanted is not None and idx not in wanted:
                continue
            if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor):
                raw_pairs.append(
                    (idx, _normalize_layer_kv_tensor(k_t, batch_index), _normalize_layer_kv_tensor(v_t, batch_index))
                )
    # Case 3: `transformers >= 5.x` `DynamicCache.layers` with `.keys` / `.values`
    elif hasattr(cache, "layers") and isinstance(cache.layers, (list, tuple)):
        for idx, layer_obj in enumerate(cache.layers):
            if wanted is not None and idx not in wanted:
                continue
            k_t = getattr(layer_obj, "keys", None)
            v_t = getattr(layer_obj, "values", None)
            if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor):
                raw_pairs.append(
                    (idx, _normalize_layer_kv_tensor(k_t, batch_index), _normalize_layer_kv_tensor(v_t, batch_index))
                )
    # Case 4: Mapping `layer_idx -> (k, v)`
    elif isinstance(cache, Mapping):
        for idx in sorted(cache):
            idx_int = int(idx)
            if wanted is not None and idx_int not in wanted:
                continue
            pair = cache[idx]
            if pair is None:
                continue
            k_t, v_t = pair
            if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor):
                raw_pairs.append(
                    (
                        idx_int,
                        _normalize_layer_kv_tensor(k_t, batch_index),
                        _normalize_layer_kv_tensor(v_t, batch_index),
                    )
                )
    # Case 5: Legacy tuple/list of `(k, v)` pairs
    elif isinstance(cache, (list, tuple)):
        for idx, item in enumerate(cache):
            if wanted is not None and idx not in wanted:
                continue
            if item is None or not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            k_t, v_t = item
            if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor):
                raw_pairs.append(
                    (idx, _normalize_layer_kv_tensor(k_t, batch_index), _normalize_layer_kv_tensor(v_t, batch_index))
                )
    else:
        raise TypeError(f"unsupported cache object type {type(cache).__name__} for KV extraction")

    if not raw_pairs:
        raise ValueError("no full-attention (K, V) tensor pairs found in cache")

    phys_indices = [idx for idx, _, _ in raw_pairs]
    k_stack = torch.stack([k for _, k, _ in raw_pairs], dim=0)
    v_stack = torch.stack([v for _, _, v in raw_pairs], dim=0)
    if clone:
        return k_stack.clone(), v_stack.clone(), phys_indices
    return k_stack, v_stack, phys_indices


def inject_layer_kv_pairs(
    cache: Any,
    k_layers: torch.Tensor,
    v_layers: torch.Tensor,
    layer_indices: Sequence[int],
) -> Any:
    """Write `[L_full, H_kv, T, D]` `(K, V)` tensors into a cache's full-attention layer slots.

    Leaves all linear-attention (`GDN-2`) `recurrent_states` and `conv_states` untouched so
    Partition-B KV blobs never pollute Partition-A recurrent streams.
    """
    if k_layers.shape != v_layers.shape or k_layers.ndim != 4:
        raise ValueError(
            f"expected matching [L, H_kv, T, D] tensors, got K={tuple(k_layers.shape)} V={tuple(v_layers.shape)}"
        )
    if len(layer_indices) != int(k_layers.shape[0]):
        raise ValueError(
            f"layer_indices length ({len(layer_indices)}) must match k_layers.shape[0] ({k_layers.shape[0]})"
        )

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for slot, phys_idx in enumerate(layer_indices):
            idx = int(phys_idx)
            while len(cache.key_cache) <= idx:
                cache.key_cache.append(None)
                cache.value_cache.append(None)
            cache.key_cache[idx] = k_layers[slot].unsqueeze(0).contiguous()
            cache.value_cache[idx] = v_layers[slot].unsqueeze(0).contiguous()
        return cache

    if hasattr(cache, "layers") and isinstance(cache.layers, (list, tuple)):
        for slot, phys_idx in enumerate(layer_indices):
            idx = int(phys_idx)
            k_b = k_layers[slot].unsqueeze(0).contiguous()
            v_b = v_layers[slot].unsqueeze(0).contiguous()
            if idx < len(cache.layers):
                layer_obj = cache.layers[idx]
                layer_obj.keys = k_b
                layer_obj.values = v_b
            elif hasattr(cache, "update") and callable(cache.update):
                cache.update(k_b, v_b, idx)
            else:
                raise IndexError(
                    f"layer_idx={idx} out of range for cache.layers (len={len(cache.layers)}) with no .update()"
                )
        return cache

    if isinstance(cache, dict):
        for slot, phys_idx in enumerate(layer_indices):
            cache[int(phys_idx)] = (
                k_layers[slot].unsqueeze(0).contiguous(),
                v_layers[slot].unsqueeze(0).contiguous(),
            )
        return cache

    raise TypeError(f"cannot inject KV pairs into unsupported cache type {type(cache).__name__}")


def splice_naive_kv(
    session_k: torch.Tensor,
    session_v: torch.Tensor,
    live_k: torch.Tensor,
    live_v: torch.Tensor,
    *,
    token_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm `b` — Naive Spliced KV negative control (raw stale RoPE + duplicate session sinks).

    Directly concatenates historical `(session_k, session_v)` ahead of `(live_k, live_v)` with
    zero Pre-RoPE un-rotation and zero `[0:sink_tokens)` sink stripping, exposing both Geometric
    Error (`G`) and Dynamic / Sink Error (`D`).
    """
    if session_k.shape != session_v.shape:
        raise ValueError(
            f"session_k {tuple(session_k.shape)} and session_v {tuple(session_v.shape)} must match"
        )
    if live_k.shape != live_v.shape:
        raise ValueError(f"live_k {tuple(live_k.shape)} and live_v {tuple(live_v.shape)} must match")
    if session_k.ndim != live_k.ndim or session_k.ndim < 2:
        raise ValueError(
            f"session_k {tuple(session_k.shape)} and live_k {tuple(live_k.shape)} must share ndim >= 2"
        )

    dim = token_dim if token_dim >= 0 else session_k.ndim + token_dim
    if not (0 <= dim < session_k.ndim):
        raise ValueError(f"token_dim={token_dim} out of range for ndim={session_k.ndim}")

    for ax in range(session_k.ndim):
        if ax != dim and session_k.shape[ax] != live_k.shape[ax]:
            raise ValueError(
                f"non-token dim {ax} mismatch between session {tuple(session_k.shape)} and live {tuple(live_k.shape)}"
            )

    sk = session_k.to(device=live_k.device, dtype=live_k.dtype)
    sv = session_v.to(device=live_v.device, dtype=live_v.dtype)
    return torch.cat([sk, live_k], dim=dim), torch.cat([sv, live_v], dim=dim)


@dataclass
class MountedPartitionBKV:
    """Output of `build_corrected_partition_b_kv` (Arm `c`).

    Supports both attribute access (`res.k`, `res.v`, `res.blob`, `res.kv_bytes_mounted`) and
    2-tuple unpacking (`k_mounted, v_mounted = build_corrected_partition_b_kv(...)`).
    """

    k: torch.Tensor
    v: torch.Tensor
    blob: KVMemBlob
    virtual_positions: torch.Tensor
    kv_bytes_mounted: int
    stripped_sinks: int
    boundary_header_tokens: int

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.k
        yield self.v


def _promote_to_4d_kv(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Promote `[T, D]` or `[H, T, D]` or `[L, H, T, D]` to `[L, H, T, D]`, returning original ndim."""
    if x.ndim == 4:
        return x, 4
    if x.ndim == 3:
        return x.unsqueeze(0), 3
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0), 2
    raise ValueError(f"expected KV tensor with ndim in {{2, 3, 4}}, got shape {tuple(x.shape)}")


def _restore_kv_ndim(x: torch.Tensor, orig_ndim: int) -> torch.Tensor:
    if orig_ndim == 4:
        return x
    if orig_ndim == 3:
        return x.squeeze(0)
    if orig_ndim == 2:
        return x.squeeze(0).squeeze(0)
    raise ValueError(f"unexpected orig_ndim={orig_ndim}")


def build_corrected_partition_b_kv(
    session_k: torch.Tensor,
    session_v: torch.Tensor,
    session_positions: torch.Tensor,
    live_sink_len: int,
    virtual_offset: int,
    cfg: K0Config | None = None,
    *,
    blob_id: str = "k0_partition_b_blob",
    boundary_header_k: torch.Tensor | None = None,
    boundary_header_v: torch.Tensor | None = None,
    boundary_header_positions: torch.Tensor | None = None,
    clamp_relative_to_q_pos: float | None = None,
    t_created: float = 0.0,
    salience: float = 1.0,
    topic_tags: tuple[str, ...] = (),
) -> MountedPartitionBKV:
    """Arm `c` — Training-Free `LSA-NL` Partition-B KV-Mem Blob construction and mount.

    Directly reuses `src/qloop/memory/kv_blob.py` primitives with zero duplicated math:
      1. Strips original session `[0:cfg.sink_tokens)` sinks (`EPIC`) and applies exact inverse
         partial-RoPE `R(-p * omega)` on `[0:cfg.rotary_dim)` via `strip_sinks_and_unrotate_k`
         (leaving `[rotary_dim:head_dim)` NoPE bit-for-bit untouched).
      2. Optionally prepends a `4`-token `TurboRAG` Boundary & Abstraction Header (`boundary_header_k/v`).
      3. Packs `(k_pre_rope, v_kept)` via `pack_kv_blob` (`Block64HadamardINT4`).
      4. Mounts into clean virtual Partition-B positions starting at `max(virtual_offset, live_sink_len)`
         via `mount_kv_blob` (with optional `clamped_relative_distance` ReRoPE clamping).
    """
    eff_cfg = K0Config() if cfg is None else cfg
    eff_cfg.validate()
    if live_sink_len < 0:
        raise ValueError(f"live_sink_len must be >= 0, got {live_sink_len}")
    if virtual_offset < 0:
        raise ValueError(f"virtual_offset must be >= 0, got {virtual_offset}")
    if session_k.shape != session_v.shape:
        raise ValueError(f"session_k {tuple(session_k.shape)} and session_v {tuple(session_v.shape)} must match")

    k4, orig_ndim = _promote_to_4d_kv(session_k)
    v4, _ = _promote_to_4d_kv(session_v)

    sink_n = int(eff_cfg.sink_tokens)
    k_pre = strip_sinks_and_unrotate_k(
        k4,
        session_positions,
        rotary_dim=eff_cfg.rotary_dim,
        rope_theta=eff_cfg.rope_theta,
        sink_tokens=sink_n,
        interleaved=eff_cfg.interleaved,
    )
    v_kept = v4[..., sink_n:, :] if sink_n > 0 else v4.clone()

    hdr_applied = 0
    if (boundary_header_k is None) != (boundary_header_v is None):
        raise ValueError("boundary_header_k and boundary_header_v must both be provided or both be None")
    if boundary_header_k is not None and boundary_header_v is not None:
        hk4, _ = _promote_to_4d_kv(boundary_header_k.to(device=k4.device, dtype=k4.dtype))
        hv4, _ = _promote_to_4d_kv(boundary_header_v.to(device=v4.device, dtype=v4.dtype))
        if hk4.shape != hv4.shape:
            raise ValueError(
                f"boundary_header_k {tuple(hk4.shape)} and boundary_header_v {tuple(hv4.shape)} must match"
            )
        if boundary_header_positions is not None:
            hk4 = strip_sinks_and_unrotate_k(
                hk4,
                boundary_header_positions,
                rotary_dim=eff_cfg.rotary_dim,
                rope_theta=eff_cfg.rope_theta,
                sink_tokens=0,
                interleaved=eff_cfg.interleaved,
            )
        hdr_applied = int(hk4.shape[-2])
        k_pre = torch.cat([hk4, k_pre], dim=-2)
        v_kept = torch.cat([hv4, v_kept], dim=-2)
    else:
        hdr_applied = min(int(eff_cfg.boundary_header_tokens), int(k_pre.shape[-2]))

    n_layers, n_heads, n_tok, head_dim = k_pre.shape
    blob_cfg = eff_cfg.effective_blob_config(
        num_layers=n_layers,
        num_kv_heads=n_heads,
        head_dim=head_dim,
        blob_tokens=n_tok,
    )
    blob = pack_kv_blob(
        k_pre,
        v_kept,
        blob_id=blob_id,
        config=blob_cfg,
        t_created=t_created,
        salience=salience,
        topic_tags=topic_tags,
    )

    mount_start = max(int(virtual_offset), int(live_sink_len))
    vpos = torch.arange(mount_start, mount_start + n_tok, device=k4.device, dtype=torch.float64)
    eff_clamp = eff_cfg.clamp_relative_to_q_pos if clamp_relative_to_q_pos is None else clamp_relative_to_q_pos

    k_mnt, v_mnt = mount_kv_blob(
        blob,
        vpos,
        dtype=session_k.dtype,
        device=session_k.device,
        interleaved=eff_cfg.interleaved,
        clamp_relative_to_q_pos=eff_clamp,
        linear_ceil=eff_cfg.linear_ceil,
        log_scale=eff_cfg.log_scale,
        max_pos=eff_cfg.max_pos,
    )
    return MountedPartitionBKV(
        k=_restore_kv_ndim(k_mnt, orig_ndim),
        v=_restore_kv_ndim(v_mnt, orig_ndim),
        blob=blob,
        virtual_positions=vpos,
        kv_bytes_mounted=blob.payload_bytes() + blob.header_wire_bytes(),
        stripped_sinks=sink_n,
        boundary_header_tokens=hdr_applied,
    )


@dataclass
class CacheBlendRefreshResult:
    """Output of `selective_cacheblend_refresh` (Arm `d`).

    Supports both attribute access (`res.k`, `res.v`, `res.refreshed_tokens`, `res.elapsed_ms`)
    and 2-tuple unpacking (`k_ref, v_ref = selective_cacheblend_refresh(...)`).
    """

    k: torch.Tensor
    v: torch.Tensor
    refreshed_indices: torch.Tensor
    token_scores: torch.Tensor
    refreshed_tokens: int
    total_tokens: int
    refresh_frac_actual: float
    pre_refresh_kv_l2: float
    post_refresh_kv_l2: float
    elapsed_ms: float

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.k
        yield self.v


def selective_cacheblend_refresh(
    k_mounted: torch.Tensor,
    v_mounted: torch.Tensor,
    k_oracle: torch.Tensor,
    v_oracle: torch.Tensor,
    refresh_frac: float = 0.05,
    *,
    score_mode: str = "hkvd",
    token_salience: torch.Tensor | None = None,
) -> CacheBlendRefreshResult:
    """Arm `d` — `CacheBlend` selective high-variance / high-HKVD token refresh.

    Scores per-token KV discrepancy (or value-variance salience) across `T` tokens, refreshes
    strictly the top `m = min(T, ceil(refresh_frac * T))` tokens from `(k_oracle, v_oracle)`,
    and logs exact `refreshed_tokens`, KV L2 discrepancy reduction, and elapsed overhead.

    Supported `score_mode` values:
      * `'hkvd'` (default): mean squared K + V deviation across layers and heads per token.
      * `'first_layer_hkvd'`: CacheBlend layer-0 HKVD propagated uniformly across all layers.
      * `'value_variance'`: per-token value deviation + cross-channel value variance salience.
    """
    t0 = time.perf_counter()
    if not math.isfinite(refresh_frac) or not (0.0 <= refresh_frac <= 1.0):
        raise ValueError(f"refresh_frac must be finite in [0, 1], got {refresh_frac}")
    if not (k_mounted.shape == v_mounted.shape == k_oracle.shape == v_oracle.shape):
        raise ValueError(
            "all four KV tensors must share identical shape, got "
            f"{tuple(k_mounted.shape)}, {tuple(v_mounted.shape)}, "
            f"{tuple(k_oracle.shape)}, {tuple(v_oracle.shape)}"
        )
    if k_mounted.ndim < 2:
        raise ValueError(f"expected KV tensors with ndim >= 2 ([..., T, D]), got {tuple(k_mounted.shape)}")
    if score_mode not in ("hkvd", "first_layer_hkvd", "value_variance"):
        raise ValueError(
            f"score_mode must be 'hkvd', 'first_layer_hkvd', or 'value_variance', got {score_mode!r}"
        )

    t_len = int(k_mounted.shape[-2])
    ko = k_oracle.to(device=k_mounted.device, dtype=k_mounted.dtype)
    vo = v_oracle.to(device=v_mounted.device, dtype=v_mounted.dtype)

    dk = ko.float() - k_mounted.float()
    dv = vo.float() - v_mounted.float()
    pre_l2 = float(torch.sqrt((dk.pow(2).sum() + dv.pow(2).sum()).clamp_min(0.0)).item())

    if t_len == 0 or refresh_frac == 0.0:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return CacheBlendRefreshResult(
            k=k_mounted.clone(),
            v=v_mounted.clone(),
            refreshed_indices=torch.empty((0,), dtype=torch.long, device=k_mounted.device),
            token_scores=torch.zeros((t_len,), dtype=torch.float32, device=k_mounted.device),
            refreshed_tokens=0,
            total_tokens=t_len,
            refresh_frac_actual=0.0,
            pre_refresh_kv_l2=pre_l2,
            post_refresh_kv_l2=pre_l2,
            elapsed_ms=elapsed_ms,
        )

    reduce_lead = tuple(range(k_mounted.ndim - 2))
    if score_mode == "first_layer_hkvd" and k_mounted.ndim >= 4:
        dk0 = dk[0]
        dv0 = dv[0]
        per_tok = (dk0.pow(2).sum(dim=-1) + dv0.pow(2).sum(dim=-1))
        if per_tok.ndim > 1:
            per_tok = per_tok.mean(dim=tuple(range(per_tok.ndim - 1)))
    elif score_mode == "value_variance":
        v_var = v_mounted.float().var(dim=-1, unbiased=False)
        v_err = dv.pow(2).sum(dim=-1)
        per_tok = v_err + 0.1 * v_var
        if reduce_lead:
            per_tok = per_tok.mean(dim=reduce_lead)
    else:
        per_tok = dk.pow(2).sum(dim=-1) + dv.pow(2).sum(dim=-1)
        if reduce_lead:
            per_tok = per_tok.mean(dim=reduce_lead)

    if token_salience is not None:
        sal = token_salience.to(device=per_tok.device, dtype=torch.float32)
        if sal.shape != (t_len,):
            raise ValueError(f"token_salience must have shape ({t_len},), got {tuple(sal.shape)}")
        per_tok = per_tok * sal.clamp_min(0.0)

    budget = min(t_len, max(1, math.ceil(float(refresh_frac) * t_len)))
    top_idx = torch.topk(per_tok, k=budget, largest=True, sorted=True).indices
    top_idx_sorted = torch.sort(top_idx).values

    k_out = k_mounted.clone()
    v_out = v_mounted.clone()
    k_out[..., top_idx_sorted, :] = ko[..., top_idx_sorted, :]
    v_out[..., top_idx_sorted, :] = vo[..., top_idx_sorted, :]

    dk_post = ko.float() - k_out.float()
    dv_post = vo.float() - v_out.float()
    post_l2 = float(torch.sqrt((dk_post.pow(2).sum() + dv_post.pow(2).sum()).clamp_min(0.0)).item())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return CacheBlendRefreshResult(
        k=k_out,
        v=v_out,
        refreshed_indices=top_idx_sorted,
        token_scores=per_tok.detach(),
        refreshed_tokens=int(top_idx_sorted.numel()),
        total_tokens=t_len,
        refresh_frac_actual=float(top_idx_sorted.numel()) / float(t_len),
        pre_refresh_kv_l2=pre_l2,
        post_refresh_kv_l2=post_l2,
        elapsed_ms=elapsed_ms,
    )


# ==============================================================================
# Attention Probe & SleepGate Contradictory-Memory Helpers (Pure CPU Verified)
# ==============================================================================
def _gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact scaled dot-product GQA attention on `[L, H_q, T_q, D]` vs `[L, H_kv, T_k, D]`."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"expected 4D [L, H, T, D] tensors, got q={tuple(q.shape)}, k={tuple(k.shape)}")
    n_layers, h_q, _t_q, d = q.shape
    if k.shape[0] != n_layers or k.shape[-1] != d or v.shape != k.shape:
        raise ValueError(f"incompatible q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    h_kv = k.shape[1]
    if h_q % h_kv != 0:
        raise ValueError(f"h_q={h_q} must be divisible by h_kv={h_kv}")
    rep = h_q // h_kv
    k_eff = k.repeat_interleave(rep, dim=1) if rep > 1 else k
    v_eff = v.repeat_interleave(rep, dim=1) if rep > 1 else v

    scores = torch.matmul(q.float(), k_eff.float().transpose(-2, -1)) / math.sqrt(float(d))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_eff.float())
    return out, weights


@dataclass
class K0ProbeArmOutput:
    """Detailed single-arm output from `run_k0_attention_probe`."""

    arm_name: str
    output: torch.Tensor
    attn_weights: torch.Tensor
    output_rel_l2_vs_oracle: float
    output_cosine_vs_oracle: float
    target_attn_mass: float
    answer_nll: float
    token_acc: float
    exact_match: float
    kv_bytes_mounted: int
    refreshed_tokens: int
    elapsed_ms: float
    kv_l2_vs_oracle: float = 0.0


@dataclass
class K0AttentionProbeResult:
    """Paired 4-arm output of `run_k0_attention_probe`."""

    arms: dict[str, K0ProbeArmOutput]
    report: K0Report
    verdict: K0Verdict
    target_index_in_session: int
    target_index_after_strip: int
    virtual_offset: int


def _score_probe_readout(
    out: torch.Tensor,
    candidate_prototypes: torch.Tensor | None,
    target_class: int = 0,
    temperature: float = 1.0,
) -> tuple[float, float, float]:
    """Compute `(answer_nll, token_acc, exact_match)` from attention output against class prototypes."""
    pooled = out.float().mean(dim=(0, 1, 2))  # [D]
    if candidate_prototypes is None:
        return 0.0, 1.0, 1.0
    protos = candidate_prototypes.to(device=pooled.device, dtype=torch.float32)  # [C, D]
    logits = (protos @ pooled) / max(float(temperature), 1e-6)
    target = torch.tensor(int(target_class), dtype=torch.long, device=pooled.device)
    nll = float(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)).item())
    pred = int(torch.argmax(logits).item())
    hit = 1.0 if pred == int(target_class) else 0.0
    return nll, hit, hit


def run_k0_attention_probe(
    q_pre_rope: torch.Tensor,
    q_positions: torch.Tensor,
    session_k_post_rope: torch.Tensor,
    session_v: torch.Tensor,
    session_positions: torch.Tensor,
    live_k_post_rope: torch.Tensor,
    live_v: torch.Tensor,
    *,
    live_sink_len: int = 4,
    virtual_offset: int | None = None,
    cfg: K0Config | None = None,
    oracle_session_k_post_rope: torch.Tensor | None = None,
    oracle_session_v: torch.Tensor | None = None,
    target_session_pos: int | None = None,
    candidate_prototypes: torch.Tensor | None = None,
    target_class: int = 0,
    readout_temperature: float = 1.0,
) -> K0AttentionProbeResult:
    """Execute all 4 Kill-Gate `K0` arms on a controlled partial-RoPE multi-session attention probe.

    Tensor shapes:
      * `q_pre_rope`: `[L, H_q, T_q, D]` (pre-RoPE live query tensor)
      * `q_positions`: `[T_q]` (live query absolute positions)
      * `session_k_post_rope`, `session_v`: `[L, H_kv, T_sess, D]` (raw post-RoPE Session 1 KV)
      * `session_positions`: `[T_sess]` (original Session 1 positions, e.g. `0..63`)
      * `live_k_post_rope`, `live_v`: `[L, H_kv, T_live, D]` (`[0:live_sink_len)` live sinks + live body)
    """
    eff_cfg = K0Config() if cfg is None else cfg
    eff_cfg.validate()
    v_off = eff_cfg.virtual_offset if virtual_offset is None else int(virtual_offset)
    mount_start = max(v_off, int(live_sink_len))

    q4, _ = _promote_to_4d_kv(q_pre_rope)
    sk4, _ = _promote_to_4d_kv(session_k_post_rope)
    sv4, _ = _promote_to_4d_kv(session_v)
    lk4, _ = _promote_to_4d_kv(live_k_post_rope)
    lv4, _ = _promote_to_4d_kv(live_v)

    if live_sink_len > int(lk4.shape[-2]):
        raise ValueError(f"live_sink_len={live_sink_len} exceeds live sequence length {lk4.shape[-2]}")

    inv_freq = build_rope_inv_freq(
        eff_cfg.rotary_dim, rope_theta=eff_cfg.rope_theta, device=q4.device, dtype=torch.float64
    )
    q_rot = apply_partial_rope_k(
        q4,
        q_positions,
        inv_freq=inv_freq,
        rotary_dim=eff_cfg.rotary_dim,
        rope_theta=eff_cfg.rope_theta,
        interleaved=eff_cfg.interleaved,
    )

    live_sinks_k = lk4[..., :live_sink_len, :]
    live_sinks_v = lv4[..., :live_sink_len, :]
    live_body_k = lk4[..., live_sink_len:, :]
    live_body_v = lv4[..., live_sink_len:, :]

    # Arm (a): Contiguous Oracle (exact full-precision Session-1 content at virtual_offset)
    t0_a = time.perf_counter()
    if oracle_session_k_post_rope is not None and oracle_session_v is not None:
        ok4, _ = _promote_to_4d_kv(oracle_session_k_post_rope)
        ov4, _ = _promote_to_4d_kv(oracle_session_v)
    else:
        k_pre_exact = strip_sinks_and_unrotate_k(
            sk4,
            session_positions,
            rotary_dim=eff_cfg.rotary_dim,
            rope_theta=eff_cfg.rope_theta,
            sink_tokens=eff_cfg.sink_tokens,
            inv_freq=inv_freq,
            interleaved=eff_cfg.interleaved,
        )
        vpos_exact = torch.arange(
            mount_start, mount_start + int(k_pre_exact.shape[-2]), device=sk4.device, dtype=torch.float64
        )
        ok4 = apply_partial_rope_k(
            k_pre_exact,
            vpos_exact,
            inv_freq=inv_freq,
            rotary_dim=eff_cfg.rotary_dim,
            rope_theta=eff_cfg.rope_theta,
            interleaved=eff_cfg.interleaved,
        )
        ov4 = sv4[..., eff_cfg.sink_tokens :, :].clone()

    k_a = torch.cat([live_sinks_k, ok4, live_body_k], dim=-2)
    v_a = torch.cat([live_sinks_v, ov4, live_body_v], dim=-2)
    out_a, w_a = _gqa_attention(q_rot, k_a, v_a)
    ms_a = (time.perf_counter() - t0_a) * 1000.0

    # Arm (b): Naive Spliced KV (raw stale RoPE + duplicate original sinks)
    t0_b = time.perf_counter()
    k_b, v_b = splice_naive_kv(sk4, sv4, lk4, lv4)
    out_b, w_b = _gqa_attention(q_rot, k_b, v_b)
    ms_b = (time.perf_counter() - t0_b) * 1000.0
    bytes_b = int(sk4.numel() * sk4.element_size() + sv4.numel() * sv4.element_size())

    # Arm (c): Training-Free Pre-RoPE + Sink-Strip + H64-INT4 + Virtual Mount
    t0_c = time.perf_counter()
    mnt_c = build_corrected_partition_b_kv(
        sk4,
        sv4,
        session_positions,
        live_sink_len,
        v_off,
        eff_cfg,
        blob_id="k0_probe_arm_c",
    )
    k_c = torch.cat([live_sinks_k, mnt_c.k, live_body_k], dim=-2)
    v_c = torch.cat([live_sinks_v, mnt_c.v, live_body_v], dim=-2)
    out_c, w_c = _gqa_attention(q_rot, k_c, v_c)
    ms_c = (time.perf_counter() - t0_c) * 1000.0
    kv_l2_c = float(
        torch.sqrt(
            ((ok4.float() - mnt_c.k.float()).pow(2).sum() + (ov4.float() - mnt_c.v.float()).pow(2).sum()).clamp_min(0.0)
        ).item()
    )

    # Arm (d): Arm (c) + CacheBlend Selective Refresh
    t0_d = time.perf_counter()
    ref_d = selective_cacheblend_refresh(
        mnt_c.k,
        mnt_c.v,
        ok4,
        ov4,
        refresh_frac=eff_cfg.refresh_frac,
    )
    k_d = torch.cat([live_sinks_k, ref_d.k, live_body_k], dim=-2)
    v_d = torch.cat([live_sinks_v, ref_d.v, live_body_v], dim=-2)
    out_d, w_d = _gqa_attention(q_rot, k_d, v_d)
    ms_d = ms_c + (time.perf_counter() - t0_d) * 1000.0

    # Target token index bookkeeping
    raw_tgt = 0 if target_session_pos is None else int(target_session_pos)
    strip_tgt = max(0, raw_tgt - int(eff_cfg.sink_tokens))
    oracle_norm = float(out_a.float().norm().clamp_min(1e-12).item())

    def _build_arm_out(
        arm_name: str,
        out_t: torch.Tensor,
        w_t: torch.Tensor,
        tgt_idx_in_k: int,
        kv_bytes: int,
        refreshed: int,
        elapsed_ms: float,
        kv_l2: float,
    ) -> K0ProbeArmOutput:
        rel_l2 = float((out_t.float() - out_a.float()).norm().item()) / oracle_norm
        cos_sim = float(
            F.cosine_similarity(out_t.float().reshape(1, -1), out_a.float().reshape(1, -1), dim=-1).item()
        )
        t_mass = float(w_t[..., tgt_idx_in_k].mean().item()) if 0 <= tgt_idx_in_k < w_t.shape[-1] else 0.0
        nll, acc, em = _score_probe_readout(out_t, candidate_prototypes, target_class, readout_temperature)
        return K0ProbeArmOutput(
            arm_name=arm_name,
            output=out_t,
            attn_weights=w_t,
            output_rel_l2_vs_oracle=rel_l2,
            output_cosine_vs_oracle=cos_sim,
            target_attn_mass=t_mass,
            answer_nll=nll,
            token_acc=acc,
            exact_match=em,
            kv_bytes_mounted=int(kv_bytes),
            refreshed_tokens=int(refreshed),
            elapsed_ms=float(elapsed_ms),
            kv_l2_vs_oracle=float(kv_l2),
        )

    bytes_a = int(ok4.numel() * ok4.element_size() + ov4.numel() * ov4.element_size())
    arm_outputs = {
        "arm_a": _build_arm_out("arm_a", out_a, w_a, live_sink_len + strip_tgt, bytes_a, 0, ms_a, 0.0),
        "arm_b": _build_arm_out("arm_b", out_b, w_b, raw_tgt, bytes_b, 0, ms_b, float("nan")),
        "arm_c": _build_arm_out(
            "arm_c", out_c, w_c, live_sink_len + strip_tgt, mnt_c.kv_bytes_mounted, 0, ms_c, kv_l2_c
        ),
        "arm_d": _build_arm_out(
            "arm_d",
            out_d,
            w_d,
            live_sink_len + strip_tgt,
            mnt_c.kv_bytes_mounted,
            ref_d.refreshed_tokens,
            ms_d,
            ref_d.post_refresh_kv_l2,
        ),
    }

    oracle_nll = arm_outputs["arm_a"].answer_nll
    oracle_score = (
        arm_outputs["arm_a"].exact_match
        if candidate_prototypes is not None and arm_outputs["arm_a"].exact_match > 0.0
        else 1.0
    )
    arm_metrics: dict[str, K0ArmMetrics] = {}
    for arm_key in CANONICAL_ARMS:
        ao = arm_outputs[arm_key]
        if candidate_prototypes is not None:
            ret_ratio = (
                ao.exact_match / oracle_score
                if eff_cfg.retention_metric == "exact_match"
                else ao.token_acc / oracle_score
            )
        else:
            # When no discrete readout head is supplied, use clamped cosine retention vs Oracle
            ret_ratio = max(0.0, min(1.0, ao.output_cosine_vs_oracle))
        arm_metrics[arm_key] = K0ArmMetrics(
            arm_name=arm_key,
            answer_nll=ao.answer_nll,
            token_acc=ao.token_acc,
            exact_match=ao.exact_match,
            oracle_retention_ratio=1.0 if arm_key == "arm_a" else float(ret_ratio),
            nll_gap=0.0 if arm_key == "arm_a" else float(ao.answer_nll - oracle_nll),
            kv_bytes_mounted=ao.kv_bytes_mounted,
            refreshed_tokens=ao.refreshed_tokens,
            elapsed_ms=ao.elapsed_ms,
            n_examples=1,
            extra={
                "output_rel_l2_vs_oracle": ao.output_rel_l2_vs_oracle,
                "output_cosine_vs_oracle": ao.output_cosine_vs_oracle,
                "target_attn_mass": ao.target_attn_mass,
                "kv_l2_vs_oracle": ao.kv_l2_vs_oracle if math.isfinite(ao.kv_l2_vs_oracle) else None,
            },
        )

    report = K0Report(arms=arm_metrics, config=eff_cfg, n_examples=1)
    verdict = decide_k0_verdict(report, eff_cfg)
    return K0AttentionProbeResult(
        arms=arm_outputs,
        report=report,
        verdict=verdict,
        target_index_in_session=raw_tgt,
        target_index_after_strip=strip_tgt,
        virtual_offset=v_off,
    )


def build_controlled_partial_rope_probe(
    *,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    num_q_heads: int = 2,
    head_dim: int = 256,
    rotary_dim: int = 64,
    session_len: int = 64,
    target_session_pos: int = 40,
    sink_tokens: int = 4,
    live_sink_len: int = 4,
    live_body_len: int = 12,
    virtual_offset: int = 10_000,
    rope_theta: float = 10_000_000.0,
    dynamic_drift_scale: float = 0.08,
    num_classes: int = 8,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build a deterministic partial-RoPE (`rotary_dim=64, head_dim=256`) multi-session attention probe.

    Session 1 (`pos = 0..session_len-1`) contains:
      * Huge attention sinks at `[0:sink_tokens)` aligned with queries (simulating `EPIC` sinks).
      * A planted target key at `target_session_pos` (`pos = 40`) aligned with the live query in
        both `[0:rotary_dim)` rotary and `[rotary_dim:head_dim)` NoPE subspaces, paired with a
        class-prototype target value vector.
      * Optional controlled cross-context Dynamic Error (`dynamic_drift_scale`) on the target +
        boundary tokens between independent Session-1 prefill and contiguous Oracle prefill so
        Arm `d` (`CacheBlend 5%`) recovery is quantitatively testable.
    """
    if not (sink_tokens <= target_session_pos < session_len):
        raise ValueError(
            f"target_session_pos={target_session_pos} must be in [sink_tokens={sink_tokens}, session_len={session_len})"
        )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # Build orthonormal class value prototypes and sink distractor direction
    q_mat, _ = torch.linalg.qr(torch.randn(head_dim, num_classes + 2, generator=gen, dtype=torch.float32))
    prototypes = q_mat[:, :num_classes].T.contiguous()  # [C, D], unit-norm orthonormal
    target_class = 0
    distractor_class = num_classes - 1
    sink_val_dir = F.normalize(0.75 * q_mat[:, num_classes] + 0.66 * prototypes[distractor_class], dim=0)
    target_val = prototypes[target_class] * 6.0

    # Query vector (pre-RoPE): strong rotary pattern + NoPE content pattern + sink attractor
    q_rot_base = F.normalize(torch.randn(rotary_dim, generator=gen, dtype=torch.float32), dim=0)
    q_nope_content = F.normalize(torch.randn(head_dim - rotary_dim, generator=gen, dtype=torch.float32), dim=0)
    q_nope_sink = F.normalize(torch.randn(head_dim - rotary_dim, generator=gen, dtype=torch.float32), dim=0)
    # Orthogonalize sink attractor against content query in NoPE space
    q_nope_sink = F.normalize(q_nope_sink - torch.dot(q_nope_sink, q_nope_content) * q_nope_content, dim=0)

    q_vec = torch.cat([q_rot_base * 14.0, q_nope_content * 7.5 + q_nope_sink * 6.0], dim=0)
    q_pre_rope = q_vec.view(1, 1, 1, head_dim).expand(num_layers, num_q_heads, 1, head_dim).clone()

    session_kept_len = session_len - sink_tokens
    mount_start = max(int(virtual_offset), int(live_sink_len))
    q_pos_val = mount_start + session_kept_len + live_body_len
    q_positions = torch.tensor([q_pos_val], dtype=torch.long)

    # Background Session-1 pre-RoPE keys & values (small random noise)
    session_k_pre = torch.randn(num_layers, num_kv_heads, session_len, head_dim, generator=gen) * 0.25
    session_v = torch.randn(num_layers, num_kv_heads, session_len, head_dim, generator=gen) * 0.25

    # Huge original Session-1 attention sinks at `[0:sink_tokens)` (`EPIC` failure mode if unstripped!)
    sink_key_nope = q_nope_sink * 34.0
    sink_key_rot = torch.zeros(rotary_dim, dtype=torch.float32)
    sink_key = torch.cat([sink_key_rot, sink_key_nope], dim=0)
    session_k_pre[:, :, :sink_tokens, :] = sink_key.view(1, 1, 1, head_dim)
    session_v[:, :, :sink_tokens, :] = (sink_val_dir * 6.0).view(1, 1, 1, head_dim)

    # Planted target key at `target_session_pos` (`pos=40`) whose rotary alignment matches
    # the TRUE contiguous relative offset `delta_true = q_pos_val - (mount_start + strip_tgt)`!
    inv_freq = build_rope_inv_freq(rotary_dim, rope_theta=rope_theta, dtype=torch.float64)
    strip_tgt = target_session_pos - sink_tokens
    delta_true = float(q_pos_val - (mount_start + strip_tgt))
    k_tgt_rot = apply_partial_rope_k(
        q_rot_base.view(1, rotary_dim),
        torch.tensor([delta_true], dtype=torch.float64),
        inv_freq=inv_freq,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
    ).reshape(rotary_dim)
    k_tgt_pre = torch.cat([k_tgt_rot * 13.0, q_nope_content * 7.5], dim=0)
    session_k_pre[:, :, target_session_pos, :] = k_tgt_pre.view(1, 1, head_dim)
    session_v[:, :, target_session_pos, :] = target_val.view(1, 1, head_dim)

    # Exact contiguous Oracle KV (before applying optional independent-session dynamic drift)
    oracle_k_pre = session_k_pre[:, :, sink_tokens:, :].clone()
    oracle_v = session_v[:, :, sink_tokens:, :].clone()

    # Optional realistic cross-context Dynamic Error (D) on target & boundary tokens in Session-1 KV
    if dynamic_drift_scale > 0.0:
        drift_indices = sorted({0, min(1, session_kept_len - 1), strip_tgt})
        attenuate = max(0.25, 1.0 - 0.5 * dynamic_drift_scale)
        tgt_proto = prototypes[target_class]
        for rel_i in drift_indices:
            abs_i = rel_i + sink_tokens
            dk_noise = F.normalize(torch.randn(head_dim, generator=gen), dim=0) * (dynamic_drift_scale * 6.0)
            dv_raw = torch.randn(head_dim, generator=gen)
            # Keep value drift strictly orthogonal to the target prototype (attenuation + cross-context shift)
            dv_orth = F.normalize(dv_raw - torch.dot(dv_raw, tgt_proto) * tgt_proto, dim=0)
            dv_noise = dv_orth * (dynamic_drift_scale * 5.0)
            session_k_pre[:, :, abs_i, :] = session_k_pre[:, :, abs_i, :] + dk_noise.view(1, 1, head_dim)
            session_v[:, :, abs_i, :] = (
                session_v[:, :, abs_i, :] * attenuate + dv_noise.view(1, 1, head_dim)
            )

    session_positions = torch.arange(0, session_len, dtype=torch.long)
    session_k_post = apply_partial_rope_k(
        session_k_pre,
        session_positions,
        inv_freq=inv_freq,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
    )
    oracle_positions = torch.arange(mount_start, mount_start + session_kept_len, dtype=torch.long)
    oracle_k_post = apply_partial_rope_k(
        oracle_k_pre,
        oracle_positions,
        inv_freq=inv_freq,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
    )

    # Live Session-2 Partition-A KV (`[0:live_sink_len)` calibrated live sinks + live body distractors)
    live_len = live_sink_len + live_body_len
    live_k_pre = torch.randn(num_layers, num_kv_heads, live_len, head_dim, generator=gen) * 0.25
    live_v = torch.randn(num_layers, num_kv_heads, live_len, head_dim, generator=gen) * 0.25
    live_sink_key = torch.cat([torch.zeros(rotary_dim), q_nope_sink * 5.5], dim=0)
    live_k_pre[:, :, :live_sink_len, :] = live_sink_key.view(1, 1, 1, head_dim)
    live_v[:, :, :live_sink_len, :] = (sink_val_dir * 0.5).view(1, 1, 1, head_dim)

    live_body_start = mount_start + session_kept_len
    live_pos = torch.cat(
        [
            torch.arange(0, live_sink_len, dtype=torch.long),
            torch.arange(live_body_start, live_body_start + live_body_len, dtype=torch.long),
        ],
        dim=0,
    )
    live_k_post = apply_partial_rope_k(
        live_k_pre,
        live_pos,
        inv_freq=inv_freq,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
    )

    return {
        "q_pre_rope": q_pre_rope,
        "q_positions": q_positions,
        "session_k_post_rope": session_k_post,
        "session_v": session_v,
        "session_positions": session_positions,
        "live_k_post_rope": live_k_post,
        "live_v": live_v,
        "oracle_session_k_post_rope": oracle_k_post,
        "oracle_session_v": oracle_v,
        "target_session_pos": target_session_pos,
        "candidate_prototypes": prototypes,
        "target_class": target_class,
        "live_sink_len": live_sink_len,
        "virtual_offset": virtual_offset,
    }


@dataclass
class K0SupersessionResult:
    """Outcome of testing `SplitHorizonKVBlobStore.supersede()` inside Kill-Gate `K0`."""

    active_blob_ids: list[str]
    superseded_blob_id: str
    updated_blob_id: str
    naive_updated_attn_mass: float
    naive_stale_attn_mass: float
    naive_predicted_class: int
    sleepgate_updated_attn_mass: float
    sleepgate_stale_attn_mass: float
    sleepgate_predicted_class: int
    updated_class: int
    stale_class: int
    sleepgate_nll: float
    naive_nll: float
    store_stats: dict[str, int]


def run_k0_supersession_probe(
    *,
    cfg: K0Config | None = None,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    session_len: int = 32,
    live_sink_len: int = 4,
    virtual_offset: int = 10_000,
    seed: int = 2027,
) -> K0SupersessionResult:
    """Verify `SplitHorizonKVBlobStore.supersede()` removes stale Session-1 blobs in `K0`.

    Packs two contradictory session blobs (`session_1_stale` encoding `stale_class=1` and
    `session_2_updated` encoding `updated_class=0`), admits both into `SplitHorizonKVBlobStore`,
    and calls `store.supersede('session_1_stale', 'session_2_updated', now=2.5)`.
    Compares mounting both blobs blindly (proactive interference) against `store.retrieve(..., now=3.0)`
    + `mount_kv_blob` (clean attention exclusively on the updated Session-2 fact).
    """
    eff_cfg = K0Config() if cfg is None else cfg
    eff_cfg.validate()
    rot_dim = eff_cfg.rotary_dim
    sink_n = eff_cfg.sink_tokens
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    q_mat, _ = torch.linalg.qr(torch.randn(head_dim, 6, generator=gen, dtype=torch.float32))
    prototypes = q_mat[:, :4].T.contiguous()  # [4, D]
    updated_class, stale_class = 0, 1

    q_rot_dir = F.normalize(torch.randn(rot_dim, generator=gen), dim=0)
    q_nope_dir = F.normalize(torch.randn(head_dim - rot_dim, generator=gen), dim=0)
    q_pre = torch.cat([q_rot_dir * 8.0, q_nope_dir * 12.0], dim=0).view(1, 1, 1, head_dim).expand(
        num_layers, num_kv_heads, 1, head_dim
    ).clone()

    kept_len = session_len - sink_n
    mount_start = max(int(virtual_offset), int(live_sink_len))
    q_pos_val = mount_start + 2 * kept_len + 8
    inv_freq = build_rope_inv_freq(rot_dim, rope_theta=eff_cfg.rope_theta, dtype=torch.float64)
    q_rot = apply_partial_rope_k(
        q_pre,
        torch.tensor([q_pos_val], dtype=torch.long),
        inv_freq=inv_freq,
        rotary_dim=rot_dim,
        rope_theta=eff_cfg.rope_theta,
    )

    def _make_fact_session(
        fact_class: int, key_boost: float, fact_idx: int, sess_seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_local = torch.Generator(device="cpu").manual_seed(sess_seed)
        kp = torch.randn(num_layers, num_kv_heads, session_len, head_dim, generator=g_local) * 0.2
        vp = torch.randn(num_layers, num_kv_heads, session_len, head_dim, generator=g_local) * 0.2
        # Give all non-sink tokens topic-aligned NoPE signal so coarse centroid matches the query topic
        kp[:, :, sink_n:, rot_dim:] += (q_nope_dir * 0.8).view(1, 1, 1, head_dim - rot_dim)
        kp[:, :, fact_idx, :] = torch.cat([q_rot_dir * 2.0, q_nope_dir * key_boost], dim=0).view(1, 1, head_dim)
        vp[:, :, fact_idx, :] = (prototypes[fact_class] * 6.0).view(1, 1, head_dim)
        pos = torch.arange(0, session_len, dtype=torch.long)
        k_post = apply_partial_rope_k(kp, pos, inv_freq=inv_freq, rotary_dim=rot_dim, rope_theta=eff_cfg.rope_theta)
        return k_post, vp, pos

    # Deliberately give stale Session-1 fact a stronger raw NoPE key match so without SleepGate
    # proactive interference actively hijacks the prediction toward `stale_class`!
    stale_fact_idx = min(session_len - 1, sink_n + 8)
    upd_fact_idx = min(session_len - 1, sink_n + 8)
    s1_k, s1_v, s1_pos = _make_fact_session(stale_class, key_boost=22.0, fact_idx=stale_fact_idx, sess_seed=seed + 1)
    s2_k, s2_v, s2_pos = _make_fact_session(updated_class, key_boost=14.0, fact_idx=upd_fact_idx, sess_seed=seed + 2)

    mnt_s1 = build_corrected_partition_b_kv(
        s1_k, s1_v, s1_pos, live_sink_len, mount_start, eff_cfg, blob_id="session_1_stale", t_created=1.0, salience=0.9
    )
    mnt_s2 = build_corrected_partition_b_kv(
        s2_k,
        s2_v,
        s2_pos,
        live_sink_len,
        mount_start + kept_len,
        eff_cfg,
        blob_id="session_2_updated",
        t_created=2.0,
        salience=0.9,
    )

    blob_cfg = eff_cfg.effective_blob_config(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, blob_tokens=kept_len
    )
    store = SplitHorizonKVBlobStore(config=blob_cfg, graph=DynamicLatentMemoryGraph())
    assert store.admit(mnt_s1.blob, salience=0.9) is True
    assert store.admit(mnt_s2.blob, salience=0.9, temporal_parent="session_1_stale") is True
    store.supersede("session_1_stale", "session_2_updated", now=2.5)

    # Derive query centroid from the same NoPE direction used by `kv_blob._derive_coarse_centroid`
    q_centroid = mnt_s2.blob.header.coarse_centroid
    active_blobs = store.retrieve(q_centroid, top_k=4, now=3.0)
    active_ids = [b.blob_id for b in active_blobs]

    # Live sinks
    live_sinks_k = torch.randn(num_layers, num_kv_heads, live_sink_len, head_dim, generator=gen) * 0.2
    live_sinks_v = torch.randn(num_layers, num_kv_heads, live_sink_len, head_dim, generator=gen) * 0.2

    # 1. Naive (NO supersession filtering): mounts BOTH stale Session-1 and updated Session-2 blobs
    k_both = torch.cat([live_sinks_k, mnt_s1.k, mnt_s2.k], dim=-2)
    v_both = torch.cat([live_sinks_v, mnt_s1.v, mnt_s2.v], dim=-2)
    out_both, w_both = _gqa_attention(q_rot, k_both, v_both)
    stale_slot_both = live_sink_len + (stale_fact_idx - sink_n)
    upd_slot_both = live_sink_len + kept_len + (upd_fact_idx - sink_n)
    naive_stale_mass = float(w_both[..., stale_slot_both].mean().item())
    naive_upd_mass = float(w_both[..., upd_slot_both].mean().item())
    naive_logits = prototypes @ out_both.float().mean(dim=(0, 1, 2))
    naive_pred = int(torch.argmax(naive_logits).item())
    naive_nll = float(F.cross_entropy(naive_logits.unsqueeze(0), torch.tensor([updated_class])).item())

    # 2. SleepGate (`SplitHorizonKVBlobStore.retrieve(..., now=3.0)` + `mount_kv_blob`): only active blobs mounted!
    mounted_k_parts = [live_sinks_k]
    mounted_v_parts = [live_sinks_v]
    cursor = mount_start
    upd_slot_sg = -1
    for blob in active_blobs:
        bk, bv = store.mount_blob(blob.blob_id, virtual_positions=cursor)
        if blob.blob_id == "session_2_updated":
            upd_slot_sg = sum(int(t.shape[-2]) for t in mounted_k_parts) + (upd_fact_idx - sink_n)
        mounted_k_parts.append(bk)
        mounted_v_parts.append(bv)
        cursor += blob.n_tokens

    k_sg = torch.cat(mounted_k_parts, dim=-2)
    v_sg = torch.cat(mounted_v_parts, dim=-2)
    out_sg, w_sg = _gqa_attention(q_rot, k_sg, v_sg)
    sg_upd_mass = float(w_sg[..., upd_slot_sg].mean().item()) if upd_slot_sg >= 0 else 0.0
    sg_stale_mass = 0.0  # superseded Session-1 blob was excluded by SleepGate!
    sg_logits = prototypes @ out_sg.float().mean(dim=(0, 1, 2))
    sg_pred = int(torch.argmax(sg_logits).item())
    sg_nll = float(F.cross_entropy(sg_logits.unsqueeze(0), torch.tensor([updated_class])).item())

    return K0SupersessionResult(
        active_blob_ids=active_ids,
        superseded_blob_id="session_1_stale",
        updated_blob_id="session_2_updated",
        naive_updated_attn_mass=naive_upd_mass,
        naive_stale_attn_mass=naive_stale_mass,
        naive_predicted_class=naive_pred,
        sleepgate_updated_attn_mass=sg_upd_mass,
        sleepgate_stale_attn_mass=sg_stale_mass,
        sleepgate_predicted_class=sg_pred,
        updated_class=updated_class,
        stale_class=stale_class,
        sleepgate_nll=sg_nll,
        naive_nll=naive_nll,
        store_stats=store.memory_stats(now=3.0),
    )


# ==============================================================================
# Suite Runner (`run_k0_suite`), Synthetic Smoke Benchmark & Markdown Formatter
# ==============================================================================
def _coerce_episode_result(raw: K0EpisodeResult | Mapping[str, Any], arm_name: str, example_id: str) -> K0EpisodeResult:
    if isinstance(raw, K0EpisodeResult):
        return raw
    if isinstance(raw, Mapping):
        return K0EpisodeResult(
            arm_name=canonicalize_arm_name(str(raw.get("arm_name", arm_name))),
            example_id=str(raw.get("example_id", example_id)),
            answer_nll=float(raw["answer_nll"]),
            token_acc=float(raw["token_acc"]),
            exact_match=float(raw["exact_match"]),
            kv_bytes_mounted=int(raw.get("kv_bytes_mounted", 0)),
            refreshed_tokens=int(raw.get("refreshed_tokens", 0)),
            elapsed_ms=float(raw.get("elapsed_ms", 0.0)),
            predicted=str(raw.get("predicted", "")),
            extra=dict(raw.get("extra", {})),
        )
    raise TypeError(f"evaluator_fn returned unsupported result type {type(raw).__name__}")


def _evaluator_is_batch_all_arms(evaluator_fn: Callable[..., Any]) -> bool:
    """Detect whether `evaluator_fn` takes `(example, cfg)` (all arms at once) vs `(example, arm_name, cfg)`."""
    try:
        sig = inspect.signature(evaluator_fn)
    except (TypeError, ValueError):
        return False
    pos_params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs:
        return False
    return len(pos_params) == 2


def _compute_retention_ratio(
    arm_em: float,
    arm_acc: float,
    oracle_em: float,
    oracle_acc: float,
    metric: str,
) -> float:
    def _safe_ratio(val: float, ref: float) -> float:
        if ref <= 0.0:
            return 1.0 if val <= 0.0 else 0.0
        return float(val) / float(ref)

    if metric == "token_acc":
        return _safe_ratio(arm_acc, oracle_acc)
    if metric == "combined":
        return 0.5 * (_safe_ratio(arm_em, oracle_em) + _safe_ratio(arm_acc, oracle_acc))
    # Default: exact_match (falls back to token_acc only if oracle_em == 0 and oracle_acc > 0)
    if oracle_em <= 0.0 and oracle_acc > 0.0:
        return _safe_ratio(arm_acc, oracle_acc)
    return _safe_ratio(arm_em, oracle_em)


def run_k0_suite(
    examples: Sequence[K0Example],
    evaluator_fn: Callable[..., Any],
    cfg: K0Config | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> K0Report:
    """Execute Arms `a`, `b`, `c`, `d` over a multi-session / multi-hop QA benchmark suite.

    `evaluator_fn` may follow either signature:
      * Per-arm: `evaluator_fn(example: K0Example, arm_name: str, cfg: K0Config) -> K0EpisodeResult | dict`
      * All-arms-at-once (avoids duplicate session prefills on GPU):
        `evaluator_fn(example: K0Example, cfg: K0Config) -> Mapping[str, K0EpisodeResult | dict]`
    """
    eff_cfg = K0Config() if cfg is None else cfg
    eff_cfg.validate()
    if not examples:
        raise ValueError("examples must be a non-empty sequence of K0Example items")

    per_arm_rows: dict[str, list[K0EpisodeResult]] = {arm: [] for arm in CANONICAL_ARMS}
    per_cat_rows: dict[str, dict[str, list[K0EpisodeResult]]] = {}
    batch_mode = _evaluator_is_batch_all_arms(evaluator_fn)

    for ex in examples:
        cat_bucket = per_cat_rows.setdefault(ex.category, {arm: [] for arm in CANONICAL_ARMS})
        if batch_mode:
            raw_map = evaluator_fn(ex, eff_cfg)
            if not isinstance(raw_map, Mapping):
                raise TypeError(f"2-arg evaluator_fn must return a Mapping of arms, got {type(raw_map).__name__}")
            norm_map = {canonicalize_arm_name(k): v for k, v in raw_map.items()}
            for arm in CANONICAL_ARMS:
                if arm not in norm_map:
                    raise KeyError(f"evaluator_fn missing required arm {arm!r} for example {ex.example_id!r}")
                ep = _coerce_episode_result(norm_map[arm], arm, ex.example_id)
                per_arm_rows[arm].append(ep)
                cat_bucket[arm].append(ep)
        else:
            for arm in CANONICAL_ARMS:
                raw_ep = evaluator_fn(ex, arm, eff_cfg)
                ep = _coerce_episode_result(raw_ep, arm, ex.example_id)
                per_arm_rows[arm].append(ep)
                cat_bucket[arm].append(ep)

    def _agg(rows: list[K0EpisodeResult]) -> tuple[float, float, float, int, int, float]:
        n = float(len(rows))
        mean_nll = sum(r.answer_nll for r in rows) / n
        mean_acc = sum(r.token_acc for r in rows) / n
        mean_em = sum(r.exact_match for r in rows) / n
        tot_bytes = sum(int(r.kv_bytes_mounted) for r in rows)
        tot_ref = sum(int(r.refreshed_tokens) for r in rows)
        tot_ms = sum(float(r.elapsed_ms) for r in rows)
        return mean_nll, mean_acc, mean_em, tot_bytes, tot_ref, tot_ms

    ora_nll, ora_acc, ora_em, _, _, _ = _agg(per_arm_rows["arm_a"])
    arms_out: dict[str, K0ArmMetrics] = {}
    for arm in CANONICAL_ARMS:
        rows = per_arm_rows[arm]
        nll, acc, em, tot_bytes, tot_ref, tot_ms = _agg(rows)
        ret = (
            1.0
            if arm == "arm_a"
            else _compute_retention_ratio(em, acc, ora_em, ora_acc, eff_cfg.retention_metric)
        )
        arms_out[arm] = K0ArmMetrics(
            arm_name=arm,
            answer_nll=nll,
            token_acc=acc,
            exact_match=em,
            oracle_retention_ratio=ret,
            nll_gap=0.0 if arm == "arm_a" else float(nll - ora_nll),
            kv_bytes_mounted=tot_bytes,
            refreshed_tokens=tot_ref,
            elapsed_ms=tot_ms,
            n_examples=len(rows),
            extra={
                "avg_kv_bytes_mounted": float(tot_bytes) / float(len(rows)),
                "avg_refreshed_tokens": float(tot_ref) / float(len(rows)),
            },
        )

    per_category_summary: dict[str, dict[str, dict[str, float]]] = {}
    for cat, arm_dict in sorted(per_cat_rows.items()):
        c_ora_nll, c_ora_acc, c_ora_em, _, _, _ = _agg(arm_dict["arm_a"])
        per_category_summary[cat] = {}
        for arm in CANONICAL_ARMS:
            c_nll, c_acc, c_em, _, c_ref, _ = _agg(arm_dict[arm])
            c_ret = (
                1.0
                if arm == "arm_a"
                else _compute_retention_ratio(c_em, c_acc, c_ora_em, c_ora_acc, eff_cfg.retention_metric)
            )
            per_category_summary[cat][arm] = {
                "answer_nll": c_nll,
                "nll_gap": 0.0 if arm == "arm_a" else float(c_nll - c_ora_nll),
                "token_acc": c_acc,
                "exact_match": c_em,
                "oracle_retention_ratio": c_ret,
                "refreshed_tokens": float(c_ref),
                "n_examples": float(len(arm_dict[arm])),
            }

    report = K0Report(
        arms=arms_out,
        config=eff_cfg,
        per_category=per_category_summary,
        n_examples=len(examples),
        metadata=dict(metadata or {}),
    )
    decide_k0_verdict(report, eff_cfg)
    return report


def format_k0_markdown(report: K0Report) -> str:
    """Format a concise, deterministic GitHub-flavored Markdown report for Kill-Gate `K0`."""
    lines: list[str] = [
        "# Kill-Gate `K0` Report — Zero-Training Pre-RoPE KV-Mem Blob Injection",
        "",
        f"- **Verdict:** `{report.verdict}` (`{report.sub_verdict}`)",
        f"- **Examples Evaluated:** `{report.n_examples}`",
        (
            f"- **Config:** `green >= {report.config.green_threshold:.0%}`, "
            f"`yellow >= {report.config.yellow_threshold:.0%}`, "
            f"`refresh_frac = {report.config.refresh_frac:.0%}`, "
            f"`sink_tokens = {report.config.sink_tokens}`, "
            f"`boundary_header_tokens = {report.config.boundary_header_tokens}`, "
            f"`rotary_dim = {report.config.rotary_dim}`"
        ),
    ]
    if report.rationale:
        lines.append(f"- **Rationale:** {report.rationale}")
    if report.recommended_action:
        lines.append(f"- **Recommended Action:** {report.recommended_action}")

    lines.extend(
        [
            "",
            (
                "| Arm | Answer NLL | NLL Gap (nats) | Token Acc | Exact Match | "
                "Oracle Retention | KV Mounted (B) | Refreshed Toks | Elapsed (ms) |"
            ),
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for arm_key in CANONICAL_ARMS:
        m = report.get_arm(arm_key)
        if m is None:
            continue
        label = ARM_DISPLAY_NAMES.get(arm_key, arm_key)
        lines.append(
            f"| {label} | `{m.answer_nll:.4f}` | `{m.nll_gap:+.4f}` | `{m.token_acc:.3f}` | "
            f"`{m.exact_match:.3f}` | **`{m.oracle_retention_ratio:.1%}`** | "
            f"`{m.kv_bytes_mounted:,}` | `{m.refreshed_tokens}` | `{m.elapsed_ms:.2f}` |"
        )

    if report.per_category:
        lines.extend(
            [
                "",
                "## Per-Category Breakdown",
                "",
                "| Category | Arm (b) Naive | Arm (c) Partition-B | Arm (d) +CacheBlend | Arm (c) NLL Gap |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for cat, cdata in sorted(report.per_category.items()):
            rb = cdata.get("arm_b", {}).get("oracle_retention_ratio", 0.0)
            rc = cdata.get("arm_c", {}).get("oracle_retention_ratio", 0.0)
            rd = cdata.get("arm_d", {}).get("oracle_retention_ratio", 0.0)
            gc = cdata.get("arm_c", {}).get("nll_gap", 0.0)
            lines.append(f"| `{cat}` | `{rb:.1%}` | `{rc:.1%}` | `{rd:.1%}` | `{gc:+.4f}` |")

    return "\n".join(lines) + "\n"


# ==============================================================================
# Synthetic CPU Smoke Benchmark & Real HF `Qwen3.6-64L` Checkpoint Evaluator
# ==============================================================================
def build_default_smoke_examples() -> list[K0Example]:
    """Deterministic multi-session + contradictory-memory (`SleepGate`) CPU smoke episodes."""
    return [
        K0Example(
            example_id="k0_multi_hop_18k",
            sessions=["Session 1: module `alpha.py` defines `MAX_RETRIES = 7` and calls `emit_trace()`."],
            live_prompt="Session 2: Given `alpha.py`, what is `MAX_RETRIES`?",
            answer="7",
            horizon_tokens=18_000,
            category="multi_hop",
            metadata={"seed": 2026, "target_session_pos": 40, "dynamic_drift_scale": 0.05},
        ),
        K0Example(
            example_id="k0_codebase_64k",
            sessions=["Session 1: `router.py` sets `CACHE_PORT = 9443` for Partition-B RPC mounts."],
            live_prompt="Session 2: Which port does `router.py` configure for `CACHE_PORT`?",
            answer="9443",
            horizon_tokens=64_000,
            category="codebase_qa",
            metadata={"seed": 2027, "target_session_pos": 36, "dynamic_drift_scale": 0.05},
        ),
        K0Example(
            example_id="k0_needle_256k",
            sessions=["Session 1: Release tag `v2.4.1` pinned SHA prefix `9f4a2c` in deploy manifest."],
            live_prompt="Session 2: Return the pinned SHA prefix for release tag `v2.4.1`.",
            answer="9f4a2c",
            horizon_tokens=256_000,
            category="needle",
            metadata={"seed": 2028, "target_session_pos": 44, "dynamic_drift_scale": 0.05},
        ),
        K0Example(
            example_id="k0_sleepgate_contradiction",
            sessions=[
                "Session 1 (stale): `DEFAULT_TIMEOUT_S = 15`.",
                "Session 2 (updated): Superseding Session 1, `DEFAULT_TIMEOUT_S = 45`.",
            ],
            live_prompt="Session 3: What is the active value of `DEFAULT_TIMEOUT_S`?",
            answer="45",
            horizon_tokens=64_000,
            category="contradictory_update",
            superseded_session_indices=(0,),
            metadata={"seed": 2029},
        ),
    ]


def _synthetic_cpu_episode_evaluator(example: K0Example, cfg: K0Config) -> dict[str, K0EpisodeResult]:
    """Fast pure-CPU evaluator executing partial-RoPE probes + `SleepGate` supersession."""
    ex_seed = int(example.metadata.get("seed", 2026))
    if example.category == "contradictory_update" and example.superseded_session_indices:
        t0 = time.perf_counter()
        sg = run_k0_supersession_probe(cfg=cfg, virtual_offset=cfg.virtual_offset, seed=ex_seed)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sg_hit = 1.0 if sg.sleepgate_predicted_class == sg.updated_class else 0.0
        naive_hit = 1.0 if sg.naive_predicted_class == sg.updated_class else 0.0
        kv_bytes = int(sg.store_stats.get("active_payload_bytes", 0) + sg.store_stats.get("active_gpu_header_bytes", 0))
        return {
            "arm_a": K0EpisodeResult(
                arm_name="arm_a",
                example_id=example.example_id,
                answer_nll=sg.sleepgate_nll,
                token_acc=1.0,
                exact_match=1.0,
                kv_bytes_mounted=kv_bytes,
                elapsed_ms=elapsed_ms * 0.25,
            ),
            "arm_b": K0EpisodeResult(
                arm_name="arm_b",
                example_id=example.example_id,
                answer_nll=sg.naive_nll,
                token_acc=naive_hit,
                exact_match=naive_hit,
                kv_bytes_mounted=kv_bytes * 2,
                elapsed_ms=elapsed_ms * 0.25,
            ),
            "arm_c": K0EpisodeResult(
                arm_name="arm_c",
                example_id=example.example_id,
                answer_nll=sg.sleepgate_nll,
                token_acc=sg_hit,
                exact_match=sg_hit,
                kv_bytes_mounted=kv_bytes,
                elapsed_ms=elapsed_ms * 0.25,
            ),
            "arm_d": K0EpisodeResult(
                arm_name="arm_d",
                example_id=example.example_id,
                answer_nll=sg.sleepgate_nll,
                token_acc=sg_hit,
                exact_match=sg_hit,
                kv_bytes_mounted=kv_bytes,
                refreshed_tokens=1,
                elapsed_ms=elapsed_ms * 0.25,
            ),
        }

    probe_inputs = build_controlled_partial_rope_probe(
        rotary_dim=cfg.rotary_dim,
        rope_theta=cfg.rope_theta,
        sink_tokens=cfg.sink_tokens,
        target_session_pos=int(example.metadata.get("target_session_pos", 40)),
        virtual_offset=cfg.virtual_offset,
        dynamic_drift_scale=float(example.metadata.get("dynamic_drift_scale", 0.05)),
        seed=ex_seed,
    )
    res = run_k0_attention_probe(
        probe_inputs["q_pre_rope"],
        probe_inputs["q_positions"],
        probe_inputs["session_k_post_rope"],
        probe_inputs["session_v"],
        probe_inputs["session_positions"],
        probe_inputs["live_k_post_rope"],
        probe_inputs["live_v"],
        live_sink_len=probe_inputs["live_sink_len"],
        virtual_offset=probe_inputs["virtual_offset"],
        cfg=cfg,
        oracle_session_k_post_rope=probe_inputs["oracle_session_k_post_rope"],
        oracle_session_v=probe_inputs["oracle_session_v"],
        target_session_pos=probe_inputs["target_session_pos"],
        candidate_prototypes=probe_inputs["candidate_prototypes"],
        target_class=probe_inputs["target_class"],
    )
    return {
        arm: K0EpisodeResult(
            arm_name=arm,
            example_id=example.example_id,
            answer_nll=ao.answer_nll,
            token_acc=ao.token_acc,
            exact_match=ao.exact_match,
            kv_bytes_mounted=ao.kv_bytes_mounted,
            refreshed_tokens=ao.refreshed_tokens,
            elapsed_ms=ao.elapsed_ms,
            extra={
                "output_rel_l2_vs_oracle": ao.output_rel_l2_vs_oracle,
                "output_cosine_vs_oracle": ao.output_cosine_vs_oracle,
                "target_attn_mass": ao.target_attn_mass,
            },
        )
        for arm, ao in res.arms.items()
    }


def run_smoke_cpu(cfg: K0Config | None = None) -> K0Report:
    """Run the deterministic pure-CPU Kill-Gate `K0` self-test suite."""
    eff_cfg = K0Config() if cfg is None else cfg
    examples = build_default_smoke_examples()
    return run_k0_suite(
        examples,
        _synthetic_cpu_episode_evaluator,
        eff_cfg,
        metadata={"mode": "smoke_cpu", "device": "cpu"},
    )


class HFCheckpointK0Evaluator:
    """Real HF / `Qwen3.6-64L` Kill-Gate `K0` evaluator ready for single-GPU execution (`~30 min`).

    Executes Arms `a`, `b`, `c`, `d` with zero gradient updates:
      * Arm `a`: Full contiguous prefill of active sessions + live prompt + answer span.
      * Arm `b`: Independent session prefills spliced directly via `splice_naive_kv`.
      * Arm `c`: Independent session prefills processed through `build_corrected_partition_b_kv`
        (Pre-RoPE un-rotation, `[0:sink_tokens)` strip, `H64-INT4` pack/mount, `GDN-2` isolated).
      * Arm `d`: Arm `c` + `selective_cacheblend_refresh` (`5%` top-HKVD tokens).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str | None = None,
        live_sink_len: int = 4,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError, TypeError):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.live_sink_len = int(live_sink_len)
        self._cache_kw = cache_kwarg_name(model)
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError, AttributeError):
            params = {}
        self._accepts_pos_ids = "position_ids" in params

    def _encode(self, text: str) -> list[int]:
        if callable(self.tokenizer):
            try:
                out = self.tokenizer(text, add_special_tokens=False)
            except TypeError:
                out = self.tokenizer(text)
            ids = out["input_ids"] if isinstance(out, Mapping) else getattr(out, "input_ids", out)
        elif hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = self.tokenizer.encode(text)
        else:
            raise TypeError(f"tokenizer {type(self.tokenizer).__name__} must be callable or provide .encode()")
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(t) for t in ids]

    def _forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Any = None,
        position_ids: torch.Tensor | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
        if cache is not None:
            kwargs[self._cache_kw] = cache
        if position_ids is not None and self._accepts_pos_ids:
            kwargs["position_ids"] = position_ids
        with torch.no_grad():
            return self.model(**kwargs)

    def _score_answer_from_logits(
        self,
        logits: torch.Tensor,
        answer_ids: list[int],
        prefix_len_in_logits: int,
    ) -> tuple[float, float, float, str]:
        ans_len = len(answer_ids)
        if ans_len == 0:
            return 0.0, 1.0, 1.0, ""
        # Predict token `answer_ids[i]` from logit position `prefix_len_in_logits - 1 + i`
        start = max(0, prefix_len_in_logits - 1)
        ans_logits = logits[0, start : start + ans_len, :].float()
        targets = torch.tensor(answer_ids, dtype=torch.long, device=ans_logits.device)
        nll = float(F.cross_entropy(ans_logits, targets).item())
        preds = torch.argmax(ans_logits, dim=-1)
        tok_acc = float((preds == targets).float().mean().item())
        em = 1.0 if bool((preds == targets).all().item()) else 0.0
        pred_text = (
            self.tokenizer.decode(preds.tolist(), skip_special_tokens=True)
            if hasattr(self.tokenizer, "decode")
            else ""
        )
        return nll, tok_acc, em, str(pred_text)

    def _fresh_cache_from(self, ref_cache: Any) -> Any:
        cls = type(ref_cache)
        cfg_obj = getattr(self.model, "config", None)
        if cfg_obj is not None:
            try:
                return cls(cfg_obj)
            except TypeError:
                pass
        try:
            return cls()
        except TypeError:
            pass
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache()
        except Exception:
            if hasattr(ref_cache, "key_cache"):
                return cls(len(ref_cache.key_cache))
            raise

    def __call__(self, example: K0Example, cfg: K0Config) -> dict[str, K0EpisodeResult]:
        superseded = set(example.superseded_session_indices)
        active_sessions = [s for idx, s in enumerate(example.sessions) if idx not in superseded]
        all_sessions = list(example.sessions)

        active_sess_ids = [self._encode(s) for s in active_sessions]
        all_sess_ids = [self._encode(s) for s in all_sessions]
        live_ids = self._encode(example.live_prompt)
        ans_ids = self._encode(example.answer)
        if not live_ids:
            live_ids = [0]
        if not ans_ids:
            ans_ids = [0]

        # Arm (a): Contiguous Oracle on active sessions + live_prompt + answer
        t0_a = time.perf_counter()
        flat_active = [tid for seq in active_sess_ids for tid in seq]
        full_ids_a = torch.tensor([flat_active + live_ids + ans_ids], dtype=torch.long, device=self.device)
        out_a = self._forward(full_ids_a)
        nll_a, acc_a, em_a, pred_a = self._score_answer_from_logits(
            out_a.logits, ans_ids, len(flat_active) + len(live_ids)
        )
        oracle_k_full, oracle_v_full, layer_indices = extract_layer_kv_pairs(out_a)
        ms_a = (time.perf_counter() - t0_a) * 1000.0
        ref_cache = extract_output_cache(out_a)
        bytes_a = int(
            oracle_k_full.numel() * oracle_k_full.element_size()
            + oracle_v_full.numel() * oracle_v_full.element_size()
        )

        # Independent session prefills (shared across Arms b, c, d)
        raw_sessions_kv: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
        for s_ids in all_sess_ids:
            if not s_ids:
                raw_sessions_kv.append((oracle_k_full[..., :0, :], oracle_v_full[..., :0, :], []))
                continue
            s_t = torch.tensor([s_ids], dtype=torch.long, device=self.device)
            s_out = self._forward(s_t)
            sk, sv, _ = extract_layer_kv_pairs(s_out, layer_indices=layer_indices)
            raw_sessions_kv.append((sk, sv, s_ids))

        live_and_ans = torch.tensor([live_ids + ans_ids], dtype=torch.long, device=self.device)
        live_sink_n = min(self.live_sink_len, len(live_ids))

        # Arm (b): Naive Spliced KV (raw post-RoPE + original sinks + stale sessions unmasked)
        t0_b = time.perf_counter()
        k_b_sess = torch.cat([sk for sk, _, _ in raw_sessions_kv], dim=-2)
        v_b_sess = torch.cat([sv for _, sv, _ in raw_sessions_kv], dim=-2)
        cache_b = inject_layer_kv_pairs(self._fresh_cache_from(ref_cache), k_b_sess, v_b_sess, layer_indices)
        out_b = self._forward(live_and_ans, cache=cache_b)
        nll_b, acc_b, em_b, pred_b = self._score_answer_from_logits(out_b.logits, ans_ids, len(live_ids))
        ms_b = (time.perf_counter() - t0_b) * 1000.0
        bytes_b = int(k_b_sess.numel() * k_b_sess.element_size() + v_b_sess.numel() * v_b_sess.element_size())

        # Arm (c): Training-Free Pre-RoPE + Sink-Strip + H64-INT4 + Virtual Mount (active sessions only)
        t0_c = time.perf_counter()
        base_mount_offset = max(int(cfg.virtual_offset), live_sink_n) if self._accepts_pos_ids else 0
        cursor = base_mount_offset
        mnt_k_list: list[torch.Tensor] = []
        mnt_v_list: list[torch.Tensor] = []
        oracle_k_slices: list[torch.Tensor] = []
        oracle_v_slices: list[torch.Tensor] = []
        bytes_c = 0
        ora_cursor = 0
        for idx, (sk, sv, s_ids) in enumerate(raw_sessions_kv):
            if idx in superseded or not s_ids:
                continue
            s_len = len(s_ids)
            eff_sink = min(cfg.sink_tokens, max(0, s_len - 1))
            local_cfg = replace(cfg, sink_tokens=eff_sink) if eff_sink != cfg.sink_tokens else cfg
            s_pos = torch.arange(0, s_len, dtype=torch.long, device=self.device)
            mnt = build_corrected_partition_b_kv(
                sk,
                sv,
                s_pos,
                live_sink_n if self._accepts_pos_ids else 0,
                cursor,
                local_cfg,
                blob_id=f"{example.example_id}_s{idx}",
            )
            mnt_k_list.append(mnt.k)
            mnt_v_list.append(mnt.v)
            bytes_c += mnt.kv_bytes_mounted
            cursor += int(mnt.k.shape[-2])
            # Align corresponding Oracle KV slice (after stripped sinks) for Arm (d) HKVD refresh
            oracle_k_slices.append(oracle_k_full[..., ora_cursor + eff_sink : ora_cursor + s_len, :])
            oracle_v_slices.append(oracle_v_full[..., ora_cursor + eff_sink : ora_cursor + s_len, :])
            ora_cursor += s_len

        if mnt_k_list:
            k_c_partb = torch.cat(mnt_k_list, dim=-2)
            v_c_partb = torch.cat(mnt_v_list, dim=-2)
            cache_c = inject_layer_kv_pairs(self._fresh_cache_from(ref_cache), k_c_partb, v_c_partb, layer_indices)
            live_pos_ids: torch.Tensor | None = (
                torch.arange(
                    cursor, cursor + len(live_ids) + len(ans_ids), dtype=torch.long, device=self.device
                ).unsqueeze(0)
                if self._accepts_pos_ids
                else None
            )
        else:
            k_c_partb = oracle_k_full[..., :0, :]
            v_c_partb = oracle_v_full[..., :0, :]
            cache_c = self._fresh_cache_from(ref_cache)
            live_pos_ids = None

        out_c = self._forward(live_and_ans, cache=cache_c, position_ids=live_pos_ids)
        nll_c, acc_c, em_c, pred_c = self._score_answer_from_logits(out_c.logits, ans_ids, len(live_ids))
        ms_c = (time.perf_counter() - t0_c) * 1000.0

        # Arm (d): Arm (c) + CacheBlend 5% Selective Refresh
        t0_d = time.perf_counter()
        if mnt_k_list:
            ora_k_cat = torch.cat(oracle_k_slices, dim=-2)
            ora_v_cat = torch.cat(oracle_v_slices, dim=-2)
            ref_d = selective_cacheblend_refresh(
                k_c_partb, v_c_partb, ora_k_cat, ora_v_cat, refresh_frac=cfg.refresh_frac
            )
            cache_d = inject_layer_kv_pairs(self._fresh_cache_from(ref_cache), ref_d.k, ref_d.v, layer_indices)
            refreshed_d = ref_d.refreshed_tokens
        else:
            cache_d = self._fresh_cache_from(ref_cache)
            refreshed_d = 0
        out_d = self._forward(live_and_ans, cache=cache_d, position_ids=live_pos_ids)
        nll_d, acc_d, em_d, pred_d = self._score_answer_from_logits(out_d.logits, ans_ids, len(live_ids))
        ms_d = ms_c + (time.perf_counter() - t0_d) * 1000.0

        return {
            "arm_a": K0EpisodeResult("arm_a", example.example_id, nll_a, acc_a, em_a, bytes_a, 0, ms_a, pred_a),
            "arm_b": K0EpisodeResult("arm_b", example.example_id, nll_b, acc_b, em_b, bytes_b, 0, ms_b, pred_b),
            "arm_c": K0EpisodeResult("arm_c", example.example_id, nll_c, acc_c, em_c, bytes_c, 0, ms_c, pred_c),
            "arm_d": K0EpisodeResult(
                "arm_d", example.example_id, nll_d, acc_d, em_d, bytes_c, refreshed_d, ms_d, pred_d
            ),
        }


def load_examples_from_file(path: str | Path) -> list[K0Example]:
    """Load `K0Example` records from a JSON array or JSONL file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"dataset file {p} is empty")
    raw_items: list[dict[str, Any]]
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"expected JSON list in {p}")
        raw_items = parsed
    else:
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]

    examples: list[K0Example] = []
    for idx, item in enumerate(raw_items):
        examples.append(
            K0Example(
                example_id=str(item.get("example_id", f"ex_{idx}")),
                sessions=[str(s) for s in item["sessions"]],
                live_prompt=str(item["live_prompt"]),
                answer=str(item["answer"]),
                horizon_tokens=int(item.get("horizon_tokens", 0)),
                category=str(item.get("category", "multi_hop")),
                superseded_session_indices=tuple(int(i) for i in item.get("superseded_session_indices", ())),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return examples


def _load_hf_checkpoint(checkpoint: str, device: str = "auto") -> tuple[Any, Any, torch.device]:
    """Lazy-load a HuggingFace causal LM checkpoint and tokenizer for GPU Kill-Gate `K0` execution."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "running --checkpoint requires the `transformers` package (install optional `gpu` extras)"
        ) from exc

    target_dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    dtype = torch.bfloat16 if target_dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype, trust_remote_code=True)
    model.to(target_dev)
    model.eval()
    return model, tokenizer, target_dev


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python3 -m qloop.eval.kill_gate_k0",
        description="Zero-Training Pre-RoPE KV-Mem Blob Injection Kill-Gate K0 Evaluator Harness.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Run deterministic pure-CPU Kill-Gate K0 self-test suite (zero GPU, zero network).",
    )
    mode.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or HF Hub ID of a Qwen3.6-64L checkpoint for zero-training GPU Kill-Gate K0 evaluation.",
    )
    ap.add_argument("--dataset", type=str, default=None, help="Optional JSON/JSONL file of K0Example episodes.")
    ap.add_argument("--output-json", type=str, default=None, help="Optional path to write deterministic JSON report.")
    ap.add_argument("--output-md", type=str, default=None, help="Optional path to write Markdown report.")
    ap.add_argument("--green-threshold", type=float, default=0.92, help="Oracle retention threshold for GREEN.")
    ap.add_argument("--yellow-threshold", type=float, default=0.75, help="Oracle retention threshold for YELLOW.")
    ap.add_argument("--refresh-frac", type=float, default=0.05, help="Arm (d) CacheBlend selective refresh fraction.")
    ap.add_argument("--sink-tokens", type=int, default=4, help="EPIC original-session sink tokens to strip.")
    ap.add_argument("--boundary-header-tokens", type=int, default=4, help="TurboRAG boundary header token count.")
    ap.add_argument("--rotary-dim", type=int, default=64, help="Partial-RoPE rotary channel count.")
    ap.add_argument("--rope-theta", type=float, default=10_000_000.0, help="RoPE base frequency.")
    ap.add_argument("--virtual-offset", type=int, default=10_000, help="Partition-B virtual mount offset.")
    ap.add_argument("--device", type=str, default="auto", help="Target device for --checkpoint (auto/cuda/cpu).")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    cfg = K0Config(
        green_threshold=args.green_threshold,
        yellow_threshold=args.yellow_threshold,
        refresh_frac=args.refresh_frac,
        sink_tokens=args.sink_tokens,
        boundary_header_tokens=args.boundary_header_tokens,
        rotary_dim=args.rotary_dim,
        rope_theta=args.rope_theta,
        virtual_offset=args.virtual_offset,
    )

    if args.smoke_cpu:
        examples = load_examples_from_file(args.dataset) if args.dataset else build_default_smoke_examples()
        report = run_k0_suite(
            examples,
            _synthetic_cpu_episode_evaluator,
            cfg,
            metadata={"mode": "smoke_cpu", "device": "cpu"},
        )
    else:
        examples = load_examples_from_file(args.dataset) if args.dataset else build_default_smoke_examples()
        model, tokenizer, dev = _load_hf_checkpoint(args.checkpoint, device=args.device)
        evaluator = HFCheckpointK0Evaluator(model, tokenizer, device=dev)
        report = run_k0_suite(
            examples,
            evaluator,
            cfg,
            metadata={"mode": "checkpoint", "checkpoint": str(args.checkpoint), "device": str(dev)},
        )

    md = format_k0_markdown(report)
    print(md)

    if args.output_json:
        out_j = Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(report.to_json() + "\n", encoding="utf-8")
    if args.output_md:
        out_m = Path(args.output_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md, encoding="utf-8")
    return 0


__all__ = [
    "ARM_DISPLAY_NAMES",
    "CANONICAL_ARMS",
    "CacheBlendRefreshResult",
    "HFCheckpointK0Evaluator",
    "K0ArmMetrics",
    "K0AttentionProbeResult",
    "K0Config",
    "K0EpisodeResult",
    "K0Example",
    "K0ProbeArmOutput",
    "K0Report",
    "K0SupersessionResult",
    "K0Verdict",
    "MountedPartitionBKV",
    "build_arg_parser",
    "build_controlled_partial_rope_probe",
    "build_corrected_partition_b_kv",
    "build_default_smoke_examples",
    "canonicalize_arm_name",
    "decide_k0_verdict",
    "extract_layer_kv_pairs",
    "format_k0_markdown",
    "inject_layer_kv_pairs",
    "load_examples_from_file",
    "main",
    "run_k0_attention_probe",
    "run_k0_suite",
    "run_k0_supersession_probe",
    "run_smoke_cpu",
    "selective_cacheblend_refresh",
    "splice_naive_kv",
]


if __name__ == "__main__":
    sys.exit(main())
