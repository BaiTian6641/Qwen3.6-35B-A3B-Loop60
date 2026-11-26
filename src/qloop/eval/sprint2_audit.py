"""Sprint 2 Scientific Validation & Causal Ablation Harness (`S2.1` + `S2.3`).

Implements the CPU-verified and GPU-ready evaluation suite defined in
`docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md` §6.2:

  1. **Leak-Free Holdout & OOD Batch Builder (`A1` / `QL60-003`):**
     Reuses `qloop.data.mixture.build_train_stream(..., split_role="eval", single_pass=True)`
     plus optional OOD sources or local JSONL/text files, recording a canonical SHA-256
     dataset digest and verifying zero document-ID overlap (`assert_zero_doc_overlap`).
     Binds cached JSON artifacts to a canonical `QL60-010` `run_identity` fingerprint so
     cross-checkpoint or cross-config cache mixing fails loudly.
  2. **`A3` Multi-Repeat Nondeterminism & Statistical Significance Gate (`run_depth_nll_sweep`):**
     Evaluates pure LM cross-entropy (valid supervised tokens only, zero aux/KL contamination
     per `QL60-009` / `C1`) across requested depths for `n_repeats` repeats plus an explicit
     `2x d=1` nondeterminism check, computing `mean_nll`, `std_nll`, `ppl`,
     `depth_gain_d1_to_dmax`, `z_score`, `exceeds_3sigma`, and head-to-head baseline gates.
  3. **`B2` / `B3` / `B4` / `B5` Length-Ladder (`8K - 1M`), Dense-vs-Sparse `D-LSA`,
     RULER Multi-Needle & `GDN-2` State Norms (`run_length_ladder_audit`):**
     Streams contexts chunk-by-chunk without materializing `[B, T, V]` full-sequence logits,
     evaluating chunked NLL, multi-needle exact-key/value retrieval accuracy & answer NLL,
     dense-vs-sparse `D-LSA` gap + live `lsa/last_recall`, and `GDN-2` recurrent state norms
     (`max_l2_norm`, `max_abs_val`, `nan_count`) across all linear-attention cache slots.
  4. **`S2.3` Zero-Training Causal Ablation Context Managers & Matrix (`run_causal_ablation_matrix`):**
     Reversible, bit-exact SHA-256-verified context managers for `full`, `no_gdn2`, `no_qed`,
     `no_neg_eigval`, `no_pass_router`, `low_anchor_probe` (`w_anchor <= 0.3` simplex shift
     per McLeish et al. arXiv:2511.07384), `lsa_dense`, and `lsa_sparse`, emitting per-ablation
     `delta_nll_vs_full`, `z_score`, and recommendation tags (`KEEP_CRITICAL`,
     `PRUNE_CANDIDATE`, `SHORTCUT_DIAGNOSTIC`).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qloop.data.formats import Example
from qloop.data.mixture import build_train_stream, doc_holdout_key
from qloop.data.packing import IGNORE, Packer, separator_id, tokenize_example
from qloop.data.tokenizer import ToyTokenizer
from qloop.eval.runners import _RUN_IDENTITY_KEYS, _checkpoint_path_digest, _model_weights_digest
from qloop.model._cache_compat import cache_kwarg_name, extract_output_cache
from qloop.model.gdn2 import GDN2Adapter, GDN2Config, GDN2Manager, install_gdn2
from qloop.model.introspect import find_lm_head, find_text_model
from qloop.model.loop import LoopController, LoopSpec, PassRouterAdapters, apply_loop
from qloop.model.lsa import LSAConfig, LSAIndexer, LSAManager, install_lsa
from qloop.model.mhc import V3_BRIDGE_WEIGHTS, LoopBridge
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.loss import lm_loss

log = logging.getLogger("qloop")

DEFAULT_GPU_LENGTHS: tuple[int, ...] = (8192, 65536, 131072, 524288, 1048576)
DEFAULT_SMOKE_LENGTHS: tuple[int, ...] = (64, 128, 256)
DEFAULT_NEEDLE_DEPTHS: tuple[float, ...] = (0.10, 0.35, 0.65, 0.90)
CANONICAL_ABLATIONS: tuple[str, ...] = (
    "full",
    "no_gdn2",
    "no_qed",
    "no_neg_eigval",
    "no_pass_router",
    "low_anchor_probe",
    "lsa_dense",
    "lsa_sparse",
)
RECOMMENDATION_TAGS: tuple[str, ...] = (
    "KEEP_CRITICAL",
    "PRUNE_CANDIDATE",
    "SHORTCUT_DIAGNOSTIC",
)
SPRINT2_SCHEMA_VERSION = "sprint2_audit_v1"

_RULER_FILLER_SENTENCES: tuple[str, ...] = (
    "The telemetry archive records steady signal propagation across seasonal intervals.",
    "Observers noted calm barometric readings along the eastern ridge before dusk.",
    "Routine calibration confirmed nominal thermal balance inside the instrument bay.",
    "Archival ledgers cataloged coastal tides, compass bearings, and vessel drafts.",
    "Standard diagnostic logs remained clear throughout the scheduled maintenance window.",
    "Field engineers documented granite strata and rainfall totals across the valley.",
)

_RULER_KEY_WORDS: tuple[str, ...] = (
    "ALPHA",
    "BRAVO",
    "CHARLIE",
    "DELTA",
    "ECHO",
    "FOXTROT",
    "GOLF",
    "HOTEL",
    "INDIA",
    "JULIET",
    "KILO",
    "LIMA",
)


# ==============================================================================
# JSON Serialization & Statistical Helpers
# ==============================================================================


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _safe_ppl(nll: float) -> float:
    if not math.isfinite(nll):
        return float("inf")
    return float(math.exp(min(max(nll, -80.0), 80.0)))


def sample_std(values: Sequence[float]) -> float:
    """Unbiased sample standard deviation (`ddof=1`), returning `0.0` for `len <= 1`."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(float(v) for v in values) / float(n)
    var = sum((float(v) - mean) ** 2 for v in values) / float(n - 1)
    return math.sqrt(max(0.0, var))


def pooled_sample_std(std_a: float, n_a: int, std_b: float, n_b: int) -> float:
    """Pooled sample standard deviation across two repeated-evaluation arms."""
    na = int(n_a)
    nb = int(n_b)
    if na <= 1 and nb <= 1:
        return 0.0
    if na <= 1:
        return float(std_b)
    if nb <= 1:
        return float(std_a)
    dof = (na - 1) + (nb - 1)
    num = (na - 1) * (float(std_a) ** 2) + (nb - 1) * (float(std_b) ** 2)
    return math.sqrt(max(0.0, num / float(max(dof, 1))))


def compute_z_and_3sigma(delta: float, pooled_std: float) -> tuple[float, bool]:
    """Compute `z_score = delta / max(pooled_std, 1e-9)` and `exceeds_3sigma` gate."""
    denom = max(float(pooled_std), 1e-9)
    z_score = float(delta) / denom
    exceeds = bool(abs(float(delta)) > 3.0 * float(pooled_std) and float(delta) > 0.0)
    return z_score, exceeds


# ==============================================================================
# Report Data Structures
# ==============================================================================


@dataclass
class EvalBatchSet:
    """Deterministic, SHA-256-fingerprinted evaluation batch bundle with canonical doc IDs."""

    name: str
    split_role: str
    batches: list[dict[str, Any]]
    doc_ids: list[str]
    dataset_sha256: str
    holdout_mod: int = 100
    holdout_eval_buckets: int = 2
    total_tokens: int = 0
    valid_loss_tokens: int = 0
    overlap_check: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split_role": self.split_role,
            "num_batches": len(self.batches),
            "num_docs": len(self.doc_ids),
            "dataset_sha256": self.dataset_sha256,
            "holdout_mod": int(self.holdout_mod),
            "holdout_eval_buckets": int(self.holdout_eval_buckets),
            "total_tokens": int(self.total_tokens),
            "valid_loss_tokens": int(self.valid_loss_tokens),
            "overlap_check": _to_jsonable(self.overlap_check),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class DepthMetrics:
    """Pure LM NLL statistics across repeated evaluations at one loop depth."""

    depth: int
    n_repeats: int
    repeat_nlls: list[float]
    mean_nll: float
    std_nll: float
    ppl: float
    valid_tokens: int
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DepthSweepResult:
    """Multi-repeat depth-scaling sweep + explicit `2x d=1` nondeterminism check (`A3`)."""

    arm_name: str
    dataset_name: str
    dataset_sha256: str
    depths: dict[int, DepthMetrics]
    n_repeats: int
    d1_repeat_nlls: list[float]
    d1_repeat_max_abs_diff: float
    d1_repeat_std: float
    depth_gain_d1_to_dmax: float
    pooled_std_d1_to_dmax: float
    z_score: float
    exceeds_3sigma: bool
    d_min: int
    d_max: int
    run_identity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "dataset_name": self.dataset_name,
            "dataset_sha256": self.dataset_sha256,
            "depths": {str(k): v.to_dict() for k, v in sorted(self.depths.items())},
            "n_repeats": int(self.n_repeats),
            "d1_repeat_nlls": [float(x) for x in self.d1_repeat_nlls],
            "d1_repeat_max_abs_diff": float(self.d1_repeat_max_abs_diff),
            "d1_repeat_std": float(self.d1_repeat_std),
            "depth_gain_d1_to_dmax": float(self.depth_gain_d1_to_dmax),
            "pooled_std_d1_to_dmax": float(self.pooled_std_d1_to_dmax),
            "z_score": float(self.z_score),
            "exceeds_3sigma": bool(self.exceeds_3sigma),
            "d_min": int(self.d_min),
            "d_max": int(self.d_max),
            "run_identity": _to_jsonable(self.run_identity),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class ArmComparisonRow:
    """Head-to-head statistical comparison of a candidate arm against a baseline arm."""

    arm_name: str
    baseline_arm: str
    arm_depth: int
    baseline_depth: int
    arm_mean_nll: float
    arm_std_nll: float
    arm_ppl: float
    baseline_mean_nll: float
    baseline_std_nll: float
    baseline_ppl: float
    delta_vs_baseline: float
    pooled_std_vs_baseline: float
    z_score_vs_baseline: float
    exceeds_3sigma: bool
    passes_s2_gate: bool
    ppl_reduction_pct: float
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GDN2StateNormTelemetry:
    """Boundedness telemetry over all linear-attention recurrent states (`B5`)."""

    num_linear_states: int
    max_l2_norm: float
    mean_l2_norm: float
    max_abs_val: float
    nan_count: int
    inf_count: int
    is_bounded: bool
    per_layer: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_linear_states": int(self.num_linear_states),
            "max_l2_norm": float(self.max_l2_norm),
            "mean_l2_norm": float(self.mean_l2_norm),
            "max_abs_val": float(self.max_abs_val),
            "nan_count": int(self.nan_count),
            "inf_count": int(self.inf_count),
            "is_bounded": bool(self.is_bounded),
            "per_layer": _to_jsonable(self.per_layer),
        }


@dataclass
class RulerMultiNeedleResult:
    """Multi-needle RULER-style key/value retrieval telemetry at a target context length."""

    target_length: int
    actual_prompt_tokens: int
    num_needles: int
    needle_depths: list[float]
    exact_match_acc: float
    needles_matched: int
    answer_token_acc: float
    answer_nll: float
    per_needle_hits: list[bool]
    decode_mode: str = "autoregressive_greedy"
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LengthLadderPoint:
    """Single context-length point of the `8K - 1M` capability & telemetry ladder."""

    length: int
    chunk_size: int
    dense_nll: float
    dense_ppl: float
    valid_tokens: int
    sparse_nll: float | None = None
    sparse_ppl: float | None = None
    dense_sparse_nll_gap: float | None = None
    lsa_last_recall: float | None = None
    lsa_sparse_active: bool = False
    ruler: RulerMultiNeedleResult | None = None
    gdn2_state_norms: GDN2StateNormTelemetry | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": int(self.length),
            "chunk_size": int(self.chunk_size),
            "dense_nll": float(self.dense_nll),
            "dense_ppl": float(self.dense_ppl),
            "valid_tokens": int(self.valid_tokens),
            "sparse_nll": None if self.sparse_nll is None else float(self.sparse_nll),
            "sparse_ppl": None if self.sparse_ppl is None else float(self.sparse_ppl),
            "dense_sparse_nll_gap": None if self.dense_sparse_nll_gap is None else float(self.dense_sparse_nll_gap),
            "lsa_last_recall": None if self.lsa_last_recall is None else float(self.lsa_last_recall),
            "lsa_sparse_active": bool(self.lsa_sparse_active),
            "ruler": None if self.ruler is None else self.ruler.to_dict(),
            "gdn2_state_norms": None if self.gdn2_state_norms is None else self.gdn2_state_norms.to_dict(),
            "elapsed_ms": float(self.elapsed_ms),
        }


@dataclass
class LengthLadderResult:
    """Complete `B2 / B3 / B4 / B5` length-ladder evaluation result."""

    arm_name: str
    depth: int
    kv_cache_quant: str
    points: list[LengthLadderPoint]
    all_states_bounded: bool
    max_dense_sparse_nll_gap: float | None = None
    min_lsa_recall: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "depth": int(self.depth),
            "kv_cache_quant": self.kv_cache_quant,
            "points": [p.to_dict() for p in self.points],
            "all_states_bounded": bool(self.all_states_bounded),
            "max_dense_sparse_nll_gap": (
                None if self.max_dense_sparse_nll_gap is None else float(self.max_dense_sparse_nll_gap)
            ),
            "min_lsa_recall": None if self.min_lsa_recall is None else float(self.min_lsa_recall),
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class AblationRow:
    """Single zero-training causal ablation outcome with restoration verification."""

    ablation: str
    mean_nll: float
    std_nll: float
    ppl: float
    repeat_nlls: list[float]
    delta_nll_vs_full: float
    pooled_std: float
    z_score: float
    exceeds_3sigma: bool
    recommendation: str
    applicable: bool = True
    is_baseline: bool = False
    restoration_verified: bool = True
    weight_sha256_before: str = ""
    weight_sha256_after: str = ""
    lsa_last_recall: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "mean_nll": float(self.mean_nll),
            "std_nll": float(self.std_nll),
            "ppl": float(self.ppl),
            "repeat_nlls": [float(x) for x in self.repeat_nlls],
            "delta_nll_vs_full": float(self.delta_nll_vs_full),
            "pooled_std": float(self.pooled_std),
            "z_score": float(self.z_score),
            "exceeds_3sigma": bool(self.exceeds_3sigma),
            "recommendation": self.recommendation,
            "applicable": bool(self.applicable),
            "is_baseline": bool(self.is_baseline),
            "restoration_verified": bool(self.restoration_verified),
            "weight_sha256_before": self.weight_sha256_before,
            "weight_sha256_after": self.weight_sha256_after,
            "lsa_last_recall": None if self.lsa_last_recall is None else float(self.lsa_last_recall),
            "diagnostics": _to_jsonable(self.diagnostics),
        }


@dataclass
class CausalAblationMatrixResult:
    """Complete `S2.3` causal ON/OFF ablation matrix result."""

    arm_name: str
    depth: int
    n_repeats: int
    rows: list[AblationRow]
    all_restorations_verified: bool
    full_weight_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_row(self, ablation: str) -> AblationRow | None:
        for r in self.rows:
            if r.ablation == ablation:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "depth": int(self.depth),
            "n_repeats": int(self.n_repeats),
            "rows": [r.to_dict() for r in self.rows],
            "all_restorations_verified": bool(self.all_restorations_verified),
            "full_weight_sha256": self.full_weight_sha256,
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class Sprint2AuditReport:
    """Unified Sprint 2 (`S2.1` + `S2.3`) validation & causal ablation report."""

    mode: str
    datasets: dict[str, dict[str, Any]]
    depth_sweeps: dict[str, dict[str, DepthSweepResult]]  # dataset_name -> arm_name -> sweep
    comparisons: dict[str, list[ArmComparisonRow]]  # dataset_name -> comparison rows
    length_ladders: dict[str, LengthLadderResult] = field(default_factory=dict)
    causal_ablations: dict[str, CausalAblationMatrixResult] = field(default_factory=dict)
    run_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPRINT2_SCHEMA_VERSION,
            "mode": self.mode,
            "datasets": _to_jsonable(self.datasets),
            "depth_sweeps": {
                ds: {arm: sw.to_dict() for arm, sw in sorted(arms.items())}
                for ds, arms in sorted(self.depth_sweeps.items())
            },
            "comparisons": {
                ds: [row.to_dict() for row in rows]
                for ds, rows in sorted(self.comparisons.items())
            },
            "length_ladders": {
                arm: ll.to_dict() for arm, ll in sorted(self.length_ladders.items())
            },
            "causal_ablations": {
                arm: ca.to_dict() for arm, ca in sorted(self.causal_ablations.items())
            },
            "run_identities": _to_jsonable(self.run_identities),
            "metadata": _to_jsonable(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# ==============================================================================
# 1. Leak-Free Holdout & OOD Batch Builder (A1 / QL60-003) & QL60-010 Identity
# ==============================================================================


def assert_zero_doc_overlap(
    train_doc_ids: Iterable[str],
    eval_doc_ids: Iterable[str],
) -> dict[str, Any]:
    """Assert exact zero overlap between canonical training and evaluation document IDs (`A1`/`QL60-003`)."""
    train_set = {str(d).strip() for d in train_doc_ids if str(d).strip()}
    eval_set = {str(d).strip() for d in eval_doc_ids if str(d).strip()}
    overlap = sorted(train_set & eval_set)
    if overlap:
        sample = overlap[:10]
        raise ValueError(
            f"Leak-free holdout violation (A1 / QL60-003): detected {len(overlap)} overlapping document ID(s) "
            f"between train ({len(train_set)} docs) and eval ({len(eval_set)} docs); sample={sample}"
        )
    return {
        "verified": True,
        "train_docs": len(train_set),
        "eval_docs": len(eval_set),
        "overlap_count": 0,
    }


def compute_batch_set_digest(
    batches: Sequence[Mapping[str, Any]],
    *,
    split_role: str = "",
    doc_ids: Sequence[str] = (),
) -> str:
    """Compute a deterministic SHA-256 digest over packed evaluation batch tensors and doc IDs."""
    h = hashlib.sha256()
    h.update(f"sprint2_batches_v1|split={split_role}|n={len(batches)}".encode())
    for idx, b in enumerate(batches):
        ids_t = torch.as_tensor(b["input_ids"], dtype=torch.int64).detach().cpu().contiguous()
        lbl_t = torch.as_tensor(b["labels"], dtype=torch.int64).detach().cpu().contiguous()
        h.update(f"\x00batch:{idx}|ids:{tuple(ids_t.shape)}|lbl:{tuple(lbl_t.shape)}".encode())
        h.update(ids_t.numpy().tobytes())
        h.update(lbl_t.numpy().tobytes())
    h.update(f"\x00docs:{len(doc_ids)}".encode())
    for d in doc_ids:
        h.update(b"\x01")
        h.update(str(d).encode("utf-8", "replace"))
    return h.hexdigest()


def _collect_tracked_stream_batches(
    stream: Any,
    n_batches: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Consume up to `n_batches` from a `TrainStream` while recording exact canonical `doc_holdout_key`s."""
    captured_keys: list[str] = []
    orig_process = stream._process

    def _tracking_process(name: str, ex: Any) -> bool:
        accepted = orig_process(name, ex)
        if accepted:
            captured_keys.append(doc_holdout_key(ex))
        return accepted

    stream._process = _tracking_process
    batches: list[dict[str, Any]] = []
    try:
        for _ in range(max(0, int(n_batches))):
            try:
                batches.append(next(stream))
            except StopIteration:
                break
            except RuntimeError as exc:
                if "produced no items" not in str(exc):
                    raise
                log.warning("[sprint2] eval stream exhausted early: %s", exc)
                break
    finally:
        stream._process = orig_process

    seen_keys: dict[str, None] = {}
    for row in batches:
        doc_t = row.get("doc_ids")
        if isinstance(doc_t, torch.Tensor):
            for raw_id in doc_t.reshape(-1).tolist():
                di = int(raw_id)
                if 0 <= di < len(captured_keys):
                    seen_keys[captured_keys[di]] = None
    if not seen_keys:
        for k in captured_keys:
            seen_keys[k] = None
    return batches, list(seen_keys.keys())


def _batch_token_counts(batches: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    total_toks = 0
    valid_loss = 0
    for b in batches:
        ids = torch.as_tensor(b["input_ids"])
        lbl = torch.as_tensor(b["labels"])
        total_toks += int(ids.numel())
        if lbl.ndim == 1:
            valid_loss += int((lbl[1:] != IGNORE).sum().item()) if lbl.numel() > 1 else 0
        else:
            valid_loss += int((lbl[:, 1:] != IGNORE).sum().item()) if lbl.shape[-1] > 1 else 0
    return total_toks, valid_loss


def build_holdout_eval_batches(
    data_cfg: Mapping[str, Any],
    tokenizer: Any,
    *,
    n_batches: int = 4,
    seed: int = 0,
    split_role: str = "eval",
    holdout_mod: int | None = None,
    holdout_eval_buckets: int | None = None,
    seq_len: int | None = None,
    label: str = "holdout",
    train_doc_ids: Iterable[str] | None = None,
    verify_against_train_split: bool = False,
    train_verify_batches: int | None = None,
) -> EvalBatchSet:
    """Build deterministic leak-free eval batches via `build_train_stream(..., split_role='eval', single_pass=True)`."""
    raw = copy.deepcopy(dict(data_cfg))
    if "sources" not in raw and isinstance(raw.get("data"), dict):
        raw = copy.deepcopy(raw["data"])
    if seq_len is not None:
        raw["seq_len"] = int(seq_len)
    eff_mod = int(holdout_mod if holdout_mod is not None else raw.get("holdout_mod", 100))
    eff_eval = int(holdout_eval_buckets if holdout_eval_buckets is not None else raw.get("holdout_eval_buckets", 2))
    raw["holdout_mod"] = eff_mod
    raw["holdout_eval_buckets"] = eff_eval

    norm_sources: list[Any] = []
    for src in list(raw.get("sources") or []):
        if isinstance(src, dict):
            s = dict(src)
            s["split_role"] = str(split_role)
            s["holdout_mod"] = eff_mod
            s["holdout_eval_buckets"] = eff_eval
            norm_sources.append(s)
        else:
            norm_sources.append(src)
    raw["sources"] = norm_sources

    stream = build_train_stream(
        raw,
        tokenizer,
        seed=int(seed),
        split_role=str(split_role),
        holdout_mod=eff_mod,
        holdout_eval_buckets=eff_eval,
        single_pass=True,
        skip_empty_sources=True,
    )
    batches, eval_doc_ids = _collect_tracked_stream_batches(stream, n_batches=int(n_batches))
    if not batches:
        raise RuntimeError(
            f"build_holdout_eval_batches({label!r}): produced 0 batches "
            f"(split_role={split_role!r}, holdout={eff_eval}/{eff_mod})"
        )

    overlap_info: dict[str, Any] = {}
    if train_doc_ids is not None:
        overlap_info = assert_zero_doc_overlap(train_doc_ids, eval_doc_ids)
    elif verify_against_train_split and split_role == "eval":
        train_cfg = copy.deepcopy(raw)
        for src in train_cfg["sources"]:
            if isinstance(src, dict):
                src["split_role"] = "train"
        tr_stream = build_train_stream(
            train_cfg,
            tokenizer,
            seed=int(seed),
            split_role="train",
            holdout_mod=eff_mod,
            holdout_eval_buckets=eff_eval,
            single_pass=True,
            skip_empty_sources=True,
        )
        tr_n = int(train_verify_batches if train_verify_batches is not None else max(n_batches * 4, 8))
        _tr_batches, sampled_train_docs = _collect_tracked_stream_batches(tr_stream, n_batches=tr_n)
        overlap_info = assert_zero_doc_overlap(sampled_train_docs, eval_doc_ids)

    digest = compute_batch_set_digest(batches, split_role=str(split_role), doc_ids=eval_doc_ids)
    total_toks, valid_loss = _batch_token_counts(batches)
    return EvalBatchSet(
        name=label,
        split_role=str(split_role),
        batches=batches,
        doc_ids=eval_doc_ids,
        dataset_sha256=digest,
        holdout_mod=eff_mod,
        holdout_eval_buckets=eff_eval,
        total_tokens=total_toks,
        valid_loss_tokens=valid_loss,
        overlap_check=overlap_info,
        metadata={"seed": int(seed), "seq_len": int(raw.get("seq_len", 4096))},
    )


def build_local_file_eval_batches(
    paths: Sequence[str | Path],
    tokenizer: Any,
    *,
    seq_len: int = 128,
    n_batches: int | None = None,
    label: str = "local_ood",
    train_doc_ids: Iterable[str] | None = None,
    respect_doc_boundaries: bool = True,
) -> EvalBatchSet:
    """Build deterministic packed eval batches from local `.jsonl`, `.json`, or `.txt` files."""
    if not paths:
        raise ValueError("build_local_file_eval_batches requires at least one file path")
    sep = separator_id(tokenizer)
    pad = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else sep
    packer = Packer(int(seq_len), int(pad), respect_doc_boundaries=bool(respect_doc_boundaries))

    doc_keys: list[str] = []
    batches: list[dict[str, Any]] = []
    doc_counter = 0

    def _emit_example(ex: Example, raw_row: dict[str, Any] | None = None) -> None:
        nonlocal doc_counter
        ids, lbls = tokenize_example(ex, tokenizer, sep)
        if not ids:
            return
        dkey = doc_holdout_key(ex, row=raw_row)
        doc_keys.append(dkey)
        batches.extend(packer.add(ids, lbls, doc_counter))
        doc_counter += 1

    for raw_path in paths:
        p = Path(raw_path)
        if not p.is_file():
            raise FileNotFoundError(f"local eval file not found: {p}")
        suffix = p.suffix.lower()
        content = p.read_text(encoding="utf-8")
        if suffix in (".jsonl", ".json"):
            if suffix == ".json":
                parsed = json.loads(content)
                rows = parsed if isinstance(parsed, list) else [parsed]
            else:
                rows = [json.loads(ln) for ln in content.splitlines() if ln.strip()]
            for ridx, row in enumerate(rows):
                if isinstance(row, str):
                    ex = Example(source=p.name, kind="text", text=row, row_index=ridx)
                    _emit_example(ex)
                elif isinstance(row, dict):
                    meta: dict[str, Any] = {}
                    for idk in ("doc_id", "id", "task_id"):
                        if row.get(idk) is not None and str(row.get(idk)).strip():
                            meta["doc_id"] = str(row[idk]).strip()
                            break
                    if isinstance(row.get("messages"), list) and row["messages"]:
                        ex = Example(
                            source=p.name,
                            kind="chat",
                            messages=row["messages"],
                            meta=meta,
                            row_index=ridx,
                        )
                    elif "prompt" in row and "response" in row:
                        msgs = [
                            {"role": "user", "content": str(row["prompt"])},
                            {"role": "assistant", "content": str(row["response"])},
                        ]
                        ex = Example(source=p.name, kind="chat", messages=msgs, meta=meta, row_index=ridx)
                    else:
                        txt = str(row.get("text", row.get("content", "")))
                        if not txt.strip():
                            continue
                        ex = Example(source=p.name, kind="text", text=txt, meta=meta, row_index=ridx)
                    _emit_example(ex, raw_row=row)
                if n_batches is not None and len(batches) >= int(n_batches):
                    break
        else:
            paragraphs = [seg.strip() for seg in content.split("\n\n") if seg.strip()]
            if not paragraphs and content.strip():
                paragraphs = [content.strip()]
            for ridx, para in enumerate(paragraphs):
                ex = Example(source=p.name, kind="text", text=para, row_index=ridx)
                _emit_example(ex)
                if n_batches is not None and len(batches) >= int(n_batches):
                    break
        if n_batches is not None and len(batches) >= int(n_batches):
            break

    if n_batches is None or len(batches) < int(n_batches):
        flushed = packer.flush()
        if flushed is not None:
            batches.append(flushed)
    if n_batches is not None:
        batches = batches[: int(n_batches)]
    if not batches:
        raise RuntimeError(f"build_local_file_eval_batches({label!r}): produced 0 batches from {paths}")

    seen: dict[str, None] = {}
    for row in batches:
        dt = row.get("doc_ids")
        if isinstance(dt, torch.Tensor):
            for raw_id in dt.reshape(-1).tolist():
                di = int(raw_id)
                if 0 <= di < len(doc_keys):
                    seen[doc_keys[di]] = None
    used_doc_ids = list(seen.keys()) if seen else list(dict.fromkeys(doc_keys))
    overlap_info = assert_zero_doc_overlap(train_doc_ids, used_doc_ids) if train_doc_ids is not None else {}
    digest = compute_batch_set_digest(batches, split_role="ood_local", doc_ids=used_doc_ids)
    total_toks, valid_loss = _batch_token_counts(batches)
    return EvalBatchSet(
        name=label,
        split_role="ood_local",
        batches=batches,
        doc_ids=used_doc_ids,
        dataset_sha256=digest,
        holdout_mod=0,
        holdout_eval_buckets=0,
        total_tokens=total_toks,
        valid_loss_tokens=valid_loss,
        overlap_check=overlap_info,
        metadata={"files": [str(Path(p).name) for p in paths], "seq_len": int(seq_len)},
    )


def build_ood_eval_batches(
    ood_sources: Sequence[Any],
    tokenizer: Any,
    *,
    seq_len: int = 128,
    n_batches: int = 4,
    seed: int = 777,
    label: str = "ood",
    train_doc_ids: Iterable[str] | None = None,
    split_role: str = "all",
    holdout_mod: int = 100,
    holdout_eval_buckets: int = 2,
) -> EvalBatchSet:
    """Build OOD evaluation batches from either `data.sources` specs or local JSONL/text file paths."""
    if not ood_sources:
        raise ValueError("build_ood_eval_batches requires non-empty ood_sources")
    if all(isinstance(s, (str, Path)) for s in ood_sources):
        return build_local_file_eval_batches(
            ood_sources,
            tokenizer,
            seq_len=seq_len,
            n_batches=n_batches,
            label=label,
            train_doc_ids=train_doc_ids,
        )
    data_cfg: dict[str, Any] = {
        "seq_len": int(seq_len),
        "sources": [dict(s) for s in ood_sources],
        "holdout_mod": int(holdout_mod),
        "holdout_eval_buckets": int(holdout_eval_buckets),
    }
    return build_holdout_eval_batches(
        data_cfg,
        tokenizer,
        n_batches=n_batches,
        seed=seed,
        split_role=split_role,
        holdout_mod=holdout_mod,
        holdout_eval_buckets=holdout_eval_buckets,
        seq_len=seq_len,
        label=label,
        train_doc_ids=train_doc_ids,
    )


def _tensor_exact_bytes(t: torch.Tensor) -> bytes:
    cpu_t = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return cpu_t.numpy().tobytes()


def compute_weights_sha256(
    model: nn.Module,
    *,
    include_buffers: bool = True,
    exact_limit: int = 50_000_000,
) -> str:
    """Compute a deterministic SHA-256 weight digest over model parameters (and optional buffers).

    Hashes every byte bit-exactly for any parameter/buffer when the model has `<= exact_limit`
    elements, and ALWAYS hashes every byte bit-exactly for all adapter/loop/gdn2/lsa/mixer/norm
    parameters on larger checkpoints while striding huge frozen expert tensors for fast GPU checks.
    """
    seen: set[int] = set()
    items: list[tuple[str, torch.Tensor]] = []
    for name, p in model.named_parameters():
        if p is not None and id(p) not in seen:
            seen.add(id(p))
            items.append((f"p:{name}", p))
    if include_buffers:
        for name, b in model.named_buffers():
            if b is not None and id(b) not in seen:
                seen.add(id(b))
                items.append((f"b:{name}", b))
    items.sort(key=lambda kv: kv[0])

    total_numel = sum(int(t.numel()) for _, t in items)
    full_exact = total_numel <= int(exact_limit)
    always_exact_tokens = ("gdn2_adapter", "loop_adapters", "lsa_indexer", "bridge", "pass_router", "pass_attn")

    h = hashlib.sha256()
    with torch.no_grad():
        for name, t in items:
            h.update(f"\x00{name}|{t.dtype}|{tuple(t.shape)}|".encode())
            n = int(t.numel())
            if n == 0:
                continue
            is_adapter = any(tok in name for tok in always_exact_tokens)
            if full_exact or is_adapter or n <= 65_536:
                h.update(_tensor_exact_bytes(t))
            else:
                flat = t.detach().reshape(-1)
                step = max(1, n // 1024)
                sample = torch.cat([flat[::step][:1024], flat[-32:]], dim=0)
                h.update(_tensor_exact_bytes(sample))
    return h.hexdigest()


def compute_ablation_state_sha256(
    model: nn.Module,
    *,
    loop: LoopController | None = None,
    gdn2: GDN2Manager | None = None,
    lsa: LSAManager | None = None,
) -> str:
    """Combined SHA-256 digest of all weights + mutable runtime ablation configuration flags."""
    h = hashlib.sha256()
    h.update(compute_weights_sha256(model, include_buffers=True).encode("ascii"))

    gdn2_adapters = _find_gdn2_adapters(model, gdn2=gdn2)
    for lid, ad in sorted(gdn2_adapters.items()):
        h.update(
            (
                f"\x00gdn2:{lid}|qed={ad.cfg.qed_enabled}:{ad.cfg.qed_max_lambda}"
                f"|neg={ad.cfg.neg_eigval_max_boost}"
            ).encode()
        )

    lsa_indexers = _find_lsa_indexers(model, lsa=lsa)
    for lid, ix in sorted(lsa_indexers.items()):
        h.update(f"\x00lsa:{lid}|mode={ix.cfg.mode}|rc={ix.cfg.recall_check_every}".encode())
    if lsa is not None:
        h.update(f"\x00lsa_mgr|mode={lsa.cfg.mode}|rc={lsa.cfg.recall_check_every}".encode())

    if loop is not None:
        h.update(f"\x00loop|active_passes={loop.active_passes}".encode())
    return h.hexdigest()


def build_sprint2_run_identity(
    model: nn.Module,
    *,
    arm_name: str,
    dataset_sha256: str,
    config_payload: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_sha: str | None = None,
    loop: LoopController | None = None,
) -> dict[str, Any]:
    """Build a `QL60-010`-compliant model/checkpoint/config/dataset identity fingerprint."""
    if model_id is not None and str(model_id).strip():
        resolved_model_id = str(model_id).strip()
    else:
        cfg = getattr(model, "config", None)
        name_or_path = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if isinstance(name_or_path, str) and name_or_path.strip():
            resolved_model_id = name_or_path.strip()
        else:
            mtype = getattr(cfg, "model_type", None) or type(model).__qualname__
            resolved_model_id = f"{type(model).__module__}.{type(model).__qualname__}:{mtype}"

    if checkpoint_sha is not None and str(checkpoint_sha).strip():
        resolved_ckpt_sha = str(checkpoint_sha).strip()
    elif checkpoint_path:
        p_dig = (
            _checkpoint_path_digest(checkpoint_path)
            if os.path.exists(checkpoint_path)
            else f"ref:{hashlib.sha256(checkpoint_path.encode('utf-8')).hexdigest()[:24]}"
        )
        resolved_ckpt_sha = f"{p_dig}|{_model_weights_digest(model, None, model)}"
    else:
        resolved_ckpt_sha = _model_weights_digest(model, None, model)

    active_passes: int | None = int(loop.active_passes) if loop is not None else 1
    cfg_raw = json.dumps(
        {
            "arm_name": arm_name,
            "config": _to_jsonable(dict(config_payload or {})),
            "schema_version": SPRINT2_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cfg_hash = hashlib.sha256(cfg_raw.encode("utf-8", "replace")).hexdigest()[:24]

    return {
        "model_id": resolved_model_id,
        "checkpoint_sha": resolved_ckpt_sha,
        "active_passes": active_passes,
        "config_hash": cfg_hash,
        "dataset_rev": str(dataset_sha256),
        "arm_name": str(arm_name),
        "schema_version": SPRINT2_SCHEMA_VERSION,
    }


def verify_or_load_cached_json(
    cache_path: str | Path,
    expected_identity: Mapping[str, Any],
    *,
    allow_mixed_cache: bool = False,
) -> dict[str, Any] | None:
    """Load cached Sprint 2 JSON only if its `QL60-010` `run_identity` matches exactly.

    Refuses silent cross-checkpoint / cross-config / cross-dataset cache mixing by raising
    `ValueError` unless `allow_mixed_cache=True` is passed explicitly.
    """
    p = Path(cache_path)
    if not p.exists():
        return None
    cached = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cached, dict):
        if not allow_mixed_cache:
            raise ValueError(f"Sprint 2 cache file {p} does not contain a JSON object")
        return None
    old_id = cached.get("run_identity")
    if not isinstance(old_id, dict):
        if not allow_mixed_cache:
            raise ValueError(
                f"Sprint 2 cache {p} lacks `run_identity` metadata (QL60-010); refusing unverified reuse. "
                "Use a fresh cache path or pass allow_mixed_cache=True."
            )
        log.warning("[sprint2] allow_mixed_cache=True: ignoring legacy cache without run_identity at %s", p)
        return None

    diffs = {
        k: (old_id.get(k), expected_identity.get(k))
        for k in _RUN_IDENTITY_KEYS
        if old_id.get(k) != expected_identity.get(k)
    }
    if diffs:
        if not allow_mixed_cache:
            raise ValueError(
                f"Sprint 2 cache identity mismatch at {p} (QL60-010): diffs={diffs}. "
                "Refusing silent cross-checkpoint/config cache mixing — use a fresh cache directory "
                "or pass allow_mixed_cache=True."
            )
        log.warning("[sprint2] allow_mixed_cache=True: overwriting mismatched cache at %s (diffs=%s)", p, diffs)
        return None
    return cached


def save_cached_json(
    cache_path: str | Path,
    payload: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> Path:
    """Persist a Sprint 2 JSON artifact stamped with its canonical `QL60-010` `run_identity`."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamped = dict(payload)
    stamped["run_identity"] = _to_jsonable(dict(run_identity))
    stamped["schema_version"] = SPRINT2_SCHEMA_VERSION
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(stamped), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


# ==============================================================================
# 2. Pure LM CE & A3 Multi-Repeat Depth NLL Sweep (QL60-009 / C1 / A3 / B4)
# ==============================================================================


def _resolve_model_device(model: nn.Module, explicit: torch.device | str | None = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _cleanup_eval_side_state(
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
) -> None:
    if loop is not None and callable(getattr(loop, "release_graph", None)):
        loop.release_graph(clear_routing=True)
    if lsa is not None:
        lsa.aux_loss()  # discard any stray pending KL buffers; never folded into LM CE
        lsa.clear_cache()


def _text_model_forward_kwargs(text_model: nn.Module) -> set[str]:
    try:
        return set(inspect.signature(text_model.forward).parameters.keys())
    except (TypeError, ValueError):
        return set()


def _prepare_batch_tensors(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
    lbl = torch.as_tensor(batch["labels"], dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if lbl.ndim == 1:
        lbl = lbl.unsqueeze(0)
    return ids.to(device=device), lbl.to(device=device)


@torch.no_grad()
def evaluate_pure_lm_nll(
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]],
    *,
    ce_chunk_size: int = 1024,
    device: torch.device | str | None = None,
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
) -> tuple[float, int]:
    """Compute exact valid-token-weighted pure LM cross-entropy (zero aux/KL contamination, `QL60-009/C1`).

    Uses `qloop.train.loss.lm_loss(..., impl="chunked")` over `text_model` hidden states so full
    `[B, T, V]` logits are never materialized.
    """
    dev = _resolve_model_device(model, device)
    was_training = model.training
    model.eval()

    text_model: nn.Module | None = None
    lm_head: nn.Module | None = None
    try:
        text_model = find_text_model(model)
        tcfg = getattr(text_model, "config", None)
        lm_head = find_lm_head(model, getattr(tcfg, "vocab_size", None))
    except RuntimeError:
        text_model = None
        lm_head = None

    total_nll_sum = 0.0
    total_valid = 0
    try:
        for batch in batches:
            ids, lbl = _prepare_batch_tensors(batch, dev)
            if ids.shape[1] < 2:
                continue
            if text_model is not None and lm_head is not None:
                out = text_model(input_ids=ids, use_cache=False)
                hidden = out.last_hidden_state
                loss_mean, n_tok = lm_loss(
                    hidden,
                    lm_head,
                    lbl,
                    impl="chunked",
                    chunk_size=int(ce_chunk_size),
                    shift=True,
                    z_loss=0.0,
                    ignore_index=IGNORE,
                )
                n_val = int(n_tok.item())
                if n_val > 0:
                    total_nll_sum += float(loss_mean.item()) * float(n_val)
                    total_valid += n_val
            else:
                out = model(input_ids=ids, use_cache=False)
                logits = out.logits
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
                shift_labels = lbl[:, 1:].reshape(-1)
                valid_mask = shift_labels != IGNORE
                n_val = int(valid_mask.sum().item())
                if n_val > 0:
                    ce_sum = F.cross_entropy(
                        shift_logits,
                        shift_labels,
                        ignore_index=IGNORE,
                        reduction="sum",
                    )
                    total_nll_sum += float(ce_sum.item())
                    total_valid += n_val
            _cleanup_eval_side_state(loop=loop, lsa=lsa)
    finally:
        _cleanup_eval_side_state(loop=loop, lsa=lsa)
        model.train(was_training)

    mean_nll = total_nll_sum / float(total_valid) if total_valid > 0 else float("nan")
    return mean_nll, total_valid


def _discover_loop_controller(model: nn.Module, loop: LoopController | None = None) -> LoopController | None:
    if loop is not None:
        return loop
    try:
        tm = find_text_model(model)
        layers = getattr(tm, "layers", None)
        if layers is not None and len(layers) > 0:
            ctl = getattr(layers[0], "_ctl", None)
            if isinstance(ctl, LoopController):
                return ctl
    except RuntimeError:
        pass
    return None


@torch.no_grad()
def run_depth_nll_sweep(
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]] | EvalBatchSet,
    *,
    depths: Sequence[int] = (1, 2, 3),
    n_repeats: int = 3,
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
    arm_name: str = "model",
    dataset_name: str = "holdout",
    dataset_sha256: str = "",
    ce_chunk_size: int = 1024,
    run_d1_nondet_check: bool = True,
    run_identity: Mapping[str, Any] | None = None,
) -> DepthSweepResult:
    """Run `A3` multi-repeat pure-LM NLL sweep across depths + explicit `2x d=1` nondeterminism check."""
    if int(n_repeats) < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if isinstance(batches, EvalBatchSet):
        dataset_name = batches.name
        dataset_sha256 = batches.dataset_sha256
        batch_list = batches.batches
    else:
        batch_list = list(batches)
        if not dataset_sha256:
            dataset_sha256 = compute_batch_set_digest(batch_list, split_role=dataset_name)
    if not batch_list:
        raise ValueError("run_depth_nll_sweep requires at least one evaluation batch")

    ctl = _discover_loop_controller(model, loop)
    max_supported = int(ctl.spec.num_passes) if ctl is not None else 1
    req_depths = sorted({int(d) for d in depths if int(d) >= 1})
    valid_depths = [d for d in req_depths if d <= max_supported]
    if not valid_depths:
        valid_depths = [1]
    if len(valid_depths) < len(req_depths):
        skipped = [d for d in req_depths if d > max_supported]
        log.info("[sprint2] %s: skipping depths %s > max_passes=%d", arm_name, skipped, max_supported)

    prev_passes = ctl.active_passes if ctl is not None else None
    per_depth: dict[int, DepthMetrics] = {}
    d1_nondet_nlls: list[float] = []

    try:
        # Explicit 2x d=1 nondeterminism check (`A3`)
        if run_d1_nondet_check:
            if ctl is not None:
                ctl.set_active_passes(1)
            for _ in range(2):
                nll_val, _ = evaluate_pure_lm_nll(
                    model, batch_list, ce_chunk_size=ce_chunk_size, loop=ctl, lsa=lsa
                )
                d1_nondet_nlls.append(float(nll_val))

        for d in valid_depths:
            if ctl is not None:
                ctl.set_active_passes(d)
            t0 = time.perf_counter()
            rep_nlls: list[float] = []
            valid_toks = 0
            for _ in range(int(n_repeats)):
                nll_val, n_tok = evaluate_pure_lm_nll(
                    model, batch_list, ce_chunk_size=ce_chunk_size, loop=ctl, lsa=lsa
                )
                rep_nlls.append(float(nll_val))
                valid_toks = n_tok
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            mean_v = sum(rep_nlls) / float(len(rep_nlls))
            std_v = sample_std(rep_nlls)
            per_depth[d] = DepthMetrics(
                depth=d,
                n_repeats=len(rep_nlls),
                repeat_nlls=rep_nlls,
                mean_nll=mean_v,
                std_nll=std_v,
                ppl=_safe_ppl(mean_v),
                valid_tokens=valid_toks,
                elapsed_ms=elapsed_ms,
            )
    finally:
        if ctl is not None and prev_passes is not None:
            ctl.set_active_passes(prev_passes)
        _cleanup_eval_side_state(loop=ctl, lsa=lsa)

    if not d1_nondet_nlls and 1 in per_depth:
        d1_nondet_nlls = list(per_depth[1].repeat_nlls[:2])

    d1_max_diff = abs(d1_nondet_nlls[0] - d1_nondet_nlls[1]) if len(d1_nondet_nlls) >= 2 else 0.0
    d1_rep_std = sample_std(d1_nondet_nlls)

    d_min = min(per_depth.keys())
    d_max = max(per_depth.keys())
    ref_d1 = per_depth[1] if 1 in per_depth else per_depth[d_min]
    ref_dmax = per_depth[d_max]
    depth_gain = float(ref_d1.mean_nll - ref_dmax.mean_nll)
    pooled = pooled_sample_std(ref_d1.std_nll, ref_d1.n_repeats, ref_dmax.std_nll, ref_dmax.n_repeats)
    z_score, exceeds = compute_z_and_3sigma(depth_gain, pooled)

    return DepthSweepResult(
        arm_name=arm_name,
        dataset_name=dataset_name,
        dataset_sha256=dataset_sha256,
        depths=per_depth,
        n_repeats=int(n_repeats),
        d1_repeat_nlls=d1_nondet_nlls,
        d1_repeat_max_abs_diff=float(d1_max_diff),
        d1_repeat_std=float(d1_rep_std),
        depth_gain_d1_to_dmax=depth_gain,
        pooled_std_d1_to_dmax=pooled,
        z_score=z_score,
        exceeds_3sigma=exceeds,
        d_min=d_min,
        d_max=d_max,
        run_identity=dict(run_identity or {}),
    )


def compare_arms_against_baseline(
    arm_results: Mapping[str, DepthSweepResult],
    baseline_arm: str,
    *,
    eval_depth: int | None = None,
    min_delta: float = 0.0,
) -> list[ArmComparisonRow]:
    """Compare every arm head-to-head against `baseline_arm` with `z_score` and `> 3sigma` gate."""
    if baseline_arm not in arm_results:
        raise KeyError(
            f"baseline_arm {baseline_arm!r} not in arm_results (available: {sorted(arm_results.keys())})"
        )
    base_res = arm_results[baseline_arm]
    base_d = (
        eval_depth
        if (eval_depth is not None and eval_depth in base_res.depths)
        else base_res.d_max
    )
    base_m = base_res.depths[base_d]

    rows: list[ArmComparisonRow] = []
    for arm_name, res in arm_results.items():
        arm_d = eval_depth if (eval_depth is not None and eval_depth in res.depths) else res.d_max
        arm_m = res.depths[arm_d]
        is_base = arm_name == baseline_arm
        delta = 0.0 if is_base else float(base_m.mean_nll - arm_m.mean_nll)
        p_std = pooled_sample_std(arm_m.std_nll, arm_m.n_repeats, base_m.std_nll, base_m.n_repeats)
        z_sc, exceeds = (0.0, False) if is_base else compute_z_and_3sigma(delta, p_std)
        passes = bool((not is_base) and exceeds and delta > float(min_delta))
        ppl_red = 0.0 if is_base else 100.0 * (1.0 - arm_m.ppl / max(base_m.ppl, 1e-12))
        rows.append(
            ArmComparisonRow(
                arm_name=arm_name,
                baseline_arm=baseline_arm,
                arm_depth=arm_d,
                baseline_depth=base_d,
                arm_mean_nll=arm_m.mean_nll,
                arm_std_nll=arm_m.std_nll,
                arm_ppl=arm_m.ppl,
                baseline_mean_nll=base_m.mean_nll,
                baseline_std_nll=base_m.std_nll,
                baseline_ppl=base_m.ppl,
                delta_vs_baseline=delta,
                pooled_std_vs_baseline=p_std,
                z_score_vs_baseline=z_sc,
                exceeds_3sigma=exceeds,
                passes_s2_gate=passes,
                ppl_reduction_pct=ppl_red,
                is_baseline=is_base,
            )
        )
    return rows


# ==============================================================================
# 3. Length-Ladder (8K - 1M), RULER Multi-Needle, D-LSA Gap & GDN-2 State Norms
# ==============================================================================


def _iter_recurrent_state_tensors(cache: Any) -> list[tuple[int, torch.Tensor]]:
    """Extract deduplicated `(slot_idx, tensor)` pairs from `cache.recurrent_states` or `cache.layers[i]`."""
    if cache is None:
        return []
    found: list[tuple[int, torch.Tensor]] = []
    seen_ids: set[int] = set()

    def _collect(slot: int, obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, torch.Tensor):
            if id(obj) not in seen_ids and obj.numel() > 0:
                seen_ids.add(id(obj))
                found.append((slot, obj))
        elif isinstance(obj, Mapping):
            for v in obj.values():
                _collect(slot, v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect(slot, v)
        elif hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            try:
                for idx in range(len(obj)):
                    _collect(slot, obj[idx])
            except Exception:
                pass

    rec_list = getattr(cache, "recurrent_states", None)
    if isinstance(rec_list, (list, tuple)):
        for idx, item in enumerate(rec_list):
            _collect(idx, item)
    elif isinstance(rec_list, Mapping):
        for idx, item in sorted(rec_list.items()):
            _collect(int(idx), item)

    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple, nn.ModuleList)):
        for idx, lyr in enumerate(layers):
            for attr in ("recurrent_states", "recurrent_state"):
                val = getattr(lyr, attr, None)
                if val is not None:
                    _collect(idx, val)
    return found


@torch.no_grad()
def inspect_gdn2_recurrent_state_norms(cache: Any) -> GDN2StateNormTelemetry:
    """Measure L2 norms, max absolute values, and NaN/Inf counts across linear-attention states (`B5`)."""
    states = _iter_recurrent_state_tensors(cache)
    if not states:
        return GDN2StateNormTelemetry(
            num_linear_states=0,
            max_l2_norm=0.0,
            mean_l2_norm=0.0,
            max_abs_val=0.0,
            nan_count=0,
            inf_count=0,
            is_bounded=True,
            per_layer=[],
        )

    per_layer: list[dict[str, Any]] = []
    l2_vals: list[float] = []
    abs_vals: list[float] = []
    total_nans = 0
    total_infs = 0

    for slot, st in states:
        sf = st.detach().float()
        n_nan = int(torch.isnan(sf).sum().item())
        n_inf = int(torch.isinf(sf).sum().item())
        total_nans += n_nan
        total_infs += n_inf
        if n_nan == 0 and n_inf == 0:
            l2 = float(torch.linalg.vector_norm(sf).item())
            m_abs = float(sf.abs().max().item())
        else:
            finite = sf[torch.isfinite(sf)]
            l2 = float("nan") if n_nan > 0 else float("inf")
            m_abs = float(finite.abs().max().item()) if finite.numel() > 0 else float("nan")
        l2_vals.append(l2)
        abs_vals.append(m_abs)
        per_layer.append(
            {
                "slot": int(slot),
                "shape": list(st.shape),
                "dtype": str(st.dtype),
                "l2_norm": l2,
                "max_abs": m_abs,
                "nan_count": n_nan,
                "inf_count": n_inf,
            }
        )

    finite_l2 = [v for v in l2_vals if math.isfinite(v)]
    finite_abs = [v for v in abs_vals if math.isfinite(v)]
    max_l2 = max(finite_l2) if (total_nans == 0 and total_infs == 0 and finite_l2) else float("inf")
    mean_l2 = (sum(finite_l2) / float(len(finite_l2))) if finite_l2 else float("nan")
    max_abs = max(finite_abs) if (total_nans == 0 and total_infs == 0 and finite_abs) else float("inf")
    bounded = bool(
        total_nans == 0 and total_infs == 0 and math.isfinite(max_l2) and math.isfinite(max_abs)
    )
    return GDN2StateNormTelemetry(
        num_linear_states=len(states),
        max_l2_norm=max_l2,
        mean_l2_norm=mean_l2,
        max_abs_val=max_abs,
        nan_count=total_nans,
        inf_count=total_infs,
        is_bounded=bounded,
        per_layer=per_layer,
    )


def _init_streaming_cache(model: nn.Module, kv_cache_quant: str) -> Any:
    """Create a `QuantizedHybridCache` when requested (`fp8`/`hadamard_int4`), else `None` (native)."""
    if kv_cache_quant in ("native", "bf16", "fp16", "fp32", ""):
        return None
    from qloop.infer.engine import QuantizedHybridCache

    tm = find_text_model(model)
    tcfg = getattr(tm, "config", None)
    layers = getattr(tm, "layers", None)
    fallback_n = len(layers) if isinstance(layers, (list, tuple, nn.ModuleList)) else 0
    num_layers = int(getattr(tcfg, "num_hidden_layers", 0) or fallback_n)
    if num_layers <= 0:
        return None
    raw_types = getattr(tcfg, "layer_types", None)
    if raw_types:
        layer_types = [str(lt) for lt in raw_types]
    elif isinstance(layers, (list, tuple, nn.ModuleList)):
        layer_types = [
            str(getattr(lyr, "layer_type", "linear_attention" if hasattr(lyr, "linear_attn") else "full_attention"))
            for lyr in layers
        ]
    else:
        layer_types = ["full_attention"] * num_layers
    return QuantizedHybridCache(num_layers=num_layers, layer_types=layer_types, mode=kv_cache_quant)


def _snapshot_linear_states(cache: Any) -> dict[str, Any]:
    """Save compact clones of `conv_states` & `recurrent_states` (`O(1)` w.r.t. sequence length)."""
    snap: dict[str, Any] = {"direct_conv": None, "direct_rec": None, "layer_states": []}
    if cache is None:
        return snap
    cs = getattr(cache, "conv_states", None)
    if isinstance(cs, list):
        snap["direct_conv"] = [t.detach().clone() if isinstance(t, torch.Tensor) else None for t in cs]
    rs = getattr(cache, "recurrent_states", None)
    if isinstance(rs, list):
        snap["direct_rec"] = [t.detach().clone() if isinstance(t, torch.Tensor) else None for t in rs]
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple, nn.ModuleList)) and snap["direct_conv"] is None and snap["direct_rec"] is None:
        for idx, lyr in enumerate(layers):
            l_cs = getattr(lyr, "conv_states", None)
            l_rs = getattr(lyr, "recurrent_states", None)
            c_copy = None
            r_copy = None
            if isinstance(l_cs, torch.Tensor):
                c_copy = l_cs.detach().clone()
            elif isinstance(l_cs, list):
                c_copy = [t.detach().clone() if isinstance(t, torch.Tensor) else None for t in l_cs]
            if isinstance(l_rs, torch.Tensor):
                r_copy = l_rs.detach().clone()
            elif isinstance(l_rs, list):
                r_copy = [t.detach().clone() if isinstance(t, torch.Tensor) else None for t in l_rs]
            if c_copy is not None or r_copy is not None:
                snap["layer_states"].append((idx, c_copy, r_copy))
    return snap


def _restore_and_crop_cache(cache: Any, prompt_len: int, linear_snap: Mapping[str, Any]) -> bool:
    """Crop KV cache back to `prompt_len` and restore linear conv/recurrent state snapshots."""
    if cache is None:
        return False
    cropped = False
    crop_fn = getattr(cache, "crop", None)
    if callable(crop_fn):
        try:
            crop_fn(int(prompt_len))
            cropped = True
        except Exception:
            cropped = False
    if not cropped:
        kc = getattr(cache, "key_cache", None)
        vc = getattr(cache, "value_cache", None)
        if isinstance(kc, list) and isinstance(vc, list):
            for i in range(len(kc)):
                if isinstance(kc[i], torch.Tensor) and kc[i].shape[-2] > prompt_len:
                    kc[i] = kc[i][..., :prompt_len, :]
                if isinstance(vc[i], torch.Tensor) and vc[i].shape[-2] > prompt_len:
                    vc[i] = vc[i][..., :prompt_len, :]
            cropped = True
        else:
            layers = getattr(cache, "layers", None)
            if isinstance(layers, (list, tuple, nn.ModuleList)):
                any_c = False
                for lyr in layers:
                    lcrop = getattr(lyr, "crop", None)
                    if callable(lcrop):
                        lcrop(int(prompt_len))
                        any_c = True
                    else:
                        k_t = getattr(lyr, "keys", None)
                        v_t = getattr(lyr, "values", None)
                        if isinstance(k_t, torch.Tensor) and k_t.shape[-2] > prompt_len:
                            lyr.keys = k_t[..., :prompt_len, :]
                            any_c = True
                        if isinstance(v_t, torch.Tensor) and v_t.shape[-2] > prompt_len:
                            lyr.values = v_t[..., :prompt_len, :]
                            any_c = True
                cropped = any_c

    if linear_snap.get("direct_conv") is not None and isinstance(getattr(cache, "conv_states", None), list):
        for i, val in enumerate(linear_snap["direct_conv"]):
            if i < len(cache.conv_states):
                cache.conv_states[i] = val.clone() if isinstance(val, torch.Tensor) else None
    if linear_snap.get("direct_rec") is not None and isinstance(getattr(cache, "recurrent_states", None), list):
        for i, val in enumerate(linear_snap["direct_rec"]):
            if i < len(cache.recurrent_states):
                cache.recurrent_states[i] = val.clone() if isinstance(val, torch.Tensor) else None
    for idx, c_copy, r_copy in linear_snap.get("layer_states", ()):
        layers = getattr(cache, "layers", None)
        if isinstance(layers, (list, tuple, nn.ModuleList)) and 0 <= idx < len(layers):
            lyr = layers[idx]
            if c_copy is not None:
                lyr.conv_states = (
                    c_copy.clone()
                    if isinstance(c_copy, torch.Tensor)
                    else [t.clone() if isinstance(t, torch.Tensor) else None for t in c_copy]
                )
            if r_copy is not None:
                lyr.recurrent_states = (
                    r_copy.clone()
                    if isinstance(r_copy, torch.Tensor)
                    else [t.clone() if isinstance(t, torch.Tensor) else None for t in r_copy]
                )
    return cropped


def _reset_lsa_recall_counters(lsa: LSAManager | None) -> dict[str, Any]:
    """Reset per-indexer recall check state so the next sparse forward samples fresh recall (`B2`)."""
    saved: dict[str, Any] = {}
    if lsa is None:
        return saved
    saved["mgr_rc"] = lsa.cfg.recall_check_every
    saved["mgr_last_recall"] = lsa.last_recall
    saved["mgr_recall_ready"] = lsa.recall_ready
    saved["ix"] = {}
    if lsa.cfg.recall_check_every <= 0:
        lsa.cfg.recall_check_every = 1
    lsa.last_recall = 0.0
    for lid, ix in lsa.indexers.items():
        saved["ix"][lid] = (
            ix.cfg.recall_check_every,
            ix._recall_checks,
            ix._last_recall_step,
            ix.last_recall,
            ix.recall_ready,
        )
        if ix.cfg.recall_check_every <= 0:
            ix.cfg.recall_check_every = 1
        ix._recall_checks = 0
        ix._last_recall_step = -1
        ix.last_recall = 0.0
    return saved


def _restore_lsa_recall_counters(lsa: LSAManager | None, saved: Mapping[str, Any]) -> None:
    if lsa is None or not saved:
        return
    lsa.cfg.recall_check_every = int(saved["mgr_rc"])
    lsa.last_recall = float(saved["mgr_last_recall"])
    lsa.recall_ready = bool(saved["mgr_recall_ready"])
    for lid, state in saved.get("ix", {}).items():
        ix = lsa.indexers.get(lid)
        if ix is not None:
            rc, checks, last_step, last_rec, ready = state
            ix.cfg.recall_check_every = int(rc)
            ix._recall_checks = int(checks)
            ix._last_recall_step = int(last_step)
            ix.last_recall = float(last_rec)
            ix.recall_ready = bool(ready)


def _step_text_model_chunk(
    text_model: nn.Module,
    chunk_ids: torch.Tensor,
    cache: Any,
    *,
    start_pos: int,
    cache_kw: str,
    fwd_params: set[str],
) -> tuple[torch.Tensor, Any]:
    kw: dict[str, Any] = {"input_ids": chunk_ids, "use_cache": True}
    if cache is not None:
        kw[cache_kw] = cache
    seq_c = int(chunk_ids.shape[1])
    if "cache_position" in fwd_params:
        kw["cache_position"] = torch.arange(start_pos, start_pos + seq_c, device=chunk_ids.device, dtype=torch.long)
    if "position_ids" in fwd_params:
        kw["position_ids"] = (
            torch.arange(start_pos, start_pos + seq_c, device=chunk_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(chunk_ids.shape[0], -1)
        )
    out = text_model(**kw)
    return out.last_hidden_state, extract_output_cache(out)


@torch.no_grad()
def stream_chunked_lm_nll(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    chunk_size: int = 4096,
    ce_chunk_size: int = 1024,
    kv_cache_quant: str = "native",
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
    sample_lsa_recall: bool = False,
) -> tuple[float, int, Any, float | None, bool]:
    """Stream an arbitrarily long sequence `[B, T]` via chunked prefill without materializing `[B, T, V]` logits.

    Carries the final hidden vector of each chunk across the chunk boundary so every one of the
    `T - 1` next-token predictions is scored identically to full-sequence cross-entropy.
    Returns `(mean_nll, valid_tokens, final_cache, lsa_recall, lsa_sparse_active)`.
    """
    dev = _resolve_model_device(model)
    ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
    lbl = ids if labels is None else (labels.unsqueeze(0) if labels.ndim == 1 else labels)
    ids = ids.to(device=dev, dtype=torch.long)
    lbl = lbl.to(device=dev, dtype=torch.long)
    seq_len = int(ids.shape[1])
    if seq_len < 2:
        raise ValueError(f"stream_chunked_lm_nll requires seq_len >= 2, got {seq_len}")

    c_step = max(2, int(chunk_size))
    was_training = model.training
    model.eval()
    text_model = find_text_model(model)
    lm_head = find_lm_head(model, getattr(getattr(text_model, "config", None), "vocab_size", None))
    if lm_head is None:
        raise RuntimeError("stream_chunked_lm_nll requires a model with an identifiable lm_head")

    cache_kw = cache_kwarg_name(text_model)
    fwd_params = _text_model_forward_kwargs(text_model)
    cache = _init_streaming_cache(model, kv_cache_quant)

    saved_recall = _reset_lsa_recall_counters(lsa) if sample_lsa_recall else {}
    if lsa is not None:
        lsa.clear_cache()

    total_nll_sum = 0.0
    total_valid = 0
    prev_last_hidden: torch.Tensor | None = None
    measured_recall: float | None = None
    sparse_active = False

    try:
        for s0 in range(0, seq_len, c_step):
            s1 = min(seq_len, s0 + c_step)
            is_last_chunk = s1 >= seq_len
            if is_last_chunk and sample_lsa_recall and lsa is not None:
                # Force recall sampling on the final chunk so recall is measured at the full target length `seq_len`!
                for ix in lsa.indexers.values():
                    ix._recall_checks = 0
                    ix._last_recall_step = -1

            h_chunk, cache = _step_text_model_chunk(
                text_model,
                ids[:, s0:s1],
                cache,
                start_pos=s0,
                cache_kw=cache_kw,
                fwd_params=fwd_params,
            )

            if s0 == 0:
                h_eval = h_chunk[:, :-1, :]
                y_eval = lbl[:, 1:s1]
            else:
                assert prev_last_hidden is not None
                h_eval = torch.cat([prev_last_hidden, h_chunk[:, :-1, :]], dim=1)
                y_eval = lbl[:, s0:s1]

            if h_eval.shape[1] > 0:
                loss_c, n_c = lm_loss(
                    h_eval,
                    lm_head,
                    y_eval,
                    impl="chunked",
                    chunk_size=int(ce_chunk_size),
                    shift=False,
                    z_loss=0.0,
                    ignore_index=IGNORE,
                )
                n_val = int(n_c.item())
                if n_val > 0:
                    total_nll_sum += float(loss_c.item()) * float(n_val)
                    total_valid += n_val
            prev_last_hidden = h_chunk[:, -1:, :].detach()

        if sample_lsa_recall and lsa is not None:
            rec_info = lsa.evaluate_recall()
            any_checks = any(ix._recall_checks > 0 for ix in lsa.indexers.values())
            sparse_active = any(
                float(ix.last_stats.get("forward_sparse", 0.0)) > 0.5 for ix in lsa.indexers.values()
            )
            if any_checks:
                measured_recall = float(rec_info["mean_recall"])
            elif not sparse_active:
                measured_recall = 1.0
            else:
                measured_recall = float(rec_info["mean_recall"])
    finally:
        if sample_lsa_recall and lsa is not None:
            _restore_lsa_recall_counters(lsa, saved_recall)
        if lsa is not None:
            lsa.aux_loss()
            lsa.clear_cache()
        if loop is not None and callable(getattr(loop, "release_graph", None)):
            loop.release_graph(clear_routing=True)
        model.train(was_training)

    mean_nll = total_nll_sum / float(total_valid) if total_valid > 0 else float("nan")
    return mean_nll, total_valid, cache, measured_recall, sparse_active


def _even_needle_depths(num_needles: int, needle_depths: Sequence[float] | None) -> list[float]:
    k = max(1, int(num_needles))
    if needle_depths is not None and len(needle_depths) == k:
        return [min(0.95, max(0.05, float(d))) for d in needle_depths]
    if k == 1:
        return [0.5]
    return [0.10 + (0.80 * i) / float(k - 1) for i in range(k)]


def build_ruler_multi_needle_episode(
    target_len: int,
    tokenizer: Any | None = None,
    *,
    num_needles: int = 4,
    needle_depths: Sequence[float] | None = DEFAULT_NEEDLE_DEPTHS,
    seed: int = 0,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Construct a deterministic RULER-style multi-needle key/value retrieval episode of length `~target_len`.

    Injects `K` unique key-value needles at fractional depths `[0.10, 0.35, 0.65, 0.90]` inside
    distractor context. Works with both real HF / `ToyTokenizer` text tokenizers and small-vocab
    token-ID-only toy models (`vocab_size < 128`).
    """
    t_len = max(32, int(target_len))
    depths = _even_needle_depths(num_needles, needle_depths)
    rng = random.Random(int(seed) * 1009 + t_len * 37 + len(depths))

    use_text = tokenizer is not None and (vocab_size is None or int(vocab_size) >= 128)
    if use_text:
        assert tokenizer is not None
        needles: list[dict[str, Any]] = []
        used_vals: set[str] = set()
        for idx, frac in enumerate(depths):
            kw = _RULER_KEY_WORDS[idx % len(_RULER_KEY_WORDS)]
            key_str = f"KEY_{idx + 1:02d}_{kw}"
            val_str = f"{rng.randint(1000, 9999)}"
            while val_str in used_vals:
                val_str = f"{rng.randint(1000, 9999)}"
            used_vals.add(val_str)
            fact_str = f" [RECORD {key_str}={val_str}] "
            needles.append(
                {
                    "index": idx,
                    "depth": frac,
                    "key": key_str,
                    "value": val_str,
                    "fact_text": fact_str,
                    "fact_ids": list(tokenizer.encode(fact_str, add_special_tokens=False)),
                }
            )

        q_keys = ", ".join(n["key"] for n in needles)
        query_text = f"\nQuestion: Recall the exact values for {q_keys}.\nAnswer: "
        answer_text = "; ".join(f"{n['key']}={n['value']}" for n in needles)
        query_ids = list(tokenizer.encode(query_text, add_special_tokens=False))
        answer_ids = list(tokenizer.encode(answer_text, add_special_tokens=False))
        if not answer_ids:
            answer_ids = [int(getattr(tokenizer, "eos_token_id", 1) or 1)]

        needle_total = sum(len(n["fact_ids"]) for n in needles)
        avail_filler = max(0, t_len - len(query_ids) - needle_total)

        filler_pool: list[int] = []
        sent_idx = 0
        while len(filler_pool) < avail_filler + 32:
            sent = _RULER_FILLER_SENTENCES[(sent_idx + int(seed)) % len(_RULER_FILLER_SENTENCES)] + " "
            filler_pool.extend(tokenizer.encode(sent, add_special_tokens=False))
            sent_idx += 1
        filler_ids = filler_pool[:avail_filler]

        # Splice needles at exact fractional depths into the filler stream
        prompt_ids: list[int] = []
        prev_cut = 0
        for nd in needles:
            cut = min(avail_filler, max(prev_cut, round(nd["depth"] * avail_filler)))
            prompt_ids.extend(filler_ids[prev_cut:cut])
            nd["token_offset"] = len(prompt_ids)
            prompt_ids.extend(nd["fact_ids"])
            prev_cut = cut
        prompt_ids.extend(filler_ids[prev_cut:])
        prompt_ids.extend(query_ids)
        return {
            "mode": "text",
            "target_length": t_len,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "answer_text": answer_text,
            "needles": needles,
            "needle_depths": depths,
        }

    # Small-vocab pure token-ID RULER episode (for minimal ToyConfig without full ASCII vocab)
    vsz = max(16, int(vocab_size or 97))
    needles_tok: list[dict[str, Any]] = []
    # Reserve token IDs: 1 = SEP, 2 = QUERY_TAG, 3..10 = keys, 11..vsz-1 = values & distractors
    key_base = 3
    val_lo = min(vsz - 4, key_base + len(depths) + 2)
    filler_lo = max(2, min(val_lo, 4))
    for idx, frac in enumerate(depths):
        k_tok = (key_base + idx) % max(2, vsz - 1) + 1
        v_tok = rng.randint(val_lo, vsz - 1) if val_lo < vsz else (k_tok % max(1, vsz - 1)) + 1
        needles_tok.append(
            {
                "index": idx,
                "depth": frac,
                "key": f"TOK_{k_tok}",
                "value": str(v_tok),
                "fact_ids": [1, k_tok, v_tok, 1],
                "value_token_id": v_tok,
            }
        )
    query_ids = [2] + [int(n["fact_ids"][1]) for n in needles_tok] + [2]
    answer_ids = [int(n["value_token_id"]) for n in needles_tok]
    needle_total = sum(len(n["fact_ids"]) for n in needles_tok)
    avail_filler = max(0, t_len - len(query_ids) - needle_total)
    filler_ids = [rng.randint(filler_lo, vsz - 1) for _ in range(avail_filler)]

    prompt_ids = []
    prev_cut = 0
    for nd in needles_tok:
        cut = min(avail_filler, max(prev_cut, round(nd["depth"] * avail_filler)))
        prompt_ids.extend(filler_ids[prev_cut:cut])
        nd["token_offset"] = len(prompt_ids)
        prompt_ids.extend(nd["fact_ids"])
        prev_cut = cut
    prompt_ids.extend(filler_ids[prev_cut:])
    prompt_ids.extend(query_ids)
    return {
        "mode": "token_ids",
        "target_length": t_len,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "answer_text": " ".join(str(x) for x in answer_ids),
        "needles": needles_tok,
        "needle_depths": depths,
    }


@torch.no_grad()
def evaluate_ruler_multi_needle(
    model: nn.Module,
    episode: Mapping[str, Any],
    *,
    tokenizer: Any | None = None,
    chunk_size: int = 4096,
    ce_chunk_size: int = 1024,
    kv_cache_quant: str = "native",
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
) -> tuple[RulerMultiNeedleResult, GDN2StateNormTelemetry]:
    """Stream a RULER multi-needle episode once, measuring greedy exact-match, answer NLL, and GDN-2 state norms."""
    t0 = time.perf_counter()
    dev = _resolve_model_device(model)
    was_training = model.training
    model.eval()
    text_model = find_text_model(model)
    lm_head = find_lm_head(model, getattr(getattr(text_model, "config", None), "vocab_size", None))
    if lm_head is None:
        raise RuntimeError("evaluate_ruler_multi_needle requires a model with an identifiable lm_head")

    prompt_ids_list = [int(x) for x in episode["prompt_ids"]]
    answer_ids_list = [int(x) for x in episode["answer_ids"]]
    prompt_len = len(prompt_ids_list)
    ans_len = len(answer_ids_list)
    prompt_t = torch.tensor([prompt_ids_list], dtype=torch.long, device=dev)
    ans_t = torch.tensor([answer_ids_list], dtype=torch.long, device=dev)

    c_step = max(2, int(chunk_size))
    cache_kw = cache_kwarg_name(text_model)
    fwd_params = _text_model_forward_kwargs(text_model)
    cache = _init_streaming_cache(model, kv_cache_quant)
    if lsa is not None:
        lsa.clear_cache()

    decode_mode = "autoregressive_greedy"
    def _lm_head_logits(h_in: torch.Tensor) -> torch.Tensor:
        w = getattr(lm_head, "weight", None)
        if isinstance(w, torch.Tensor) and w.is_floating_point() and h_in.dtype != w.dtype:
            return lm_head(h_in.to(w.dtype))
        return lm_head(h_in)

    try:
        # 1. Stream prompt chunks once
        last_prompt_h: torch.Tensor | None = None
        for s0 in range(0, prompt_len, c_step):
            s1 = min(prompt_len, s0 + c_step)
            h_c, cache = _step_text_model_chunk(
                text_model,
                prompt_t[:, s0:s1],
                cache,
                start_pos=s0,
                cache_kw=cache_kw,
                fwd_params=fwd_params,
            )
            last_prompt_h = h_c[:, -1:, :].detach()
        assert last_prompt_h is not None

        # 2. Capture GDN-2 recurrent state norms at exact prompt horizon
        state_norms = inspect_gdn2_recurrent_state_norms(cache)
        linear_snap = _snapshot_linear_states(cache)

        # 3. Autoregressive greedy generation for `ans_len` tokens
        gen_ids: list[int] = []
        cur_h = last_prompt_h
        cur_pos = prompt_len
        for step_i in range(ans_len):
            logits_1 = _lm_head_logits(cur_h[:, -1:, :])
            nxt_tok = int(logits_1.argmax(dim=-1).item())
            gen_ids.append(nxt_tok)
            if step_i + 1 < ans_len:
                nxt_t = torch.tensor([[nxt_tok]], dtype=torch.long, device=dev)
                cur_h, cache = _step_text_model_chunk(
                    text_model,
                    nxt_t,
                    cache,
                    start_pos=cur_pos,
                    cache_kw=cache_kw,
                    fwd_params=fwd_params,
                )
                cur_pos += 1

        # 4. Crop KV cache + restore linear states for exact teacher-forced answer NLL & token accuracy
        restored_ok = _restore_and_crop_cache(cache, prompt_len, linear_snap)
        if lsa is not None:
            lsa.clear_cache()

        if restored_ok:
            if ans_len > 1:
                h_ans, _ = _step_text_model_chunk(
                    text_model,
                    ans_t[:, :-1],
                    cache,
                    start_pos=prompt_len,
                    cache_kw=cache_kw,
                    fwd_params=fwd_params,
                )
                h_for_ans = torch.cat([last_prompt_h, h_ans], dim=1)
            else:
                h_for_ans = last_prompt_h
            ans_loss, _ = lm_loss(
                h_for_ans,
                lm_head,
                ans_t,
                impl="chunked",
                chunk_size=max(1, min(int(ce_chunk_size), ans_len)),
                shift=False,
                z_loss=0.0,
                ignore_index=IGNORE,
            )
            answer_nll = float(ans_loss.item())
            # Token-level teacher-forced accuracy (tiny `[1, ans_len, V]` only over the short answer)
            ans_preds = _lm_head_logits(h_for_ans).argmax(dim=-1).squeeze(0).tolist()
            answer_tok_acc = sum(int(p == g) for p, g in zip(ans_preds, answer_ids_list)) / float(max(1, ans_len))
        else:
            decode_mode = "autoregressive_greedy_no_crop"
            logits_first = _lm_head_logits(last_prompt_h).squeeze(0).float()
            answer_nll = float(F.cross_entropy(logits_first, ans_t[:, 0]).item())
            answer_tok_acc = float(gen_ids[0] == answer_ids_list[0]) if gen_ids else 0.0
    finally:
        if lsa is not None:
            lsa.aux_loss()
            lsa.clear_cache()
        if loop is not None and callable(getattr(loop, "release_graph", None)):
            loop.release_graph(clear_routing=True)
        model.train(was_training)

    needles = list(episode["needles"])
    if episode.get("mode") == "text" and tokenizer is not None:
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        hits = [str(nd["value"]) in pred_text for nd in needles]
    else:
        hits = [
            (idx < len(gen_ids) and int(gen_ids[idx]) == int(nd["value_token_id"]))
            for idx, nd in enumerate(needles)
        ]

    n_hits = sum(1 for h in hits if h)
    em_acc = float(n_hits) / float(max(1, len(needles)))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ruler_res = RulerMultiNeedleResult(
        target_length=int(episode["target_length"]),
        actual_prompt_tokens=prompt_len,
        num_needles=len(needles),
        needle_depths=[float(d) for d in episode["needle_depths"]],
        exact_match_acc=em_acc,
        needles_matched=n_hits,
        answer_token_acc=float(answer_tok_acc),
        answer_nll=float(answer_nll),
        per_needle_hits=hits,
        decode_mode=decode_mode,
        elapsed_ms=elapsed_ms,
    )
    return ruler_res, state_norms


def _build_length_probe_tokens(
    target_len: int,
    *,
    tokenizer: Any | None,
    source_tokens: torch.Tensor | None,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    """Build deterministic `[1, target_len]` token IDs for chunked NLL length-ladder probing."""
    t_len = max(4, int(target_len))
    if source_tokens is not None and source_tokens.numel() >= 2:
        flat = source_tokens.detach().reshape(-1).to(dtype=torch.long, device="cpu")
        if flat.numel() >= t_len:
            return flat[:t_len].unsqueeze(0)
        reps = (t_len + int(flat.numel()) - 1) // int(flat.numel())
        return flat.repeat(reps)[:t_len].unsqueeze(0)
    if tokenizer is not None and vocab_size >= 128:
        ids: list[int] = []
        idx = 0
        while len(ids) < t_len:
            sent = _RULER_FILLER_SENTENCES[(idx + int(seed)) % len(_RULER_FILLER_SENTENCES)] + " "
            ids.extend(tokenizer.encode(sent, add_special_tokens=False))
            idx += 1
        return torch.tensor([ids[:t_len]], dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed) + t_len)
    return torch.randint(0, max(2, int(vocab_size)), (1, t_len), generator=gen, dtype=torch.long)


@torch.no_grad()
def run_length_ladder_audit(
    model: nn.Module,
    *,
    lengths: Sequence[int] = DEFAULT_GPU_LENGTHS,
    tokenizer: Any | None = None,
    source_tokens: torch.Tensor | None = None,
    chunk_size: int = 4096,
    ce_chunk_size: int = 1024,
    kv_cache_quant: str = "native",
    depth: int | None = None,
    loop: LoopController | None = None,
    lsa: LSAManager | None = None,
    num_needles: int = 4,
    needle_depths: Sequence[float] | None = DEFAULT_NEEDLE_DEPTHS,
    seed: int = 0,
    arm_name: str = "model",
) -> LengthLadderResult:
    """Run `B2 / B3 / B4 / B5` length-ladder evaluation (chunked NLL, dense vs sparse D-LSA, RULER, GDN-2 norms)."""
    ctl = _discover_loop_controller(model, loop)
    prev_passes = ctl.active_passes if ctl is not None else None
    eff_depth = 1
    if ctl is not None:
        eff_depth = int(ctl.spec.num_passes if depth is None else min(int(depth), ctl.spec.num_passes))
        ctl.set_active_passes(eff_depth)

    tm = find_text_model(model)
    vocab_size = int(getattr(getattr(tm, "config", None), "vocab_size", 0) or tm.embed_tokens.weight.shape[0])

    points: list[LengthLadderPoint] = []
    try:
        for raw_len in lengths:
            t_len = int(raw_len)
            t0 = time.perf_counter()
            eff_chunk = max(2, min(int(chunk_size), t_len))
            seq_ids = _build_length_probe_tokens(
                t_len,
                tokenizer=tokenizer,
                source_tokens=source_tokens,
                vocab_size=vocab_size,
                seed=seed,
            )

            # (a) Dense chunked NLL + GDN-2 state norms
            dense_ctx = lsa.temporary_mode("dense") if lsa is not None else contextlib.nullcontext()
            with dense_ctx:
                dense_nll, valid_toks, dense_cache, _, _ = stream_chunked_lm_nll(
                    model,
                    seq_ids,
                    seq_ids,
                    chunk_size=eff_chunk,
                    ce_chunk_size=ce_chunk_size,
                    kv_cache_quant=kv_cache_quant,
                    loop=ctl,
                    lsa=lsa,
                    sample_lsa_recall=False,
                )
                dense_norms = inspect_gdn2_recurrent_state_norms(dense_cache)
            del dense_cache

            # (b) Sparse D-LSA chunked NLL + live `lsa/last_recall` (`B2`)
            sparse_nll: float | None = None
            sparse_ppl: float | None = None
            ds_gap: float | None = None
            lsa_recall: float | None = None
            sparse_active = False
            sparse_norms: GDN2StateNormTelemetry | None = None
            if lsa is not None:
                with lsa.temporary_mode("sparse"):
                    s_nll, _, sparse_cache, rec_val, s_act = stream_chunked_lm_nll(
                        model,
                        seq_ids,
                        seq_ids,
                        chunk_size=eff_chunk,
                        ce_chunk_size=ce_chunk_size,
                        kv_cache_quant=kv_cache_quant,
                        loop=ctl,
                        lsa=lsa,
                        sample_lsa_recall=True,
                    )
                    sparse_norms = inspect_gdn2_recurrent_state_norms(sparse_cache)
                del sparse_cache
                sparse_nll = float(s_nll)
                sparse_ppl = _safe_ppl(sparse_nll)
                ds_gap = float(sparse_nll - dense_nll)
                lsa_recall = rec_val
                sparse_active = bool(s_act)

            # (c) Synthetic multi-needle RULER retrieval + GDN-2 state norm telemetry (`B3` / `B5`)
            episode = build_ruler_multi_needle_episode(
                t_len,
                tokenizer=tokenizer,
                num_needles=num_needles,
                needle_depths=needle_depths,
                seed=seed,
                vocab_size=vocab_size,
            )
            ruler_res, ruler_norms = evaluate_ruler_multi_needle(
                model,
                episode,
                tokenizer=tokenizer,
                chunk_size=eff_chunk,
                ce_chunk_size=ce_chunk_size,
                kv_cache_quant=kv_cache_quant,
                loop=ctl,
                lsa=lsa,
            )

            # Prefer the highest-norm telemetry snapshot across dense/sparse/RULER for strict boundedness check
            chosen_norms = sparse_norms if (sparse_norms is not None and sparse_norms.num_linear_states > 0) else (
                dense_norms if dense_norms.num_linear_states > 0 else ruler_norms
            )
            if (
                ruler_norms.num_linear_states > 0
                and chosen_norms.num_linear_states > 0
                and (not ruler_norms.is_bounded or ruler_norms.max_l2_norm > chosen_norms.max_l2_norm)
            ):
                chosen_norms = ruler_norms

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            points.append(
                LengthLadderPoint(
                    length=t_len,
                    chunk_size=eff_chunk,
                    dense_nll=float(dense_nll),
                    dense_ppl=_safe_ppl(dense_nll),
                    valid_tokens=int(valid_toks),
                    sparse_nll=sparse_nll,
                    sparse_ppl=sparse_ppl,
                    dense_sparse_nll_gap=ds_gap,
                    lsa_last_recall=lsa_recall,
                    lsa_sparse_active=sparse_active,
                    ruler=ruler_res,
                    gdn2_state_norms=chosen_norms,
                    elapsed_ms=elapsed_ms,
                )
            )
    finally:
        if ctl is not None and prev_passes is not None:
            ctl.set_active_passes(prev_passes)
        _cleanup_eval_side_state(loop=ctl, lsa=lsa)

    all_bounded = all(
        (p.gdn2_state_norms is None or p.gdn2_state_norms.is_bounded) for p in points
    )
    gaps = [p.dense_sparse_nll_gap for p in points if p.dense_sparse_nll_gap is not None]
    recalls = [p.lsa_last_recall for p in points if p.lsa_last_recall is not None]
    return LengthLadderResult(
        arm_name=arm_name,
        depth=eff_depth,
        kv_cache_quant=kv_cache_quant,
        points=points,
        all_states_bounded=all_bounded,
        max_dense_sparse_nll_gap=max(gaps) if gaps else None,
        min_lsa_recall=min(recalls) if recalls else None,
        metadata={"lengths": [int(x) for x in lengths], "num_needles": int(num_needles)},
    )


# ==============================================================================
# 4. S2.3 Zero-Training Reversible Causal Ablation Context Managers & Matrix
# ==============================================================================


def _find_gdn2_adapters(model: nn.Module, gdn2: GDN2Manager | None = None) -> dict[int, GDN2Adapter]:
    found: dict[int, GDN2Adapter] = {}
    seen_ids: set[int] = set()
    if gdn2 is not None:
        for lid, ad in gdn2.adapters.items():
            found[int(lid)] = ad
            seen_ids.add(id(ad))
    for mod in model.modules():
        if isinstance(mod, GDN2Adapter) and id(mod) not in seen_ids:
            seen_ids.add(id(mod))
            found[int(mod.layer_id)] = mod
    return found


def _find_pass_routers(model: nn.Module, loop: LoopController | None = None) -> list[PassRouterAdapters]:
    found: list[PassRouterAdapters] = []
    seen_ids: set[int] = set()
    if loop is not None and loop.adapters.pass_router is not None:
        pr = loop.adapters.pass_router
        found.append(pr)
        seen_ids.add(id(pr))
    for mod in model.modules():
        if isinstance(mod, PassRouterAdapters) and id(mod) not in seen_ids:
            seen_ids.add(id(mod))
            found.append(mod)
    return found


def _find_loop_bridges(model: nn.Module, loop: LoopController | None = None) -> list[LoopBridge]:
    found: list[LoopBridge] = []
    seen_ids: set[int] = set()
    if loop is not None and isinstance(loop.adapters.bridge, LoopBridge):
        br = loop.adapters.bridge
        found.append(br)
        seen_ids.add(id(br))
    for mod in model.modules():
        if isinstance(mod, LoopBridge) and id(mod) not in seen_ids:
            seen_ids.add(id(mod))
            found.append(mod)
    return found


def _find_lsa_indexers(model: nn.Module, lsa: LSAManager | None = None) -> dict[int, LSAIndexer]:
    found: dict[int, LSAIndexer] = {}
    seen_ids: set[int] = set()
    if lsa is not None:
        for lid, ix in lsa.indexers.items():
            found[int(lid)] = ix
            seen_ids.add(id(ix))
    for mod in model.modules():
        if isinstance(mod, LSAIndexer) and id(mod) not in seen_ids:
            seen_ids.add(id(mod))
            found[int(mod.layer_id)] = mod
    return found


@contextlib.contextmanager
def ablate_full(model: nn.Module) -> Iterator[dict[str, Any]]:
    """Reference baseline context manager (no ablation applied)."""
    del model
    yield {"applicable": True, "ablation": "full"}


@contextlib.contextmanager
def ablate_no_gdn2(
    model: nn.Module,
    *,
    gdn2: GDN2Manager | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily disable/bypass all `GDN2Adapter` residuals cleanly, restoring bit-exact weights on exit."""
    adapters = _find_gdn2_adapters(model, gdn2=gdn2)
    saved_tensors: list[tuple[nn.Parameter, torch.Tensor]] = []
    saved_stats: list[tuple[GDN2Adapter, dict[str, float]]] = []
    with torch.no_grad():
        for ad in adapters.values():
            saved_stats.append((ad, dict(ad._last_stats)))
            for p in ad._zero_check_params():
                saved_tensors.append((p, p.detach().clone()))
                p.zero_()
            ad._is_zero_cache = None
    try:
        yield {"applicable": len(adapters) > 0, "num_gdn2_adapters": len(adapters)}
    finally:
        with torch.no_grad():
            for p, orig in saved_tensors:
                p.copy_(orig)
            for ad, st in saved_stats:
                ad._is_zero_cache = None
                ad._pending_stat_tensors.clear()
                ad._last_stats = st


@contextlib.contextmanager
def ablate_no_qed(
    model: nn.Module,
    *,
    gdn2: GDN2Manager | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily zero `qed_lambda_raw` across all `GDN2Adapter`s while keeping base GDN-2 gates active."""
    adapters = _find_gdn2_adapters(model, gdn2=gdn2)
    saved_qed: list[tuple[GDN2Adapter, torch.Tensor]] = []
    with torch.no_grad():
        for ad in adapters.values():
            if ad.qed_lambda_raw is not None:
                saved_qed.append((ad, ad.qed_lambda_raw.detach().clone()))
                ad.qed_lambda_raw.zero_()
                ad._is_zero_cache = None
    try:
        yield {"applicable": len(saved_qed) > 0, "num_qed_adapters": len(saved_qed)}
    finally:
        with torch.no_grad():
            for ad, orig in saved_qed:
                assert ad.qed_lambda_raw is not None
                ad.qed_lambda_raw.copy_(orig)
                ad._is_zero_cache = None


@contextlib.contextmanager
def ablate_no_neg_eigval(
    model: nn.Module,
    *,
    gdn2: GDN2Manager | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily clamp `neg_eigval_max_boost` and `neg_gamma_raw` to `0.0` so eigenvalues stay in `[0, 1]`."""
    adapters = _find_gdn2_adapters(model, gdn2=gdn2)
    saved_raw: list[tuple[GDN2Adapter, torch.Tensor]] = []
    saved_cfgs: dict[int, tuple[GDN2Config, float]] = {}
    with torch.no_grad():
        for ad in adapters.values():
            if ad.neg_gamma_raw is not None:
                saved_raw.append((ad, ad.neg_gamma_raw.detach().clone()))
                ad.neg_gamma_raw.zero_()
                ad._is_zero_cache = None
            if id(ad.cfg) not in saved_cfgs:
                saved_cfgs[id(ad.cfg)] = (ad.cfg, float(ad.cfg.neg_eigval_max_boost))
                ad.cfg.neg_eigval_max_boost = 0.0
    try:
        yield {
            "applicable": len(saved_raw) > 0,
            "num_neg_eigval_adapters": len(saved_raw),
        }
    finally:
        with torch.no_grad():
            for cfg_obj, orig_boost in saved_cfgs.values():
                cfg_obj.neg_eigval_max_boost = orig_boost
            for ad, orig in saved_raw:
                assert ad.neg_gamma_raw is not None
                ad.neg_gamma_raw.copy_(orig)
                ad._is_zero_cache = None


@contextlib.contextmanager
def ablate_no_pass_router(
    model: nn.Module,
    *,
    loop: LoopController | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily zero `PassRouterAdapters` projection deltas, biases, and `repel` on `d >= 2`."""
    routers = _find_pass_routers(model, loop=loop)
    saved: list[tuple[nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for pr in routers:
            w_param = pr.up if pr.rank > 0 else pr.weight
            for p in (w_param, pr.bias, pr.repel):
                saved.append((p, p.detach().clone()))
                p.zero_()
    try:
        yield {"applicable": len(routers) > 0, "num_pass_routers": len(routers)}
    finally:
        with torch.no_grad():
            for p, orig in saved:
                p.copy_(orig)


@contextlib.contextmanager
def ablate_low_anchor_probe(
    model: nn.Module,
    *,
    loop: LoopController | None = None,
    max_anchor_weight: float = 0.25,
) -> Iterator[dict[str, Any]]:
    """Temporarily shift `LoopBridge`/`ReEntryBridge` simplex weights so `w_anchor <= max_anchor_weight <= 0.3`.

    Preserves the relative ratio between `w_h_prev` (index 0) and `w_rms` (index 2) while
    renormalizing `[w_h_prev, w_anchor, w_rms]` onto the 2-simplex (McLeish et al. arXiv:2511.07384),
    and restores exact original `mix_logits` in `finally:`.
    """
    cap = float(max_anchor_weight)
    if not (0.0 < cap <= 0.3):
        raise ValueError(f"max_anchor_weight must be in (0, 0.3], got {max_anchor_weight}")
    bridges = _find_loop_bridges(model, loop=loop)
    saved: list[tuple[LoopBridge, torch.Tensor]] = []
    before_weights: list[list[float]] = []
    after_weights: list[list[float]] = []

    with torch.no_grad():
        for br in bridges:
            saved.append((br, br.mix_logits.detach().clone()))
            new_logits = br.mix_logits.detach().clone()
            for p_idx in range(br.n_extra_passes):
                w = br.weights(p_idx).detach().to(torch.float64)
                w0, w1, w2 = float(w[0].item()), float(w[1].item()), float(w[2].item())
                before_weights.append([w0, w1, w2])
                if w1 > cap:
                    new_w1 = cap
                    rem = 1.0 - new_w1
                    old_rem = w0 + w2
                    if old_rem > 1e-12:
                        new_w0 = rem * (w0 / old_rem)
                        new_w2 = rem * (w2 / old_rem)
                    else:
                        new_w0 = 0.5 * rem
                        new_w2 = 0.5 * rem
                    eps = 1e-12
                    vals = [max(new_w0, eps), max(new_w1, eps), max(new_w2, eps)]
                    s = sum(vals)
                    vals = [v / s for v in vals]
                    new_logits[p_idx] = torch.tensor(
                        [math.log(v) for v in vals],
                        device=br.mix_logits.device,
                        dtype=br.mix_logits.dtype,
                    )
            br.mix_logits.copy_(new_logits)
            for p_idx in range(br.n_extra_passes):
                wa = br.weights(p_idx).detach().to(torch.float64)
                after_weights.append([float(wa[0].item()), float(wa[1].item()), float(wa[2].item())])

    try:
        yield {
            "applicable": len(bridges) > 0,
            "num_bridges": len(bridges),
            "max_anchor_weight": cap,
            "weights_before": before_weights,
            "weights_after": after_weights,
        }
    finally:
        with torch.no_grad():
            for br, orig in saved:
                br.mix_logits.copy_(orig)


@contextlib.contextmanager
def ablate_lsa_mode(
    model: nn.Module,
    mode: str,
    *,
    lsa: LSAManager | None = None,
    sample_recall: bool = False,
) -> Iterator[dict[str, Any]]:
    """Temporarily toggle `LSAManager` / `LSAIndexer` mode (`'dense'` vs `'sparse'`) and restore on exit."""
    if mode not in ("dense", "sparse"):
        raise ValueError(f"ablate_lsa_mode expects 'dense' or 'sparse', got {mode!r}")
    indexers = _find_lsa_indexers(model, lsa=lsa)
    saved_mgr_mode = lsa.cfg.mode if lsa is not None else None
    saved_ix_modes = {lid: ix.cfg.mode for lid, ix in indexers.items()}
    saved_recall = _reset_lsa_recall_counters(lsa) if (sample_recall and mode == "sparse") else {}

    if lsa is not None:
        lsa.clear_cache()
        lsa.set_mode(mode)
    else:
        for ix in indexers.values():
            ix._cache.clear()
            ix.cfg.mode = mode

    info: dict[str, Any] = {
        "applicable": len(indexers) > 0,
        "mode": mode,
        "num_lsa_indexers": len(indexers),
    }
    try:
        yield info
    finally:
        if sample_recall and mode == "sparse" and lsa is not None:
            rec = lsa.evaluate_recall()
            info["lsa_last_recall"] = float(rec["mean_recall"])
            _restore_lsa_recall_counters(lsa, saved_recall)
        if lsa is not None:
            lsa.clear_cache()
            if saved_mgr_mode is not None:
                lsa.cfg.mode = saved_mgr_mode
        for lid, orig_m in saved_ix_modes.items():
            indexers[lid].cfg.mode = orig_m
            indexers[lid]._cache.clear()


@contextlib.contextmanager
def causal_ablation_context(
    model: nn.Module,
    ablation: str,
    *,
    loop: LoopController | None = None,
    gdn2: GDN2Manager | None = None,
    lsa: LSAManager | None = None,
    max_anchor_weight: float = 0.25,
) -> Iterator[dict[str, Any]]:
    """Unified reversible context manager dispatching all `S2.3` zero-training causal ablations."""
    key = str(ablation).strip().lower()
    if key == "full":
        with ablate_full(model) as info:
            yield info
    elif key == "no_gdn2":
        with ablate_no_gdn2(model, gdn2=gdn2) as info:
            yield info
    elif key == "no_qed":
        with ablate_no_qed(model, gdn2=gdn2) as info:
            yield info
    elif key == "no_neg_eigval":
        with ablate_no_neg_eigval(model, gdn2=gdn2) as info:
            yield info
    elif key == "no_pass_router":
        with ablate_no_pass_router(model, loop=loop) as info:
            yield info
    elif key == "low_anchor_probe":
        with ablate_low_anchor_probe(model, loop=loop, max_anchor_weight=max_anchor_weight) as info:
            yield info
    elif key == "lsa_dense":
        with ablate_lsa_mode(model, "dense", lsa=lsa, sample_recall=False) as info:
            yield info
    elif key == "lsa_sparse":
        with ablate_lsa_mode(model, "sparse", lsa=lsa, sample_recall=True) as info:
            yield info
    else:
        raise ValueError(f"unknown S2.3 causal ablation {ablation!r}; expected one of {CANONICAL_ABLATIONS}")


def _recommend_ablation_tag(
    ablation: str,
    delta_nll_vs_full: float,
    exceeds_3sigma: bool,
    *,
    min_keep_delta: float,
    lsa_last_recall: float | None,
    recall_threshold: float,
    max_sparse_nll_gap: float,
    dense_nll_ref: float | None,
    this_mean_nll: float,
) -> str:
    if ablation == "full":
        return "KEEP_CRITICAL"
    if ablation in ("low_anchor_probe", "lsa_dense"):
        return "SHORTCUT_DIAGNOSTIC"
    if ablation == "lsa_sparse":
        gap = (this_mean_nll - dense_nll_ref) if dense_nll_ref is not None else delta_nll_vs_full
        recall_ok = lsa_last_recall is None or float(lsa_last_recall) >= float(recall_threshold)
        if recall_ok and gap <= float(max_sparse_nll_gap):
            return "KEEP_CRITICAL"
        return "PRUNE_CANDIDATE"
    # Removal ablations (`no_gdn2`, `no_qed`, `no_neg_eigval`, `no_pass_router`):
    # Positive `delta_nll_vs_full` means removing the component degraded LM NLL -> component helps.
    if exceeds_3sigma and float(delta_nll_vs_full) > float(min_keep_delta):
        return "KEEP_CRITICAL"
    return "PRUNE_CANDIDATE"


@torch.no_grad()
def run_causal_ablation_matrix(
    model: nn.Module,
    batches: Sequence[Mapping[str, Any]] | EvalBatchSet,
    *,
    ablations: Sequence[str] = CANONICAL_ABLATIONS,
    n_repeats: int = 3,
    depth: int | None = None,
    loop: LoopController | None = None,
    gdn2: GDN2Manager | None = None,
    lsa: LSAManager | None = None,
    max_anchor_weight: float = 0.25,
    min_keep_delta: float = 1e-4,
    recall_threshold: float = 0.70,
    max_sparse_nll_gap: float = 0.05,
    ce_chunk_size: int = 1024,
    verify_sha256: bool = True,
    arm_name: str = "model",
) -> CausalAblationMatrixResult:
    """Execute the `S2.3` reversible causal ablation matrix with bit-exact SHA-256 restoration checks."""
    if int(n_repeats) < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    batch_list = batches.batches if isinstance(batches, EvalBatchSet) else list(batches)
    if not batch_list:
        raise ValueError("run_causal_ablation_matrix requires at least one evaluation batch")

    ctl = _discover_loop_controller(model, loop)
    prev_passes = ctl.active_passes if ctl is not None else None
    eff_depth = 1
    if ctl is not None:
        eff_depth = int(ctl.spec.num_passes if depth is None else min(int(depth), ctl.spec.num_passes))
        ctl.set_active_passes(eff_depth)

    ordered = list(dict.fromkeys(str(a).strip().lower() for a in ablations))
    ordered = ["full", *(a for a in ordered if a != "full")]

    baseline_sha = compute_ablation_state_sha256(model, loop=ctl, gdn2=gdn2, lsa=lsa) if verify_sha256 else ""
    raw_entries: list[dict[str, Any]] = []

    try:
        for abl_name in ordered:
            sha_before = compute_ablation_state_sha256(model, loop=ctl, gdn2=gdn2, lsa=lsa) if verify_sha256 else ""
            rep_nlls: list[float] = []
            lsa_rec_val: float | None = None

            with causal_ablation_context(
                model,
                abl_name,
                loop=ctl,
                gdn2=gdn2,
                lsa=lsa,
                max_anchor_weight=max_anchor_weight,
            ) as ctx_info:
                for _ in range(int(n_repeats)):
                    nll_v, _ = evaluate_pure_lm_nll(
                        model, batch_list, ce_chunk_size=ce_chunk_size, loop=ctl, lsa=lsa
                    )
                    rep_nlls.append(float(nll_v))
                if abl_name == "lsa_sparse" and lsa is not None:
                    lsa_rec_val = float(lsa.evaluate_recall()["mean_recall"])

            if lsa_rec_val is None and "lsa_last_recall" in ctx_info:
                lsa_rec_val = float(ctx_info["lsa_last_recall"])

            sha_after = compute_ablation_state_sha256(model, loop=ctl, gdn2=gdn2, lsa=lsa) if verify_sha256 else ""
            restored_ok = (sha_before == sha_after) if verify_sha256 else True
            if verify_sha256 and not restored_ok:
                raise RuntimeError(
                    f"S2.3 ablation {abl_name!r} failed bit-exact weight/state restoration! "
                    f"before={sha_before[:16]} after={sha_after[:16]}"
                )

            mean_v = sum(rep_nlls) / float(len(rep_nlls))
            std_v = sample_std(rep_nlls)
            raw_entries.append(
                {
                    "ablation": abl_name,
                    "mean_nll": mean_v,
                    "std_nll": std_v,
                    "repeat_nlls": rep_nlls,
                    "applicable": bool(ctx_info.get("applicable", True)),
                    "restoration_verified": restored_ok,
                    "sha_before": sha_before,
                    "sha_after": sha_after,
                    "lsa_last_recall": lsa_rec_val,
                    "diagnostics": dict(ctx_info),
                }
            )
    finally:
        if ctl is not None and prev_passes is not None:
            ctl.set_active_passes(prev_passes)
        _cleanup_eval_side_state(loop=ctl, lsa=lsa)

    full_ent = raw_entries[0]
    dense_ent = next((e for e in raw_entries if e["ablation"] == "lsa_dense"), None)
    dense_ref = float(dense_ent["mean_nll"]) if dense_ent is not None else None

    rows: list[AblationRow] = []
    for ent in raw_entries:
        name = ent["ablation"]
        is_base = name == "full"
        delta = 0.0 if is_base else float(ent["mean_nll"] - full_ent["mean_nll"])
        p_std = pooled_sample_std(
            ent["std_nll"], len(ent["repeat_nlls"]), full_ent["std_nll"], len(full_ent["repeat_nlls"])
        )
        z_sc, exceeds = (0.0, False) if is_base else compute_z_and_3sigma(delta, p_std)
        rec_tag = _recommend_ablation_tag(
            name,
            delta,
            exceeds,
            min_keep_delta=min_keep_delta,
            lsa_last_recall=ent["lsa_last_recall"],
            recall_threshold=recall_threshold,
            max_sparse_nll_gap=max_sparse_nll_gap,
            dense_nll_ref=dense_ref,
            this_mean_nll=float(ent["mean_nll"]),
        )
        rows.append(
            AblationRow(
                ablation=name,
                mean_nll=float(ent["mean_nll"]),
                std_nll=float(ent["std_nll"]),
                ppl=_safe_ppl(float(ent["mean_nll"])),
                repeat_nlls=ent["repeat_nlls"],
                delta_nll_vs_full=delta,
                pooled_std=p_std,
                z_score=z_sc,
                exceeds_3sigma=exceeds,
                recommendation=rec_tag,
                applicable=ent["applicable"],
                is_baseline=is_base,
                restoration_verified=ent["restoration_verified"],
                weight_sha256_before=ent["sha_before"],
                weight_sha256_after=ent["sha_after"],
                lsa_last_recall=ent["lsa_last_recall"],
                diagnostics=ent["diagnostics"],
            )
        )

    return CausalAblationMatrixResult(
        arm_name=arm_name,
        depth=eff_depth,
        n_repeats=int(n_repeats),
        rows=rows,
        all_restorations_verified=all(r.restoration_verified for r in rows),
        full_weight_sha256=baseline_sha,
        metadata={"max_anchor_weight": float(max_anchor_weight)},
    )


# ==============================================================================
# 5. Markdown Report Formatter
# ==============================================================================


def format_sprint2_markdown(report: Sprint2AuditReport) -> str:
    """Format a `Sprint2AuditReport` into a clean, publication-grade GitHub Markdown report."""
    lines: list[str] = [
        "# Sprint 2 Scientific Validation & Causal Ablation Report (`S2.1` + `S2.3`)",
        "",
        f"- **Mode:** `{report.mode}`",
        f"- **Schema:** `{SPRINT2_SCHEMA_VERSION}`",
    ]
    if report.metadata:
        meta_bits = [f"`{k}={v}`" for k, v in sorted(report.metadata.items()) if v is not None]
        if meta_bits:
            lines.append(f"- **Run Metadata:** {', '.join(meta_bits)}")
    lines.append("")

    # Section 1: Datasets & Leak-Free Verification
    lines.append("## 1. Leak-Free Holdout & OOD Dataset Audit (`A1` / `QL60-003`)")
    lines.append("")
    lines.append("| Dataset | Role | Batches | Docs | Valid Loss Tokens | Overlap Check | SHA-256 (prefix) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ds_name, ds_meta in sorted(report.datasets.items()):
        ov = ds_meta.get("overlap_check") or {}
        ov_str = (
            f"PASS (0/{ov.get('eval_docs', 0)})"
            if ov.get("verified")
            else "N/A"
        )
        lines.append(
            f"| `{ds_name}` | `{ds_meta.get('split_role', '?')}` | {ds_meta.get('num_batches', 0)} | "
            f"{ds_meta.get('num_docs', 0)} | {ds_meta.get('valid_loss_tokens', 0)} | {ov_str} | "
            f"`{str(ds_meta.get('dataset_sha256', ''))[:16]}` |"
        )
    lines.append("")

    # Section 2: A3 Multi-Repeat Depth Sweeps & Head-to-Head Comparisons
    lines.append("## 2. Multi-Repeat Pure LM Depth Sweep & Head-to-Head Gate (`A3` / `B4` / `QL60-009`)")
    lines.append("")
    for ds_name, arm_map in sorted(report.depth_sweeps.items()):
        lines.append(f"### Dataset: `{ds_name}`")
        lines.append("")
        lines.append(
            "| Arm | Depths | `d=1` Mean ± Std | `d_max` Mean ± Std | `PPL(d_max)` | "
            "`d1->dmax` Gain | `z`-score | `>3σ` | `2× d=1` Max Diff |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for arm_name, sw in sorted(arm_map.items()):
            m1 = sw.depths.get(1) or sw.depths[sw.d_min]
            mmax = sw.depths[sw.d_max]
            d_list = ",".join(str(d) for d in sorted(sw.depths.keys()))
            lines.append(
                f"| `{arm_name}` | `{d_list}` | `{m1.mean_nll:.5f} ± {m1.std_nll:.5f}` | "
                f"`{mmax.mean_nll:.5f} ± {mmax.std_nll:.5f}` | `{mmax.ppl:.3f}` | "
                f"`{sw.depth_gain_d1_to_dmax:+.5f}` | `{sw.z_score:+.2f}` | "
                f"{'YES' if sw.exceeds_3sigma else 'NO'} | `{sw.d1_repeat_max_abs_diff:.2e}` |"
            )
        lines.append("")

        comp_rows = report.comparisons.get(ds_name) or []
        if comp_rows:
            lines.append(
                "| Arm | Baseline | Depth | Mean NLL | Δ vs Baseline | `z`-score | PPL Δ% | S2 Gate |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            for cr in comp_rows:
                gate_str = "BASELINE" if cr.is_baseline else ("PASS" if cr.passes_s2_gate else "FAIL")
                lines.append(
                    f"| `{cr.arm_name}` | `{cr.baseline_arm}` | `d={cr.arm_depth}` vs `d={cr.baseline_depth}` | "
                    f"`{cr.arm_mean_nll:.5f}` | `{cr.delta_vs_baseline:+.5f}` | "
                    f"`{cr.z_score_vs_baseline:+.2f}` | `{cr.ppl_reduction_pct:+.2f}%` | **{gate_str}** |"
                )
            lines.append("")

    # Section 3: Length Ladder (8K - 1M), RULER Multi-Needle, D-LSA Gap & GDN-2 State Norms
    if report.length_ladders:
        lines.append("## 3. Long-Context Length Ladder, Multi-Needle RULER & `GDN-2` State Norms (`B2`–`B5`)")
        lines.append("")
        for arm_name, ll in sorted(report.length_ladders.items()):
            lines.append(
                f"### Arm: `{arm_name}` (`d={ll.depth}`, `kv_cache_quant={ll.kv_cache_quant}`, "
                f"`all_states_bounded={ll.all_states_bounded}`)"
            )
            lines.append("")
            lines.append(
                "| Context Len | Dense NLL | Sparse NLL | Sparse−Dense Gap | `lsa/last_recall` | "
                "RULER EM | RULER Ans NLL | GDN-2 `max_l2` | GDN-2 `max_abs` | NaNs |"
            )
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for pt in ll.points:
                s_nll = "N/A" if pt.sparse_nll is None else f"{pt.sparse_nll:.5f}"
                s_gap = "N/A" if pt.dense_sparse_nll_gap is None else f"{pt.dense_sparse_nll_gap:+.5f}"
                l_rec = "N/A" if pt.lsa_last_recall is None else f"{pt.lsa_last_recall:.4f}"
                r_em = (
                    f"{pt.ruler.needles_matched}/{pt.ruler.num_needles} ({pt.ruler.exact_match_acc:.1%})"
                    if pt.ruler is not None
                    else "N/A"
                )
                r_nll = f"{pt.ruler.answer_nll:.4f}" if pt.ruler is not None else "N/A"
                st = pt.gdn2_state_norms
                st_l2 = f"{st.max_l2_norm:.3f}" if st is not None else "N/A"
                st_abs = f"{st.max_abs_val:.3f}" if st is not None else "N/A"
                st_nan = str(st.nan_count) if st is not None else "0"
                lines.append(
                    f"| `{pt.length}` | `{pt.dense_nll:.5f}` | `{s_nll}` | `{s_gap}` | `{l_rec}` | "
                    f"{r_em} | `{r_nll}` | `{st_l2}` | `{st_abs}` | `{st_nan}` |"
                )
            lines.append("")

    # Section 4: S2.3 Causal Ablations
    if report.causal_ablations:
        lines.append("## 4. Zero-Training Causal Ablation Matrix (`S2.3`)")
        lines.append("")
        for arm_name, ca in sorted(report.causal_ablations.items()):
            lines.append(
                f"### Arm: `{arm_name}` (`d={ca.depth}`, `repeats={ca.n_repeats}`, "
                f"`all_restorations_verified={ca.all_restorations_verified}`, "
                f"`sha256={ca.full_weight_sha256[:16]}`)"
            )
            lines.append("")
            lines.append(
                "| Ablation | Mean ± Std NLL | `Δ NLL vs Full` | `z`-score | `lsa/last_recall` | "
                "SHA-256 Restored | Recommendation |"
            )
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            for row in ca.rows:
                rec_s = "N/A" if row.lsa_last_recall is None else f"{row.lsa_last_recall:.4f}"
                lines.append(
                    f"| `{row.ablation}` | `{row.mean_nll:.5f} ± {row.std_nll:.5f}` | "
                    f"`{row.delta_nll_vs_full:+.5f}` | `{row.z_score:+.2f}` | `{rec_s}` | "
                    f"{'PASS' if row.restoration_verified else 'FAIL'} | **{row.recommendation}** |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ==============================================================================
# 6. CPU Smoke Builder, GPU Arm Runner & CLI
# ==============================================================================


def build_smoke_cpu_stack(
    *,
    seed: int = 0,
    num_passes: int = 3,
    trained_perturbation: bool = True,
    install_retrofit: bool = True,
) -> dict[str, Any]:
    """Build a deterministic pure-CPU `ToyForCausalLM` stack with `GDN-2` + `D-LSA` + `Loop`."""
    tok = ToyTokenizer()
    torch.manual_seed(int(seed))
    cfg = ToyConfig(
        vocab_size=len(tok),
        hidden_size=16,
        num_hidden_layers=8,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=2,
        linear_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    model = ToyForCausalLM(cfg).eval()
    gdn2: GDN2Manager | None = None
    lsa: LSAManager | None = None
    loop: LoopController | None = None

    if install_retrofit:
        gdn2 = install_gdn2(
            model,
            GDN2Config(
                rank=4,
                chunk_size=16,
                decay_log_min=-3.0,
                decay_log_max=0.5,
                qed_enabled=True,
                qed_max_lambda=0.35,
                neg_eigval_max_boost=0.5,
                backend="pytorch",
            ),
        )
        lsa = install_lsa(
            model,
            LSAConfig(
                mode="sparse",
                index_dim=4,
                index_on="nope",
                rotary_dim=4,
                pool=2,
                block_size=8,
                avg_bucket_keys=4,
                max_bucket_keys=8,
                num_select=2,
                window=8,
                sinks=2,
                query_chunk=8,
                distill_chunk=4,
                num_query_passes=num_passes,
                recall_check_every=1,
            ),
        )
        loop = apply_loop(
            model,
            LoopSpec(
                prelude_end=2,
                core_end=6,
                num_passes=num_passes,
                target_virtual_layers=2 + 4 * num_passes + 2,
                kv_share=True,
                bridge_init=V3_BRIDGE_WEIGHTS,
                bridge_rank=4,
                bridge_mode="subspace",
                per_pass_router=True,
                router_adapter_rank=4,
                router_repel_init=0.0,
            ),
        )

        if trained_perturbation:
            gen = torch.Generator().manual_seed(int(seed) + 9001)
            with torch.no_grad():
                for ad in gdn2.adapters.values():
                    ad.erase_up.weight.normal_(0.0, 0.25, generator=gen)
                    ad.write_up.weight.normal_(0.0, 0.25, generator=gen)
                    ad.decay_up.weight.normal_(0.0, 0.25, generator=gen)
                    if ad.qed_lambda_raw is not None:
                        ad.qed_lambda_raw.uniform_(0.4, 0.8, generator=gen)
                    if ad.neg_gamma_raw is not None:
                        ad.neg_gamma_raw.uniform_(0.4, 0.8, generator=gen)
                    ad._is_zero_cache = None

                loop.adapters.gates.fill_(0.20)
                if loop.adapters.pass_router is not None:
                    pr = loop.adapters.pass_router
                    if pr.rank > 0:
                        pr.up.normal_(0.0, 0.30, generator=gen)
                    else:
                        pr.weight.normal_(0.0, 0.30, generator=gen)
                    pr.bias.normal_(0.0, 0.10, generator=gen)
                    pr.repel.fill_(0.75)

    return {"model": model, "tokenizer": tok, "gdn2": gdn2, "lsa": lsa, "loop": loop}


def run_smoke_cpu(
    *,
    seed: int = 42,
    repeats: int = 3,
    lengths: Sequence[int] = DEFAULT_SMOKE_LENGTHS,
    eval_batches: int = 2,
    seq_len: int = 48,
) -> Sprint2AuditReport:
    """Run the complete pure-CPU Sprint 2 (`S2.1` + `S2.3`) verification suite in `< 10s`."""
    t_start = time.perf_counter()
    base_stack = build_smoke_cpu_stack(seed=seed, num_passes=1, install_retrofit=False)
    cand_stack = build_smoke_cpu_stack(seed=seed, num_passes=3, trained_perturbation=True, install_retrofit=True)
    tok = cand_stack["tokenizer"]

    holdout_cfg = {
        "seq_len": int(seq_len),
        "holdout_mod": 10,
        "holdout_eval_buckets": 2,
        "sources": [
            {"name": "toy_text", "weight": 0.6, "toy_rows": 120},
            {"name": "toy_cot", "weight": 0.4, "toy_rows": 120},
        ],
    }
    holdout_set = build_holdout_eval_batches(
        holdout_cfg,
        tok,
        n_batches=int(eval_batches),
        seed=int(seed),
        split_role="eval",
        holdout_mod=10,
        holdout_eval_buckets=2,
        label="holdout",
        verify_against_train_split=True,
        train_verify_batches=8,
    )
    ood_set = build_ood_eval_batches(
        [{"name": "toy_text", "weight": 1.0, "toy_rows": 80}],
        tok,
        seq_len=int(seq_len),
        n_batches=int(eval_batches),
        seed=int(seed) + 5555,
        label="ood",
        train_doc_ids=None,
    )

    datasets = {
        holdout_set.name: holdout_set.summary_dict(),
        ood_set.name: ood_set.summary_dict(),
    }

    arms = {
        "stock_toy_d1": base_stack,
        "looped_gdn2_lsa_d3": cand_stack,
    }
    depth_sweeps: dict[str, dict[str, DepthSweepResult]] = {}
    comparisons: dict[str, list[ArmComparisonRow]] = {}
    run_identities: dict[str, dict[str, Any]] = {}

    for ds in (holdout_set, ood_set):
        depth_sweeps[ds.name] = {}
        for arm_name, st in arms.items():
            rid = build_sprint2_run_identity(
                st["model"],
                arm_name=arm_name,
                dataset_sha256=ds.dataset_sha256,
                config_payload={"mode": "smoke_cpu", "repeats": repeats, "dataset": ds.name},
                loop=st["loop"],
            )
            run_identities[f"{ds.name}:{arm_name}"] = rid
            depth_sweeps[ds.name][arm_name] = run_depth_nll_sweep(
                st["model"],
                ds,
                depths=(1, 2, 3),
                n_repeats=int(repeats),
                loop=st["loop"],
                lsa=st["lsa"],
                arm_name=arm_name,
                ce_chunk_size=32,
                run_d1_nondet_check=True,
                run_identity=rid,
            )
        comparisons[ds.name] = compare_arms_against_baseline(
            depth_sweeps[ds.name],
            baseline_arm="stock_toy_d1",
        )

    length_ladder = run_length_ladder_audit(
        cand_stack["model"],
        lengths=lengths,
        tokenizer=tok,
        chunk_size=32,
        ce_chunk_size=32,
        kv_cache_quant="native",
        depth=3,
        loop=cand_stack["loop"],
        lsa=cand_stack["lsa"],
        num_needles=4,
        needle_depths=DEFAULT_NEEDLE_DEPTHS,
        seed=int(seed),
        arm_name="looped_gdn2_lsa_d3",
    )

    ablation_matrix = run_causal_ablation_matrix(
        cand_stack["model"],
        holdout_set,
        ablations=CANONICAL_ABLATIONS,
        n_repeats=int(repeats),
        depth=3,
        loop=cand_stack["loop"],
        gdn2=cand_stack["gdn2"],
        lsa=cand_stack["lsa"],
        max_anchor_weight=0.25,
        ce_chunk_size=32,
        verify_sha256=True,
        arm_name="looped_gdn2_lsa_d3",
    )

    elapsed_s = time.perf_counter() - t_start
    return Sprint2AuditReport(
        mode="smoke_cpu",
        datasets=datasets,
        depth_sweeps=depth_sweeps,
        comparisons=comparisons,
        length_ladders={"looped_gdn2_lsa_d3": length_ladder},
        causal_ablations={"looped_gdn2_lsa_d3": ablation_matrix},
        run_identities=run_identities,
        metadata={
            "device": "cpu",
            "seed": int(seed),
            "repeats": int(repeats),
            "elapsed_s": round(elapsed_s, 3),
        },
    )


def parse_arm_specs(
    arms_raw: Sequence[str] | str | None,
    *,
    default_config: str | None = None,
) -> list[dict[str, Any]]:
    """Parse `--arms` specifications from JSON file, inline JSON, or `name=ckpt[:config]` CLI strings."""
    if not arms_raw:
        if default_config is None:
            return []
        return [{"name": Path(default_config).stem, "config": default_config, "checkpoint": None}]

    if isinstance(arms_raw, str):
        items: list[str] = [arms_raw]
    else:
        items = list(arms_raw)

    parsed: list[dict[str, Any]] = []
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        if os.path.isfile(s) and s.endswith(".json"):
            data = json.loads(Path(s).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    entry = dict(v) if isinstance(v, dict) else {"checkpoint": v}
                    entry.setdefault("name", str(k))
                    entry.setdefault("config", default_config)
                    parsed.append(entry)
            elif isinstance(data, list):
                for entry in data:
                    e = dict(entry)
                    e.setdefault("config", default_config)
                    parsed.append(e)
            else:
                raise ValueError(f"unsupported JSON arms structure in {s}")
        elif s.startswith(("[", "{")):
            data = json.loads(s)
            if isinstance(data, list):
                for entry in data:
                    e = dict(entry)
                    e.setdefault("config", default_config)
                    parsed.append(e)
            elif isinstance(data, dict):
                if "name" in data:
                    e = dict(data)
                    e.setdefault("config", default_config)
                    parsed.append(e)
                else:
                    for k, v in data.items():
                        e = dict(v) if isinstance(v, dict) else {"checkpoint": v}
                        e.setdefault("name", str(k))
                        e.setdefault("config", default_config)
                        parsed.append(e)
        else:
            for token in s.split(","):
                tok = token.strip()
                if not tok:
                    continue
                if "=" in tok:
                    name, rest = tok.split("=", 1)
                    if "@" in rest and not os.path.exists(rest):
                        ckpt_part, cfg_part = rest.split("@", 1)
                        parsed.append(
                            {
                                "name": name.strip(),
                                "checkpoint": None if ckpt_part.strip() in ("", "base", "none") else ckpt_part.strip(),
                                "config": cfg_part.strip() or default_config,
                            }
                        )
                    else:
                        ckpt_val = None if rest.strip() in ("", "base", "none") else rest.strip()
                        is_base = rest.strip() == "base"
                        parsed.append(
                            {
                                "name": name.strip(),
                                "checkpoint": ckpt_val,
                                "config": default_config,
                                "base_only": is_base,
                            }
                        )
                else:
                    parsed.append({"name": tok, "checkpoint": None, "config": default_config})
    return parsed


def _load_arm_stack(arm_spec: Mapping[str, Any], *, overrides: Sequence[str] = ()) -> tuple[Any, str | None]:
    """Instantiate a `qloop.train.stack.Stack` from an arm spec and optionally restore its checkpoint."""
    from qloop.config import load_config
    from qloop.train.checkpoint import CheckpointManager
    from qloop.train.stack import build_stack

    cfg_path = arm_spec.get("config")
    if not cfg_path:
        raise ValueError(f"arm {arm_spec.get('name')!r} requires a YAML `config` path (pass --config or in --arms)")
    arm_overrides = list(overrides) + list(arm_spec.get("overrides") or [])
    cfg = load_config(str(cfg_path), overrides=arm_overrides)
    cfg_dict = cfg.to_dict()

    if arm_spec.get("base_only"):
        for section in ("loop", "gdn2", "lsa", "hssd", "experts"):
            if isinstance(cfg_dict.get(section), dict):
                cfg_dict[section]["enabled"] = False
    else:
        for flag_k, sec in (
            ("loop_enabled", "loop"),
            ("gdn2_enabled", "gdn2"),
            ("lsa_enabled", "lsa"),
            ("hssd_enabled", "hssd"),
        ):
            if flag_k in arm_spec and arm_spec[flag_k] is not None:
                cfg_dict.setdefault(sec, {})["enabled"] = bool(arm_spec[flag_k])

    stack = build_stack(cfg_dict)
    ckpt_spec = arm_spec.get("checkpoint")
    resolved_ckpt: str | None = None
    if ckpt_spec:
        from qloop.hub import resolve_run_path

        ckpt_str = resolve_run_path(str(ckpt_spec)) or str(ckpt_spec)
        if os.path.isfile(ckpt_str):
            saved = torch.load(ckpt_str, map_location="cpu")
            params = saved["params"] if isinstance(saved, dict) and "params" in saved else saved
            stack.restore_checkpoint(params, strict=False)
            resolved_ckpt = ckpt_str
        elif os.path.isdir(ckpt_str):
            if os.path.isfile(os.path.join(ckpt_str, "trainable.pt")):
                saved_params = torch.load(os.path.join(ckpt_str, "trainable.pt"), map_location="cpu")
                smeta: dict[str, Any] = {}
                state_fp = os.path.join(ckpt_str, "state.json")
                if os.path.isfile(state_fp):
                    smeta = (json.loads(Path(state_fp).read_text(encoding="utf-8")).get("stack") or {})
                sc = float(smeta.get("expert_delta_scale") or 1.0)
                stack.restore_checkpoint(saved_params, strict=False, expert_delta_scale=sc, prior_stack_meta=smeta)
                resolved_ckpt = ckpt_str
            else:
                ck = CheckpointManager(ckpt_str, keep_last=0).load(map_location="cpu", restore_rng=False)
                if ck is None:
                    raise FileNotFoundError(f"no checkpoint found under {ckpt_str!r}")
                smeta = (ck.get("state") or {}).get("stack") or {}
                sc = float(smeta.get("expert_delta_scale") or 1.0)
                stack.restore_checkpoint(ck["params"], strict=False, expert_delta_scale=sc, prior_stack_meta=smeta)
                resolved_ckpt = str(ck["path"])
        else:
            raise FileNotFoundError(f"checkpoint path not found: {ckpt_str!r}")

    stack.model.eval()
    return stack, resolved_ckpt


def _parse_int_list(raw: Sequence[str] | str) -> list[int]:
    tokens = [raw] if isinstance(raw, str) else list(raw)
    out: list[int] = []
    for tok in tokens:
        for part in str(tok).split(","):
            p = part.strip()
            if p:
                out.append(int(p))
    return out


def run_sprint2_audit_from_config(
    *,
    config_path: str,
    arms: Sequence[Mapping[str, Any]],
    depths: Sequence[int] = (1, 2, 3),
    lengths: Sequence[int] = DEFAULT_GPU_LENGTHS,
    repeats: int = 3,
    eval_batches: int = 4,
    ood_files: Sequence[str | Path] = (),
    chunk_size: int = 4096,
    ce_chunk_size: int = 1024,
    kv_cache_quant: str = "fp8",
    baseline_arm: str | None = None,
    ablation_arm: str | None = None,
    skip_length_ladder: bool = False,
    skip_ablations: bool = False,
    cache_dir: str | Path | None = None,
    allow_mixed_cache: bool = False,
    seed: int = 42,
    overrides: Sequence[str] = (),
) -> Sprint2AuditReport:
    """Execute full GPU/CPU Sprint 2 audit over configured YAML + arms with `QL60-010` cache binding."""
    from qloop.config import load_config

    if not arms:
        raise ValueError("run_sprint2_audit_from_config requires at least one arm")
    t_start = time.perf_counter()
    base_cfg = load_config(config_path, overrides=list(overrides)).to_dict()
    base_name = baseline_arm or str(arms[0]["name"])
    target_ablation_arm = ablation_arm or str(arms[-1]["name"])

    datasets_meta: dict[str, dict[str, Any]] = {}
    depth_sweeps: dict[str, dict[str, DepthSweepResult]] = {}
    length_ladders: dict[str, LengthLadderResult] = {}
    causal_ablations: dict[str, CausalAblationMatrixResult] = {}
    run_identities: dict[str, dict[str, Any]] = {}

    eval_sets: list[EvalBatchSet] | None = None

    for arm_spec in arms:
        arm_name = str(arm_spec["name"])
        stack, ckpt_path = _load_arm_stack(arm_spec, overrides=overrides)
        try:
            if eval_sets is None:
                holdout_set = build_holdout_eval_batches(
                    base_cfg.get("data") or {},
                    stack.tokenizer,
                    n_batches=int(eval_batches),
                    seed=int(seed),
                    split_role="eval",
                    label="holdout",
                )
                eval_sets = [holdout_set]
                if ood_files:
                    ood_seq_len = int((base_cfg.get("data") or {}).get("seq_len", 4096))
                    ood_set = build_local_file_eval_batches(
                        ood_files,
                        stack.tokenizer,
                        seq_len=min(ood_seq_len, 8192),
                        n_batches=int(eval_batches),
                        label="ood",
                        train_doc_ids=None,
                    )
                    eval_sets.append(ood_set)
                for ds in eval_sets:
                    datasets_meta[ds.name] = ds.summary_dict()
                    depth_sweeps[ds.name] = {}

            for ds in eval_sets:
                rid = build_sprint2_run_identity(
                    stack.model,
                    arm_name=arm_name,
                    dataset_sha256=ds.dataset_sha256,
                    config_payload={
                        "depths": [int(d) for d in depths],
                        "repeats": int(repeats),
                        "dataset": ds.name,
                        "arm": dict(arm_spec),
                    },
                    checkpoint_path=ckpt_path,
                    loop=stack.loop,
                )
                run_identities[f"{ds.name}:{arm_name}"] = rid

                cached_sw: dict[str, Any] | None = None
                cpath: Path | None = None
                if cache_dir is not None:
                    safe_arm = arm_name.replace("/", "_").replace("@", "_")
                    cpath = Path(cache_dir) / f"sweep_{ds.name}_{safe_arm}.json"
                    cached_sw = verify_or_load_cached_json(cpath, rid, allow_mixed_cache=allow_mixed_cache)

                if cached_sw is not None and "sweep" in cached_sw:
                    sw_raw = cached_sw["sweep"]
                    sw = DepthSweepResult(
                        arm_name=sw_raw["arm_name"],
                        dataset_name=sw_raw["dataset_name"],
                        dataset_sha256=sw_raw["dataset_sha256"],
                        depths={int(k): DepthMetrics(**v) for k, v in sw_raw["depths"].items()},
                        n_repeats=int(sw_raw["n_repeats"]),
                        d1_repeat_nlls=list(sw_raw["d1_repeat_nlls"]),
                        d1_repeat_max_abs_diff=float(sw_raw["d1_repeat_max_abs_diff"]),
                        d1_repeat_std=float(sw_raw["d1_repeat_std"]),
                        depth_gain_d1_to_dmax=float(sw_raw["depth_gain_d1_to_dmax"]),
                        pooled_std_d1_to_dmax=float(sw_raw["pooled_std_d1_to_dmax"]),
                        z_score=float(sw_raw["z_score"]),
                        exceeds_3sigma=bool(sw_raw["exceeds_3sigma"]),
                        d_min=int(sw_raw["d_min"]),
                        d_max=int(sw_raw["d_max"]),
                        run_identity=rid,
                    )
                else:
                    sw = run_depth_nll_sweep(
                        stack.model,
                        ds,
                        depths=depths,
                        n_repeats=int(repeats),
                        loop=stack.loop,
                        lsa=stack.lsa,
                        arm_name=arm_name,
                        ce_chunk_size=int(ce_chunk_size),
                        run_d1_nondet_check=True,
                        run_identity=rid,
                    )
                    if cpath is not None:
                        save_cached_json(cpath, {"sweep": sw.to_dict()}, rid)
                depth_sweeps[ds.name][arm_name] = sw

            if not skip_length_ladder:
                length_ladders[arm_name] = run_length_ladder_audit(
                    stack.model,
                    lengths=lengths,
                    tokenizer=stack.tokenizer,
                    chunk_size=int(chunk_size),
                    ce_chunk_size=int(ce_chunk_size),
                    kv_cache_quant=kv_cache_quant,
                    loop=stack.loop,
                    lsa=stack.lsa,
                    seed=int(seed),
                    arm_name=arm_name,
                )

            if not skip_ablations and arm_name == target_ablation_arm and eval_sets:
                causal_ablations[arm_name] = run_causal_ablation_matrix(
                    stack.model,
                    eval_sets[0],
                    ablations=CANONICAL_ABLATIONS,
                    n_repeats=int(repeats),
                    loop=stack.loop,
                    gdn2=stack.gdn2,
                    lsa=stack.lsa,
                    ce_chunk_size=int(ce_chunk_size),
                    verify_sha256=True,
                    arm_name=arm_name,
                )
        finally:
            del stack
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparisons = {
        ds_name: compare_arms_against_baseline(arm_map, baseline_arm=base_name)
        for ds_name, arm_map in depth_sweeps.items()
    }
    elapsed_s = time.perf_counter() - t_start
    return Sprint2AuditReport(
        mode="config_arms",
        datasets=datasets_meta,
        depth_sweeps=depth_sweeps,
        comparisons=comparisons,
        length_ladders=length_ladders,
        causal_ablations=causal_ablations,
        run_identities=run_identities,
        metadata={
            "config": str(config_path),
            "baseline_arm": base_name,
            "ablation_arm": target_ablation_arm,
            "repeats": int(repeats),
            "seed": int(seed),
            "elapsed_s": round(elapsed_s, 3),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m qloop.eval.sprint2_audit",
        description="Sprint 2 Scientific Validation & Causal Ablation Harness (S2.1 + S2.3).",
    )
    ap.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Run complete pure-CPU S2.1 + S2.3 self-test on ToyForCausalLM + GDN-2 + D-LSA + Loop (< 10s).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Stage YAML config path (supports _base_ inheritance) for GPU/local evaluation.",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Arm specs as JSON file path, inline JSON, or name=ckpt specs (use name=base for stock baseline).",
    )
    ap.add_argument("--baseline-arm", type=str, default=None, help="Arm name to serve as head-to-head baseline.")
    ap.add_argument("--ablation-arm", type=str, default=None, help="Arm name to run S2.3 causal ablations on.")
    ap.add_argument(
        "--depths",
        nargs="+",
        default=["1,2,3"],
        help="Loop depths to sweep (e.g. --depths 1,2,3 or --depths 1 2 3).",
    )
    ap.add_argument(
        "--lengths",
        nargs="+",
        default=None,
        help="Context lengths for length ladder (default: 64,128,256 in --smoke-cpu; 8192..1048576 on GPU).",
    )
    ap.add_argument("--repeats", type=int, default=3, help="Number of repeats per depth/ablation (A3 default: 3).")
    ap.add_argument("--eval-batches", type=int, default=None, help="Number of leak-free holdout batches.")
    ap.add_argument("--ood-file", action="append", default=[], help="Optional local JSONL/JSON/text OOD file path.")
    ap.add_argument("--chunk-size", type=int, default=4096, help="Streaming prefill chunk size on GPU.")
    ap.add_argument("--ce-chunk-size", type=int, default=1024, help="Token chunk size for chunked CE loss.")
    ap.add_argument(
        "--kv-cache-quant",
        type=str,
        default="fp8",
        choices=("native", "fp8", "hadamard_int4"),
        help="KV cache quantization mode for GPU length-ladder streaming.",
    )
    ap.add_argument("--cache-dir", type=str, default=None, help="Optional QL60-010 identity-bound cache directory.")
    ap.add_argument("--allow-mixed-cache", action="store_true", help="Allow overwriting mismatched QL60-010 cache.")
    ap.add_argument("--skip-length-ladder", action="store_true", help="Skip length-ladder & RULER audit.")
    ap.add_argument("--skip-ablations", action="store_true", help="Skip S2.3 causal ablation matrix.")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed.")
    ap.add_argument("-s", "--set", dest="overrides", action="append", default=[], help="Config key=value override.")
    ap.add_argument(
        "--output-json",
        "--out-json",
        dest="output_json",
        type=str,
        default=None,
        help="Optional path to write deterministic JSON report.",
    )
    ap.add_argument(
        "--output-md",
        "--out-md",
        dest="output_md",
        type=str,
        default=None,
        help="Optional path to write Markdown report.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.smoke_cpu:
        lengths = _parse_int_list(args.lengths) if args.lengths else list(DEFAULT_SMOKE_LENGTHS)
        eval_b = int(args.eval_batches) if args.eval_batches is not None else 2
        report = run_smoke_cpu(
            seed=int(args.seed),
            repeats=int(args.repeats),
            lengths=lengths,
            eval_batches=eval_b,
        )
    else:
        if not args.config:
            ap.error("must specify either --smoke-cpu or --config <yaml>")
        depths = _parse_int_list(args.depths)
        lengths = _parse_int_list(args.lengths) if args.lengths else list(DEFAULT_GPU_LENGTHS)
        arms = parse_arm_specs(args.arms, default_config=args.config)
        eval_b = int(args.eval_batches) if args.eval_batches is not None else 4
        report = run_sprint2_audit_from_config(
            config_path=args.config,
            arms=arms,
            depths=depths,
            lengths=lengths,
            repeats=int(args.repeats),
            eval_batches=eval_b,
            ood_files=args.ood_file,
            chunk_size=int(args.chunk_size),
            ce_chunk_size=int(args.ce_chunk_size),
            kv_cache_quant=str(args.kv_cache_quant),
            baseline_arm=args.baseline_arm,
            ablation_arm=args.ablation_arm,
            skip_length_ladder=bool(args.skip_length_ladder),
            skip_ablations=bool(args.skip_ablations),
            cache_dir=args.cache_dir,
            allow_mixed_cache=bool(args.allow_mixed_cache),
            seed=int(args.seed),
            overrides=args.overrides,
        )

    md = format_sprint2_markdown(report)
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
    "CANONICAL_ABLATIONS",
    "DEFAULT_GPU_LENGTHS",
    "DEFAULT_NEEDLE_DEPTHS",
    "DEFAULT_SMOKE_LENGTHS",
    "RECOMMENDATION_TAGS",
    "SPRINT2_SCHEMA_VERSION",
    "AblationRow",
    "ArmComparisonRow",
    "CausalAblationMatrixResult",
    "DepthMetrics",
    "DepthSweepResult",
    "EvalBatchSet",
    "GDN2StateNormTelemetry",
    "LengthLadderPoint",
    "LengthLadderResult",
    "RulerMultiNeedleResult",
    "Sprint2AuditReport",
    "ablate_full",
    "ablate_low_anchor_probe",
    "ablate_lsa_mode",
    "ablate_no_gdn2",
    "ablate_no_neg_eigval",
    "ablate_no_pass_router",
    "ablate_no_qed",
    "assert_zero_doc_overlap",
    "build_arg_parser",
    "build_holdout_eval_batches",
    "build_local_file_eval_batches",
    "build_ood_eval_batches",
    "build_ruler_multi_needle_episode",
    "build_smoke_cpu_stack",
    "build_sprint2_run_identity",
    "causal_ablation_context",
    "compare_arms_against_baseline",
    "compute_ablation_state_sha256",
    "compute_batch_set_digest",
    "compute_weights_sha256",
    "compute_z_and_3sigma",
    "evaluate_pure_lm_nll",
    "evaluate_ruler_multi_needle",
    "format_sprint2_markdown",
    "inspect_gdn2_recurrent_state_norms",
    "main",
    "parse_arm_specs",
    "pooled_sample_std",
    "run_causal_ablation_matrix",
    "run_depth_nll_sweep",
    "run_length_ladder_audit",
    "run_smoke_cpu",
    "run_sprint2_audit_from_config",
    "sample_std",
    "save_cached_json",
    "stream_chunked_lm_nll",
    "verify_or_load_cached_json",
]


if __name__ == "__main__":
    sys.exit(main())
