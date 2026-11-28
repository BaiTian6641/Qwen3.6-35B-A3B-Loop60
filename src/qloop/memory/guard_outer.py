"""GUARD G4 — Unified Dual-Loop `LSA-NL` Fast Outer Loop Controller.

Connects Google `GUARD` (`arXiv:2611.23476`) item-level `Soft-HURT` monotonic gating and
directional regret tracking (`TaskRegretTracker`) with the three-tier `LSA-NL` Split-Horizon
Continuum Memory stack:

  * **Tier 1 (KV-Mem Blobs + DLMG):** `propose_kv_rule()` evaluates candidate `KVMemBlob`s
    (preserving optional `gdn2_recurrent_states`/`gdn2_conv_states` untouched), enforces
    asymmetric `Soft-HURT` gating matching `G1` (`src/qloop/train/rl.py`), admits passing rules
    through `SplitHorizonKVBlobStore.admit()` (respecting salience gating, semantic links, and
    bi-temporal supersession), and logs regressions to `TaskRegretTracker` without mutating
    `kv_store` or baseline scores on rejection.
  * **Tier 2 (ESM Null-Space Compilation):** `graduate_kv_to_esm()` selects hardened blobs via
    `kv_store.select_for_graduation()`, compiles them into `EpisodicSideMemoryBank` (`P_perp`
    AlphaEdit DGKS), verifies them under `Soft-HURT`, tombstones/supersedes promoted blobs in
    `kv_store`, and performs exact reverse-order transactional rollback via
    `esm_bank.rollback_entry()` on rejection OR on any exception raised by `eval_fn`.
  * **Tier 3 (Slow Inner Loop Bridge):** `export_slow_inner_bundle()` writes a contamination-free
    `guard_outer_state.json` summary and `slow_inner_distill_queue.jsonl` prioritization queue
    (graduated rules, newly hurt IDs, chronic unsolved IDs — zero raw benchmark text).
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from qloop.train.sirsi import (
    REGRET_NEWLY_HURT,
    REGRET_PINNED_PASS,
    REGRET_UNSOLVED_CHRONIC,
    REGRET_UNSOLVED_NOVEL,
    TaskRegretTracker,
)

from .engram import AnchoredMoEEngramBank
from .esm import EpisodicSideMemoryBank, ESMLedgerEntry
from .kv_blob import KVMemBlob, SplitHorizonKVBlobStore

SCHEMA_VERSION = "guard_g4_outer_v1"

_BLOCKED_CONTAMINATION_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "prompts",
        "completion",
        "completions",
        "bad_completion",
        "raw_text",
        "eval_text",
        "benchmark_text",
        "answer",
        "ground_truth",
        "solution",
        "samples",
        "messages",
        "response",
        "responses",
    }
)


def _clamp01(value: float) -> float:
    """Clamp item score to `[0.0, 1.0]` matching `G1`, mapping `NaN` safely to `0.0`."""
    v = float(value)
    if math.isnan(v):
        return 0.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _sanitize_meta(obj: Any) -> Any:
    """Recursively strip raw eval-text keys and coerce values to JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, Mapping):
        clean: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if ks.strip().lower() in _BLOCKED_CONTAMINATION_KEYS:
                continue
            clean[ks] = _sanitize_meta(v)
        return clean
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_meta(item) for item in obj]
    return str(obj)


def _normalize_compiled_entries(
    raw: tuple[int, ESMLedgerEntry] | Sequence[tuple[int, ESMLedgerEntry]],
) -> list[tuple[int, ESMLedgerEntry]]:
    """Normalize single `(layer_idx, entry)` or sequence of pairs into a validated list."""
    if (
        isinstance(raw, tuple)
        and len(raw) == 2
        and isinstance(raw[0], int)
        and not isinstance(raw[0], bool)
        and isinstance(raw[1], ESMLedgerEntry)
    ):
        return [(int(raw[0]), raw[1])]

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(
            "compile_fn must return (layer_idx, ESMLedgerEntry) or a sequence of such pairs; "
            f"got {type(raw).__name__}"
        )

    pairs: list[tuple[int, ESMLedgerEntry]] = []
    for item in raw:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], int)
            or isinstance(item[0], bool)
            or not isinstance(item[1], ESMLedgerEntry)
        ):
            raise TypeError(f"expected (int, ESMLedgerEntry) pair from compile_fn, got {item!r}")
        pairs.append((int(item[0]), item[1]))
    return pairs


@dataclass
class SoftHurtGateConfig:
    """Configuration for item-level asymmetric `Soft-HURT` monotonic gating (parity with `G1`)."""

    lambda_hurt: float = 2.0
    min_soft_hurt: float = 0.0
    max_hurt_frac: float = 0.05
    max_bench_drop: float = 0.025
    pass_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.lambda_hurt = float(self.lambda_hurt)
        self.min_soft_hurt = float(self.min_soft_hurt)
        self.max_hurt_frac = float(self.max_hurt_frac)
        self.max_bench_drop = float(self.max_bench_drop)
        self.pass_threshold = float(self.pass_threshold)
        self.validate()

    def validate(self) -> None:
        if not math.isfinite(self.lambda_hurt) or self.lambda_hurt < 0.0:
            raise ValueError(f"lambda_hurt must be finite and >= 0, got {self.lambda_hurt}")
        if not math.isfinite(self.min_soft_hurt):
            raise ValueError(f"min_soft_hurt must be finite, got {self.min_soft_hurt}")
        if not math.isfinite(self.max_hurt_frac) or not (0.0 <= self.max_hurt_frac <= 1.0):
            raise ValueError(f"max_hurt_frac must be finite and in [0, 1], got {self.max_hurt_frac}")
        if not math.isfinite(self.max_bench_drop) or self.max_bench_drop < 0.0:
            raise ValueError(f"max_bench_drop must be finite and >= 0, got {self.max_bench_drop}")
        if not math.isfinite(self.pass_threshold) or not (0.0 <= self.pass_threshold <= 1.0):
            raise ValueError(f"pass_threshold must be finite and in [0, 1], got {self.pass_threshold}")

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_hurt": float(self.lambda_hurt),
            "min_soft_hurt": float(self.min_soft_hurt),
            "max_hurt_frac": float(self.max_hurt_frac),
            "max_bench_drop": float(self.max_bench_drop),
            "pass_threshold": float(self.pass_threshold),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | SoftHurtGateConfig | None) -> SoftHurtGateConfig:
        if isinstance(d, cls):
            return d
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown soft_hurt.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)


@dataclass(frozen=True)
class SoftHurtOutcome:
    """Contamination-free outcome of a `Soft-HURT` baseline-vs-candidate evaluation."""

    promoted: bool
    n_items: int
    gain_mean: float
    hurt_mean: float
    hurt_frac: float
    soft_hurt_score: float
    max_bench_drop_seen: float
    bench_deltas: dict[str, float]
    newly_hurt_ids: tuple[str, ...]
    newly_fixed_ids: tuple[str, ...]
    pinned_passed_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_diff_dict(self) -> dict[str, Any]:
        """Return a contamination-free JSON-serializable audit dictionary (IDs & metrics only)."""
        return {
            "promoted": bool(self.promoted),
            "n_items": int(self.n_items),
            "gain_mean": float(self.gain_mean),
            "hurt_mean": float(self.hurt_mean),
            "hurt_frac": float(self.hurt_frac),
            "soft_hurt_score": float(self.soft_hurt_score),
            "max_bench_drop_seen": float(self.max_bench_drop_seen),
            "bench_deltas": {str(k): float(v) for k, v in sorted(self.bench_deltas.items())},
            "n_newly_hurt": len(self.newly_hurt_ids),
            "n_newly_fixed": len(self.newly_fixed_ids),
            "n_pinned_passed": len(self.pinned_passed_ids),
            "newly_hurt_ids": list(self.newly_hurt_ids),
            "newly_fixed_ids": list(self.newly_fixed_ids),
            "pinned_passed_ids": list(self.pinned_passed_ids),
            "rejection_reasons": list(self.rejection_reasons),
        }


def evaluate_soft_hurt(
    baseline_items: Mapping[str, float],
    candidate_items: Mapping[str, float],
    *,
    cfg: SoftHurtGateConfig | None = None,
    bench_map: Mapping[str, str] | None = None,
) -> SoftHurtOutcome:
    """Evaluate candidate item scores against baseline using exact `GUARD G1` `Soft-HURT` math.

    Semantics (100% consistent with `GRPOTrainer.gate(mode="soft_hurt")` in `src/qloop/train/rl.py`):
      * All item scores in both `baseline_items` and `candidate_items` are clamped to `[0.0, 1.0]`
        so bonus scores (e.g. KernelBench speedup bonuses > 1.0) can never mask correctness drops.
      * When `baseline_items` is empty (initial bootstrap pass), promotes cleanly with zero gain/hurt
        and populates `pinned_passed_ids` with all candidate items scoring `>= cfg.pass_threshold`.
      * On non-empty baseline, computes over common item IDs:
          - `newly_hurt_ids`: `base[k] >= tau` and `cand[k] < tau` (pass -> fail flip).
          - `newly_fixed_ids`: `base[k] < tau` and `cand[k] >= tau` (fail -> pass flip).
          - `gain_mean = mean(max(0, cand[k] - base[k]))`,
            `hurt_mean = mean(max(0, base[k] - cand[k]))`,
            `soft_hurt_score = gain_mean - lambda_hurt * hurt_mean`,
            `hurt_frac = len(newly_hurt_ids) / len(common)`.
          - Per-benchmark signed mean deltas (`cand_mean - base_mean`) and `max_bench_drop_seen`
            when `bench_map` is provided (`item_id -> benchmark_name`).
      * Promotion requires:
          1. If `hurt_mean > 0`: strictly positive `soft_hurt_score > 0` AND
             `soft_hurt_score >= min_soft_hurt`; else (`hurt_mean == 0`):
             `soft_hurt_score >= min_soft_hurt`.
          2. `hurt_frac <= max_hurt_frac`.
          3. `max_bench_drop_seen <= max_bench_drop`.
    """
    gcfg = cfg if cfg is not None else SoftHurtGateConfig()
    gcfg.validate()
    tau = gcfg.pass_threshold

    base = {str(k): _clamp01(v) for k, v in baseline_items.items()}
    cand = {str(k): _clamp01(v) for k, v in candidate_items.items()}
    pinned_passed = tuple(sorted(k for k, v in cand.items() if v >= tau))

    # Bootstrap / initial baseline pass when no baseline exists yet.
    if not base:
        boot_deltas: dict[str, float] = {}
        if bench_map is not None:
            for item_id in cand:
                b_name = bench_map.get(item_id)
                if b_name is not None:
                    boot_deltas[str(b_name)] = 0.0
        return SoftHurtOutcome(
            promoted=True,
            n_items=len(cand),
            gain_mean=0.0,
            hurt_mean=0.0,
            hurt_frac=0.0,
            soft_hurt_score=0.0,
            max_bench_drop_seen=0.0,
            bench_deltas=dict(sorted(boot_deltas.items())),
            newly_hurt_ids=(),
            newly_fixed_ids=(),
            pinned_passed_ids=pinned_passed,
            rejection_reasons=(),
        )

    common = sorted(k for k in cand if k in base)
    if not common:
        return SoftHurtOutcome(
            promoted=False,
            n_items=0,
            gain_mean=0.0,
            hurt_mean=0.0,
            hurt_frac=0.0,
            soft_hurt_score=0.0,
            max_bench_drop_seen=0.0,
            bench_deltas={},
            newly_hurt_ids=(),
            newly_fixed_ids=(),
            pinned_passed_ids=pinned_passed,
            rejection_reasons=("no_common_items",),
        )

    n = len(common)
    newly_fixed = tuple(k for k in common if base[k] < tau and cand[k] >= tau)
    newly_hurt = tuple(k for k in common if base[k] >= tau and cand[k] < tau)

    gain_mean = sum(max(0.0, cand[k] - base[k]) for k in common) / float(n)
    hurt_mean = sum(max(0.0, base[k] - cand[k]) for k in common) / float(n)
    soft_hurt_score = gain_mean - gcfg.lambda_hurt * hurt_mean
    hurt_frac = float(len(newly_hurt)) / float(n)

    bench_deltas: dict[str, float] = {}
    max_bench_drop_seen = 0.0
    worst_bench = ""
    if bench_map is not None:
        grouped: dict[str, list[str]] = {}
        for item_id in common:
            b_name = bench_map.get(item_id)
            if b_name is not None:
                grouped.setdefault(str(b_name), []).append(item_id)
        for b_name in sorted(grouped):
            b_ids = grouped[b_name]
            b_base = sum(base[i] for i in b_ids) / float(len(b_ids))
            b_cand = sum(cand[i] for i in b_ids) / float(len(b_ids))
            delta = b_cand - b_base
            bench_deltas[b_name] = float(delta)
            drop = max(0.0, b_base - b_cand)
            if drop > max_bench_drop_seen:
                max_bench_drop_seen = float(drop)
                worst_bench = b_name

    reasons: list[str] = []
    eps = 1e-12
    if hurt_mean > 0.0:
        if soft_hurt_score <= eps:
            reasons.append(f"soft_hurt_non_positive({soft_hurt_score:.6f}<=0.0)")
        elif soft_hurt_score + eps < gcfg.min_soft_hurt:
            reasons.append(f"soft_hurt_below_min({soft_hurt_score:.6f}<{gcfg.min_soft_hurt:.6f})")
    elif soft_hurt_score + eps < gcfg.min_soft_hurt:
        reasons.append(f"soft_hurt_below_min({soft_hurt_score:.6f}<{gcfg.min_soft_hurt:.6f})")

    if hurt_frac > gcfg.max_hurt_frac + eps:
        reasons.append(f"hurt_frac_exceeded({hurt_frac:.6f}>{gcfg.max_hurt_frac:.6f})")

    if max_bench_drop_seen > gcfg.max_bench_drop + eps:
        reasons.append(
            f"bench_drop_exceeded({worst_bench}:{max_bench_drop_seen:.6f}>{gcfg.max_bench_drop:.6f})"
        )

    return SoftHurtOutcome(
        promoted=len(reasons) == 0,
        n_items=n,
        gain_mean=float(gain_mean),
        hurt_mean=float(hurt_mean),
        hurt_frac=float(hurt_frac),
        soft_hurt_score=float(soft_hurt_score),
        max_bench_drop_seen=float(max_bench_drop_seen),
        bench_deltas=bench_deltas,
        newly_hurt_ids=newly_hurt,
        newly_fixed_ids=newly_fixed,
        pinned_passed_ids=pinned_passed,
        rejection_reasons=tuple(reasons),
    )


@dataclass
class GuardOuterConfig:
    """Configuration for `LSANLGuardOuterLoop` dual-loop fast outer-loop controller."""

    soft_hurt: SoftHurtGateConfig = field(default_factory=SoftHurtGateConfig)
    graduation_max_per_cycle: int = 4
    graduation_min_hits: int | None = None
    graduation_min_stability: float | None = None
    rollback_esm_on_reject: bool = True
    chronic_fail_streak: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.soft_hurt, SoftHurtGateConfig):
            self.soft_hurt = SoftHurtGateConfig.from_dict(self.soft_hurt)
        self.graduation_max_per_cycle = int(self.graduation_max_per_cycle)
        self.graduation_min_hits = None if self.graduation_min_hits is None else int(self.graduation_min_hits)
        self.graduation_min_stability = (
            None if self.graduation_min_stability is None else float(self.graduation_min_stability)
        )
        self.rollback_esm_on_reject = bool(self.rollback_esm_on_reject)
        self.chronic_fail_streak = int(self.chronic_fail_streak)
        self.validate()

    def validate(self) -> None:
        self.soft_hurt.validate()
        if self.graduation_max_per_cycle < 1:
            raise ValueError(f"graduation_max_per_cycle must be >= 1, got {self.graduation_max_per_cycle}")
        if self.graduation_min_hits is not None and self.graduation_min_hits < 1:
            raise ValueError(f"graduation_min_hits must be >= 1, got {self.graduation_min_hits}")
        if self.graduation_min_stability is not None and (
            not math.isfinite(self.graduation_min_stability) or self.graduation_min_stability <= 0.0
        ):
            raise ValueError(
                f"graduation_min_stability must be finite and > 0, got {self.graduation_min_stability}"
            )
        if self.chronic_fail_streak < 1:
            raise ValueError(f"chronic_fail_streak must be >= 1, got {self.chronic_fail_streak}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "soft_hurt": self.soft_hurt.to_dict(),
            "graduation_max_per_cycle": int(self.graduation_max_per_cycle),
            "graduation_min_hits": self.graduation_min_hits,
            "graduation_min_stability": self.graduation_min_stability,
            "rollback_esm_on_reject": bool(self.rollback_esm_on_reject),
            "chronic_fail_streak": int(self.chronic_fail_streak),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | GuardOuterConfig | None) -> GuardOuterConfig:
        if isinstance(d, cls):
            return d
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown guard_outer.* keys {sorted(bad)}; valid: {sorted(known)}")
        if "soft_hurt" in raw and raw["soft_hurt"] is not None:
            raw["soft_hurt"] = SoftHurtGateConfig.from_dict(raw["soft_hurt"])
        return cls(**raw)


@dataclass(frozen=True)
class GuardProposalResult:
    """Result record for one Tier-1 `KVMemBlob` rule proposal."""

    blob_id: str
    outcome: SoftHurtOutcome
    admitted: bool
    promoted: bool
    superseded_ids: tuple[str, ...] = ()
    semantic_links_added: tuple[tuple[str, float], ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    has_gdn2_state: bool = False
    rule_meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "kv_proposal",
            "blob_id": str(self.blob_id),
            "admitted": bool(self.admitted),
            "promoted": bool(self.promoted),
            "superseded_ids": list(self.superseded_ids),
            "semantic_links_added": [[str(dst), float(w)] for dst, w in self.semantic_links_added],
            "rejection_reasons": list(self.rejection_reasons),
            "has_gdn2_state": bool(self.has_gdn2_state),
            "rule_meta": _sanitize_meta(self.rule_meta),
            "timestamp": float(self.timestamp),
            "outcome": self.outcome.to_diff_dict(),
        }


@dataclass(frozen=True)
class GuardGraduationResult:
    """Result record for one Tier-2 `KVMemBlob -> ESM` null-space graduation attempt."""

    blob_id: str
    outcome: SoftHurtOutcome
    promoted: bool
    compiled_entries: tuple[tuple[int, str], ...] = ()
    rolled_back_entries: tuple[tuple[int, str], ...] = ()
    superseded_in_kv: bool = False
    max_post_leakage: float = 0.0
    rejection_reasons: tuple[str, ...] = ()
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "esm_graduation",
            "blob_id": str(self.blob_id),
            "promoted": bool(self.promoted),
            "compiled_entries": [[int(lyr), str(eid)] for lyr, eid in self.compiled_entries],
            "rolled_back_entries": [[int(lyr), str(eid)] for lyr, eid in self.rolled_back_entries],
            "superseded_in_kv": bool(self.superseded_in_kv),
            "max_post_leakage": float(self.max_post_leakage),
            "rejection_reasons": list(self.rejection_reasons),
            "timestamp": float(self.timestamp),
            "outcome": self.outcome.to_diff_dict(),
        }


class LSANLGuardOuterLoop:
    """Unified `GUARD G4` + `LSA-NL` Fast Outer Loop Controller across KV-Mem, ESM, and Slow-Inner RL."""

    def __init__(
        self,
        kv_store: SplitHorizonKVBlobStore,
        *,
        esm_bank: EpisodicSideMemoryBank | None = None,
        engram_bank: AnchoredMoEEngramBank | None = None,
        config: GuardOuterConfig | None = None,
        regret_tracker: TaskRegretTracker | None = None,
    ) -> None:
        self.kv_store = kv_store
        self.esm_bank = esm_bank
        self.engram_bank = engram_bank
        self.config = config if config is not None else GuardOuterConfig()
        self.regret_tracker = regret_tracker if regret_tracker is not None else TaskRegretTracker()

        self.baseline_items: dict[str, float] = {}
        self.bench_map: dict[str, str] = {}
        self.pinned_passed: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.graduated_rules: list[dict[str, Any]] = []
        self.admitted_rule_meta: dict[str, dict[str, Any]] = {}

        self._hurt_event_counts: dict[str, int] = {}
        self._cycle_count: int = 0
        self._promoted_kv_count: int = 0
        self._rejected_kv_count: int = 0
        self._salience_rejected_kv_count: int = 0
        self._promoted_esm_count: int = 0
        self._rejected_esm_count: int = 0
        self._rollback_esm_count: int = 0
        self._clock: float = 0.0

    def _resolve_time(self, now: float | None, *, floor_time: float = 0.0) -> float:
        if now is not None:
            t = float(now)
            self._clock = max(self._clock, t)
            return t
        self._clock = max(self._clock, float(floor_time))
        return self._clock

    def _resolve_bench_map(self, bench_map: Mapping[str, str] | None) -> dict[str, str] | None:
        if bench_map is None:
            return dict(self.bench_map) if self.bench_map else None
        merged = dict(self.bench_map)
        for k, v in bench_map.items():
            merged[str(k)] = str(v)
        return merged

    def _record_promoted_scores(self, cand_clamped: Mapping[str, float], outcome: SoftHurtOutcome) -> None:
        if not self.baseline_items:
            self.baseline_items = dict(cand_clamped)
        else:
            self.baseline_items.update(cand_clamped)
        self.pinned_passed |= set(outcome.pinned_passed_ids)
        for item_id in outcome.newly_hurt_ids:
            self._hurt_event_counts[item_id] = self._hurt_event_counts.get(item_id, 0) + 1
        for item_id in sorted(cand_clamped):
            self.regret_tracker.classify_and_update(item_id, cand_clamped[item_id])

    def _record_rejected_hurt_ids(
        self,
        cand_clamped: Mapping[str, float],
        newly_hurt_ids: Sequence[str],
    ) -> None:
        for item_id in newly_hurt_ids:
            self._hurt_event_counts[item_id] = self._hurt_event_counts.get(item_id, 0) + 1
            if item_id not in self.regret_tracker.states and item_id in self.baseline_items:
                self.regret_tracker.classify_and_update(item_id, self.baseline_items[item_id])
            if item_id in cand_clamped:
                self.regret_tracker.classify_and_update(item_id, cand_clamped[item_id])

    def initialize_baseline(
        self,
        item_scores: Mapping[str, float],
        *,
        bench_map: Mapping[str, str] | None = None,
    ) -> None:
        """Seed the controller baseline item scores, benchmark map, pinned set, and regret tracker."""
        clamped = {str(k): _clamp01(v) for k, v in item_scores.items()}
        self.baseline_items = dict(clamped)
        if bench_map is not None:
            self.bench_map = {str(k): str(v) for k, v in bench_map.items()}
        tau = self.config.soft_hurt.pass_threshold
        self.pinned_passed = {k for k, v in clamped.items() if v >= tau}
        for item_id in sorted(clamped):
            self.regret_tracker.classify_and_update(item_id, clamped[item_id])

    def propose_kv_rule(
        self,
        blob: KVMemBlob,
        *,
        eval_fn: Callable[[KVMemBlob], Mapping[str, float]],
        supersede_ids: Sequence[str] = (),
        semantic_links: Sequence[tuple[str, float]] = (),
        bench_map: Mapping[str, str] | None = None,
        rule_meta: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> GuardProposalResult:
        """Evaluate and monotonically gate a Tier-1 `KVMemBlob` rule under `Soft-HURT`."""
        self._cycle_count += 1
        t_now = self._resolve_time(now, floor_time=max(blob.header.t_valid, blob.header.t_created))
        eff_bench_map = self._resolve_bench_map(bench_map)
        if bench_map is not None:
            for k, v in bench_map.items():
                self.bench_map[str(k)] = str(v)

        has_gdn2 = bool(blob.gdn2_recurrent_states or blob.gdn2_conv_states)
        clean_meta = _sanitize_meta(rule_meta) if rule_meta is not None else {}

        raw_cand = eval_fn(blob)
        cand_clamped = {str(k): _clamp01(v) for k, v in raw_cand.items()}
        outcome = evaluate_soft_hurt(
            self.baseline_items,
            cand_clamped,
            cfg=self.config.soft_hurt,
            bench_map=eff_bench_map,
        )

        if not outcome.promoted:
            self._rejected_kv_count += 1
            self._record_rejected_hurt_ids(cand_clamped, outcome.newly_hurt_ids)
            result = GuardProposalResult(
                blob_id=blob.blob_id,
                outcome=outcome,
                admitted=False,
                promoted=False,
                rejection_reasons=outcome.rejection_reasons,
                has_gdn2_state=has_gdn2,
                rule_meta=clean_meta,
                timestamp=t_now,
            )
            self.history.append(result.to_dict())
            return result

        # Soft-HURT promoted: attempt salience-gated admission into `SplitHorizonKVBlobStore`.
        # NOTE: `kv_store.admit(blob)` preserves `blob.gdn2_recurrent_states` and `gdn2_conv_states` untouched.
        admitted = self.kv_store.admit(blob)
        if not admitted:
            self._salience_rejected_kv_count += 1
            return GuardProposalResult(
                blob_id=blob.blob_id,
                outcome=outcome,
                admitted=False,
                promoted=False,
                rejection_reasons=("kv_salience_rejected",),
                has_gdn2_state=has_gdn2,
                rule_meta=clean_meta,
                timestamp=t_now,
            )

        links_added: list[tuple[str, float]] = []
        for dst_id, weight in semantic_links:
            dst = str(dst_id)
            w = float(weight)
            if dst != blob.blob_id and dst in self.kv_store.graph.nodes and w > 0.0:
                self.kv_store.graph.add_semantic_edge(blob.blob_id, dst, w)
                links_added.append((dst, w))

        superseded: list[str] = []
        for old_id in supersede_ids:
            oid = str(old_id)
            if oid != blob.blob_id and oid in self.kv_store.blobs:
                self.kv_store.supersede(oid, blob.blob_id, now=t_now)
                superseded.append(oid)

        self._promoted_kv_count += 1
        self.admitted_rule_meta[blob.blob_id] = dict(clean_meta)
        self._record_promoted_scores(cand_clamped, outcome)

        result = GuardProposalResult(
            blob_id=blob.blob_id,
            outcome=outcome,
            admitted=True,
            promoted=True,
            superseded_ids=tuple(superseded),
            semantic_links_added=tuple(links_added),
            rejection_reasons=(),
            has_gdn2_state=has_gdn2,
            rule_meta=clean_meta,
            timestamp=t_now,
        )
        self.history.append(result.to_dict())
        return result

    def _select_graduation_candidates(self, *, now: float | None = None) -> list[KVMemBlob]:
        """Select graduation candidates via `kv_store.select_for_graduation`, honoring config overrides."""
        orig_hits = self.kv_store.config.graduation_hits
        orig_stab = self.kv_store.config.graduation_min_stability
        try:
            if self.config.graduation_min_hits is not None:
                self.kv_store.config.graduation_hits = int(self.config.graduation_min_hits)
            if self.config.graduation_min_stability is not None:
                self.kv_store.config.graduation_min_stability = float(self.config.graduation_min_stability)
            ready = self.kv_store.select_for_graduation(now=now)
        finally:
            self.kv_store.config.graduation_hits = orig_hits
            self.kv_store.config.graduation_min_stability = orig_stab
        return ready[: self.config.graduation_max_per_cycle]

    def _rollback_esm_entries(
        self,
        compiled_pairs: Sequence[tuple[int, ESMLedgerEntry]],
    ) -> tuple[tuple[int, str], ...]:
        assert self.esm_bank is not None
        rolled: list[tuple[int, str]] = []
        for layer_idx, entry in reversed(compiled_pairs):
            self.esm_bank.rollback_entry(int(layer_idx), entry.entry_id)
            rolled.append((int(layer_idx), str(entry.entry_id)))
        return tuple(rolled)

    def _supersede_graduated_blob(self, blob: KVMemBlob, esm_target_id: str, *, now: float) -> bool:
        """Bi-temporally tombstone a graduated KV blob in `kv_store` & `DLMG` (`t_invalid = now`)."""
        t_inv = float(now)
        blob.header.t_invalid = t_inv
        if blob.blob_id in self.kv_store.graph.nodes:
            self.kv_store.graph.nodes[blob.blob_id].t_invalid = t_inv
            self.kv_store.graph.supersession_edges[blob.blob_id] = str(esm_target_id)
        return True

    def graduate_kv_to_esm(
        self,
        *,
        compile_fn: Callable[
            [KVMemBlob, EpisodicSideMemoryBank],
            tuple[int, ESMLedgerEntry] | Sequence[tuple[int, ESMLedgerEntry]],
        ],
        eval_fn: Callable[[KVMemBlob, EpisodicSideMemoryBank], Mapping[str, float]],
        bench_map: Mapping[str, str] | None = None,
        now: float | None = None,
    ) -> list[GuardGraduationResult]:
        """Graduate hardened `KVMemBlob`s into `ESM` with `Soft-HURT` gating and transactional rollback."""
        if self.esm_bank is None:
            raise RuntimeError("graduate_kv_to_esm requires esm_bank to be provided on LSANLGuardOuterLoop")

        eff_bench_map = self._resolve_bench_map(bench_map)
        if bench_map is not None:
            for k, v in bench_map.items():
                self.bench_map[str(k)] = str(v)

        candidates = self._select_graduation_candidates(now=now)
        results: list[GuardGraduationResult] = []

        for blob in candidates:
            self._cycle_count += 1
            t_now = self._resolve_time(
                now,
                floor_time=max(blob.header.t_last_access, blob.header.t_valid, blob.header.t_created),
            )
            compiled_pairs = _normalize_compiled_entries(compile_fn(blob, self.esm_bank))
            compiled_refs = tuple((int(lyr), str(entry.entry_id)) for lyr, entry in compiled_pairs)
            max_post_leak = max((float(entry.post_leakage) for _, entry in compiled_pairs), default=0.0)

            try:
                raw_cand = eval_fn(blob, self.esm_bank)
            except BaseException as exc:
                rolled_refs = self._rollback_esm_entries(compiled_pairs)
                self._rejected_esm_count += 1
                self._rollback_esm_count += 1
                self.history.append(
                    {
                        "kind": "esm_graduation_exception_rollback",
                        "blob_id": blob.blob_id,
                        "compiled_entries": [[lyr, eid] for lyr, eid in compiled_refs],
                        "rolled_back_entries": [[lyr, eid] for lyr, eid in rolled_refs],
                        "error_type": type(exc).__name__,
                        "timestamp": t_now,
                    }
                )
                raise

            cand_clamped = {str(k): _clamp01(v) for k, v in raw_cand.items()}
            outcome = evaluate_soft_hurt(
                self.baseline_items,
                cand_clamped,
                cfg=self.config.soft_hurt,
                bench_map=eff_bench_map,
            )

            if outcome.promoted:
                target_tag = (
                    "esm:" + ",".join(f"L{lyr}:{eid}" for lyr, eid in compiled_refs)
                    if compiled_refs
                    else f"esm:{blob.blob_id}"
                )
                superseded_ok = self._supersede_graduated_blob(blob, target_tag, now=t_now)
                self._promoted_esm_count += 1
                self._record_promoted_scores(cand_clamped, outcome)

                grad_res = GuardGraduationResult(
                    blob_id=blob.blob_id,
                    outcome=outcome,
                    promoted=True,
                    compiled_entries=compiled_refs,
                    rolled_back_entries=(),
                    superseded_in_kv=superseded_ok,
                    max_post_leakage=max_post_leak,
                    rejection_reasons=(),
                    timestamp=t_now,
                )
                grad_record = {
                    **grad_res.to_dict(),
                    "topic_tags": list(blob.header.topic_tags),
                    "hit_count": int(blob.header.hit_count),
                    "stability": float(blob.header.stability),
                    "salience": float(blob.header.salience),
                    "rule_meta": _sanitize_meta(self.admitted_rule_meta.get(blob.blob_id, {})),
                }
                self.graduated_rules.append(grad_record)
                self.history.append(grad_res.to_dict())
                results.append(grad_res)
            else:
                rolled_refs: tuple[tuple[int, str], ...] = ()
                if self.config.rollback_esm_on_reject and compiled_pairs:
                    rolled_refs = self._rollback_esm_entries(compiled_pairs)
                    self._rollback_esm_count += 1
                self._rejected_esm_count += 1
                self._record_rejected_hurt_ids(cand_clamped, outcome.newly_hurt_ids)

                grad_res = GuardGraduationResult(
                    blob_id=blob.blob_id,
                    outcome=outcome,
                    promoted=False,
                    compiled_entries=compiled_refs,
                    rolled_back_entries=rolled_refs,
                    superseded_in_kv=False,
                    max_post_leakage=max_post_leak,
                    rejection_reasons=outcome.rejection_reasons,
                    timestamp=t_now,
                )
                self.history.append(grad_res.to_dict())
                results.append(grad_res)

        return results

    def _classify_regret_bucket(self, item_id: str) -> str:
        st = self.regret_tracker.states.get(item_id)
        tau = self.config.soft_hurt.pass_threshold
        if item_id in self._hurt_event_counts and (
            st is None or st.last_pass_rate < st.best_pass_rate or st.last_pass_rate < tau
        ):
            return REGRET_NEWLY_HURT
        if st is None:
            return REGRET_UNSOLVED_NOVEL
        drop_eps = float(self.regret_tracker.cfg.regret_drop_eps)
        if (st.ever_passed and st.last_pass_rate < st.best_pass_rate - drop_eps) or (
            st.best_pass_rate >= self.config.soft_hurt.pass_threshold and st.last_pass_rate == 0.0
        ):
            return REGRET_NEWLY_HURT
        if st.last_pass_rate > 0.0:
            return REGRET_PINNED_PASS
        if st.consec_fails >= self.config.chronic_fail_streak:
            return REGRET_UNSOLVED_CHRONIC
        return REGRET_UNSOLVED_NOVEL

    def _regret_summary(self) -> dict[str, Any]:
        bucket_counts: dict[str, int] = {
            REGRET_NEWLY_HURT: 0,
            REGRET_UNSOLVED_CHRONIC: 0,
            REGRET_UNSOLVED_NOVEL: 0,
            REGRET_PINNED_PASS: 0,
        }
        all_ids = sorted(set(self.regret_tracker.states) | set(self._hurt_event_counts))
        ever_passed = 0
        for item_id in all_ids:
            bucket = self._classify_regret_bucket(item_id)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            st = self.regret_tracker.states.get(item_id)
            if st is not None and st.ever_passed:
                ever_passed += 1
        return {
            "n_tracked": len(all_ids),
            "n_ever_passed": ever_passed,
            "n_newly_hurt": bucket_counts[REGRET_NEWLY_HURT],
            "n_chronic_unsolved": bucket_counts[REGRET_UNSOLVED_CHRONIC],
            "n_novel_unsolved": bucket_counts[REGRET_UNSOLVED_NOVEL],
            "n_pinned_pass": bucket_counts[REGRET_PINNED_PASS],
            "bucket_counts": bucket_counts,
        }

    def export_slow_inner_bundle(self, out_dir: str | Path) -> dict[str, Any]:
        """Export contamination-free outer-loop state and prioritized slow-inner distillation queue."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        bench_means: dict[str, float] = {}
        if self.bench_map and self.baseline_items:
            grouped: dict[str, list[float]] = {}
            for item_id, sc in self.baseline_items.items():
                b_name = self.bench_map.get(item_id)
                if b_name is not None:
                    grouped.setdefault(b_name, []).append(sc)
            for b_name in sorted(grouped):
                vals = grouped[b_name]
                bench_means[b_name] = sum(vals) / float(len(vals))

        esm_stats: dict[str, Any] | None = None
        if self.esm_bank is not None:
            active_entries = 0
            total_entries = 0
            for lyr in self.esm_bank.anchor_layers:
                ledger = self.esm_bank.anchor(lyr).ledger
                total_entries += len(ledger)
                active_entries += sum(1 for e in ledger if e.active)
            esm_stats = {
                "anchor_layers": list(self.esm_bank.anchor_layers),
                "shard_bytes": int(self.esm_bank.shard_bytes()),
                "shard_mib": float(self.esm_bank.shard_mib()),
                "active_ledger_entries": active_entries,
                "total_ledger_entries": total_entries,
            }

        engram_stats: dict[str, Any] | None = None
        if self.engram_bank is not None:
            engram_stats = {
                "anchor_layers": list(self.engram_bank.anchor_layers),
                "param_summary": _sanitize_meta(self.engram_bank.param_summary()),
            }

        state_summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "cycle_count": int(self._cycle_count),
            "config": self.config.to_dict(),
            "counts": {
                "promoted_kv": int(self._promoted_kv_count),
                "rejected_kv": int(self._rejected_kv_count),
                "salience_rejected_kv": int(self._salience_rejected_kv_count),
                "promoted_esm": int(self._promoted_esm_count),
                "rejected_esm": int(self._rejected_esm_count),
                "rollback_esm": int(self._rollback_esm_count),
                "graduated_rules": len(self.graduated_rules),
                "history_events": len(self.history),
            },
            "baseline_summary": {
                "n_items": len(self.baseline_items),
                "mean_score": (
                    sum(self.baseline_items.values()) / float(len(self.baseline_items))
                    if self.baseline_items
                    else 0.0
                ),
                "n_pinned_passed": len(self.pinned_passed),
                "bench_means": bench_means,
            },
            "kv_store_stats": self.kv_store.memory_stats(),
            "esm_bank_stats": esm_stats,
            "engram_bank_stats": engram_stats,
            "regret_tracker_summary": self._regret_summary(),
            "graduated_rules": [_sanitize_meta(r) for r in self.graduated_rules],
        }

        queue_records: list[dict[str, Any]] = []

        # 1. Graduated ESM rules -> slow-inner LoRA/Engram consolidation queue
        for rule in self.graduated_rules:
            outcome_d = rule.get("outcome") or {}
            sh_score = float(outcome_d.get("soft_hurt_score", 0.0))
            n_fixed = int(outcome_d.get("n_newly_fixed", 0))
            priority = 2.0 + max(0.0, sh_score) + 0.1 * float(n_fixed)
            queue_records.append(
                {
                    "record_type": "graduated_rule",
                    "item_id": str(rule["blob_id"]),
                    "blob_id": str(rule["blob_id"]),
                    "regret_bucket": REGRET_PINNED_PASS,
                    "priority": float(priority),
                    "compiled_entries": rule.get("compiled_entries", []),
                    "max_post_leakage": float(rule.get("max_post_leakage", 0.0)),
                    "soft_hurt_score": sh_score,
                    "topic_tags": [str(t) for t in (rule.get("topic_tags") or ())],
                    "rule_meta": _sanitize_meta(rule.get("rule_meta", {})),
                }
            )

        # 2. Newly hurt & chronic unsolved items -> slow-inner targeted RL / curriculum sculpting queue
        emitted_hurt: set[str] = set()
        candidate_ids = sorted(set(self.regret_tracker.states) | set(self._hurt_event_counts))
        for item_id in candidate_ids:
            st = self.regret_tracker.states.get(item_id)
            hurt_cnt = int(self._hurt_event_counts.get(item_id, 0))
            bucket = self._classify_regret_bucket(item_id)
            if hurt_cnt > 0 or bucket == REGRET_NEWLY_HURT:
                drop_mag = max(0.0, (st.best_pass_rate - st.last_pass_rate)) if st is not None else 1.0
                priority = 3.0 + self.config.soft_hurt.lambda_hurt * drop_mag + 0.25 * float(max(1, hurt_cnt))
                emitted_hurt.add(item_id)
                queue_records.append(
                    {
                        "record_type": "newly_hurt_item",
                        "item_id": item_id,
                        "benchmark": self.bench_map.get(item_id),
                        "regret_bucket": REGRET_NEWLY_HURT,
                        "priority": float(priority),
                        "hurt_events": hurt_cnt,
                        "baseline_score": (
                            float(self.baseline_items[item_id]) if item_id in self.baseline_items else None
                        ),
                        "best_pass_rate": float(st.best_pass_rate) if st is not None else None,
                        "last_pass_rate": float(st.last_pass_rate) if st is not None else None,
                    }
                )
            elif (
                st is not None
                and (not st.ever_passed)
                and st.consec_fails >= self.config.chronic_fail_streak
                and item_id not in emitted_hurt
            ):
                priority = 1.0 + 0.1 * float(min(st.consec_fails, 10))
                queue_records.append(
                    {
                        "record_type": "chronic_unsolved_item",
                        "item_id": item_id,
                        "benchmark": self.bench_map.get(item_id),
                        "regret_bucket": REGRET_UNSOLVED_CHRONIC,
                        "priority": float(priority),
                        "consec_fails": int(st.consec_fails),
                        "seen": int(st.seen),
                        "baseline_score": (
                            float(self.baseline_items[item_id]) if item_id in self.baseline_items else None
                        ),
                    }
                )

        queue_records.sort(key=lambda r: (-float(r["priority"]), str(r["record_type"]), str(r["item_id"])))

        state_path = out_path / "guard_outer_state.json"
        queue_path = out_path / "slow_inner_distill_queue.jsonl"

        state_path.write_text(json.dumps(state_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with queue_path.open("w", encoding="utf-8") as qf:
            for rec in queue_records:
                qf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return {
            "out_dir": str(out_path),
            "state_path": str(state_path),
            "queue_path": str(queue_path),
            "n_queue_records": len(queue_records),
            "state": state_summary,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize controller state into a JSON-compatible dictionary for cross-session persistence."""
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "baseline_items": {str(k): float(v) for k, v in sorted(self.baseline_items.items())},
            "bench_map": {str(k): str(v) for k, v in sorted(self.bench_map.items())},
            "pinned_passed": sorted(self.pinned_passed),
            "regret_tracker": self.regret_tracker.state_dict(),
            "history": [_sanitize_meta(h) for h in self.history],
            "graduated_rules": [_sanitize_meta(r) for r in self.graduated_rules],
            "admitted_rule_meta": {
                str(k): _sanitize_meta(v) for k, v in sorted(self.admitted_rule_meta.items())
            },
            "hurt_event_counts": {str(k): int(v) for k, v in sorted(self._hurt_event_counts.items())},
            "counts": {
                "cycle_count": int(self._cycle_count),
                "promoted_kv": int(self._promoted_kv_count),
                "rejected_kv": int(self._rejected_kv_count),
                "salience_rejected_kv": int(self._salience_rejected_kv_count),
                "promoted_esm": int(self._promoted_esm_count),
                "rejected_esm": int(self._rejected_esm_count),
                "rollback_esm": int(self._rollback_esm_count),
            },
            "clock": float(self._clock),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore controller state from a dictionary produced by `state_dict()`."""
        if "config" in state and isinstance(state["config"], Mapping):
            self.config = GuardOuterConfig.from_dict(state["config"])
        self.baseline_items = {
            str(k): _clamp01(v) for k, v in (state.get("baseline_items") or {}).items()
        }
        self.bench_map = {str(k): str(v) for k, v in (state.get("bench_map") or {}).items()}
        self.pinned_passed = {str(k) for k in (state.get("pinned_passed") or ())}
        if isinstance(state.get("regret_tracker"), Mapping):
            self.regret_tracker.load_state_dict(dict(state["regret_tracker"]))
        self.history = [dict(h) for h in (state.get("history") or ()) if isinstance(h, Mapping)]
        self.graduated_rules = [
            dict(r) for r in (state.get("graduated_rules") or ()) if isinstance(r, Mapping)
        ]
        self.admitted_rule_meta = {
            str(k): dict(v)
            for k, v in (state.get("admitted_rule_meta") or {}).items()
            if isinstance(v, Mapping)
        }
        self._hurt_event_counts = {
            str(k): int(v) for k, v in (state.get("hurt_event_counts") or {}).items()
        }
        counts = state.get("counts") or {}
        self._cycle_count = int(counts.get("cycle_count", 0))
        self._promoted_kv_count = int(counts.get("promoted_kv", 0))
        self._rejected_kv_count = int(counts.get("rejected_kv", 0))
        self._salience_rejected_kv_count = int(counts.get("salience_rejected_kv", 0))
        self._promoted_esm_count = int(counts.get("promoted_esm", 0))
        self._rejected_esm_count = int(counts.get("rejected_esm", 0))
        self._rollback_esm_count = int(counts.get("rollback_esm", 0))
        self._clock = float(state.get("clock", 0.0))


__all__ = [
    "SCHEMA_VERSION",
    "GuardGraduationResult",
    "GuardOuterConfig",
    "GuardProposalResult",
    "LSANLGuardOuterLoop",
    "SoftHurtGateConfig",
    "SoftHurtOutcome",
    "evaluate_soft_hurt",
]
