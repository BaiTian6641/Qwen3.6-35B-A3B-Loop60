"""Fast pure-CPU verification suite for Zero-Training Kill-Gate `K0` Evaluator Harness.

Covers (`tests/test_kill_gate_k0.py`, zero network, zero GPU):
  1. Geometric + Sink Error reproduction & recovery on a controlled partial-RoPE
     (`rotary_dim=64, head_dim=256`) multi-session attention probe (`pos=0..63`, huge sinks
     at `[0:4)`, planted target at `pos=40`, mounted at `virtual_offset=10000`), verifying:
       * Arm `b` (Naive Splice) fails due to stale RoPE + duplicate sinks,
       * Arm `c` (`Pre-RoPE + sink-strip + H64-INT4 + virtual mount`) closely matches Arm `a`
         (Contiguous Oracle) while keeping `[64:256)` NoPE bitwise orthogonal,
       * Arm `d` (`+ 5% CacheBlend selective refresh`) strictly reduces remaining KV, attention,
         and NLL discrepancy over Arm `c` while refreshing `<= ceil(0.05 * T)` tokens.
  2. Automated falsification decision tree (`decide_k0_verdict()`) exact `GREEN`, `YELLOW`,
     and `RED` verdicts + rationales on boundary ratios (`0.93`, `0.85`, `0.60`, `c=0.80/d=0.94`,
     plus NLL-gap & naive-separation edge cases).
  3. Contradictory/stale-memory `SleepGate` integration (`SplitHorizonKVBlobStore.supersede()`)
     removing superseded Session-1 facts so queries attend cleanly to the updated Session-2 blob.
  4. Suite runner (`run_k0_suite`), cache extraction/injection across HF/Toy cache layouts, and
     end-to-end `HFCheckpointK0Evaluator` execution on CPU `ToyForCausalLM`.
  5. Deterministic JSON serialization, `format_k0_markdown()`, dataset loader, and `--smoke-cpu`
     CLI exit code `0`.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from qloop.eval.kill_gate_k0 import (
    CANONICAL_ARMS,
    HFCheckpointK0Evaluator,
    K0ArmMetrics,
    K0Config,
    K0EpisodeResult,
    K0Example,
    K0Report,
    _clone_linear_states_into,
    _clone_state_value,
    build_controlled_partial_rope_probe,
    build_corrected_partition_b_kv,
    decide_k0_verdict,
    extract_layer_kv_pairs,
    format_k0_markdown,
    inject_layer_kv_pairs,
    load_examples_from_file,
    main,
    run_k0_attention_probe,
    run_k0_suite,
    run_k0_supersession_probe,
    run_smoke_cpu,
    selective_cacheblend_refresh,
    splice_naive_kv,
)
from qloop.eval.testing import ByteTokenizer
from qloop.infer.engine import QuantizedHybridCache
from qloop.infer.hf_custom_modeling import Qwen36LoopHSSDConfig, Qwen36LoopHSSDForCausalLM
from qloop.model.toy import ToyCausalLMOutput, ToyConfig, ToyForCausalLM, ToyHybridCache


# ==============================================================================
# 1. Geometric + Sink Error Reproduction & Recovery (Partial-RoPE 64/256 Probe)
# ==============================================================================
def test_k0_geometric_and_sink_error_reproduction_and_recovery():
    cfg = K0Config(
        rotary_dim=64,
        rope_theta=10_000_000.0,
        sink_tokens=4,
        boundary_header_tokens=4,
        refresh_frac=0.05,
        virtual_offset=10_000,
    )
    probe = build_controlled_partial_rope_probe(
        num_layers=2,
        num_kv_heads=2,
        num_q_heads=4,  # GQA (2x query heads per KV head)
        head_dim=256,
        rotary_dim=64,
        session_len=64,
        target_session_pos=40,
        sink_tokens=4,
        live_sink_len=4,
        live_body_len=12,
        virtual_offset=10_000,
        rope_theta=10_000_000.0,
        dynamic_drift_scale=0.08,
        seed=2026,
    )
    res = run_k0_attention_probe(
        probe["q_pre_rope"],
        probe["q_positions"],
        probe["session_k_post_rope"],
        probe["session_v"],
        probe["session_positions"],
        probe["live_k_post_rope"],
        probe["live_v"],
        live_sink_len=probe["live_sink_len"],
        virtual_offset=probe["virtual_offset"],
        cfg=cfg,
        oracle_session_k_post_rope=probe["oracle_session_k_post_rope"],
        oracle_session_v=probe["oracle_session_v"],
        target_session_pos=probe["target_session_pos"],
        candidate_prototypes=probe["candidate_prototypes"],
        target_class=probe["target_class"],
    )

    arm_a = res.arms["arm_a"]
    arm_b = res.arms["arm_b"]
    arm_c = res.arms["arm_c"]
    arm_d = res.arms["arm_d"]

    # Arm (a) Oracle hits target cleanly
    assert arm_a.exact_match == 1.0
    assert arm_a.target_attn_mass > 0.90

    # Arm (b) Naive Splice FAILS due to stale RoPE + duplicate original sinks at [0:4)
    assert arm_b.exact_match == 0.0
    assert arm_b.target_attn_mass < 0.05
    assert arm_b.output_rel_l2_vs_oracle > 0.80
    assert arm_b.answer_nll > arm_a.answer_nll + 2.0

    # Arm (c) Pre-RoPE + sink-strip + H64-INT4 + virtual mount closely matches Oracle Arm (a)
    assert arm_c.exact_match == 1.0
    assert arm_c.target_attn_mass > 0.90
    assert arm_c.output_cosine_vs_oracle > 0.98
    assert arm_c.output_rel_l2_vs_oracle < 0.15
    assert arm_c.kv_bytes_mounted < arm_a.kv_bytes_mounted
    assert res.verdict.verdict == "GREEN"

    # Arm (d) +5% CacheBlend selective refresh strictly reduces discrepancy while refreshing <= ceil(0.05 * T)
    t_kept = 64 - cfg.sink_tokens  # 60 tokens
    max_refresh_budget = math.ceil(0.05 * t_kept)  # 3 tokens
    assert 0 < arm_d.refreshed_tokens <= max_refresh_budget
    assert arm_d.kv_l2_vs_oracle < arm_c.kv_l2_vs_oracle
    assert arm_d.output_rel_l2_vs_oracle < arm_c.output_rel_l2_vs_oracle
    assert arm_d.answer_nll < arm_c.answer_nll

    # Pure H64-INT4 regime (zero extra synthetic drift): Arm (c) is near-exact AND Arm (d) still improves!
    probe_pure = build_controlled_partial_rope_probe(
        head_dim=256,
        rotary_dim=64,
        session_len=64,
        target_session_pos=40,
        virtual_offset=10_000,
        dynamic_drift_scale=0.0,
        seed=2026,
    )
    res_pure = run_k0_attention_probe(
        probe_pure["q_pre_rope"],
        probe_pure["q_positions"],
        probe_pure["session_k_post_rope"],
        probe_pure["session_v"],
        probe_pure["session_positions"],
        probe_pure["live_k_post_rope"],
        probe_pure["live_v"],
        live_sink_len=probe_pure["live_sink_len"],
        virtual_offset=probe_pure["virtual_offset"],
        cfg=cfg,
        target_session_pos=probe_pure["target_session_pos"],
        candidate_prototypes=probe_pure["candidate_prototypes"],
        target_class=probe_pure["target_class"],
    )
    assert res_pure.arms["arm_b"].exact_match == 0.0
    assert res_pure.arms["arm_c"].exact_match == 1.0
    assert res_pure.arms["arm_c"].output_cosine_vs_oracle > 0.995
    assert 0 < res_pure.arms["arm_d"].refreshed_tokens <= max_refresh_budget
    assert res_pure.arms["arm_d"].kv_l2_vs_oracle < res_pure.arms["arm_c"].kv_l2_vs_oracle
    assert res_pure.arms["arm_d"].output_rel_l2_vs_oracle < res_pure.arms["arm_c"].output_rel_l2_vs_oracle


def test_k0_partition_b_nope_orthogonality_rerope_clamp_and_boundary_header():
    torch.manual_seed(7)
    cfg = K0Config(rotary_dim=64, sink_tokens=4, boundary_header_tokens=4, rope_theta=10_000_000.0)
    k_sess = torch.randn(2, 2, 36, 256, dtype=torch.float32)
    v_sess = torch.randn_like(k_sess)
    pos = torch.arange(0, 36, dtype=torch.long)

    # Mount at two distant virtual offsets AND with extreme 3M ReRoPE clamping
    mnt_10k = build_corrected_partition_b_kv(k_sess, v_sess, pos, live_sink_len=4, virtual_offset=10_000, cfg=cfg)
    mnt_200k = build_corrected_partition_b_kv(
        k_sess,
        v_sess,
        pos,
        live_sink_len=4,
        virtual_offset=200_000,
        cfg=cfg,
        clamp_relative_to_q_pos=3_000_000.0,
    )
    # 2-tuple unpacking works seamlessly
    k_unpacked, v_unpacked = mnt_10k
    assert torch.equal(k_unpacked, mnt_10k.k)
    assert torch.equal(v_unpacked, mnt_10k.v)

    # `[64:256)` NoPE subspace and Values are strictly bit-identical across virtual positions & clamping!
    assert torch.equal(mnt_10k.k[..., 64:], mnt_200k.k[..., 64:])
    assert torch.equal(mnt_10k.v, mnt_200k.v)
    assert not torch.allclose(mnt_10k.k[..., :64], mnt_200k.k[..., :64])
    assert int(mnt_10k.virtual_positions[0].item()) == 10_000

    # Mounting with virtual_offset < live_sink_len automatically avoids colliding with live sinks `[0:live_sink_len)`
    mnt_zero = build_corrected_partition_b_kv(k_sess, v_sess, pos, live_sink_len=4, virtual_offset=0, cfg=cfg)
    assert int(mnt_zero.virtual_positions[0].item()) == 4

    # Explicit 4-token TurboRAG Boundary & Abstraction Header prepending
    hdr_k = torch.randn(2, 2, 4, 256, dtype=torch.float32)
    hdr_v = torch.randn(2, 2, 4, 256, dtype=torch.float32)
    mnt_hdr = build_corrected_partition_b_kv(
        k_sess,
        v_sess,
        pos,
        live_sink_len=4,
        virtual_offset=10_000,
        cfg=cfg,
        boundary_header_k=hdr_k,
        boundary_header_v=hdr_v,
    )
    assert mnt_hdr.boundary_header_tokens == 4
    assert mnt_hdr.k.shape[-2] == (36 - 4) + 4

    # Test selective_cacheblend_refresh modes & 0% no-op budget
    k_ora = mnt_10k.k + 0.05 * torch.randn_like(mnt_10k.k)
    v_ora = mnt_10k.v + 0.05 * torch.randn_like(mnt_10k.v)
    for mode in ("hkvd", "first_layer_hkvd", "value_variance"):
        r = selective_cacheblend_refresh(mnt_10k.k, mnt_10k.v, k_ora, v_ora, refresh_frac=0.05, score_mode=mode)
        assert r.refreshed_tokens == math.ceil(0.05 * mnt_10k.k.shape[-2])
        assert r.post_refresh_kv_l2 < r.pre_refresh_kv_l2
        rk, rv = r  # 2-tuple unpacking
        assert torch.equal(rk, r.k) and torch.equal(rv, r.v)

    r_zero = selective_cacheblend_refresh(mnt_10k.k, mnt_10k.v, k_ora, v_ora, refresh_frac=0.0)
    assert r_zero.refreshed_tokens == 0
    assert torch.equal(r_zero.k, mnt_10k.k)

    # 5D `[B, L, H, T, D]` `first_layer_hkvd` MUST slice layer 0 (`[:, 0]`), NOT batch 0 (`[0]`)!
    k5_mnt = torch.zeros(2, 3, 2, 20, 64, dtype=torch.float32)
    v5_mnt = torch.zeros_like(k5_mnt)
    k5_ora = torch.zeros_like(k5_mnt)
    v5_ora = torch.zeros_like(k5_mnt)
    # Batch 0, Layer 0: small drift at t=5; Batch 1, Layer 0: large drift at t=11;
    # Batch 0, Layer 1: huge drift at t=17 (must be ignored by `first_layer_hkvd`!).
    k5_ora[0, 0, :, 5, :] = 0.5
    k5_ora[1, 0, :, 11, :] = 4.0
    k5_ora[0, 1, :, 17, :] = 100.0
    r5 = selective_cacheblend_refresh(k5_mnt, v5_mnt, k5_ora, v5_ora, refresh_frac=0.05, score_mode="first_layer_hkvd")
    assert r5.refreshed_tokens == 1
    assert r5.refreshed_indices.tolist() == [11]


# ==============================================================================
# 2. Decision Tree Tests (`GREEN`, `YELLOW`, `RED` + Boundary Edge Cases)
# ==============================================================================
def _make_report_for_ratios(
    *,
    r_b: float = 0.25,
    r_c: float,
    r_d: float,
    nll_a: float = 0.20,
    nll_b: float = 2.40,
    nll_c: float = 0.22,
    nll_d: float = 0.21,
    cfg: K0Config | None = None,
) -> K0Report:
    eff_cfg = K0Config() if cfg is None else cfg
    arms = {
        "arm_a": K0ArmMetrics("arm_a", nll_a, 1.0, 1.0, 1.0, 0.0, 1_000_000, 0, 10.0, 20),
        "arm_b": K0ArmMetrics("arm_b", nll_b, r_b, r_b, r_b, nll_b - nll_a, 1_000_000, 0, 10.0, 20),
        "arm_c": K0ArmMetrics("arm_c", nll_c, r_c, r_c, r_c, nll_c - nll_a, 250_000, 0, 12.0, 20),
        "arm_d": K0ArmMetrics("arm_d", nll_d, r_d, r_d, r_d, nll_d - nll_a, 250_000, 15, 14.0, 20),
    }
    return K0Report(arms=arms, config=eff_cfg, n_examples=20)


def test_k0_decision_tree_boundary_verdicts():
    # Case 1: GREEN — Arm c = 0.93 >= 0.92, beats Arm b = 0.25, NLL gap = +0.02 <= 0.05
    rep_green = _make_report_for_ratios(r_b=0.25, r_c=0.93, r_d=0.96, nll_a=0.20, nll_c=0.22, nll_d=0.205)
    v_green = decide_k0_verdict(rep_green)
    assert v_green.verdict == "GREEN"
    assert v_green.sub_verdict == "GREEN"
    assert rep_green.verdict == "GREEN"
    assert "93.0%" in v_green.rationale
    assert "training-free" in v_green.recommended_action.lower()

    # Case 2: YELLOW-B — Arm c = 0.85 in [0.75, 0.92) and Arm d = 0.89 < 0.92
    rep_yellow_b = _make_report_for_ratios(r_b=0.25, r_c=0.85, r_d=0.89, nll_a=0.20, nll_c=0.28, nll_d=0.24)
    v_yb = decide_k0_verdict(rep_yellow_b)
    assert v_yb.verdict == "YELLOW"
    assert v_yb.sub_verdict == "YELLOW_B_BLOCK_ATTN_CALIB"
    assert "85.0%" in v_yb.rationale
    assert "block-attention" in v_yb.recommended_action.lower()

    # Case 3: YELLOW-A — Arm c = 0.80, Arm d = 0.94 (+5% CacheBlend crosses >= 0.92!)
    rep_yellow_a = _make_report_for_ratios(r_b=0.20, r_c=0.80, r_d=0.94, nll_a=0.20, nll_c=0.31, nll_d=0.22)
    v_ya = decide_k0_verdict(rep_yellow_a)
    assert v_ya.verdict == "YELLOW"
    assert v_ya.sub_verdict == "YELLOW_A_SELECTIVE_REFRESH"
    assert "80.0%" in v_ya.rationale and "94.0%" in v_ya.rationale
    assert "cacheblend" in v_ya.recommended_action.lower()

    # Case 4: RED — Both Arm c = 0.60 and Arm d = 0.68 strictly below 0.75 -> KILL & pivot to ESM
    rep_red = _make_report_for_ratios(r_b=0.20, r_c=0.60, r_d=0.68, nll_a=0.20, nll_c=0.85, nll_d=0.72)
    v_red = decide_k0_verdict(rep_red)
    assert v_red.verdict == "RED"
    assert v_red.sub_verdict == "RED_PIVOT_ESM"
    assert "60.0%" in v_red.rationale and "68.0%" in v_red.rationale
    assert "esm" in v_red.recommended_action.lower()

    # Edge Case 5: Arm c = 0.93 accuracy but NLL gap = +0.12 > 0.05 -> cannot pass GREEN blindly!
    rep_nll_fail = _make_report_for_ratios(r_b=0.25, r_c=0.93, r_d=0.91, nll_a=0.20, nll_c=0.32, nll_d=0.27)
    v_nll = decide_k0_verdict(rep_nll_fail)
    assert v_nll.verdict == "YELLOW"
    assert "nll gap" in v_nll.rationale.lower()

    # Edge Case 6: Arm c = 0.94 does NOT beat naive Arm b = 0.95 -> YELLOW warning (probe didn't separate)
    rep_no_sep = _make_report_for_ratios(r_b=0.95, r_c=0.94, r_d=0.91, nll_a=0.20, nll_b=0.21, nll_c=0.22, nll_d=0.22)
    v_nosep = decide_k0_verdict(rep_no_sep)
    assert v_nosep.verdict == "YELLOW"
    assert "did not beat" in v_nosep.rationale.lower()

    # Edge Case 7 (Blocker 4C): YELLOW_A requires BOTH `nll_ok_d` AND `beats_naive_d`
    rep_ya_nll_fail = _make_report_for_ratios(r_b=0.20, r_c=0.80, r_d=0.95, nll_a=0.20, nll_c=0.35, nll_d=1.50)
    v_ya_nll = decide_k0_verdict(rep_ya_nll_fail)
    assert v_ya_nll.verdict == "YELLOW"
    assert v_ya_nll.sub_verdict == "YELLOW_B_BLOCK_ATTN_CALIB"
    assert "arm d nll gap" in v_ya_nll.rationale.lower()

    rep_ya_naive_fail = _make_report_for_ratios(
        r_b=0.98, r_c=0.95, r_d=0.96, nll_a=0.20, nll_b=0.205, nll_c=0.21, nll_d=0.21
    )
    v_ya_naive = decide_k0_verdict(rep_ya_naive_fail)
    assert v_ya_naive.verdict == "YELLOW"
    assert v_ya_naive.sub_verdict == "YELLOW_B_BLOCK_ATTN_CALIB"
    assert "arm d" in v_ya_naive.rationale.lower() and "did not beat" in v_ya_naive.rationale.lower()

    # Edge Case 8 (Blocker 4C): Degenerate 0.0-recall Oracle MUST NEVER pass GREEN or YELLOW_A!
    zero_ex = [K0Example("zero_1", ["s1"], "q1", "a1")]

    def all_zero_eval(ex: K0Example, arm_name: str, cfg: K0Config) -> K0EpisodeResult:
        nlls = {"arm_a": 5.00, "arm_b": 5.10, "arm_c": 5.02, "arm_d": 5.01}
        return K0EpisodeResult(arm_name, ex.example_id, nlls[arm_name], 0.0, 0.0)

    rep_zero = run_k0_suite(zero_ex, all_zero_eval, K0Config())
    assert rep_zero.verdict == "RED"
    assert rep_zero.sub_verdict == "INVALID_ORACLE"
    for arm_k in CANONICAL_ARMS:
        assert rep_zero.arms[arm_k].oracle_retention_ratio == 0.0
    assert "0.0%" in rep_zero.rationale

    # Config validation checks
    assert K0Config.from_dict({"green_threshold": 0.90, "refresh_frac": 0.10}).green_threshold == 0.90
    with pytest.raises(ValueError, match="unknown"):
        K0Config.from_dict({"not_a_k0_key": 123})
    with pytest.raises(ValueError, match="yellow_threshold"):
        K0Config(green_threshold=0.80, yellow_threshold=0.90)


# ==============================================================================
# 3. Stale/Contradictory Memory Episode (`SplitHorizonKVBlobStore.supersede()`)
# ==============================================================================
def test_k0_supersession_removes_stale_session1_facts():
    cfg = K0Config(rotary_dim=64, sink_tokens=4, rope_theta=10_000_000.0, virtual_offset=10_000)
    sg = run_k0_supersession_probe(cfg=cfg, seed=2027)

    # SleepGate tombstone excluded superseded Session-1 blob and kept updated Session-2 blob active
    assert sg.superseded_blob_id not in sg.active_blob_ids
    assert sg.updated_blob_id in sg.active_blob_ids
    assert sg.store_stats["tombstoned_blobs"] == 1
    assert sg.store_stats["active_blobs"] == 1

    # Without SleepGate: proactive interference from stale Session-1 hijacks attention & prediction!
    assert sg.naive_stale_attn_mass > sg.naive_updated_attn_mass
    assert sg.naive_predicted_class == sg.stale_class
    assert sg.naive_predicted_class != sg.updated_class

    # With SplitHorizonKVBlobStore.supersede(): query attends cleanly to updated Session-2 blob!
    assert sg.sleepgate_stale_attn_mass == 0.0
    assert sg.sleepgate_updated_attn_mass > 0.90
    assert sg.sleepgate_predicted_class == sg.updated_class
    assert sg.sleepgate_nll < sg.naive_nll


# ==============================================================================
# 4. Suite Runner, Cache Extraction/Injection & ToyModel HF-Interface E2E
# ==============================================================================
def test_k0_suite_runner_cache_extraction_and_toy_model_evaluator():
    # 4a. Test `extract_layer_kv_pairs` + `inject_layer_kv_pairs` across all cache conventions
    k0 = torch.randn(1, 2, 10, 64)
    v0 = torch.randn(1, 2, 10, 64)
    k1 = torch.randn(1, 2, 10, 64)
    v1 = torch.randn(1, 2, 10, 64)

    # Hybrid `.key_cache`/`.value_cache` with `None` for linear-attention layers
    hybrid_cache = SimpleNamespace(
        key_cache=[None, k0, None, k1],
        value_cache=[None, v0, None, v1],
        recurrent_states=["keep_me_intact", None, "keep_me_2", None],
    )
    ks, vs, idxs = extract_layer_kv_pairs(SimpleNamespace(cache=hybrid_cache))
    assert idxs == [1, 3]
    assert ks.shape == (2, 2, 10, 64)
    assert torch.equal(ks[0], k0[0]) and torch.equal(vs[1], v1[0])

    # Inject updated full-attention KV while leaving linear-attention recurrent_states untouched
    target_cache = SimpleNamespace(
        key_cache=[None, None, None, None],
        value_cache=[None, None, None, None],
        recurrent_states=["live_gdn_state_0", None, "live_gdn_state_2", None],
    )
    inject_layer_kv_pairs(target_cache, ks * 2.0, vs * 3.0, idxs)
    assert target_cache.key_cache[0] is None
    assert torch.equal(target_cache.key_cache[1], k0 * 2.0)
    assert target_cache.recurrent_states == ["live_gdn_state_0", None, "live_gdn_state_2", None]

    # `transformers >= 5.x` `.layers[i].keys/.values` layout (pre-populated AND empty dynamic `.update()`)
    hf5_cache = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=None, values=None),
            SimpleNamespace(keys=k0, values=v0),
        ]
    )
    ks5, vs5, idxs5 = extract_layer_kv_pairs(SimpleNamespace(past_key_values=hf5_cache))
    assert idxs5 == [1] and torch.equal(ks5[0], k0[0]) and torch.equal(vs5[0], v0[0])

    class _DynamicV5Stub:
        def __init__(self) -> None:
            self.layers: list[SimpleNamespace] = []

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            while len(self.layers) <= layer_idx:
                self.layers.append(SimpleNamespace(keys=None, values=None))
            self.layers[layer_idx].keys = k
            self.layers[layer_idx].values = v
            return k, v

    dyn5 = _DynamicV5Stub()
    inject_layer_kv_pairs(dyn5, ks, vs, idxs)
    ks_dyn5, vs_dyn5, idxs_dyn5 = extract_layer_kv_pairs(dyn5)
    assert idxs_dyn5 == [1, 3] and torch.equal(ks_dyn5, ks) and torch.equal(vs_dyn5, vs)

    # Legacy tuple & dict caches
    ks_t, _, idxs_t = extract_layer_kv_pairs((None, (k0, v0), (k1, v1)))
    assert idxs_t == [1, 2] and ks_t.shape == (2, 2, 10, 64)
    ks_d, _, idxs_d = extract_layer_kv_pairs({3: (k0, v0), 7: (k1, v1)})
    assert idxs_d == [3, 7] and ks_d.shape == (2, 2, 10, 64)

    # `splice_naive_kv` basic shape & concatenation contract
    spliced_k, spliced_v = splice_naive_kv(ks, vs, ks * 0.5, vs * 0.5)
    assert spliced_k.shape == (2, 2, 20, 64)
    assert torch.equal(spliced_k[..., :10, :], ks)
    assert torch.equal(spliced_v[..., 10:, :], vs * 0.5)

    # 4b. Test 3-arg per-arm `run_k0_suite(examples, evaluator_fn, cfg)`
    examples = [
        K0Example("e1", ["s1"], "q1", "a1", category="multi_hop"),
        K0Example("e2", ["s1", "s2"], "q2", "a2", category="contradictory_update", superseded_session_indices=(0,)),
    ]

    def per_arm_eval(ex: K0Example, arm_name: str, cfg: K0Config) -> K0EpisodeResult:
        scores = {
            "arm_a": (0.10, 1.0, 1.0, 0),
            "arm_b": (2.50, 0.0, 0.0, 0),
            "arm_c": (0.12, 1.0, 1.0, 0),
            "arm_d": (0.11, 1.0, 1.0, 2),
        }
        nll, acc, em, ref = scores[arm_name]
        return K0EpisodeResult(arm_name, ex.example_id, nll, acc, em, kv_bytes_mounted=1024, refreshed_tokens=ref)

    rep_suite = run_k0_suite(examples, per_arm_eval, K0Config())
    assert rep_suite.verdict == "GREEN"
    assert rep_suite.n_examples == 2
    assert set(rep_suite.arms) == set(CANONICAL_ARMS)
    assert set(rep_suite.per_category) == {"multi_hop", "contradictory_update"}
    assert rep_suite.arms["arm_d"].refreshed_tokens == 4

    # 4c. End-to-end `HFCheckpointK0Evaluator` on CPU `ToyForCausalLM` (zero GPU, zero network!)
    torch.manual_seed(0)
    toy_cfg = ToyConfig(
        vocab_size=260,
        hidden_size=32,
        num_hidden_layers=8,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.5,  # rotary_dim = 64, head_dim = 128 (divisible by H64!)
        rope_theta=10_000.0,
    )
    toy_model = ToyForCausalLM(toy_cfg).eval()
    tok = ByteTokenizer()
    hf_eval = HFCheckpointK0Evaluator(toy_model, tok, device="cpu", live_sink_len=2)
    k0_toy_cfg = K0Config(rotary_dim=64, rope_theta=10_000.0, sink_tokens=2, refresh_frac=0.10)
    toy_ex = K0Example(
        example_id="toy_e2e",
        sessions=["alpha beta gamma delta epsilon zeta"],
        live_prompt="query alpha: ",
        answer="ok",
    )
    toy_report = run_k0_suite([toy_ex], hf_eval, k0_toy_cfg)
    assert set(toy_report.arms) == set(CANONICAL_ARMS)
    assert toy_report.verdict in ("GREEN", "YELLOW", "RED")
    assert toy_report.arms["arm_c"].kv_bytes_mounted > 0
    assert toy_report.arms["arm_d"].refreshed_tokens >= 1


# ==============================================================================
# 5. Report Serialization, Markdown Formatter & `--smoke-cpu` CLI Smoke Test
# ==============================================================================
def test_k0_report_json_markdown_and_cli_smoke_cpu(tmp_path: Path):
    report = run_smoke_cpu()
    assert report.verdict == "GREEN"
    assert report.n_examples == 4
    assert report.arms["arm_b"].oracle_retention_ratio == 0.0
    assert report.arms["arm_c"].oracle_retention_ratio == 1.0
    assert report.arms["arm_d"].oracle_retention_ratio == 1.0
    assert report.arms["arm_d"].answer_nll <= report.arms["arm_c"].answer_nll

    # Deterministic JSON serialization round-trip
    js1 = report.to_json()
    js2 = report.to_json()
    assert js1 == js2
    parsed = json.loads(js1)
    assert parsed["verdict"] == "GREEN"
    assert set(parsed["arms"].keys()) == set(CANONICAL_ARMS)
    assert "contradictory_update" in parsed["per_category"]

    # Concise Markdown formatting check
    md = format_k0_markdown(report)
    assert "# Kill-Gate `K0` Report" in md
    assert "`GREEN`" in md
    assert "Arm (a) Contiguous Oracle" in md
    assert "Arm (b) Naive Spliced KV" in md
    assert "Arm (c) Pre-RoPE + Sink-Strip + H64-INT4" in md
    assert "Arm (d) Arm (c) + CacheBlend Refresh" in md
    assert "## Per-Category Breakdown" in md

    # Dataset loader (JSON array + JSONL)
    ds_jsonl = tmp_path / "k0_ds.jsonl"
    ds_jsonl.write_text(
        json.dumps({"example_id": "j1", "sessions": ["s1"], "live_prompt": "q1", "answer": "a1"}) + "\n",
        encoding="utf-8",
    )
    loaded = load_examples_from_file(ds_jsonl)
    assert len(loaded) == 1 and loaded[0].example_id == "j1"

    # In-process `main(["--smoke-cpu", ...])` check (verifying both `--output-*` and `--out-*` aliases!)
    out_json = tmp_path / "k0_report.json"
    out_md = tmp_path / "k0_report.md"
    rc = main(["--smoke-cpu", "--output-json", str(out_json), "--output-md", str(out_md)])
    assert rc == 0
    assert out_json.exists() and json.loads(out_json.read_text(encoding="utf-8"))["verdict"] == "GREEN"
    assert out_md.exists() and "Kill-Gate `K0` Report" in out_md.read_text(encoding="utf-8")

    out_json_alias = tmp_path / "k0_report_alias.json"
    out_md_alias = tmp_path / "k0_report_alias.md"
    rc_alias = main(["--smoke-cpu", "--out-json", str(out_json_alias), "--out-md", str(out_md_alias)])
    assert rc_alias == 0
    assert out_json_alias.exists() and json.loads(out_json_alias.read_text(encoding="utf-8"))["verdict"] == "GREEN"
    assert out_md_alias.exists() and "Kill-Gate `K0` Report" in out_md_alias.read_text(encoding="utf-8")

    # Subprocess CLI entrypoint check (`python3 -m qloop.eval.kill_gate_k0 --smoke-cpu`)
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
    proc = subprocess.run(
        [sys.executable, "-m", "qloop.eval.kill_gate_k0", "--smoke-cpu"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "Kill-Gate `K0` Report" in proc.stdout
    assert "`GREEN`" in proc.stdout


# ==============================================================================
# 6. Audit Regressions: QuantizedHybridCache, TF 5.24 DynamicLayer, LoopHSSD & PosAware
# ==============================================================================
@pytest.mark.parametrize("mode", ["fp8", "hadamard_int4", "bf16"])
def test_k0_quantized_hybrid_cache_fp8_and_hadamard_int4_extract_inject_update(mode: str):
    torch.manual_seed(11)
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    qc = QuantizedHybridCache(num_layers=8, layer_types=layer_types, mode=mode)
    k3 = torch.randn(1, 2, 12, 128, dtype=torch.float32)
    v3 = torch.randn(1, 2, 12, 128, dtype=torch.float32)
    k7 = torch.randn(1, 2, 12, 128, dtype=torch.float32)
    v7 = torch.randn(1, 2, 12, 128, dtype=torch.float32)
    qc.update(k3, v3, 3)
    qc.update(k7, v7, 7)
    qc.conv_states[0] = torch.ones(1, 4, 3)
    qc.recurrent_states[1] = torch.full((1, 2, 8, 8), 2.5)

    # Extract must succeed cleanly across all QuantizedHybridCache modes (`fp8`, `hadamard_int4`, `bf16`)
    ks, vs, idxs = extract_layer_kv_pairs(qc)
    assert idxs == [3, 7]
    assert ks.shape == (2, 2, 12, 128)
    assert vs.shape == (2, 2, 12, 128)
    assert float((ks[0] - k3[0]).norm() / k3[0].norm()) < 0.15

    # Inject into a fresh QuantizedHybridCache preserving GDN-2 linear states & supporting subsequent `.update()`
    qc_fresh = QuantizedHybridCache(num_layers=8, layer_types=layer_types, mode=mode)
    conv_sentinel = torch.full((1, 4, 3), 7.0)
    rec_sentinel = torch.full((1, 2, 8, 8), -3.0)
    qc_fresh.conv_states[0] = conv_sentinel
    qc_fresh.recurrent_states[1] = rec_sentinel

    inject_layer_kv_pairs(qc_fresh, ks, vs, idxs)
    assert qc_fresh.get_seq_length() == 12
    assert qc_fresh.get_seq_length(3) == 12
    assert qc_fresh.get_seq_length(7) == 12

    # Subsequent live decode/prefill update MUST append cleanly to T = 12 + 4 = 16 (never silently drop!)
    k_live = torch.randn(1, 2, 4, 128, dtype=torch.float32)
    v_live = torch.randn(1, 2, 4, 128, dtype=torch.float32)
    k_cat, v_cat = qc_fresh.update(k_live, v_live, 3)
    qc_fresh.update(k_live, v_live, 7)
    assert k_cat.shape == (1, 2, 16, 128)
    assert v_cat.shape == (1, 2, 16, 128)
    assert qc_fresh.get_seq_length(3) == 16
    assert qc_fresh.get_seq_length(7) == 16
    # GDN-2 linear states strictly isolated and untouched
    assert torch.equal(qc_fresh.conv_states[0], conv_sentinel)
    assert torch.equal(qc_fresh.recurrent_states[1], rec_sentinel)

    # Verify HFCheckpointK0Evaluator._fresh_cache_from reconstructs QuantizedHybridCache cleanly
    toy_m = ToyForCausalLM(ToyConfig(num_hidden_layers=8, head_dim=128, partial_rotary_factor=0.5))
    ev = HFCheckpointK0Evaluator(toy_m, ByteTokenizer(), device="cpu")
    qc_new = ev._fresh_cache_from(qc)
    assert isinstance(qc_new, QuantizedHybridCache)
    assert qc_new.mode == mode and qc_new.num_layers == 8 and qc_new.get_seq_length() == 0

    # Also verify Track-C style QuantizedHybridCache (with `.layers` + `._k_dtypes` present simultaneously)
    qc_fresh.layers = [SimpleNamespace(keys=None, values=None, is_initialized=False) for _ in range(8)]
    qc_fresh._k_dtypes = [torch.float32] * 8
    qc_fresh._v_dtypes = [torch.float32] * 8
    ks_tc, vs_tc, idxs_tc = extract_layer_kv_pairs(qc_fresh)
    assert idxs_tc == [3, 7] and ks_tc.shape[0] == 2 and vs_tc.shape[0] == 2
    inject_layer_kv_pairs(qc_fresh, ks, vs, idxs)
    assert qc_fresh.get_seq_length(3) == 12
    k_cat2, _ = qc_fresh.update(k_live, v_live, 3)
    assert k_cat2.shape == (1, 2, 16, 128)


def test_k0_tf524_dynamic_layer_uninitialized_inject_and_update():
    """Verify Blocker 2B: HF 5.20/5.24 pre-populated `DynamicLayer(is_initialized=False)` inject + update."""
    torch.manual_seed(17)

    class _TF524DynamicLayer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.is_initialized: bool = False

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.is_initialized:
                self.keys = k
                self.values = v
                self.is_initialized = True
            else:
                assert self.keys is not None and self.values is not None
                self.keys = torch.cat([self.keys, k], dim=-2)
                self.values = torch.cat([self.values, v], dim=-2)
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return int(self.keys.shape[-2]) if (self.is_initialized and self.keys is not None) else 0

        def reset(self) -> None:
            self.keys = None
            self.values = None
            self.is_initialized = False

    class _TF524LinearLayer:
        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [torch.ones(1, 4, 3)]
            self.recurrent_states: list[torch.Tensor | None] = [torch.full((1, 2, 4, 4), 5.0)]

    class _TF524HybridDynamicCache:
        def __init__(self) -> None:
            self.layers: list[object] = [
                _TF524LinearLayer(),
                _TF524LinearLayer(),
                _TF524LinearLayer(),
                _TF524DynamicLayer(),
                _TF524LinearLayer(),
                _TF524LinearLayer(),
                _TF524LinearLayer(),
                _TF524DynamicLayer(),
            ]

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            layer = self.layers[layer_idx]
            assert isinstance(layer, _TF524DynamicLayer)
            return layer.update(k, v)

        def get_seq_length(self, layer_idx: int = 3) -> int:
            layer = self.layers[layer_idx]
            return layer.get_seq_length() if isinstance(layer, _TF524DynamicLayer) else 0

    cache = _TF524HybridDynamicCache()
    k12 = torch.randn(2, 2, 12, 64)
    v12 = torch.randn(2, 2, 12, 64)

    inject_layer_kv_pairs(cache, k12, v12, [3, 7])
    layer3 = cache.layers[3]
    layer7 = cache.layers[7]
    assert isinstance(layer3, _TF524DynamicLayer) and isinstance(layer7, _TF524DynamicLayer)
    assert layer3.is_initialized is True and layer7.is_initialized is True
    assert cache.get_seq_length(3) == 12
    assert cache.get_seq_length(7) == 12

    # Subsequent live update MUST concatenate to T = 12 + 4 = 16 (never overwrite!)
    k4 = torch.randn(1, 2, 4, 64)
    v4 = torch.randn(1, 2, 4, 64)
    k_cat, v_cat = cache.update(k4, v4, 3)
    assert k_cat.shape == (1, 2, 16, 64)
    assert torch.equal(k_cat[:, :, :12], k12[0:1])
    assert torch.equal(k_cat[:, :, 12:], k4)
    assert cache.get_seq_length(3) == 16

    # Extract cleanly recovers initialized full-attention slots while ignoring linear layers
    ks_out, vs_out, idxs_out = extract_layer_kv_pairs(cache, layer_indices=[3])
    assert idxs_out == [3] and ks_out.shape == (1, 2, 16, 64) and torch.equal(vs_out[0], v_cat[0])
    # Linear layers remain strictly untouched
    lin0 = cache.layers[0]
    assert isinstance(lin0, _TF524LinearLayer)
    assert lin0.conv_states[0] is not None and torch.equal(lin0.conv_states[0], torch.ones(1, 4, 3))


def test_k0_hf_evaluator_qwen36_loop_hssd_and_pos_aware_arm_d_alignment():
    """Verify Blockers 4A, 4B, and Issue 4D on `Qwen36LoopHSSDForCausalLM` and a `position_ids`-aware model."""
    torch.manual_seed(23)
    tok = ByteTokenizer()

    # Part A (Blocker 4B): `Qwen36LoopHSSDForCausalLM` end-to-end (outer pack config != inner `_text_model.config`)
    toy_cfg = ToyConfig(
        vocab_size=260,
        hidden_size=32,
        num_hidden_layers=8,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.5,  # rotary_dim = 64
        rope_theta=10_000.0,
    )
    backbone = ToyForCausalLM(toy_cfg).eval()
    pack_cfg = Qwen36LoopHSSDConfig(
        base_config={
            "model_type": "toy",
            "num_hidden_layers": 8,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
        },
        qloop_v3_arch={},
    )
    wrapper = Qwen36LoopHSSDForCausalLM(pack_cfg).eval()
    wrapper.backbone = backbone

    k0_cfg = K0Config(rotary_dim=64, rope_theta=10_000.0, sink_tokens=2, refresh_frac=0.10, virtual_offset=10_000)
    ev_wrap = HFCheckpointK0Evaluator(wrapper, tok, device="cpu", live_sink_len=2)
    ex = K0Example(
        example_id="loop_hssd_e2e",
        sessions=["session one defines target_key = 42 in alpha.py"],
        live_prompt="Live sink prefix. Query target_key: ",
        answer="42",
    )
    rep_wrap = run_k0_suite([ex], ev_wrap, k0_cfg)
    assert set(rep_wrap.arms) == set(CANONICAL_ARMS)
    assert rep_wrap.arms["arm_c"].kv_bytes_mounted > 0
    assert rep_wrap.arms["arm_d"].refreshed_tokens >= 1
    assert ev_wrap.last_diagnostics["live_sink_tokens_prefilled"] == 2
    assert ev_wrap.last_diagnostics["post_refresh_kv_l2"] < ev_wrap.last_diagnostics["pre_refresh_kv_l2"]

    # Part B (Blocker 4A + Issue 4D): `position_ids`-aware wrapper verifying RoPE re-alignment at virtual_offset=10_000
    class _PosAwareToyCausalLM(torch.nn.Module):
        """Toy causal LM whose forward honors explicit `position_ids` identically to HF `Qwen3.6`."""

        def __init__(self, inner: ToyForCausalLM) -> None:
            super().__init__()
            self.inner = inner
            self.config = inner.config

        def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            past_key_values: ToyHybridCache | None = None,
            use_cache: bool = False,
        ) -> ToyCausalLMOutput:
            tm = self.inner.model
            h = tm.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = ToyHybridCache(self.config)
            start = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(start, start + h.shape[1], device=h.device)
            pos_ids = cache_position[None].expand(h.shape[0], -1) if position_ids is None else position_ids
            pos_emb = tm.rotary_emb(h, pos_ids)
            for layer in tm.layers[: self.config.num_hidden_layers]:
                h = layer(
                    h,
                    position_embeddings=pos_emb,
                    attention_mask=None,
                    position_ids=pos_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
            h = tm.norm(h)
            return ToyCausalLMOutput(self.inner.lm_head(h), None, past_key_values)

    pos_model = _PosAwareToyCausalLM(backbone).eval()
    ev_pos = HFCheckpointK0Evaluator(pos_model, tok, device="cpu", live_sink_len=4)
    assert ev_pos._accepts_pos_ids is True

    # Verify Arm (d) RoPE re-alignment + live Partition-A sinks at `virtual_offset = 10_000`
    ep_map = ev_pos(ex, k0_cfg)
    diag = ev_pos.last_diagnostics
    assert diag["live_sink_tokens_prefilled"] == 4
    assert diag["base_mount_offset"] == 10_000
    assert diag["mounted_total_tokens_arm_c"] == 4 + diag["part_b_tokens"]
    assert diag["post_refresh_kv_l2"] < diag["pre_refresh_kv_l2"]
    # Without Blocker 4A's RoPE re-alignment, `rot_l2_pre` is ~14x larger than `nope_l2_pre`!
    # With proper un-rotate + re-rotate to `mnt.effective_positions`, rotary and NoPE discrepancies match scale:
    assert diag["rot_l2_pre"] <= 2.5 * diag["nope_l2_pre"] + 1e-3, (
        f"Arm (d) rotary error swamped NoPE error (unaligned RoPE!): "
        f"rot={diag['rot_l2_pre']}, nope={diag['nope_l2_pre']}"
    )
    assert ep_map["arm_c"].extra["live_sink_tokens_prefilled"] == 4
    assert ep_map["arm_d"].refreshed_tokens == math.ceil(k0_cfg.refresh_frac * diag["part_b_tokens"])


def test_k0_clone_linear_states_cyclic_quantized_cache_no_recursion():
    """Verify Blocker R1: `_clone_linear_states_into` & `_clone_state_value` handle Track-C cyclic proxies safely."""
    torch.manual_seed(31)

    class _SingleSlotStateList:
        """Mirrors Track-C `QuantizedHybridCache` proxy holding back-reference to parent cache."""

        def __init__(self, cache: object, attr: str, layer_idx: int) -> None:
            self._cache = cache
            self._attr = attr
            self._layer_idx = layer_idx

        def __getitem__(self, idx: int) -> torch.Tensor | None:
            assert idx == 0
            return getattr(self._cache, self._attr)[self._layer_idx]

    class _CyclicLayerProxy:
        def __init__(self, cache: object, layer_idx: int) -> None:
            self._cache = cache
            self._layer_idx = layer_idx

        @property
        def conv_states(self) -> _SingleSlotStateList:
            return _SingleSlotStateList(self._cache, "conv_states", self._layer_idx)

        @property
        def recurrent_states(self) -> _SingleSlotStateList:
            return _SingleSlotStateList(self._cache, "recurrent_states", self._layer_idx)

    layer_types = ["linear_attention", "full_attention"]
    src = QuantizedHybridCache(num_layers=2, layer_types=layer_types, mode="fp8")
    dst = QuantizedHybridCache(num_layers=2, layer_types=layer_types, mode="fp8")
    src.layers = [_CyclicLayerProxy(src, 0), _CyclicLayerProxy(src, 1)]
    dst.layers = [_CyclicLayerProxy(dst, 0), _CyclicLayerProxy(dst, 1)]

    conv0 = torch.randn(1, 4, 3)
    rec0 = torch.randn(1, 2, 4, 4)
    src.conv_states[0] = conv0
    src.recurrent_states[0] = rec0

    # Must complete without RecursionError / stack overflow and clone tensors independently
    _clone_linear_states_into(src, dst)
    assert dst.conv_states[0] is not None and torch.equal(dst.conv_states[0], conv0)
    assert dst.recurrent_states[0] is not None and torch.equal(dst.recurrent_states[0], rec0)
    dst.conv_states[0].zero_()
    dst.recurrent_states[0].zero_()
    assert not torch.equal(src.conv_states[0], dst.conv_states[0])
    assert not torch.equal(src.recurrent_states[0], dst.recurrent_states[0])

    # Direct cyclic graph passed into `_clone_state_value` terminates cleanly via cycle memo
    cloned_proxy = _clone_state_value(src.layers[0].conv_states)
    assert isinstance(cloned_proxy, _SingleSlotStateList)
    assert cloned_proxy._layer_idx == 0


def test_k0_clone_linear_states_tf524_linear_attention_layer_metadata():
    """Verify Blocker R2: TF 5.24 `LinearAttentionLayer` flags & metadata preserved across clone + update."""
    torch.manual_seed(37)

    class _TF524LinearAttentionLayer:
        supports_early_init = False

        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[torch.Tensor | None] = [None]
            self.conv_kernel_size: dict[int, int] = {}
            self.is_conv_states_initialized: dict[int, bool] = {0: False}
            self.is_recurrent_states_initialized: dict[int, bool] = {0: False}
            self.has_previous_state: dict[int, bool] = {0: False}
            self.batch_size: int | None = None

        def update_conv_state(
            self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None
        ) -> torch.Tensor:
            if not self.is_conv_states_initialized[state_idx]:
                self.conv_kernel_size[state_idx] = (
                    conv_states.shape[-1] if conv_kernel_size is None else int(conv_kernel_size)
                )
                self.conv_states[state_idx] = torch.zeros(
                    (*conv_states.shape[:-1], self.conv_kernel_size[state_idx]),
                    dtype=conv_states.dtype,
                    device=conv_states.device,
                )
                self.is_conv_states_initialized[state_idx] = True
                self.batch_size = int(conv_states.shape[0])
            if not self.has_previous_state[state_idx]:
                full_cs = conv_states
                if full_cs.shape[-1] < self.conv_kernel_size[state_idx]:
                    full_cs = F.pad(full_cs, (self.conv_kernel_size[state_idx] - full_cs.shape[-1], 0), value=0)
                self.has_previous_state[state_idx] = True
            else:
                prev = self.conv_states[state_idx]
                assert prev is not None
                full_cs = torch.cat([prev, conv_states], dim=-1)
            dst_buf = self.conv_states[state_idx]
            assert dst_buf is not None
            dst_buf.copy_(full_cs[..., -self.conv_kernel_size[state_idx] :])
            return full_cs

        def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
            if not self.is_recurrent_states_initialized[state_idx]:
                self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
                self.is_recurrent_states_initialized[state_idx] = True
            dst_buf = self.recurrent_states[state_idx]
            assert dst_buf is not None
            dst_buf.copy_(recurrent_states)
            return dst_buf

    class _TF524FullLayer:
        supports_early_init = True

        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.is_initialized: bool = False

    class _TF524DynamicCache:
        def __init__(self) -> None:
            self.layers: list[object] = [_TF524LinearAttentionLayer(), _TF524FullLayer()]

        def has_previous_state(self, layer_idx: int) -> bool:
            lyr = self.layers[layer_idx]
            assert isinstance(lyr, _TF524LinearAttentionLayer)
            return bool(lyr.has_previous_state[0])

        def get_recurrent_state(self, layer_idx: int) -> torch.Tensor | None:
            lyr = self.layers[layer_idx]
            assert isinstance(lyr, _TF524LinearAttentionLayer)
            return lyr.recurrent_states[0] if lyr.has_previous_state[0] else None

        def update_conv_state(
            self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None
        ) -> torch.Tensor:
            lyr = self.layers[layer_idx]
            assert isinstance(lyr, _TF524LinearAttentionLayer)
            return lyr.update_conv_state(conv_states, conv_kernel_size=conv_kernel_size)

        def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            lyr = self.layers[layer_idx]
            assert isinstance(lyr, _TF524LinearAttentionLayer)
            return lyr.update_recurrent_state(recurrent_states)

    src_524 = _TF524DynamicCache()
    dst_524 = _TF524DynamicCache()
    conv_init = torch.randn(1, 4, 3)
    rec_init = torch.randn(1, 2, 4, 4)
    src_524.update_conv_state(conv_init, 0, conv_kernel_size=3)
    src_524.update_recurrent_state(rec_init, 0)

    _clone_linear_states_into(src_524, dst_524)

    # All LinearAttentionLayer flags & tensors must be faithfully cloned
    assert dst_524.has_previous_state(0) is True
    dst_rec = dst_524.get_recurrent_state(0)
    assert dst_rec is not None and torch.equal(dst_rec, rec_init)
    dst_lin = dst_524.layers[0]
    src_lin = src_524.layers[0]
    assert isinstance(dst_lin, _TF524LinearAttentionLayer) and isinstance(src_lin, _TF524LinearAttentionLayer)
    assert dst_lin.is_conv_states_initialized == {0: True}
    assert dst_lin.is_recurrent_states_initialized == {0: True}
    assert dst_lin.conv_kernel_size == {0: 3}
    assert dst_lin.batch_size == 1

    # Subsequent update_conv_state(len=2) MUST concatenate prior 3 + new 2 = 5 (never overwrite with zeros!)
    conv_next = torch.randn(1, 4, 2)
    full_cs = dst_524.update_conv_state(conv_next, 0)
    assert full_cs.shape == (1, 4, 5)
    assert torch.equal(full_cs[..., :3], conv_init)
    assert torch.equal(full_cs[..., 3:], conv_next)
    # Source cache stays untouched (independent clone)
    assert src_lin.conv_states[0] is not None and torch.equal(src_lin.conv_states[0], conv_init)


def test_k0_tf524_readonly_is_initialized_property_inject_and_update():
    """Verify `inject_layer_kv_pairs` handles read-only `@property def is_initialized` layers without `.reset()`."""
    torch.manual_seed(41)

    class _ReadOnlyInitDynamicLayer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None

        @property
        def is_initialized(self) -> bool:
            return self.keys is not None and self.keys.numel() > 0

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.is_initialized:
                self.keys = k
                self.values = v
            else:
                assert self.keys is not None and self.values is not None
                self.keys = torch.cat([self.keys, k], dim=-2)
                self.values = torch.cat([self.values, v], dim=-2)
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return int(self.keys.shape[-2]) if self.is_initialized and self.keys is not None else 0

    class _ReadOnlyInitCache:
        def __init__(self) -> None:
            self.layers: list[_ReadOnlyInitDynamicLayer] = [_ReadOnlyInitDynamicLayer(), _ReadOnlyInitDynamicLayer()]

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.layers[layer_idx].update(k, v)

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self.layers[layer_idx].get_seq_length()

    cache = _ReadOnlyInitCache()
    k10 = torch.randn(2, 2, 10, 64)
    v10 = torch.randn(2, 2, 10, 64)
    # Must NOT raise `AttributeError: can't set attribute` on read-only `@property def is_initialized`!
    inject_layer_kv_pairs(cache, k10, v10, [0, 1])
    assert cache.layers[0].is_initialized is True
    assert cache.get_seq_length(0) == 10
    assert cache.get_seq_length(1) == 10

    k4 = torch.randn(1, 2, 4, 64)
    v4 = torch.randn(1, 2, 4, 64)
    k_cat, v_cat = cache.update(k4, v4, 0)
    assert k_cat.shape == (1, 2, 14, 64) and v_cat.shape == (1, 2, 14, 64)
    assert torch.equal(k_cat[:, :, :10], k10[0:1]) and torch.equal(k_cat[:, :, 10:], k4)

    # Also test direct fallback branch (no `.update()` on cache or layer, read-only `@property def is_initialized`)
    class _NoUpdateReadOnlyLayer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None

        @property
        def is_initialized(self) -> bool:
            return self.keys is not None and self.keys.numel() > 0

    stub_cache = SimpleNamespace(layers=[_NoUpdateReadOnlyLayer()])
    inject_layer_kv_pairs(stub_cache, k10[:1], v10[:1], [0])
    assert stub_cache.layers[0].is_initialized is True
    assert stub_cache.layers[0].keys is not None and torch.equal(stub_cache.layers[0].keys, k10[:1])
