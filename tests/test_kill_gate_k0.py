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

from qloop.eval.kill_gate_k0 import (
    CANONICAL_ARMS,
    HFCheckpointK0Evaluator,
    K0ArmMetrics,
    K0Config,
    K0EpisodeResult,
    K0Example,
    K0Report,
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
from qloop.model.toy import ToyConfig, ToyForCausalLM


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

    # In-process `main(["--smoke-cpu", ...])` check
    out_json = tmp_path / "k0_report.json"
    out_md = tmp_path / "k0_report.md"
    rc = main(["--smoke-cpu", "--output-json", str(out_json), "--output-md", str(out_md)])
    assert rc == 0
    assert out_json.exists() and json.loads(out_json.read_text(encoding="utf-8"))["verdict"] == "GREEN"
    assert out_md.exists() and "Kill-Gate `K0` Report" in out_md.read_text(encoding="utf-8")

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
