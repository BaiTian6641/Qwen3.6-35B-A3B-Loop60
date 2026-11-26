"""Tests for Sprint 2 Scientific Validation & Causal Ablation Harness (`src/qloop/eval/sprint2_audit.py`).

Covers:
  1. Leak-free holdout & OOD batch builder (`A1` / `QL60-003`), canonical SHA-256 dataset digest,
     `assert_zero_doc_overlap`, local JSONL/text OOD loading, and `QL60-010` `run_identity` cache checks.
  2. `A3` 3-repeat nondeterminism + `2x d=1` check, pure LM CE (`QL60-009` / `C1`, zero aux/KL),
     `> 3sigma` statistical gate, and head-to-head baseline comparison.
  3. `B2` / `B3` / `B4` / `B5` length ladder: chunked streaming CE parity vs full CE, `GDN-2`
     recurrent state norms across `cache.recurrent_states` & `cache.layers[i].recurrent_states`,
     multi-needle RULER exact-key/value episodes, and dense-vs-sparse `D-LSA` + live `lsa/last_recall`.
  4. `S2.3` reversible causal ablation context managers (`full`, `no_gdn2`, `no_qed`, `no_neg_eigval`,
     `no_pass_router`, `low_anchor_probe`, `lsa_dense`, `lsa_sparse`) with exception-safe bit-exact
     SHA-256 restoration and recommendation tags.
  5. `--smoke-cpu` CLI, config + checkpoint + cache E2E, Markdown formatting, and hygiene checks.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml

from qloop.data.tokenizer import ToyTokenizer
from qloop.eval.sprint2_audit import (
    CANONICAL_ABLATIONS,
    RECOMMENDATION_TAGS,
    DepthMetrics,
    DepthSweepResult,
    ablate_low_anchor_probe,
    ablate_lsa_mode,
    ablate_no_gdn2,
    ablate_no_neg_eigval,
    ablate_no_pass_router,
    ablate_no_qed,
    assert_zero_doc_overlap,
    build_holdout_eval_batches,
    build_local_file_eval_batches,
    build_ood_eval_batches,
    build_ruler_multi_needle_episode,
    build_smoke_cpu_stack,
    build_sprint2_run_identity,
    causal_ablation_context,
    compare_arms_against_baseline,
    compute_ablation_state_sha256,
    compute_weights_sha256,
    compute_z_and_3sigma,
    evaluate_pure_lm_nll,
    evaluate_ruler_multi_needle,
    format_sprint2_markdown,
    inspect_gdn2_recurrent_state_norms,
    main,
    parse_arm_specs,
    run_causal_ablation_matrix,
    run_depth_nll_sweep,
    run_length_ladder_audit,
    run_smoke_cpu,
    run_sprint2_audit_from_config,
    save_cached_json,
    stream_chunked_lm_nll,
    verify_or_load_cached_json,
)
from qloop.model.toy import build_toy
from qloop.train.checkpoint import CheckpointManager
from qloop.train.stack import build_stack

# ==============================================================================
# 1. Leak-Free Holdout / OOD Batch Builder & QL60-010 Cache Identity
# ==============================================================================


def test_assert_zero_doc_overlap_passes_and_rejects_leak() -> None:
    res = assert_zero_doc_overlap(["doc_a", "doc_b", "doc_c"], ["doc_d", "doc_e"])
    assert res == {"verified": True, "train_docs": 3, "eval_docs": 2, "overlap_count": 0}

    with pytest.raises(ValueError, match="Leak-free holdout violation"):
        assert_zero_doc_overlap(["doc_a", "doc_b"], ["doc_b", "doc_c"])


def test_holdout_eval_batches_deterministic_and_zero_train_overlap() -> None:
    tok = ToyTokenizer()
    data_cfg = {
        "seq_len": 64,
        "holdout_mod": 10,
        "holdout_eval_buckets": 2,
        "sources": [
            {"name": "toy_text", "weight": 0.5, "toy_rows": 120},
            {"name": "toy_cot", "weight": 0.5, "toy_rows": 120},
        ],
    }
    eval_a = build_holdout_eval_batches(
        data_cfg,
        tok,
        n_batches=2,
        seed=17,
        split_role="eval",
        verify_against_train_split=True,
        train_verify_batches=8,
    )
    eval_b = build_holdout_eval_batches(
        data_cfg,
        tok,
        n_batches=2,
        seed=17,
        split_role="eval",
    )
    assert eval_a.dataset_sha256 == eval_b.dataset_sha256
    assert len(eval_a.dataset_sha256) == 64
    assert eval_a.overlap_check["verified"] is True
    assert eval_a.overlap_check["overlap_count"] == 0
    assert len(eval_a.doc_ids) > 0
    assert eval_a.valid_loss_tokens > 0
    for ba, bb in zip(eval_a.batches, eval_b.batches):
        assert torch.equal(ba["input_ids"], bb["input_ids"])
        assert torch.equal(ba["labels"], bb["labels"])


def test_local_file_and_ood_eval_batches_with_overlap_guard(tmp_path: Path) -> None:
    tok = ToyTokenizer()
    jsonl_file = tmp_path / "ood.jsonl"
    rows = [
        {"id": "ood-doc-1", "text": "Quantum optics experiments measure photon entanglement across fiber links."},
        {
            "id": "ood-doc-2",
            "prompt": "Compute 12 + 30.",
            "response": "12 + 30 equals 42.",
        },
        {
            "id": "ood-doc-3",
            "messages": [
                {"role": "user", "content": "Name one prime."},
                {"role": "assistant", "content": "Seven is a prime number."},
            ],
        },
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    txt_file = tmp_path / "ood_notes.txt"
    txt_file.write_text(
        "Paragraph one describes basalt geology on volcanic islands.\n\n"
        "Paragraph two describes maritime navigation by sextant.",
        encoding="utf-8",
    )

    batch_set = build_local_file_eval_batches([jsonl_file, txt_file], tok, seq_len=64, label="ood_files")
    assert len(batch_set.batches) >= 1
    assert batch_set.valid_loss_tokens > 0
    assert "id:ood-doc-1" in batch_set.doc_ids

    # Passing disjoint train doc IDs succeeds
    batch_ok = build_ood_eval_batches([jsonl_file], tok, seq_len=64, train_doc_ids=["id:train-only-99"])
    assert batch_ok.overlap_check["overlap_count"] == 0

    # Passing overlapping train doc ID fails loudly
    with pytest.raises(ValueError, match="Leak-free holdout violation"):
        build_ood_eval_batches([jsonl_file], tok, seq_len=64, train_doc_ids=["id:ood-doc-1"])


def test_ql60_010_run_identity_cache_refuses_cross_checkpoint_mixing(tmp_path: Path) -> None:
    st_a = build_smoke_cpu_stack(seed=1, num_passes=2, trained_perturbation=True)
    st_b = build_smoke_cpu_stack(seed=2, num_passes=2, trained_perturbation=True)

    id_a = build_sprint2_run_identity(
        st_a["model"],
        arm_name="arm_a",
        dataset_sha256="deadbeef" * 8,
        config_payload={"repeats": 3},
        loop=st_a["loop"],
    )
    id_b = build_sprint2_run_identity(
        st_b["model"],
        arm_name="arm_a",
        dataset_sha256="deadbeef" * 8,
        config_payload={"repeats": 3},
        loop=st_b["loop"],
    )
    assert id_a["checkpoint_sha"] != id_b["checkpoint_sha"]

    cache_file = tmp_path / "sweep_cache.json"
    save_cached_json(cache_file, {"metric": 1.234}, id_a)

    # Matching identity loads cleanly
    loaded = verify_or_load_cached_json(cache_file, id_a)
    assert loaded is not None and loaded["metric"] == pytest.approx(1.234)

    # Different checkpoint weights MUST raise ValueError
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_or_load_cached_json(cache_file, id_b)

    # Different dataset digest MUST raise ValueError
    id_a_diff_ds = dict(id_a, dataset_rev="cafebabe" * 8)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_or_load_cached_json(cache_file, id_a_diff_ds)

    # Legacy cache missing `run_identity` MUST raise ValueError
    legacy_file = tmp_path / "legacy.json"
    legacy_file.write_text(json.dumps({"metric": 2.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks `run_identity`"):
        verify_or_load_cached_json(legacy_file, id_a)

    # Explicit opt-in allow_mixed_cache=True allows overwrite path (returns None)
    assert verify_or_load_cached_json(cache_file, id_b, allow_mixed_cache=True) is None


# ==============================================================================
# 2. Pure LM CE (QL60-009 / C1), A3 Multi-Repeat Sweep & Baseline Comparison
# ==============================================================================


def test_pure_lm_nll_matches_naive_valid_token_ce_with_zero_aux_contamination() -> None:
    st = build_smoke_cpu_stack(seed=5, num_passes=2, trained_perturbation=True)
    model = st["model"]
    # Inject synthetic aux/KL tensors to prove evaluate_pure_lm_nll never folds them into LM NLL
    if st["lsa"] is not None:
        first_ix = next(iter(st["lsa"].indexers.values()))
        first_ix.pending.append(torch.tensor(42.0))
    if st["loop"] is not None:
        st["loop"].ctx.aux["synthetic_aux"] = torch.tensor(17.5)

    vocab = len(st["tokenizer"])
    gen = torch.Generator().manual_seed(11)
    # Two batches with UNEQUAL valid token counts (10 vs 30) to verify exact token-weighted averaging
    ids1 = torch.randint(0, vocab, (40,), generator=gen)
    lbl1 = ids1.clone()
    lbl1[:30] = -100  # 10 valid shifted targets
    ids2 = torch.randint(0, vocab, (40,), generator=gen)
    lbl2 = ids2.clone()
    lbl2[:10] = -100  # 30 valid shifted targets
    ids3 = torch.randint(0, vocab, (16,), generator=gen)
    lbl3 = torch.full_like(ids3, -100)  # 0 valid targets (must be safely skipped without NaN)

    batches = [
        {"input_ids": ids1, "labels": lbl1},
        {"input_ids": ids2, "labels": lbl2},
        {"input_ids": ids3, "labels": lbl3},
    ]

    mean_nll, n_valid = evaluate_pure_lm_nll(
        model, batches, ce_chunk_size=7, loop=st["loop"], lsa=st["lsa"]
    )
    assert n_valid == 40

    # Exact naive reference
    with torch.no_grad():
        tot_ce = 0.0
        for b in batches[:2]:
            logits = model(b["input_ids"].unsqueeze(0), use_cache=False).logits
            tot_ce += float(
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    b["labels"].unsqueeze(0)[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
            )
    ref_nll = tot_ce / 40.0
    assert mean_nll == pytest.approx(ref_nll, rel=1e-5, abs=1e-5)
    # Ensure pending aux losses were flushed cleanly
    if st["lsa"] is not None:
        assert st["lsa"].aux_loss() is None


def test_depth_nll_sweep_3_repeats_and_baseline_significance_gate() -> None:
    st_base = build_smoke_cpu_stack(seed=3, num_passes=1, install_retrofit=False)
    st_cand = build_smoke_cpu_stack(seed=3, num_passes=3, trained_perturbation=True, install_retrofit=True)
    tok = st_cand["tokenizer"]
    eval_set = build_holdout_eval_batches(
        {
            "seq_len": 48,
            "holdout_mod": 10,
            "holdout_eval_buckets": 2,
            "sources": [{"name": "toy_text", "weight": 1.0, "toy_rows": 100}],
        },
        tok,
        n_batches=2,
        seed=9,
    )

    sw_base = run_depth_nll_sweep(
        st_base["model"], eval_set, depths=(1, 2, 3), n_repeats=3, arm_name="stock_d1", ce_chunk_size=16
    )
    sw_cand = run_depth_nll_sweep(
        st_cand["model"],
        eval_set,
        depths=(1, 2, 3),
        n_repeats=3,
        loop=st_cand["loop"],
        lsa=st_cand["lsa"],
        arm_name="cand_d3",
        ce_chunk_size=16,
    )
    assert list(sw_base.depths.keys()) == [1]
    assert list(sw_cand.depths.keys()) == [1, 2, 3]
    assert len(sw_cand.d1_repeat_nlls) == 2
    assert sw_cand.d1_repeat_max_abs_diff == pytest.approx(0.0, abs=1e-7)
    assert sw_cand.d1_repeat_std == pytest.approx(0.0, abs=1e-7)
    for d in (1, 2, 3):
        m = sw_cand.depths[d]
        assert m.n_repeats == 3
        assert math.isfinite(m.mean_nll) and math.isfinite(m.ppl)
        assert m.std_nll == pytest.approx(0.0, abs=1e-7)
    # Non-zero trained loop gates must make d=3 differ from d=1
    assert abs(sw_cand.depth_gain_d1_to_dmax) > 1e-5

    # Verify exact 3-sigma statistical gating math under noisy repeats
    z_ok, ex_ok = compute_z_and_3sigma(0.010, 0.002)
    assert z_ok == pytest.approx(5.0) and ex_ok is True
    z_low, ex_low = compute_z_and_3sigma(0.004, 0.002)
    assert z_low == pytest.approx(2.0) and ex_low is False
    z_neg, ex_neg = compute_z_and_3sigma(-0.020, 0.002)
    assert z_neg == pytest.approx(-10.0) and ex_neg is False

    # Synthetic comparison table checking PASS vs FAIL vs BASELINE
    def _mock_sweep(name: str, vals: list[float]) -> DepthSweepResult:
        mean_v = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1))
        dm = DepthMetrics(
            depth=1, n_repeats=len(vals), repeat_nlls=vals, mean_nll=mean_v, std_nll=sd, ppl=math.exp(mean_v),
            valid_tokens=100,
        )
        return DepthSweepResult(
            arm_name=name, dataset_name="holdout", dataset_sha256="0" * 64, depths={1: dm}, n_repeats=len(vals),
            d1_repeat_nlls=vals[:2], d1_repeat_max_abs_diff=abs(vals[0] - vals[1]), d1_repeat_std=sd,
            depth_gain_d1_to_dmax=0.0, pooled_std_d1_to_dmax=0.0, z_score=0.0, exceeds_3sigma=False, d_min=1, d_max=1,
        )

    comp = compare_arms_against_baseline(
        {
            "baseline": _mock_sweep("baseline", [1.000, 1.002, 0.998]),
            "winner": _mock_sweep("winner", [0.950, 0.952, 0.948]),
            "noise_only": _mock_sweep("noise_only", [0.999, 1.001, 0.997]),
            "regressed": _mock_sweep("regressed", [1.050, 1.052, 1.048]),
        },
        baseline_arm="baseline",
    )
    by_name = {r.arm_name: r for r in comp}
    assert by_name["baseline"].is_baseline is True and by_name["baseline"].passes_s2_gate is False
    assert by_name["winner"].passes_s2_gate is True and by_name["winner"].z_score_vs_baseline > 10.0
    assert by_name["noise_only"].passes_s2_gate is False
    assert by_name["regressed"].passes_s2_gate is False and by_name["regressed"].delta_vs_baseline < 0.0


# ==============================================================================
# 3. Length-Ladder, Chunked Streaming CE Parity, RULER & GDN-2 State Norms
# ==============================================================================


def test_stream_chunked_lm_nll_matches_full_sequence_ce_across_chunk_boundaries() -> None:
    st = build_smoke_cpu_stack(seed=13, num_passes=2, trained_perturbation=True)
    model = st["model"].double()
    vocab = len(st["tokenizer"])
    ids = torch.randint(0, vocab, (1, 64), generator=torch.Generator().manual_seed(77))

    # Full-sequence reference in dense mode
    with st["lsa"].temporary_mode("dense"):
        full_nll, full_n = evaluate_pure_lm_nll(
            model, [{"input_ids": ids[0], "labels": ids[0]}], ce_chunk_size=16, loop=st["loop"], lsa=st["lsa"]
        )
        stream_nll, stream_n, cache, _, _ = stream_chunked_lm_nll(
            model, ids, ids, chunk_size=16, ce_chunk_size=8, loop=st["loop"], lsa=st["lsa"]
        )

    assert full_n == 63 and stream_n == 63
    assert stream_nll == pytest.approx(full_nll, rel=1e-9, abs=1e-9)

    norms = inspect_gdn2_recurrent_state_norms(cache)
    assert norms.num_linear_states > 0
    assert norms.is_bounded is True
    assert norms.nan_count == 0 and norms.inf_count == 0
    assert norms.max_l2_norm > 0.0


def test_inspect_gdn2_recurrent_state_norms_handles_hf524_layers_and_nan_detection() -> None:
    class _LayerHolder:
        def __init__(self, state: torch.Tensor | None) -> None:
            self.recurrent_states = [state] if state is not None else []

    class _MockHF524Cache:
        def __init__(self, tensors: list[torch.Tensor]) -> None:
            self.layers = [_LayerHolder(t) for t in tensors]

    t0 = torch.tensor([3.0, 4.0])  # l2=5.0, max_abs=4.0
    t1 = torch.tensor([1.0, float("nan"), 2.0])
    good = inspect_gdn2_recurrent_state_norms(_MockHF524Cache([t0]))
    assert good.num_linear_states == 1
    assert good.max_l2_norm == pytest.approx(5.0)
    assert good.max_abs_val == pytest.approx(4.0)
    assert good.is_bounded is True

    bad = inspect_gdn2_recurrent_state_norms(_MockHF524Cache([t0, t1]))
    assert bad.num_linear_states == 2
    assert bad.nan_count == 1
    assert bad.is_bounded is False


def test_length_ladder_ruler_sparse_recall_and_fp8_cache() -> None:
    st = build_smoke_cpu_stack(seed=19, num_passes=3, trained_perturbation=True)
    tok = st["tokenizer"]

    ladder = run_length_ladder_audit(
        st["model"],
        lengths=(64, 128),
        tokenizer=tok,
        chunk_size=32,
        ce_chunk_size=32,
        kv_cache_quant="native",
        depth=3,
        loop=st["loop"],
        lsa=st["lsa"],
        num_needles=4,
        seed=19,
        arm_name="smoke_d3",
    )
    assert len(ladder.points) == 2
    assert ladder.all_states_bounded is True
    for pt in ladder.points:
        assert math.isfinite(pt.dense_nll)
        assert pt.sparse_nll is not None and math.isfinite(pt.sparse_nll)
        assert pt.dense_sparse_nll_gap == pytest.approx(pt.sparse_nll - pt.dense_nll)
        # B2: live sparse recall must be measured (non-zero in (0, 1]) when L > sinks + window
        assert pt.lsa_sparse_active is True
        assert pt.lsa_last_recall is not None and 0.0 < pt.lsa_last_recall <= 1.0
        assert pt.ruler is not None and pt.ruler.num_needles == 4
        assert math.isfinite(pt.ruler.answer_nll)
        assert pt.gdn2_state_norms is not None
        assert pt.gdn2_state_norms.num_linear_states > 0
        assert pt.gdn2_state_norms.is_bounded is True

    # Also verify small-vocab ToyForCausalLM (`vocab_size=97`) RULER fallback + FP8 QuantizedHybridCache streaming
    tiny_model = build_toy(seed=7)
    ep_small = build_ruler_multi_needle_episode(64, tokenizer=None, num_needles=4, seed=7, vocab_size=97)
    assert ep_small["mode"] == "token_ids"
    assert max(ep_small["prompt_ids"] + ep_small["answer_ids"]) < 97
    r_small, st_small = evaluate_ruler_multi_needle(
        tiny_model, ep_small, chunk_size=32, kv_cache_quant="fp8"
    )
    assert math.isfinite(r_small.answer_nll)
    assert st_small.num_linear_states > 0 and st_small.is_bounded is True


# ==============================================================================
# 4. S2.3 Zero-Training Reversible Causal Ablations & Bit-Exact Restoration
# ==============================================================================


def test_every_s23_ablation_context_modifies_only_target_and_restores_bit_exact() -> None:
    st = build_smoke_cpu_stack(seed=23, num_passes=3, trained_perturbation=True)
    model, loop, gdn2, lsa = st["model"], st["loop"], st["gdn2"], st["lsa"]
    ids = torch.randint(0, len(st["tokenizer"]), (1, 32), generator=torch.Generator().manual_seed(23))

    sha_orig = compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa)
    w_sha_orig = compute_weights_sha256(model)
    with torch.no_grad():
        logits_orig = model(ids, use_cache=False).logits.clone()
    lsa.clear_cache()

    # 1) no_gdn2 disables all GDN2 adapters
    with ablate_no_gdn2(model, gdn2=gdn2) as info:
        assert info["applicable"] is True
        assert all(ad.is_zero() for ad in gdn2.adapters.values())
        with torch.no_grad():
            logits_no_gdn2 = model(ids, use_cache=False).logits
        assert not torch.allclose(logits_no_gdn2, logits_orig)
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 2) no_qed zeroes qed_lambda while keeping base GDN-2 erase/write/decay + neg_eigval active
    with ablate_no_qed(model, gdn2=gdn2) as info:
        assert info["applicable"] is True
        for ad in gdn2.adapters.values():
            lam = ad.qed_lambda()
            assert lam is not None and bool((lam == 0).all())
            assert not ad.is_zero()
            neg_sc = ad.neg_eigval_scale()
            assert neg_sc is not None and float((neg_sc - 1.0).detach().abs().max()) > 1e-4
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 3) no_neg_eigval clamps neg_eigval_scale to exact 1.0 while keeping QED & base GDN-2 active
    with ablate_no_neg_eigval(model, gdn2=gdn2) as info:
        assert info["applicable"] is True
        for ad in gdn2.adapters.values():
            assert ad.cfg.neg_eigval_max_boost == 0.0
            neg_sc = ad.neg_eigval_scale()
            assert neg_sc is not None and torch.allclose(neg_sc, torch.ones_like(neg_sc))
            lam = ad.qed_lambda()
            assert lam is not None and float(lam.detach().abs().max()) > 1e-4
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 4) no_pass_router zeroes PassRouterAdapters deltas & repel on d >= 2
    with ablate_no_pass_router(model, loop=loop) as info:
        assert info["applicable"] is True
        pr = loop.adapters.pass_router
        assert pr is not None
        assert pr.is_zero(0, 0) and pr.is_zero(1, 0)
        assert float(pr.repel.detach().abs().max()) == 0.0
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 5) low_anchor_probe shifts w_anchor <= 0.3 while renormalizing on the simplex (preserving w0:w2 ratio)
    br = loop.adapters.bridge
    orig_w = br.weights(0).detach()
    orig_w0, orig_w1, orig_w2 = float(orig_w[0]), float(orig_w[1]), float(orig_w[2])
    assert orig_w1 == pytest.approx(0.85, abs=1e-5)
    with ablate_low_anchor_probe(model, loop=loop, max_anchor_weight=0.25) as info:
        assert info["applicable"] is True
        for p_idx in range(br.n_extra_passes):
            w = br.weights(p_idx).detach()
            assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
            assert float(w[1]) <= 0.3 and float(w[1]) == pytest.approx(0.25, abs=1e-5)
            assert float(w[0] / w[2]) == pytest.approx(orig_w0 / orig_w2, rel=1e-5)
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 6) lsa_dense / lsa_sparse mode toggle restores original mode
    orig_mode = lsa.cfg.mode
    with ablate_lsa_mode(model, "dense", lsa=lsa):
        assert lsa.cfg.mode == "dense"
        assert all(ix.cfg.mode == "dense" for ix in lsa.indexers.values())
    assert lsa.cfg.mode == orig_mode
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig

    # 7) Exception safety: even if an exception occurs inside an ablation, weights & state restore bit-exactly!
    abl_ctx = causal_ablation_context(model, "no_gdn2", loop=loop, gdn2=gdn2, lsa=lsa)
    with pytest.raises(RuntimeError, match="forced failure"), abl_ctx:
        raise RuntimeError("forced failure")
    assert compute_ablation_state_sha256(model, loop=loop, gdn2=gdn2, lsa=lsa) == sha_orig
    assert compute_weights_sha256(model) == w_sha_orig

    lsa.clear_cache()
    with torch.no_grad():
        logits_after = model(ids, use_cache=False).logits
    assert torch.equal(logits_after, logits_orig)


def test_run_causal_ablation_matrix_emits_tags_and_live_recall() -> None:
    st = build_smoke_cpu_stack(seed=31, num_passes=3, trained_perturbation=True)
    tok = st["tokenizer"]
    eval_set = build_holdout_eval_batches(
        {
            "seq_len": 48,
            "holdout_mod": 10,
            "holdout_eval_buckets": 2,
            "sources": [{"name": "toy_text", "weight": 1.0, "toy_rows": 100}],
        },
        tok,
        n_batches=2,
        seed=31,
    )
    mat = run_causal_ablation_matrix(
        st["model"],
        eval_set,
        ablations=CANONICAL_ABLATIONS,
        n_repeats=2,
        depth=3,
        loop=st["loop"],
        gdn2=st["gdn2"],
        lsa=st["lsa"],
        ce_chunk_size=16,
        verify_sha256=True,
        arm_name="smoke_d3",
    )
    assert mat.all_restorations_verified is True
    assert [r.ablation for r in mat.rows] == list(CANONICAL_ABLATIONS)
    for row in mat.rows:
        assert row.recommendation in RECOMMENDATION_TAGS
        assert row.restoration_verified is True
        assert row.weight_sha256_before == row.weight_sha256_after == mat.full_weight_sha256
    assert mat.get_row("low_anchor_probe").recommendation == "SHORTCUT_DIAGNOSTIC"  # type: ignore[union-attr]
    # Every trained perturbation ablation must visibly shift NLL relative to full
    for name in ("no_gdn2", "no_qed", "no_neg_eigval", "no_pass_router", "low_anchor_probe"):
        row = mat.get_row(name)
        assert row is not None and abs(row.delta_nll_vs_full) > 1e-6
    sparse_row = mat.get_row("lsa_sparse")
    assert sparse_row is not None and sparse_row.lsa_last_recall is not None and sparse_row.lsa_last_recall > 0.0


# ==============================================================================
# 5. CLI, --smoke-cpu, Config+Checkpoint E2E & Hygiene Guards
# ==============================================================================


def test_smoke_cpu_cli_json_md_and_runtime(tmp_path: Path) -> None:
    out_json = tmp_path / "sprint2_smoke.json"
    out_md = tmp_path / "sprint2_smoke.md"
    t0 = time.perf_counter()
    rc = main(["--smoke-cpu", "--output-json", str(out_json), "--output-md", str(out_md)])
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 10.0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["mode"] == "smoke_cpu"
    assert "holdout" in payload["datasets"] and "ood" in payload["datasets"]
    assert payload["datasets"]["holdout"]["overlap_check"]["verified"] is True
    assert "looped_gdn2_lsa_d3" in payload["length_ladders"]
    assert payload["length_ladders"]["looped_gdn2_lsa_d3"]["all_states_bounded"] is True
    assert payload["causal_ablations"]["looped_gdn2_lsa_d3"]["all_restorations_verified"] is True

    md = out_md.read_text(encoding="utf-8")
    assert "## 1. Leak-Free Holdout" in md
    assert "## 2. Multi-Repeat Pure LM Depth Sweep" in md
    assert "## 3. Long-Context Length Ladder" in md
    assert "## 4. Zero-Training Causal Ablation Matrix" in md


def test_config_and_checkpoint_arms_e2e_with_ql60_010_cache(tmp_path: Path) -> None:
    cfg_dict: dict[str, Any] = {
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {"seed": 4, "num_hidden_layers": 16, "hidden_size": 32, "num_experts": 8},
        },
        "gdn2": {
            "enabled": True,
            "rank": 4,
            "chunk_size": 16,
            "qed_enabled": True,
            "neg_eigval_max_boost": 0.5,
            "backend": "pytorch",
        },
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "index_dim": 4,
            "block_size": 8,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "window": 8,
            "sinks": 2,
            "num_select": 2,
            "recall_check_every": 1,
        },
        "loop": {
            "enabled": True,
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 2,
            "target_virtual_layers": 24,
            "bridge_init": [0.05, 0.85, 0.10],
            "bridge_mode": "subspace",
            "bridge_rank": 4,
            "per_pass_router": True,
            "router_adapter_rank": 4,
        },
        "train": {"train_loop_adapters": True, "train_gdn2": True, "train_lsa": True},
        "data": {
            "seq_len": 48,
            "holdout_mod": 10,
            "holdout_eval_buckets": 2,
            "sources": [{"name": "toy_text", "weight": 1.0, "toy_rows": 80}],
        },
    }
    cfg_path = tmp_path / "toy_stage.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    # Save a perturbed trainable checkpoint
    st_train = build_stack(cfg_dict)
    with torch.no_grad():
        assert st_train.loop is not None
        st_train.loop.adapters.gates.fill_(0.15)
    run_dir = tmp_path / "run_cand"
    ckpt_mgr = CheckpointManager(str(run_dir))
    ckpt_mgr.save(10, st_train.checkpoint_params(), None, {"stage": "toy_s2", "stack": st_train.describe()})

    ood_file = tmp_path / "local_ood.jsonl"
    ood_lines = [
        json.dumps({"id": f"loc-{i}", "text": f"Local OOD validation sample number {i} " * 4})
        for i in range(6)
    ]
    ood_file.write_text("\n".join(ood_lines) + "\n", encoding="utf-8")

    arms = parse_arm_specs([f"stock=base,cand={run_dir}"], default_config=str(cfg_path))
    assert len(arms) == 2 and arms[0]["base_only"] is True and arms[1]["checkpoint"] == str(run_dir)

    cache_dir = tmp_path / "eval_cache"
    rep1 = run_sprint2_audit_from_config(
        config_path=str(cfg_path),
        arms=arms,
        depths=(1, 2),
        lengths=(48,),
        repeats=2,
        eval_batches=1,
        ood_files=[ood_file],
        chunk_size=24,
        ce_chunk_size=16,
        kv_cache_quant="native",
        baseline_arm="stock",
        ablation_arm="cand",
        cache_dir=cache_dir,
        seed=4,
    )
    assert "holdout" in rep1.depth_sweeps and "ood" in rep1.depth_sweeps
    assert "cand" in rep1.causal_ablations and rep1.causal_ablations["cand"].all_restorations_verified
    assert format_sprint2_markdown(rep1).startswith("# Sprint 2 Scientific Validation")

    # Second call with identical checkpoint hits QL60-010 cache cleanly
    rep2 = run_sprint2_audit_from_config(
        config_path=str(cfg_path),
        arms=arms,
        depths=(1, 2),
        lengths=(48,),
        repeats=2,
        eval_batches=1,
        ood_files=[ood_file],
        chunk_size=24,
        ce_chunk_size=16,
        kv_cache_quant="native",
        baseline_arm="stock",
        ablation_arm="cand",
        skip_length_ladder=True,
        skip_ablations=True,
        cache_dir=cache_dir,
        seed=4,
    )
    assert rep2.depth_sweeps["holdout"]["cand"].depths[2].mean_nll == pytest.approx(
        rep1.depth_sweeps["holdout"]["cand"].depths[2].mean_nll
    )

    # Mutating checkpoint weights on disk MUST trigger QL60-010 cache mismatch refusal!
    with torch.no_grad():
        st_train.loop.adapters.gates.fill_(0.45)
    ckpt_mgr.save(20, st_train.checkpoint_params(), None, {"stage": "toy_s2", "stack": st_train.describe()})
    with pytest.raises(ValueError, match="identity mismatch"):
        run_sprint2_audit_from_config(
            config_path=str(cfg_path),
            arms=arms,
            depths=(1, 2),
            lengths=(48,),
            repeats=2,
            eval_batches=1,
            ood_files=[ood_file],
            chunk_size=24,
            ce_chunk_size=16,
            kv_cache_quant="native",
            baseline_arm="stock",
            ablation_arm="cand",
            skip_length_ladder=True,
            skip_ablations=True,
            cache_dir=cache_dir,
            seed=4,
        )


def test_sprint2_audit_hygiene_no_forbidden_memory_or_user_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_text = (repo_root / "src" / "qloop" / "eval" / "sprint2_audit.py").read_text(encoding="utf-8")
    test_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_mem = "H" + "BM"
    forbidden_home = "/" + "Users/"
    assert forbidden_mem not in src_text.upper()
    assert forbidden_mem not in test_text.upper()
    assert forbidden_home not in src_text
    assert forbidden_home not in test_text


def test_s23_ablations_graceful_on_base_model_without_retrofits() -> None:
    base = build_smoke_cpu_stack(seed=5, num_passes=1, install_retrofit=False)
    tok = base["tokenizer"]
    eval_set = build_holdout_eval_batches(
        {
            "seq_len": 32,
            "holdout_mod": 10,
            "holdout_eval_buckets": 2,
            "sources": [{"name": "toy_text", "weight": 1.0, "toy_rows": 60}],
        },
        tok,
        n_batches=1,
        seed=5,
    )
    mat = run_causal_ablation_matrix(
        base["model"],
        eval_set,
        ablations=CANONICAL_ABLATIONS,
        n_repeats=1,
        ce_chunk_size=16,
        verify_sha256=True,
        arm_name="base_only",
    )
    assert mat.all_restorations_verified is True
    for name in ("no_gdn2", "no_qed", "no_neg_eigval", "no_pass_router", "low_anchor_probe"):
        row = mat.get_row(name)
        assert row is not None and row.applicable is False and row.delta_nll_vs_full == pytest.approx(0.0)


def test_run_smoke_cpu_direct_api() -> None:
    rep = run_smoke_cpu(seed=7, repeats=2, lengths=(64, 128), eval_batches=1, seq_len=48)
    d = rep.to_dict()
    assert d["mode"] == "smoke_cpu"
    assert d["causal_ablations"]["looped_gdn2_lsa_d3"]["all_restorations_verified"] is True
    assert d["length_ladders"]["looped_gdn2_lsa_d3"]["all_states_bounded"] is True
    # Confirm deterministic JSON serialization succeeds
    parsed = json.loads(rep.to_json())
    assert parsed["schema_version"] == d["schema_version"]


# ==============================================================================
# 6. Audit Blocker Regressions (B1-B5 + Gate 3/4 Hardening)
# ==============================================================================


def test_b1_compute_z_and_3sigma_zero_std_and_nonfinite_guards() -> None:
    # B1: Tiny fp noise (1e-12, 1e-16) with pooled_std=0.0 MUST NOT falsely pass 3-sigma!
    z_tiny, ex_tiny = compute_z_and_3sigma(1e-12, 0.0)
    assert z_tiny == pytest.approx(1e-3)
    assert ex_tiny is False

    z_eps, ex_eps = compute_z_and_3sigma(1e-16, 0.0)
    assert z_eps == pytest.approx(1e-7)
    assert ex_eps is False

    # Boundary at exactly 3.0x denom is False; strictly above 3.0x denom is True
    z_eq, ex_eq = compute_z_and_3sigma(3e-9, 0.0)
    assert z_eq == pytest.approx(3.0)
    assert ex_eq is False
    z_above, ex_above = compute_z_and_3sigma(1e-5, 0.0)
    assert z_above > 3.0 and ex_above is True

    # Non-finite inputs never pass
    z_nan, ex_nan = compute_z_and_3sigma(float("nan"), 0.0)
    assert math.isnan(z_nan) and ex_nan is False
    z_inf, ex_inf = compute_z_and_3sigma(float("inf"), 0.0)
    assert math.isnan(z_inf) and ex_inf is False


def test_b2_readonly_property_proxies_tf524_metadata_anslen1_bf16_and_cycles() -> None:
    from qloop.eval.sprint2_audit import (
        _init_streaming_cache,
        _iter_recurrent_state_tensors,
        _restore_and_crop_cache,
        _snapshot_linear_states,
    )
    from qloop.infer.engine import QuantizedHybridCache

    # 1) Read-only @property top-level list view + read-only @property layer slot proxy + TF 5.24 metadata
    class _ReadOnlySlotProxy:
        def __init__(self, backing: list[Any], idx: int) -> None:
            self._backing = backing
            self._idx = idx

        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> Any:
            if i != 0:
                raise IndexError(i)
            return self._backing[self._idx]

        def __setitem__(self, i: int, val: Any) -> None:
            if i != 0:
                raise IndexError(i)
            self._backing[self._idx] = val

    class _StrictTF524LinearLayer:
        def __init__(self, backing_conv: list[Any], backing_rec: list[Any], idx: int) -> None:
            self._backing_conv = backing_conv
            self._backing_rec = backing_rec
            self._idx = idx
            self.has_previous_state: dict[int, bool] = {0: True}
            self.is_conv_states_initialized: bool = True
            self.is_recurrent_states_initialized: bool = True
            self.conv_kernel_size: int = 4
            self.batch_size: int = 1

        @property
        def conv_states(self) -> _ReadOnlySlotProxy:
            return _ReadOnlySlotProxy(self._backing_conv, self._idx)

        @property
        def recurrent_states(self) -> _ReadOnlySlotProxy:
            return _ReadOnlySlotProxy(self._backing_rec, self._idx)

    class _StrictTF524FullLayer:
        def __init__(self, seq_len: int) -> None:
            self.keys = torch.ones((1, 2, seq_len, 8))
            self.values = torch.ones((1, 2, seq_len, 8))

    class _StrictTF524PropertyCache:
        def __init__(self, seq_len: int) -> None:
            self._conv_backing: list[Any] = [torch.full((1, 4, 2), 1.0), None]
            self._rec_backing: list[Any] = [torch.full((1, 2, 4, 4), 2.0), None]
            self.layers: list[Any] = [
                _StrictTF524LinearLayer(self._conv_backing, self._rec_backing, 0),
                _StrictTF524FullLayer(seq_len),
            ]

        @property
        def conv_states(self) -> list[Any]:
            # Fresh throwaway list view — mutating this directly would silently drop writes!
            return list(self._conv_backing)

        @property
        def recurrent_states(self) -> list[Any]:
            return list(self._rec_backing)

        def crop(self, max_length: int) -> None:
            # Simulate destructive crop that wipes linear states & metadata flags
            self.layers[1].keys = self.layers[1].keys[..., :max_length, :]
            self.layers[1].values = self.layers[1].values[..., :max_length, :]
            self._conv_backing[0] = torch.full((1, 4, 2), 77.0)
            self._rec_backing[0] = torch.full((1, 2, 4, 4), 99.0)
            lin = self.layers[0]
            lin.has_previous_state[0] = False
            lin.is_conv_states_initialized = False
            lin.is_recurrent_states_initialized = False
            lin.conv_kernel_size = 0
            lin.batch_size = 0

    strict_cache = _StrictTF524PropertyCache(seq_len=16)
    snap = _snapshot_linear_states(strict_cache)
    assert snap["direct_conv"] is None and snap["direct_rec"] is None
    assert len(snap["layer_states"]) == 1

    # Mutate & crop, then restore -> MUST restore underlying tensors & ALL TF 5.24 metadata without AttributeError!
    cropped = _restore_and_crop_cache(strict_cache, prompt_len=12, linear_snap=snap)
    assert cropped is True
    assert strict_cache.layers[1].keys.shape[-2] == 12
    assert torch.equal(strict_cache._conv_backing[0], torch.full((1, 4, 2), 1.0))
    assert torch.equal(strict_cache._rec_backing[0], torch.full((1, 2, 4, 4), 2.0))
    lin0 = strict_cache.layers[0]
    assert lin0.has_previous_state == {0: True}
    assert lin0.is_conv_states_initialized is True
    assert lin0.is_recurrent_states_initialized is True
    assert lin0.conv_kernel_size == 4
    assert lin0.batch_size == 1

    # 2) ans_len == 1 on a no-.crop() layer-only TF 5.24 cache stays cropped=True (no false fallback!)
    class _NoCropLayerCache:
        def __init__(self, seq_len: int) -> None:
            self.layers = [_StrictTF524FullLayer(seq_len)]

    nc_cache = _NoCropLayerCache(seq_len=16)
    assert _restore_and_crop_cache(nc_cache, prompt_len=16, linear_snap={"layer_states": []}) is True

    # 3) Cyclic self-referential container protection in `_iter_recurrent_state_tensors`
    class _CyclicProxy:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.items: list[Any] = [tensor, self]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Any:
            return self.items[idx]

    class _CyclicCache:
        def __init__(self) -> None:
            self.recurrent_states = [_CyclicProxy(torch.ones(2, 2))]

    extracted = _iter_recurrent_state_tensors(_CyclicCache())
    assert len(extracted) == 1

    # 4) `kv_cache_quant="bf16"` instantiates `QuantizedHybridCache(mode="bf16")` cleanly
    tiny = build_toy(seed=11)
    bf16_cache = _init_streaming_cache(tiny, "bf16")
    assert isinstance(bf16_cache, QuantizedHybridCache) and bf16_cache.mode == "bf16"
    ids = torch.randint(0, 97, (1, 32), generator=torch.Generator().manual_seed(11))
    nll_bf16, n_tok, out_cache, _, _ = stream_chunked_lm_nll(
        tiny, ids, ids, chunk_size=16, ce_chunk_size=16, kv_cache_quant="bf16"
    )
    assert n_tok == 31 and math.isfinite(nll_bf16)
    assert isinstance(out_cache, QuantizedHybridCache) and out_cache.mode == "bf16"


def test_b3_parse_arm_specs_base_at_yaml_and_step_tags() -> None:
    # 1) base@<cfg.yaml> MUST preserve base_only=True
    arms1 = parse_arm_specs(
        ["stock_qwen36_yarn4x=base@configs/stages/s2_v2_joint_bptt.yaml"],
        default_config="configs/base.yaml",
    )
    assert len(arms1) == 1
    assert arms1[0]["name"] == "stock_qwen36_yarn4x"
    assert arms1[0]["base_only"] is True
    assert arms1[0]["checkpoint"] is None
    assert arms1[0]["config"] == "configs/stages/s2_v2_joint_bptt.yaml"

    # 2) @600 step tag MUST NOT be misinterpreted as config path!
    arms2 = parse_arm_specs(
        ["s2_v2=runs/s2_v2_joint_bptt@600"],
        default_config="configs/stages/s2_v2_joint_bptt.yaml",
    )
    assert arms2[0]["checkpoint"] == "runs/s2_v2_joint_bptt@600"
    assert arms2[0]["config"] == "configs/stages/s2_v2_joint_bptt.yaml"
    assert arms2[0]["base_only"] is False

    # 3) hf://owner/repo@stage preserves full HF URI and optional trailing @config.yaml
    arms3 = parse_arm_specs(
        [
            "hub_s2=hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s2_v2_joint_bptt",
            "hub_s1=hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s1_loop_a6b@configs/stages/s1_loop_a6b.yaml",
        ],
        default_config="configs/stages/s2_v2_joint_bptt.yaml",
    )
    assert arms3[0]["checkpoint"] == "hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s2_v2_joint_bptt"
    assert arms3[0]["config"] == "configs/stages/s2_v2_joint_bptt.yaml"
    assert arms3[1]["checkpoint"] == "hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s1_loop_a6b"
    assert arms3[1]["config"] == "configs/stages/s1_loop_a6b.yaml"


def test_b4_load_arm_stack_stage_sync_readonly_and_hf_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qloop.eval.sprint2_audit import _load_arm_stack

    # Build an S1-only checkpoint (loop ONLY, NO gdn2, NO lsa)
    s1_cfg: dict[str, Any] = {
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {"seed": 9, "num_hidden_layers": 8, "hidden_size": 16, "full_attention_interval": 2,
                    "num_experts": 4},
        },
        "loop": {
            "enabled": True,
            "prelude_end": 2,
            "core_end": 6,
            "num_passes": 2,
            "target_virtual_layers": 12,
            "bridge_init": [0.05, 0.85, 0.10],
            "bridge_mode": "subspace",
            "bridge_rank": 4,
            "per_pass_router": True,
            "router_adapter_rank": 4,
        },
        "gdn2": {"enabled": False},
        "lsa": {"enabled": False},
        "train": {"train_loop_adapters": True},
    }
    st_s1 = build_stack(s1_cfg)
    with torch.no_grad():
        assert st_s1.loop is not None
        st_s1.loop.adapters.gates.fill_(0.25)
    ids = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(9))
    st_s1.model.eval()
    with torch.no_grad():
        s1_ref_logits = st_s1.model(ids, use_cache=False).logits.clone()

    s1_run = tmp_path / "run_s1"
    s1_step_dir = CheckpointManager(str(s1_run)).save(
        10, st_s1.checkpoint_params(), None, {"stage": "s1_loop_a6b", "stack": st_s1.describe()}
    )

    # S2-v2 default config has GDN-2 + sparse D-LSA enabled!
    s2_cfg = dict(s1_cfg)
    s2_cfg["gdn2"] = {"enabled": True, "rank": 4, "chunk_size": 16, "qed_enabled": True, "backend": "pytorch"}
    s2_cfg["lsa"] = {
        "enabled": True,
        "mode": "sparse",
        "index_dim": 4,
        "block_size": 8,
        "pool": 2,
        "avg_bucket_keys": 4,
        "max_bucket_keys": 8,
        "window": 8,
        "sinks": 2,
        "num_select": 2,
    }
    s2_cfg_path = tmp_path / "s2_v2.yaml"
    s2_cfg_path.write_text(yaml.safe_dump(s2_cfg), encoding="utf-8")

    # 1) Loading S1 checkpoint under S2 config MUST auto-disable GDN-2 and LSA -> bit-exact S1 logits!
    loaded_s1, resolved_s1 = _load_arm_stack({"name": "s1", "config": str(s2_cfg_path), "checkpoint": str(s1_run)})
    assert loaded_s1.gdn2 is None and loaded_s1.lsa is None and loaded_s1.loop is not None
    assert resolved_s1 == s1_step_dir
    with torch.no_grad():
        loaded_logits = loaded_s1.model(ids, use_cache=False).logits
    assert torch.equal(loaded_logits, s1_ref_logits)

    # 2) Direct `.pt` file loads sibling `state.json` AND `<run>/checkpoints` dir resolves read-only
    direct_pt = Path(s1_step_dir) / "trainable.pt"
    loaded_pt, _ = _load_arm_stack({"name": "s1_pt", "config": str(s2_cfg_path), "checkpoint": str(direct_pt)})
    assert loaded_pt.gdn2 is None and loaded_pt.lsa is None
    loaded_ckdir, _ = _load_arm_stack(
        {"name": "s1_ck", "config": str(s2_cfg_path), "checkpoint": str(s1_run / "checkpoints")}
    )
    assert loaded_ckdir.gdn2 is None and loaded_ckdir.lsa is None

    # 3) Non-run directory raises FileNotFoundError WITHOUT creating a side-effect `checkpoints/` folder!
    empty_dir = tmp_path / "not_a_run_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_arm_stack({"name": "bad", "config": str(s2_cfg_path), "checkpoint": str(empty_dir)})
    assert list(empty_dir.iterdir()) == []

    # 4) `hf://owner/repo@stage` delegates cleanly to `qloop.hub.load_qloop_from_hub`
    import qloop.hub as hub_mod

    called: dict[str, Any] = {}

    def _fake_hub_load(repo_id: str, *, stage_name: str | None = None, **kwargs: Any) -> Any:
        called["repo_id"] = repo_id
        called["stage_name"] = stage_name
        return st_s1

    monkeypatch.setattr(hub_mod, "load_qloop_from_hub", _fake_hub_load)
    hub_st, hub_lbl = _load_arm_stack(
        {"name": "hub_s2", "checkpoint": "hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s2_v2_joint_bptt"}
    )
    assert hub_st is st_s1
    assert called == {"repo_id": "TassP0lar/Qwen3.6-35B-A3B-Loop64", "stage_name": "s2_v2_joint_bptt"}
    assert hub_lbl == "hf://TassP0lar/Qwen3.6-35B-A3B-Loop64@s2_v2_joint_bptt"


def test_b5_eval_seq_len_cap_disables_decontam_and_verifies_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qloop.data.mixture as mix_mod

    def _forbid_decontam(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("eval-only holdout stream MUST NOT invoke decontaminator benchmark downloads!")

    monkeypatch.setattr(mix_mod, "default_eval_providers", _forbid_decontam)

    cfg_dict: dict[str, Any] = {
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {"seed": 2, "num_hidden_layers": 8, "hidden_size": 16, "full_attention_interval": 2,
                    "num_experts": 4},
        },
        "loop": {"enabled": False},
        "data": {
            "seq_len": 131072,
            "holdout_mod": 10,
            "holdout_eval_buckets": 2,
            "sources": [{"name": "toy_text", "weight": 1.0, "toy_rows": 80}],
            "decontaminate": {"enabled": True, "ngram": 13, "groups": ["aime_2026"]},
        },
    }
    cfg_path = tmp_path / "long_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    rep = run_sprint2_audit_from_config(
        config_path=str(cfg_path),
        arms=[{"name": "stock", "config": str(cfg_path), "base_only": True}],
        depths=(1,),
        repeats=1,
        eval_batches=1,
        eval_seq_len=48,
        verify_train_overlap=True,
        skip_length_ladder=True,
        skip_ablations=True,
        seed=2,
    )
    assert rep.metadata["eval_seq_len"] == 48
    assert rep.datasets["holdout"]["metadata"]["seq_len"] == 48
    assert rep.datasets["holdout"]["overlap_check"]["verified"] is True
    assert rep.datasets["holdout"]["overlap_check"]["overlap_count"] == 0


def test_minor_gate4_ablate_lsa_mode_autodiscovers_manager_when_lsa_none() -> None:
    st = build_smoke_cpu_stack(seed=41, num_passes=2, trained_perturbation=True)
    model = st["model"]
    ids = torch.randint(0, len(st["tokenizer"]), (1, 32), generator=torch.Generator().manual_seed(41))
    with ablate_lsa_mode(model, "sparse", lsa=None, sample_recall=True) as info, torch.no_grad():
        model(ids, use_cache=False)
    assert info["applicable"] is True
    assert "lsa_last_recall" in info and 0.0 < float(info["lsa_last_recall"]) <= 1.0
