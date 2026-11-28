"""Comprehensive CPU tests for `GUARD G4` Unified Dual-Loop `LSA-NL` Fast Outer Loop Controller."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from qloop.memory import (
    EpisodicSideMemoryBank,
    ESMConfig,
    ESMLedgerEntry,
    GuardGraduationResult,
    GuardOuterConfig,
    GuardProposalResult,
    KVBlobConfig,
    KVMemBlob,
    LSANLGuardOuterLoop,
    SoftHurtGateConfig,
    SoftHurtOutcome,
    SplitHorizonKVBlobStore,
    evaluate_soft_hurt,
    pack_kv_blob,
)
from qloop.memory.engram import AnchoredMoEEngramBank, EngramConfig
from qloop.train.sirsi import (
    REGRET_NEWLY_HURT,
    REGRET_PINNED_PASS,
    REGRET_UNSOLVED_CHRONIC,
    TaskRegretTracker,
)


def _make_kv_cfg(**overrides: Any) -> KVBlobConfig:
    defaults: dict[str, Any] = {
        "num_global_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 128,
        "rotary_dim": 64,
        "blob_tokens": 8,
        "index_dim": 32,
        "salience_threshold": 0.25,
        "dedup_cos_threshold": 0.90,
        "min_retention": 0.15,
        "initial_stability": 1.0,
        "hardening_factor": 2.0,
        "graduation_hits": 2,
        "graduation_min_stability": 4.0,
    }
    defaults.update(overrides)
    return KVBlobConfig(**defaults)


def _make_blob(
    blob_id: str,
    centroid_dir: torch.Tensor,
    *,
    cfg: KVBlobConfig,
    t_created: float = 0.0,
    salience: float = 0.85,
    tags: tuple[str, ...] = (),
    with_gdn2: bool = False,
) -> KVMemBlob:
    k = torch.randn(cfg.num_global_layers, cfg.num_kv_heads, cfg.blob_tokens, cfg.head_dim)
    v = torch.randn_like(k)
    c_mat = F.normalize(centroid_dir.float(), dim=-1).unsqueeze(0).expand(cfg.num_global_layers, -1).contiguous()
    blob = pack_kv_blob(
        k,
        v,
        blob_id=blob_id,
        config=cfg,
        coarse_centroid=c_mat,
        t_created=t_created,
        t_valid=t_created,
        salience=salience,
        topic_tags=tags,
    )
    if with_gdn2:
        blob.gdn2_recurrent_states = {0: torch.randn(1, 2, 8, 8, dtype=torch.float32)}
        blob.gdn2_conv_states = {0: torch.randn(1, 16, 4, dtype=torch.float32)}
    return blob


def _make_esm_bank(d_model: int = 32, r_hidden: int = 96, layers: tuple[int, ...] = (15, 19)) -> EpisodicSideMemoryBank:
    cfg = ESMConfig(
        d_model=d_model,
        r_hidden=r_hidden,
        anchor_layers=layers,
        scope_threshold=0.65,
        ridge_alpha=1e-4,
        max_rel_delta_norm=0.03,
        seed=2026,
    )
    return EpisodicSideMemoryBank(cfg)


# ==============================================================================
# 1. `Soft-HURT` Math Edge Cases (100% Parity with `G1` in `src/qloop/train/rl.py`)
# ==============================================================================


def test_soft_hurt_math_clamping_asymmetry_caps_and_bootstrap() -> None:
    cfg = SoftHurtGateConfig(
        lambda_hurt=2.0,
        min_soft_hurt=0.0,
        max_hurt_frac=0.05,
        max_bench_drop=0.025,
        pass_threshold=0.5,
    )

    # 1a. Empty baseline -> bootstrap promotes cleanly and populates pinned_passed_ids
    boot = evaluate_soft_hurt(
        {},
        {"a0": 1.0, "a1": 0.0, "a2": 1.7},
        cfg=cfg,
        bench_map={"a0": "m", "a1": "m", "a2": "m"},
    )
    assert isinstance(boot, SoftHurtOutcome)
    assert boot.promoted is True
    assert boot.n_items == 3
    assert boot.gain_mean == 0.0 and boot.hurt_mean == 0.0 and boot.soft_hurt_score == 0.0
    assert boot.pinned_passed_ids == ("a0", "a2")
    assert boot.newly_hurt_ids == () and boot.newly_fixed_ids == ()
    assert boot.bench_deltas == {"m": 0.0}
    assert boot.rejection_reasons == ()

    # 1b. Clamping `[0, 1]`: bonus score > 1.0 (e.g., speedup 1.8) cannot mask a regression, negative clamped to 0
    base_clamp = {"i0": 1.0, "i1": 1.0, "i2": 0.0, "i3": 0.0}
    cand_clamp = {"i0": 1.8, "i1": 0.0, "i2": -0.5, "i3": 0.0}
    out_clamp = evaluate_soft_hurt(base_clamp, cand_clamp, cfg=cfg)
    assert out_clamp.promoted is False
    assert out_clamp.gain_mean == pytest.approx(0.0)
    assert out_clamp.hurt_mean == pytest.approx(0.25)
    assert out_clamp.soft_hurt_score == pytest.approx(-0.5)
    assert out_clamp.newly_hurt_ids == ("i1",)
    assert out_clamp.pinned_passed_ids == ("i0",)

    # 1c. Asymmetric lambda_hurt=2.0 rejects even when RAW mean increases!
    # 100 items: 3 fixed (+0.03), 2 hurt (-0.02) => raw delta = +0.01 > 0, hurt_frac = 0.02 <= 0.05,
    # BUT soft_hurt = 0.03 - 2.0 * 0.02 = -0.01 < 0 -> REJECTED!
    base_100 = {f"t{i}": (1.0 if i < 50 else 0.0) for i in range(100)}
    cand_raw_up = dict(base_100)
    cand_raw_up["t50"] = cand_raw_up["t51"] = cand_raw_up["t52"] = 1.0
    cand_raw_up["t0"] = cand_raw_up["t1"] = 0.0
    out_asym = evaluate_soft_hurt(base_100, cand_raw_up, cfg=cfg)
    assert out_asym.promoted is False
    assert out_asym.gain_mean == pytest.approx(0.03)
    assert out_asym.hurt_mean == pytest.approx(0.02)
    assert out_asym.hurt_frac == pytest.approx(0.02)
    assert out_asym.soft_hurt_score == pytest.approx(-0.01)
    assert out_asym.newly_hurt_ids == ("t0", "t1")
    assert out_asym.newly_fixed_ids == ("t50", "t51", "t52")
    assert any("soft_hurt_non_positive" in r for r in out_asym.rejection_reasons)

    # 1d. Exact zero tie with hurt > 0 (2 fixed, 1 hurt => soft_hurt = 0.0) -> REJECTED (G1 parity!)
    base_20 = {f"a{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    cand_tie = {**base_20, "a0": 0.0, "a10": 1.0, "a11": 1.0}
    out_tie = evaluate_soft_hurt(base_20, cand_tie, cfg=cfg)
    assert out_tie.promoted is False
    assert out_tie.soft_hurt_score == pytest.approx(0.0)
    assert any("soft_hurt_non_positive" in r for r in out_tie.rejection_reasons)

    # 1e. `max_hurt_frac` rejection despite positive `soft_hurt_score`:
    # 20 items: 6 fixed (+0.30), 2 hurt (-0.10) => soft_hurt = +0.10 > 0, BUT hurt_frac = 0.10 > 0.05 -> REJECTED
    cand_hf = {**base_20, "a0": 0.0, "a1": 0.0, "a10": 1.0, "a11": 1.0, "a12": 1.0, "a13": 1.0, "a14": 1.0, "a15": 1.0}
    out_hf = evaluate_soft_hurt(base_20, cand_hf, cfg=cfg)
    assert out_hf.promoted is False
    assert out_hf.soft_hurt_score == pytest.approx(0.10)
    assert out_hf.hurt_frac == pytest.approx(0.10)
    assert any("hurt_frac_exceeded" in r for r in out_hf.rejection_reasons)

    # 1f. `max_bench_drop` rejection despite positive overall `soft_hurt_score` and `hurt_frac <= 0.05`:
    # bench_a (10 items): +4 fixed (+0.40); bench_b (10 items): -1 hurt (-0.10 > 0.025)
    bench_map = {**{f"a{i}": "bench_a" for i in range(10)}, **{f"b{i}": "bench_b" for i in range(10)}}
    base_2b = {
        **{f"a{i}": (1.0 if i < 5 else 0.0) for i in range(10)},
        **{f"b{i}": (1.0 if i < 5 else 0.0) for i in range(10)},
    }
    cand_bdrop = {**base_2b, "a5": 1.0, "a6": 1.0, "a7": 1.0, "a8": 1.0, "b0": 0.0}
    out_bdrop = evaluate_soft_hurt(base_2b, cand_bdrop, cfg=cfg, bench_map=bench_map)
    assert out_bdrop.promoted is False
    assert out_bdrop.soft_hurt_score == pytest.approx(0.10)
    assert out_bdrop.hurt_frac == pytest.approx(0.05)
    assert out_bdrop.bench_deltas["bench_a"] == pytest.approx(0.40)
    assert out_bdrop.bench_deltas["bench_b"] == pytest.approx(-0.10)
    assert out_bdrop.max_bench_drop_seen == pytest.approx(0.10)
    assert any("bench_drop_exceeded" in r and "bench_b" in r for r in out_bdrop.rejection_reasons)

    # 1g. JSON-serializable diff dict & no-common-items guard
    diff = out_bdrop.to_diff_dict()
    assert json.loads(json.dumps(diff))["newly_hurt_ids"] == ["b0"]
    out_disjoint = evaluate_soft_hurt({"x0": 1.0}, {"y0": 1.0}, cfg=cfg)
    assert out_disjoint.promoted is False
    assert out_disjoint.rejection_reasons == ("no_common_items",)


# ==============================================================================
# 2. Tier-1 `propose_kv_rule`: Promote, GDN-2 Preservation, Salience & Soft-HURT Reject
# ==============================================================================


def test_tier1_propose_kv_rule_promote_gdn2_salience_and_soft_hurt_reject() -> None:
    torch.manual_seed(7)
    kv_cfg = _make_kv_cfg(salience_threshold=0.30)
    store = SplitHorizonKVBlobStore(config=kv_cfg)
    loop = LSANLGuardOuterLoop(store, config=GuardOuterConfig())

    base_scores = {f"q{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    bench_map = {f"q{i}": ("math" if i < 10 else "code") for i in range(20)}
    loop.initialize_baseline(base_scores, bench_map=bench_map)
    assert len(loop.pinned_passed) == 10

    basis = torch.eye(32, dtype=torch.float32)

    # 2a. Seed an initial active blob so we can test `supersede_ids` and `semantic_links`
    blob_v1 = _make_blob("rule_v1", basis[0], cfg=kv_cfg, t_created=1.0, salience=0.80, tags=("algebra",))
    blob_anchor = _make_blob("anchor_rule", basis[1], cfg=kv_cfg, t_created=1.0, salience=0.80, tags=("number_theory",))
    assert store.admit(blob_v1) is True
    assert store.admit(blob_anchor) is True

    # 2b. Promotion with GDN-2 states + supersede_ids + semantic_links
    blob_v2 = _make_blob(
        "rule_v2",
        basis[0] + 0.05 * basis[2],
        cfg=kv_cfg,
        t_created=2.0,
        salience=0.90,
        tags=("algebra", "verified"),
        with_gdn2=True,
    )
    assert blob_v2.gdn2_recurrent_states is not None and blob_v2.gdn2_conv_states is not None
    rec_snap = blob_v2.gdn2_recurrent_states[0].clone()
    conv_snap = blob_v2.gdn2_conv_states[0].clone()

    cand_good = {**base_scores, "q10": 1.0, "q11": 1.0}
    res_ok = loop.propose_kv_rule(
        blob_v2,
        eval_fn=lambda _b: cand_good,
        supersede_ids=("rule_v1",),
        semantic_links=(("anchor_rule", 0.77),),
        rule_meta={"source": "outer_test", "prompt": "SECRET_PROMPT_MUST_NOT_LEAK"},
        now=2.5,
    )
    assert isinstance(res_ok, GuardProposalResult)
    assert res_ok.promoted is True and res_ok.admitted is True
    assert res_ok.has_gdn2_state is True
    assert res_ok.superseded_ids == ("rule_v1",)
    assert res_ok.semantic_links_added == (("anchor_rule", 0.77),)
    assert "prompt" not in res_ok.rule_meta
    assert "rule_v2" in store.blobs
    # GDN-2 recurrent & conv states preserved bit-identically!
    stored_v2 = store.blobs["rule_v2"]
    assert stored_v2.gdn2_recurrent_states is not None and stored_v2.gdn2_conv_states is not None
    assert torch.equal(stored_v2.gdn2_recurrent_states[0], rec_snap)
    assert torch.equal(stored_v2.gdn2_conv_states[0], conv_snap)
    # Supersession tombstoned rule_v1 and semantic link added in DLMG
    assert store.blobs["rule_v1"].header.is_active(2.5) is False
    assert store.graph.semantic_edges["rule_v2"]["anchor_rule"] == pytest.approx(0.77)
    assert loop.baseline_items["q10"] == 1.0 and loop.baseline_items["q11"] == 1.0
    assert {"q10", "q11"}.issubset(loop.pinned_passed)
    assert len(loop.history) == 1

    # 2c. Salience-gate rejection: Soft-HURT passes, but salience < 0.30 -> admitted=False, baseline UNCHANGED!
    blob_low_sal = _make_blob("low_sal_rule", basis[3], cfg=kv_cfg, t_created=3.0, salience=0.10)
    cand_low_sal = {**loop.baseline_items, "q12": 1.0}
    baseline_before = dict(loop.baseline_items)
    pinned_before = set(loop.pinned_passed)
    history_len_before = len(loop.history)

    res_sal = loop.propose_kv_rule(blob_low_sal, eval_fn=lambda _b: cand_low_sal, now=3.0)
    assert res_sal.outcome.promoted is True
    assert res_sal.admitted is False
    assert res_sal.promoted is False
    assert res_sal.rejection_reasons == ("kv_salience_rejected",)
    assert "low_sal_rule" not in store.blobs
    assert loop.baseline_items == baseline_before
    assert loop.pinned_passed == pinned_before
    assert len(loop.history) == history_len_before

    # 2d. Soft-HURT rejection: kv_store & baseline untouched; newly_hurt_ids tracked in regret_tracker & history
    blob_bad = _make_blob("regressing_rule", basis[4], cfg=kv_cfg, t_created=4.0, salience=0.95)
    cand_bad = {**loop.baseline_items, "q0": 0.0, "q1": 0.0, "q13": 1.0}
    store_ids_before = set(store.blobs.keys())
    res_bad = loop.propose_kv_rule(blob_bad, eval_fn=lambda _b: cand_bad, now=4.0)
    assert res_bad.promoted is False and res_bad.admitted is False
    assert res_bad.outcome.promoted is False
    assert res_bad.outcome.newly_hurt_ids == ("q0", "q1")
    assert set(store.blobs.keys()) == store_ids_before
    assert loop.baseline_items == baseline_before
    assert len(loop.history) == history_len_before + 1
    assert loop.regret_tracker.states["q0"].last_pass_rate == 0.0
    assert loop.regret_tracker.states["q0"].best_pass_rate == 1.0
    assert loop.regret_tracker.states["q1"].last_pass_rate == 0.0


# ==============================================================================
# 3. Tier-2 `graduate_kv_to_esm`: Real `ESM` Promote, Multi-Layer Rollback, Exception Safety
# ==============================================================================


def test_tier2_graduate_kv_to_esm_promote_rollback_and_exception_safety() -> None:
    torch.manual_seed(2026)
    d_model = 32
    r_hidden = 96
    kv_cfg = _make_kv_cfg(graduation_hits=2, graduation_min_stability=4.0)
    store = SplitHorizonKVBlobStore(config=kv_cfg)
    esm_bank = _make_esm_bank(d_model=d_model, r_hidden=r_hidden, layers=(15, 19))

    # Protect general-corpus subspace on both layers
    h_general = torch.zeros(12, d_model, dtype=torch.float32)
    h_general[:, :16] = torch.randn(12, 16)
    for lyr in (15, 19):
        esm_bank.register_general_subspace(lyr, h_general, energy_threshold=0.995)

    loop = LSANLGuardOuterLoop(
        store,
        esm_bank=esm_bank,
        config=GuardOuterConfig(graduation_max_per_cycle=4),
    )
    base_scores = {f"m{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    loop.initialize_baseline(base_scores, bench_map={f"m{i}": "aime" for i in range(20)})

    # Require esm_bank error check on a bankless controller
    no_esm_loop = LSANLGuardOuterLoop(store)
    with pytest.raises(ValueError, match="esm_bank"):
        no_esm_loop.graduate_kv_to_esm(compile_fn=lambda _b, _e: [], eval_fn=lambda _b, _e: base_scores)

    basis = torch.eye(32, dtype=torch.float32)
    b_promote = _make_blob("grad_good", basis[0], cfg=kv_cfg, t_created=0.0, salience=0.90, tags=("calculus",))
    b_reject = _make_blob("grad_bad", basis[1], cfg=kv_cfg, t_created=0.0, salience=0.85, tags=("geometry",))
    b_crash = _make_blob("grad_crash", basis[2], cfg=kv_cfg, t_created=0.0, salience=0.80, tags=("algebra",))

    assert store.admit(b_promote) is True
    assert store.admit(b_reject) is True
    assert store.admit(b_crash) is True

    # Harden all three so they qualify for graduation (`hits >= 2, stability >= 4.0`)
    for bid in ("grad_good", "grad_bad", "grad_crash"):
        store.record_hit(bid, now=1.0)
        store.record_hit(bid, now=2.0)

    def _compile_multi_layer(blob: KVMemBlob, bank: EpisodicSideMemoryBank) -> list[tuple[int, ESMLedgerEntry]]:
        entries: list[tuple[int, ESMLedgerEntry]] = []
        for idx, lyr in enumerate((15, 19)):
            h_trig = torch.zeros(3, d_model, dtype=torch.float32)
            h_trig[:, 16:] = F.normalize(torch.randn(3, 16), dim=-1) * 10.0
            y_tgt = F.normalize(torch.randn(3, d_model), dim=-1) * 0.15
            entry, leak = bank.compile_nullspace_ridge(
                lyr, h_trig, y_tgt, entry_id=f"{blob.blob_id}_L{lyr}_{idx}"
            )
            assert leak < 1e-12
            entries.append((lyr, entry))
        return entries

    # 3a. Promotion case: graduate ONLY `grad_good` first (via graduation_max_per_cycle=1, highest salience)
    loop.config.graduation_max_per_cycle = 1
    res_prom_list = loop.graduate_kv_to_esm(
        compile_fn=_compile_multi_layer,
        eval_fn=lambda _b, _e: {**loop.baseline_items, "m10": 1.0, "m11": 1.0},
        now=3.0,
    )
    assert len(res_prom_list) == 1
    res_prom = res_prom_list[0]
    assert isinstance(res_prom, GuardGraduationResult)
    assert res_prom.blob_id == "grad_good"
    assert res_prom.promoted is True
    assert res_prom.superseded_in_kv is True
    assert res_prom.max_post_leakage < 1e-12
    assert len(res_prom.compiled_entries) == 2
    assert res_prom.rolled_back_entries == ()
    # Blob tombstoned in kv_store, ESM weights retained, graduated_rules populated
    assert store.blobs["grad_good"].header.is_active(3.0) is False
    assert store.blobs["grad_good"].header.is_active() is False
    assert float(esm_bank.anchor(15).w_down.abs().max()) > 0.0
    assert float(esm_bank.anchor(19).w_down.abs().max()) > 0.0
    assert len(loop.graduated_rules) == 1
    assert loop.baseline_items["m10"] == 1.0 and loop.baseline_items["m11"] == 1.0

    # Snapshot ESM weights & basis across BOTH layers before rejection test
    snap_w15 = esm_bank.anchor(15).w_down.detach().clone()
    snap_w19 = esm_bank.anchor(19).w_down.detach().clone()
    snap_b15 = esm_bank.anchor(15).active_basis.detach().clone()
    snap_b19 = esm_bank.anchor(19).active_basis.detach().clone()
    snap_p15 = esm_bank.anchor(15).scope_prototypes.detach().clone()
    snap_p19 = esm_bank.anchor(19).scope_prototypes.detach().clone()

    # 3b. Rejection case: `grad_bad` is next highest salience; multi-layer ESM write MUST roll back `< 1e-6`!
    res_rej_list = loop.graduate_kv_to_esm(
        compile_fn=_compile_multi_layer,
        eval_fn=lambda _b, _e: {**loop.baseline_items, "m0": 0.0, "m1": 0.0, "m12": 1.0},
        now=4.0,
    )
    assert len(res_rej_list) == 1
    res_rej = res_rej_list[0]
    assert res_rej.blob_id == "grad_bad"
    assert res_rej.promoted is False
    assert res_rej.superseded_in_kv is False
    # Rolled back in exact reverse order!
    assert res_rej.rolled_back_entries == tuple(reversed(res_rej.compiled_entries))
    assert float((esm_bank.anchor(15).w_down - snap_w15).abs().max()) < 1e-6
    assert float((esm_bank.anchor(19).w_down - snap_w19).abs().max()) < 1e-6
    assert torch.equal(esm_bank.anchor(15).active_basis, snap_b15)
    assert torch.equal(esm_bank.anchor(19).active_basis, snap_b19)
    assert torch.equal(esm_bank.anchor(15).scope_prototypes, snap_p15)
    assert torch.equal(esm_bank.anchor(19).scope_prototypes, snap_p19)
    # Rejected blob stays ACTIVE in kv_store
    assert store.blobs["grad_bad"].header.is_active(4.0) is True
    assert loop.baseline_items["m0"] == 1.0

    # 3c. Exception-in-`eval_fn` safety: MUST still roll back multi-layer ESM cleanly before propagating!
    # Give `grad_crash` an extra hit so it sorts ahead of `grad_bad`
    store.record_hit("grad_crash", now=4.5)

    def _exploding_eval(_b: KVMemBlob, _e: EpisodicSideMemoryBank) -> dict[str, float]:
        raise ValueError("synthetic eval failure mid-graduation")

    with pytest.raises(ValueError, match="synthetic eval failure"):
        loop.graduate_kv_to_esm(
            compile_fn=_compile_multi_layer,
            eval_fn=_exploding_eval,
            now=5.0,
        )

    assert float((esm_bank.anchor(15).w_down - snap_w15).abs().max()) < 1e-6
    assert float((esm_bank.anchor(19).w_down - snap_w19).abs().max()) < 1e-6
    assert torch.equal(esm_bank.anchor(15).active_basis, snap_b15)
    assert torch.equal(esm_bank.anchor(19).active_basis, snap_b19)
    assert store.blobs["grad_crash"].header.is_active(5.0) is True
    assert loop.history[-1]["kind"] == "esm_graduation_exception_rollback"
    assert loop.history[-1]["blob_id"] == "grad_crash"
    assert loop.history[-1]["error_type"] == "ValueError"
    # General-corpus tokens STILL suffer exact 0.0 WISE drift after promote + rollback + crash-rollback
    assert torch.equal(esm_bank(15, h_general), torch.zeros_like(h_general))
    assert torch.equal(esm_bank(19, h_general), torch.zeros_like(h_general))


def test_tier2_graduate_partial_compile_and_malformed_eval_rollback() -> None:
    torch.manual_seed(303)
    d_model = 32
    r_hidden = 96
    kv_cfg = _make_kv_cfg(graduation_hits=1, graduation_min_stability=2.0)
    store = SplitHorizonKVBlobStore(config=kv_cfg)
    esm_bank = _make_esm_bank(d_model=d_model, r_hidden=r_hidden, layers=(15, 19))

    # Pre-seed an existing promoted entry on layer 15 so rollback must preserve prior state cleanly
    h_pre = F.normalize(torch.randn(2, d_model), dim=-1) * 8.0
    y_pre = F.normalize(torch.randn(2, d_model), dim=-1) * 0.10
    esm_bank.compile_nullspace_ridge(15, h_pre, y_pre, entry_id="pre_existing_L15")

    loop = LSANLGuardOuterLoop(
        store,
        esm_bank=esm_bank,
        config=GuardOuterConfig(graduation_max_per_cycle=1),
    )
    loop.initialize_baseline({"m0": 1.0, "m1": 0.0}, bench_map={"m0": "aime", "m1": "aime"})

    basis = torch.eye(32, dtype=torch.float32)
    b_target = _make_blob("b_partial", basis[0], cfg=kv_cfg, t_created=0.0, salience=0.90)
    assert store.admit(b_target) is True
    store.record_hit("b_partial", now=1.0)

    snap_w15 = esm_bank.anchor(15).w_down.detach().clone()
    snap_w19 = esm_bank.anchor(19).w_down.detach().clone()
    snap_b15 = esm_bank.anchor(15).active_basis.detach().clone()
    snap_b19 = esm_bank.anchor(19).active_basis.detach().clone()
    snap_p15 = esm_bank.anchor(15).scope_prototypes.detach().clone()
    snap_p19 = esm_bank.anchor(19).scope_prototypes.detach().clone()
    active_l15_before = [e.entry_id for e in esm_bank.anchor(15).ledger if e.active]
    active_l19_before = [e.entry_id for e in esm_bank.anchor(19).ledger if e.active]

    # (a) Partial multi-layer `compile_fn` crash: layer 15 compiles, layer 19 raises before returning
    def _partial_crashing_compile(blob: KVMemBlob, bank: EpisodicSideMemoryBank) -> list[tuple[int, ESMLedgerEntry]]:
        h_trig = F.normalize(torch.randn(2, d_model), dim=-1) * 8.0
        y_tgt = F.normalize(torch.randn(2, d_model), dim=-1) * 0.10
        bank.compile_nullspace_ridge(15, h_trig, y_tgt, entry_id=f"{blob.blob_id}_L15_partial")
        raise RuntimeError("layer 19 compile failed mid-transaction")

    with pytest.raises(RuntimeError, match="layer 19 compile failed"):
        loop.graduate_kv_to_esm(
            compile_fn=_partial_crashing_compile,
            eval_fn=lambda _b, _e: {"m0": 1.0, "m1": 1.0},
            now=2.0,
        )

    assert float((esm_bank.anchor(15).w_down - snap_w15).abs().max()) < 1e-6
    assert float((esm_bank.anchor(19).w_down - snap_w19).abs().max()) < 1e-6
    assert torch.equal(esm_bank.anchor(15).active_basis, snap_b15)
    assert torch.equal(esm_bank.anchor(19).active_basis, snap_b19)
    assert torch.equal(esm_bank.anchor(15).scope_prototypes, snap_p15)
    assert torch.equal(esm_bank.anchor(19).scope_prototypes, snap_p19)
    assert [e.entry_id for e in esm_bank.anchor(15).ledger if e.active] == active_l15_before
    assert [e.entry_id for e in esm_bank.anchor(19).ledger if e.active] == active_l19_before
    assert store.blobs["b_partial"].header.is_active(2.0) is True
    assert loop.history[-1]["kind"] == "esm_graduation_exception_rollback"
    assert loop.history[-1]["rolled_back_entries"] == [[15, "b_partial_L15_partial"]]

    # (b) Malformed `eval_fn` payloads (`None` and non-numeric score) roll back multi-layer writes cleanly
    compile_counter = {"n": 0}

    def _ok_multi_compile(blob: KVMemBlob, bank: EpisodicSideMemoryBank) -> list[tuple[int, ESMLedgerEntry]]:
        compile_counter["n"] += 1
        tag = compile_counter["n"]
        out: list[tuple[int, ESMLedgerEntry]] = []
        for lyr in (15, 19):
            h_trig = F.normalize(torch.randn(2, d_model), dim=-1) * 8.0
            y_tgt = F.normalize(torch.randn(2, d_model), dim=-1) * 0.10
            entry, _ = bank.compile_nullspace_ridge(lyr, h_trig, y_tgt, entry_id=f"{blob.blob_id}_L{lyr}_m{tag}")
            out.append((lyr, entry))
        return out

    with pytest.raises(AttributeError):
        loop.graduate_kv_to_esm(
            compile_fn=_ok_multi_compile,
            eval_fn=lambda _b, _e: None,  # type: ignore[arg-type,return-value]
            now=3.0,
        )

    assert float((esm_bank.anchor(15).w_down - snap_w15).abs().max()) < 1e-6
    assert float((esm_bank.anchor(19).w_down - snap_w19).abs().max()) < 1e-6
    assert [e.entry_id for e in esm_bank.anchor(15).ledger if e.active] == active_l15_before
    assert [e.entry_id for e in esm_bank.anchor(19).ledger if e.active] == active_l19_before
    assert store.blobs["b_partial"].header.is_active(3.0) is True

    with pytest.raises(ValueError):
        loop.graduate_kv_to_esm(
            compile_fn=_ok_multi_compile,
            eval_fn=lambda _b, _e: {"m0": "not_a_number", "m1": 1.0},  # type: ignore[arg-type,return-value]
            now=4.0,
        )

    assert float((esm_bank.anchor(15).w_down - snap_w15).abs().max()) < 1e-6
    assert float((esm_bank.anchor(19).w_down - snap_w19).abs().max()) < 1e-6
    assert torch.equal(esm_bank.anchor(15).active_basis, snap_b15)
    assert torch.equal(esm_bank.anchor(19).active_basis, snap_b19)
    assert torch.equal(esm_bank.anchor(15).scope_prototypes, snap_p15)
    assert torch.equal(esm_bank.anchor(19).scope_prototypes, snap_p19)
    assert [e.entry_id for e in esm_bank.anchor(15).ledger if e.active] == active_l15_before
    assert [e.entry_id for e in esm_bank.anchor(19).ledger if e.active] == active_l19_before
    assert store.blobs["b_partial"].header.is_active(4.0) is True


# ==============================================================================
# 4. Tier-3 `export_slow_inner_bundle` + `state_dict` / `load_state_dict` Round-Trip
# ==============================================================================


def test_tier3_export_slow_inner_bundle_and_state_dict_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(99)
    kv_cfg = _make_kv_cfg(graduation_hits=1, graduation_min_stability=2.0)
    store = SplitHorizonKVBlobStore(config=kv_cfg)
    esm_bank = _make_esm_bank(d_model=32, r_hidden=64, layers=(15,))
    engram_bank = AnchoredMoEEngramBank(
        EngramConfig(
            d_model=32,
            anchor_layers=(15,),
            tier1a_buckets=16,
            tier1b_buckets=16,
            tier1_dim=8,
            num_experts=2,
            top_k=1,
            moe_intermediate_size=16,
            shared_intermediate_size=16,
        )
    )
    tracker = TaskRegretTracker()
    cfg = GuardOuterConfig(
        soft_hurt=SoftHurtGateConfig(lambda_hurt=2.0, max_hurt_frac=0.25, max_bench_drop=0.25),
        graduation_max_per_cycle=2,
        graduation_min_hits=1,
        graduation_min_stability=2.0,
        chronic_fail_streak=2,
    )
    loop = LSANLGuardOuterLoop(
        store,
        esm_bank=esm_bank,
        engram_bank=engram_bank,
        config=cfg,
        regret_tracker=tracker,
    )

    # Baseline: i0..i3 pass, i4 & i5 fail (seen once -> consec_fails=1)
    bench_map = {f"i{k}": ("math" if k < 3 else "code") for k in range(6)}
    loop.initialize_baseline({"i0": 1.0, "i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 0.0, "i5": 0.0}, bench_map=bench_map)

    basis = torch.eye(32, dtype=torch.float32)
    b1 = _make_blob("kv_rule_1", basis[0], cfg=kv_cfg, t_created=1.0, salience=0.9, tags=("math",))
    # Promote b1: i4 fixed (0 -> 1), i5 stays 0 (consec_fails -> 2 => chronic unsolved!)
    res1 = loop.propose_kv_rule(
        b1,
        eval_fn=lambda _b: {"i0": 1.0, "i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 1.0, "i5": 0.0},
        rule_meta={"topic": "modular_arith", "prompt": "NEVER_SERIALIZE_THIS_EVAL_PROMPT"},
        now=1.5,
    )
    assert res1.promoted is True

    # Reject b2 which hurts i0 -> tracked as newly_hurt
    b2 = _make_blob("kv_rule_bad", basis[1], cfg=kv_cfg, t_created=2.0, salience=0.9)
    res2 = loop.propose_kv_rule(
        b2,
        eval_fn=lambda _b: {"i0": 0.0, "i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 1.0, "i5": 0.0},
        now=2.0,
    )
    assert res2.promoted is False

    # Harden b1 and graduate it to ESM (single `(layer_idx, entry)` tuple return format!)
    store.record_hit("kv_rule_1", now=2.5)

    def _compile_single(blob: KVMemBlob, bank: EpisodicSideMemoryBank) -> tuple[int, ESMLedgerEntry]:
        h_trig = F.normalize(torch.randn(2, 32), dim=-1) * 8.0
        y_tgt = F.normalize(torch.randn(2, 32), dim=-1) * 0.10
        entry, _ = bank.compile_nullspace_ridge(15, h_trig, y_tgt, entry_id=f"esm_{blob.blob_id}")
        return (15, entry)

    grads = loop.graduate_kv_to_esm(
        compile_fn=_compile_single,
        eval_fn=lambda _b, _e: {"i0": 1.0, "i1": 1.0, "i2": 1.0, "i3": 1.0, "i4": 1.0, "i5": 0.0},
        now=3.0,
    )
    assert len(grads) == 1 and grads[0].promoted is True

    # Export slow-inner bundle and verify contamination firewall + contents
    bundle_dir = tmp_path / "slow_inner_bundle"
    bundle = loop.export_slow_inner_bundle(bundle_dir)
    state_path = Path(bundle["state_path"])
    queue_path = Path(bundle["queue_path"])
    assert state_path.exists() and queue_path.exists()

    state_raw = state_path.read_text(encoding="utf-8")
    queue_raw = queue_path.read_text(encoding="utf-8")
    assert "NEVER_SERIALIZE_THIS_EVAL_PROMPT" not in state_raw
    assert "NEVER_SERIALIZE_THIS_EVAL_PROMPT" not in queue_raw

    state_json = json.loads(state_raw)
    assert state_json["counts"]["promoted_kv"] == 1
    assert state_json["counts"]["rejected_kv"] == 1
    assert state_json["counts"]["promoted_esm"] == 1
    assert state_json["esm_bank_stats"]["shard_mib"] == pytest.approx(esm_bank.shard_mib())
    assert state_json["engram_bank_stats"]["anchor_layers"] == [15]
    assert state_json["kv_store_stats"]["tombstoned_blobs"] == 1
    assert state_json["baseline_summary"]["n_items"] == 6

    q_rows = [json.loads(line) for line in queue_raw.splitlines() if line.strip()]
    assert len(q_rows) == bundle["n_queue_records"]
    by_type = {(r["record_type"], r["item_id"]): r for r in q_rows}
    assert ("graduated_rule", "kv_rule_1") in by_type
    assert ("newly_hurt_item", "i0") in by_type
    assert ("chronic_unsolved_item", "i5") in by_type
    assert by_type[("newly_hurt_item", "i0")]["regret_bucket"] == REGRET_NEWLY_HURT
    assert by_type[("chronic_unsolved_item", "i5")]["regret_bucket"] == REGRET_UNSOLVED_CHRONIC
    assert by_type[("graduated_rule", "kv_rule_1")]["regret_bucket"] == REGRET_PINNED_PASS
    # Queue is strictly sorted by descending priority
    priorities = [r["priority"] for r in q_rows]
    assert priorities == sorted(priorities, reverse=True)

    # Full `state_dict` / `load_state_dict` JSON round-trip
    sd = loop.state_dict()
    sd_roundtrip = json.loads(json.dumps(sd))
    loop2 = LSANLGuardOuterLoop(SplitHorizonKVBlobStore(config=kv_cfg), esm_bank=esm_bank)
    loop2.load_state_dict(sd_roundtrip)
    assert loop2.state_dict() == sd
    assert loop2.baseline_items == loop.baseline_items
    assert loop2.pinned_passed == loop.pinned_passed
    assert loop2.bench_map == loop.bench_map
    assert loop2.graduated_rules == loop.graduated_rules
    assert loop2.regret_tracker.state_dict() == loop.regret_tracker.state_dict()


def test_config_validation_bootstrap_and_graduation_overrides() -> None:
    torch.manual_seed(123)
    with pytest.raises(ValueError, match="lambda_hurt"):
        SoftHurtGateConfig(lambda_hurt=-0.1)
    with pytest.raises(ValueError, match="max_hurt_frac"):
        SoftHurtGateConfig(max_hurt_frac=1.5)
    with pytest.raises(ValueError, match="graduation_max_per_cycle"):
        GuardOuterConfig(graduation_max_per_cycle=0)
    with pytest.raises(ValueError, match="unknown"):
        GuardOuterConfig.from_dict({"bad_key": 1})

    # `min_soft_hurt > 0` blocks zero-gain ties AND sub-threshold positive soft_hurt
    strict_cfg = SoftHurtGateConfig(lambda_hurt=2.0, min_soft_hurt=0.05)
    base = {f"k{i}": (1.0 if i < 10 else 0.0) for i in range(20)}
    assert evaluate_soft_hurt(base, base, cfg=strict_cfg).promoted is False
    assert evaluate_soft_hurt(base, {**base, "k10": 1.0}, cfg=strict_cfg).promoted is True  # +0.05 == min_soft_hurt

    # Bootstrap `propose_kv_rule` without prior `initialize_baseline`, then override looser graduation thresholds
    kv_cfg = _make_kv_cfg(graduation_hits=10, graduation_min_stability=100.0)
    store = SplitHorizonKVBlobStore(config=kv_cfg)
    esm_bank = _make_esm_bank(d_model=32, r_hidden=64, layers=(15,))
    outer_cfg = GuardOuterConfig(graduation_min_hits=1, graduation_min_stability=2.0)
    loop = LSANLGuardOuterLoop(store, esm_bank=esm_bank, config=outer_cfg)

    basis = torch.eye(32, dtype=torch.float32)
    b0 = _make_blob("boot_rule", basis[0], cfg=kv_cfg, t_created=0.0, salience=0.9)
    res_boot = loop.propose_kv_rule(b0, eval_fn=lambda _b: {"x0": 1.0, "x1": 0.0}, now=1.0)
    assert res_boot.promoted is True and res_boot.admitted is True
    assert loop.baseline_items == {"x0": 1.0, "x1": 0.0}
    assert loop.pinned_passed == {"x0"}

    # Single hit gives hit_count=1, stability=2.0 — below kv_cfg (10, 100.0), but matches outer_cfg override!
    store.record_hit("boot_rule", now=1.5)
    assert store.select_for_graduation() == []

    def _compile(blob: KVMemBlob, bank: EpisodicSideMemoryBank) -> tuple[int, ESMLedgerEntry]:
        h_t = F.normalize(torch.randn(2, 32), dim=-1) * 8.0
        y_t = F.normalize(torch.randn(2, 32), dim=-1) * 0.1
        entry, _ = bank.compile_nullspace_ridge(15, h_t, y_t, entry_id=f"esm_{blob.blob_id}")
        return (15, entry)

    grads = loop.graduate_kv_to_esm(
        compile_fn=_compile,
        eval_fn=lambda _b, _e: {"x0": 1.0, "x1": 1.0},
        now=2.0,
    )
    assert len(grads) == 1 and grads[0].promoted is True
    # Original `kv_store.config` graduation thresholds MUST be restored cleanly!
    assert store.config.graduation_hits == 10
    assert store.config.graduation_min_stability == 100.0
    # Graduated blob is excluded from retrieval and cleanly garbage-collected on forgetting cycle
    assert store.retrieve(basis[0], top_k=4, now=2.0) == []
    evicted = store.run_forgetting_cycle(now=2.0)
    assert evicted == ["boot_rule"]

