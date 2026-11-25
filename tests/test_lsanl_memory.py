"""Fast CPU verification suite for Phase-2A/2B `LSA-NL` Memory Primitives (`src/qloop/memory/`).

Verifies:
  1. Pre-RoPE sink stripping & un-rotation (`strip_sinks_and_unrotate_k`), round-trip error `< 1e-5`
     on `[0:64)` in fp32, exact `0.0` diff (`torch.equal`) on `[64:256)` (`192`-D NoPE), Group-0 vs
     Groups `1..3` orthogonality through `Block64HadamardINT4`, and exact `2.8125 MiB` payload &
     `1,344 B` header byte formulas.
  2. `SplitHorizonKVBlobStore` 4-stage `10B`-token lifecycle: salience cherry-picking reject/accept,
     `EpiCache` duplicate compaction, Ebbinghaus ACT-R spaced-repetition hardening (`S -> 2S`) vs
     forgetting eviction, bi-temporal `supersede()` exclusion from `DLMG` PPR, and graduation selection.
  3. `DynamicLatentMemoryGraph` (`DLMG`) 2-hop Personalized PageRank (`HippoRAG 2`) test: on a
     planted 2-hop cluster graph (`A -> B -> C` with `cos(A, C) == 0`), PPR ranks `C` in top-2 while
     flat cosine completely misses `C`.
  4. `EpisodicSideMemoryBank` (`ESM`): Coda `L39` guard (`ValueError` unless `allow_coda=True`),
     `144.0 MiB` shard accounting, Step-0 `torch.equal` identity, `Float64 DGKS` null-space leakage
     `< 1e-12` across 3 sequential writes + general-corpus subspace protection, exact `0.0` `WISE`
     scope-gate perturbation on orthogonal general tokens, trust-region relative norm cap, and exact
     transactional `rollback_entry` restoration (`< 1e-6`).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from qloop.memory import (
    DEFAULT_CORE_LOOP_SPEC,
    DynamicLatentMemoryGraph,
    EpisodicSideMemoryBank,
    ESMConfig,
    KVBlobConfig,
    KVBlobHeader,
    SplitHorizonKVBlobStore,
    mount_kv_blob,
    pack_kv_blob,
    strip_sinks_and_unrotate_k,
    theoretical_h64_int4_bytes,
    theoretical_header_wire_bytes,
)
from qloop.model.loop import LoopSpec
from qloop.model.pos_rerope import apply_partial_rope_k, build_rope_inv_freq


# ==============================================================================
# 1. Pre-RoPE Un-Rotation, Group-0/NoPE Orthogonality & Exact Byte Formulas
# ==============================================================================
def test_prerope_roundtrip_h64_orthogonality_and_exact_byte_formulas():
    torch.manual_seed(2026)
    cfg = KVBlobConfig()
    assert cfg.header_wire_bytes() == 1344
    assert theoretical_header_wire_bytes(10, 64) == 1344
    assert theoretical_h64_int4_bytes(10, 2, 256, 512) == 2_949_120
    assert math.isclose(theoretical_h64_int4_bytes(10, 2, 256, 512) / (1024**2), 2.8125, rel_tol=0.0, abs_tol=0.0)
    assert cfg.theoretical_payload_bytes() == 2_949_120

    # 1a. Exact Pre-RoPE un-rotation + re-rotation round-trip on `[0:64)` and exact 0.0 on `[64:256)`
    n_layers, n_heads, n_tok_with_sinks, head_dim = 2, 2, 68, 256
    rot_dim = 64
    k_raw = torch.randn(n_layers, n_heads, n_tok_with_sinks, head_dim, dtype=torch.float32)
    orig_positions = torch.arange(12_345, 12_345 + n_tok_with_sinks, dtype=torch.long)
    inv_freq = build_rope_inv_freq(rot_dim, rope_theta=cfg.rope_theta, dtype=torch.float64)

    k_post_rope = apply_partial_rope_k(k_raw, orig_positions, inv_freq=inv_freq, rotary_dim=rot_dim)
    assert torch.equal(k_post_rope[..., rot_dim:], k_raw[..., rot_dim:])

    # Strip `[0:4)` sinks (`EPIC` rule) and un-rotate `[0:64)` back to Pre-RoPE
    k_unrot = strip_sinks_and_unrotate_k(
        k_post_rope,
        orig_positions,
        rotary_dim=rot_dim,
        rope_theta=cfg.rope_theta,
        sink_tokens=cfg.sink_tokens,
    )
    assert k_unrot.shape == (n_layers, n_heads, n_tok_with_sinks - cfg.sink_tokens, head_dim)
    # Bitwise exact `0.0` diff on all `192` NoPE dims `[64:256)`!
    assert torch.equal(k_unrot[..., rot_dim:], k_raw[..., cfg.sink_tokens :, rot_dim:])
    unrot_rot_err = float((k_unrot[..., :rot_dim] - k_raw[..., cfg.sink_tokens :, :rot_dim]).abs().max())
    assert unrot_rot_err < 1e-5

    # Re-rotate back to original positions -> exact round-trip `< 1e-5` fp32 and `0.0` on NoPE
    k_rerot = apply_partial_rope_k(k_unrot, orig_positions[cfg.sink_tokens :], inv_freq=inv_freq, rotary_dim=rot_dim)
    rerot_err = float((k_rerot[..., :rot_dim] - k_post_rope[..., cfg.sink_tokens :, :rot_dim]).abs().max())
    assert rerot_err < 1e-5
    assert torch.equal(k_rerot[..., rot_dim:], k_raw[..., cfg.sink_tokens :, rot_dim:])

    # 1b. Group-0 `[0:64)` vs Groups `1..3` `[64:256)` orthogonality through `Block64HadamardINT4`
    v_kept = torch.randn_like(k_unrot)
    v_kept[..., :4] *= 10.0
    v_kept[..., 64:68] *= 10.0
    blob_a = pack_kv_blob(k_unrot, v_kept, blob_id="blob_a", config=cfg)

    # Perturb ONLY `[0:64)` rotary slice arbitrarily -> Groups `1..3` packed bytes & mounted NoPE MUST be bit-exact!
    k_unrot_perturbed = k_unrot.clone()
    k_unrot_perturbed[..., :rot_dim] = torch.randn_like(k_unrot[..., :rot_dim]) * 7.5
    blob_b = pack_kv_blob(k_unrot_perturbed, v_kept, blob_id="blob_b", config=cfg)

    assert torch.equal(blob_a.packed_k[..., 1:, :], blob_b.packed_k[..., 1:, :])
    assert torch.equal(blob_a.header.coarse_centroid, blob_b.header.coarse_centroid)

    # Mount at two distant Partition-B virtual positions (including ReRoPE clamping at 3M query offset)
    k_mnt_1, v_mnt_1 = mount_kv_blob(blob_a, virtual_positions=50_000)
    k_mnt_2, v_mnt_2 = mount_kv_blob(blob_a, virtual_positions=180_000, clamp_relative_to_q_pos=3_000_000.0)
    k_mnt_b, _ = mount_kv_blob(blob_b, virtual_positions=50_000)

    # Zero mutation of `[64:256)` NoPE dims across virtual positions AND across Group-0 perturbations
    assert torch.equal(k_mnt_1[..., rot_dim:], k_mnt_2[..., rot_dim:])
    assert torch.equal(k_mnt_1[..., rot_dim:], k_mnt_b[..., rot_dim:])
    assert not torch.allclose(k_mnt_1[..., :rot_dim], k_mnt_2[..., :rot_dim])
    assert torch.equal(v_mnt_1, v_mnt_2)
    # Outlier Value reconstruction fidelity stays well under 0.085
    v_rel_err = float((v_mnt_1 - v_kept).norm() / v_kept.norm())
    assert v_rel_err < 0.085

    # 1c. Canonical full-scale `10L, 2H, 256D, 512T` blob exact `2.8125 MiB` payload & `1,344 B` header
    k_full = torch.randn(10, 2, 512, 256, dtype=torch.float32)
    v_full = torch.randn(10, 2, 512, 256, dtype=torch.float32)
    full_blob = pack_kv_blob(k_full, v_full, blob_id="canonical_512", config=cfg)
    assert full_blob.payload_bytes() == 2_949_120
    assert full_blob.payload_bytes() == theoretical_h64_int4_bytes(10, 2, 256, 512)
    assert full_blob.header_wire_bytes() == 1344
    assert full_blob.header.coarse_centroid.shape == (10, 64)
    assert full_blob.header.coarse_centroid.dtype == torch.float16
    layer_norms = full_blob.header.coarse_centroid.float().norm(dim=-1)
    assert float((layer_norms - 1.0).abs().max()) < 2e-3


# ==============================================================================
# 2. `SplitHorizonKVBlobStore` 4-Stage `10B`-Token Lifecycle
# ==============================================================================
def _make_synthetic_blob(
    blob_id: str,
    centroid_dir: torch.Tensor,
    *,
    cfg: KVBlobConfig,
    t_created: float = 0.0,
    salience: float = 0.8,
    tags: tuple[str, ...] = (),
) -> object:
    """Create a lightweight 2-layer, 4-token packed `KVMemBlob` with an exact centroid direction."""
    k = torch.randn(cfg.num_global_layers, cfg.num_kv_heads, cfg.blob_tokens, cfg.head_dim)
    v = torch.randn_like(k)
    c_mat = F.normalize(centroid_dir.float(), dim=-1).unsqueeze(0).expand(cfg.num_global_layers, -1).contiguous()
    return pack_kv_blob(
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


def test_split_horizon_store_4stage_lifecycle():
    torch.manual_seed(42)
    cfg = KVBlobConfig(
        num_global_layers=2,
        num_kv_heads=2,
        head_dim=128,
        rotary_dim=64,
        blob_tokens=8,
        index_dim=32,
        salience_threshold=0.25,
        dedup_cos_threshold=0.90,
        min_retention=0.15,
        initial_stability=1.0,
        hardening_factor=2.0,
        graduation_hits=4,
        graduation_min_stability=8.0,
    )
    store = SplitHorizonKVBlobStore(config=cfg)

    basis = torch.eye(32, dtype=torch.float32)

    # Stage 1: Salience cherry-picking (reject below threshold, admit above)
    low_sal = _make_synthetic_blob("noise_0", basis[0], cfg=cfg, t_created=0.0, salience=0.10)
    assert store.admit(low_sal, salience=0.10) is False
    assert "noise_0" not in store.blobs
    assert store.memory_stats()["rejected_blobs"] == 1

    b_hard = _make_synthetic_blob("hardened_ep", basis[1], cfg=cfg, t_created=0.0, salience=0.90, tags=("core",))
    b_dup = _make_synthetic_blob(
        "hardened_ep_dup",
        basis[1] + 0.05 * basis[2],
        cfg=cfg,
        t_created=0.1,
        salience=0.60,
        tags=("dup_tag",),
    )
    b_stale = _make_synthetic_blob("unaccessed_ep", basis[3], cfg=cfg, t_created=0.0, salience=0.70)
    b_fact_v1 = _make_synthetic_blob("fact_v1", basis[4], cfg=cfg, t_created=0.0, salience=0.80, tags=("config",))
    b_fact_v2 = _make_synthetic_blob(
        "fact_v2", basis[4] + 0.02 * basis[5], cfg=cfg, t_created=2.0, salience=0.85, tags=("config",)
    )

    assert store.admit(b_hard) is True
    assert store.admit(b_dup) is True
    assert store.admit(b_stale) is True
    assert store.admit(b_fact_v1) is True
    assert len(store) == 4

    # Stage 2: EpiCache duplicate compaction merges `hardened_ep_dup` into `hardened_ep`
    merged = store.compact_duplicates(cos_threshold=0.90)
    assert merged == [("hardened_ep_dup", "hardened_ep")]
    assert "hardened_ep_dup" not in store.blobs
    assert "dup_tag" in store.blobs["hardened_ep"].header.topic_tags
    assert store.memory_stats()["merged_blobs"] == 1

    # Stage 3a: Ebbinghaus ACT-R spaced-repetition hardening (`S -> 2S` per hit)
    for step, t_hit in enumerate((0.5, 1.0, 1.5, 2.0), start=1):
        hdr = store.record_hit("hardened_ep", now=t_hit)
        assert hdr.hit_count == step
        assert math.isclose(hdr.stability, float(2**step), rel_tol=0.0, abs_tol=1e-12)
    assert store.blobs["hardened_ep"].header.stability == 16.0

    # Stage 3b: Bi-temporal supersession (`SleepGate` + `Zep/Graphiti`) excludes old blob from DLMG PPR
    assert store.admit(b_fact_v2) is True
    store.supersede("fact_v1", "fact_v2", now=2.5)
    assert store.blobs["fact_v1"].header.t_invalid == 2.5
    assert store.blobs["fact_v1"].header.is_active(2.0) is True
    assert store.blobs["fact_v1"].header.is_active(2.5) is False

    stats_pre_forget = store.memory_stats()
    assert stats_pre_forget["tombstoned_blobs"] == 1
    assert stats_pre_forget["active_blobs"] == 3
    ppr_hits = store.graph.retrieve_ppr(basis[4], top_k=4, alpha=0.85, n_iters=15)
    assert "fact_v2" in ppr_hits
    assert "fact_v1" not in ppr_hits

    # Stage 3c: Active forgetting cycle at t=5.0
    # - `unaccessed_ep`: R(5.0) = exp(-5 / 1) = 0.0067 < 0.15 -> EVICTED
    # - `fact_v1`: tombstoned at 2.5 <= 5.0 -> EVICTED
    # - `hardened_ep`: R(5.0) = exp(-3 / 16) = 0.829 >= 0.15 -> SURVIVES
    # - `fact_v2`: R(5.0) = exp(-3 / 1) = 0.0498 < 0.15 unless hit; hit once at t=4.0 so it survives!
    store.record_hit("fact_v2", now=4.0)  # S = 2.0, R(5.0) = exp(-1/2) = 0.606 >= 0.15
    evicted = set(store.run_forgetting_cycle(now=5.0))
    assert evicted == {"unaccessed_ep", "fact_v1"}
    assert "hardened_ep" in store.blobs
    assert "fact_v2" in store.blobs
    assert "unaccessed_ep" not in store.blobs
    assert "fact_v1" not in store.blobs

    # Stage 4: Graduation selection returns only hardened episodes (`hits >= 4` and `S >= 8.0`)
    grad = store.select_for_graduation()
    assert [b.blob_id for b in grad] == ["hardened_ep"]

    # Retrieve & mount end-to-end from store
    retrieved = store.retrieve(basis[1], top_k=2)
    assert retrieved[0].blob_id == "hardened_ep"
    k_mnt, v_mnt = store.mount_blob("hardened_ep", virtual_positions=4096)
    assert k_mnt.shape == (cfg.num_global_layers, cfg.num_kv_heads, cfg.blob_tokens, cfg.head_dim)
    assert v_mnt.shape == k_mnt.shape

    # Exact memory stats & from_dict validation
    stats = store.memory_stats()
    assert stats["active_blobs"] == 2
    assert stats["tombstoned_blobs"] == 0
    assert stats["evicted_blobs"] == 2
    expected_payload = sum(b.payload_bytes() for b in store.blobs.values())
    expected_headers = sum(b.header_wire_bytes() for b in store.blobs.values())
    assert stats["payload_bytes"] == expected_payload
    assert stats["gpu_header_bytes"] == expected_headers
    assert stats["gpu_header_bytes"] == 2 * theoretical_header_wire_bytes(2, 32)
    assert KVBlobConfig.from_dict({"salience_threshold": 0.30}).salience_threshold == 0.30
    with pytest.raises(ValueError, match="unknown"):
        KVBlobConfig.from_dict({"not_a_real_key": 1})

    # Verify NaN/non-finite stability rejection, stability upper cap (`max_stability`), and
    # supersession_edges remapping/pruning on `compact_duplicates` & `remove_node`
    with pytest.raises(ValueError, match="finite"):
        KVBlobConfig(initial_stability=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        KVBlobHeader(blob_id="nan_hdr", coarse_centroid=torch.ones(2, 32), stability=float("nan"))

    # Repeated hardening saturates cleanly at `max_stability` without overflowing to `inf`
    cap_cfg = KVBlobConfig(
        num_global_layers=2,
        num_kv_heads=2,
        head_dim=128,
        rotary_dim=64,
        blob_tokens=4,
        index_dim=32,
        max_stability=1e6,
    )
    cap_store = SplitHorizonKVBlobStore(config=cap_cfg)
    b_cap = _make_synthetic_blob("cap_ep", basis[6], cfg=cap_cfg, salience=0.9)
    assert cap_store.admit(b_cap) is True
    for i in range(1200):
        hdr_cap = cap_store.record_hit("cap_ep", now=float(i) * 0.01)
    assert math.isfinite(hdr_cap.stability)
    assert hdr_cap.stability == cap_cfg.max_stability

    # Supersession remapping on duplicate merge: `old -> dup` remaps to `old -> keeper`, zero dangling IDs!
    b_old = _make_synthetic_blob("sup_old", basis[7], cfg=cap_cfg, t_created=0.0, salience=0.5)
    b_keep = _make_synthetic_blob("sup_keep", basis[8], cfg=cap_cfg, t_created=1.0, salience=0.95)
    b_dup_new = _make_synthetic_blob("sup_dup", basis[8] + 0.02 * basis[9], cfg=cap_cfg, t_created=1.1, salience=0.6)
    assert cap_store.admit(b_old) is True
    assert cap_store.admit(b_keep) is True
    assert cap_store.admit(b_dup_new) is True
    cap_store.supersede("sup_old", "sup_dup", now=2.0)
    assert cap_store.graph.supersession_edges["sup_old"] == "sup_dup"
    merged_sup = cap_store.compact_duplicates(cos_threshold=0.90)
    assert ("sup_dup", "sup_keep") in merged_sup
    assert cap_store.graph.supersession_edges["sup_old"] == "sup_keep"
    assert all(dst in cap_store.graph.nodes for dst in cap_store.graph.supersession_edges.values())
    cap_store.graph.remove_node("sup_keep")
    assert "sup_old" not in cap_store.graph.supersession_edges


# ==============================================================================
# 3. `DynamicLatentMemoryGraph` (`DLMG`) 2-Hop Personalized PageRank Test
# ==============================================================================
def test_dlmg_two_hop_ppr_ranks_c_in_top2_while_flat_cosine_misses():
    d_idx = 64
    n_layers = 4
    eye = torch.eye(d_idx, dtype=torch.float32)

    def _hdr(bid: str, vec: torch.Tensor) -> KVBlobHeader:
        c = F.normalize(vec, dim=-1).unsqueeze(0).expand(n_layers, -1).contiguous()
        return KVBlobHeader(blob_id=bid, coarse_centroid=c, n_tokens=512)

    # Plant a 2-hop cluster graph A <-> {B1, B2, B3} <-> C plus 8 distractors D1..D8:
    #   * Query matches A (`cos(q, A) = 1.0`).
    #   * Target answer C is strictly orthogonal to A (`cos(q, C) = 0.0`).
    #   * Bridge cluster B1..B3 connects A (`cos=0.35`) to C (`cos=0.45`) with `cos(B_i, B_j) = 0.325 < 0.34`.
    #   * Distractors D1..D8 have `cos(q, D_k) = 0.25 > 0.0 = cos(q, C)`, so flat cosine ranks ALL 8
    #     distractors strictly ahead of C, completely missing C!
    e_a = eye[0]
    e_c = eye[1]

    graph = DynamicLatentMemoryGraph(semantic_top_m=4, semantic_min_cos=0.34)
    graph.register_node(_hdr("A", e_a))
    graph.register_node(_hdr("C", e_c))

    for idx in range(3):
        e_b = eye[2 + idx]
        vec_b = 0.35 * e_a + 0.45 * e_c + math.sqrt(1.0 - 0.35**2 - 0.45**2) * e_b
        graph.register_node(_hdr(f"B{idx + 1}", vec_b))

    for idx in range(8):
        e_d = eye[10 + idx]
        vec_d = 0.25 * e_a + math.sqrt(1.0 - 0.25**2) * e_d
        graph.register_node(_hdr(f"D{idx + 1}", vec_d))

    query = e_a

    # Flat cosine misses C completely in top-2 (and in fact ranks C last at #13!)
    flat_top2 = graph.retrieve_cosine(query, top_k=2)
    flat_all = graph.retrieve_cosine(query, top_k=13)
    assert "C" not in flat_top2
    assert flat_all.index("C") == 12  # dead last behind A, B1..B3, and all 8 distractors!

    # Damped HippoRAG-2 PPR (`alpha=0.85, n_iters=15`) spreads mass A -> {B1,B2,B3} -> C and ranks C in top-2!
    ppr_top2 = graph.retrieve_ppr(query, top_k=2, alpha=0.85, n_iters=15)
    assert ppr_top2[0] == "A"
    assert "C" in ppr_top2
    assert ppr_top2.index("C") == 1

    # Verify Hebbian co-access edges strengthen recall between unrelated nodes
    graph.record_co_access(["A", "D1"], weight_step=3.0)
    assert graph.hebbian_edges["A"]["D1"] == 3.0


# ==============================================================================
# 4. `EpisodicSideMemoryBank` (`ESM`): Coda Guard, Step-0, DGKS, WISE, Rollback
# ==============================================================================
def test_esm_coda_guard_and_shard_bytes_formula():
    # Default config matches `LoopSpec` `[12, 36)` core full-attention anchors & exact `144.0 MiB` shard
    cfg_default = ESMConfig()
    assert cfg_default.anchor_layers == (15, 19, 23, 27, 31, 35)
    assert cfg_default.theoretical_shard_bytes() == 144 * 1024 * 1024
    assert EpisodicSideMemoryBank.theoretical_shard_bytes(6, 2048, 2048) == 144 * 1024 * 1024
    full_bank = EpisodicSideMemoryBank(cfg_default)
    assert full_bank.shard_bytes() == 150_994_944
    assert full_bank.shard_mib() == 144.0
    # Verify actual `state_dict()` weight bytes (`w_in` bf16 + `w_down` fp32) match `shard_bytes()` byte-for-byte!
    actual_sd_bytes = sum(
        t.numel() * t.element_size()
        for k, t in full_bank.state_dict().items()
        if k.endswith(("w_in", "w_down"))
    )
    assert actual_sd_bytes == full_bank.shard_bytes() == 150_994_944 == 144 * 1024 * 1024
    assert ESMConfig.from_dict({"anchor_layers": [15, 19], "scope_threshold": 0.70}).anchor_layers == (15, 19)
    with pytest.raises(ValueError, match="unknown"):
        ESMConfig.from_dict({"bad_key": 42})
    with pytest.raises(ValueError, match="scope_threshold"):
        ESMConfig(d_model=32, r_hidden=64, scope_threshold=0.0)
    with pytest.raises(ValueError, match="scope_threshold"):
        ESMConfig(d_model=32, r_hidden=64, scope_threshold=-0.2)

    cfg_from_spec = ESMConfig.from_loop_spec(DEFAULT_CORE_LOOP_SPEC)
    assert cfg_from_spec.anchor_layers == (15, 19, 23, 27, 31, 35)
    assert cfg_from_spec.coda_start == 36

    # Strictly rejects Coda layer `L39` (and `L36`) unless `allow_coda=True`
    with pytest.raises(ValueError, match="Coda"):
        ESMConfig(d_model=64, r_hidden=128, anchor_layers=(15, 39))
    with pytest.raises(ValueError, match="Coda"):
        ESMConfig(d_model=64, r_hidden=128, anchor_layers=(15, 36))
    with pytest.raises(ValueError, match="Coda"):
        EpisodicSideMemoryBank(
            ESMConfig(d_model=64, r_hidden=128, anchor_layers=(15, 31), coda_start=36),
            loop_spec=LoopSpec(prelude_end=8, core_end=30, num_passes=2, target_virtual_layers=None),
        )

    # Explicit `allow_coda=True` override works cleanly
    cfg_coda_ok = ESMConfig(d_model=32, r_hidden=64, anchor_layers=(15, 39), allow_coda=True)
    bank_coda = EpisodicSideMemoryBank(cfg_coda_ok)
    assert bank_coda.has_layer(39)
    assert bank_coda.shard_bytes() == 2 * 64 * 32 * 6


def test_esm_step0_dgks_nullspace_wise_normcap_and_rollback():
    torch.manual_seed(2026)
    d_model = 64
    r_hidden = 192
    cfg = ESMConfig(
        d_model=d_model,
        r_hidden=r_hidden,
        anchor_layers=(15, 19, 23, 27, 31, 35),
        scope_threshold=0.65,
        ridge_alpha=1e-4,
        max_rel_delta_norm=0.03,
    )
    bank = EpisodicSideMemoryBank(cfg)
    assert bank.shard_bytes() == 6 * r_hidden * d_model * 6
    anchor = bank.anchor(15)
    assert anchor.w_in.dtype == torch.bfloat16
    assert anchor.w_in.requires_grad is False
    assert not hasattr(anchor, "w_gate") and not hasattr(anchor, "w_up")
    assert anchor.w_down.dtype == torch.float32
    assert anchor.active_basis.dtype == torch.float64

    # 4a. Step-0 bit-identical forward (`torch.equal` to exact zeros across dtypes)
    for dt in (torch.float32, torch.bfloat16, torch.float64):
        h_probe = torch.randn(2, 5, d_model, dtype=dt)
        out0 = bank(15, h_probe)
        assert torch.equal(out0, torch.zeros_like(h_probe))
        assert torch.equal(h_probe + out0, h_probe)

    # Partition d_model=64 into cleanly separated subspaces so orthogonal general tokens stay orthogonal
    # to trigger prototypes (`cos == 0.0 < 0.65`), while SwiGLU features still have non-zero overlap
    # with general subspace `U_gen` to genuinely stress AlphaEdit null-space projection!
    h_general = torch.zeros(24, d_model, dtype=torch.float32)
    h_general[:, :32] = torch.randn(24, 32)
    u_gen = bank.register_general_subspace(15, h_general, energy_threshold=0.995)
    assert u_gen.dtype == torch.float64 and u_gen.shape[0] >= 1
    assert float((u_gen @ u_gen.T - torch.eye(u_gen.shape[0], dtype=torch.float64)).abs().max()) < 1e-12

    # 4b. Three sequential writes with general-corpus + prior-write null-space leakage < 1e-12
    trigger_sets: list[torch.Tensor] = []
    target_sets: list[torch.Tensor] = []
    entries = []
    out_after_write2: torch.Tensor | None = None

    for w_idx in range(3):
        h_trig = torch.zeros(4, d_model, dtype=torch.float32)
        h_trig[:, 32:] = F.normalize(torch.randn(4, 32), dim=-1) * 10.0
        # Target strictly inside trust-region norm cap (`0.015 * ||h|| < 0.03 * ||h||`)
        y_tgt = F.normalize(torch.randn(4, d_model), dim=-1) * 0.15
        trigger_sets.append(h_trig)
        target_sets.append(y_tgt)

        entry, max_leak = bank.compile_nullspace_ridge(15, h_trig, y_tgt, entry_id=f"ep_{w_idx + 1}")
        entries.append(entry)
        assert entry.active is True
        assert entry.n_probes == 4
        assert entry.pre_leakage > 1e-5, f"expected non-zero pre-projection leakage, got {entry.pre_leakage}"
        assert max_leak < 1e-12, f"write {w_idx + 1} null-space leakage {max_leak} >= 1e-12"
        assert entry.post_leakage == max_leak

        # Direct check against general-corpus basis `U_gen`
        gen_leak = float((u_gen @ entry.delta_w_down).abs().max().item())
        assert gen_leak < 1e-12

        # Direct check against ALL prior writes' SwiGLU features (zero catastrophic interference!)
        for prev_i in range(w_idx):
            prev_phi = anchor.compute_features(trigger_sets[prev_i], dtype=torch.float64)
            cross_leak = float((prev_phi @ entry.delta_w_down).abs().max().item())
            assert cross_leak < 1e-12

        # Check active_basis mutual orthonormality stays at machine epsilon
        u_all = anchor.active_basis
        ortho_err = float((u_all @ u_all.T - torch.eye(u_all.shape[0], dtype=torch.float64)).abs().max().item())
        assert ortho_err < 1e-12

        if w_idx == 1:
            h_eval_all = torch.cat([trigger_sets[0], trigger_sets[1], h_general[:4]], dim=0)
            out_after_write2 = bank(15, h_eval_all).clone()

    # 4c. WISE scope gate: orthogonal general tokens (`cos == 0.0 < 0.65`) get EXACT `0.0` perturbation!
    out_general = bank(15, h_general)
    assert torch.equal(out_general, torch.zeros_like(h_general))

    # In-scope triggers recall non-zero targets accurately (including Write #1 after Writes #2 & #3)
    out_w1 = bank(15, trigger_sets[0])
    assert float(out_w1.norm()) > 0.0
    rel_err_w1 = float((out_w1 - target_sets[0]).norm() / target_sets[0].norm())
    assert rel_err_w1 < 0.15

    # 4d. Exact transactional `rollback_entry` restores Write-2 output to `< 1e-6`
    assert out_after_write2 is not None
    h_eval_with_w3 = torch.cat([trigger_sets[0], trigger_sets[1], h_general[:4]], dim=0)
    out_w3_before_rb = bank(15, trigger_sets[2])
    assert float(out_w3_before_rb.norm()) > 0.05

    rb_entry = bank.rollback_entry(15, "ep_3")
    assert rb_entry.active is False
    out_after_rb = bank(15, h_eval_with_w3)
    rb_diff = float((out_after_rb - out_after_write2).abs().max().item())
    assert rb_diff < 1e-6

    # Also test relative norm cap (`||delta|| <= max_rel_delta_norm * ||h||`) on an intentionally huge target!
    h_huge_trig = torch.zeros(2, d_model, dtype=torch.float32)
    h_huge_trig[:, 32:] = F.normalize(torch.randn(2, 32), dim=-1) * 5.0
    y_huge_tgt = F.normalize(torch.randn(2, d_model), dim=-1) * 50.0  # 10x ||h|| >> 0.03 * ||h||
    _, huge_leak = bank.compile_nullspace_ridge(15, h_huge_trig, y_huge_tgt, entry_id="ep_huge")
    assert huge_leak < 1e-12
    out_capped = bank(15, h_huge_trig)
    h_norms = h_huge_trig.norm(dim=-1)
    d_norms = out_capped.norm(dim=-1)
    assert bool((d_norms <= (cfg.max_rel_delta_norm * h_norms + 1e-5)).all())
    assert bool((d_norms > 0.02 * h_norms).all())

    # Verify `load_state_dict` -> `compile` -> `rollback` restores EXACT pre-reload state (Blocker 3)
    sd = {k: v.clone() for k, v in bank.state_dict().items()}
    pre_reload_out = bank(15, trigger_sets[0]).clone()
    pre_reload_w_down = anchor.w_down.clone()
    pre_reload_basis = anchor.active_basis.clone()
    pre_reload_protos = anchor.scope_prototypes.clone()

    reloaded = EpisodicSideMemoryBank(cfg)
    reloaded.load_state_dict(sd)
    assert torch.equal(reloaded(15, trigger_sets[0]), pre_reload_out)
    assert torch.equal(reloaded(15, h_general), torch.zeros_like(h_general))

    h_post = torch.zeros(2, d_model, dtype=torch.float32)
    h_post[:, 32:] = F.normalize(torch.randn(2, 32), dim=-1) * 10.0
    y_post = F.normalize(torch.randn(2, d_model), dim=-1) * 0.15
    _, post_reload_leak = reloaded.compile_nullspace_ridge(15, h_post, y_post, entry_id="ep_post_reload")
    assert post_reload_leak < 1e-12
    assert reloaded.anchor(15).active_basis.shape[0] > pre_reload_basis.shape[0]
    reloaded.rollback_entry(15, "ep_post_reload")
    assert torch.equal(reloaded.anchor(15).w_down, pre_reload_w_down)
    assert torch.equal(reloaded.anchor(15).active_basis, pre_reload_basis)
    assert torch.equal(reloaded.anchor(15).scope_prototypes, pre_reload_protos)
    assert torch.equal(reloaded(15, trigger_sets[0]), pre_reload_out)

    # Exact Out-of-Order `rollback_entry` via Deterministic Replay (Blocker 2):
    # Write `ep_1`, `ep_2`, roll back `ep_1`, then write `ep_3_ooo` ->
    #   (a) `ep_2` recall stays exact (not degraded!), and
    #   (b) `ep_3_ooo`'s actual leakage onto `ep_2`'s features & `U_gen` is strictly `< 1e-12`!
    ooo_bank = EpisodicSideMemoryBank(cfg)
    u_gen_ooo = ooo_bank.register_general_subspace(15, h_general, energy_threshold=0.995)
    ooo_bank.compile_nullspace_ridge(15, trigger_sets[0], target_sets[0], entry_id="ep_1")
    ooo_bank.compile_nullspace_ridge(15, trigger_sets[1], target_sets[1], entry_id="ep_2")

    ooo_bank.rollback_entry(15, "ep_1")
    out_ep2_after_ooo = ooo_bank(15, trigger_sets[1])
    ep2_rel_err = float((out_ep2_after_ooo - target_sets[1]).norm() / target_sets[1].norm())
    assert ep2_rel_err < 0.05, f"ep_2 recall degraded after rolling back ep_1: rel_err={ep2_rel_err}"

    ep3_entry, ep3_leak = ooo_bank.compile_nullspace_ridge(15, trigger_sets[2], target_sets[2], entry_id="ep_3_ooo")
    assert ep3_leak < 1e-12
    phi_ep2 = ooo_bank.anchor(15).compute_features(trigger_sets[1], dtype=torch.float64)
    ep3_leak_on_ep2 = float((phi_ep2 @ ep3_entry.delta_w_down).abs().max().item())
    ep3_leak_on_gen = float((u_gen_ooo @ ep3_entry.delta_w_down).abs().max().item())
    assert ep3_leak_on_ep2 < 1e-12, f"post-rollback ep_3 leaked onto ep_2: {ep3_leak_on_ep2}"
    assert ep3_leak_on_gen < 1e-12, f"post-rollback ep_3 leaked onto U_gen: {ep3_leak_on_gen}"
    out_ep2_after_ep3 = ooo_bank(15, trigger_sets[1])
    assert float((out_ep2_after_ep3 - out_ep2_after_ooo).abs().max().item()) < 1e-6

    # Edge-case guards: shape[-1] mismatch raises ValueError; exhausted null space & zero-norm
    # triggers do NOT pollute `scope_prototypes`
    with pytest.raises(ValueError, match="d_model"):
        ooo_bank.compile_nullspace_ridge(15, torch.randn(2, 4), torch.randn(2, 4), entry_id="bad_shape")

    tiny_cfg = ESMConfig(d_model=8, r_hidden=4, anchor_layers=(15,), scope_threshold=0.65)
    tiny_bank = EpisodicSideMemoryBank(tiny_cfg)
    # Zero-norm trigger produces zero update and appends 0 prototypes
    tiny_bank.compile_nullspace_ridge(15, torch.zeros(1, 8), torch.ones(1, 8), entry_id="zero_trig")
    assert tiny_bank.anchor(15).scope_prototypes.shape[0] == 0
    # Exhaust all 4 dimensions of `r_hidden=4`
    tiny_bank.compile_nullspace_ridge(15, torch.randn(4, 8), torch.randn(4, 8) * 0.01, entry_id="fill_4")
    protos_before_exhaust = tiny_bank.anchor(15).scope_prototypes.shape[0]
    assert tiny_bank.anchor(15).active_basis.shape[0] == 4
    exh_entry, exh_leak = tiny_bank.compile_nullspace_ridge(
        15, torch.randn(2, 8), torch.randn(2, 8) * 0.01, entry_id="exhausted"
    )
    assert exh_leak == 0.0
    assert float(exh_entry.delta_w_down.abs().max().item()) == 0.0
    assert tiny_bank.anchor(15).scope_prototypes.shape[0] == protos_before_exhaust

    # Full rollback of all remaining active entries restores exact Step-0 `torch.equal` zeros!
    for eid in ("ep_huge", "ep_2", "ep_1"):
        bank.rollback_entry(15, eid)
    assert anchor.scope_prototypes.shape[0] == 0
    assert torch.equal(bank(15, trigger_sets[0]), torch.zeros_like(trigger_sets[0]))
