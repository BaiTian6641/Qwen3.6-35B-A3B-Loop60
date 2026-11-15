"""D-LSA correctness, scale, compaction, recall, hybrid mode, and zero-sync verification suite."""
from __future__ import annotations

import copy
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from qloop.model.loop import apply_loop
from qloop.model.lsa import (
    _DENSE_FN,
    LSAConfig,
    LSAIndexer,
    apply_yarn,
    compute_lsa_recall,
    install_lsa,
    lsa_attention,
    members_table,
    ward_cluster,
)
from qloop.model.toy import ToyConfig, ToyForCausalLM, build_toy, dense_causal_attention

SMALL = {
    "index_dim": 4,
    "block_size": 8,
    "pool": 2,
    "avg_bucket_keys": 4,
    "max_bucket_keys": 8,
    "window": 8,
    "sinks": 2,
    "num_select": 2,
    "query_chunk": 5,
    "distill_chunk": 4,
}


class _ZeroHostSyncGuard(TorchDispatchMode):
    """Fail immediately if any ATen op triggers a GPU->CPU scalar read or dynamic-shape nonzero sync."""

    FORBIDDEN_OPS: ClassVar[set[str]] = {
        "_local_scalar_dense",
        "item",
        "nonzero",
        "masked_select",
    }

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_name = getattr(getattr(func, "_schema", None), "name", str(func))
        short = op_name.split("::")[-1].split(".")[0]
        if short in self.FORBIDDEN_OPS:
            self.violations.append(op_name)
        return func(*args, **(kwargs or {}))


def _ids(t: int = 40, b: int = 1, seed: int = 5) -> torch.Tensor:
    return torch.randint(0, 97, (b, t), generator=torch.Generator().manual_seed(seed))


# ============================================================================ core / regression tests


def test_ward_cluster_separates_and_respects_cap():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(6, 3, generator=g) * 0.01
    b = torch.randn(6, 3, generator=g) * 0.01 + 5.0
    x = torch.cat([a, b])[torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])].unsqueeze(0)
    lab = ward_cluster(x, n_clusters=2, max_size=6)[0]
    assert lab.unique().numel() == 2
    assert len(set(lab[0::2].tolist())) == 1 and len(set(lab[1::2].tolist())) == 1
    capped = ward_cluster(x, n_clusters=1, max_size=4)[0]
    assert torch.bincount(capped).max() <= 4
    mem = members_table(lab.unsqueeze(0), 6)[0]
    for rep in lab.unique().tolist():
        assert sorted(m for m in mem[rep].tolist() if m >= 0) == sorted(
            (lab == rep).nonzero().flatten().tolist()
        )


def test_sparse_with_all_buckets_equals_dense():
    model = build_toy().double()
    ids = _ids(t=37)
    ref = model(ids).logits
    install_lsa(model, dict(SMALL, mode="sparse", num_select=10_000))
    out = model(ids).logits
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("search", ["flat", "tree"])
def test_sparse_decode_matches_sparse_full_forward(search):
    model = build_toy().double()
    ids = _ids(t=42, b=2)
    dense = model(ids).logits
    mgr = install_lsa(model, dict(SMALL, mode="sparse", search=search, tree_group=2, tree_beam=1))
    full = model(ids).logits
    assert not torch.allclose(full, dense), "selection must actually drop keys in this config"
    mgr.clear_cache()
    out = model(ids[:, :3], use_cache=True)
    cache, steps = out.past_key_values, [out.logits]
    for t in range(3, ids.shape[1]):
        out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, 1), full, rtol=1e-9, atol=1e-10)
    assert mgr.stats(), "stats should be recorded"


def test_distill_mode_keeps_dense_output_and_trains_only_indexer():
    model = build_toy()
    ids = _ids(t=48)
    ref = model(ids).logits.detach()
    for p in model.parameters():
        p.requires_grad_(False)
    mgr = install_lsa(model, dict(SMALL, mode="distill", distill_query_frac=1.0))
    params = mgr.trainable_parameters()
    opt = torch.optim.Adam(params, lr=0.05)
    first = last = None
    for _ in range(15):
        out = model(ids).logits
        loss = mgr.aux_loss()
        assert loss is not None and torch.isfinite(loss)
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.testing.assert_close(out.detach(), ref)  # base attention untouched in distill mode
    assert last < first * 0.9, f"KL should decrease: {first} -> {last}"
    assert all(p.grad is not None for p in params)


def test_lsa_composes_with_loop_and_kv_share():
    model = build_toy().double()
    ids = _ids(t=40)
    ref = model(ids).logits
    mgr = install_lsa(model, dict(SMALL, mode="sparse", num_select=10_000))
    ctl = apply_loop(
        model, {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}
    )
    torch.testing.assert_close(model(ids).logits, ref, rtol=1e-10, atol=1e-11)  # gate=0 + all buckets
    # pass-2 of a core full-attn layer reuses pass-1's bucket index (one cache entry, no re-clustering)
    ix = mgr.indexers[7]
    calls = {"n": 0}
    orig = ix._cluster_blocks

    def counting(z):
        calls["n"] += 1
        return orig(z)

    ix._cluster_blocks = counting
    mgr.clear_cache()
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.3)
    model(ids)
    assert calls["n"] == 1
    mgr.uninstall()
    ctl.unwrap()
    assert torch.equal(model(ids).logits, ref)


def test_yarn_and_config_validation():
    cfg = ToyConfig()
    cfg.max_position_embeddings = 262144
    rope = apply_yarn(cfg, 1_048_576)
    assert rope["factor"] == 4.0 and cfg.max_position_embeddings == 1_048_576
    with pytest.raises(ValueError):
        LSAConfig(block_size=10, pool=4).validate()
    with pytest.raises(ValueError, match="unknown"):
        LSAConfig.from_dict({"bogus": 1})
    for mode in ("off", "dense", "distill", "sparse", "hybrid"):
        LSAConfig(mode=mode).validate()
    with pytest.raises(ValueError):
        LSAConfig(mode="invalid_mode").validate()


def test_lsa_real_scale_shapes_smoke():
    """Qwen3.6 attention geometry (16 q heads / 2 kv heads / head_dim 256 / rotary 64) on a short sequence."""
    cfg = LSAConfig(
        mode="sparse", rotary_dim=64, block_size=64, window=64, sinks=4, num_select=4, query_chunk=64
    )
    ix = LSAIndexer(16, 2, 256, cfg, layer_id=3)
    mod = torch.nn.Module()
    mod.lsa_indexer = ix
    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        q, k, v = torch.randn(1, 16, 200, 256), torch.randn(1, 2, 200, 256), torch.randn(1, 2, 200, 256)
        out, _ = lsa_attention(mod, q, k, v)
    finally:
        _DENSE_FN[0] = saved
    assert out.shape == (1, 200, 16, 256) and torch.isfinite(out).all()
    assert ix.last_stats["keys_per_query"] < 200


def test_install_rejects_non_full_attention_layers():
    model = build_toy()
    with pytest.raises(ValueError, match="not full-attention"):
        install_lsa(copy.deepcopy(model), dict(SMALL, layers=[0]))


# ============================================================================ D7.1: 2K-8K exact parity


@pytest.mark.parametrize("per_head_select", [False, True])
@pytest.mark.parametrize("seq_len", [2048, 8192])
def test_sparse_equals_dense_multiblock_2k_to_8k_both_head_modes(
    per_head_select: bool, seq_len: int
):
    """D7.1 + D4 + D5 + D6: Exact `sparse == dense` (`atol < 1e-5`) at 2K and 8K when all buckets selected."""
    g_rng = torch.Generator().manual_seed(101 + seq_len + int(per_head_select))
    b, hq, g_kv, dh = 1, 8, 2, 32
    # Use 64-query suffix at 8K so dense reference is instant on CPU while exercising all 16 blocks (8,192 keys)!
    # At 2K, run full 512 causal queries spanning multiple block boundaries (`qpos = 1536..2047`).
    tq = 512 if seq_len == 2048 else 64
    q = torch.randn(b, hq, tq, dh, generator=g_rng, dtype=torch.float32)
    k = torch.randn(b, g_kv, seq_len, dh, generator=g_rng, dtype=torch.float32)
    v = torch.randn(b, g_kv, seq_len, dh, generator=g_rng, dtype=torch.float32)

    # Note: at seq_len=8192 with block_size=512, avg_bucket_keys=8 -> 16 * 64 = 1,024 active slots
    # and 7,616 historical keys (> default max_selected_keys=768!). Setting num_select >= S_active
    # must still capture 100% of historical keys with zero truncation!
    s_active_expected = (seq_len // 512) * 64
    cfg = LSAConfig(
        mode="sparse",
        index_dim=16,
        rotary_dim=8,
        block_size=512,
        pool=4,
        avg_bucket_keys=8,
        max_bucket_keys=32,
        num_select=s_active_expected,
        window=512,
        sinks=64,
        query_chunk=128,
        per_head_select=per_head_select,
        compact_active_slots=True,
        max_selected_keys=768,
    )
    ix = LSAIndexer(hq, g_kv, dh, cfg, layer_id=0)
    if per_head_select:
        assert ix.wq.shape == (1, hq, dh - 8, 16)
        with torch.no_grad():
            ix.wq.normal_(generator=g_rng)

    mod = torch.nn.Module()
    mod.lsa_indexer = ix
    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        with torch.no_grad():
            ref, _ = dense_causal_attention(mod, q, k, v)
            out, _ = lsa_attention(mod, q, k, v)
    finally:
        _DENSE_FN[0] = saved

    max_err = float((out - ref).abs().max())
    assert max_err < 1e-5, f"Expected sparse == dense (< 1e-5), got max_err={max_err}"


def test_tight_valid_member_packing_cap_and_priority():
    """D6: `_pack_valid_members` shrinks gather width to `min(max_valid, max_selected_keys)` and keeps top priority."""
    from qloop.model.lsa import _pack_valid_members

    # 2 queries, Kb = 64 raw member slots (mostly -1 padding, only 6 valid keys in query 0 and 10 in query 1)
    mem = torch.full((1, 2, 2, 64), -1, dtype=torch.long)
    # Query 0 valid keys: [100, 101, 102, 200, 201, 202] (6 valid)
    mem[0, :, 0, :6] = torch.tensor([100, 101, 102, 200, 201, 202])
    # Query 1 valid keys: 10 valid keys in priority order [500..509]
    mem[0, :, 1, 4:14] = torch.arange(500, 510)
    m_b = mem >= 0

    # Case 1: max_valid = 10 < eff_cap = 768 -> L_pack shrinks dynamically to 10 (NOT 64 or 768!)
    packed_small = _pack_valid_members(mem, m_b, tk=1024, eff_cap=768, sync_free=False)
    assert packed_small is not None
    mem_s, mask_s, n_sel_s = packed_small
    assert mem_s.shape == (1, 2, 2, 10)
    assert n_sel_s == 2 * (6 + 10)
    assert sorted(mem_s[0, 0, 0][mask_s[0, 0, 0]].tolist()) == [100, 101, 102, 200, 201, 202]
    assert sorted(mem_s[0, 0, 1][mask_s[0, 0, 1]].tolist()) == list(range(500, 510))

    # Case 2: eff_cap = 5 < max_valid = 10 -> keeps ONLY the 5 highest-priority (earliest) keys!
    packed_capped = _pack_valid_members(mem, m_b, tk=1024, eff_cap=5, sync_free=False)
    assert packed_capped is not None
    mem_c, mask_c, _ = packed_capped
    assert mem_c.shape == (1, 2, 2, 5)
    assert mem_c[0, 0, 0][mask_c[0, 0, 0]].tolist() == [100, 101, 102, 200, 201]
    assert mem_c[0, 0, 1][mask_c[0, 0, 1]].tolist() == [500, 501, 502, 503, 504]


# ============================================================================ D7.2: Per-head recall & D4


def test_compute_lsa_recall_and_per_head_orthogonal_subspace_specialization():
    """D7.2 + D4: `compute_lsa_recall` verifies high recall and `per_head_select=True` advantage on orthogonal heads."""
    torch.manual_seed(2026)
    b, g_kv, hq, tk, tq, dh = 1, 1, 4, 1024, 32, 64
    rot = 16  # 48 NoPE dims = 4 orthogonal 12-d subspaces, one per query head h in 0..3!
    sub_d = (dh - rot) // hq  # 12

    k = torch.randn(b, g_kv, tk, dh) * 0.05
    q = torch.zeros(b, hq, tq, dh)

    # Plant 2 separated 8-key clusters per head across `[64 : 768)` (outside sinks=64 & window=256).
    # Head `h` exclusively queries NoPE subspace `[rot + h*12 : rot + (h+1)*12]`!
    for h in range(hq):
        lo_d = rot + h * sub_d
        hi_d = lo_d + sub_d
        head_dir = F.normalize(torch.randn(sub_d), dim=-1)
        start_a = 64 + h * 160
        start_b = start_a + 64
        k[0, 0, start_a : start_a + 8, lo_d:hi_d] = head_dir * 8.0
        k[0, 0, start_b : start_b + 8, lo_d:hi_d] = head_dir * 8.0
        q[0, h, :, lo_d:hi_d] = head_dir * 4.0

    common_kw = dict(
        mode="distill",
        index_dim=16,
        rotary_dim=rot,
        block_size=256,
        pool=4,
        avg_bucket_keys=8,
        max_bucket_keys=32,
        num_select=2,  # Tight budget: 2 buckets/head captures both clusters; 2 shared across 4 heads cannot!
        window=256,
        sinks=64,
        distill_query_frac=1.0,
        temperature=0.1,
        recall_threshold=0.80,
    )

    ix_group = LSAIndexer(hq, g_kv, dh, LSAConfig(**common_kw, per_head_select=False), layer_id=0)
    ix_head = LSAIndexer(hq, g_kv, dh, LSAConfig(**common_kw, per_head_select=True), layer_id=0)

    # Train both indexers via `_distill_kl` on `(q, k)` for 40 Adam steps
    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        for ix in (ix_group, ix_head):
            mod = torch.nn.Module()
            mod.lsa_indexer = ix
            opt = torch.optim.Adam(ix.parameters(), lr=0.05)
            v_dummy = torch.zeros_like(k)
            for _ in range(40):
                opt.zero_grad()
                lsa_attention(mod, q, k, v_dummy)
                losses = ix.pop_losses()
                assert losses
                losses[0].backward()
                opt.step()
    finally:
        _DENSE_FN[0] = saved

    idx_g = ix_group.build(k)
    idx_h = ix_head.build(k)
    assert idx_g is not None and idx_h is not None
    sel_g = ix_group.select(q, idx_g, qpos=torch.arange(tk - tq, tk))
    sel_h = ix_head.select(q, idx_h, qpos=torch.arange(tk - tq, tk))
    assert sel_g.shape == (b, g_kv, tq, 2)
    assert sel_h.shape == (b, hq, tq, 2)

    rec_group = compute_lsa_recall(q, k, idx_g, sel_g, top_k_keys=16, window=256, sinks=64)
    rec_head = compute_lsa_recall(q, k, idx_h, sel_h, top_k_keys=16, window=256, sinks=64)

    assert len(rec_head["per_head_recall"]) == hq
    assert rec_head["mean_recall"] >= 0.90, f"Expected >= 0.90 recall, got {rec_head}"
    assert rec_head["min_head_recall"] >= 0.85, f"Expected >= 0.85 min head recall, got {rec_head}"
    # Per-head selection strictly improves both mean and worst-head recall under tight bucket budget
    assert rec_head["mean_recall"] > rec_group["mean_recall"] + 0.30
    assert rec_head["min_head_recall"] > rec_group["min_head_recall"] + 0.50

    # Verify `evaluate_recall` updates `last_recall` and flips `recall_ready` to True
    assert not ix_head.recall_ready
    eval_res = ix_head.evaluate_recall(q, k, top_k_keys=16)
    assert ix_head.recall_ready and ix_head.last_recall >= 0.90
    assert abs(eval_res["mean_recall"] - rec_head["mean_recall"]) < 1e-9


# ============================================================================ D7.3: 1M slot compaction


def test_1m_equivalent_slot_compaction_and_slot_scores_memory():
    """D7.3 + D5: 2,048 blocks (1M tokens) compacts `262,144 -> 131,072` active slots (50% reduction)."""
    torch.manual_seed(42)
    b, g_kv, hq, dh, idx_dim = 1, 2, 16, 64, 64

    cfg_compact = LSAConfig(
        mode="sparse",
        index_dim=idx_dim,
        rotary_dim=16,
        block_size=512,
        pool=4,
        avg_bucket_keys=8,
        max_bucket_keys=32,
        num_select=64,
        window=512,
        sinks=64,
        compact_active_slots=True,
    )
    cfg_full = copy.deepcopy(cfg_compact)
    cfg_full.compact_active_slots = False

    assert cfg_compact.slots_per_block() == 64
    assert cfg_full.slots_per_block() == 128

    ix_compact = LSAIndexer(hq, g_kv, dh, cfg_compact, layer_id=0)
    ix_full = LSAIndexer(hq, g_kv, dh, cfg_full, layer_id=0)
    ix_full.load_state_dict(ix_compact.state_dict())

    # 1) Verify exact output equivalence between compacted & uncompacted indexers on 8 blocks (4,096 tokens)
    # plus incremental block-append cache (`nb=4 -> nb=8`)
    k_4k = torch.randn(b, g_kv, 4096, dh)
    q_tail = torch.randn(b, hq, 64, dh)
    qpos_tail = torch.arange(4096 - 64, 4096)

    bi_c_part = ix_compact.build(k_4k[:, :, :2048], cache_key="seq_inc")
    assert bi_c_part is not None and bi_c_part.centroids.shape[2] == 4 * 64
    bi_c = ix_compact.build(k_4k, cache_key="seq_inc")
    bi_f = ix_full.build(k_4k)
    assert bi_c is not None and bi_f is not None
    assert bi_c.centroids.shape == (b, g_kv, 8 * 64, idx_dim)
    assert bi_f.centroids.shape == (b, g_kv, 8 * 128, idx_dim)

    # Fresh build must match incremental build bit-for-bit
    ix_compact._cache.clear()
    bi_c_fresh = ix_compact.build(k_4k)
    assert bi_c_fresh is not None
    torch.testing.assert_close(bi_c.centroids, bi_c_fresh.centroids)
    assert torch.equal(bi_c.members, bi_c_fresh.members)
    assert torch.equal(bi_c.key2slot, bi_c_fresh.key2slot)

    # Top-64 selected member key sets must match identically between compacted and uncompacted indices
    qz_tail = ix_compact.query_latent(q_tail)
    top_c = ix_compact.slot_scores(qz_tail, bi_c, qpos_tail).topk(64, dim=-1).values
    top_f = ix_full.slot_scores(qz_tail, bi_f, qpos_tail).topk(64, dim=-1).values
    torch.testing.assert_close(top_c, top_f, rtol=1e-6, atol=1e-6)

    # 2) True 1M-token scale (`nb = 2,048` blocks = `1,048,576` tokens):
    # Cluster 4 representative blocks and tile across 2,048 blocks to verify full 1M BucketIndex & slot_scores!
    nb_1m = 2048
    z_seed = F.normalize(torch.randn(b, g_kv, 4, 512, idx_dim), dim=-1)
    lab_c, slab_c, scnt_c, umem_c = ix_compact._cluster_blocks(z_seed)
    cent_c4 = ix_compact._centroids(z_seed.view(b, g_kv, 4, 128, 4, idx_dim).sum(4), slab_c, scnt_c, 64)

    lab_f, slab_f, scnt_f, umem_f = ix_full._cluster_blocks(z_seed)
    cent_f4 = ix_full._centroids(z_seed.view(b, g_kv, 4, 128, 4, idx_dim).sum(4), slab_f, scnt_f, 128)

    rep = nb_1m // 4
    bi_1m_compact = ix_compact._assemble(
        lab_c.repeat(1, 1, rep, 1),
        slab_c.repeat(1, 1, rep, 1),
        scnt_c.repeat(1, 1, rep, 1),
        umem_c.repeat(1, 1, rep, 1, 1),
        cent_c4.repeat(1, 1, rep, 1, 1),
    )
    bi_1m_full = ix_full._assemble(
        lab_f.repeat(1, 1, rep, 1),
        slab_f.repeat(1, 1, rep, 1),
        scnt_f.repeat(1, 1, rep, 1),
        umem_f.repeat(1, 1, rep, 1, 1),
        cent_f4.repeat(1, 1, rep, 1, 1),
    )

    assert bi_1m_full.centroids.shape == (1, 2, 262_144, 64)
    assert bi_1m_compact.centroids.shape == (1, 2, 131_072, 64)
    assert bi_1m_compact.centroids.shape[2] * 2 == bi_1m_full.centroids.shape[2]

    bytes_compact = bi_1m_compact.centroids.nbytes + bi_1m_compact.members.nbytes + bi_1m_compact.active.nbytes
    bytes_full = bi_1m_full.centroids.nbytes + bi_1m_full.members.nbytes + bi_1m_full.active.nbytes
    assert bytes_compact * 2 == bytes_full

    # Score a 64-query chunk at position ~1M against all 131,072 active slots (`[1, 2, 64, 131072]`)
    qz_1m = F.normalize(torch.randn(1, 2, 64, 64), dim=-1)
    qpos_1m = torch.arange(1_048_576 - 64, 1_048_576)
    scores_1m = ix_compact.slot_scores(qz_1m, bi_1m_compact, qpos_1m)
    assert scores_1m.shape == (1, 2, 64, 131_072)
    # Last block (`block 2047`, `slot_end == 1_048_576 > frontier == 1_048_064`) must be causally masked to -inf!
    assert bool((scores_1m[..., -64:] == float("-inf")).all())
    assert bool(torch.isfinite(scores_1m[..., :-64]).all())


# ============================================================================ D7.4: Hybrid mode


def test_hybrid_mode_alternating_sparse_forward_and_dual_gradients():
    """D7.4 + D1: `mode='hybrid'` alternates sparse forward + KL and routes downstream grads sparsely."""
    torch.manual_seed(314)
    t_cfg = ToyConfig(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=1,
        full_attention_interval=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(t_cfg).double()
    ids = _ids(t=40, b=1, seed=17)

    with torch.no_grad():
        ref_dense = model(ids).logits.detach().clone()

    # Compare against pure sparse mode with identical indexer seed
    model_sparse_ref = copy.deepcopy(model)
    torch.manual_seed(999)
    install_lsa(model_sparse_ref, dict(SMALL, mode="sparse", num_select=1))
    with torch.no_grad():
        ref_sparse = model_sparse_ref(ids).logits.detach().clone()
    assert not torch.allclose(ref_sparse, ref_dense), "num_select=1 must differ from dense"

    # Install `hybrid` mode with unreachable initial threshold `0.999` first -> stays dense until `recall_ready`
    torch.manual_seed(999)
    mgr = install_lsa(
        model,
        dict(
            SMALL,
            mode="hybrid",
            num_select=1,
            recall_threshold=0.999,
            hybrid_sparse_every=2,
            distill_weight=1.0,
            distill_query_frac=1.0,
        ),
    )
    ix = mgr.indexers[0]

    # Step 0 before recall_ready: forward is DENSE, but KL loss still trains W_q_idx & W_k_idx!
    mgr.set_step(0)
    model.zero_grad(set_to_none=True)
    out_pre = model(ids).logits
    torch.testing.assert_close(out_pre.detach(), ref_dense, rtol=1e-10, atol=1e-11)
    assert ix.last_stats["forward_sparse"] == 0.0
    kl_pre = mgr.aux_loss()
    assert kl_pre is not None
    kl_pre.backward()
    assert ix.W_q_idx.grad is not None and float(ix.W_q_idx.grad.norm()) > 0.0
    assert ix.W_k_idx.grad is not None and float(ix.W_k_idx.grad.norm()) > 0.0

    # Unlock `recall_ready`: Step 2 (`2 % 2 == 0`) runs SPARSE forward + KL; Step 3 (`3 % 2 == 1`) runs DENSE + KL!
    mgr.recall_ready = True
    ix.recall_ready = True

    mgr.set_step(2)
    model.zero_grad(set_to_none=True)
    out_even = model(ids).logits
    torch.testing.assert_close(out_even.detach(), ref_sparse, rtol=1e-10, atol=1e-11)
    assert ix.last_stats["forward_sparse"] == 1.0
    kl_even = mgr.aux_loss()
    assert kl_even is not None
    (out_even.sum() + kl_even).backward()
    assert ix.W_q_idx.grad is not None and float(ix.W_q_idx.grad.norm()) > 0.0
    assert ix.W_k_idx.grad is not None and float(ix.W_k_idx.grad.norm()) > 0.0
    attn0 = model.model.layers[0].self_attn
    assert attn0.q_proj.weight.grad is not None and float(attn0.q_proj.weight.grad.norm()) > 0.0
    assert attn0.o_proj.weight.grad is not None and float(attn0.o_proj.weight.grad.norm()) > 0.0

    # Prove downstream value gradients flow EXCLUSIVELY through the sparse key subset on even hybrid steps!
    q_raw = torch.randn(1, 4, 8, 8, dtype=torch.float64, requires_grad=True)
    k_raw = torch.randn(1, 2, 40, 8, dtype=torch.float64, requires_grad=True)
    v_raw = torch.randn(1, 2, 40, 8, dtype=torch.float64, requires_grad=True)
    mod_probe = torch.nn.Module()
    mod_probe.lsa_indexer = ix
    ix.pop_losses()
    out_probe, _ = lsa_attention(mod_probe, q_raw, k_raw, v_raw)
    kl_probe = ix.pop_losses()[0]
    (out_probe.sum() + kl_probe).backward()
    assert v_raw.grad is not None
    # Historical region is `[sinks=2 : 40 - window - 7 = 25]`: with `num_select=1` (<= 4 keys/bucket),
    # at least half of historical positions `2..24` are unselected and MUST have exact `0.0` value gradient!
    hist_v_grad_norms = v_raw.grad[0, :, 2:24].abs().sum(dim=-1)  # [2, 22]
    assert int((hist_v_grad_norms == 0.0).sum()) >= 20
    assert int((hist_v_grad_norms > 0.0).sum()) >= 1

    # Odd step (`step = 3`): alternates back to DENSE forward + KL!
    mgr.set_step(3)
    model.zero_grad(set_to_none=True)
    out_odd = model(ids).logits
    torch.testing.assert_close(out_odd.detach(), ref_dense, rtol=1e-10, atol=1e-11)
    assert ix.last_stats["forward_sparse"] == 0.0
    assert mgr.aux_loss() is not None

    # Verify `temporary_mode("off")` and restoration on exception
    with pytest.raises(RuntimeError, match="boom"), mgr.temporary_mode("off"):
        assert mgr.cfg.mode == "off"
        torch.testing.assert_close(model(ids).logits.detach(), ref_dense, rtol=1e-10, atol=1e-11)
        raise RuntimeError("boom")
    assert mgr.cfg.mode == "hybrid"


# ============================================================================ D7.5: Zero host sync


def test_zero_host_sync_ward_cluster_and_indexer_build():
    """D7.5 + D8: Zero `.item()` / `bool(tensor)` / `nonzero` syncs in `ward_cluster` and `LSAIndexer.build`."""
    torch.manual_seed(77)
    cfg = LSAConfig(
        mode="sparse",
        index_dim=16,
        rotary_dim=8,
        block_size=64,
        pool=4,
        avg_bucket_keys=8,
        max_bucket_keys=16,
        num_select=4,
        window=64,
        sinks=4,
        compact_active_slots=True,
    )
    ix = LSAIndexer(8, 2, 32, cfg, layer_id=0).eval()
    x_ward = torch.randn(6, 16, 8)
    k_short = torch.randn(1, 2, 128, 32)
    k_long = torch.cat([k_short, torch.randn(1, 2, 128, 32)], dim=2)

    with torch.no_grad(), _ZeroHostSyncGuard() as sync_guard:
        # 1) Both normal and cap-constrained `ward_cluster` execute 0 host syncs
        lab_norm = ward_cluster(x_ward, n_clusters=8, max_size=4)
        lab_capped = ward_cluster(x_ward, n_clusters=1, max_size=3)
        # 2) Initial `LSAIndexer.build` + incremental `cache_key` append execute 0 host syncs
        b1 = ix.build(k_short, cache_key="seq_zero_sync")
        b2 = ix.build(k_long, cache_key="seq_zero_sync")
        # 3) Unkeyed `build` + in-place mutation rebuild execute 0 host syncs
        k_mut = k_short.clone()
        b_mut_1 = ix.build(k_mut)
        c1_snap = b_mut_1.centroids.clone() if b_mut_1 is not None else None
        k_mut.roll(shifts=8, dims=-1)
        k_mut[:, :, :64] = torch.randn_like(k_mut[:, :, :64])
        b_mut_2 = ix.build(k_mut)

    assert sync_guard.violations == [], f"Unexpected host-sync ops: {sync_guard.violations}"
    assert lab_norm.shape == (6, 16) and lab_capped.shape == (6, 16)
    assert b1 is not None and b2 is not None
    assert b1.n_blocks == 2 and b2.n_blocks == 4
    assert torch.equal(b2.centroids[:, :, : b1.centroids.shape[2]], b1.centroids)
    # Verify in-place mutation without `cache_key` did NOT return stale cached centroids!
    assert b_mut_2 is not None and c1_snap is not None
    assert not torch.allclose(b_mut_2.centroids, c1_snap)
