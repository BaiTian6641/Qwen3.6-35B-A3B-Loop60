"""Unit & parity tests for single-Blackwell `s2_joint_bptt` throughput optimizations."""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from qloop.data.decontam import Decontaminator, ngram_hashes, normalize_words
from qloop.data.packing import Packer
from qloop.model.hssd import (
    HSSDCacheState,
    HSSDConfig,
    _replay_chunk_on_bank,
    chunk_hssd,
    install_hssd,
)
from qloop.model.lsa import (
    LSAConfig,
    LSAIndexer,
    _distill_kl,
    _pack_valid_members,
    _run_detached_distill_kl,
    _sparse_chunk,
    _TritonSparseGatherAttnFn,
    install_lsa,
)
from qloop.model.quant_autograd import (
    _contiguous_id_slice,
    _maybe_triton_dequant_nvfp4,
    _slice_expert_scale_batch,
    dequantize_nvfp4_tensor,
)
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train import bptt as bptt_mod
from qloop.train.bptt import (
    BPTTConfig,
    SegmentedKVBPTTRunner,
    _gather_prior_field,
    _offload_prior_record,
)
from qloop.train.optim import FusedStepOptimizer
from qloop.train.trainer import _AsyncBatchPrefetcher, _to_device


# ==============================================================================
# 1. Optimizer (`optim.py`): Batched non-fused AdamWLite + _foreach_norm + guard
# ==============================================================================
def _make_mixed_opt_params(seed: int = 17) -> list[tuple[str, nn.Parameter]]:
    g = torch.Generator().manual_seed(seed)
    return [
        ("layers.0.self_attn.q_norm.weight", nn.Parameter(torch.randn(16, generator=g, dtype=torch.float32))),
        ("layers.0.self_attn.k_proj.weight", nn.Parameter(torch.randn(16, 16, generator=g, dtype=torch.bfloat16))),
        ("layers.1.self_attn.v_proj.weight", nn.Parameter(torch.randn(16, 16, generator=g, dtype=torch.bfloat16))),
        ("layers.0.mlp.gate_proj.weight", nn.Parameter(torch.randn(128, 64, generator=g, dtype=torch.bfloat16))),
    ]


def test_optim_batched_nonfused_parity_and_atomic_nonfinite_guard(monkeypatch):
    """Batched `_foreach_norm` + `_step_batched_adamw_lite` matches unbatched reference & keeps atomic guard."""
    p_fast = _make_mixed_opt_params(seed=101)
    p_ref = [(n, nn.Parameter(p.detach().clone())) for n, p in p_fast]

    ocfg = {
        "lr": 1e-2,
        "weight_decay": 0.01,
        "warmup_steps": 2,
        "total_steps": 10,
        "fused": False,
        "max_grad_norm": 0.5,
        "adamw_lite_bf16_master": True,
        "apollo_min_params": 4096,
        "apollo_rank": 8,
        "apollo_backend": "fallback",
        "seed": 42,
    }
    opt_fast = FusedStepOptimizer.from_config(p_fast, dict(ocfg), grad_accum=1)
    opt_ref = FusedStepOptimizer.from_config(p_ref, dict(ocfg), grad_accum=1)

    # Force reference optimizer to take per-param unbatched path by hiding `_step_batched_adamw_lite`
    # and `_foreach_norm` during reference steps.
    g = torch.Generator().manual_seed(202)
    for _step in range(3):
        for (_nf, pf), (_nr, pr) in zip(p_fast, p_ref):
            grad = torch.randn(pf.shape, generator=g, dtype=pf.dtype) * 2.0
            pf.grad = grad.clone()
            pr.grad = grad.clone()

        st_fast = opt_fast.step()
        with monkeypatch.context() as mp:
            mp.setattr(opt_ref, "_step_batched_adamw_lite", lambda *a, **kw: None)
            # Call single-param fallback via `_step_one` for every param on reference
            orig_nonfused = opt_ref._step_nonfused

            def _unbatched_ref() -> None:
                active = [e for e in opt_ref.entries.values() if e.param.grad is not None]
                if not active:
                    return
                with torch.no_grad():
                    sq_sum = 0.0
                    for e in active:
                        assert e.param.grad is not None
                        dt = torch.float64 if e.param.grad.dtype == torch.float64 else torch.float32
                        gn = torch.linalg.vector_norm(e.param.grad.detach(), dtype=dt)
                        fin = torch.isfinite(gn)
                        opt_ref._param_finite[e.name] = fin
                        if bool(fin):
                            sq_sum += float(gn.double().square())
                    opt_ref._step_gnorm_sq = torch.tensor(sq_sum, dtype=torch.float64)
                    if opt_ref.skip_nonfinite and not all(bool(opt_ref._param_finite[e.name]) for e in active):
                        opt_ref._atomic_step_aborted = True
                        return
                    for e in active:
                        opt_ref._stepped.add(e.name)
                    if opt_ref.max_grad_norm is not None:
                        tot_exact = math.sqrt(max(0.0, sq_sum))
                        coef_exact = min(1.0, float(opt_ref.max_grad_norm) / (tot_exact + 1e-6))
                        for e in active:
                            assert e.param.grad is not None
                            e.param.grad.mul_(coef_exact)
                lrf = opt_ref.lr_factor
                if opt_ref._torch_adamw is not None and any(e.opt is opt_ref._torch_adamw for e in active):
                    for pg in opt_ref._torch_adamw.param_groups:
                        pg["lr"] = float(pg.get("_qloop_base_lr", pg["lr"])) * lrf
                    opt_ref._torch_adamw.step()
                for e in active:
                    if e.opt is opt_ref._torch_adamw:
                        continue
                    for pg in e.opt.param_groups:
                        pg["lr"] = e.base_lr * lrf
                    e.opt.step()
                    opt_ref._clamp_projected_param(e.param, e.opt)

            mp.setattr(opt_ref, "_step_nonfused", _unbatched_ref)
            st_ref = opt_ref.step()
            _ = orig_nonfused

        assert math.isclose(
            st_fast["optim/global_grad_norm"], st_ref["optim/global_grad_norm"], rel_tol=1e-5, abs_tol=1e-6
        )
        for (name, pf), (_, pr) in zip(p_fast, p_ref):
            tol = 1e-2 if pf.dtype == torch.bfloat16 else 1e-6
            torch.testing.assert_close(pf, pr, rtol=tol, atol=tol, msg=f"mismatch on {name}")

    # Verify state_dict round-trip compatibility
    sd = opt_fast.state_dict()
    opt_reload = FusedStepOptimizer.from_config(
        [(n, nn.Parameter(p.detach().clone())) for n, p in p_fast], dict(ocfg), grad_accum=1
    )
    opt_reload.load_state_dict(sd)
    assert opt_reload.step_count == opt_fast.step_count

    # Atomic whole-step non-finite guard: poison ONE tensor -> ZERO params updated
    snap = [p.detach().clone() for _, p in p_fast]
    for i, (_, pf) in enumerate(p_fast):
        pf.grad = torch.ones_like(pf)
        if i == 1:
            pf.grad.view(-1)[0] = float("nan")
    st_bad = opt_fast.step()
    assert st_bad["optim/stepped_params"] == 0.0
    assert st_bad["optim/skipped_params"] >= 1.0
    for (_, pf), before in zip(p_fast, snap):
        torch.testing.assert_close(pf, before, rtol=0.0, atol=0.0)


# ==============================================================================
# 2. NVFP4 Dequant & Contiguous Expert Slicing (`quant_autograd.py`)
# ==============================================================================
def _pack_e2m1_nibbles(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return ((lo.to(torch.uint8) & 0xF) | ((hi.to(torch.uint8) & 0xF) << 4)).contiguous()


def test_nvfp4_contiguous_slicing_and_dequant_cpu():
    assert _contiguous_id_slice(torch.tensor([64, 65, 66, 67])) == slice(64, 68)
    assert _contiguous_id_slice(torch.tensor([0, 2, 4])) is None
    assert _contiguous_id_slice(torch.tensor([5])) == slice(5, 6)

    scale_3d = torch.randn(16, 4, 8)
    eids = torch.arange(4, 12, dtype=torch.long)
    sl = _contiguous_id_slice(eids)
    assert sl == slice(4, 12)
    out_fast = _slice_expert_scale_batch(scale_3d, eids, num_experts=16, id_slice=sl)
    out_ref = _slice_expert_scale_batch(scale_3d, eids, num_experts=16, id_slice=None)
    assert out_fast is not None and out_ref is not None
    torch.testing.assert_close(out_fast, out_ref, rtol=0.0, atol=0.0)

    # E2M1 dequant sanity across 2D & 3D + divisor / multiplier
    g = torch.Generator().manual_seed(7)
    lo = torch.randint(0, 16, (4, 8, 16), generator=g)
    hi = torch.randint(0, 16, (4, 8, 16), generator=g)
    packed = _pack_e2m1_nibbles(lo, hi)  # [4, 8, 16] -> [4, 8, 32]
    ws = torch.rand(4, 8, 2, generator=g) + 0.25
    gs = torch.tensor([1.5, 2.0, 0.5, 3.0]).view(4, 1, 1)
    y_div = dequantize_nvfp4_tensor(packed, ws, gs, dtype=torch.float32, global_scale_mode="divisor")
    y_mul = dequantize_nvfp4_tensor(packed, ws, 1.0 / gs, dtype=torch.float32, global_scale_mode="multiplier")
    assert y_div.shape == (4, 8, 32)
    torch.testing.assert_close(y_div, y_mul, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton NVFP4 dequant parity")
@pytest.mark.parametrize("mode", ["divisor", "multiplier"])
@pytest.mark.parametrize("shape_3d", [False, True])
def test_cuda_triton_nvfp4_dequant_parity(mode: str, shape_3d: bool):
    g = torch.Generator().manual_seed(33)
    if shape_3d:
        lo = torch.randint(0, 16, (8, 32, 64), generator=g)
        hi = torch.randint(0, 16, (8, 32, 64), generator=g)
        ws = (torch.rand(8, 32, 8, generator=g) + 0.1).to(dtype=torch.bfloat16, device="cuda")
        gs = (torch.rand(8, 1, 1, generator=g) + 0.5).to(dtype=torch.float32, device="cuda")
    else:
        lo = torch.randint(0, 16, (64, 64), generator=g)
        hi = torch.randint(0, 16, (64, 64), generator=g)
        ws = (torch.rand(64, 8, generator=g) + 0.1).to(dtype=torch.bfloat16, device="cuda")
        gs = torch.tensor(2.25, dtype=torch.float32, device="cuda")
    packed = _pack_e2m1_nibbles(lo, hi).to("cuda")

    tri = _maybe_triton_dequant_nvfp4(
        packed, ws, gs, dtype=torch.bfloat16, group_size=16, global_scale_mode=mode
    )
    assert tri is not None
    # Force eager PyTorch path by computing on CPU then casting
    ref = dequantize_nvfp4_tensor(
        packed.cpu(), ws.cpu(), gs.cpu(), dtype=torch.bfloat16, group_size=16, global_scale_mode=mode
    ).to("cuda")
    torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)


# ==============================================================================
# 3. HSSD v3.2 (`hssd.py`): `_all_swa_valid` propagation & replay bank parity
# ==============================================================================
def test_hssd_cache_state_all_swa_valid_and_replay_bank_parity():
    st = HSSDCacheState.zeros(
        batch_size=1,
        num_pages=4,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
        chunk_size=4,
        window_chunks=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        compact_decode_cache=True,
    )
    assert st._all_swa_valid is None
    st._all_swa_valid = True
    assert st.clone()._all_swa_valid is True
    assert st.detach()._all_swa_valid is True
    assert st.contiguous()._all_swa_valid is True
    assert st.to(dtype=torch.float32)._all_swa_valid is True
    st_dst = HSSDCacheState.zeros(
        1,
        4,
        2,
        8,
        8,
        chunk_size=4,
        window_chunks=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        compact_decode_cache=True,
    )
    st_dst.copy_(st)
    assert st_dst._all_swa_valid is True
    st_dst.zero_()
    assert st_dst._all_swa_valid is False

    # Replay bank parity: sync_free=True vs sync_free=False must be bit-identical
    g = torch.Generator().manual_seed(9)
    max_win = (2 + 1) * 4
    bank = torch.randn(1, 4, 2, 8, 8, generator=g)
    ke = torch.randn(1, 2, max_win, 8, generator=g)
    ve = torch.randn(1, 2, max_win, 8, generator=g)
    gg = torch.rand(1, max_win, 2, 2, generator=g)
    gam = torch.rand(1, max_win, 2, 2, 8, generator=g) * 0.9
    pidx = torch.randint(0, 4, (1, max_win, 2), generator=g)
    rbuf = (ke, ve, gg, gam, pidx)
    out_c = _replay_chunk_on_bank(bank, rbuf, chunk_idx=0, chunk_size=4, window_chunks=2, sync_free=False)
    out_u = _replay_chunk_on_bank(bank, rbuf, chunk_idx=0, chunk_size=4, window_chunks=2, sync_free=True)
    torch.testing.assert_close(out_c, out_u, rtol=1e-6, atol=1e-6)

    # Multi-segment chunk_hssd parity with state carry
    q = torch.randn(1, 16, 2, 8, generator=g)
    k = torch.randn(1, 16, 2, 8, generator=g)
    v = torch.randn(1, 16, 2, 8, generator=g)
    gamma = (torch.full((4,), 0.95), torch.full((1, 16, 2, 4), 0.95))
    bg = torch.full((1, 16, 2, 1), 0.5)
    wg = torch.full((1, 16, 2, 1), 0.5)
    th = torch.zeros(1, 16, 2, 4)
    sb = torch.zeros(1, 16, 2, 1)
    ss = torch.ones(2)
    st_th = torch.zeros(2)
    wl = torch.randn(1, 16, 4, generator=g)
    rl = torch.randn(1, 16, 4, generator=g)
    full_o, _ = chunk_hssd(
        q, k, v, gamma, bg, wg, th, sb, ss, st_th, wl, rl,
        q_swa=q, k_swa=k, v_swa=v,
        num_pages=4, chunk_size=4, window_chunks=2, top_k_pages=2, segment_chunks=2, checkpoint_segments=False,
    )
    o1, s1 = chunk_hssd(
        q[:, :8], k[:, :8], v[:, :8], (gamma[0], gamma[1][:, :8]), bg[:, :8], wg[:, :8],
        th[:, :8], sb[:, :8], ss, st_th, wl[:, :8], rl[:, :8],
        q_swa=q[:, :8], k_swa=k[:, :8], v_swa=v[:, :8],
        num_pages=4, chunk_size=4, window_chunks=2, top_k_pages=2, segment_chunks=2, checkpoint_segments=False,
        output_final_state=True,
    )
    o2, _ = chunk_hssd(
        q[:, 8:], k[:, 8:], v[:, 8:], (gamma[0], gamma[1][:, 8:]), bg[:, 8:], wg[:, 8:],
        th[:, 8:], sb[:, 8:], ss, st_th, wl[:, 8:], rl[:, 8:],
        q_swa=q[:, 8:], k_swa=k[:, 8:], v_swa=v[:, 8:],
        num_pages=4, chunk_size=4, window_chunks=2, top_k_pages=2, segment_chunks=2, checkpoint_segments=False,
        initial_state=s1, output_final_state=True,
    )
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), full_o, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 4. D-LSA (`lsa.py`): Cached BucketIndex, analytic distill KL, CUDA Triton parity
# ==============================================================================
def test_lsa_bucket_index_reuse_and_pack_members_dtype():
    cfg = LSAConfig(
        mode="sparse",
        index_dim=16,
        rotary_dim=4,
        pool=2,
        block_size=16,
        avg_bucket_keys=4,
        max_bucket_keys=8,
        num_select=4,
        window=16,
        sinks=4,
        query_chunk=8,
    )
    ix = LSAIndexer(num_heads=4, num_kv_heads=2, head_dim=16, cfg=cfg, layer_id=0)
    k = torch.randn(1, 2, 64, 16)
    with torch.no_grad():
        bi1 = ix.build(k)
        bi2 = ix.build(k)
        assert bi1 is not None and bi2 is bi1  # exact cached assembled BucketIndex reuse!
        ix.wk.add_(0.01)  # bumps wk._version -> forces clean rebuild
        bi3 = ix.build(k)
        assert bi3 is not None and bi3 is not bi1

    # `_pack_valid_members` int32 vs int64 output parity
    mem = torch.tensor([[[[5, 2, 5, -1, 1, 9, 2, 4]]]], dtype=torch.long)
    m_b = mem >= 0
    p_i64 = _pack_valid_members(mem, m_b, tk=64, eff_cap=6, sync_free=True, out_dtype=torch.long)
    p_i32 = _pack_valid_members(mem, m_b, tk=64, eff_cap=6, sync_free=True, out_dtype=torch.int32)
    assert p_i64 is not None and p_i32 is not None
    assert p_i32[0].dtype == torch.int32
    torch.testing.assert_close(p_i32[0].long(), p_i64[0])
    torch.testing.assert_close(p_i32[1], p_i64[1])


@pytest.mark.parametrize("per_head", [False, True])
def test_lsa_fast_detached_distill_kl_matches_autograd_reference(per_head: bool):
    """Analytic `(d_qz, d_cent)` + single `autograd.grad` matches reference `_distill_kl` to fp32/fp64 precision."""
    cfg = LSAConfig(
        mode="distill",
        index_dim=16,
        rotary_dim=4,
        pool=2,
        block_size=16,
        avg_bucket_keys=4,
        max_bucket_keys=8,
        num_select=4,
        window=16,
        sinks=4,
        distill_weight=0.5,
        distill_query_frac=0.35,  # forces non-divisible nq % distill_chunk != 0!
        distill_chunk=5,
        per_head_select=per_head,
    )
    torch.manual_seed(123)
    ix_fast = LSAIndexer(num_heads=4, num_kv_heads=2, head_dim=16, cfg=cfg, layer_id=0).double()
    ix_ref = copy.deepcopy(ix_fast)

    q = torch.randn(1, 4, 48, 16, dtype=torch.float64)
    k = torch.randn(1, 2, 48, 16, dtype=torch.float64)
    qpos_all = torch.arange(48)
    scaling = 16 ** -0.5

    with torch.no_grad():
        index_fast = ix_fast.build(k)
        index_ref = ix_ref.build(k)
    assert index_fast is not None and index_ref is not None

    # Same RNG seed so both pick the exact same query subset!
    torch.manual_seed(999)
    _run_detached_distill_kl(q, k, qpos_all, index_fast, ix_fast, ix_fast.active_wq(pass_idx=1), scaling, cur_pass=1)
    loss_fast = ix_fast.pop_losses()[0]
    loss_fast.backward()

    # Compute exact reference using original chunk-by-chunk `_distill_kl` + autograd
    torch.manual_seed(999)
    cand = torch.nonzero((qpos_all // cfg.block_size) > 0).flatten()
    nq = max(1, int(cand.numel() * cfg.distill_query_frac))
    pick = cand[torch.randperm(cand.numel())[:nq]].sort().values
    qg = q.view(1, 2, 2, 48, 16)
    kls = []
    # Build differentiable centroids matching ix_ref.wk
    bs, pool, n_units = cfg.block_size, cfg.pool, cfg.block_size // cfg.pool
    spb = index_ref.slots_per_block
    nb = index_ref.n_blocks
    z_all = ix_ref.key_latent(k[:, :, : nb * bs]).view(1, 2, nb, n_units, pool, -1).sum(4)
    assert index_ref.slot_labels is not None and index_ref.slot_counts is not None
    cent_live = ix_ref._centroids(z_all, index_ref.slot_labels, index_ref.slot_counts, spb).reshape(1, 2, nb * spb, -1)
    from dataclasses import replace as dc_replace

    idx_live = dc_replace(index_ref, centroids=cent_live)
    for s0 in range(0, pick.numel(), cfg.distill_chunk):
        sel = pick[s0 : s0 + cfg.distill_chunk]
        qz = ix_ref.query_latent(q[:, :, sel], pass_idx=1)
        kl = _distill_kl(qg[:, :, :, sel], qz, qpos_all[sel], k.detach(), idx_live, ix_ref, scaling)
        if kl is not None:
            kls.append(kl)
    loss_ref = cfg.distill_weight * torch.stack(kls).mean()
    loss_ref.backward()

    torch.testing.assert_close(loss_fast.detach(), loss_ref.detach(), rtol=1e-10, atol=1e-10)
    assert ix_fast.wq.grad is not None and ix_ref.wq.grad is not None
    assert ix_fast.wk.grad is not None and ix_ref.wk.grad is not None
    torch.testing.assert_close(ix_fast.wq.grad, ix_ref.wq.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(ix_fast.wk.grad, ix_ref.wk.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton LSA kernel parity")
@pytest.mark.parametrize("with_buckets", [False, True])
def test_cuda_triton_sparse_gather_attn_fwd_bwd_parity(with_buckets: bool):
    """Fused Triton Sparse-Gather FlashAttention matches PyTorch `_sparse_chunk` in fwd + dq/dk/dv bwd."""
    torch.manual_seed(55)
    b, g, hg, dh = 1, 2, 8, 64
    tk, tq = 1024, 512
    q_base = tk - tq
    cfg = LSAConfig(
        mode="sparse",
        index_dim=32,
        rotary_dim=16,
        pool=4,
        block_size=128,
        avg_bucket_keys=8,
        max_bucket_keys=32,
        num_select=8,
        window=128,
        sinks=32,
        max_selected_keys=128,
        query_chunk=128,
    )
    ix = LSAIndexer(num_heads=g * hg, num_kv_heads=g, head_dim=dh, cfg=cfg, layer_id=0).cuda()
    q_ref = torch.randn(b, g, hg, tq, dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_ref = torch.randn(b, g, tk, dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v_ref = torch.randn(b, g, tk, dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    scaling = dh ** -0.5
    qpos_all = torch.arange(q_base, tk, device="cuda")
    with torch.no_grad():
        index = ix.build(k_ref) if with_buckets else None
        qz_all = ix.query_latent(q_ref.detach().reshape(b, g * hg, tq, dh), pass_idx=1)

    # PyTorch reference across chunks
    ref_chunks = []
    for c0 in range(0, tq, cfg.query_chunk):
        c1 = min(tq, c0 + cfg.query_chunk)
        ref_chunks.append(
            _sparse_chunk(
                q_ref[:, :, :, c0:c1],
                qz_all[:, :, c0:c1],
                qpos_all[c0:c1],
                k_ref,
                v_ref,
                index,
                ix,
                scaling,
                cur_pass=1,
                q_bounds=(q_base + c0, q_base + c1 - 1),
                record_stats=False,
            )
        )
    out_ref = torch.cat(ref_chunks, dim=3)

    from qloop.model.lsa import _build_sparse_mem_table

    mem_idx = _build_sparse_mem_table(
        qz_all,
        qpos_all,
        index,
        ix,
        tk=tk,
        q_base_pos=q_base,
        cur_pass=1,
        record_tail_len=cfg.query_chunk,
        record_stats=False,
    )
    out_tri = _TritonSparseGatherAttnFn.apply(
        q_tri, k_tri, v_tri, mem_idx, q_base, cfg.resolved_window(), min(cfg.sinks, tk), scaling
    )

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_tri.backward(grad_out)

    for label, a, b_t in [
        ("out", out_tri, out_ref),
        ("dq", q_tri.grad, q_ref.grad),
        ("dk", k_tri.grad, k_ref.grad),
        ("dv", v_tri.grad, v_ref.grad),
    ]:
        assert a is not None and b_t is not None
        cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b_t.float().flatten(), dim=0).item()
        assert cos > 0.9995, f"{label} cosine similarity too low: {cos}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton LSA distill parity")
def test_cuda_triton_distill_kl_parity():
    torch.manual_seed(77)
    cfg = LSAConfig(
        mode="distill",
        index_dim=32,
        rotary_dim=16,
        pool=4,
        block_size=128,
        avg_bucket_keys=8,
        max_bucket_keys=32,
        num_select=8,
        window=128,
        sinks=32,
        distill_weight=1.0,
        distill_query_frac=0.25,
        distill_chunk=16,
    )
    ix_tri = LSAIndexer(num_heads=16, num_kv_heads=2, head_dim=64, cfg=cfg, layer_id=0).cuda()
    ix_ref = copy.deepcopy(ix_tri)
    q = torch.randn(1, 16, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.bfloat16)
    qpos_all = torch.arange(512, device="cuda")
    scaling = 64 ** -0.5

    with torch.no_grad():
        idx_t = ix_tri.build(k)
        idx_r = ix_ref.build(k)
    assert idx_t is not None and idx_r is not None

    torch.manual_seed(4242)
    _run_detached_distill_kl(q, k, qpos_all, idx_t, ix_tri, ix_tri.active_wq(pass_idx=1), scaling, cur_pass=1)
    loss_tri = ix_tri.pop_losses()[0]
    loss_tri.backward()

    # Force PyTorch fallback via float32 cast
    torch.manual_seed(4242)
    _run_detached_distill_kl(
        q.float(), k.float(), qpos_all, idx_r, ix_ref, ix_ref.active_wq(pass_idx=1), scaling, cur_pass=1
    )
    loss_ref = ix_ref.pop_losses()[0]
    loss_ref.backward()

    assert math.isclose(float(loss_tri), float(loss_ref), rel_tol=1e-2, abs_tol=1e-2)
    assert ix_tri.wq.grad is not None and ix_ref.wq.grad is not None
    assert ix_tri.wk.grad is not None and ix_ref.wk.grad is not None
    cos_wq = torch.nn.functional.cosine_similarity(
        ix_tri.wq.grad.float().flatten(), ix_ref.wq.grad.float().flatten(), dim=0
    ).item()
    cos_wk = torch.nn.functional.cosine_similarity(
        ix_tri.wk.grad.float().flatten(), ix_ref.wk.grad.float().flatten(), dim=0
    ).item()
    assert cos_wq > 0.999 and cos_wk > 0.999


# ==============================================================================
# 5. Segmented K/V-Replay BPTT (`bptt.py`): auto offload, loss_scale, gather
# ==============================================================================
def test_bptt_config_auto_offload_and_gather_parity(monkeypatch):
    cfg_auto = BPTTConfig.from_dict({"offload_prior_hidden": "auto", "hbm_headroom_gib": 8.0})
    assert cfg_auto.offload_prior_hidden == "auto"
    assert cfg_auto.resolve_offload_to_cpu(torch.device("cpu"), estimated_bytes=10**9) is False

    # Simulate CUDA with plenty of headroom -> HBM-resident (`False`), vs tight VRAM -> CPU offload (`True`)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _d: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (24 * (1024**3), 96 * (1024**3)))
    assert cfg_auto.resolve_offload_to_cpu(torch.device("cuda:0"), estimated_bytes=4 * (1024**3)) is False
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (6 * (1024**3), 24 * (1024**3)))
    assert cfg_auto.resolve_offload_to_cpu(torch.device("cuda:0"), estimated_bytes=4 * (1024**3)) is True

    with pytest.raises(ValueError):
        BPTTConfig.from_dict({"offload_prior_hidden": "invalid_mode"})

    # `_gather_prior_field` parity across B=1 and B=2 with unequal segment lengths
    g = torch.Generator().manual_seed(19)
    for bsz in (1, 2):
        segs = [
            torch.randn(bsz, 7, 12, generator=g),
            torch.randn(bsz, 11, 12, generator=g),
            torch.randn(bsz, 5, 12, generator=g),
        ]
        idx = torch.randint(0, 23, (bsz, 15), generator=g)
        got = _gather_prior_field(segs, idx)
        ref = torch.cat(segs, dim=1).gather(1, idx.unsqueeze(-1).expand(-1, -1, 12))
        torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)

    # CPU fallback for `_offload_prior_record`
    u = torch.randn(1, 8, 16)
    c = torch.randn(1, 8, 4)
    s = torch.randn(1, 8, 4)
    rec = _offload_prior_record(u, c, s, offload_to_cpu=True, copy_stream=None)
    rec.wait_ready()
    torch.testing.assert_close(rec.u, u)


def test_bptt_loss_scale_and_offload_modes_exact_gradient_parity():
    """`loss_scale=1/accum` matches grad-scaling hooks exactly; HBM vs CPU offload produce bit-exact grads."""
    tcfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    def _build_toy_model() -> ToyForCausalLM:
        torch.manual_seed(42)
        m = ToyForCausalLM(tcfg)
        install_hssd(
            m,
            HSSDConfig(
                chunk_size=4,
                window_chunks=2,
                num_pages=4,
                k_pages=2,
                segment_chunks=2,
                checkpoint_segments=False,
            ),
        )
        install_lsa(
            m,
            LSAConfig(
                mode="sparse",
                index_dim=8,
                rotary_dim=2,
                pool=2,
                block_size=8,
                avg_bucket_keys=4,
                max_bucket_keys=4,
                num_select=2,
                window=8,
                sinks=2,
                query_chunk=8,
                distill_weight=0.0,
            ),
        )
        # Freeze q_proj on full-attention layers to also exercise the zero-duplicate-q_proj path!
        for name, p in m.named_parameters():
            if "q_proj" in name:
                p.requires_grad_(False)
        return m

    m_scale = _build_toy_model()
    m_hooks = _build_toy_model()
    m_hbm = _build_toy_model()

    ids = torch.randint(0, 64, (1, 32), generator=torch.Generator().manual_seed(7))
    labels = ids.clone()

    # 1. `loss_scale=0.5` with `offload_prior_hidden=True`
    r_scale = SegmentedKVBPTTRunner(
        m_scale,
        cfg=BPTTConfig(n_segments=2, seg_len=16, replay_k=4, offload_prior_hidden=True, query_sim_chunk=8),
    )
    res_scale = r_scale.run_step(ids, labels=labels, backward=True, loss_scale=0.5)

    # 2. Legacy `scale_hooks` with `loss_scale=1.0`
    hooks = []
    for p in m_hooks.parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(lambda g: g * 0.5))
    try:
        r_hooks = SegmentedKVBPTTRunner(
            m_hooks,
            cfg=BPTTConfig(n_segments=2, seg_len=16, replay_k=4, offload_prior_hidden=True, query_sim_chunk=8),
        )
        res_hooks = r_hooks.run_step(ids, labels=labels, backward=True, loss_scale=1.0)
    finally:
        for h in hooks:
            h.remove()

    # 3. `offload_prior_hidden=False` (HBM-resident) with `loss_scale=0.5`
    r_hbm = SegmentedKVBPTTRunner(
        m_hbm,
        cfg=BPTTConfig(n_segments=2, seg_len=16, replay_k=4, offload_prior_hidden=False, query_sim_chunk=8),
    )
    res_hbm = r_hbm.run_step(ids, labels=labels, backward=True, loss_scale=0.5)

    assert math.isclose(res_scale.total_loss, res_hooks.total_loss, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(res_scale.total_loss, res_hbm.total_loss, rel_tol=1e-6, abs_tol=1e-6)
    assert res_hbm.stats["bptt/offload_to_cpu"] == 0.0
    for (name, p_s), (_, p_h), (_, p_m) in zip(
        m_scale.named_parameters(), m_hooks.named_parameters(), m_hbm.named_parameters()
    ):
        if not p_s.requires_grad:
            continue
        assert (p_s.grad is None) == (p_h.grad is None) == (p_m.grad is None)
        if p_s.grad is not None:
            torch.testing.assert_close(p_s.grad, p_h.grad, rtol=1e-5, atol=1e-6, msg=f"scale vs hooks {name}")
            torch.testing.assert_close(p_s.grad, p_m.grad, rtol=1e-5, atol=1e-6, msg=f"offload vs hbm {name}")


# ==============================================================================
# 6. Async Prefetcher, Decontam & Packer (`trainer.py`, `decontam.py`, `packing.py`)
# ==============================================================================
class _StatefulDummyStream:
    def __init__(self, total: int = 20, fail_at: int | None = None) -> None:
        self.cursor = 0
        self.total = total
        self.fail_at = fail_at

    def __iter__(self) -> _StatefulDummyStream:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.fail_at is not None and self.cursor == self.fail_at:
            raise RuntimeError(f"boom at {self.cursor}")
        if self.cursor >= self.total:
            raise StopIteration
        v = self.cursor
        self.cursor += 1
        t = torch.arange(v * 8, (v + 1) * 8, dtype=torch.long)
        return {"input_ids": t, "labels": t.clone(), "position_ids": torch.arange(8, dtype=torch.long)}

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, sd: dict[str, int]) -> None:
        self.cursor = int(sd["cursor"])


def test_async_prefetcher_exact_deterministic_resume_and_error_propagation():
    # Consume 3 items with maxsize=3 (producer reads ahead!), save state_dict, resume fresh stream
    stream = _StatefulDummyStream(total=12)
    pf = _AsyncBatchPrefetcher(stream, maxsize=3, pin_memory=False)
    try:
        consumed = [pf.next_item() for _ in range(3)]
        sd_at_3 = pf.state_dict()
        assert sd_at_3 == {"cursor": 3}
        # Even though the background stream cursor advanced ahead of 3, saved state is EXACTLY 3!
        pf.close()
        assert pf.state_dict() == {"cursor": 3}
    finally:
        pf.close()

    # Resume a fresh stream from `sd_at_3` -> must yield items 3, 4, 5, 6 bit-identically
    resumed = _StatefulDummyStream(total=12)
    resumed.load_state_dict(sd_at_3)
    pf2 = _AsyncBatchPrefetcher(resumed, maxsize=2, pin_memory=False)
    try:
        tail = [pf2.next_item() for _ in range(4)]
    finally:
        pf2.close()

    ref_stream = _StatefulDummyStream(total=12)
    expected = [next(ref_stream) for _ in range(7)]
    for got, exp in zip(consumed + tail, expected):
        torch.testing.assert_close(got["input_ids"], exp["input_ids"])
        torch.testing.assert_close(got["labels"], exp["labels"])

    # `_to_device` single-item zero-copy squeeze & multi-item stack
    d1 = _to_device([expected[0]], torch.device("cpu"), pass_pos=True)
    assert d1["input_ids"].shape == (1, 8) and d1["position_ids"].shape == (1, 8)
    d2 = _to_device(expected[:2], torch.device("cpu"), pass_pos=False)
    assert d2["input_ids"].shape == (2, 8) and "position_ids" not in d2

    # Clean StopIteration propagation
    pf_end = _AsyncBatchPrefetcher(_StatefulDummyStream(total=2), maxsize=2, pin_memory=False)
    try:
        _ = pf_end.next_item()
        _ = pf_end.next_item()
        with pytest.raises(StopIteration):
            pf_end.next_item()
    finally:
        pf_end.close()

    # Clean exception propagation
    pf_err = _AsyncBatchPrefetcher(_StatefulDummyStream(total=5, fail_at=1), maxsize=2, pin_memory=False)
    try:
        _ = pf_err.next_item()
        with pytest.raises(RuntimeError, match="boom at 1"):
            pf_err.next_item()
    finally:
        pf_err.close()


def test_decontaminator_short_circuit_exact_parity():
    dec = Decontaminator(n=5, min_unique=3, short_min_words=3)
    dec.add_texts("eval_a", ["the quick brown fox jumps over the lazy dog", "alpha beta gamma"])

    cases = [
        "completely unrelated sentence with plenty of unique words here",
        "prefix text before the quick brown fox jumps somewhere inside",
        "0 0 0 0 0 0 0 0 0 0 0 0 0",  # low-diversity below min_unique
        "short text containing alpha beta gamma right in middle",
        "too short",
    ]
    for txt in cases:
        got = dec.is_contaminated(txt)
        # Reference implementation using original `ngram_hashes`
        words = normalize_words(txt)
        ref: tuple[bool, str | None] = (False, None)
        for h in ngram_hashes(words, dec.n, dec.min_unique):
            if h in dec._grams:
                ref = (True, dec._grams[h])
                break
        if not ref[0]:
            for k, tbl in dec._short.items():
                for i in range(len(words) - k + 1):
                    g_val = tbl.get(hash(tuple(words[i : i + k])))
                    if g_val is not None:
                        ref = (True, g_val)
                        break
                if ref[0]:
                    break
        assert got == ref, f"decontam mismatch on {txt!r}: {got} vs {ref}"


def test_packer_vectorized_emit_exact_parity():
    def _ref_pos(docs: list[int]) -> list[int]:
        pos: list[int] = []
        prev: int | None = None
        p = 0
        for d in docs:
            if d == -1:
                pos.append(0)
                continue
            p = p + 1 if d == prev else 0
            pos.append(p)
            prev = d
        return pos

    pk = Packer(seq_len=16, pad_id=0, respect_doc_boundaries=True, drop_no_loss=False)
    rows = []
    rows.extend(pk.add([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], doc_id=10))
    rows.extend(pk.add([6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], doc_id=20))
    rows.extend(pk.add([12, 13, 14, 15, 16, 17, 18], [12, 13, 14, 15, 16, 17, 18], doc_id=30))
    flushed = pk.flush()
    if flushed is not None:
        rows.append(flushed)

    assert len(rows) == 2
    for row in rows:
        docs_list = row["doc_ids"].tolist()
        assert row["position_ids"].tolist() == _ref_pos(docs_list)
        assert row["n_docs"] == len({d for d in docs_list if d != -1})

    # Verify synthetic `-1` doc_id fallback matches too
    pk2 = Packer(seq_len=8, pad_id=0, respect_doc_boundaries=False, drop_no_loss=False)
    r_mid = pk2.add([1, 2], [1, 2], doc_id=1)
    r_mid += pk2.add([3, 4], [3, 4], doc_id=-1)
    r_mid += pk2.add([5, 6, 7, 8], [5, 6, 7, 8], doc_id=1)
    assert len(r_mid) == 1
    assert r_mid[0]["position_ids"].tolist() == _ref_pos(r_mid[0]["doc_ids"].tolist())
    assert r_mid[0]["n_docs"] == 1


# Keep `bptt_mod` imported cleanly for regression checks
def test_bptt_module_exports_chunked_replay_indexer_kl():
    assert callable(bptt_mod._chunked_replay_indexer_kl)
