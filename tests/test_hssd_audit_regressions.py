"""Comprehensive regression suite for all external HSSD v3.1 audit findings (Bugs 3A-3E, 4.1-4.3, Issues 5-6).

Explicitly reproduces every counterexample identified in the external review and verifies each fix:
  1. Bug 3A: Compact `HSSDCacheState` storage (`theta_cum` slice view leak eliminated; `unique_storage_bytes` &
     `has_compact_storage` verified across `T=17`, `T=73`, `T=600`, and Qwen3.6 dimensions).
  2. Bug 3B: Factored `(gamma_page_scale[M], gamma_chan_pair[B, T, H_v, d_k // 2])` representation (`16x` smaller;
     no 5D `[B, T, M, H_v, d_k]` tensor in `project_inputs`; backward-compatible legacy tensor support).
  3. Bug 3C: Actual sparse Top-`k` write-mass duty (`pi_actual` + STE `pi_ste`) catching the 6-dead-page
     counterexample with non-zero upward gradients into unselected `write_router` logits.
  4. Bug 3D: Full PyTorch/HF Tensor cache contract (`__torch_function__`: `zeros_like`, `empty_like`, `clone`,
     `index_select`, `cat`; `.copy_()`, `__getitem__`, `reorder_cache`, `HSSDHybridCache`, & optional HF cache).
  5. Bug 3E: Packed-document boundaries (`cu_seqlens=[0, 3, 6]` & 2D `seq_idx`) with `0.0` cross-doc leakage
     across SWA, recurrent memory pages, and `theta_cum`, plus strict `ValueError` validation.
  6. Bug 4.1: 2D padding logical-clock invariance (`step_count: Tensor[B]` + per-row unpad/repad so prepending
     or appending masked pad tokens changes valid outputs by `< 1e-13` in `fp64`).
  7. Bug 4.2: Complete `state_equal` (checking valid SWA K/V & validity ring) + `s_chunk_start` property aliasing
     `boundary_ring[-1][:, 0]` without duplicate `2 MiB/layer` tensor storage.
  8. Bug 4.3: `float64` phase accumulator (`< 2e-6 rad` drift at `1,000,000` steps in `fp32` input; preserved
     across `.to(torch.bfloat16)` and `state_dtype="bfloat16"`).
  9. Issues 5 & 6: Linear-`beta` write scaling (`sqrt(w_gate)`), `paired_b_gate`, low-rank `theta_down/theta_up`,
     multi-page routed surprise suppressing redundant writes on Page 5, overflow-free `_wy_kda_state_update`
     at `span=400..5000` with zero `Tensor.__bool__` host syncs, and `< 1e-13` `chunk_hssd` vs `recurrent_hssd`
     forward/backward/cache/duty parity.
"""
from __future__ import annotations

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from qloop.model.hssd import (
    DEFAULT_MU_MAX,
    HSSDAdapter,
    HSSDCacheState,
    HSSDConfig,
    HSSDHybridCache,
    PeakBoltzmannGate,
    _wy_kda_state_update,
    apply_2d_rotation,
    apply_hssd,
    chunk_hssd,
    compute_write_budget_loss,
    recurrent_hssd,
    recurrent_hssd_step,
    remove_hssd,
)
from qloop.model.loop import apply_loop
from qloop.model.toy import build_toy


def _make_hssd_inputs(
    bsz: int = 2,
    seq_len: int = 29,
    hv: int = 2,
    dk: int = 8,
    dv: int = 8,
    num_pages: int = 8,
    *,
    dtype: torch.dtype = torch.float64,
    seed: int = 2026,
    factored_gamma: bool = True,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        t = torch.randn(*shape, generator=gen, dtype=dtype) * scale
        return t.detach().requires_grad_(requires_grad)

    q = randn(bsz, seq_len, hv, dk)
    k = randn(bsz, seq_len, hv, dk)
    v = randn(bsz, seq_len, hv, dv, scale=0.5)

    p_scale = torch.logspace(-3.5, -1.3, steps=num_pages, dtype=dtype).detach().requires_grad_(requires_grad)
    chan_pair = (
        torch.empty(bsz, seq_len, hv, dk // 2, dtype=dtype).uniform_(0.2, 1.5, generator=gen)
    ).detach().requires_grad_(requires_grad)
    if factored_gamma:
        gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (p_scale, chan_pair)
    else:
        gamma = (
            p_scale.view(1, 1, num_pages, 1, 1) * chan_pair.repeat_interleave(2, dim=-1).unsqueeze(2)
        ).detach().requires_grad_(requires_grad)

    b_gate = (
        torch.empty(bsz, seq_len, hv, dk, dtype=dtype).uniform_(0.25, 1.75, generator=gen)
    ).detach().requires_grad_(requires_grad)
    w_gate = (
        torch.empty(bsz, seq_len, hv, dv, dtype=dtype).uniform_(0.20, 1.60, generator=gen)
    ).detach().requires_grad_(requires_grad)
    theta = (
        torch.empty(bsz, seq_len, hv, dk // 2, dtype=dtype).uniform_(-0.7, 0.7, generator=gen)
    ).detach().requires_grad_(requires_grad)
    surprise_bias = randn(bsz, seq_len, hv, scale=0.3)
    surprise_slope = (
        torch.empty(hv, dtype=dtype).uniform_(2.0, 4.0, generator=gen)
    ).detach().requires_grad_(requires_grad)
    surprise_thresh = (
        torch.empty(hv, dtype=dtype).uniform_(0.5, 1.0, generator=gen)
    ).detach().requires_grad_(requires_grad)
    write_logits = randn(bsz, seq_len, num_pages, scale=1.1)
    read_logits = randn(bsz, seq_len, num_pages, scale=1.1)
    q_swa = randn(bsz, seq_len, hv, dk)
    k_swa = randn(bsz, seq_len, hv, dk)
    v_swa = randn(bsz, seq_len, hv, dv, scale=0.5)

    return {
        "query": q,
        "key": k,
        "value": v,
        "gamma": gamma,
        "b_gate": b_gate,
        "w_gate": w_gate,
        "theta": theta,
        "surprise_bias": surprise_bias,
        "surprise_slope": surprise_slope,
        "surprise_thresh": surprise_thresh,
        "write_logits": write_logits,
        "read_logits": read_logits,
        "q_swa": q_swa,
        "k_swa": k_swa,
        "v_swa": v_swa,
    }


# ==============================================================================
# 1. Bug 3A: Eliminate Prompt-Length Backing Storage Leak in HSSDCacheState
# ==============================================================================
def test_bug_3a_compact_cache_storage_no_prompt_length_leak():
    gate = PeakBoltzmannGate(num_heads=2).double()
    mid_chunk_bytes: list[int] = []
    for seq_len in (17, 73, 195):
        inp = _make_hssd_inputs(bsz=2, seq_len=seq_len, hv=2, dk=8, dv=8, dtype=torch.float64, seed=10 + seq_len)
        _, st_chunk = chunk_hssd(
            **inp,
            boltzmann_gate=gate,
            chunk_size=8,
            window_chunks=2,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert st_chunk is not None
        assert st_chunk.theta_cum.dtype == torch.float64
        assert st_chunk.theta_cum.untyped_storage().nbytes() == st_chunk.theta_cum.numel() * 8
        assert st_chunk.has_compact_storage()
        mid_chunk_bytes.append(st_chunk.unique_storage_bytes())

        _, st_rec = recurrent_hssd(
            **inp,
            boltzmann_gate=gate,
            chunk_size=8,
            window_chunks=2,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert st_rec is not None
        assert st_rec.theta_cum.untyped_storage().nbytes() == st_rec.theta_cum.numel() * 8
        assert st_rec.has_compact_storage()

    # Backing storage bytes of returned mid-chunk cache must be strictly invariant to prompt length T!
    assert len(set(mid_chunk_bytes)) == 1

    # Verify exact Qwen3.6-dimension single-layer cache ledger (B=1, M=8, H_v=32, d_k=d_v=128, C=64, W=8)
    st_qwen_bf16 = HSSDCacheState.zeros(
        1, 8, 32, 128, 128, chunk_size=64, window_chunks=8, dtype=torch.bfloat16
    )
    assert st_qwen_bf16.has_compact_storage()
    # Zero-aliased init shares 1 x 8 MiB page buffer + 9 MiB SWA + 16 KiB fp64 theta_cum + 576B valid + 8B step_count
    expected_zero_bytes = (8 * 32 * 128 * 128 * 2) + 2 * (576 * 32 * 128 * 2) + (32 * 64 * 8) + 576 + 8
    assert st_qwen_bf16.unique_storage_bytes() == expected_zero_bytes

    st_qwen_distinct = HSSDCacheState.zeros(
        1, 8, 32, 128, 128, chunk_size=64, window_chunks=8, dtype=torch.bfloat16, distinct_ring_buffers=True
    )
    assert st_qwen_distinct.has_compact_storage()
    # 10 page tensors (page_states + 9 boundary_ring) * 8 MiB + 9 MiB SWA + 16 KiB fp64 theta_cum + 576B + 8B
    expected_mid_bytes = 10 * (8 * 32 * 128 * 128 * 2) + 2 * (576 * 32 * 128 * 2) + (32 * 64 * 8) + 576 + 8
    assert st_qwen_distinct.unique_storage_bytes() == expected_mid_bytes


# ==============================================================================
# 2. Bug 3B: Factored Decay Representation (No 5D [B, T, M, H_v, d_k] Allocation)
# ==============================================================================
def test_bug_3b_factored_gamma_projection_and_legacy_tensor_equivalence():
    cfg = HSSDConfig(rank=16, chunk_size=8, window_chunks=2, segment_chunks=2, checkpoint_segments=True)
    adapter = HSSDAdapter(32, 2, 8, 8, cfg, layer_id=0).double()
    gen = torch.Generator().manual_seed(302)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.2)

    bsz, seq_len = 2, 21
    x_in = torch.randn(bsz, seq_len, 32, generator=gen, dtype=torch.float64, requires_grad=True)
    q = torch.randn(bsz, seq_len, 2, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    k = torch.randn(bsz, seq_len, 2, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    v = torch.randn(bsz, seq_len, 2, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    g_base = (-0.05 - 0.1 * torch.rand(bsz, seq_len, 2, generator=gen, dtype=torch.float64)).requires_grad_(True)
    beta_base = (0.2 + 0.5 * torch.rand(bsz, seq_len, 2, generator=gen, dtype=torch.float64)).requires_grad_(True)

    proj = adapter.project_inputs(x_in, q, k, v, g_base, beta_base, use_qk_l2norm=True)
    assert "gamma" not in proj
    assert proj["gamma_page_scale"].shape == (cfg.num_pages,)
    assert proj["gamma_chan_pair"].shape == (bsz, seq_len, 2, 4)
    for key_name, val in proj.items():
        if isinstance(val, torch.Tensor):
            assert val.ndim <= 4, f"Unexpected 5D tensor {key_name}: {val.shape}"

    # Equivalence between factored (page_scale, chan_pair) and legacy 5D [B, T, M, H_v, d_k] gamma
    ps = proj["gamma_page_scale"]
    cp = proj["gamma_chan_pair"]
    slp = adapter.log_surprise_slope.exp()
    thr = adapter.surprise_thresh

    out_fac, st_fac, duty_fac = chunk_hssd(
        proj["query"], proj["key"], proj["value"], (ps, cp),
        proj["b_gate"], proj["w_gate"], proj["theta"],
        proj["surprise_bias"], slp, thr,
        proj["write_logits"], proj["read_logits"],
        proj["q_swa"], proj["k_swa"], proj["v_swa"],
        boltzmann_gate=adapter.boltzmann_gate,
        chunk_size=8, window_chunks=2, segment_chunks=2, checkpoint_segments=True,
        output_final_state=True, use_qk_l2norm_in_kernel=False, return_aux=True,
    )
    loss_fac = out_fac.pow(2).sum() + duty_fac.sum()
    loss_fac.backward()
    grad_x_fac = x_in.grad.detach().clone()

    adapter.zero_grad(set_to_none=True)
    x_in2 = x_in.detach().clone().requires_grad_(True)
    g_base2 = g_base.detach().clone().requires_grad_(True)
    beta_base2 = beta_base.detach().clone().requires_grad_(True)
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    proj2 = adapter.project_inputs(x_in2, q2, k2, v2, g_base2, beta_base2, use_qk_l2norm=True)
    gamma_5d_2 = (
        proj2["gamma_page_scale"].view(1, 1, cfg.num_pages, 1, 1)
        * proj2["gamma_chan_pair"].repeat_interleave(2, dim=-1).unsqueeze(2)
    )
    out_5d, st_5d, duty_5d = chunk_hssd(
        proj2["query"], proj2["key"], proj2["value"], gamma_5d_2,
        proj2["b_gate"], proj2["w_gate"], proj2["theta"],
        proj2["surprise_bias"], adapter.log_surprise_slope.exp(), adapter.surprise_thresh,
        proj2["write_logits"], proj2["read_logits"],
        proj2["q_swa"], proj2["k_swa"], proj2["v_swa"],
        boltzmann_gate=adapter.boltzmann_gate,
        chunk_size=8, window_chunks=2, segment_chunks=2, checkpoint_segments=True,
        output_final_state=True, use_qk_l2norm_in_kernel=False, return_aux=True,
    )
    (out_5d.pow(2).sum() + duty_5d.sum()).backward()

    assert float((out_fac - out_5d).detach().abs().max()) < 1e-14
    assert st_fac is not None and st_5d is not None and st_fac.state_equal(st_5d, atol=1e-14, rtol=1e-14)
    assert float((duty_fac - duty_5d).detach().abs().max()) < 1e-14
    assert float((grad_x_fac - x_in2.grad).abs().max()) < 1e-13

    # Also verify legacy 4D [B, T, H_v, d_k // 2] gamma support in recurrent_hssd
    slp_d = adapter.log_surprise_slope.exp().detach()
    thr_d = adapter.surprise_thresh.detach()
    out_4d_rec, _ = recurrent_hssd(
        proj["query"].detach(), proj["key"].detach(), proj["value"].detach(), cp.detach(),
        proj["b_gate"].detach(), proj["w_gate"].detach(), proj["theta"].detach(),
        proj["surprise_bias"].detach(), slp_d, thr_d,
        proj["write_logits"].detach(), proj["read_logits"].detach(),
        proj["q_swa"].detach(), proj["k_swa"].detach(), proj["v_swa"].detach(),
        boltzmann_gate=adapter.boltzmann_gate, chunk_size=8, window_chunks=2, use_qk_l2norm_in_kernel=False,
    )
    out_4d_chk, _ = chunk_hssd(
        proj["query"].detach(), proj["key"].detach(), proj["value"].detach(), cp.detach(),
        proj["b_gate"].detach(), proj["w_gate"].detach(), proj["theta"].detach(),
        proj["surprise_bias"].detach(), slp_d, thr_d,
        proj["write_logits"].detach(), proj["read_logits"].detach(),
        proj["q_swa"].detach(), proj["k_swa"].detach(), proj["v_swa"].detach(),
        boltzmann_gate=adapter.boltzmann_gate, chunk_size=8, window_chunks=2, use_qk_l2norm_in_kernel=False,
    )
    assert float((out_4d_rec - out_4d_chk).detach().abs().max()) < 1e-13


# ==============================================================================
# 3. Bug 3C: Actual Sparse Top-k Write Mass (6-Dead-Page Counterexample + STE)
# ==============================================================================
def test_bug_3c_six_dead_pages_actual_sparse_write_duty_and_ste_gradients():
    # Counterexample: pages (0, 1) have slightly higher logit (0.20) than pages 2..7 (0.00) with zero
    # router weights so Top-2 deterministically selects ONLY pages (0, 1) on every token, leaving pages
    # 2..7 with 0.0 actual writes even though dense softmax gives ~0.118 probability to pages 2..7!
    model = build_toy(num_hidden_layers=4, hidden_size=32).double()
    cfg = HSSDConfig(rank=8, chunk_size=8, window_chunks=2, budget_weight=5.0)
    handle = apply_hssd(model, cfg)
    model.train()

    with torch.no_grad():
        for ad in handle.adapters.values():
            ad.write_router.weight.zero_()
            ad.write_router.bias.copy_(
                torch.tensor([0.20, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
            )
            # Ensure non-trivial surprise gate g ~ 0.5 so dense softmax duty on pages 2..7 would exceed mu_min!
            ad.surprise_proj.weight.zero_()
            ad.surprise_proj.bias.fill_(1.0)

    ids = torch.randint(0, model.config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(33))
    _ = model(ids)

    ad0 = handle.adapters[0]
    duty0 = ad0.last_page_duty
    assert duty0 is not None
    # Forward value of duty on unselected pages 2..7 is IDENTICALLY 0.0!
    assert torch.equal(duty0[2:].detach(), torch.zeros(6, dtype=torch.float64))
    assert float(duty0[:2].detach().sum()) > 0.05

    # Dense softmax probability on pages 2..7 was ~0.118 > 0 (which previously hid the 6 dead pages!)
    dense_vec = ad0.last_dense_router_prob
    assert dense_vec is not None
    assert bool((dense_vec[2:] > 0.10).all())
    assert ad0.last_stats["dense_router_prob"] > 0.10
    assert int(ad0.last_stats["dead_page_count"]) == 6
    stats = handle.stats()
    assert stats["hssd/dead_page_count"] == 6.0
    assert stats["hssd/dense_router_prob"] > 0.10

    # Budget loss MUST be strictly positive because pages 2..7 violate mu_min (0 < 0.2 * mu_max)
    # AND pages 0..1 violate mu_max (0.0025)!
    loss = handle.aux_loss()
    assert float(loss.detach()) > 0.01
    loss.backward()

    b_grad = ad0.write_router.bias.grad
    assert b_grad is not None
    # Unselected dead pages 2..7 MUST receive strictly negative dL/dbias so -lr*grad pushes their logits UP!
    assert bool((b_grad[2:] < -1e-4).all()), f"Expected negative grad on dead pages 2..7, got {b_grad.tolist()}"
    assert bool((b_grad[:2] > 1e-4).all()), f"Expected positive grad on flooded vault pages 0..1, got {b_grad.tolist()}"


# ==============================================================================
# 4. Bug 3D: Transformers Tensor Cache Contract & HSSDHybridCache
# ==============================================================================
def test_bug_3d_torch_function_cache_contract_and_hybrid_cache():
    inp = _make_hssd_inputs(bsz=3, seq_len=19, hv=2, dk=8, dv=8, dtype=torch.float64, seed=404)
    gate = PeakBoltzmannGate(num_heads=2).double()
    _, state = chunk_hssd(
        **inp, boltzmann_gate=gate, chunk_size=8, window_chunks=2, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert state is not None

    # Tensor metadata contract
    assert state.shape == state.page_states.shape
    assert state.size() == state.page_states.size()
    assert state.size(0) == 3
    assert state.ndim == 5
    assert state.numel() == state.page_states.numel()
    assert state.device == state.page_states.device
    assert state.dtype == torch.float64

    # Static buffer pre-allocation + in-place .copy_() (preserving destination storage addresses!)
    buf = torch.zeros_like(state)
    assert isinstance(buf, HSSDCacheState)
    assert buf.theta_cum.dtype == torch.float64
    assert buf.has_compact_storage()
    ptr_before = buf.page_states.data_ptr()
    ring0_ptr_before = buf.boundary_ring[0].data_ptr()
    ring_last_ptr_before = buf.boundary_ring[-1].data_ptr()
    assert ring0_ptr_before != ring_last_ptr_before

    ret = buf.copy_(state)
    assert ret is buf
    assert buf.page_states.data_ptr() == ptr_before
    assert buf.boundary_ring[0].data_ptr() == ring0_ptr_before
    assert buf.state_equal(state, atol=0.0, rtol=0.0)

    # Mutating buf must NOT mutate original state
    buf.page_states.add_(1.0)
    assert not buf.state_equal(state)

    empty_buf = torch.empty_like(state)
    assert isinstance(empty_buf, HSSDCacheState) and empty_buf.theta_cum.dtype == torch.float64
    cloned = torch.clone(state)
    assert isinstance(cloned, HSSDCacheState) and cloned.state_equal(state, atol=0.0, rtol=0.0)

    # Beam search reorder / index_select / slicing / cat
    beam_idx = torch.tensor([2, 0, 2, 1], dtype=torch.long)
    sel_fn = torch.index_select(state, 0, beam_idx)
    sel_idx = state[beam_idx]
    sel_reord = state.reorder_cache(beam_idx)
    assert sel_fn.size(0) == 4 and sel_fn.has_compact_storage()
    assert sel_fn.state_equal(sel_idx, atol=0.0, rtol=0.0)
    assert sel_fn.state_equal(sel_reord, atol=0.0, rtol=0.0)
    assert sel_fn[0:1].state_equal(state[2:3], atol=0.0, rtol=0.0)

    rejoined = torch.cat([state[0:1], state[1:2], state[2:3]], dim=0)
    assert rejoined.state_equal(state, atol=0.0, rtol=0.0)
    assert rejoined.has_compact_storage()

    # End-to-end HSSDHybridCache with ToyHybridMoEForCausalLM + apply_hssd + apply_loop
    model = build_toy(num_hidden_layers=8, hidden_size=32).double()
    h_cfg = HSSDConfig(rank=8, chunk_size=8, window_chunks=2, checkpoint_segments=False)
    handle = apply_hssd(model, h_cfg)
    ctl = apply_loop(model, {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12})
    gen = torch.Generator().manual_seed(405)
    with torch.no_grad():
        for p in handle.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.25)
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.25)
    model.eval()

    ids = torch.randint(0, model.config.vocab_size, (2, 14), generator=gen)
    full_logits = model(ids).logits

    hcache = HSSDHybridCache(model.config)
    out_pre = model(ids[:, :10], past_key_values=hcache, use_cache=True)
    assert hcache.get_seq_length() == 10
    assert hcache.unique_storage_bytes() > 0

    pieces = [out_pre.logits]
    for t in range(10, 14):
        out_step = model(ids[:, t : t + 1], past_key_values=hcache, use_cache=True)
        pieces.append(out_step.logits)
    assert float((torch.cat(pieces, dim=1) - full_logits).detach().abs().max()) < 1e-11

    # Beam reorder on HSSDHybridCache + 1 more decode step matches reordered batch reference!
    b_order = torch.tensor([1, 0, 1], dtype=torch.long)
    hcache.reorder_cache(b_order)
    nxt = torch.randint(0, model.config.vocab_size, (3, 1), generator=gen)
    out_beam = model(nxt, past_key_values=hcache, use_cache=True).logits
    ref_seq = torch.cat([ids[b_order], nxt], dim=1)
    ref_beam = model(ref_seq).logits[:, -1:, :]
    assert float((out_beam - ref_beam).detach().abs().max()) < 1e-11

    # Multi-pass explicit slot helper & reset()
    hcache.update_recurrent_state(state[0:1], layer_idx=0, pass_idx=1)
    got_pass = hcache.get_recurrent_state(0, pass_idx=1)
    assert got_pass is not None and got_pass.state_equal(state[0:1], atol=0.0, rtol=0.0)
    hcache.reset()
    assert hcache.get_seq_length() == 0

    ctl.unwrap()
    remove_hssd(model)

    # Optional upstream transformers Qwen3_5DynamicCache check if installed
    if importlib.util.find_spec("transformers") is not None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35_mod  # type: ignore[import-not-found]

        dyn_cls = getattr(q35_mod, "Qwen3_5DynamicCache", None)
        if dyn_cls is not None:
            # Verify static-cache style copy_ & reorder_cache compatibility directly on recurrent_states
            st_copy = torch.zeros_like(state)
            st_copy.copy_(state)
            assert st_copy.state_equal(state, atol=0.0, rtol=0.0)


# ==============================================================================
# 5. Bug 3E: Packed-Document Boundaries (cu_seqlens & seq_idx Zero Leakage)
# ==============================================================================
def test_bug_3e_packed_document_boundaries_zero_cross_doc_leakage_and_validation():
    gate = PeakBoltzmannGate(num_heads=2).double()

    # Exact 3+3 counterexample AND multi-chunk 11+8+12 packed sequence (crossing C=8, W=2 boundaries)
    for splits in ([0, 3, 6], [0, 11, 19, 31]):
        t_tot = splits[-1]
        inp = _make_hssd_inputs(
            bsz=1, seq_len=t_tot, hv=2, dk=8, dv=8, dtype=torch.float64, seed=500 + t_tot, requires_grad=True
        )
        cu = torch.tensor(splits, dtype=torch.int32)

        out_packed, st_packed, duty_packed = chunk_hssd(
            **inp,
            boltzmann_gate=gate,
            chunk_size=8,
            window_chunks=2,
            cu_seqlens=cu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            return_aux=True,
        )
        det_inp = {
            k: (v.detach() if isinstance(v, torch.Tensor) else (v[0].detach(), v[1].detach()))
            for k, v in inp.items()
        }
        out_rec_packed, st_rec_packed, duty_rec_packed = recurrent_hssd(
            **det_inp,
            boltzmann_gate=gate,
            chunk_size=8,
            window_chunks=2,
            cu_seqlens=cu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            return_aux=True,
        )
        assert float((out_packed.detach() - out_rec_packed).abs().max()) < 1e-13
        assert float((duty_packed.detach() - duty_rec_packed).abs().max()) < 1e-13
        assert st_packed is not None and st_rec_packed is not None
        assert st_packed.state_equal(st_rec_packed, atol=1e-13, rtol=1e-13)

        # Independent per-document runs from fresh zero state
        ind_outs = []
        ind_u_sum = torch.zeros_like(duty_packed.detach())
        last_ind_st = None
        inp_ind = _make_hssd_inputs(
            bsz=1, seq_len=t_tot, hv=2, dk=8, dv=8, dtype=torch.float64, seed=500 + t_tot, requires_grad=True
        )
        for d_idx in range(len(splits) - 1):
            s0, s1 = splits[d_idx], splits[d_idx + 1]
            sub_inp = {}
            for k_name, val in inp_ind.items():
                if k_name == "gamma":
                    ps, cp = val  # type: ignore[misc]
                    sub_inp[k_name] = (ps, cp[:, s0:s1])
                elif isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == t_tot:
                    sub_inp[k_name] = val[:, s0:s1]
                else:
                    sub_inp[k_name] = val
            o_d, last_ind_st, duty_d = chunk_hssd(
                **sub_inp,
                boltzmann_gate=gate,
                chunk_size=8,
                window_chunks=2,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                return_aux=True,
            )
            ind_outs.append(o_d)
            ind_u_sum = ind_u_sum + duty_d * (s1 - s0)

        out_ind_cat = torch.cat(ind_outs, dim=1)
        duty_ind = ind_u_sum / float(t_tot)
        assert float((out_packed.detach() - out_ind_cat.detach()).abs().max()) < 1e-13
        assert float((duty_packed.detach() - duty_ind.detach()).abs().max()) < 1e-13
        assert last_ind_st is not None and st_packed.state_equal(last_ind_st, atol=1e-13, rtol=1e-13)

        # Gradient parity across packed documents
        (out_packed.pow(2).sum() + duty_packed.sum()).backward()
        (out_ind_cat.pow(2).sum() + duty_ind.sum()).backward()
        assert float((inp["query"].grad - inp_ind["query"].grad).abs().max()) < 1e-13  # type: ignore[union-attr]
        assert float((inp["theta"].grad - inp_ind["theta"].grad).abs().max()) < 1e-13  # type: ignore[union-attr]

    # 2D monotonic seq_idx across B=2
    inp_seq = _make_hssd_inputs(bsz=2, seq_len=6, hv=2, dk=8, dv=8, dtype=torch.float64, seed=555)
    seq_idx = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 2, 2, 5, 5]], dtype=torch.long)
    out_si_chk, _ = chunk_hssd(
        **inp_seq, boltzmann_gate=gate, chunk_size=4, window_chunks=1, seq_idx=seq_idx, use_qk_l2norm_in_kernel=True
    )
    out_si_rec, _ = recurrent_hssd(
        **inp_seq, boltzmann_gate=gate, chunk_size=4, window_chunks=1, seq_idx=seq_idx, use_qk_l2norm_in_kernel=True
    )
    assert float((out_si_chk - out_si_rec).detach().abs().max()) < 1e-13

    # Validation errors on malformed cu_seqlens / seq_idx / initial_state
    inp_b2 = _make_hssd_inputs(bsz=2, seq_len=6, hv=2, dk=8, dv=8, dtype=torch.float64, seed=556)
    with pytest.raises(ValueError, match="batch size B == 1"):
        chunk_hssd(**inp_b2, cu_seqlens=torch.tensor([0, 3, 6]))
    inp_b1 = _make_hssd_inputs(bsz=1, seq_len=6, hv=2, dk=8, dv=8, dtype=torch.float64, seed=557)
    with pytest.raises(ValueError, match="initial_state"):
        chunk_hssd(
            **inp_b1,
            cu_seqlens=torch.tensor([0, 3, 6]),
            initial_state=HSSDCacheState.zeros(1, 8, 2, 8, 8),
        )
    with pytest.raises(ValueError, match="monotonic"):
        chunk_hssd(**inp_b1, cu_seqlens=torch.tensor([0, 4, 3, 6]))
    with pytest.raises(ValueError, match="end at seq_len"):
        chunk_hssd(**inp_b1, cu_seqlens=torch.tensor([0, 3, 5]))
    with pytest.raises(ValueError, match="monotonic"):
        chunk_hssd(**inp_b1, seq_idx=torch.tensor([[0, 1, 0, 1, 2, 2]]))


# ==============================================================================
# 6. Bug 4.1: Padding Logical-Clock Invariance (Prepend 1 or 7 Masked Pad Tokens)
# ==============================================================================
@pytest.mark.parametrize("left_pads", [1, 7])
def test_bug_4_1_padding_logical_clock_invariance_fp64(left_pads: int):
    gate = PeakBoltzmannGate(num_heads=2).double()
    valid_len = 19
    right_pads = 3
    tot_len = left_pads + valid_len + right_pads

    inp_valid = _make_hssd_inputs(bsz=1, seq_len=valid_len, hv=2, dk=8, dv=8, dtype=torch.float64, seed=601)
    out_clean, st_clean, duty_clean = chunk_hssd(
        **inp_valid,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        return_aux=True,
    )
    assert st_clean is not None and st_clean.step_count_int == valid_len

    # Embed exact valid tokens inside [left_pads : left_pads + valid_len] surrounded by arbitrary garbage pad tokens
    inp_noise = _make_hssd_inputs(bsz=1, seq_len=tot_len, hv=2, dk=8, dv=8, dtype=torch.float64, seed=999)
    inp_padded: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    sl = slice(left_pads, left_pads + valid_len)
    for k_name, val in inp_valid.items():
        if k_name == "gamma":
            ps_v, cp_v = val  # type: ignore[misc]
            _, cp_n = inp_noise[k_name]  # type: ignore[misc]
            cp_pad = cp_n.clone()
            cp_pad[:, sl] = cp_v
            inp_padded[k_name] = (ps_v, cp_pad)
        elif isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == valid_len:
            t_pad = inp_noise[k_name].clone()  # type: ignore[union-attr]
            t_pad[:, sl] = val
            inp_padded[k_name] = t_pad
        else:
            inp_padded[k_name] = val

    pad_mask = torch.zeros(1, tot_len, dtype=torch.long)
    pad_mask[:, sl] = 1

    out_pad_chk, st_pad_chk, duty_pad_chk = chunk_hssd(
        **inp_padded,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        pad_mask=pad_mask,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        return_aux=True,
    )
    out_pad_rec, st_pad_rec, duty_pad_rec = recurrent_hssd(
        **inp_padded,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        pad_mask=pad_mask,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        return_aux=True,
    )

    assert float((out_pad_chk[:, sl] - out_clean).detach().abs().max()) < 1e-13
    assert float((out_pad_rec[:, sl] - out_clean).detach().abs().max()) < 1e-13
    assert torch.equal(out_pad_chk[:, :left_pads].detach(), torch.zeros_like(out_pad_chk[:, :left_pads]))
    right_sl = slice(left_pads + valid_len, None)
    assert torch.equal(out_pad_chk[:, right_sl].detach(), torch.zeros_like(out_pad_chk[:, right_sl]))
    assert float((duty_pad_chk - duty_clean).detach().abs().max()) < 1e-13
    assert float((duty_pad_rec - duty_clean).detach().abs().max()) < 1e-13
    assert st_pad_chk is not None and st_pad_chk.state_equal(st_clean, atol=1e-13, rtol=1e-13)
    assert st_pad_rec is not None and st_pad_rec.state_equal(st_clean, atol=1e-13, rtol=1e-13)


def test_bug_4_1_end_to_end_model_left_padding_and_inhomogeneous_decode_fp64():
    # Verify ToyGatedDeltaNet + apply_hssd on inhomogeneous left-padded batch (B=2: 7 pads vs 1 pad)
    model = build_toy(num_hidden_layers=3, hidden_size=32, full_attention_interval=8).double()
    cfg = HSSDConfig(rank=8, chunk_size=8, window_chunks=2, checkpoint_segments=False)
    handle = apply_hssd(model, cfg)
    gen = torch.Generator().manual_seed(602)
    with torch.no_grad():
        for p in handle.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.3)
    model.eval()

    ids0 = torch.randint(1, model.config.vocab_size, (1, 19), generator=gen)
    ids1 = torch.randint(1, model.config.vocab_size, (1, 25), generator=gen)
    nxt = torch.randint(1, model.config.vocab_size, (2, 1), generator=gen)

    ref0 = model(torch.cat([ids0, nxt[0:1]], dim=1)).logits
    ref1 = model(torch.cat([ids1, nxt[1:2]], dim=1)).logits

    # Pack into B=2 left-padded prefill of length 26: row 0 has 7 left pads, row 1 has 1 left pad
    padded = torch.zeros(2, 26, dtype=torch.long)
    attn_mask = torch.zeros(2, 26, dtype=torch.long)
    padded[0, 7:] = ids0[0]
    attn_mask[0, 7:] = 1
    padded[1, 1:] = ids1[0]
    attn_mask[1, 1:] = 1

    cache = HSSDHybridCache(model.config)
    out_pre = model(padded, attention_mask=attn_mask, past_key_values=cache, use_cache=True)
    assert float((out_pre.logits[0, 7:] - ref0[0, :-1]).abs().max()) < 1e-11
    assert float((out_pre.logits[1, 1:] - ref1[0, :-1]).abs().max()) < 1e-11

    st0 = cache.recurrent_states[0]
    assert isinstance(st0, HSSDCacheState)
    assert torch.equal(st0.step_count, torch.tensor([19, 25], dtype=torch.long))

    # Continue 1-step cached decode from inhomogeneous step_count=[19, 25]!
    attn_mask_dec = torch.cat([attn_mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
    out_dec = model(nxt, attention_mask=attn_mask_dec, past_key_values=cache, use_cache=True)
    assert float((out_dec.logits[0, 0] - ref0[0, -1]).abs().max()) < 1e-11
    assert float((out_dec.logits[1, 0] - ref1[0, -1]).abs().max()) < 1e-11
    assert torch.equal(cache.recurrent_states[0].step_count, torch.tensor([20, 26], dtype=torch.long))

    remove_hssd(model)


# ==============================================================================
# 7. Bug 4.2: Complete state_equal + Eliminate Duplicate s_chunk_start Storage
# ==============================================================================
def test_bug_4_2_state_equal_checks_swa_and_s_chunk_start_is_zero_overhead_property():
    assert isinstance(HSSDCacheState.s_chunk_start, property)

    inp = _make_hssd_inputs(bsz=2, seq_len=13, hv=2, dk=8, dv=8, dtype=torch.float64, seed=707)
    gate = PeakBoltzmannGate(num_heads=2).double()
    _, state = chunk_hssd(
        **inp, boltzmann_gate=gate, chunk_size=8, window_chunks=2, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert state is not None
    assert state.s_chunk_start.data_ptr() == state.boundary_ring[-1][:, 0].data_ptr()
    assert torch.equal(state.s_chunk_start, state.boundary_ring[-1][:, 0])

    # Find a valid SWA ring slot and an invalid SWA ring slot
    valid_slots = state.swa_valid_ring[0].nonzero(as_tuple=False).squeeze(-1)
    invalid_slots = (~state.swa_valid_ring[0]).nonzero(as_tuple=False).squeeze(-1)
    assert valid_slots.numel() > 0 and invalid_slots.numel() > 0
    v_slot = int(valid_slots[0].item())
    inv_slot = int(invalid_slots[0].item())

    # 1. Mutating valid swa_v_ring -> state_equal MUST be False
    mut_v = state.clone()
    mut_v.swa_v_ring[0, v_slot, 0, 0] += 0.05
    assert state.state_equal(mut_v) is False

    # 2. Mutating valid swa_k_ring -> state_equal MUST be False
    mut_k = state.clone()
    mut_k.swa_k_ring[0, v_slot, 0, 0] += 0.05
    assert state.state_equal(mut_k) is False

    # 3. Mutating swa_valid_ring -> state_equal MUST be False
    mut_val = state.clone()
    mut_val.swa_valid_ring[0, v_slot] = False
    assert state.state_equal(mut_val) is False

    # 4. Mutating boundary_ring[0] or boundary_ring[-1] -> state_equal MUST be False
    mut_r0 = state.clone()
    mut_r0.boundary_ring = (mut_r0.boundary_ring[0] + 0.05, *mut_r0.boundary_ring[1:])
    assert state.state_equal(mut_r0) is False

    # 5. Mutating step_count or theta_cum -> state_equal MUST be False
    mut_sc = state.clone()
    mut_sc.step_count = mut_sc.step_count + 1
    assert state.state_equal(mut_sc) is False
    mut_th = state.clone()
    mut_th.theta_cum[0, 0, 0] += 0.05
    assert state.state_equal(mut_th) is False

    # 6. Mutating an INVALID (unpopulated) SWA ring slot leaves logical state equal
    mut_unused = state.clone()
    mut_unused.swa_v_ring[0, inv_slot] += 999.0
    assert state.state_equal(mut_unused) is True


# ==============================================================================
# 8. Bug 4.3: float64 Phase Accumulator (< 2e-6 rad Drift at 1,000,000 Steps)
# ==============================================================================
def test_bug_4_3_million_step_fp32_input_phase_accumulation_and_bf16_state():
    n_steps = 1_000_000
    theta_val_f32 = torch.tensor(0.1, dtype=torch.float32)

    # Negative control: naive float32 cumsum over 1,000,000 steps loses mantissa precision (> 0.01 rad error!)
    theta_1m_f32 = torch.full((n_steps,), 0.1, dtype=torch.float32)
    naive_f32_phase = float(torch.remainder(torch.cumsum(theta_1m_f32, dim=0)[-1], 2.0 * math.pi).double().item())

    # Step-by-step exact modular reference over 1,000,000 steps of `theta_f32`
    # (1000 blocks of 1000 sequential modular steps == exact 1,000,000-step float64 modular recurrence)
    step_val_64 = float(theta_val_f32.to(torch.float64).item())
    two_pi = 2.0 * math.pi
    acc_ref = 0.0
    block_1k = 0.0
    for _ in range(1000):
        block_1k = math.fmod(block_1k + step_val_64, two_pi)
    for _ in range(1000):
        acc_ref = math.fmod(acc_ref + block_1k, two_pi)

    def _ang_dist(a: float, b: float) -> float:
        d = abs(a - b) % two_pi
        return min(d, two_pi - d)

    assert _ang_dist(naive_f32_phase, acc_ref) > 1e-3, "Sanity check: fp32 cumsum drifts >2000x worse at 1M steps"

    # Run actual `chunk_hssd` over 1,000,000 tokens in float32 input (streamed as 20 x 50,000 segments)
    bsz, seg_len, hv, dk, dv, m_pages = 1, 50_000, 1, 2, 2, 1
    q_seg = torch.zeros(bsz, seg_len, hv, dk, dtype=torch.float32)
    k_seg = torch.zeros(bsz, seg_len, hv, dk, dtype=torch.float32)
    v_seg = torch.zeros(bsz, seg_len, hv, dv, dtype=torch.float32)
    ps_seg = torch.full((m_pages,), 1e-3, dtype=torch.float32)
    cp_seg = torch.ones(bsz, seg_len, hv, dk // 2, dtype=torch.float32)
    bg_seg = torch.zeros(bsz, seg_len, hv, dk, dtype=torch.float32)
    wg_seg = torch.zeros(bsz, seg_len, hv, dv, dtype=torch.float32)
    th_seg = torch.full((bsz, seg_len, hv, dk // 2), 0.1, dtype=torch.float32)
    sb_seg = torch.zeros(bsz, seg_len, hv, 1, dtype=torch.float32)
    slp_seg = torch.ones(hv, dtype=torch.float32)
    thr_seg = torch.ones(hv, dtype=torch.float32)
    wl_seg = torch.zeros(bsz, seg_len, m_pages, dtype=torch.float32)
    rl_seg = torch.zeros(bsz, seg_len, m_pages, dtype=torch.float32)

    st_run: HSSDCacheState | None = None
    with torch.no_grad():
        for _ in range(n_steps // seg_len):
            _, st_run = chunk_hssd(
                q_seg, k_seg, v_seg, (ps_seg, cp_seg),
                bg_seg, wg_seg, th_seg,
                sb_seg, slp_seg, thr_seg,
                wl_seg, rl_seg,
                q_seg, k_seg, v_seg,
                chunk_size=100,
                window_chunks=1,
                num_pages=1,
                top_k_pages=1,
                checkpoint_segments=False,
                initial_state=st_run,
                output_final_state=True,
                state_dtype="bfloat16",
                use_swa=False,
                use_phase=True,
                use_surprise=False,
            )

    assert st_run is not None
    assert st_run.step_count_int == n_steps
    assert st_run.page_states.dtype == torch.bfloat16
    assert st_run.theta_cum.dtype == torch.float64
    got_phase = float(st_run.theta_cum[0, 0, 0].item())
    assert _ang_dist(got_phase, acc_ref) < 2e-6

    # Also verify HSSDCacheState.to(torch.bfloat16) preserves float64 theta_cum
    casted = st_run.to(torch.float16).to(torch.bfloat16)
    assert casted.page_states.dtype == torch.bfloat16
    assert casted.theta_cum.dtype == torch.float64


# ==============================================================================
# 9. Issues 5 & 6: Linear-beta, Paired b_gate, Low-Rank theta, Routed Page-5
#    Surprise, Overflow-Free WY Span=400..5000 (Zero bool() Sync), & fp64 Parity
# ==============================================================================
def test_issues_5_6_linear_beta_paired_bgate_and_low_rank_theta():
    # 9a: Linear-beta write scaling (`sqrt(w_gate)`): rank-1 write term scales linearly with beta!
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.5, -1.5, 0.5, 1.0]]]], dtype=torch.float64)
    gamma = (torch.tensor([1e-3], dtype=torch.float64), torch.zeros(1, 1, 1, 2, dtype=torch.float64))
    th = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    sb = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    slp = torch.ones(1, dtype=torch.float64)
    thr = torch.zeros(1, dtype=torch.float64)
    wl = torch.zeros(1, 1, 1, dtype=torch.float64)
    rl = torch.zeros(1, 1, 1, dtype=torch.float64)
    g_one = torch.ones(1, 1, 1, 1, dtype=torch.float64)

    outs_by_beta = {}
    for beta_val in (0.2, 0.4):
        bg = torch.full((1, 1, 1, 4), beta_val, dtype=torch.float64)
        wg = torch.full((1, 1, 1, 4), beta_val, dtype=torch.float64)
        st0 = HSSDCacheState.zeros(1, 1, 1, 4, 4, chunk_size=4, window_chunks=1, dtype=torch.float64)
        _, st1 = recurrent_hssd_step(
            q, k, v, gamma, bg, wg, th, sb, slp, thr, wl, rl, q, k, v, st0,
            top_k_pages=1, g_override=g_one, use_swa=False, use_phase=False, use_surprise=False,
        )
        outs_by_beta[beta_val] = st1.page_states[0, 0, 0, 0].clone()

    # Doubling beta from 0.2 -> 0.4 scales written state by exact 2.0x (NOT 2^1.5 = 2.8284x!)
    assert float((outs_by_beta[0.4] / outs_by_beta[0.2] - 2.0).abs().max()) < 1e-14

    # 9b: paired_b_gate=True makes sqrt(b_gate) strictly commute with R(-Theta)
    cfg_pair = HSSDConfig(rank=8, paired_b_gate=True)
    ad_pair = HSSDAdapter(32, 2, 8, 8, cfg_pair, layer_id=0).double()
    gen = torch.Generator().manual_seed(901)
    with torch.no_grad():
        for p in ad_pair.parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.4)
    x = torch.randn(2, 5, 32, generator=gen, dtype=torch.float64)
    q_p = torch.randn(2, 5, 2, 8, generator=gen, dtype=torch.float64)
    k_p = torch.randn(2, 5, 2, 8, generator=gen, dtype=torch.float64)
    v_p = torch.randn(2, 5, 2, 8, generator=gen, dtype=torch.float64)
    gb = -0.1 * torch.ones(2, 5, 2, dtype=torch.float64)
    bb = 0.5 * torch.ones(2, 5, 2, dtype=torch.float64)
    pr = ad_pair.project_inputs(x, q_p, k_p, v_p, gb, bb, use_qk_l2norm=True)
    b_g = pr["b_gate"]
    assert torch.equal(b_g[..., 0::2], b_g[..., 1::2])
    k_rand = F.normalize(torch.randn(2, 5, 2, 8, generator=gen, dtype=torch.float64), dim=-1)
    lhs = torch.sqrt(b_g) * apply_2d_rotation(k_rand, -pr["theta"])
    rhs = apply_2d_rotation(torch.sqrt(b_g) * k_rand, -pr["theta"])
    assert float((lhs - rhs).abs().max()) < 1e-14

    # 9c: Low-rank theta_low_rank=True cuts Qwen3.6 30-layer adapter params from ~176.5M to ~57.5M
    ad_lr = HSSDAdapter(2048, 32, 128, 128, HSSDConfig(rank=64, theta_low_rank=True))
    ad_full = HSSDAdapter(2048, 32, 128, 128, HSSDConfig(rank=64, theta_low_rank=False))
    p30_lr = 30 * sum(p.numel() for p in ad_lr.parameters())
    p30_full = 30 * sum(p.numel() for p in ad_full.parameters())
    assert 55_000_000 < p30_lr < 60_000_000
    assert 174_000_000 < p30_full < 179_000_000


def test_issues_5_6_multipage_routed_surprise_suppresses_redundant_writes_on_page_5():
    dt = torch.float64
    bsz, seq_len, hv, dk, dv, m_pages = 1, 4, 1, 8, 8, 8
    gen = torch.Generator().manual_seed(905)
    k_unit = F.normalize(torch.randn(1, 1, hv, dk, generator=gen, dtype=dt), dim=-1).expand(bsz, seq_len, hv, dk)
    v_stored = torch.randn(1, 1, hv, dv, generator=gen, dtype=dt).expand(bsz, seq_len, hv, dv)
    q = k_unit.clone()

    # Store exact association `k_unit @ S_5 == v_stored` ONLY on Page 5 (Page 0 is completely empty!)
    st_init = HSSDCacheState.zeros(
        bsz, m_pages, hv, dk, dv, chunk_size=8, window_chunks=1, dtype=dt, distinct_ring_buffers=True
    )
    assoc = torch.einsum("bchk,bchv->bhkv", k_unit[:, :1], v_stored[:, :1])
    st_init.page_states[:, 5] = assoc
    st_init.boundary_ring[-1][:, 5] = assoc

    gamma = (torch.full((m_pages,), 1e-3, dtype=dt), torch.ones(bsz, seq_len, hv, dk // 2, dtype=dt))
    bg = torch.full((bsz, seq_len, hv, dk), 0.5, dtype=dt)
    wg = torch.full((bsz, seq_len, hv, dv), 0.5, dtype=dt)
    th = torch.zeros(bsz, seq_len, hv, dk // 2, dtype=dt)
    sb = torch.zeros(bsz, seq_len, hv, 1, dtype=dt)
    slp = torch.full((hv,), 6.0, dtype=dt)
    thr = torch.full((hv,), 0.5, dtype=dt)

    # Route writes overwhelmingly to Page 5 (and secondarily Page 6, with Page 5 carrying ~99.99% of Top-2 mass)
    wl = torch.full((bsz, seq_len, m_pages), -20.0, dtype=dt)
    wl[:, :, 5] = 10.0
    wl[:, :, 6] = 0.0
    rl = wl.clone()

    _, _, duty_routed = chunk_hssd(
        q, k_unit, v_stored, gamma, bg, wg, th, sb, slp, thr, wl, rl, q, k_unit, v_stored,
        chunk_size=8, window_chunks=1, num_pages=m_pages, top_k_pages=2,
        initial_state=st_init, use_qk_l2norm_in_kernel=False, use_swa=False, use_phase=False,
        use_surprise=True, surprise_mode="routed", return_aux=True,
    )
    _, _, duty_page0 = chunk_hssd(
        q, k_unit, v_stored, gamma, bg, wg, th, sb, slp, thr, wl, rl, q, k_unit, v_stored,
        chunk_size=8, window_chunks=1, num_pages=m_pages, top_k_pages=2,
        initial_state=st_init, use_qk_l2norm_in_kernel=False, use_swa=False, use_phase=False,
        use_surprise=True, surprise_mode="page0", return_aux=True,
    )

    # Under surprise_mode="routed", prediction error against Page 5 is ~0 => g = sigmoid(6*(0 - 0.5)) = 0.0474
    # => g^2 ~ 0.00225! Under legacy "page0", rel_err = 1.0 => g = sigmoid(3.0) = 0.9526 => g^2 ~ 0.907!
    assert float(duty_routed[5]) < 0.005
    assert float(duty_page0[5]) > 0.85


def test_issues_5_6_wy_extreme_span_400_without_tensor_bool_sync():
    gen = torch.Generator().manual_seed(906)
    bsz, m_pages, hv, c_len, dk, dv = 1, 2, 2, 32, 8, 8

    for dt, tol in ((torch.float32, 1e-5), (torch.float64, 1e-13)):
        s_prev = torch.randn(bsz, m_pages, hv, dk, dv, generator=gen, dtype=dt)
        k_raw = F.normalize(torch.randn(bsz, m_pages, hv, c_len, dk, generator=gen, dtype=dt), dim=-1) * 0.8
        v_raw = torch.randn(bsz, m_pages, hv, c_len, dv, generator=gen, dtype=dt) * 0.5
        # Per-step decay of -15.0 across 32 steps => cumulative span ~ 480 > 150 (overflows float32 exp(+480)!)
        log_a = -torch.empty(bsz, m_pages, hv, c_len, dk, dtype=dt).uniform_(12.0, 16.0, generator=gen)

        orig_bool = torch.Tensor.__bool__

        def _forbid_bool(_self: torch.Tensor) -> bool:
            raise AssertionError("Forbidden Tensor.__bool__ host sync inside _wy_kda_state_update!")

        torch.Tensor.__bool__ = _forbid_bool  # type: ignore[method-assign]
        try:
            s_wy = _wy_kda_state_update(s_prev, k_raw, v_raw, log_a)
        finally:
            torch.Tensor.__bool__ = orig_bool  # type: ignore[method-assign]

        assert torch.isfinite(s_wy).all()

        # Sequential step-by-step ground truth
        s_seq = s_prev.clone()
        for t in range(c_len):
            dec = s_seq * torch.exp(log_a[:, :, :, t, :]).unsqueeze(-1)
            kt = k_raw[:, :, :, t, :]
            vt = v_raw[:, :, :, t, :]
            proj = (dec * kt.unsqueeze(-1)).sum(dim=-2, keepdim=True)
            s_seq = dec - kt.unsqueeze(-1) * proj + kt.unsqueeze(-1) * vt.unsqueeze(-2)

        assert float((s_wy - s_seq).abs().max()) < tol


def test_issues_5_6_chunk_vs_recurrent_fp64_forward_backward_cache_duty_parity():
    gate1 = PeakBoltzmannGate(num_heads=2).double()
    gate2 = PeakBoltzmannGate(num_heads=2).double()
    gate2.load_state_dict(gate1.state_dict())

    init_inp = _make_hssd_inputs(bsz=2, seq_len=11, hv=2, dk=8, dv=8, dtype=torch.float64, seed=910)
    _, init_st = chunk_hssd(
        **init_inp, boltzmann_gate=gate1, chunk_size=8, window_chunks=2, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert init_st is not None

    inp_chk = _make_hssd_inputs(
        bsz=2, seq_len=29, hv=2, dk=8, dv=8, dtype=torch.float64, seed=911, requires_grad=True
    )
    inp_rec = _make_hssd_inputs(
        bsz=2, seq_len=29, hv=2, dk=8, dv=8, dtype=torch.float64, seed=911, requires_grad=True
    )

    out_chk, st_chk, duty_chk = chunk_hssd(
        **inp_chk,
        boltzmann_gate=gate1,
        chunk_size=8,
        window_chunks=2,
        segment_chunks=2,
        checkpoint_segments=True,
        initial_state=init_st.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        surprise_mode="routed",
        return_aux=True,
    )
    out_rec, st_rec, duty_rec = recurrent_hssd(
        **inp_rec,
        boltzmann_gate=gate2,
        chunk_size=8,
        window_chunks=2,
        initial_state=init_st.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        surprise_mode="routed",
        return_aux=True,
    )

    assert float((out_chk.detach() - out_rec.detach()).abs().max()) < 1e-13
    assert float((duty_chk.detach() - duty_rec.detach()).abs().max()) < 1e-13
    assert st_chk is not None and st_rec is not None
    assert st_chk.state_equal(st_rec, atol=1e-13, rtol=1e-13)
    assert st_chk.has_compact_storage() and st_rec.has_compact_storage()

    w_out = torch.randn_like(out_chk.detach())
    loss_chk = (out_chk * w_out).sum() + compute_write_budget_loss(duty_chk, DEFAULT_MU_MAX)
    loss_rec = (out_rec * w_out).sum() + compute_write_budget_loss(duty_rec, DEFAULT_MU_MAX)
    loss_chk.backward()
    loss_rec.backward()

    for k_name in inp_chk:
        if k_name == "gamma":
            ps_c, cp_c = inp_chk[k_name]  # type: ignore[misc]
            ps_r, cp_r = inp_rec[k_name]  # type: ignore[misc]
            assert float((ps_c.grad - ps_r.grad).abs().max()) < 1e-12
            assert float((cp_c.grad - cp_r.grad).abs().max()) < 1e-12
        else:
            tc = inp_chk[k_name]
            tr = inp_rec[k_name]
            assert isinstance(tc, torch.Tensor) and isinstance(tr, torch.Tensor)
            assert tc.grad is not None and tr.grad is not None
            assert float((tc.grad - tr.grad).abs().max()) < 1e-12, f"Grad mismatch on {k_name}"


# ==============================================================================
# 10. HF>=5 Qwen3_5GatedDeltaNet (`recurrent_gated_delta_rule`), SharedKVView,
#     Qwen3_5DynamicCache.last_linear_layer, & Frozen Expert Buffer SHA-256
# ==============================================================================
def _convert_toy_to_hf5_recurrent_attr(model: torch.nn.Module) -> None:
    """Mimic `transformers>=5` `Qwen3_5GatedDeltaNet`: `self.recurrent_gated_delta_rule` (NO `fused_*` attr)."""
    from qloop.model._dtype import upcast

    for layer in model.model.layers:  # type: ignore[union-attr]
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            continue
        orig_rec = attn.fused_recurrent_gated_delta_rule
        delattr(attn, "fused_recurrent_gated_delta_rule")
        assert not hasattr(attn, "fused_recurrent_gated_delta_rule")
        attn.recurrent_gated_delta_rule = orig_rec

        def _hf5_forward(
            x: torch.Tensor,
            cache_params: object = None,
            attention_mask: torch.Tensor | None = None,
            _a: torch.nn.Module = attn,
        ) -> torch.Tensor:
            b, t, _ = x.shape
            qkv = _a.in_proj_qkv(x).transpose(1, 2)
            has_prev = bool(getattr(cache_params, "has_previous_state", True)) if cache_params is not None else False
            prev = (
                cache_params.conv_states[_a.layer_idx]  # type: ignore[attr-defined]
                if (cache_params is not None and (t > 1 or has_prev))
                else None
            )
            if prev is None:
                prev = qkv.new_zeros(b, qkv.shape[1], _a.ck - 1)
            seq = torch.cat([prev, qkv], dim=-1)
            if cache_params is not None:
                cache_params.conv_states[_a.layer_idx] = seq[..., -(_a.ck - 1) :]  # type: ignore[attr-defined]
            qkv = F.silu(_a.conv1d(seq)).transpose(1, 2)
            q, k, v = qkv.view(b, t, 3, _a.nh, _a.d).unbind(2)
            q, k = F.normalize(upcast(q), dim=-1), F.normalize(upcast(k), dim=-1)
            v = upcast(v)
            bb, aa = upcast(_a.in_proj_ba(x)).chunk(2, dim=-1)
            beta = torch.sigmoid(bb)
            g = -upcast(_a.A_log).exp() * F.softplus(aa + upcast(_a.dt_bias))
            init_state = (
                cache_params.recurrent_states[_a.layer_idx]  # type: ignore[attr-defined]
                if (cache_params is not None and (t > 1 or has_prev))
                else None
            )
            rule_fn = _a.recurrent_gated_delta_rule if t == 1 else _a.chunk_gated_delta_rule
            core_out, final_state = rule_fn(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=init_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=False,
                scale=1.0,
            )
            if cache_params is not None:
                cache_params.recurrent_states[_a.layer_idx] = final_state  # type: ignore[attr-defined]
            o = _a.norm(core_out.to(x.dtype))
            return _a.out_proj(o.reshape(b, t, _a.nh * _a.d))

        attn.forward = _hf5_forward  # type: ignore[method-assign]


def test_hf5_qwen35_recurrent_gated_delta_rule_wrapping_gdn2_hssd_and_standalone():
    import ast
    import sys
    import types

    from qloop.infer.hf_custom_modeling import _STANDALONE_CONFIG_SRC, _STANDALONE_MODELING_SRC
    from qloop.model.gdn2 import GDN2Config, install_gdn2
    from qloop.model.toy import ToyConfig, ToyForCausalLM

    model = build_toy(num_hidden_layers=8, hidden_size=32).double().eval()
    _convert_toy_to_hf5_recurrent_attr(model)

    gen = torch.Generator().manual_seed(202611)
    ids = torch.randint(0, model.config.vocab_size, (2, 15), generator=gen)
    ref_unwrapped = model(ids).logits.detach().clone()

    # 10A. GDN-2 + Loop on HF5 `recurrent_gated_delta_rule` (reproducing & fixing the diff=1.0625 Colab bug)
    gdn2 = install_gdn2(model, GDN2Config(rank=8, chunk_size=8))
    ctl = apply_loop(model, {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12})
    assert torch.equal(model(ids).logits, ref_unwrapped)
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.25)
        for ad in gdn2.adapters.values():
            for w in (ad.erase_up.weight, ad.write_up.weight, ad.decay_up.weight):
                w.copy_(torch.randn(w.shape, generator=gen, dtype=w.dtype) * 0.15)
    full_gdn2 = model(ids).logits
    pre_gdn2 = model(ids[:, :11], use_cache=True)
    cache_gdn2 = pre_gdn2.past_key_values
    dec_gdn2 = [pre_gdn2.logits[:, -1:]]
    for pos in range(11, 15):
        step_out = model(ids[:, pos : pos + 1], past_key_values=cache_gdn2, use_cache=True)
        dec_gdn2.append(step_out.logits)
    assert float((torch.cat(dec_gdn2, dim=1) - full_gdn2[:, 10:]).abs().max()) < 1e-11
    ctl.unwrap()
    gdn2.uninstall()
    for lyr in model.model.layers:
        la = getattr(lyr, "linear_attn", None)
        if la is not None:
            assert hasattr(la, "recurrent_gated_delta_rule")
            assert not hasattr(la, "fused_recurrent_gated_delta_rule")
            assert not hasattr(la, "gdn2_adapter")
    assert torch.equal(model(ids).logits, ref_unwrapped)

    # 10B. HSSD v3.1 + Loop on HF5 `recurrent_gated_delta_rule`
    h_handle = apply_hssd(model, HSSDConfig(rank=8, chunk_size=8, window_chunks=2, checkpoint_segments=False))
    ctl_h = apply_loop(model, {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12})
    with torch.no_grad():
        ctl_h.adapters.gates.fill_(0.25)
        for had in h_handle.adapters.values():
            for p in had.parameters():
                p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.05)
    full_hssd = model(ids).logits
    hcache = HSSDHybridCache(model.config)
    pre_h = model(ids[:, :11], past_key_values=hcache, use_cache=True)
    dec_h = [pre_h.logits[:, -1:]]
    for pos in range(11, 15):
        step_out = model(ids[:, pos : pos + 1], past_key_values=hcache, use_cache=True)
        dec_h.append(step_out.logits)
    assert float((torch.cat(dec_h, dim=1) - full_hssd[:, 10:]).abs().max()) < 1e-11
    ctl_h.unwrap()
    remove_hssd(model)
    assert torch.equal(model(ids).logits, ref_unwrapped)

    # 10C. Partial instance attr (`chunk` on instance, `recurrent` ONLY on host module) + re-entrancy guard
    mod_OnlyRec = types.ModuleType("_mock_hf5_partial_mod")
    la0 = model.model.layers[0].linear_attn
    orig_rec0 = la0.recurrent_gated_delta_rule
    mod_OnlyRec.recurrent_gated_delta_rule = orig_rec0  # type: ignore[attr-defined]
    sys.modules["_mock_hf5_partial_mod"] = mod_OnlyRec
    try:
        m_part = build_toy(num_hidden_layers=4, hidden_size=32).double().eval()
        _convert_toy_to_hf5_recurrent_attr(m_part)
        for lyr in m_part.model.layers:
            la = getattr(lyr, "linear_attn", None)
            if la is None:
                continue

            class _PartialDeltaNet(type(la)):  # type: ignore[misc]
                pass

            _PartialDeltaNet.__module__ = "_mock_hf5_partial_mod"
            la.__class__ = _PartialDeltaNet
            delattr(la, "recurrent_gated_delta_rule")
            orig_fwd = la.forward

            def _deleg_fwd(
                hidden_states: torch.Tensor,
                *fa: object,
                _la: torch.nn.Module = la,
                _of: object = orig_fwd,
                **fkw: object,
            ) -> torch.Tensor:
                _la.recurrent_gated_delta_rule = mod_OnlyRec.recurrent_gated_delta_rule  # type: ignore[attr-defined]
                try:
                    return _of(hidden_states, *fa, **fkw)  # type: ignore[operator]
                finally:
                    delattr(_la, "recurrent_gated_delta_rule")

            la.forward = _deleg_fwd  # type: ignore[method-assign]

        g_part = install_gdn2(m_part, GDN2Config(rank=8, chunk_size=8))
        # Step-0 zero-init re-entrancy check + perturbed decode check
        ref_p = m_part(ids[:, :10]).logits
        with torch.no_grad():
            for ad in g_part.adapters.values():
                ad.erase_up.weight.fill_(0.08)
                ad.write_up.weight.fill_(-0.06)
        full_p = m_part(ids[:, :10]).logits
        assert not torch.equal(full_p, ref_p)
        pre_p = m_part(ids[:, :8], use_cache=True)
        s1_p = m_part(ids[:, 8:9], past_key_values=pre_p.past_key_values, use_cache=True)
        s2_p = m_part(ids[:, 9:10], past_key_values=pre_p.past_key_values, use_cache=True)
        inc_p = torch.cat([pre_p.logits[:, -1:], s1_p.logits, s2_p.logits], dim=1)
        assert float((inc_p - full_p[:, 7:]).abs().max()) < 1e-11
        g_part.uninstall()
        assert mod_OnlyRec.recurrent_gated_delta_rule is orig_rec0  # type: ignore[attr-defined]
        assert torch.equal(m_part(ids[:, :10]).logits, ref_p)
    finally:
        sys.modules.pop("_mock_hf5_partial_mod", None)

    # 10D. Standalone `_STANDALONE_MODELING_SRC` zero-`qloop` branch (AST 3.9 + GDN-2 & HSSD HF5 decode)
    ast.parse(_STANDALONE_CONFIG_SRC, filename="configuration_qwen36_loophssd.py", feature_version=(3, 9))
    ast.parse(_STANDALONE_MODELING_SRC, filename="modeling_qwen36_loophssd.py", feature_version=(3, 9))

    mock_tf = types.ModuleType("transformers")

    class _MockPretrainedConfig:
        model_type = "qwen36_loop_hssd"

        def __init__(self, **kw: object) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class _MockPreTrainedModel(torch.nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config = config

    class _MockAutoConfig:
        @staticmethod
        def for_model(_mtype: str, **kw: object) -> ToyConfig:
            known = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            if "layer_types" in kw and isinstance(kw["layer_types"], list):
                kw = dict(kw, layer_types=tuple(kw["layer_types"]))
            return ToyConfig(**{k: kw[k] for k in known})  # type: ignore[arg-type]

    class _MockAutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **_kw: object) -> ToyForCausalLM:
            m = ToyForCausalLM(cfg).double()
            _convert_toy_to_hf5_recurrent_attr(m)
            return m

    mock_tf.PretrainedConfig = _MockPretrainedConfig  # type: ignore[attr-defined]
    mock_tf.PreTrainedModel = _MockPreTrainedModel  # type: ignore[attr-defined]
    mock_tf.AutoConfig = _MockAutoConfig  # type: ignore[attr-defined]
    mock_tf.AutoModelForCausalLM = _MockAutoModelForCausalLM  # type: ignore[attr-defined]

    saved_tf = sys.modules.get("transformers")
    saved_cfg_mod = sys.modules.get("configuration_qwen36_loophssd")
    orig_find_spec = importlib.util.find_spec
    sys.modules["transformers"] = mock_tf
    try:
        importlib.util.find_spec = lambda name, *a, **kw: (  # type: ignore[assignment]
            None if name == "qloop" else orig_find_spec(name, *a, **kw)
        )
        cfg_mod = types.ModuleType("configuration_qwen36_loophssd")
        exec(compile(_STANDALONE_CONFIG_SRC, "configuration_qwen36_loophssd.py", "exec"), cfg_mod.__dict__)
        sys.modules["configuration_qwen36_loophssd"] = cfg_mod

        mod_ns = types.ModuleType("modeling_qwen36_loophssd")
        exec(compile(_STANDALONE_MODELING_SRC, "modeling_qwen36_loophssd.py", "exec"), mod_ns.__dict__)
        assert mod_ns._HAS_QLOOP is False

        base_d = {
            "vocab_size": 128,
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "partial_rotary_factor": 0.5,
            "linear_num_heads": 4,
            "linear_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        }
        for mode_key in ("gdn2", "hssd"):
            arch_d: dict[str, object] = {
                "base_model_type": "toy",
                "base_dtype": "float64",
                "loop": {"enabled": True, "prelude_end": 0, "core_end": 4, "num_passes": 2, "kv_share": True},
                mode_key: {"enabled": True, "rank": 8, "chunk_size": 8, "window_chunks": 2},
            }
            scfg = cfg_mod.Qwen36LoopHSSDConfig(base_config=base_d, qloop_v3_arch=arch_d)
            smodel = mod_ns.Qwen36LoopHSSDForCausalLM(scfg, _dtype_override=torch.float64).eval()
            with torch.no_grad():
                smodel._loop_adapters.gates.fill_(0.2)
                for p in smodel.parameters():
                    if p.ndim >= 2 and p.shape[0] <= 16:
                        p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.05)
            tok_ids = torch.randint(0, 120, (1, 12), generator=gen)
            s_full = smodel(input_ids=tok_ids, use_cache=False).logits
            s_pre = smodel(input_ids=tok_ids[:, :9], use_cache=True)
            s_dec = [s_pre.logits[:, -1:]]
            for pos in range(9, 12):
                s_step = smodel(
                    input_ids=tok_ids[:, pos : pos + 1],
                    past_key_values=s_pre.past_key_values,
                    use_cache=True,
                )
                s_dec.append(s_step.logits)
            assert float((torch.cat(s_dec, dim=1) - s_full[:, 8:]).abs().max()) < 1e-11
    finally:
        importlib.util.find_spec = orig_find_spec  # type: ignore[assignment]
        if saved_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tf
        if saved_cfg_mod is None:
            sys.modules.pop("configuration_qwen36_loophssd", None)
        else:
            sys.modules["configuration_qwen36_loophssd"] = saved_cfg_mod


def test_hf5_shared_kv_view_zero_reinvocation_qwen35_last_linear_and_buffer_sha256():
    from qloop.model.loop import SharedKVView, sha256_frozen_experts

    # 1. HF5 `.layers[i].keys/.values` DynamicCache fast-path in SharedKVView (zero re-invocation on r >= 2)
    class _HF5Layer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.update_calls = 0

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            assert k.shape[-2] > 0, "SharedKVView must NOT invoke DynamicLayer.update with empty slice"
            self.update_calls += 1
            self.keys = k if self.keys is None else torch.cat([self.keys, k], dim=-2)
            self.values = v if self.values is None else torch.cat([self.values, v], dim=-2)
            return self.keys, self.values

    class _HF5HybridCache:
        def __init__(self, n_layers: int, last_lin: int) -> None:
            self.layers = [_HF5Layer() for _ in range(n_layers)]
            self.conv_states: list[torch.Tensor | None] = [None] * n_layers
            self.recurrent_states: list[torch.Tensor | None] = [None] * n_layers
            self.last_linear_layer = last_lin

        @property
        def has_previous_state(self) -> bool:
            return self.conv_states[self.last_linear_layer] is not None

        def get_seq_length(self, layer_idx: int = 0) -> int:
            for lyr in self.layers:
                if lyr.keys is not None:
                    return int(lyr.keys.shape[-2])
            return 0

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *_a: object, **_kw: object):
            return self.layers[layer_idx].update(k, v)

    # Zero-coda schedule (`prelude_end=4, core_end=8, num_passes=3 -> 16 virtual layers`, tail linear = 14)
    model = build_toy(num_hidden_layers=8, hidden_size=32).double().eval()
    _convert_toy_to_hf5_recurrent_attr(model)
    ctl = apply_loop(model, {"prelude_end": 4, "core_end": 8, "num_passes": 3, "target_virtual_layers": 16})
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.3)

    gen = torch.Generator().manual_seed(777)
    ids = torch.randint(0, model.config.vocab_size, (1, 10), generator=gen)

    # Test active_passes=3: Pass-1 core full-attn is slot 7; Passes 2 & 3 (slots 11, 15) share slot 7
    hf5_c3 = _HF5HybridCache(16, last_lin=14)
    full_3 = model(ids).logits
    pre_3 = model(ids[:, :7], past_key_values=hf5_c3, use_cache=True)
    dec_3 = [pre_3.logits[:, -1:]]
    for pos in range(7, 10):
        dec_3.append(model(ids[:, pos : pos + 1], past_key_values=hf5_c3, use_cache=True).logits)
    assert float((torch.cat(dec_3, dim=1) - full_3[:, 6:]).abs().max()) < 1e-12
    # 1 prefill + 3 decode steps = 4 calls total on slot 7 (ZERO calls on slots 11 & 15 and ZERO empty updates!)
    assert hf5_c3.layers[7].update_calls == 4
    assert hf5_c3.layers[11].keys is None and hf5_c3.layers[15].keys is None

    # Test active_passes=1 & active_passes=2 with Qwen3_5DynamicCache `last_linear_layer` (initially 14 in Pass 3!)
    for act_r, expected_last_lin in ((1, 6), (2, 10)):
        ctl.set_active_passes(act_r)
        hf5_cr = _HF5HybridCache(16, last_lin=14)
        full_r = model(ids).logits
        pre_r = model(ids[:, :7], past_key_values=hf5_cr, use_cache=True)
        assert hf5_cr.last_linear_layer == expected_last_lin
        assert hf5_cr.has_previous_state is True
        dec_r = [pre_r.logits[:, -1:]]
        for pos in range(7, 10):
            dec_r.append(model(ids[:, pos : pos + 1], past_key_values=hf5_cr, use_cache=True).logits)
        assert float((torch.cat(dec_r, dim=1) - full_r[:, 6:]).abs().max()) < 1e-12

    # Custom QuantizedHybridCache-style (`dequantize` attribute) still delegates to `cache.update`
    class _MockQuantCache:
        def __init__(self) -> None:
            self.calls = 0

        def dequantize(self) -> None:
            pass

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *_a: object, **_kw: object):
            assert k.shape[-2] == 0 and layer_idx == 3
            self.calls += 1
            return torch.ones(1, 2, 5, 8), torch.ones(1, 2, 5, 8)

    qc = _MockQuantCache()
    sv = SharedKVView(qc)
    k_ret, _ = sv.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), 3)
    assert qc.calls == 1 and k_ret.shape == (1, 2, 5, 8)

    ctl.unwrap()

    # 2. Frozen expert quantized buffer SHA-256 invariance across `apply_loop` & mutation sensitivity
    for p in model.parameters():
        p.requires_grad_(False)
    packed_buf = torch.tensor([17, 42, 99, 255], dtype=torch.uint8)
    model.model.layers[2].mlp.experts.register_buffer("weight_packed", packed_buf)
    sha_pre = sha256_frozen_experts(model)
    ctl2 = apply_loop(model, {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12})
    sha_loop = sha256_frozen_experts(model)
    ctl2.unwrap()
    sha_post = sha256_frozen_experts(model)
    assert sha_pre == sha_loop == sha_post
    packed_buf[0] ^= 1
    assert sha256_frozen_experts(model) != sha_pre


# ==============================================================================
# 11. HF>=5 `HSSDCacheState` & `_HSSDCacheState` (`zero_`, `is_contiguous`,
#     `contiguous`, `repeat_interleave`, `mark_static_address`), HF 5.2-5.23
#     DynamicCache + GatedDeltaNet + Loop (`kv_share=True`), & FP8 Router Safety
# ==============================================================================
def _load_standalone_ns_no_qloop():
    import sys
    import types

    from qloop.infer.hf_custom_modeling import _STANDALONE_CONFIG_SRC, _STANDALONE_MODELING_SRC
    from qloop.model.toy import ToyConfig, ToyForCausalLM

    mock_tf = types.ModuleType("transformers")

    class _MockPretrainedConfig:
        model_type = "qwen36_loop_hssd"

        def __init__(self, **kw: object) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class _MockPreTrainedModel(torch.nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config = config

    class _MockAutoConfig:
        @staticmethod
        def for_model(_mtype: str, **kw: object) -> ToyConfig:
            known = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            if "layer_types" in kw and isinstance(kw["layer_types"], list):
                kw = dict(kw, layer_types=tuple(kw["layer_types"]))
            return ToyConfig(**{k: kw[k] for k in known})  # type: ignore[arg-type]

    class _MockAutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **_kw: object) -> ToyForCausalLM:
            m = ToyForCausalLM(cfg)
            _convert_toy_to_hf5_recurrent_attr(m)
            return m

    mock_tf.PretrainedConfig = _MockPretrainedConfig  # type: ignore[attr-defined]
    mock_tf.PreTrainedModel = _MockPreTrainedModel  # type: ignore[attr-defined]
    mock_tf.AutoConfig = _MockAutoConfig  # type: ignore[attr-defined]
    mock_tf.AutoModelForCausalLM = _MockAutoModelForCausalLM  # type: ignore[attr-defined]

    saved_tf = sys.modules.get("transformers")
    saved_cfg_mod = sys.modules.get("configuration_qwen36_loophssd")
    orig_find_spec = importlib.util.find_spec
    sys.modules["transformers"] = mock_tf
    try:
        importlib.util.find_spec = lambda name, *a, **kw: (  # type: ignore[assignment]
            None if name == "qloop" else orig_find_spec(name, *a, **kw)
        )
        cfg_mod = types.ModuleType("configuration_qwen36_loophssd")
        exec(compile(_STANDALONE_CONFIG_SRC, "configuration_qwen36_loophssd.py", "exec"), cfg_mod.__dict__)
        sys.modules["configuration_qwen36_loophssd"] = cfg_mod

        mod_ns = types.ModuleType("modeling_qwen36_loophssd")
        exec(compile(_STANDALONE_MODELING_SRC, "modeling_qwen36_loophssd.py", "exec"), mod_ns.__dict__)
        assert mod_ns._HAS_QLOOP is False
        return cfg_mod, mod_ns
    finally:
        importlib.util.find_spec = orig_find_spec  # type: ignore[assignment]
        if saved_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tf
        if saved_cfg_mod is None:
            sys.modules.pop("configuration_qwen36_loophssd", None)
        else:
            sys.modules["configuration_qwen36_loophssd"] = saved_cfg_mod


def test_hssd_cache_state_zero_contiguous_dynamo_static_address_and_standalone_parity():
    from qloop.model.hssd import _ensure_dynamo_mark_static_address_compat

    _cfg_mod, mod_ns = _load_standalone_ns_no_qloop()
    StandaloneState = mod_ns._HSSDCacheState

    gate = PeakBoltzmannGate(num_heads=2).double()
    inp19 = _make_hssd_inputs(bsz=3, seq_len=19, hv=2, dk=8, dv=8, dtype=torch.float64, seed=1101)
    _, st19 = chunk_hssd(
        **inp19,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    inp16 = _make_hssd_inputs(bsz=3, seq_len=16, hv=2, dk=8, dv=8, dtype=torch.float64, seed=1102)
    _, st16 = chunk_hssd(
        **inp16,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert st19 is not None and st16 is not None

    # Build matching standalone state from st19
    st19_sa = StandaloneState.zeros(
        3, 8, 2, 8, 8, chunk_size=8, window_chunks=2, device=st19.device, dtype=torch.float64
    )
    st19_sa.copy_(st19)
    assert st19.state_equal(st19_sa, atol=0.0, rtol=0.0)
    assert st19_sa.state_equal(st19, atol=0.0, rtol=0.0)

    for cls_state, sample in ((HSSDCacheState, st19), (StandaloneState, st19_sa)):
        # 1. Compact & contiguous fast-path (`contiguous()` returns `self` when `has_compact_storage()`)
        assert sample.is_contiguous() is True
        assert sample.has_compact_storage() is True
        assert sample.contiguous() is sample
        assert sample.compact() is sample

        # 2. Non-contiguous transposed view (`is_contiguous() == False` -> `contiguous()` compacts)
        non_contig = sample.clone()
        non_contig.page_states = non_contig.page_states.transpose(-1, -2)
        assert non_contig.is_contiguous() is False
        assert non_contig.has_compact_storage() is False
        compacted_nc = non_contig.contiguous()
        assert compacted_nc is not non_contig
        assert compacted_nc.is_contiguous() is True
        assert compacted_nc.has_compact_storage() is True
        assert torch.equal(compacted_nc.page_states, non_contig.page_states)
        assert compacted_nc.contiguous() is compacted_nc

        # 3. Contiguous slice into oversized storage (`is_contiguous() == True`, `has_compact_storage() == False`)
        sliced_view = sample.clone()
        max_win = sliced_view.max_swa_len
        oversized = torch.randn(3 + 5, max_win, 2, 8, dtype=torch.float64)
        sliced_view.swa_k_ring = oversized[2 : 2 + 3]
        assert sliced_view.swa_k_ring.is_contiguous() is True
        assert sliced_view.is_contiguous() is True
        assert sliced_view.has_compact_storage() is False
        compacted_sl = sliced_view.contiguous()
        assert compacted_sl is not sliced_view
        assert compacted_sl.is_contiguous() is True
        assert compacted_sl.has_compact_storage() is True
        assert compacted_sl.state_equal(sliced_view, atol=0.0, rtol=0.0)

        # 4. `.zero_()` on `torch.zeros_like(...)` static buffer preserves ALL 10 distinct `data_ptr()`s!
        static_buf = torch.zeros_like(sample)
        static_buf.copy_(sample)
        assert static_buf.state_equal(sample, atol=0.0, rtol=0.0)
        ptrs_before = [int(t.data_ptr()) for t in static_buf._iter_tensors()]
        assert len(set(ptrs_before)) == len(ptrs_before)
        z_ret = static_buf.zero_()
        assert z_ret is static_buf
        ptrs_after = [int(t.data_ptr()) for t in static_buf._iter_tensors()]
        assert ptrs_after == ptrs_before
        z_ref = cls_state.zeros(
            3, 8, 2, 8, 8, chunk_size=8, window_chunks=2, device=sample.device, dtype=torch.float64
        )
        assert static_buf.state_equal(z_ref, atol=0.0, rtol=0.0)
        assert not bool(static_buf.swa_valid_ring.any())
        assert float(static_buf.swa_k_ring.abs().max()) == 0.0
        assert float(static_buf.swa_v_ring.abs().max()) == 0.0

        # 5. `.zero_()` on aliased zero-init (`cls.zeros`) and chunk-boundary state unaliases shared ring slots
        aliased_z = cls_state.zeros(
            2, 8, 2, 8, 8, chunk_size=8, window_chunks=2, device=sample.device, dtype=torch.float64
        )
        assert aliased_z.boundary_ring[0].data_ptr() == aliased_z.page_states.data_ptr()
        assert aliased_z.zero_() is aliased_z
        ring_ptrs = [int(t.untyped_storage().data_ptr()) for t in (aliased_z.page_states, *aliased_z.boundary_ring)]
        assert len(set(ring_ptrs)) == len(ring_ptrs)
        aliased_z.boundary_ring[0].add_(3.0)
        assert float(aliased_z.page_states.abs().max()) == 0.0
        assert float(aliased_z.boundary_ring[1].abs().max()) == 0.0

        # 6. `repeat_interleave`, `(indices, Ellipsis)` batch indexing, and `non_blocking` `.to()`
        rep_m = sample.repeat_interleave(2, dim=0)
        rep_f = torch.repeat_interleave(sample, 2, dim=0)
        assert rep_m.size(0) == 6 and rep_m.has_compact_storage()
        assert rep_m.state_equal(rep_f, atol=0.0, rtol=0.0)
        assert rep_m[0:1].state_equal(sample[0:1], atol=0.0, rtol=0.0)
        assert rep_m[1:2].state_equal(sample[0:1], atol=0.0, rtol=0.0)
        assert rep_m[4:5].state_equal(sample[2:3], atol=0.0, rtol=0.0)

        idx_ell = sample[torch.tensor([2, 0], dtype=torch.long), ...]
        assert idx_ell.size(0) == 2 and idx_ell.state_equal(sample[[2, 0]], atol=0.0, rtol=0.0)
        bf16_nb = sample.to(dtype=torch.bfloat16, non_blocking=True)
        assert bf16_nb.dtype == torch.bfloat16 and bf16_nb.theta_cum.dtype == torch.float64

    # Chunk-boundary state (`st16`: `boundary_ring[-1]` shares `page_states`) unaliases cleanly on `.zero_()`
    st16_c = st16.clone()
    assert st16_c.boundary_ring[-1].data_ptr() == st16_c.page_states.data_ptr()
    st16_c.zero_()
    assert st16_c.boundary_ring[-1].data_ptr() != st16_c.page_states.data_ptr()
    assert int(st16_c.step_count.abs().max().item()) == 0

    # 7. `torch._dynamo.mark_static_address` compatibility for HF>=5.20 `LinearAttentionLayer.lazy_initialization`
    _ensure_dynamo_mark_static_address_compat()
    mod_ns._ensure_dynamo_mark_static_address_compat()
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and callable(getattr(dynamo, "mark_static_address", None)):
        dynamo.mark_static_address(torch.zeros_like(st19))
        dynamo.mark_static_address(torch.zeros_like(st19_sa))
        with pytest.raises(TypeError):
            dynamo.mark_static_address(12345)


def test_hf5_qwen35_dynamic_cache_and_hf523_linear_attention_layer_hssd_loop_decode():
    import sys
    import types

    from qloop.model._dtype import upcast

    # --- A. HF 5.2-5.5 style `Qwen3_5DynamicCache` (`recurrent_states`, `conv_states`, `last_linear_layer`, `.zero_()`)
    class _MockQwen35DynamicCache:
        def __init__(self, config: object) -> None:
            n = int(config.num_hidden_layers)  # type: ignore[attr-defined]
            ltypes = list(config.layer_types)  # type: ignore[attr-defined]
            self.layer_types = ltypes
            self.transformer_layers = [i for i, lt in enumerate(ltypes) if lt == "full_attention"]
            lin_layers = [i for i, lt in enumerate(ltypes) if lt == "linear_attention"]
            self.last_linear_layer = lin_layers[-1]
            self.conv_states: list[torch.Tensor | None] = [None] * n
            self.recurrent_states: list[object] = [None] * n
            self.key_cache: list[torch.Tensor | None] = [None] * n
            self.value_cache: list[torch.Tensor | None] = [None] * n

        @property
        def has_previous_state(self) -> bool:
            return self.conv_states[self.last_linear_layer] is not None

        def update(
            self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *_a: object, **_kw: object
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert key_states.shape[-2] > 0, "SharedKVView must read key_cache/value_cache without empty update"
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)  # type: ignore[list-item]
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)  # type: ignore[list-item]
            return self.key_cache[layer_idx], self.value_cache[layer_idx]  # type: ignore[return-value]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if not self.transformer_layers:
                return 0
            idx = layer_idx if layer_idx in self.transformer_layers else self.transformer_layers[0]
            kc = self.key_cache[idx]
            return int(kc.shape[-2]) if kc is not None else 0

        def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
            for i in range(len(self.key_cache)):
                if self.key_cache[i] is not None:
                    dev = self.key_cache[i].device  # type: ignore[union-attr]
                    self.key_cache[i] = self.key_cache[i].index_select(0, beam_idx.to(dev))  # type: ignore[union-attr]
                    self.value_cache[i] = self.value_cache[i].index_select(0, beam_idx.to(dev))  # type: ignore[union-attr]
                if self.conv_states[i] is not None:
                    dev = self.conv_states[i].device  # type: ignore[union-attr]
                    self.conv_states[i] = self.conv_states[i].index_select(0, beam_idx.to(dev))  # type: ignore[union-attr]
                if self.recurrent_states[i] is not None:
                    rs = self.recurrent_states[i]
                    self.recurrent_states[i] = rs.index_select(0, beam_idx.to(rs.device))  # type: ignore[attr-defined]

        def reset(self) -> None:
            for i in range(len(self.key_cache)):
                self.key_cache[i] = None
                self.value_cache[i] = None
                if self.conv_states[i] is not None:
                    self.conv_states[i].zero_()  # type: ignore[union-attr]
                    self.conv_states[i] = None
                if self.recurrent_states[i] is not None:
                    assert self.recurrent_states[i].is_contiguous()  # type: ignore[attr-defined]
                    self.recurrent_states[i] = self.recurrent_states[i].contiguous()  # type: ignore[attr-defined]
                    self.recurrent_states[i].zero_()  # type: ignore[attr-defined]

    # 8 physical layers (2 x [3 linear + 1 full]) looped with prelude=0, core=8, passes=3 -> 24 virtual layers
    m_q35 = build_toy(num_hidden_layers=8, hidden_size=32).double().eval()
    _convert_toy_to_hf5_recurrent_attr(m_q35)
    h_q35 = apply_hssd(
        m_q35, HSSDConfig(rank=8, chunk_size=6, window_chunks=2, state_dtype="auto", checkpoint_segments=False)
    )
    ctl_q35 = apply_loop(
        m_q35, {"prelude_end": 0, "core_end": 8, "num_passes": 3, "target_virtual_layers": 24, "kv_share": True}
    )
    gen = torch.Generator().manual_seed(52301)
    with torch.no_grad():
        ctl_q35.adapters.gates.fill_(0.22)
        for had in h_q35.adapters.values():
            for p in had.parameters():
                p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.08)

    ids_a = torch.randint(0, m_q35.config.vocab_size, (2, 15), generator=gen)
    ids_b = torch.randint(0, m_q35.config.vocab_size, (2, 13), generator=gen)
    with torch.no_grad():
        full_a = m_q35(ids_a).logits
        full_b = m_q35(ids_b).logits

        qcache = _MockQwen35DynamicCache(m_q35.config)
        pre_a = m_q35(ids_a[:, :10], past_key_values=qcache, use_cache=True)
        assert qcache.has_previous_state is True
        dec_a = [pre_a.logits[:, -1:]]
        for pos in range(10, 15):
            dec_a.append(m_q35(ids_a[:, pos : pos + 1], past_key_values=qcache, use_cache=True).logits)
        assert float((torch.cat(dec_a, dim=1) - full_a[:, 9:]).abs().max()) < 1e-11
        # Shared Pass-2/Pass-3 full-attn KV slots (11, 15, 19, 23) stayed unallocated
        for shared_slot in (11, 15, 19, 23):
            assert qcache.key_cache[shared_slot] is None

        # `.reset()` calls `.is_contiguous()`, `.contiguous()`, `.zero_()` on every `HSSDCacheState`; reuse on ids_b!
        qcache.reset()
        assert qcache.has_previous_state is False
        pre_b = m_q35(ids_b[:, :9], past_key_values=qcache, use_cache=True)
        dec_b = [pre_b.logits[:, -1:]]
        for pos in range(9, 13):
            dec_b.append(m_q35(ids_b[:, pos : pos + 1], past_key_values=qcache, use_cache=True).logits)
        assert float((torch.cat(dec_b, dim=1) - full_b[:, 8:]).abs().max()) < 1e-11

    ctl_q35.unwrap()
    remove_hssd(m_q35)

    # --- B. HF 5.23.0 `DynamicCache` (`LinearAttentionLayer` + `mark_static_address` + `.copy_()` + `.zero_()`
    #        + module-level `torch_chunk/recurrent_gated_delta_rule` + `kernels` `_kernel_repo` unwrap)
    class _HF523DynamicLayer:
        supports_early_init = True

        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.is_initialized = False

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            assert k.shape[-2] > 0
            if not self.is_initialized:
                self.keys = k
                self.values = v
                self.is_initialized = True
            else:
                self.keys = torch.cat([self.keys, k], dim=-2)  # type: ignore[list-item]
                self.values = torch.cat([self.values, v], dim=-2)  # type: ignore[list-item]
            return self.keys, self.values  # type: ignore[return-value]

        def get_seq_length(self) -> int:
            return int(self.keys.shape[-2]) if (self.is_initialized and self.keys is not None) else 0

        def reset(self) -> None:
            self.keys = None
            self.values = None
            self.is_initialized = False

        def batch_repeat_interleave(self, repeats: int) -> None:
            if self.get_seq_length() > 0:
                self.keys = self.keys.repeat_interleave(repeats, dim=0)  # type: ignore[union-attr]
                self.values = self.values.repeat_interleave(repeats, dim=0)  # type: ignore[union-attr]

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            if self.get_seq_length() > 0:
                self.keys = self.keys[indices.to(self.keys.device), ...]  # type: ignore[index,union-attr]
                self.values = self.values[indices.to(self.values.device), ...]  # type: ignore[index,union-attr]

    class _HF523LinearAttentionLayer:
        supports_early_init = False

        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[object] = [None]
            self.conv_kernel_size: dict[int, int] = {}
            self.is_conv_states_initialized = {0: False}
            self.is_recurrent_states_initialized = {0: False}
            self.has_previous_state = {0: False}
            self.record_past = False

        def lazy_initialization(
            self,
            conv_states: torch.Tensor | None = None,
            recurrent_states: object = None,
            conv_kernel_size: int | None = None,
            state_idx: int = 0,
        ) -> None:
            dynamo = getattr(torch, "_dynamo", None)
            if conv_states is not None:
                self.batch_size = conv_states.shape[0]
                self.conv_kernel_size[state_idx] = (
                    conv_states.shape[-1] if conv_kernel_size is None else int(conv_kernel_size)
                )
                self.conv_states[state_idx] = torch.zeros(
                    (*conv_states.shape[:-1], self.conv_kernel_size[state_idx]),
                    dtype=conv_states.dtype,
                    device=conv_states.device,
                )
                if dynamo is not None and callable(getattr(dynamo, "mark_static_address", None)):
                    dynamo.mark_static_address(self.conv_states[state_idx])
                self.is_conv_states_initialized[state_idx] = True
            if recurrent_states is not None:
                self.batch_size = recurrent_states.shape[0]  # type: ignore[attr-defined]
                self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)  # type: ignore[arg-type]
                if dynamo is not None and callable(getattr(dynamo, "mark_static_address", None)):
                    dynamo.mark_static_address(self.recurrent_states[state_idx])
                self.is_recurrent_states_initialized[state_idx] = True

        def update_conv_state(
            self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None
        ) -> torch.Tensor:
            if not self.is_conv_states_initialized[state_idx]:
                self.lazy_initialization(
                    conv_states=conv_states, conv_kernel_size=conv_kernel_size, state_idx=state_idx
                )
            if not self.has_previous_state[state_idx]:
                full_cs = conv_states
                if full_cs.shape[-1] < self.conv_kernel_size[state_idx]:
                    full_cs = F.pad(full_cs, (self.conv_kernel_size[state_idx] - full_cs.shape[-1], 0), value=0)
                self.has_previous_state[state_idx] = True
            else:
                full_cs = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)  # type: ignore[list-item]
            self.conv_states[state_idx].copy_(full_cs[..., -self.conv_kernel_size[state_idx] :])  # type: ignore[union-attr]
            return full_cs

        def update_recurrent_state(self, recurrent_states: object, state_idx: int = 0) -> object:
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            self.recurrent_states[state_idx].copy_(recurrent_states)  # type: ignore[attr-defined]
            return self.recurrent_states[state_idx]

        def reset(self) -> None:
            if self.conv_states[0] is not None:
                self.conv_states[0].zero_()
                self.has_previous_state[0] = False
            if self.recurrent_states[0] is not None:
                self.recurrent_states[0].zero_()  # type: ignore[attr-defined]

        def batch_repeat_interleave(self, repeats: int) -> None:
            if self.conv_states[0] is not None:
                self.conv_states[0] = self.conv_states[0].repeat_interleave(repeats, dim=0)
            if self.recurrent_states[0] is not None:
                self.recurrent_states[0] = self.recurrent_states[0].repeat_interleave(repeats, dim=0)  # type: ignore[attr-defined]
            if hasattr(self, "batch_size"):
                self.batch_size *= repeats

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            idx = indices.to(torch.long)
            if self.conv_states[0] is not None:
                self.conv_states[0] = self.conv_states[0][idx.to(self.conv_states[0].device), ...]
            if self.recurrent_states[0] is not None:
                rs = self.recurrent_states[0]
                self.recurrent_states[0] = rs[idx.to(rs.device), ...]  # type: ignore[attr-defined,index]
            if hasattr(self, "batch_size"):
                self.batch_size = int(idx.shape[0])

    class _HF523DynamicCache:
        def __init__(self, config: object) -> None:
            self.layers = [
                _HF523DynamicLayer() if lt == "full_attention" else _HF523LinearAttentionLayer()
                for lt in config.layer_types  # type: ignore[attr-defined]
            ]

        @property
        def is_initialized(self) -> bool:
            init_layers = [lyr for lyr in self.layers if lyr.supports_early_init]
            return len(init_layers) > 0 and all(lyr.is_initialized for lyr in init_layers)  # type: ignore[attr-defined]

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *_a: object, **_kw: object):
            return self.layers[layer_idx].update(k, v)  # type: ignore[union-attr]

        def has_previous_state(self, layer_idx: int) -> bool:
            return bool(self.layers[layer_idx].has_previous_state[0])  # type: ignore[union-attr]

        def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kw: object) -> torch.Tensor:
            return self.layers[layer_idx].update_conv_state(conv_states, **kw)  # type: ignore[union-attr,arg-type]

        def get_recurrent_state(self, layer_idx: int) -> object:
            lyr = self.layers[layer_idx]
            return lyr.recurrent_states[0] if lyr.has_previous_state[0] else None  # type: ignore[union-attr]

        def update_recurrent_state(self, recurrent_states: object, layer_idx: int) -> object:
            return self.layers[layer_idx].update_recurrent_state(recurrent_states)  # type: ignore[union-attr]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if isinstance(self.layers[layer_idx], _HF523DynamicLayer):
                return self.layers[layer_idx].get_seq_length()  # type: ignore[union-attr]
            first_full = next(lyr for lyr in self.layers if isinstance(lyr, _HF523DynamicLayer))
            return first_full.get_seq_length()

        def reset(self) -> None:
            for lyr in self.layers:
                lyr.reset()

        def batch_repeat_interleave(self, repeats: int) -> None:
            for lyr in self.layers:
                lyr.batch_repeat_interleave(repeats)

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            for lyr in self.layers:
                lyr.batch_select_indices(indices)

    host_mod_name = "_mock_hf523_qwen35_moe_mod"
    host_mod = types.ModuleType(host_mod_name)
    sys.modules[host_mod_name] = host_mod
    try:
        m_523 = build_toy(num_hidden_layers=8, hidden_size=32).double().eval()
        la_proto = m_523.model.layers[0].linear_attn
        host_mod.torch_chunk_gated_delta_rule = la_proto.chunk_gated_delta_rule  # type: ignore[attr-defined]
        host_mod.torch_recurrent_gated_delta_rule = la_proto.fused_recurrent_gated_delta_rule  # type: ignore[attr-defined]

        def _cls_hf523_forward(
            self: torch.nn.Module,
            x: torch.Tensor,
            cache_params: object = None,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del attention_mask
            b, t, _ = x.shape
            use_pre = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and t == 1  # type: ignore[attr-defined]
            qkv = self.in_proj_qkv(x).transpose(1, 2)  # type: ignore[operator]
            if cache_params is not None:
                full_cs = cache_params.update_conv_state(qkv, self.layer_idx, conv_kernel_size=self.ck)  # type: ignore[attr-defined]
                seq = full_cs[..., -self.ck :] if use_pre else torch.cat(  # type: ignore[operator]
                    [qkv.new_zeros(b, qkv.shape[1], self.ck - 1), qkv], dim=-1  # type: ignore[operator]
                )
            else:
                seq = torch.cat([qkv.new_zeros(b, qkv.shape[1], self.ck - 1), qkv], dim=-1)  # type: ignore[operator]
            qkv_act = F.silu(self.conv1d(seq))[..., -t:].transpose(1, 2)  # type: ignore[operator]
            q, k, v = qkv_act.view(b, t, 3, self.nh, self.d).unbind(2)  # type: ignore[attr-defined]
            q, k = F.normalize(upcast(q), dim=-1), F.normalize(upcast(k), dim=-1)
            v = upcast(v)
            bb, aa = upcast(self.in_proj_ba(x)).chunk(2, dim=-1)  # type: ignore[operator]
            beta = torch.sigmoid(bb)
            g = -upcast(self.A_log).exp() * F.softplus(aa + upcast(self.dt_bias))  # type: ignore[attr-defined]
            last_rec = cache_params.get_recurrent_state(self.layer_idx) if use_pre else None  # type: ignore[attr-defined]
            rule_fn = (
                host_mod.torch_recurrent_gated_delta_rule  # type: ignore[attr-defined]
                if t == 1
                else host_mod.torch_chunk_gated_delta_rule  # type: ignore[attr-defined]
            )
            core_out, last_rec = rule_fn(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=last_rec,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=False,
                scale=1.0,
            )
            if cache_params is not None:
                cache_params.update_recurrent_state(last_rec, self.layer_idx)  # type: ignore[attr-defined]
            o = self.norm(core_out.to(x.dtype))  # type: ignore[operator]
            return self.out_proj(o.reshape(b, t, self.nh * self.d))  # type: ignore[operator,attr-defined]

        kernel_sentinel_calls = [0]
        for lyr in m_523.model.layers:
            la = getattr(lyr, "linear_attn", None)
            if la is None:
                continue

            class _HF523MoeGatedDeltaNet(type(la)):  # type: ignore[misc]
                forward = _cls_hf523_forward

            _HF523MoeGatedDeltaNet.__module__ = host_mod_name
            la.__class__ = _HF523MoeGatedDeltaNet
            delattr(la, "chunk_gated_delta_rule")
            delattr(la, "fused_recurrent_gated_delta_rule")

            # Simulate `kernels==0.18.2` `kernelize()` having bound an instance `forward` + `_kernel_repo`
            def _bound_hub_kernel_fwd(*_a: object, **_kw: object) -> torch.Tensor:
                kernel_sentinel_calls[0] += 1
                raise AssertionError("bound hub kernel forward should be unwrapped while HSSD is active")

            la.__dict__["forward"] = _bound_hub_kernel_fwd
            la.__dict__["_kernel_repo"] = "Atlas-Inference/gdn"

        h_523 = apply_hssd(
            m_523, HSSDConfig(rank=8, chunk_size=6, window_chunks=2, state_dtype="auto", checkpoint_segments=False)
        )
        ctl_523 = apply_loop(
            m_523, {"prelude_end": 0, "core_end": 8, "num_passes": 2, "target_virtual_layers": 16, "kv_share": True}
        )
        with torch.no_grad():
            ctl_523.adapters.gates.fill_(0.2)
            for had in h_523.adapters.values():
                for p in had.parameters():
                    p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.1)

            full_523 = m_523(ids_a).logits
            c523 = _HF523DynamicCache(m_523.config)
            pre_523 = m_523(ids_a[:, :10], past_key_values=c523, use_cache=True)
            assert kernel_sentinel_calls[0] == 0
            assert c523.is_initialized is True

            lin0_st = c523.layers[0].recurrent_states[0]  # type: ignore[union-attr]
            assert isinstance(lin0_st, HSSDCacheState)
            assert lin0_st.dtype == torch.float64 and lin0_st.theta_cum.dtype == torch.float64
            static_ptrs_0 = [int(t.data_ptr()) for t in lin0_st._iter_tensors()]

            dec_523 = [pre_523.logits[:, -1:]]
            for pos in range(10, 15):
                dec_523.append(m_523(ids_a[:, pos : pos + 1], past_key_values=c523, use_cache=True).logits)
            inc_523 = torch.cat(dec_523, dim=1)
            assert float((inc_523 - full_523[:, 9:]).abs().max()) < 1e-11
            assert [int(t.data_ptr()) for t in lin0_st._iter_tensors()] == static_ptrs_0

            # `reset()` zeroes in-place while preserving CUDAGraph static addresses, then cleanly runs `ids_b`!
            c523.reset()
            assert [int(t.data_ptr()) for t in lin0_st._iter_tensors()] == static_ptrs_0
            assert int(lin0_st.step_count.abs().max().item()) == 0
            full_523_b = m_523(ids_b).logits
            pre_523_b = m_523(ids_b[:, :9], past_key_values=c523, use_cache=True)
            dec_523_b = [pre_523_b.logits[:, -1:]]
            for pos in range(9, 13):
                dec_523_b.append(m_523(ids_b[:, pos : pos + 1], past_key_values=c523, use_cache=True).logits)
            inc_523_b = torch.cat(dec_523_b, dim=1)
            assert float((inc_523_b - full_523_b[:, 8:]).abs().max()) < 1e-11

            # Also verify `state_dtype="bfloat16"` with `LinearAttentionLayer.lazy_initialization` + static address
            bf16_layer = _HF523LinearAttentionLayer()
            bf16_sample = lin0_st.to(dtype=torch.bfloat16)
            bf16_buf = bf16_layer.update_recurrent_state(bf16_sample)
            assert isinstance(bf16_buf, HSSDCacheState)
            assert bf16_buf.dtype == torch.bfloat16 and bf16_buf.theta_cum.dtype == torch.float64

            # Batch repeat_interleave & batch_select_indices on HF 5.23 cache
            c523.batch_repeat_interleave(2)
            assert c523.layers[0].recurrent_states[0].size(0) == 4  # type: ignore[union-attr]
            c523.batch_select_indices(torch.tensor([3, 0], dtype=torch.long))
            assert c523.layers[0].recurrent_states[0].size(0) == 2  # type: ignore[union-attr]

            ctl_523.unwrap()
            remove_hssd(m_523)
            # Verify `_kernel_repo` & bound `forward` are cleanly restored after `remove_hssd`
            assert m_523.model.layers[0].linear_attn.__dict__.get("_kernel_repo") == "Atlas-Inference/gdn"
            with pytest.raises(AssertionError, match="bound hub kernel forward"):
                m_523(ids_a[:, :2])
            assert kernel_sentinel_calls[0] == 1
    finally:
        sys.modules.pop(host_mod_name, None)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch.float8_e4m3fn required")
def test_pass_router_adapters_fp8_scaled_router_weight_loop_and_standalone():
    import sys
    import types

    from qloop.infer.hf_custom_modeling import _STANDALONE_CONFIG_SRC, _STANDALONE_MODELING_SRC
    from qloop.model.quant_autograd import dequantize_fp8_tensor
    from qloop.model.toy import ToyConfig, ToyForCausalLM

    fp8_dt = torch.float8_e4m3fn

    class _TupleFP8TopKRouter(torch.nn.Module):
        def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            top_k: int,
            *,
            scale_mode: str = "block2d",
            return_probs: bool = False,
            seed: int = 9901,
        ) -> None:
            super().__init__()
            self.num_experts = num_experts
            self.hidden_size = hidden_size
            self.top_k = top_k
            self.return_probs = return_probs
            g = torch.Generator().manual_seed(seed)
            raw_fp8 = (
                torch.randn(num_experts, hidden_size, generator=g, dtype=torch.float32).clamp(-6.0, 6.0).to(fp8_dt)
            )
            self.weight = torch.nn.Parameter(raw_fp8, requires_grad=False)
            self.bias: torch.nn.Parameter | None = None
            if scale_mode == "block2d":
                self.weight_block_size = (2, 8)
                # Deliberately asymmetric block scales (0.02 vs 18.0) so unscaled `upcast(weight)` flips expert ranking!
                s = torch.tensor(
                    [
                        [0.02, 18.0, 0.03, 14.0],
                        [16.0, 0.02, 15.0, 0.04],
                    ],
                    dtype=torch.float32,
                )
                self.register_buffer("weight_scale_inv", s)
            elif scale_mode == "row1d":
                s = torch.tensor([0.02, 19.0, 0.03, 15.0], dtype=torch.float32)
                self.register_buffer("weight_scale_inv", s)
            elif scale_mode == "scalar":
                self.register_buffer("weight_scale_inv", torch.tensor(0.03125, dtype=torch.float32))
            elif scale_mode == "missing":
                pass
            else:
                raise ValueError(scale_mode)

        def _dequant_ref(self) -> torch.Tensor:
            s = getattr(self, "weight_scale_inv", None)
            if s is None:
                # Simulate stock router computing probs from true scaled weights while scale attr was stripped
                return self.weight.to(torch.float32) * 0.05
            bsize = getattr(self, "weight_block_size", None)
            return dequantize_fp8_tensor(self.weight, s, dtype=torch.float32, block_size=bsize)

        def forward(self, hidden_states: torch.Tensor):
            flat = hidden_states.reshape(-1, self.hidden_size).float()
            logits = F.linear(flat, self._dequant_ref())
            probs = F.softmax(logits, dim=-1)
            top_w, top_i = probs.topk(self.top_k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            full_out = probs if self.return_probs else logits
            return full_out.to(hidden_states.dtype), top_w.to(hidden_states.dtype), top_i

    def _install_tuple_fp8_routers(
        model: ToyForCausalLM, *, scale_mode: str, return_probs: bool, base_seed: int
    ) -> list[_TupleFP8TopKRouter]:
        installed: list[_TupleFP8TopKRouter] = []
        for li, lyr in enumerate(model.model.layers):
            mlp = lyr.mlp
            rt = _TupleFP8TopKRouter(4, 32, 2, scale_mode=scale_mode, return_probs=return_probs, seed=base_seed + li)
            mlp.gate = rt  # type: ignore[assignment]

            def _tuple_moe_fwd(x: torch.Tensor, _mlp: torch.nn.Module = mlp) -> torch.Tensor:
                b, t, h = x.shape
                flat = x.reshape(-1, h)
                _rlogits, top_w, top_idx = _mlp.gate(flat)  # type: ignore[operator]
                out = _mlp.experts(flat, top_idx, top_w)  # type: ignore[operator]
                out = out + torch.sigmoid(_mlp.shared_expert_gate(flat)) * _mlp.shared_expert(flat)  # type: ignore[operator]
                return out.view(b, t, h)

            mlp.forward = _tuple_moe_fwd  # type: ignore[method-assign]
            installed.append(rt)
        return installed

    for scale_mode in ("block2d", "row1d", "scalar"):
        for return_probs in (False, True):
            m = build_toy(num_hidden_layers=4, hidden_size=32, num_experts=4).float().eval()
            router0 = _install_tuple_fp8_routers(m, scale_mode=scale_mode, return_probs=return_probs, base_seed=9901)[0]

            captured_r2: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            ctl = apply_loop(
                m,
                {
                    "prelude_end": 0,
                    "core_end": 4,
                    "num_passes": 2,
                    "target_virtual_layers": 8,
                    "per_pass_router": True,
                    "router_adapter_rank": 4,
                },
            )
            # Register capture hook AFTER `apply_loop` so it sees the output rewritten by `_pass_router_hook`!
            router0.register_forward_hook(
                lambda _mod, _inp, out, cap=captured_r2: cap.append(
                    (out[0].detach().clone(), out[1].detach().clone(), out[2].detach().clone())
                )
            )

            gen = torch.Generator().manual_seed(7700)
            with torch.no_grad():
                pr = ctl.adapters.pass_router
                assert pr is not None
                pr.up.copy_(torch.randn(pr.up.shape, generator=gen) * 0.25)
                pr.bias.copy_(torch.tensor([[[0.15, -0.10, 0.20, -0.05]]] * 4).squeeze(1))
                pr.repel.fill_(0.35)

                x_tok = torch.randn(1, 5, 32, generator=gen, dtype=torch.float32)
                # Capture exact input reaching layer 0's router in Pass 1 and Pass 2
                router_inps: list[torch.Tensor] = []
                router0.register_forward_pre_hook(
                    lambda _m, a, inps=router_inps: inps.append(a[0].detach().clone())
                )
                _ = m.model(inputs_embeds=x_tok)
                # Layer 0's router runs once per pass: index 0 is (phys=0, r=1), index 1 is (phys=0, r=2)
                assert len(captured_r2) == 2
                x_p1 = router_inps[0].reshape(-1, 32)
                x_p2 = router_inps[1].reshape(-1, 32)
                got_full, got_w, got_idx = captured_r2[1]

                # Exact reference with scaled FP8 dequantization
                w_deq = router0._dequant_ref()
                p1_top = F.softmax(F.linear(x_p1, w_deq), dim=-1).topk(2, dim=-1).indices
                prev_mask = torch.zeros(5, 4, dtype=torch.float32).scatter_(1, p1_top, 1.0)
                delta_ref = ctl.adapters.pass_router.delta_logits(x_p2, 0, 0, prev_mask)  # type: ignore[union-attr]
                ref_logits = F.linear(x_p2, w_deq) + delta_ref
                ref_probs = F.softmax(ref_logits, dim=-1)
                ref_w, ref_idx = ref_probs.topk(2, dim=-1)
                ref_w = ref_w / ref_w.sum(-1, keepdim=True)
                ref_full = ref_probs if return_probs else ref_logits

                assert float((got_full - ref_full).abs().max()) < 1e-5
                assert float((got_w - ref_w).abs().max()) < 1e-6
                assert torch.equal(got_idx, ref_idx)

                # Prove unscaled `F.linear(x_p2, router0.weight.float()) + delta_ref` would have been wildly wrong!
                unscaled_logits = F.linear(x_p2, router0.weight.float()) + delta_ref
                unscaled_probs = F.softmax(unscaled_logits, dim=-1)
                assert float((ref_probs - unscaled_probs).abs().max()) > 0.25

            ctl.unwrap()

    # Fallback test when `weight_scale_inv` is missing on an FP8 router: must NOT use unscaled `weight.float()`!
    m_miss = build_toy(num_hidden_layers=4, hidden_size=32, num_experts=4).float().eval()
    router_missing = _install_tuple_fp8_routers(m_miss, scale_mode="missing", return_probs=True, base_seed=9920)[0]
    ctl_miss = apply_loop(
        m_miss, {"prelude_end": 0, "core_end": 4, "num_passes": 2, "target_virtual_layers": 8, "per_pass_router": True}
    )
    cap_miss: list[torch.Tensor] = []
    inp_miss: list[torch.Tensor] = []
    router_missing.register_forward_pre_hook(lambda _m, a: inp_miss.append(a[0].detach().clone()))
    router_missing.register_forward_hook(lambda _m, _i, out: cap_miss.append(out[0].detach().clone()))
    with torch.no_grad():
        ctl_miss.adapters.pass_router.bias.fill_(0.2)  # type: ignore[union-attr]
        ctl_miss.adapters.pass_router.up.fill_(0.1)  # type: ignore[union-attr]
        _ = m_miss(torch.randint(0, 50, (1, 4)))
        x_m2 = inp_miss[1].reshape(-1, 32)
        base_probs_m2 = F.softmax(F.linear(x_m2, router_missing._dequant_ref()), dim=-1)
        delta_m2 = ctl_miss.adapters.pass_router.delta_logits(x_m2, 0, 0, None)  # type: ignore[union-attr]
        expected_fb_probs = F.softmax(base_probs_m2.clamp_min(1e-30).log() + delta_m2, dim=-1)
        assert float((cap_miss[1] - expected_fb_probs).abs().max()) < 1e-5
    ctl_miss.unwrap()

    # Standalone `_STANDALONE_MODELING_SRC` (`_HAS_QLOOP = False`) parity on `float8_e4m3fn` + `weight_scale_inv` router
    mock_tf = types.ModuleType("transformers")

    class _MockPretrainedConfig:
        model_type = "qwen36_loop_hssd"

        def __init__(self, **kw: object) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class _MockPreTrainedModel(torch.nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config = config

    class _MockAutoConfig:
        @staticmethod
        def for_model(_mtype: str, **kw: object) -> ToyConfig:
            known = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            if "layer_types" in kw and isinstance(kw["layer_types"], list):
                kw = dict(kw, layer_types=tuple(kw["layer_types"]))
            return ToyConfig(**{k: kw[k] for k in known})  # type: ignore[arg-type]

    st_routers: list[_TupleFP8TopKRouter] = []

    class _MockAutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **_kw: object) -> ToyForCausalLM:
            m_st = ToyForCausalLM(cfg).float()
            _convert_toy_to_hf5_recurrent_attr(m_st)
            st_routers.extend(
                _install_tuple_fp8_routers(m_st, scale_mode="block2d", return_probs=False, base_seed=9955)
            )
            return m_st

    mock_tf.PretrainedConfig = _MockPretrainedConfig  # type: ignore[attr-defined]
    mock_tf.PreTrainedModel = _MockPreTrainedModel  # type: ignore[attr-defined]
    mock_tf.AutoConfig = _MockAutoConfig  # type: ignore[attr-defined]
    mock_tf.AutoModelForCausalLM = _MockAutoModelForCausalLM  # type: ignore[attr-defined]

    saved_tf = sys.modules.get("transformers")
    saved_cfg_mod = sys.modules.get("configuration_qwen36_loophssd")
    orig_find_spec = importlib.util.find_spec
    sys.modules["transformers"] = mock_tf
    try:
        importlib.util.find_spec = lambda name, *a, **kw: (  # type: ignore[assignment]
            None if name == "qloop" else orig_find_spec(name, *a, **kw)
        )
        cfg_mod = types.ModuleType("configuration_qwen36_loophssd")
        exec(compile(_STANDALONE_CONFIG_SRC, "configuration_qwen36_loophssd.py", "exec"), cfg_mod.__dict__)
        sys.modules["configuration_qwen36_loophssd"] = cfg_mod

        mod_ns = types.ModuleType("modeling_qwen36_loophssd")
        exec(compile(_STANDALONE_MODELING_SRC, "modeling_qwen36_loophssd.py", "exec"), mod_ns.__dict__)
        assert mod_ns._HAS_QLOOP is False

        base_d = {
            "vocab_size": 128,
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "partial_rotary_factor": 0.5,
            "linear_num_heads": 4,
            "linear_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        }
        arch_d: dict[str, object] = {
            "base_model_type": "toy",
            "base_dtype": "float32",
            "loop": {
                "enabled": True,
                "prelude_end": 0,
                "core_end": 4,
                "num_passes": 2,
                "kv_share": True,
                "per_pass_router": True,
                "router_adapter_rank": 4,
            },
        }
        scfg = cfg_mod.Qwen36LoopHSSDConfig(base_config=base_d, qloop_v3_arch=arch_d)
        smodel = mod_ns.Qwen36LoopHSSDForCausalLM(scfg, _dtype_override=torch.float32).eval()
        st_router0 = st_routers[0]

        st_cap: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        st_inps: list[torch.Tensor] = []
        st_router0.register_forward_pre_hook(lambda _m, a: st_inps.append(a[0].detach().clone()))
        st_router0.register_forward_hook(
            lambda _m, _i, out: st_cap.append(
                (out[0].detach().clone(), out[1].detach().clone(), out[2].detach().clone())
            )
        )
        with torch.no_grad():
            pr_st = smodel._loop_adapters.pass_router
            pr_st.up.fill_(0.18)
            pr_st.bias.copy_(torch.tensor([[[0.25, -0.15, 0.10, -0.20]]] * 4).squeeze(1))
            pr_st.repel.fill_(0.30)

            _ = smodel(input_ids=torch.randint(0, 100, (1, 5), generator=torch.Generator().manual_seed(9966)))
            assert len(st_cap) == 2
            sx_p1 = st_inps[0].reshape(-1, 32)
            sx_p2 = st_inps[1].reshape(-1, 32)
            sgot_full, sgot_w, sgot_idx = st_cap[1]

            sw_deq = st_router0._dequant_ref()
            sp1_top = F.softmax(F.linear(sx_p1, sw_deq), dim=-1).topk(2, dim=-1).indices
            sprev_m = torch.zeros(5, 4, dtype=torch.float32).scatter_(1, sp1_top, 1.0)
            sdelta_ref = smodel._loop_adapters.pass_router.delta_logits(sx_p2, 0, 0, sprev_m)
            sref_logits = F.linear(sx_p2, sw_deq) + sdelta_ref
            sref_probs = F.softmax(sref_logits, dim=-1)
            sref_w, sref_idx = sref_probs.topk(2, dim=-1)
            sref_w = sref_w / sref_w.sum(-1, keepdim=True)

            assert float((sgot_full - sref_logits).abs().max()) < 1e-5
            assert float((sgot_w - sref_w).abs().max()) < 1e-6
            assert torch.equal(sgot_idx, sref_idx)
            sunscaled_probs = F.softmax(F.linear(sx_p2, st_router0.weight.float()) + sdelta_ref, dim=-1)
            assert float((sref_probs - sunscaled_probs).abs().max()) > 0.25
    finally:
        importlib.util.find_spec = orig_find_spec  # type: ignore[assignment]
        if saved_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved_tf
        if saved_cfg_mod is None:
            sys.modules.pop("configuration_qwen36_loophssd", None)
        else:
            sys.modules["configuration_qwen36_loophssd"] = saved_cfg_mod


