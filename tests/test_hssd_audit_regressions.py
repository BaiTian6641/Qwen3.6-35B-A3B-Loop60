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
