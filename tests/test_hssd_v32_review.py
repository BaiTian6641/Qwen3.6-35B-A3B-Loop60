"""Comprehensive `float64` & systems verification suite for HSSD v3.2 review fixes (`H1`–`H9` & `5.2`)."""
from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import qloop.model.hssd as hssd_mod
from qloop.model.hssd import (
    DEFAULT_MU_MAX,
    HSSDAdapter,
    HSSDCacheState,
    HSSDConfig,
    PeakBoltzmannGate,
    _compute_root_surprise_and_routing,
    apply_2d_rotation,
    apply_2d_rotation_trig,
    apply_hssd,
    chunk_hssd,
    compute_2d_trig,
    compute_write_budget_loss,
    recurrent_hssd,
    recurrent_hssd_step,
)
from qloop.model.probes import measure_hssd_tf32_drift, profile_hssd_layer
from qloop.model.toy import ToyConfig, ToyForCausalLM


def _synth_v32_inputs(
    *,
    bsz: int = 2,
    seq_len: int = 48,
    hv: int = 4,
    dk: int = 16,
    dv: int = 16,
    num_pages: int = 8,
    seed: int = 3201,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    hdk = dk // 2
    q = F.normalize(torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dtype), dim=-1)
    v = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=dtype) * 0.5

    ps = torch.exp(torch.linspace(math.log(1e-4), math.log(5e-2), num_pages, dtype=dtype))
    ch = torch.empty(bsz, seq_len, hv, hdk, dtype=dtype).uniform_(0.5, 1.5, generator=gen)
    b_gate = torch.empty(bsz, seq_len, hv, dk, dtype=dtype).uniform_(0.35, 1.75, generator=gen)
    w_gate = torch.empty(bsz, seq_len, hv, dv, dtype=dtype).uniform_(0.25, 0.95, generator=gen)
    theta = torch.empty(bsz, seq_len, hv, hdk, dtype=dtype).uniform_(-0.6 * math.pi, 0.6 * math.pi, generator=gen)

    sb = torch.randn(bsz, seq_len, hv, generator=gen, dtype=dtype) * 0.3 + 1.0
    slp = torch.full((hv,), 3.0, dtype=dtype)
    thr = torch.full((hv,), 0.85, dtype=dtype)

    wl = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=dtype)
    rl = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=dtype)
    qs = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dtype)
    ks = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dtype)
    vs = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=dtype) * 0.5
    swa_slope = torch.linspace(0.05, 0.65, hv, dtype=dtype)

    if requires_grad:
        for t in (q, k, v, ps, ch, b_gate, w_gate, theta, sb, slp, thr, wl, rl, qs, ks, vs, swa_slope):
            t.requires_grad_(True)

    return {
        "query": q,
        "key": k,
        "value": v,
        "gamma": (ps, ch),
        "b_gate": b_gate,
        "w_gate": w_gate,
        "theta": theta,
        "surprise_bias": sb,
        "surprise_slope": slp,
        "surprise_thresh": thr,
        "write_logits": wl,
        "read_logits": rl,
        "q_swa": qs,
        "k_swa": ks,
        "v_swa": vs,
        "swa_recency_slope": swa_slope,
    }


# ==============================================================================
# 1. H1 & H6: Full-Strength Selected-Page Writes, Exact Overwrite & Write Budget
# ==============================================================================
def test_h1_h6_full_strength_selected_page_writes_and_exact_overwrite_fp64() -> None:
    """Verify writing (k, v1) then overwriting (k, v2) with b=w=g=1 returns exact v2 (`0.0` attenuation)."""
    dt = torch.float64
    bsz, hv, dk, dv, m_pages, k_pages = 1, 2, 8, 8, 8, 2
    c_size, w_chunks = 4, 1  # far horizon activates after (W + 1) * C = 8 steps -> at chunk 2 (t >= 8)

    gen = torch.Generator().manual_seed(101)
    k_vec = F.normalize(torch.randn(bsz, 1, hv, dk, generator=gen, dtype=dt), dim=-1)
    v1 = torch.randn(bsz, 1, hv, dv, generator=gen, dtype=dt)
    v2 = torch.randn(bsz, 1, hv, dv, generator=gen, dtype=dt)

    # Sequence of 12 tokens (3 chunks of 4):
    #   t=0: write (k_vec, v1) with g=1, b=1, w=1
    #   t=1: overwrite (k_vec, v2) with g=1, b=1, w=1
    #   t=2..10: g=0 no-ops so boundary_ring[0] advances past chunk 0 once t >= 8!
    #   t=11 (chunk 2): query q = k_vec (with scale=1.0, use_swa=False) reads S_boundary[1]!
    seq_len = 12
    q_seq = torch.zeros(bsz, seq_len, hv, dk, dtype=dt)
    k_seq = torch.zeros(bsz, seq_len, hv, dk, dtype=dt)
    v_seq = torch.zeros(bsz, seq_len, hv, dv, dtype=dt)
    g_ovr = torch.zeros(bsz, seq_len, hv, dtype=dt)

    k_seq[:, 0:1] = k_vec
    v_seq[:, 0:1] = v1
    g_ovr[:, 0:1] = 1.0

    k_seq[:, 1:2] = k_vec
    v_seq[:, 1:2] = v2
    g_ovr[:, 1:2] = 1.0

    q_seq[:, -1:] = k_vec

    # Unequal logits on top-2 pages (pages 6 and 7) so split-mode attenuates unequally
    logits_base = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.7, 2.4], dtype=dt)
    wl = logits_base.view(1, 1, m_pages).expand(bsz, seq_len, m_pages).clone()
    rl = wl.clone()

    zeros_gamma = torch.zeros(bsz, seq_len, hv, dk // 2, dtype=dt)
    ones_b = torch.ones(bsz, seq_len, hv, dk, dtype=dt)
    ones_w = torch.ones(bsz, seq_len, hv, dv, dtype=dt)
    zeros_theta = torch.zeros(bsz, seq_len, hv, dk // 2, dtype=dt)
    sb = torch.zeros(bsz, seq_len, hv, dtype=dt)
    slp = torch.full((hv,), 3.0, dtype=dt)
    thr = torch.full((hv,), 0.85, dtype=dt)
    zeros_swa_k = torch.zeros(bsz, seq_len, hv, dk, dtype=dt)
    zeros_swa_v = torch.zeros(bsz, seq_len, hv, dv, dtype=dt)

    out_full, st_full = chunk_hssd(
        q_seq,
        k_seq,
        v_seq,
        zeros_gamma,
        ones_b,
        ones_w,
        zeros_theta,
        sb,
        slp,
        thr,
        wl,
        rl,
        zeros_swa_k,
        zeros_swa_k,
        zeros_swa_v,
        chunk_size=c_size,
        window_chunks=w_chunks,
        num_pages=m_pages,
        top_k_pages=k_pages,
        output_final_state=True,
        g_override=g_ovr,
        scale=1.0,
        use_swa=False,
        use_phase=False,
        write_mode="full",
    )
    assert st_full is not None
    retrieved_full = out_full[:, -1:]
    err_full = (retrieved_full - v2).abs().max().item()
    assert err_full < 1e-12, f"write_mode='full' overwrite must recover exact v2, got max_err={err_full:.3e}"

    out_rec, _ = recurrent_hssd(
        q_seq,
        k_seq,
        v_seq,
        zeros_gamma,
        ones_b,
        ones_w,
        zeros_theta,
        sb,
        slp,
        thr,
        wl,
        rl,
        zeros_swa_k,
        zeros_swa_k,
        zeros_swa_v,
        chunk_size=c_size,
        window_chunks=w_chunks,
        num_pages=m_pages,
        top_k_pages=k_pages,
        g_override=g_ovr,
        scale=1.0,
        use_swa=False,
        use_phase=False,
        write_mode="full",
    )
    assert (out_full - out_rec).abs().max().item() < 1e-12

    # Legacy split mode exhibits the 0.25*v1 + 0.5*v2 attenuation/residual artifact:
    out_split, st_split = chunk_hssd(
        q_seq,
        k_seq,
        v_seq,
        zeros_gamma,
        ones_b,
        ones_w,
        zeros_theta,
        sb,
        slp,
        thr,
        wl,
        rl,
        zeros_swa_k,
        zeros_swa_k,
        zeros_swa_v,
        chunk_size=c_size,
        window_chunks=w_chunks,
        num_pages=m_pages,
        top_k_pages=k_pages,
        output_final_state=True,
        g_override=g_ovr,
        scale=1.0,
        use_swa=False,
        use_phase=False,
        write_mode="split",
    )
    assert st_split is not None
    assert (out_split[:, -1:] - v2).abs().max().item() > 0.1

    # H6: Relative surprise error on already-stored (k_vec, v2) drops to ~0.0 under write_mode="full"!
    pi_w = F.softmax(wl[:, :1], dim=-1)
    tv, ti = torch.topk(pi_w, k=k_pages, dim=-1)
    tw = tv / tv.sum(dim=-1, keepdim=True)
    pred_full = hssd_mod._routed_bank_readout(k_vec, st_full.s_chunk_start_bank, ti, tw)
    eps_full = (v2 - pred_full).norm(dim=-1) / (v2.norm(dim=-1) + 1e-6)
    assert eps_full.max().item() < 1e-6

    pred_split = hssd_mod._routed_bank_readout(k_vec, st_split.s_chunk_start_bank, ti, tw)
    eps_split = (v2 - pred_split).norm(dim=-1) / (v2.norm(dim=-1) + 1e-6)
    assert eps_split.min().item() > 0.25


def test_h1_write_budget_u_sum_exact_accounting_and_ste_router_gradients() -> None:
    dt = torch.float64
    gen = torch.Generator().manual_seed(102)
    bsz, c_len, hv, dk, dv, m_pages, k_pages = 2, 16, 4, 8, 8, 8, 2

    k_hat = F.normalize(torch.randn(bsz, c_len, hv, dk, generator=gen, dtype=dt), dim=-1)
    v_c = torch.randn(bsz, c_len, hv, dv, generator=gen, dtype=dt)
    s_bank = torch.randn(bsz, m_pages, hv, dk, dv, generator=gen, dtype=dt) * 0.1
    sb = torch.randn(bsz, c_len, hv, generator=gen, dtype=dt)
    slp = torch.full((hv,), 3.0, dtype=dt)
    thr = torch.full((hv,), 0.85, dtype=dt)
    wl = torch.randn(bsz, c_len, m_pages, generator=gen, dtype=dt).requires_grad_(True)

    g_page, u_sum, valid_toks, top_idx, g_active = _compute_root_surprise_and_routing(
        k_hat,
        v_c,
        s_bank,
        sb,
        slp,
        thr,
        wl,
        k_pages,
        wmask_c=None,
        g_override_c=None,
        use_surprise=True,
        surprise_mode="routed",
        write_mode="full",
    )
    assert valid_toks.item() == float(bsz * c_len)

    # Full-strength write means g_active == g_root on every selected slot
    g_root_ref = g_active[:, :, 0, :]
    assert torch.allclose(g_active[:, :, 1, :], g_root_ref, atol=1e-12, rtol=0.0)

    # Exact page write-mass accounting: u_sum[m] == sum_{(b,t): m in top_idx} mean_h(g_root^2)
    g_sq_mean = (g_root_ref * g_root_ref).mean(dim=-1)  # [B, C]
    for m in range(m_pages):
        sel_m = (top_idx == m).any(dim=-1).to(dtype=dt)
        expected_u_m = (sel_m * g_sq_mean).sum()
        assert abs(u_sum[m].item() - expected_u_m.item()) < 1e-12

    # Total write mass across M pages equals k * sum(g_sq_mean)
    assert abs(u_sum.sum().item() - k_pages * g_sq_mean.sum().item()) < 1e-12

    # STE gradients flow into write_logits from BOTH write-budget loss AND g_page writes
    duty = u_sum / valid_toks
    budget_loss = compute_write_budget_loss(duty, DEFAULT_MU_MAX)
    (budget_loss + (g_page * 0.37).sum()).backward()
    assert wl.grad is not None and torch.isfinite(wl.grad).all()
    assert wl.grad.abs().max().item() > 1e-5


# ==============================================================================
# 2. H2: Teacher-Matched Init, Stock Decay Seed, Scheduled Blend & Teacher Hooks
# ==============================================================================
def test_h2_teacher_matched_initialization_and_stock_decay_seeding() -> None:
    cfg = HSSDConfig()
    assert cfg.write_mode == "full"
    assert cfg.surprise_bias_init == 1.5
    assert cfg.tau_init == 1.0
    assert cfg.use_stock_decay_init is True

    ad = HSSDAdapter(hidden_size=32, num_v_heads=4, head_k_dim=16, head_v_dim=16, cfg=cfg, layer_id=0)
    assert torch.allclose(ad.surprise_proj.bias, torch.full((4,), 1.5))
    assert torch.allclose(ad.boltzmann_gate.tau, torch.full((4,), 1.0))

    # Step-0 SWA Boltzmann gate with tau_init=1.0 is ~10x more active than legacy tau_init=2.2 (~1.1%)
    # and reaches balanced ~0.45-0.55 weight on a moderate local key match (where legacy stayed <0.08).
    gen = torch.Generator().manual_seed(202)
    scores = torch.randn(2, 4, 8, 32, generator=gen)
    legacy_gate = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=2.2)
    mean_hit = float(ad.boltzmann_gate(scores).mean().item())
    legacy_hit = float(legacy_gate(scores).mean().item())
    assert legacy_hit < 0.02 and mean_hit > 8.0 * legacy_hit
    scores_match = scores.clone()
    scores_match[..., 0] = 3.5
    match_hit = float(ad.boltzmann_gate(scores_match).mean().item())
    assert 0.30 < match_hit < 0.80, f"Expected balanced local-match p_hit with tau_init=1.0, got {match_hit:.4f}"

    # Step-0 write_router and read_router select the EXACT same working pages {M-2, M-1} = {6, 7}
    x_in = torch.randn(2, 16, 32, generator=gen)
    q = torch.randn(2, 16, 4, 16, generator=gen)
    k = torch.randn(2, 16, 4, 16, generator=gen)
    v = torch.randn(2, 16, 4, 16, generator=gen)
    g_stock = torch.full((2, 16, 4), -0.02)
    beta = torch.full((2, 16, 4), 0.5)

    proj = ad.project_inputs(x_in, q, k, v, g_stock, beta, use_qk_l2norm=True)
    w_top = torch.topk(proj["write_logits"], k=cfg.k_pages, dim=-1).indices
    r_top = torch.topk(proj["read_logits"], k=cfg.k_pages, dim=-1).indices
    w_set = {tuple(sorted(row.tolist())) for row in w_top.reshape(-1, cfg.k_pages)}
    r_set = {tuple(sorted(row.tolist())) for row in r_top.reshape(-1, cfg.k_pages)}
    assert w_set == {(6, 7)} and r_set == {(6, 7)}, f"Expected matched working pages (6, 7), got {w_set} vs {r_set}"

    # Stock decay seeding: working page M-1 == 1.0x (-g_base) == 0.02, vault page 0 scaled down by 1e-4 / 5e-2
    ps, ch = proj["gamma_page_scale"], proj["gamma_chan_pair"]
    eff_gamma_m_last = ps[-1] * ch
    eff_gamma_vault0 = ps[0] * ch
    assert torch.allclose(eff_gamma_m_last, torch.full_like(ch, 0.02), atol=1e-6)
    assert torch.allclose(eff_gamma_vault0, torch.full_like(ch, 0.02 * (1e-4 / 5e-2)), atol=1e-7)

    # Synthetic g_base == 0 fallback uses page_gamma_prior cleanly
    proj_zero_g = ad.project_inputs(x_in, q, k, v, torch.zeros_like(g_stock), beta, use_qk_l2norm=True)
    ps0, ch0 = proj_zero_g["gamma_page_scale"], proj_zero_g["gamma_chan_pair"]
    assert torch.allclose(ps0[-1] * ch0, torch.full_like(ch0, cfg.decay_gamma_max), atol=1e-6)
    assert torch.allclose(ps0[0] * ch0, torch.full_like(ch0, cfg.decay_gamma_min), atol=1e-7)


def test_h2_scheduled_blend_exact_kl_zero_ste_grads_and_teacher_hooks() -> None:
    torch.manual_seed(203)
    tcfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
    )
    stock_model = ToyForCausalLM(tcfg).eval()
    hssd_model = ToyForCausalLM(tcfg).eval()
    hssd_model.load_state_dict(stock_model.state_dict())

    handle = apply_hssd(
        hssd_model,
        HSSDConfig(
            num_pages=4,
            top_k_pages=2,
            rank=8,
            chunk_size=8,
            window_chunks=2,
            blend_init=0.0,
            blend_warmup_steps=10,
        ),
    )
    ids = torch.tensor([[3, 11, 27, 42, 19, 8, 51, 14, 33, 22, 7, 45]], dtype=torch.long)

    with torch.no_grad():
        stock_logits = stock_model(ids, use_cache=False).logits

    # At lam == 0.0: forward output equals stock_model EXACTLY (diff == 0.0, KL == 0.0) while HSSD gets STE grads!
    assert handle.blend == 0.0
    hssd_model.zero_grad(set_to_none=True)
    lam0_logits = hssd_model(ids, use_cache=False).logits
    assert torch.equal(lam0_logits.detach(), stock_logits), (
        f"Expected bit-exact stock logits at blend=0.0, max_diff={(lam0_logits - stock_logits).abs().max().item():.3e}"
    )
    loss0 = (lam0_logits.float() ** 2).mean()
    loss0.backward()

    grad_norms = [
        p.grad.abs().max().item()
        for ad in handle.adapters.values()
        for p in ad.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0 and max(grad_norms) > 0.0, "HSSD parameters must receive step-0 STE grads at lam=0.0"

    # Linear blend warmup schedule
    assert abs(handle.step_blend(0) - 0.0) < 1e-12
    assert abs(handle.step_blend(4) - 0.4) < 1e-12
    assert abs(handle.step_blend(10) - 1.0) < 1e-12
    assert abs(handle.step_blend(25) - 1.0) < 1e-12

    # Teacher bypass + Online Layer-Wise Teacher-Forced Warm-Up callback (`5.2 #2`)
    captured_calls: list[tuple[int, torch.Size, torch.Size]] = []

    def _teacher_cb(
        layer_idx: int,
        adapter: HSSDAdapter,
        orig_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        x_in: torch.Tensor,
        mor_wmask: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> None:
        del adapter, key, value, g, beta, mor_wmask, attention_mask
        captured_calls.append((layer_idx, orig_out.shape, x_in.shape))
        assert orig_out.shape[:2] == query.shape[:2]

    with torch.no_grad(), handle.teacher_layerwise_Hook(_teacher_cb):
        t_logits = hssd_model(ids, use_cache=False).logits
    assert torch.equal(t_logits, stock_logits)
    assert [c[0] for c in captured_calls] == [0, 2]
    assert handle.bypass_to_orig is False


# ==============================================================================
# 3. H3: Local SWA Recency Slope (fp64 Parity, Zero-Init Grad & Newer-Key Bias)
# ==============================================================================
def test_h3_local_swa_recency_slope_fp64_parity_grad_and_newer_key_preference() -> None:
    inp = _synth_v32_inputs(bsz=2, seq_len=40, hv=4, dk=12, dv=12, seed=303, requires_grad=True)
    gate = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)
    swa_slp_zero = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    # 3a: Non-zero gradient at swa_recency_slope == 0.0
    out_z, _ = chunk_hssd(
        inp["query"],
        inp["key"],
        inp["value"],
        inp["gamma"],
        inp["b_gate"],
        inp["w_gate"],
        inp["theta"],
        inp["surprise_bias"],
        inp["surprise_slope"],
        inp["surprise_thresh"],
        inp["write_logits"],
        inp["read_logits"],
        inp["q_swa"],
        inp["k_swa"],
        inp["v_swa"],
        boltzmann_gate=gate,
        swa_recency_slope=swa_slp_zero,
        chunk_size=8,
        window_chunks=2,
        segment_chunks=2,
    )
    (out_z * torch.sin(torch.arange(out_z.numel(), dtype=torch.float64).view_as(out_z))).sum().backward()
    assert swa_slp_zero.grad is not None and swa_slp_zero.grad.abs().max().item() > 1e-6

    # 3b: Exact fp64 chunk == recurrent == recurrent_hssd_step parity with non-zero swa_recency_slope
    inp2 = _synth_v32_inputs(bsz=2, seq_len=40, hv=4, dk=12, dv=12, seed=304, requires_grad=False)
    out_c, st_c = chunk_hssd(
        inp2["query"],
        inp2["key"],
        inp2["value"],
        inp2["gamma"],
        inp2["b_gate"],
        inp2["w_gate"],
        inp2["theta"],
        inp2["surprise_bias"],
        inp2["surprise_slope"],
        inp2["surprise_thresh"],
        inp2["write_logits"],
        inp2["read_logits"],
        inp2["q_swa"],
        inp2["k_swa"],
        inp2["v_swa"],
        boltzmann_gate=gate,
        swa_recency_slope=inp2["swa_recency_slope"],
        chunk_size=8,
        window_chunks=2,
        segment_chunks=2,
        output_final_state=True,
    )
    out_r, st_r = recurrent_hssd(
        inp2["query"],
        inp2["key"],
        inp2["value"],
        inp2["gamma"],
        inp2["b_gate"],
        inp2["w_gate"],
        inp2["theta"],
        inp2["surprise_bias"],
        inp2["surprise_slope"],
        inp2["surprise_thresh"],
        inp2["write_logits"],
        inp2["read_logits"],
        inp2["q_swa"],
        inp2["k_swa"],
        inp2["v_swa"],
        boltzmann_gate=gate,
        swa_recency_slope=inp2["swa_recency_slope"],
        chunk_size=8,
        window_chunks=2,
        segment_chunks=2,
        output_final_state=True,
    )
    assert (out_c - out_r).abs().max().item() < 1e-12
    assert st_c is not None and st_r is not None and st_c.state_equal(st_r, atol=1e-12, rtol=1e-12)

    # 3c: Repeated-key local window test: positive slope favors newer (k, v2) over older (k, v1)
    dt = torch.float64
    bsz, t_len, hv, dk, dv = 1, 8, 2, 8, 8
    k_match = torch.zeros(bsz, 1, hv, dk, dtype=dt)
    k_match[..., 0] = 4.0
    k_orth = torch.zeros(bsz, 1, hv, dk, dtype=dt)
    k_orth[..., 1] = -4.0

    ks = k_orth.expand(bsz, t_len, hv, dk).clone()
    vs = torch.zeros(bsz, t_len, hv, dv, dtype=dt)
    ks[:, 0:1] = k_match
    vs[:, 0, :, 0] = 1.0  # older v1 = e_0 at t=0
    ks[:, 6:7] = k_match
    vs[:, 6, :, 1] = 1.0  # newer v2 = e_1 at t=6

    qs = torch.zeros(bsz, t_len, hv, dk, dtype=dt)
    qs[:, 7:8] = k_match  # query at t=7

    high_hit_gate = PeakBoltzmannGate(hv, kappa_init=10.0, tau_init=-10.0).to(dtype=dt)
    z_qkv = torch.zeros(bsz, t_len, hv, dk, dtype=dt)
    z_gam = torch.zeros(bsz, t_len, hv, dk // 2, dtype=dt)
    z_th = torch.zeros(bsz, t_len, hv, dk // 2, dtype=dt)
    z_sb = torch.zeros(bsz, t_len, hv, dtype=dt)
    slp = torch.ones(hv, dtype=dt)
    thr = torch.zeros(hv, dtype=dt)
    z_rt = torch.zeros(bsz, t_len, 4, dtype=dt)

    out_flat, _ = chunk_hssd(
        z_qkv, z_qkv, torch.zeros_like(vs), z_gam, z_qkv, torch.zeros_like(vs), z_th,
        z_sb, slp, thr, z_rt, z_rt, qs, ks, vs,
        boltzmann_gate=high_hit_gate,
        swa_recency_slope=torch.zeros(hv, dtype=dt),
        chunk_size=8, window_chunks=1, num_pages=4, top_k_pages=2,
        g_override=torch.zeros(bsz, t_len, hv, dtype=dt),
    )
    out_sloped, _ = chunk_hssd(
        z_qkv, z_qkv, torch.zeros_like(vs), z_gam, z_qkv, torch.zeros_like(vs), z_th,
        z_sb, slp, thr, z_rt, z_rt, qs, ks, vs,
        boltzmann_gate=high_hit_gate,
        swa_recency_slope=torch.full((hv,), 3.0, dtype=dt),
        chunk_size=8, window_chunks=1, num_pages=4, top_k_pages=2,
        g_override=torch.zeros(bsz, t_len, hv, dtype=dt),
    )
    # Flat slope weights t=0 and t=6 identically; positive slope strongly favors newer t=6 (channel 1)
    assert abs(out_flat[0, 7, 0, 0].item() - out_flat[0, 7, 0, 1].item()) < 1e-12
    assert out_sloped[0, 7, 0, 1].item() > 0.85 and out_sloped[0, 7, 0, 0].item() < 0.15


# ==============================================================================
# 4. H4: Read-Ahead Far Readout (1-Bank Checkpoint State + fp64 Backward Parity)
# ==============================================================================
def test_h4_read_ahead_far_readout_checkpoint_single_bank_and_fp64_backward_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bsz, seq_len, hv, dk, dv, m_pages = 2, 96, 4, 12, 12, 8
    c_size, w_chunks, seg_chunks = 8, 2, 2  # 12 chunks -> 6 checkpointed segments

    inp_c = _synth_v32_inputs(
        bsz=bsz, seq_len=seq_len, hv=hv, dk=dk, dv=dv, num_pages=m_pages, seed=404, requires_grad=True
    )
    inp_r = _synth_v32_inputs(
        bsz=bsz, seq_len=seq_len, hv=hv, dk=dk, dv=dv, num_pages=m_pages, seed=404, requires_grad=True
    )
    gate_c = PeakBoltzmannGate(hv, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)
    gate_r = PeakBoltzmannGate(hv, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)

    real_ckpt = hssd_mod.checkpoint
    ckpt_bank_counts: list[int] = []
    ckpt_out_counts: list[int] = []

    def _spy_ckpt(fn: Any, *args: Any, **kwargs: Any) -> Any:
        n_banks = sum(
            1
            for a in args
            if isinstance(a, torch.Tensor) and a.shape == (bsz, m_pages, hv, dk, dv)
        )
        ckpt_bank_counts.append(n_banks)
        res = real_ckpt(fn, *args, **kwargs)
        ckpt_out_counts.append(len(res))
        return res

    monkeypatch.setattr(hssd_mod, "checkpoint", _spy_ckpt)

    out_c, final_c = chunk_hssd(
        inp_c["query"],
        inp_c["key"],
        inp_c["value"],
        inp_c["gamma"],
        inp_c["b_gate"],
        inp_c["w_gate"],
        inp_c["theta"],
        inp_c["surprise_bias"],
        inp_c["surprise_slope"],
        inp_c["surprise_thresh"],
        inp_c["write_logits"],
        inp_c["read_logits"],
        inp_c["q_swa"],
        inp_c["k_swa"],
        inp_c["v_swa"],
        boltzmann_gate=gate_c,
        swa_recency_slope=inp_c["swa_recency_slope"],
        chunk_size=c_size,
        window_chunks=w_chunks,
        num_pages=m_pages,
        top_k_pages=2,
        segment_chunks=seg_chunks,
        checkpoint_segments=True,
        output_final_state=False,
    )
    assert final_c is None
    # H4 invariant: ONLY 1 bank (`p_cur`) enters each segment checkpoint (NOT 1 + W + 1 = 4 or 10 banks!)
    assert ckpt_bank_counts == [1] * 6, f"Expected [1]*6 checkpoint bank args, got {ckpt_bank_counts}"
    assert ckpt_out_counts == [5] * 6, f"Expected 5 tuple outputs (0 saved boundary banks), got {ckpt_out_counts}"

    out_r, _ = recurrent_hssd(
        inp_r["query"],
        inp_r["key"],
        inp_r["value"],
        inp_r["gamma"],
        inp_r["b_gate"],
        inp_r["w_gate"],
        inp_r["theta"],
        inp_r["surprise_bias"],
        inp_r["surprise_slope"],
        inp_r["surprise_thresh"],
        inp_r["write_logits"],
        inp_r["read_logits"],
        inp_r["q_swa"],
        inp_r["k_swa"],
        inp_r["v_swa"],
        boltzmann_gate=gate_r,
        swa_recency_slope=inp_r["swa_recency_slope"],
        chunk_size=c_size,
        window_chunks=w_chunks,
        num_pages=m_pages,
        top_k_pages=2,
        segment_chunks=seg_chunks,
        output_final_state=False,
    )
    assert (out_c - out_r).abs().max().item() < 1e-11

    grad_w = torch.cos(torch.arange(out_c.numel(), dtype=torch.float64).view_as(out_c))
    (out_c * grad_w).sum().backward()
    (out_r * grad_w).sum().backward()

    for key_name in (
        "query",
        "key",
        "value",
        "b_gate",
        "w_gate",
        "theta",
        "surprise_bias",
        "write_logits",
        "read_logits",
        "q_swa",
        "k_swa",
        "v_swa",
        "swa_recency_slope",
    ):
        gc = inp_c[key_name].grad
        gr = inp_r[key_name].grad
        assert gc is not None and gr is not None
        diff = (gc - gr).abs().max().item()
        assert diff < 1e-10, f"Backward grad mismatch on {key_name}: {diff:.3e}"
    assert (inp_c["gamma"][0].grad - inp_r["gamma"][0].grad).abs().max().item() < 1e-10
    assert (inp_c["gamma"][1].grad - inp_r["gamma"][1].grad).abs().max().item() < 1e-10


# ==============================================================================
# 5. Section 5.2 #4, #6, #8: Active Compaction, Segment-Start WY & Compile
# ==============================================================================
def test_speed_52_4_active_page_compaction_fp64_forward_and_backward_parity() -> None:
    inp_a = _synth_v32_inputs(bsz=2, seq_len=64, hv=4, dk=12, dv=12, num_pages=8, seed=501, requires_grad=True)
    inp_b = _synth_v32_inputs(bsz=2, seq_len=64, hv=4, dk=12, dv=12, num_pages=8, seed=501, requires_grad=True)
    # Bias router away from pages 0, 1 so untouched pages & uneven token counts are actively exercised
    bias_skew = torch.tensor([-8.0, -7.0, 0.2, 0.5, 0.1, 0.4, 1.2, 1.5], dtype=torch.float64)
    with torch.no_grad():
        inp_a["write_logits"].add_(bias_skew)
        inp_b["write_logits"].add_(bias_skew)
        inp_a["read_logits"].add_(bias_skew)
        inp_b["read_logits"].add_(bias_skew)

    gate_a = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)
    gate_b = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)

    out_comp, st_comp, duty_comp = chunk_hssd(
        inp_a["query"], inp_a["key"], inp_a["value"], inp_a["gamma"],
        inp_a["b_gate"], inp_a["w_gate"], inp_a["theta"],
        inp_a["surprise_bias"], inp_a["surprise_slope"], inp_a["surprise_thresh"],
        inp_a["write_logits"], inp_a["read_logits"],
        inp_a["q_swa"], inp_a["k_swa"], inp_a["v_swa"],
        boltzmann_gate=gate_a, swa_recency_slope=inp_a["swa_recency_slope"],
        chunk_size=16, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=2,
        checkpoint_segments=False, output_final_state=True,
        compact_active_pages=True, return_aux=True,
    )
    out_dense, st_dense, duty_dense = chunk_hssd(
        inp_b["query"], inp_b["key"], inp_b["value"], inp_b["gamma"],
        inp_b["b_gate"], inp_b["w_gate"], inp_b["theta"],
        inp_b["surprise_bias"], inp_b["surprise_slope"], inp_b["surprise_thresh"],
        inp_b["write_logits"], inp_b["read_logits"],
        inp_b["q_swa"], inp_b["k_swa"], inp_b["v_swa"],
        boltzmann_gate=gate_b, swa_recency_slope=inp_b["swa_recency_slope"],
        chunk_size=16, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=2,
        checkpoint_segments=False, output_final_state=True,
        compact_active_pages=False, return_aux=True,
    )
    assert (out_comp - out_dense).abs().max().item() < 1e-12
    assert (duty_comp - duty_dense).abs().max().item() < 1e-12
    assert st_comp is not None and st_dense is not None and st_comp.state_equal(st_dense, atol=1e-12, rtol=1e-12)

    w_g = torch.sin(torch.arange(out_comp.numel(), dtype=torch.float64).view_as(out_comp))
    ((out_comp * w_g).sum() + duty_comp.sum()).backward()
    ((out_dense * w_g).sum() + duty_dense.sum()).backward()
    for k_nm in ("query", "key", "value", "b_gate", "w_gate", "write_logits", "read_logits"):
        assert (inp_a[k_nm].grad - inp_b[k_nm].grad).abs().max().item() < 1e-11


def test_speed_52_6_8_segment_start_batched_wy_fp64_parity_and_compile_chunks() -> None:
    inp_c = _synth_v32_inputs(bsz=2, seq_len=72, hv=4, dk=12, dv=12, num_pages=8, seed=506, requires_grad=True)
    inp_r = _synth_v32_inputs(bsz=2, seq_len=72, hv=4, dk=12, dv=12, num_pages=8, seed=506, requires_grad=True)
    gate_c = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)
    gate_r = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)

    # 9 chunks of 8, segment_chunks=3 -> 3 batched-WY segments
    out_c, st_c = chunk_hssd(
        inp_c["query"], inp_c["key"], inp_c["value"], inp_c["gamma"],
        inp_c["b_gate"], inp_c["w_gate"], inp_c["theta"],
        inp_c["surprise_bias"], inp_c["surprise_slope"], inp_c["surprise_thresh"],
        inp_c["write_logits"], inp_c["read_logits"],
        inp_c["q_swa"], inp_c["k_swa"], inp_c["v_swa"],
        boltzmann_gate=gate_c, swa_recency_slope=inp_c["swa_recency_slope"],
        chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=3,
        surprise_mode="segment_start", output_final_state=True,
    )
    out_r, st_r = recurrent_hssd(
        inp_r["query"], inp_r["key"], inp_r["value"], inp_r["gamma"],
        inp_r["b_gate"], inp_r["w_gate"], inp_r["theta"],
        inp_r["surprise_bias"], inp_r["surprise_slope"], inp_r["surprise_thresh"],
        inp_r["write_logits"], inp_r["read_logits"],
        inp_r["q_swa"], inp_r["k_swa"], inp_r["v_swa"],
        boltzmann_gate=gate_r, swa_recency_slope=inp_r["swa_recency_slope"],
        chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=3,
        surprise_mode="segment_start", output_final_state=True,
    )
    assert (out_c - out_r).abs().max().item() < 1e-11
    assert st_c is not None and st_r is not None and st_c.state_equal(st_r, atol=1e-11, rtol=1e-11)

    w_g = torch.cos(torch.arange(out_c.numel(), dtype=torch.float64).view_as(out_c))
    (out_c * w_g).sum().backward()
    (out_r * w_g).sum().backward()
    for k_nm in ("query", "key", "value", "write_logits", "read_logits", "surprise_bias"):
        assert (inp_c[k_nm].grad - inp_r[k_nm].grad).abs().max().item() < 1e-10

    # Unaligned mid-segment prefill (T=29) + chunk continuation (T=43) under segment_start
    with torch.no_grad():
        inp_s = _synth_v32_inputs(bsz=1, seq_len=72, hv=4, dk=12, dv=12, num_pages=8, seed=507)
        o_pre, st_pre = chunk_hssd(
            inp_s["query"][:, :29], inp_s["key"][:, :29], inp_s["value"][:, :29],
            (inp_s["gamma"][0], inp_s["gamma"][1][:, :29]),
            inp_s["b_gate"][:, :29], inp_s["w_gate"][:, :29], inp_s["theta"][:, :29],
            inp_s["surprise_bias"][:, :29], inp_s["surprise_slope"], inp_s["surprise_thresh"],
            inp_s["write_logits"][:, :29], inp_s["read_logits"][:, :29],
            inp_s["q_swa"][:, :29], inp_s["k_swa"][:, :29], inp_s["v_swa"][:, :29],
            boltzmann_gate=gate_c, swa_recency_slope=inp_s["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=3,
            surprise_mode="segment_start", output_final_state=True,
        )
        o_cont, _ = chunk_hssd(
            inp_s["query"][:, 29:], inp_s["key"][:, 29:], inp_s["value"][:, 29:],
            (inp_s["gamma"][0], inp_s["gamma"][1][:, 29:]),
            inp_s["b_gate"][:, 29:], inp_s["w_gate"][:, 29:], inp_s["theta"][:, 29:],
            inp_s["surprise_bias"][:, 29:], inp_s["surprise_slope"], inp_s["surprise_thresh"],
            inp_s["write_logits"][:, 29:], inp_s["read_logits"][:, 29:],
            inp_s["q_swa"][:, 29:], inp_s["k_swa"][:, 29:], inp_s["v_swa"][:, 29:],
            boltzmann_gate=gate_c, swa_recency_slope=inp_s["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=3,
            surprise_mode="segment_start", initial_state=st_pre, output_final_state=False,
        )
        o_full_ref, _ = recurrent_hssd(
            inp_s["query"], inp_s["key"], inp_s["value"], inp_s["gamma"],
            inp_s["b_gate"], inp_s["w_gate"], inp_s["theta"],
            inp_s["surprise_bias"], inp_s["surprise_slope"], inp_s["surprise_thresh"],
            inp_s["write_logits"], inp_s["read_logits"],
            inp_s["q_swa"], inp_s["k_swa"], inp_s["v_swa"],
            boltzmann_gate=gate_c, swa_recency_slope=inp_s["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2, segment_chunks=3,
            surprise_mode="segment_start",
        )
        assert (torch.cat([o_pre, o_cont], dim=1) - o_full_ref).abs().max().item() < 1e-11

        # 5.2 #8: compile_chunks=True vs False parity
        inp_f32 = _synth_v32_inputs(bsz=1, seq_len=32, hv=2, dk=8, dv=8, num_pages=4, seed=508, dtype=torch.float32)
        gate_32 = PeakBoltzmannGate(2).to(dtype=torch.float32)
        o_eager, _ = chunk_hssd(
            inp_f32["query"], inp_f32["key"], inp_f32["value"], inp_f32["gamma"],
            inp_f32["b_gate"], inp_f32["w_gate"], inp_f32["theta"],
            inp_f32["surprise_bias"], inp_f32["surprise_slope"], inp_f32["surprise_thresh"],
            inp_f32["write_logits"], inp_f32["read_logits"],
            inp_f32["q_swa"], inp_f32["k_swa"], inp_f32["v_swa"],
            boltzmann_gate=gate_32, swa_recency_slope=inp_f32["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=4, top_k_pages=2, segment_chunks=2,
            compile_chunks=False,
        )
        o_comp_fn, _ = chunk_hssd(
            inp_f32["query"], inp_f32["key"], inp_f32["value"], inp_f32["gamma"],
            inp_f32["b_gate"], inp_f32["w_gate"], inp_f32["theta"],
            inp_f32["surprise_bias"], inp_f32["surprise_slope"], inp_f32["surprise_thresh"],
            inp_f32["write_logits"], inp_f32["read_logits"],
            inp_f32["q_swa"], inp_f32["k_swa"], inp_f32["v_swa"],
            boltzmann_gate=gate_32, swa_recency_slope=inp_f32["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=4, top_k_pages=2, segment_chunks=2,
            compile_chunks=True,
        )
        assert (o_eager - o_comp_fn).abs().max().item() < 1e-5


# ==============================================================================
# 6. H5, H7, H8, H9 & Probes: Compact Cache, Horizon Vault, Zero-Sync, Trig
# ==============================================================================
class _SyncDetectorMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.sync_ops: list[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = str(func)
        if "_local_scalar_dense" in name or "aten.nonzero" in name:
            self.sync_ops.append(name)
        return func(*args, **(kwargs or {}))


def test_h5_h7_h8_h9_compact_decode_cache_horizon_budget_zero_sync_trig_and_probes() -> None:
    # H7: Horizon-scaled vault page write budgets
    cfg_def = HSSDConfig()
    assert cfg_def.effective_mu_max() == DEFAULT_MU_MAX

    cfg_256k = HSSDConfig(horizon_tokens=262_144)
    eff_mu = cfg_256k.effective_mu_max()
    want_vault = 160.0 / 262_144.0
    assert abs(eff_mu[0] - want_vault) < 1e-12
    assert abs(eff_mu[1] - want_vault) < 1e-12
    assert eff_mu[2:] == DEFAULT_MU_MAX[2:]

    # H9: Single-eval 2D rotation trig + 10^6-step float64 phase accuracy
    gen = torch.Generator().manual_seed(609)
    x_rot = torch.randn(2, 5, 4, 16, generator=gen, dtype=torch.float64)
    th_rot = torch.randn(2, 5, 4, 8, generator=gen, dtype=torch.float64)
    cos_t, sin_t = compute_2d_trig(th_rot, x_rot.dtype)
    assert torch.equal(apply_2d_rotation_trig(x_rot, cos_t, sin_t), apply_2d_rotation(x_rot, th_rot))

    dtheta = torch.full((1, 1, 4), 0.123456789012345, dtype=torch.float64)
    phase_acc = torch.zeros((1, 1, 4), dtype=torch.float64)
    # Simulate 1,000,000 steps via 1,000 blocks of 1,000 float64 remainder updates
    step_1k = dtheta * 1000.0
    for _ in range(1000):
        phase_acc = torch.remainder(phase_acc + step_1k, 2.0 * math.pi)
    exact_1m = torch.remainder(dtheta * 1_000_000.0, 2.0 * math.pi)
    v_probe = torch.randn(1, 1, 4, 8, generator=gen, dtype=torch.float64)
    r_iter = apply_2d_rotation(v_probe, phase_acc.unsqueeze(1))
    r_exact = apply_2d_rotation(v_probe, exact_1m.unsqueeze(1))
    assert (r_iter - r_exact).abs().max().item() < 1e-10

    # H5: Compact Lagged-Replay Decode Cache memory footprint (Qwen3.6 geometry: H=32, D=128, V=128, M=8, k=2, bf16)
    st_qwen_ring = HSSDCacheState.zeros(
        1, 8, 32, 128, 128, chunk_size=64, window_chunks=8, compact_decode_cache=False,
        dtype=torch.bfloat16, distinct_ring_buffers=True,
    )
    st_qwen_comp = HSSDCacheState(
        step_count=0,
        chunk_size=64,
        window_chunks=8,
        page_states=torch.zeros(1, 8, 32, 128, 128, dtype=torch.bfloat16),
        theta_cum=torch.zeros(1, 32, 64, dtype=torch.float64),
        boundary_ring=None,
        swa_k_ring=torch.zeros(1, 576, 32, 128, dtype=torch.bfloat16),
        swa_v_ring=torch.zeros(1, 576, 32, 128, dtype=torch.bfloat16),
        swa_valid_ring=torch.zeros(1, 576, dtype=torch.bool),
        s_far_bank=torch.zeros(1, 8, 32, 128, 128, dtype=torch.bfloat16),
        s_chunk_start_bank=torch.zeros(1, 8, 32, 128, 128, dtype=torch.bfloat16),
        replay_buf=(
            torch.zeros(1, 32, 576, 128, dtype=torch.bfloat16),
            torch.zeros(1, 32, 576, 128, dtype=torch.bfloat16),
            torch.zeros(1, 576, 2, 32, dtype=torch.bfloat16),
            torch.zeros(1, 576, 2, 32, 64, dtype=torch.bfloat16),
            torch.zeros(1, 576, 2, dtype=torch.long),
        ),
    )
    ring_bytes = st_qwen_ring.unique_storage_bytes()
    comp_bytes = st_qwen_comp.unique_storage_bytes()
    one_bank_bytes = st_qwen_comp.page_states.numel() * st_qwen_comp.page_states.element_size()
    assert st_qwen_comp.replay_buf is not None
    lagged_comp_bytes = one_bank_bytes + sum(t.numel() * t.element_size() for t in st_qwen_comp.replay_buf)
    lagged_ring_bytes = 8 * one_bank_bytes
    assert lagged_comp_bytes < 0.35 * lagged_ring_bytes, (
        f"Expected ~3x horizon-ring savings (H5), got {lagged_comp_bytes} vs {lagged_ring_bytes}"
    )
    assert comp_bytes < 0.55 * ring_bytes, (
        f"Expected ~2x total cache reduction (H5), got compact={comp_bytes} vs ring={ring_bytes}"
    )

    # H5 + H8: fp64 parity of compact lagged-replay decode vs explicit 9-bank ring across > (W + 3)*C steps
    # AND zero GPU->CPU host syncs (`_SyncDetectorMode`) + static CUDAGraph buffer addresses!
    inp64 = _synth_v32_inputs(bsz=2, seq_len=44, hv=4, dk=12, dv=12, num_pages=8, seed=605)
    gate64 = PeakBoltzmannGate(4, kappa_init=2.0, tau_init=1.0).to(dtype=torch.float64)

    with torch.no_grad():
        # Prefill 14 tokens (unaligned) -> decode 30 tokens step-by-step across chunk_size=8, window_chunks=2
        o_pre_c, st_pre_c = chunk_hssd(
            inp64["query"][:, :14], inp64["key"][:, :14], inp64["value"][:, :14],
            (inp64["gamma"][0], inp64["gamma"][1][:, :14]),
            inp64["b_gate"][:, :14], inp64["w_gate"][:, :14], inp64["theta"][:, :14],
            inp64["surprise_bias"][:, :14], inp64["surprise_slope"], inp64["surprise_thresh"],
            inp64["write_logits"][:, :14], inp64["read_logits"][:, :14],
            inp64["q_swa"][:, :14], inp64["k_swa"][:, :14], inp64["v_swa"][:, :14],
            boltzmann_gate=gate64, swa_recency_slope=inp64["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2,
            output_final_state=True, compact_decode_cache=True,
        )
        o_pre_r, st_pre_r = chunk_hssd(
            inp64["query"][:, :14], inp64["key"][:, :14], inp64["value"][:, :14],
            (inp64["gamma"][0], inp64["gamma"][1][:, :14]),
            inp64["b_gate"][:, :14], inp64["w_gate"][:, :14], inp64["theta"][:, :14],
            inp64["surprise_bias"][:, :14], inp64["surprise_slope"], inp64["surprise_thresh"],
            inp64["write_logits"][:, :14], inp64["read_logits"][:, :14],
            inp64["q_swa"][:, :14], inp64["k_swa"][:, :14], inp64["v_swa"][:, :14],
            boltzmann_gate=gate64, swa_recency_slope=inp64["swa_recency_slope"],
            chunk_size=8, window_chunks=2, num_pages=8, top_k_pages=2,
            output_final_state=True, compact_decode_cache=False,
        )
        assert st_pre_c is not None and st_pre_r is not None
        assert st_pre_c.is_compact_ring is True and st_pre_r.is_compact_ring is False
        assert (o_pre_c - o_pre_r).abs().max().item() < 1e-12

        # Allocate static CUDAGraph-ready state buffer for compact decode
        static_c = HSSDCacheState.empty_like_state(st_pre_c)
        static_c.copy_(st_pre_c)
        ptrs_before = [t.data_ptr() for t in static_c._iter_tensors()]

        cur_r = st_pre_r
        dec_outs_c: list[torch.Tensor] = []
        dec_outs_r: list[torch.Tensor] = []

        with _SyncDetectorMode() as sync_det:
            for t in range(14, 44):
                sl = slice(t, t + 1)
                oc_t, static_c = recurrent_hssd_step(
                    inp64["query"][:, sl], inp64["key"][:, sl], inp64["value"][:, sl],
                    (inp64["gamma"][0], inp64["gamma"][1][:, sl]),
                    inp64["b_gate"][:, sl], inp64["w_gate"][:, sl], inp64["theta"][:, sl],
                    inp64["surprise_bias"][:, sl], inp64["surprise_slope"], inp64["surprise_thresh"],
                    inp64["write_logits"][:, sl], inp64["read_logits"][:, sl],
                    inp64["q_swa"][:, sl], inp64["k_swa"][:, sl], inp64["v_swa"][:, sl],
                    static_c,
                    boltzmann_gate=gate64, swa_recency_slope=inp64["swa_recency_slope"],
                    top_k_pages=2, compact_decode_cache=True, inplace_state=True,
                )
                dec_outs_c.append(oc_t)

        assert sync_det.sync_ops == [], f"Expected zero host sync ops during decode, saw {sync_det.sync_ops}"
        assert [t.data_ptr() for t in static_c._iter_tensors()] == ptrs_before

        for t in range(14, 44):
            sl = slice(t, t + 1)
            or_t, cur_r = recurrent_hssd_step(
                inp64["query"][:, sl], inp64["key"][:, sl], inp64["value"][:, sl],
                (inp64["gamma"][0], inp64["gamma"][1][:, sl]),
                inp64["b_gate"][:, sl], inp64["w_gate"][:, sl], inp64["theta"][:, sl],
                inp64["surprise_bias"][:, sl], inp64["surprise_slope"], inp64["surprise_thresh"],
                inp64["write_logits"][:, sl], inp64["read_logits"][:, sl],
                inp64["q_swa"][:, sl], inp64["k_swa"][:, sl], inp64["v_swa"][:, sl],
                cur_r,
                boltzmann_gate=gate64, swa_recency_slope=inp64["swa_recency_slope"],
                top_k_pages=2, compact_decode_cache=False, inplace_state=False,
            )
            dec_outs_r.append(or_t)

        assert (torch.cat(dec_outs_c, dim=1) - torch.cat(dec_outs_r, dim=1)).abs().max().item() < 1e-12
        # Synthesized `.boundary_ring` on compact cache matches explicit 9-entry ring to < 1e-12!
        for b_comp, b_ring in zip(static_c.boundary_ring, cur_r.boundary_ring):
            assert (b_comp - b_ring).abs().max().item() < 1e-12

    # Probes: profile_hssd_layer(..., compare_tf32=True) & measure_hssd_tf32_drift
    prof = profile_hssd_layer(
        hidden_size=32,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        seq_len=32,
        stream_chunk_len=16,
        rank=8,
        compare_tf32=True,
        drift_seq_len=32,
    )
    for req_key in (
        "allow_tf32",
        "compact_active_pages",
        "compact_decode_cache",
        "decode_state_bytes",
        "tokens_per_sec",
        "tf32_comparison",
    ):
        assert req_key in prof, f"Missing {req_key} in profile_hssd_layer output"
    assert prof["tokens_per_sec"] > 0.0
    assert prof["decode_state_bytes"] > 0
    assert "drift" in prof["tf32_comparison"] and "prefill_speedup" in prof["tf32_comparison"]

    drift = measure_hssd_tf32_drift(
        hidden_size=32,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        seq_len=32,
        stream_chunk_len=16,
        rank=8,
    )
    assert math.isfinite(drift["tail_output_max_abs_diff"]) and math.isfinite(drift["rel_page_state_l2_drift"])
