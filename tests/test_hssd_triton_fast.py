"""Unit and CUDA parity tests for HSSD v3.2 Triton fast path and Root Cause A/B optimizations."""
from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import qloop.model.hssd as hssd_mod
from qloop.model.hssd import (
    HSSDAdapter,
    HSSDConfig,
    PeakBoltzmannGate,
    _build_active_page_plan,
    _build_multi_span_active_plans,
    chunk_hssd,
    recurrent_hssd,
    recurrent_hssd_step,
)


def test_active_page_plan_helpers_remain_intact_cpu() -> None:
    """Verify `_build_active_page_plan` and `_build_multi_span_active_plans` stay intact for CPU & large B*M."""
    act_bm = torch.zeros((2, 8, 16), dtype=torch.bool)
    act_bm[0, 1, 3:7] = True
    act_bm[1, 6, 0:2] = True
    plan = _build_active_page_plan(act_bm)
    assert plan.n_act == 2
    assert plan.max_tok == 4
    assert plan.lin_bm is not None and plan.lin_bm.tolist() == [1, 14]

    t_idx_0 = torch.zeros((2, 16, 2), dtype=torch.long)
    t_idx_1 = torch.full((2, 16, 2), 7, dtype=torch.long)
    plans = _build_multi_span_active_plans([t_idx_0, t_idx_1], num_pages=8)
    assert len(plans) == 2
    assert plans[0].n_act == 2 and plans[1].n_act == 2


def test_fp64_chunk_vs_recurrent_parity_and_gradcheck_preserved() -> None:
    """Verify `fp64` chunk_hssd forward & backward stay bit-accurate (`< 1e-12`) against `recurrent_hssd`."""
    g = torch.Generator().manual_seed(202611)
    bsz, seq_len, hv, dk, dv, num_pages, k_pages = 2, 40, 2, 8, 8, 6, 2

    def _mk() -> dict[str, Any]:
        gen = torch.Generator().manual_seed(202611)
        return {
            "query": torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64, requires_grad=True),
            "key": F.normalize(
                torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64), dim=-1
            ).detach().requires_grad_(True),
            "value": torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64, requires_grad=True),
            "ps": torch.linspace(0.005, 0.05, num_pages, dtype=torch.float64, requires_grad=True),
            "ch": (0.01 + 0.03 * torch.rand(bsz, seq_len, hv, dk // 2, generator=gen, dtype=torch.float64))
            .detach()
            .requires_grad_(True),
            "bg": (0.2 + 1.5 * torch.rand(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64))
            .detach()
            .requires_grad_(True),
            "wg": (0.1 + 0.8 * torch.rand(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64))
            .detach()
            .requires_grad_(True),
            "th": (0.25 * torch.randn(bsz, seq_len, hv, dk // 2, generator=gen, dtype=torch.float64))
            .detach()
            .requires_grad_(True),
            "sb": (0.3 * torch.randn(bsz, seq_len, hv, generator=gen, dtype=torch.float64))
            .detach()
            .requires_grad_(True),
            "slp": torch.full((hv,), 3.0, dtype=torch.float64, requires_grad=True),
            "thr": torch.full((hv,), 0.75, dtype=torch.float64, requires_grad=True),
            "wl": torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=torch.float64, requires_grad=True),
            "rl": torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=torch.float64, requires_grad=True),
            "qs": torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64, requires_grad=True),
            "ks": torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64, requires_grad=True),
            "vs": torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64, requires_grad=True),
            "gate": PeakBoltzmannGate(hv).double(),
            "sw_slp": torch.linspace(0.02, 0.08, hv, dtype=torch.float64, requires_grad=True),
        }

    d_c = _mk()
    d_r = _mk()
    grad_o = torch.randn(bsz, seq_len, hv, dv, generator=g, dtype=torch.float64)

    out_c, st_c, du_c = chunk_hssd(
        d_c["query"],
        d_c["key"],
        d_c["value"],
        (d_c["ps"], d_c["ch"]),
        d_c["bg"],
        d_c["wg"],
        d_c["th"],
        d_c["sb"],
        d_c["slp"],
        d_c["thr"],
        d_c["wl"],
        d_c["rl"],
        d_c["qs"],
        d_c["ks"],
        d_c["vs"],
        boltzmann_gate=d_c["gate"],
        swa_recency_slope=d_c["sw_slp"],
        chunk_size=16,
        window_chunks=1,
        num_pages=num_pages,
        top_k_pages=k_pages,
        segment_chunks=2,
        output_final_state=True,
        return_aux=True,
    )
    ((out_c * grad_o).sum() + du_c.sum()).backward()

    out_r, st_r, du_r = recurrent_hssd(
        d_r["query"],
        d_r["key"],
        d_r["value"],
        (d_r["ps"], d_r["ch"]),
        d_r["bg"],
        d_r["wg"],
        d_r["th"],
        d_r["sb"],
        d_r["slp"],
        d_r["thr"],
        d_r["wl"],
        d_r["rl"],
        d_r["qs"],
        d_r["ks"],
        d_r["vs"],
        boltzmann_gate=d_r["gate"],
        swa_recency_slope=d_r["sw_slp"],
        chunk_size=16,
        window_chunks=1,
        num_pages=num_pages,
        top_k_pages=k_pages,
        segment_chunks=2,
        output_final_state=True,
        return_aux=True,
    )
    ((out_r * grad_o).sum() + du_r.sum()).backward()

    assert st_c is not None and st_r is not None
    assert float((out_c.detach() - out_r.detach()).abs().max()) < 1e-12
    assert float((st_c.page_states.detach() - st_r.page_states.detach()).abs().max()) < 1e-12
    for key_n in ("query", "key", "value", "ps", "ch", "bg", "wg", "th", "sb", "wl", "rl", "qs", "ks", "vs"):
        assert float((d_c[key_n].grad - d_r[key_n].grad).abs().max()) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fast-path parity tests")
@pytest.mark.parametrize(
    ("seq_len", "surprise_mode", "write_mode", "use_wm"),
    [
        (300, "routed", "full", False),
        (640, "routed", "full", True),
        (640, "segment_start", "full", False),
        (200, "page0", "split", False),
        (200, "none", "full", False),
    ],
)
def test_cuda_triton_fast_path_fwd_bwd_and_outer_ckpt_parity(
    seq_len: int,
    surprise_mode: str,
    write_mode: str,
    use_wm: bool,
) -> None:
    """Verify CUDA BF16 Triton fast path (incl. VirtualLayer outer checkpoint) matches pure FP32 reference."""
    assert hssd_mod._HAS_TRITON
    device = torch.device("cuda")
    cfg = HSSDConfig(
        rank=32,
        chunk_size=64,
        window_chunks=2,
        num_pages=8,
        k_pages=2,
        segment_chunks=4,
        surprise_mode=surprise_mode,
        write_mode=write_mode,
        state_dtype="bfloat16",
    )
    ad_ref = HSSDAdapter(256, 8, 64, 64, cfg, 0).to(device=device, dtype=torch.float32)
    ad_tri = HSSDAdapter(256, 8, 64, 64, cfg, 0).to(device=device, dtype=torch.float32)
    ad_tri.load_state_dict(ad_ref.state_dict())

    torch.manual_seed(4200 + seq_len)
    x_r = torch.randn(1, seq_len, 256, device=device, dtype=torch.float32, requires_grad=True)
    q_r = torch.randn(1, seq_len, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    k_r = torch.randn(1, seq_len, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
    v_r = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float32, requires_grad=True)
    g_r = -F.softplus(torch.randn(1, seq_len, 8, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    b_r = torch.sigmoid(torch.randn(1, seq_len, 8, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    wm = (torch.rand(1, seq_len, device=device) > 0.25).float() if use_wm else None

    x_t = x_r.detach().to(torch.bfloat16).requires_grad_(True)
    q_t = q_r.detach().to(torch.bfloat16).requires_grad_(True)
    k_t = k_r.detach().to(torch.bfloat16).requires_grad_(True)
    v_t = v_r.detach().to(torch.bfloat16).requires_grad_(True)
    g_t = g_r.detach().clone().requires_grad_(True)
    b_t = b_r.detach().to(torch.bfloat16).requires_grad_(True)

    pr_r = ad_ref.project_inputs(x_r, q_r, k_r, v_r, g_r, b_r, use_qk_l2norm=True)
    out_r, _, du_r = chunk_hssd(
        pr_r["query"],
        pr_r["key"],
        pr_r["value"],
        (pr_r["gamma_page_scale"], pr_r["gamma_chan_pair"]),
        pr_r["b_gate"],
        pr_r["w_gate"],
        pr_r["theta"],
        pr_r["surprise_bias"],
        ad_ref.log_surprise_slope.exp(),
        ad_ref.surprise_thresh,
        pr_r["write_logits"],
        pr_r["read_logits"],
        pr_r["q_swa"],
        pr_r["k_swa"],
        pr_r["v_swa"],
        boltzmann_gate=ad_ref.boltzmann_gate,
        swa_recency_slope=ad_ref.swa_recency_slope,
        chunk_size=64,
        window_chunks=2,
        num_pages=8,
        top_k_pages=2,
        segment_chunks=4,
        write_mask=wm,
        surprise_mode=surprise_mode,
        write_mode=write_mode,
        return_aux=True,
    )
    grad_o = torch.randn_like(out_r)
    ((out_r * grad_o).sum() + du_r.sum()).backward()

    def _wrapped_student(xt: torch.Tensor, qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor):
        pr_t = ad_tri.project_inputs(xt, qt, kt, vt, g_t, b_t, use_qk_l2norm=True)
        o_t, _, d_t = chunk_hssd(
            pr_t["query"],
            pr_t["key"],
            pr_t["value"],
            (pr_t["gamma_page_scale"], pr_t["gamma_chan_pair"]),
            pr_t["b_gate"],
            pr_t["w_gate"],
            pr_t["theta"],
            pr_t["surprise_bias"],
            ad_tri.log_surprise_slope.exp(),
            ad_tri.surprise_thresh,
            pr_t["write_logits"],
            pr_t["read_logits"],
            pr_t["q_swa"],
            pr_t["k_swa"],
            pr_t["v_swa"],
            boltzmann_gate=ad_tri.boltzmann_gate,
            swa_recency_slope=ad_tri.swa_recency_slope,
            chunk_size=64,
            window_chunks=2,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=4,
            write_mask=wm,
            surprise_mode=surprise_mode,
            write_mode=write_mode,
            return_aux=True,
        )
        return o_t, d_t

    out_t, du_t = checkpoint(_wrapped_student, x_t, q_t, k_t, v_t, use_reentrant=False)
    ((out_t.float() * grad_o).sum() + du_t.sum()).backward()

    assert torch.isfinite(out_t).all()
    assert F.cosine_similarity(out_r.flatten(), out_t.float().flatten(), dim=0).item() > 0.9995
    assert F.cosine_similarity(v_r.grad.flatten(), v_t.grad.float().flatten(), dim=0).item() > 0.999
    if float(x_r.grad.norm()) > 1e-4:
        assert F.cosine_similarity(x_r.grad.flatten(), x_t.grad.float().flatten(), dim=0).item() > 0.999
    else:
        assert float((x_r.grad - x_t.grad.float()).abs().max()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for BF16 prefill+decode parity test")
@pytest.mark.parametrize("compact_decode_cache", [True, False])
def test_cuda_triton_prefill_to_recurrent_decode_parity(compact_decode_cache: bool) -> None:
    """Verify BF16 Triton prefill (`output_final_state=True`) hands off cleanly to `recurrent_hssd_step`."""
    device = torch.device("cuda")
    cfg = HSSDConfig(
        rank=32,
        chunk_size=64,
        window_chunks=4,
        num_pages=8,
        k_pages=2,
        segment_chunks=4,
        state_dtype="bfloat16",
        compact_decode_cache=compact_decode_cache,
    )
    ad = HSSDAdapter(256, 8, 64, 64, cfg, 0).to(device=device, dtype=torch.float32).eval()
    torch.manual_seed(9100)
    t_pre, t_dec = 340, 5
    t_tot = t_pre + t_dec
    x = torch.randn(1, t_tot, 256, device=device, dtype=torch.bfloat16)
    q = torch.randn(1, t_tot, 4, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, t_tot, 4, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, t_tot, 8, 64, device=device, dtype=torch.bfloat16)
    g = -F.softplus(torch.randn(1, t_tot, 8, device=device, dtype=torch.float32))
    b = torch.sigmoid(torch.randn(1, t_tot, 8, device=device, dtype=torch.bfloat16))

    with torch.no_grad():
        pr = ad.project_inputs(x, q, k, v, g, b, use_qk_l2norm=True)
        out_rec, _ = recurrent_hssd(
            pr["query"],
            pr["key"],
            pr["value"],
            (pr["gamma_page_scale"], pr["gamma_chan_pair"]),
            pr["b_gate"],
            pr["w_gate"],
            pr["theta"],
            pr["surprise_bias"],
            ad.log_surprise_slope.exp(),
            ad.surprise_thresh,
            pr["write_logits"],
            pr["read_logits"],
            pr["q_swa"],
            pr["k_swa"],
            pr["v_swa"],
            boltzmann_gate=ad.boltzmann_gate,
            swa_recency_slope=ad.swa_recency_slope,
            chunk_size=64,
            window_chunks=4,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=4,
            output_final_state=True,
            state_dtype="bfloat16",
            compact_decode_cache=compact_decode_cache,
        )

        pr_pre = {kn: (vt[:, :t_pre] if vt.ndim >= 2 else vt) for kn, vt in pr.items()}
        out_p, st = chunk_hssd(
            pr_pre["query"],
            pr_pre["key"],
            pr_pre["value"],
            (pr_pre["gamma_page_scale"], pr_pre["gamma_chan_pair"]),
            pr_pre["b_gate"],
            pr_pre["w_gate"],
            pr_pre["theta"],
            pr_pre["surprise_bias"],
            ad.log_surprise_slope.exp(),
            ad.surprise_thresh,
            pr_pre["write_logits"],
            pr_pre["read_logits"],
            pr_pre["q_swa"],
            pr_pre["k_swa"],
            pr_pre["v_swa"],
            boltzmann_gate=ad.boltzmann_gate,
            swa_recency_slope=ad.swa_recency_slope,
            chunk_size=64,
            window_chunks=4,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=4,
            output_final_state=True,
            state_dtype="bfloat16",
            compact_decode_cache=compact_decode_cache,
        )
        pieces = [out_p]
        for idx in range(t_pre, t_tot):
            sl = slice(idx, idx + 1)
            o_i, st = recurrent_hssd_step(
                pr["query"][:, sl],
                pr["key"][:, sl],
                pr["value"][:, sl],
                (pr["gamma_page_scale"], pr["gamma_chan_pair"][:, sl]),
                pr["b_gate"][:, sl],
                pr["w_gate"][:, sl],
                pr["theta"][:, sl],
                pr["surprise_bias"][:, sl],
                ad.log_surprise_slope.exp(),
                ad.surprise_thresh,
                pr["write_logits"][:, sl],
                pr["read_logits"][:, sl],
                pr["q_swa"][:, sl],
                pr["k_swa"][:, sl],
                pr["v_swa"][:, sl],
                st,
                boltzmann_gate=ad.boltzmann_gate,
                swa_recency_slope=ad.swa_recency_slope,
                top_k_pages=2,
                segment_chunks=4,
                state_dtype="bfloat16",
                compact_decode_cache=compact_decode_cache,
            )
            pieces.append(o_i)
        out_full = torch.cat(pieces, dim=1)
        assert F.cosine_similarity(out_rec.float().flatten(), out_full.float().flatten(), dim=0).item() > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for aligned-key WY stability regression")
def test_cuda_triton_aligned_keys_no_neumann_blowup() -> None:
    """Regression: collinear consecutive keys (`||L|| >> 1`) must invert `(I + L)` without Neumann cancellation."""
    device = torch.device("cuda")
    torch.manual_seed(777)
    bsz, seq_len, hv, dk, dv = 1, 4096, 4, 128, 128
    base_dir = F.normalize(torch.randn(bsz, 1, hv, dk, device=device, dtype=torch.float32), dim=-1)
    k_aligned = F.normalize(base_dir + 0.02 * torch.randn(bsz, seq_len, hv, dk, device=device), dim=-1)
    q_r = F.normalize(base_dir + 0.1 * torch.randn(bsz, seq_len, hv, dk, device=device), dim=-1).requires_grad_(True)
    k_r = k_aligned.detach().clone().requires_grad_(True)
    v_r = (5.0 * torch.randn(bsz, seq_len, hv, dv, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    ps = torch.linspace(0.002, 1.0, 8, device=device, dtype=torch.float32)
    ch = torch.full((bsz, seq_len, hv, dk // 2), 1e-4, device=device, dtype=torch.float32)
    bg_r = torch.full((bsz, seq_len, hv, dk), 1.85, device=device, dtype=torch.float32, requires_grad=True)
    wg_r = torch.ones((bsz, seq_len, hv, dv), device=device, dtype=torch.float32, requires_grad=True)
    th = torch.zeros((bsz, seq_len, hv, dk // 2), device=device, dtype=torch.float32)
    sb = torch.full((bsz, seq_len, hv), 2.0, device=device, dtype=torch.float32)
    slp = torch.full((hv,), 3.0, device=device, dtype=torch.float32)
    thr = torch.full((hv,), 0.85, device=device, dtype=torch.float32)
    wl = torch.linspace(-0.5, 0.5, 8, device=device).view(1, 1, 8).expand(bsz, seq_len, 8).clone()
    rl = wl.clone()
    qs_r = q_r.detach().clone()
    ks_r = k_r.detach().clone()
    vs_r = v_r.detach().clone()

    out_ref, _ = chunk_hssd(
        q_r, k_r, v_r, (ps, ch), bg_r, wg_r, th, sb, slp, thr, wl, rl, qs_r, ks_r, vs_r,
        chunk_size=64, window_chunks=8, num_pages=8, top_k_pages=2, segment_chunks=4,
    )
    out_ref.sum().backward()

    q_t = q_r.detach().to(torch.bfloat16).requires_grad_(True)
    k_t = k_r.detach().to(torch.bfloat16).requires_grad_(True)
    v_t = v_r.detach().to(torch.bfloat16).requires_grad_(True)
    bg_t = bg_r.detach().to(torch.bfloat16).requires_grad_(True)
    wg_t = wg_r.detach().to(torch.bfloat16).requires_grad_(True)
    out_tri, _ = chunk_hssd(
        q_t, k_t, v_t, (ps, ch), bg_t, wg_t, th, sb, slp, thr, wl, rl,
        qs_r.to(torch.bfloat16), ks_r.to(torch.bfloat16), vs_r.to(torch.bfloat16),
        chunk_size=64, window_chunks=8, num_pages=8, top_k_pages=2, segment_chunks=4,
    )
    out_tri.float().sum().backward()

    assert torch.isfinite(out_tri).all()
    assert torch.isfinite(k_t.grad).all() and torch.isfinite(v_t.grad).all()
    assert F.cosine_similarity(out_ref.flatten(), out_tri.float().flatten(), dim=0).item() > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sync_debug_mode('error') test")
def test_cuda_zero_host_sync_under_sync_debug_error() -> None:
    """Ensure `project_inputs` + `chunk_hssd(..., return_aux=True)` fwd+bwd triggers 0 CUDA host syncs."""
    device = torch.device("cuda")
    cfg = HSSDConfig(rank=32, chunk_size=64, window_chunks=2, num_pages=8, k_pages=2, segment_chunks=4)
    ad = HSSDAdapter(256, 8, 64, 64, cfg, 0).to(device=device, dtype=torch.float32)
    x = torch.randn(1, 256, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    q = torch.randn(1, 256, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 256, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 256, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    g = -F.softplus(torch.randn(1, 256, 8, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    b = torch.sigmoid(torch.randn(1, 256, 8, device=device, dtype=torch.bfloat16)).detach().requires_grad_(True)

    def _step() -> None:
        ad.zero_grad(set_to_none=True)
        x.grad = q.grad = k.grad = v.grad = g.grad = b.grad = None
        pr = ad.project_inputs(x, q, k, v, g, b, use_qk_l2norm=True)
        out, _, duty = chunk_hssd(
            pr["query"],
            pr["key"],
            pr["value"],
            (pr["gamma_page_scale"], pr["gamma_chan_pair"]),
            pr["b_gate"],
            pr["w_gate"],
            pr["theta"],
            pr["surprise_bias"],
            ad.log_surprise_slope.exp(),
            ad.surprise_thresh,
            pr["write_logits"],
            pr["read_logits"],
            pr["q_swa"],
            pr["k_swa"],
            pr["v_swa"],
            boltzmann_gate=ad.boltzmann_gate,
            swa_recency_slope=ad.swa_recency_slope,
            chunk_size=64,
            window_chunks=2,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=4,
            return_aux=True,
        )
        (out.sum() + duty.sum()).backward()

    # Warm up Triton compilation outside sync_debug_mode("error")
    _step()
    torch.cuda.synchronize()

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _step()
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for differentiable write_mask parity test")
@pytest.mark.parametrize(("use_surprise", "nonzero_theta"), [(True, False), (True, True), (False, False)])
def test_cuda_differentiable_write_mask_grad_parity(use_surprise: bool, nonzero_theta: bool) -> None:
    """Ensure `write_mask.requires_grad=True` propagates exact `g_root`, `theta`, and `aux` gradients on CUDA BF16."""
    device = torch.device("cuda")
    torch.manual_seed(5150)
    bsz, seq_len, hv, dk, dv = 1, 256, 4, 64, 64
    q_r = F.normalize(torch.randn(bsz, seq_len, hv, dk, device=device, dtype=torch.float32), dim=-1)
    k_r = F.normalize(torch.randn(bsz, seq_len, hv, dk, device=device, dtype=torch.float32), dim=-1)
    v_r = torch.randn(bsz, seq_len, hv, dv, device=device, dtype=torch.float32)
    ps = torch.linspace(0.005, 0.5, 8, device=device, dtype=torch.float32)
    ch = torch.full((bsz, seq_len, hv, dk // 2), 0.02, device=device, dtype=torch.float32)
    bg = torch.full((bsz, seq_len, hv, dk), 0.8, device=device, dtype=torch.float32)
    wg = torch.full((bsz, seq_len, hv, dv), 0.8, device=device, dtype=torch.float32)
    th = (
        0.2 * torch.randn(bsz, seq_len, hv, dk // 2, device=device, dtype=torch.float32)
        if nonzero_theta
        else torch.zeros((bsz, seq_len, hv, dk // 2), device=device, dtype=torch.float32)
    )
    sb = torch.zeros((bsz, seq_len, hv), device=device, dtype=torch.float32)
    slp = torch.full((hv,), 3.0, device=device, dtype=torch.float32)
    thr = torch.full((hv,), 0.8, device=device, dtype=torch.float32)
    wl = torch.randn(bsz, seq_len, 8, device=device, dtype=torch.float32)
    rl = torch.randn(bsz, seq_len, 8, device=device, dtype=torch.float32)
    wm_base = (0.6 + 0.35 * torch.rand(bsz, seq_len, device=device, dtype=torch.float32))
    grad_o = torch.randn(bsz, seq_len, hv, dv, device=device, dtype=torch.float32)

    wm_r = wm_base.detach().clone().requires_grad_(True)
    out_r, _, du_r = chunk_hssd(
        q_r, k_r, v_r, (ps, ch), bg, wg, th, sb, slp, thr, wl, rl, q_r, k_r, v_r,
        chunk_size=64, window_chunks=1, num_pages=8, top_k_pages=2, segment_chunks=2,
        write_mask=wm_r, use_surprise=use_surprise, return_aux=True,
    )
    ((out_r * grad_o).sum() + 0.5 * du_r.sum()).backward()

    wm_t = wm_base.detach().clone().requires_grad_(True)
    out_t, _, du_t = chunk_hssd(
        q_r.to(torch.bfloat16), k_r.to(torch.bfloat16), v_r.to(torch.bfloat16),
        (ps, ch), bg.to(torch.bfloat16), wg.to(torch.bfloat16), th, sb, slp, thr, wl, rl,
        q_r.to(torch.bfloat16), k_r.to(torch.bfloat16), v_r.to(torch.bfloat16),
        chunk_size=64, window_chunks=1, num_pages=8, top_k_pages=2, segment_chunks=2,
        write_mask=wm_t, use_surprise=use_surprise, return_aux=True,
    )
    ((out_t.float() * grad_o).sum() + 0.5 * du_t.sum()).backward()

    assert wm_t.grad is not None and wm_r.grad is not None
    assert float(wm_r.grad.norm()) > 0.1
    assert F.cosine_similarity(wm_r.grad.flatten(), wm_t.grad.float().flatten(), dim=0).item() > 0.999


def test_hssd_hybrid_cache_init_transformers_v5_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `HSSDHybridCache` handles `transformers >= 5.x` `Cache.__init__` raising `ValueError`."""
    class _FakeV5Cache:
        def __init__(self) -> None:
            raise ValueError("You should provide exactly one of layers or layer_class_to_replicate")

    monkeypatch.setattr(hssd_mod, "_HFCacheBase", _FakeV5Cache)
    cache = hssd_mod.HSSDHybridCache(num_layers=4)
    assert len(cache.recurrent_states) == 4


def test_phase2_theta_horizon_jacobian_and_blend_backward_stability() -> None:
    """Regression test for s1b Phase-2 multi-layer gradient explosion root causes:
    1) `theta` cumulative-phase Jacobian (`||cumsum_T||_2 = O(T)`) stays O(1) across `T = 64 -> 512`
       while keeping forward outputs bit-identical.
    2) Scheduled output blend (`0 <= lam < 1`) preserves live `(1 - lam) * d(stock)/dx + lam * d(hssd)/dx`
       residual Jacobians without `25x` `max(lam, 0.1)` inflation.
    3) Subnormal `beta` (`1e-38`) and dead-conv near-zero `q/k` (`silu(-25)`) produce finite bounded grads.
    """
    torch.manual_seed(9001)
    cfg = HSSDConfig(
        rank=16,
        chunk_size=16,
        window_chunks=2,
        num_pages=4,
        k_pages=2,
        theta_low_rank=True,
        segment_chunks=2,
        checkpoint_segments=False,
        blend_init=0.0,
        blend_warmup_steps=250,
    )
    ad = HSSDAdapter(64, 4, 16, 16, cfg, 0)
    with torch.no_grad():
        assert ad.theta_up is not None
        ad.theta_up.weight.normal_(0.0, 0.25)
        ad.swa_v_up.weight.normal_(0.0, 0.05)

    # 1) Compare per-token normalized residual Jacobian through theta at T=64 vs T=512 (8x longer)
    dx_norms: list[float] = []
    for seq_len in (64, 512):
        ad.zero_grad(set_to_none=True)
        x = torch.randn(1, seq_len, 64, requires_grad=True)
        q = torch.randn(1, seq_len, 2, 16)
        k = torch.randn(1, seq_len, 2, 16)
        v = torch.randn(1, seq_len, 4, 16)
        g = -0.001 * torch.ones(1, seq_len, 4)
        b = 0.5 * torch.ones(1, seq_len, 4)
        pr = ad.project_inputs(x, q, k, v, g, b, use_qk_l2norm=True)
        out, _, _ = chunk_hssd(
            pr["query"],
            pr["key"],
            pr["value"],
            (pr["gamma_page_scale"], pr["gamma_chan_pair"]),
            pr["b_gate"],
            pr["w_gate"],
            pr["theta"],
            pr["surprise_bias"],
            ad.log_surprise_slope.exp(),
            ad.surprise_thresh,
            pr["write_logits"],
            pr["read_logits"],
            pr["q_swa"],
            pr["k_swa"],
            pr["v_swa"],
            boltzmann_gate=ad.boltzmann_gate,
            swa_recency_slope=ad.swa_recency_slope,
            chunk_size=16,
            window_chunks=2,
            num_pages=4,
            top_k_pages=2,
            segment_chunks=2,
            return_aux=True,
        )
        u = torch.randn_like(out)
        u = u / u.norm()
        (out * u).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        dx_norms.append(float(x.grad.norm().item()))
    # Without chunk_size / seq_len normalization, dx_norms[1] grew ~8x over dx_norms[0]; now stays bounded.
    assert dx_norms[1] < 2.0 * dx_norms[0]

    # 2) Subnormal beta (1e-38 in bf16) + dead-conv silu(-25) q/k must stay 100% finite & bounded
    ad.zero_grad(set_to_none=True)
    x_edge = torch.randn(1, 32, 64, dtype=torch.bfloat16, requires_grad=True)
    q_edge = F.silu(torch.full((1, 32, 2, 16), -25.0, dtype=torch.bfloat16)).detach().requires_grad_(True)
    k_edge = F.silu(torch.full((1, 32, 2, 16), -25.0, dtype=torch.bfloat16)).detach().requires_grad_(True)
    v_edge = torch.randn(1, 32, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    g_edge = torch.full((1, 32, 4), -0.1, dtype=torch.float32, requires_grad=True)
    b_edge = torch.full((1, 32, 4), 1.5e-38, dtype=torch.bfloat16, requires_grad=True)
    pr_e = ad.project_inputs(x_edge, q_edge, k_edge, v_edge, g_edge, b_edge, use_qk_l2norm=True)
    out_e, _, du_e = chunk_hssd(
        pr_e["query"],
        pr_e["key"],
        pr_e["value"],
        (pr_e["gamma_page_scale"], pr_e["gamma_chan_pair"]),
        pr_e["b_gate"],
        pr_e["w_gate"],
        pr_e["theta"],
        pr_e["surprise_bias"],
        ad.log_surprise_slope.exp(),
        ad.surprise_thresh,
        pr_e["write_logits"],
        pr_e["read_logits"],
        pr_e["q_swa"],
        pr_e["k_swa"],
        pr_e["v_swa"],
        boltzmann_gate=ad.boltzmann_gate,
        swa_recency_slope=ad.swa_recency_slope,
        chunk_size=16,
        window_chunks=2,
        num_pages=4,
        top_k_pages=2,
        segment_chunks=2,
        return_aux=True,
    )
    (out_e.float().sum() + du_e.sum()).backward()
    for t_chk in (x_edge, q_edge, k_edge, v_edge, g_edge, b_edge, *ad.parameters()):
        if t_chk.grad is not None:
            assert torch.isfinite(t_chk.grad).all()

    # 3) Scheduled blend backward consistency at lam = 0.004 (no detached stock_out or 25x lam_eff=0.1 jump)
    from qloop.model.toy import ToyConfig, ToyForCausalLM

    tcfg = ToyConfig(
        vocab_size=32,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "linear_attention"),
    )
    stock_m = ToyForCausalLM(tcfg).eval()
    hssd_m = ToyForCausalLM(tcfg).eval()
    hssd_m.load_state_dict(stock_m.state_dict())
    h_handle = hssd_mod.apply_hssd(
        hssd_m,
        HSSDConfig(
            rank=8,
            chunk_size=8,
            window_chunks=1,
            num_pages=4,
            k_pages=2,
            blend_init=0.0,
            blend_warmup_steps=250,
        ),
    )
    for ad_i in h_handle.adapters.values():
        with torch.no_grad():
            ad_i.swa_v_up.weight.normal_(0.0, 0.2)
    ids = torch.randint(0, 32, (1, 16))
    emb_s = stock_m.model.embed_tokens(ids).detach().requires_grad_(True)
    logits_s = stock_m.lm_head(stock_m.model(inputs_embeds=emb_s).last_hidden_state)
    logits_s.pow(2).mean().backward()

    h_handle.set_blend(0.004)
    emb_h = hssd_m.model.embed_tokens(ids).detach().requires_grad_(True)
    logits_h = hssd_m.lm_head(hssd_m.model(inputs_embeds=emb_h).last_hidden_state)
    logits_h.pow(2).mean().backward()
    assert emb_s.grad is not None and emb_h.grad is not None
    assert F.cosine_similarity(emb_s.grad.flatten(), emb_h.grad.flatten(), dim=0).item() > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for window_chunks=8 Triton norm parity test")
def test_cuda_triton_window8_norm_and_edge_case_grad_parity() -> None:
    """Verify exact gradient norm & cosine parity (including dq, dk, dtheta_up, dswa_v_up) at window_chunks=8."""
    device = torch.device("cuda")
    cfg = HSSDConfig(
        rank=64,
        chunk_size=64,
        window_chunks=8,
        num_pages=8,
        k_pages=2,
        theta_low_rank=True,
        surprise_mode="routed",
        write_mode="full",
        segment_chunks=4,
    )
    torch.manual_seed(2026)
    ad_ref = HSSDAdapter(2048, 32, 128, 128, cfg, 0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        assert ad_ref.theta_up is not None
        ad_ref.theta_up.weight.normal_(0.0, 0.05)
        ad_ref.swa_v_up.weight.normal_(0.0, 0.02)
    ad_tri = HSSDAdapter(2048, 32, 128, 128, cfg, 0).to(device=device, dtype=torch.float32)
    ad_tri.load_state_dict(ad_ref.state_dict())

    seq_len = 1024
    x_0 = torch.randn(1, seq_len, 2048, device=device, dtype=torch.float32)
    pre_q = torch.randn(1, seq_len, 16, 128, device=device)
    pre_k = torch.randn(1, seq_len, 16, 128, device=device)
    pre_q[:, 70:74, 0, :] = -20.0  # near-zero dead-conv head
    pre_k[:, 140:144, 1, :] = -20.0
    q_0 = F.silu(pre_q)
    k_0 = F.silu(pre_k)
    v_0 = F.silu(torch.randn(1, seq_len, 32, 128, device=device))
    g_0 = -F.softplus(torch.randn(1, seq_len, 32, device=device))
    b_0 = torch.sigmoid(torch.randn(1, seq_len, 32, device=device))
    b_0[:, 50:55, 2] = 1.5e-38  # subnormal bf16 beta
    grad_o = torch.randn(1, seq_len, 32, 128, device=device, dtype=torch.float32)

    xr = x_0.to(torch.bfloat16).float().detach().requires_grad_(True)
    qr = q_0.to(torch.bfloat16).float().detach().requires_grad_(True)
    kr = k_0.to(torch.bfloat16).float().detach().requires_grad_(True)
    vr = v_0.to(torch.bfloat16).float().detach().requires_grad_(True)
    gr = g_0.detach().clone().requires_grad_(True)
    br = b_0.to(torch.bfloat16).float().detach().requires_grad_(True)

    orig_tri = hssd_mod._HAS_TRITON
    hssd_mod._HAS_TRITON = False
    try:
        pr_r = ad_ref.project_inputs(xr, qr, kr, vr, gr, br, use_qk_l2norm=True)
        out_r, _, du_r = chunk_hssd(
            pr_r["query"],
            pr_r["key"],
            pr_r["value"],
            (pr_r["gamma_page_scale"], pr_r["gamma_chan_pair"]),
            pr_r["b_gate"],
            pr_r["w_gate"],
            pr_r["theta"],
            pr_r["surprise_bias"],
            ad_ref.log_surprise_slope.exp(),
            ad_ref.surprise_thresh,
            pr_r["write_logits"],
            pr_r["read_logits"],
            pr_r["q_swa"],
            pr_r["k_swa"],
            pr_r["v_swa"],
            boltzmann_gate=ad_ref.boltzmann_gate,
            swa_recency_slope=ad_ref.swa_recency_slope,
            chunk_size=64,
            window_chunks=8,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=4,
            return_aux=True,
        )
        ((out_r.float() * grad_o).sum() + du_r.sum()).backward()
    finally:
        hssd_mod._HAS_TRITON = orig_tri

    xt = x_0.to(torch.bfloat16).detach().requires_grad_(True)
    qt = q_0.to(torch.bfloat16).detach().requires_grad_(True)
    kt = k_0.to(torch.bfloat16).detach().requires_grad_(True)
    vt = v_0.to(torch.bfloat16).detach().requires_grad_(True)
    gt = g_0.detach().clone().requires_grad_(True)
    bt = b_0.to(torch.bfloat16).detach().requires_grad_(True)
    pr_t = ad_tri.project_inputs(xt, qt, kt, vt, gt, bt, use_qk_l2norm=True)
    out_t, _, du_t = chunk_hssd(
        pr_t["query"],
        pr_t["key"],
        pr_t["value"],
        (pr_t["gamma_page_scale"], pr_t["gamma_chan_pair"]),
        pr_t["b_gate"],
        pr_t["w_gate"],
        pr_t["theta"],
        pr_t["surprise_bias"],
        ad_tri.log_surprise_slope.exp(),
        ad_tri.surprise_thresh,
        pr_t["write_logits"],
        pr_t["read_logits"],
        pr_t["q_swa"],
        pr_t["k_swa"],
        pr_t["v_swa"],
        boltzmann_gate=ad_tri.boltzmann_gate,
        swa_recency_slope=ad_tri.swa_recency_slope,
        chunk_size=64,
        window_chunks=8,
        num_pages=8,
        top_k_pages=2,
        segment_chunks=4,
        return_aux=True,
    )
    ((out_t.float() * grad_o).sum() + du_t.sum()).backward()

    assert ad_tri.theta_up is not None and ad_ref.theta_up is not None
    for name, tv, rv in [
        ("out", out_t.float(), out_r.float()),
        ("dx", xt.grad.float(), xr.grad.float()),
        ("dq", qt.grad.float(), qr.grad.float()),
        ("dk", kt.grad.float(), kr.grad.float()),
        ("dv", vt.grad.float(), vr.grad.float()),
        ("dg", gt.grad.float(), gr.grad.float()),
        ("db", bt.grad.float(), br.grad.float()),
        ("dtheta_up", ad_tri.theta_up.weight.grad.float(), ad_ref.theta_up.weight.grad.float()),
        ("dswa_v_up", ad_tri.swa_v_up.weight.grad.float(), ad_ref.swa_v_up.weight.grad.float()),
    ]:
        assert torch.isfinite(tv).all(), f"non-finite {name}"
        ratio = tv.norm().item() / max(rv.norm().item(), 1e-12)
        cos = F.cosine_similarity(tv.flatten(), rv.flatten(), dim=0).item()
        assert 0.97 <= ratio <= 1.03, f"{name} norm ratio={ratio:.4f}"
        assert cos > 0.995, f"{name} cos={cos:.6f}"
