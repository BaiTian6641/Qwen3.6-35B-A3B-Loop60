"""Comprehensive tests for GDN-2-1M + Bounded QED + Soft Negative Eigenvalues + `fla` DPLR/GDN-2 Backend.

Covers:
  1. `GDN2Config` validation & `asymmetric_tanh_bound` (`C^1` unit slope at 0, bit-exact 0, 20.1x horizon).
  2. IEEE-754 Step-0 bit-identity (`max_diff == 0.0`) with all 1M + QED + soft neg-eigval flags enabled,
     plus `GDN2Adapter.is_zero()`, Step-1 gradient flow into `qed_lambda_raw` & `neg_gamma_raw`, and
     exact telemetry keys (`gdn2/qed_lambda_abs_mean`, `gdn2/neg_eigval_scale_mean`,
     `gdn2/decay_horizon_mult_max`, `gdn2/erase_write_l1`, `gdn2/decay_spread`, `gdn2/adapter_norm`).
  3. Exact fp64/fp32/bf16 forward & backward parity between `recurrent_gdn2`, `chunk_gdn2`, and the
     exact algebraic DPLR formula `S_t = (Diag(exp(gk)) + b a^T) S_{t-1} + k v^T`, `o_t = S_t^T (q * scale)`
     (including GQA `H_k < H_v`, `initial_state`, `output_final_state`, `b_gate` in `(0, 1.5)`, extreme decay,
     and `qed_lambda` / `neg_gamma` gradients), plus `qed_dplr_prep` analytical backward & gradcheck parity
     and Triton `tl.int64` indexing verification (and live CUDA Triton tests when CUDA is available).
  4. Mocked `fla` DPLR & `fla.ops.gdn2` dispatch:
     - `T=1` no-grad -> `fused_recurrent_dplr_delta_rule` / `fused_recurrent_gdn2`
     - `T=1` with grad -> `chunk_dplr_delta_rule` / `fla.ops.gdn2.chunk_gdn2` (never inference-only recurrent)
     - `T>1` -> `chunk_dplr_delta_rule` / `fla.ops.gdn2.chunk_gdn2`
     - No double-scaling of `q_hat`, GQA expansion before `fla`, exact argument mapping, clean CPU/fp64 fallback.
  5. End-to-end `s3_v2_long_1m_lsa` (no `gdn2`) -> `s2_v2_joint_bptt` (`gdn2` enabled, 1 BPTT step w/ 88L-style
     3-pass loop + `transformers 5.24`-style `DynamicCache`) -> `s4_v2_rl` (config + `slap_target_groups`
     isolation + GRPO step) -> `eval` (`_stack_with_ckpt` & `_pipeline_stage_cmds`) & `load_qloop_from_hub`.
"""
from __future__ import annotations

import ast
import copy
import math
import os
import pathlib
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from qloop.cli import _pipeline_stage_cmds, _stack_with_ckpt
from qloop.config import Cfg, load_config
from qloop.hub import V3_GDN_PIPELINE_ORDER, HubSyncManager, load_qloop_from_hub
from qloop.model import gdn2 as gdn2_mod
from qloop.model.gdn2 import (
    GDN2Config,
    _qed_dplr_prep_bwd_math,
    asymmetric_tanh_bound,
    chunk_gdn2,
    install_gdn2,
    l2norm,
    qed_dplr_prep,
    qed_dplr_prep_pytorch,
    recurrent_gdn2,
)
from qloop.model.probes import surgery_check
from qloop.model.toy import ToyConfig, build_toy
from qloop.train.bptt import BPTTConfig, SegmentedKVBPTTRunner
from qloop.train.checkpoint import CheckpointManager
from qloop.train.rl import GRPOTrainer
from qloop.train.sirsi import resolve_slap_target_params
from qloop.train.trainer import LMTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]

TOY_LOOP_3PASS = {
    "prelude_end": 0,
    "core_end": 8,
    "num_passes": 3,
    "target_virtual_layers": 24,
    "kv_share": True,
    "bridge_mode": "subspace",
    "bridge_rank": 8,
    "bridge_init": [0.05, 0.85, 0.10],
    "attn_adapter_rank": 4,
    "per_pass_router": True,
    "router_adapter_rank": 4,
    "gate_init": 0.0,
    "write_mask": True,
}

FULL_1M_QED_CFG: dict[str, Any] = {
    "rank": 8,
    "chunk_size": 8,
    "segment_chunks": 2,
    "decay_log_min": -3.0,
    "decay_log_max": 0.5,
    "qed_enabled": True,
    "qed_max_lambda": 0.35,
    "neg_eigval_max_boost": 0.5,
    "backend": "auto",
}


def _dplr_exact_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exact algebraic reference for `fla` DPLR:
        S_t = (Diag(exp(gk_t)) + b_t a_t^T) S_{t-1} + k_t v_t^T,
        o_t = S_t^T (q_t * scale).
    """
    init_dtype = q.dtype
    qf, kf, vf = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    af, bf, gkf = a.to(torch.float64), b.to(torch.float64), gk.to(torch.float64)
    bsz, seq_len, hv, dk = kf.shape
    dv = vf.shape[-1]
    sc = (dk ** -0.5) if scale is None else float(scale)
    state = (
        qf.new_zeros(bsz, hv, dk, dv)
        if initial_state is None
        else initial_state.to(torch.float64).clone()
    )
    outs = []
    for t in range(seq_len):
        diag_term = state * gkf[:, t].exp().unsqueeze(-1)
        a_s = (state * af[:, t].unsqueeze(-1)).sum(dim=-2)  # [B, H, V] = a_t^T S_{t-1}
        rank1_term = bf[:, t].unsqueeze(-1) * a_s.unsqueeze(-2)
        write_term = kf[:, t].unsqueeze(-1) * vf[:, t].unsqueeze(-2)
        state = diag_term + rank1_term + write_term
        outs.append((state * (qf[:, t] * sc).unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=1).to(init_dtype)
    st_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    return out, (state.to(st_dt) if output_final_state else None)


# ============================================================================ 1. Config & Asymmetric Bound


def test_gdn2_1m_config_validation_and_asymmetric_tanh_bound() -> None:
    # Backward-compatible defaults
    default_cfg = GDN2Config()
    default_cfg.validate()
    assert default_cfg.decay_log_min is None
    assert default_cfg.decay_log_max is None
    assert default_cfg.qed_enabled is False
    assert default_cfg.qed_max_lambda == pytest.approx(0.35)
    assert default_cfg.neg_eigval_max_boost == pytest.approx(0.0)
    assert default_cfg.backend == "auto"

    # Full 1M + QED + soft neg-eigval config
    cfg = GDN2Config.from_dict({"enabled": True, **FULL_1M_QED_CFG})
    cfg.validate()

    # Invalid parameter rejection
    with pytest.raises(ValueError, match="decay_log_min"):
        GDN2Config(decay_log_min=0.0).validate()
    with pytest.raises(ValueError, match="decay_log_min"):
        GDN2Config(decay_log_min=0.5).validate()
    with pytest.raises(ValueError, match="decay_log_max"):
        GDN2Config(decay_log_max=0.0).validate()
    with pytest.raises(ValueError, match="decay_log_max"):
        GDN2Config(decay_log_max=-0.2).validate()
    with pytest.raises(ValueError, match="qed_max_lambda"):
        GDN2Config(qed_max_lambda=-0.1).validate()
    with pytest.raises(ValueError, match="neg_eigval_max_boost"):
        GDN2Config(neg_eigval_max_boost=-0.1).validate()
    with pytest.raises(ValueError, match="neg_eigval_max_boost"):
        GDN2Config(neg_eigval_max_boost=1.5).validate()
    with pytest.raises(ValueError, match="backend"):
        GDN2Config(backend="bogus").validate()

    # Asymmetric tanh bound: bit-exact 0, C^1 derivative == 1 at 0, range (-3.0, +0.5), 20.1x horizon
    for dt in (torch.float32, torch.float64):
        z = torch.zeros(4, dtype=dt)
        assert torch.equal(asymmetric_tanh_bound(z, -3.0, 0.5), z)

    eps = 1e-6
    x_pos = torch.tensor([eps], dtype=torch.float64, requires_grad=True)
    x_neg = torch.tensor([-eps], dtype=torch.float64, requires_grad=True)
    x_zero = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    asymmetric_tanh_bound(x_pos, -3.0, 0.5).backward()
    asymmetric_tanh_bound(x_neg, -3.0, 0.5).backward()
    asymmetric_tanh_bound(x_zero, -3.0, 0.5).backward()
    assert x_pos.grad is not None and x_neg.grad is not None and x_zero.grad is not None
    assert float(x_pos.grad) == pytest.approx(1.0, abs=1e-9)
    assert float(x_neg.grad) == pytest.approx(1.0, abs=1e-9)
    assert float(x_zero.grad) == pytest.approx(1.0, abs=1e-12)

    extremes = torch.tensor([-100.0, 100.0], dtype=torch.float64)
    bounded = asymmetric_tanh_bound(extremes, -3.0, 0.5)
    assert float(bounded[0]) == pytest.approx(-3.0, abs=1e-10)
    assert float(bounded[1]) == pytest.approx(0.5, abs=1e-10)
    assert math.exp(-float(bounded[0])) == pytest.approx(math.exp(3.0), rel=1e-9)  # ~20.0855x horizon


# ========================================================================= 2. IEEE-754 Step-0 Bit-Identity & Telemetry


def test_step0_ieee754_bit_identity_with_1m_qed_neg_eigval_and_telemetry() -> None:
    torch.manual_seed(11)
    model = build_toy().double()
    ids = torch.randint(4, model.config.vocab_size, (2, 17))
    model.eval()
    with torch.no_grad():
        ref_eval = model(ids).logits

    # Surgery check with all 1M + QED + soft neg-eigval flags enabled
    rep = surgery_check(
        model,
        ids[:1],
        {
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 2,
            "target_virtual_layers": 24,
            "per_pass_router": True,
            "router_adapter_rank": 4,
        },
        None,
        decode_steps=4,
        atol=1e-8,
        gdn2_cfg=FULL_1M_QED_CFG,
    )
    assert rep["init_identity_maxdiff"] == 0.0
    assert rep["decode_vs_full_maxdiff"] < 1e-9

    # Install GDN-2-1M + QED + soft neg-eigval directly and verify bit-identity in eval & exact parity in train
    gdn2 = install_gdn2(model, FULL_1M_QED_CFG)
    assert len(gdn2.adapters) == 12
    for ad in gdn2.adapters.values():
        assert ad.qed_lambda_raw is not None and ad.qed_lambda_raw.shape == (ad.num_v_heads,)
        assert ad.neg_gamma_raw is not None and ad.neg_gamma_raw.shape == (ad.num_v_heads,)
        assert ad.is_zero() is True

    model.eval()
    with torch.no_grad():
        out_eval = model(ids).logits
    assert float((out_eval - ref_eval).abs().max()) == 0.0
    assert torch.equal(out_eval, ref_eval)

    st0 = gdn2.stats()
    assert st0["gdn2/adapter_norm"] == 0.0
    assert st0["gdn2/qed_lambda_abs_mean"] == 0.0
    assert st0["gdn2/neg_eigval_scale_mean"] == 1.0
    assert st0["gdn2/decay_horizon_mult_max"] == 1.0
    assert st0["gdn2/erase_write_l1"] == 0.0
    assert st0["gdn2/decay_spread"] == 0.0

    # `is_zero()` detects non-zero `qed_lambda_raw` and `neg_gamma_raw` individually
    ad0 = next(iter(gdn2.adapters.values()))
    assert ad0.qed_lambda_raw is not None and ad0.neg_gamma_raw is not None
    with torch.no_grad():
        ad0.qed_lambda_raw.fill_(0.1)
    assert ad0.is_zero() is False
    with torch.no_grad():
        ad0.qed_lambda_raw.zero_()
        ad0.neg_gamma_raw.fill_(0.1)
    assert ad0.is_zero() is False
    with torch.no_grad():
        ad0.neg_gamma_raw.zero_()
    assert ad0.is_zero() is True

    # Train mode Step-0 executes full compute_gates + chunk_gdn2 path and matches stock to ~1e-11
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in gdn2.trainable_parameters():
        p.requires_grad_(True)
    out_train = model(ids, labels=ids)
    torch.testing.assert_close(out_train.logits.detach(), ref_eval, rtol=1e-11, atol=1e-11)

    # Step-1 gradient flow must reach EVERY new parameter (`qed_lambda_raw`, `neg_gamma_raw`, `erase/write/decay_up`)
    out_train.loss.backward()
    for lid, ad in gdn2.adapters.items():
        for pname, p in ad.named_parameters():
            if pname == "down.weight":
                continue  # down.weight gets grad starting step 2 once up weights leave 0
            assert p.grad is not None, f"L{lid}.{pname} got None grad at step 1"
            assert torch.isfinite(p.grad).all(), f"L{lid}.{pname} got non-finite grad"
            assert float(p.grad.abs().max()) > 0.0, f"L{lid}.{pname} got zero grad at step 1"

    # Train 3 AdamW steps and verify all 6 telemetry keys move in expected ranges + cached decode parity
    opt = torch.optim.AdamW(gdn2.trainable_parameters(), lr=5e-2)
    opt.step()
    for _ in range(3):
        opt.zero_grad()
        model(ids, labels=ids).loss.backward()
        opt.step()

    st = gdn2.stats()
    assert st["gdn2/adapter_norm"] > 0.0
    assert 0.0 < st["gdn2/qed_lambda_abs_mean"] < FULL_1M_QED_CFG["qed_max_lambda"]
    assert 0.5 < st["gdn2/neg_eigval_scale_mean"] < 1.5
    assert st["gdn2/neg_eigval_scale_mean"] != 1.0
    assert st["gdn2/decay_horizon_mult_max"] > 1.0
    assert st["gdn2/erase_write_l1"] > 0.0
    assert st["gdn2/decay_spread"] > 0.0

    # Prefill (`chunk_gdn2`) + autoregressive T=1 decode (`recurrent_gdn2`) parity after training QED + neg_eigval
    model.eval()
    full_logits = model(ids).logits
    pre = model(ids[:, :7], use_cache=True)
    cache, steps = pre.past_key_values, [pre.logits]
    for t in range(7, ids.shape[1]):
        step_o = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        steps.append(step_o.logits)
    inc_logits = torch.cat(steps, dim=1)
    torch.testing.assert_close(inc_logits, full_logits, rtol=1e-9, atol=1e-10)

    # Clean uninstall restores bit-exact stock model
    gdn2.uninstall()
    assert torch.equal(model(ids).logits, ref_eval)


# ============================================================================ 3. Exact DPLR, Chunk & Recurrent Parity


@pytest.mark.parametrize("use_l2", [True, False])
@pytest.mark.parametrize("extreme_decay", [False, True])
def test_exact_fp64_fwd_bwd_parity_recurrent_chunk_and_dplr_algebra(use_l2: bool, extreme_decay: bool) -> None:
    """Verify exact fp64 forward + backward parity across:
    1) `recurrent_gdn2` (with QED + soft neg-eigval `b in (0, 1.5)` + GQA)
    2) `chunk_gdn2` (padded, 2 checkpointed WY segments)
    3) Exact algebraic DPLR formula `S_t = (Diag(exp(gk)) + b a^T) S_{t-1} + k v^T` fed by `qed_dplr_prep_pytorch`
    """
    torch.manual_seed(123 + int(use_l2) + 10 * int(extreme_decay))
    bsz, seq_len = 2, 37  # non-divisible by chunk_size=16; T_pad=48 > seg_len=32 -> 2 checkpointed segments
    hq, hk, hv, dk, dv = 2, 2, 4, 10, 12  # GQA H_k < H_v and non-power-of-2 dk != dv

    q0 = torch.randn(bsz, seq_len, hq, dk, dtype=torch.float64)
    k0 = torch.randn(bsz, seq_len, hk, dk, dtype=torch.float64)
    if not use_l2:
        q0 = l2norm(q0, dim=-1)
        k0 = l2norm(k0, dim=-1)
    v0 = torch.randn(bsz, seq_len, hv, dv, dtype=torch.float64)
    if extreme_decay:
        g0 = -(10.0 + 3.0 * torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64))
    else:
        g0 = -(1e-4 + 0.30 * torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64))
    # Soft negative eigenvalues: b_gate spans (0.05, 1.48)
    b0 = 0.05 + 1.43 * torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64)
    w0 = 0.05 + 0.90 * torch.rand(bsz, seq_len, hv, dv, dtype=torch.float64)
    lam0 = torch.tensor([-0.31, 0.27, -0.14, 0.33], dtype=torch.float64)
    s0 = 0.25 * torch.randn(bsz, hv, dk, dv, dtype=torch.float64)

    def _clone_leaves():
        return [t.detach().clone().requires_grad_(True) for t in (q0, k0, v0, g0, b0, w0, lam0, s0)]

    rec_in = _clone_leaves()
    chk_in = _clone_leaves()
    dplr_in = _clone_leaves()

    out_rec, st_rec = recurrent_gdn2(
        rec_in[0],
        rec_in[1],
        rec_in[2],
        rec_in[3],
        rec_in[4],
        rec_in[5],
        qed_lambda=rec_in[6],
        backend="pytorch",
        initial_state=rec_in[7],
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_l2,
        scale=None,
    )
    out_chk, st_chk = chunk_gdn2(
        chk_in[0],
        chk_in[1],
        chk_in[2],
        chk_in[3],
        chk_in[4],
        chk_in[5],
        qed_lambda=chk_in[6],
        backend="pytorch",
        chunk_size=16,
        segment_chunks=2,
        checkpoint_segments=True,
        initial_state=chk_in[7],
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_l2,
        scale=None,
    )
    qh, kh, vw, a_dplr, gk_dplr = qed_dplr_prep_pytorch(
        dplr_in[0],
        dplr_in[1],
        dplr_in[2],
        dplr_in[3],
        dplr_in[4],
        dplr_in[5],
        dplr_in[6],
        use_qk_l2norm=use_l2,
        out_dtype=torch.float64,
    )
    out_dplr, st_dplr = _dplr_exact_ref(
        qh,
        kh,
        vw,
        a_dplr,
        kh,
        gk_dplr,
        scale=dk ** -0.5,
        initial_state=dplr_in[7],
        output_final_state=True,
    )

    assert st_rec is not None and st_chk is not None and st_dplr is not None
    torch.testing.assert_close(out_chk, out_rec, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(st_chk, st_rec, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(out_dplr, out_rec, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(st_dplr, st_rec, rtol=1e-11, atol=1e-11)

    go = torch.randn_like(out_rec)
    gs = torch.randn_like(st_rec)
    ((out_rec * go).sum() + (st_rec * gs).sum()).backward()
    ((out_chk * go).sum() + (st_chk * gs).sum()).backward()
    ((out_dplr * go).sum() + (st_dplr * gs).sum()).backward()

    names = ("q", "k", "v", "g_vec", "b_gate", "w_gate", "qed_lambda", "initial_state")
    for nm, t_rec, t_chk, t_dplr in zip(names, rec_in, chk_in, dplr_in):
        assert t_rec.grad is not None and t_chk.grad is not None and t_dplr.grad is not None, nm
        torch.testing.assert_close(t_chk.grad, t_rec.grad, rtol=1e-9, atol=1e-9, msg=f"chunk vs rec grad {nm}")
        torch.testing.assert_close(t_dplr.grad, t_rec.grad, rtol=1e-10, atol=1e-10, msg=f"dplr vs rec grad {nm}")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_l2", [True, False])
@pytest.mark.parametrize("has_qed", [True, False])
def test_qed_dplr_prep_analytical_backward_parity_and_gradcheck(
    dtype: torch.dtype, use_l2: bool, has_qed: bool
) -> None:
    """Verify the analytical backward formulas used by `_qed_dplr_prep_bwd_kernel` match PyTorch autograd
    in fp64, fp32, and bf16 (incl. GQA H_q != H_k < H_v, g_vec clamping across 0, and partial None grads).
    """
    torch.manual_seed(99)
    bsz, seq_len, hq, hk, hv, dk, dv = 2, 7, 4, 2, 4, 10, 12  # asymmetric H_q != H_k, dk != dv
    q = torch.randn(bsz, seq_len, hq, dk, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, seq_len, hk, dk, dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, seq_len, hv, dv, dtype=dtype, requires_grad=True)
    # Include positive values to verify clamping at g <= 0 has exact zero subgradient for g > 0
    g_dt = torch.float64 if dtype == torch.float64 else torch.float32
    g = (torch.randn(bsz, seq_len, hv, dk, dtype=g_dt) * 0.4 - 0.2).detach().requires_grad_(True)
    b = (0.1 + 1.35 * torch.rand(bsz, seq_len, hv, dk, dtype=g.dtype)).detach().requires_grad_(True)
    w = (0.1 + 0.85 * torch.rand(bsz, seq_len, hv, dv, dtype=g.dtype)).detach().requires_grad_(True)
    lam = (
        torch.tensor([-0.3, 0.25, -0.15, 0.34], dtype=g.dtype, requires_grad=True)
        if has_qed
        else None
    )

    qh, kh, vw, a_out, gk_out = qed_dplr_prep_pytorch(
        q, k, v, g, b, w, lam, use_qk_l2norm=use_l2, out_dtype=dtype
    )
    dqh = torch.randn_like(qh)
    dkh = torch.randn_like(kh)
    dvw = torch.randn_like(vw)
    da = torch.randn_like(a_out)
    dgk = torch.randn_like(gk_out)

    loss = (
        (qh.float() * dqh.float()).sum()
        + (kh.float() * dkh.float()).sum()
        + (vw.float() * dvw.float()).sum()
        + (a_out.float() * da.float()).sum()
        + (gk_out.float() * dgk.float()).sum()
    )
    if dtype == torch.float64:
        loss = (qh * dqh).sum() + (kh * dkh).sum() + (vw * dvw).sum() + (a_out * da).sum() + (gk_out * dgk).sum()
    loss.backward()

    ana = _qed_dplr_prep_bwd_math(
        q.detach(),
        k.detach(),
        v.detach(),
        g.detach(),
        b.detach(),
        w.detach(),
        None if lam is None else lam.detach(),
        dqh,
        dkh,
        dvw,
        da,
        dgk,
        use_qk_l2norm=use_l2,
    )
    ref_tensors = (q, k, v, g, b, w, lam)
    for name, ref_t, got_g in zip(("q", "k", "v", "g", "b", "w", "lam"), ref_tensors, ana):
        if ref_t is None:
            assert got_g is None
            continue
        assert ref_t.grad is not None and got_g is not None, name
        if dtype == torch.float64:
            torch.testing.assert_close(got_g, ref_t.grad, rtol=1e-11, atol=1e-11, msg=f"fp64 {name}")
        elif dtype == torch.float32:
            torch.testing.assert_close(got_g, ref_t.grad, rtol=1e-5, atol=1e-5, msg=f"fp32 {name}")
        else:
            cos = float(F.cosine_similarity(got_g.float().reshape(1, -1), ref_t.grad.float().reshape(1, -1)))
            assert cos > 0.999, f"bf16 {name} cos={cos}"

    # Verify g > 0 has exact zero gradient
    pos_mask = g.detach() > 0.0
    if pos_mask.any():
        assert torch.equal(ana[3][pos_mask], torch.zeros_like(ana[3][pos_mask]))

    # Verify partial None upstream cotangents don't crash and produce exact grads
    ana_partial = _qed_dplr_prep_bwd_math(
        q.detach(),
        k.detach(),
        v.detach(),
        g.detach(),
        b.detach(),
        w.detach(),
        None if lam is None else lam.detach(),
        None,
        dkh,
        None,
        da,
        None,
        use_qk_l2norm=use_l2,
    )
    assert torch.equal(ana_partial[2], torch.zeros_like(v))
    assert torch.equal(ana_partial[5], torch.zeros_like(w))

    # Finite-difference gradcheck in fp64
    if dtype == torch.float64 and has_qed:
        q_gc = torch.randn(1, 3, 2, 4, dtype=torch.float64, requires_grad=True)
        k_gc = torch.randn(1, 3, 2, 4, dtype=torch.float64, requires_grad=True)
        v_gc = torch.randn(1, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        g_gc = (-0.1 - 0.4 * torch.rand(1, 3, 4, 4, dtype=torch.float64)).detach().requires_grad_(True)
        b_gc = (0.2 + 1.1 * torch.rand(1, 3, 4, 4, dtype=torch.float64)).detach().requires_grad_(True)
        w_gc = (0.2 + 0.6 * torch.rand(1, 3, 4, 5, dtype=torch.float64)).detach().requires_grad_(True)
        l_gc = torch.tensor([-0.2, 0.15, -0.1, 0.28], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda qq, kk, vv, gg, bb, ww, ll: qed_dplr_prep_pytorch(
                qq, kk, vv, gg, bb, ww, ll, use_qk_l2norm=use_l2, out_dtype=torch.float64
            ),
            (q_gc, k_gc, v_gc, g_gc, b_gc, w_gc, l_gc),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )


def test_gdn2_triton_kernels_use_int64_indexing() -> None:
    """Every `@triton.jit` kernel in `src/qloop/model/gdn2.py` must use `tl.int64` indexing for 1M safety."""
    src = (ROOT / "src" / "qloop" / "model" / "gdn2.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    kernels: list[str] = []
    missing_int64: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decs = [ast.unparse(d) for d in node.decorator_list]
            if any("triton.jit" in d for d in decs):
                kernels.append(node.name)
                body_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                if "tl.int64" not in body_src:
                    missing_int64.append(node.name)
    assert kernels == ["_qed_dplr_prep_fwd_kernel", "_qed_dplr_prep_bwd_kernel"]
    assert not missing_int64, f"GDN-2 Triton kernels missing tl.int64 indexing: {missing_int64}"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and gdn2_mod._HAS_TRITON),
    reason="CUDA + Triton required for live GPU fused prep kernel test",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_l2", [True, False])
def test_cuda_triton_qed_dplr_prep_fwd_bwd_parity(dtype: torch.dtype, use_l2: bool) -> None:
    """Live CUDA parity test: fused Triton `qed_dplr_prep` vs `qed_dplr_prep_pytorch` (Qwen3.6 dims)."""
    torch.manual_seed(7)
    dev = torch.device("cuda")
    bsz, seq_len, hq, hk, hv, dk, dv = 1, 64, 16, 16, 32, 128, 128
    q = torch.randn(bsz, seq_len, hq, dk, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, seq_len, hk, dk, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, seq_len, hv, dv, device=dev, dtype=dtype, requires_grad=True)
    g = (-0.01 - 0.3 * torch.rand(bsz, seq_len, hv, dk, device=dev, dtype=torch.float32)).detach().requires_grad_(True)
    b = (0.05 + 1.4 * torch.rand(bsz, seq_len, hv, dk, device=dev, dtype=torch.float32)).detach().requires_grad_(True)
    w = (0.05 + 0.9 * torch.rand(bsz, seq_len, hv, dv, device=dev, dtype=torch.float32)).detach().requires_grad_(True)
    lam = (0.3 * torch.randn(hv, device=dev, dtype=torch.float32)).detach().requires_grad_(True)

    ref_in = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, b, w, lam)]
    out_tri = qed_dplr_prep(q, k, v, g, b, w, lam, use_qk_l2norm=use_l2, out_dtype=dtype)
    out_ref = qed_dplr_prep_pytorch(*ref_in, use_qk_l2norm=use_l2, out_dtype=dtype)
    for o_t, o_r in zip(out_tri, out_ref):
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(o_t, o_r, rtol=tol, atol=tol)

    weights = [torch.randn_like(o) for o in out_tri]
    sum((o.float() * wt.float()).sum() for o, wt in zip(out_tri, weights)).backward()
    sum((o.float() * wt.float()).sum() for o, wt in zip(out_ref, weights)).backward()
    for t_tri, t_ref in zip((q, k, v, g, b, w, lam), ref_in):
        assert t_tri.grad is not None and t_ref.grad is not None
        cos = float(F.cosine_similarity(t_tri.grad.float().reshape(1, -1), t_ref.grad.float().reshape(1, -1)))
        assert cos > (0.99999 if dtype == torch.float32 else 0.999)

    # Also test T=1 no-grad direct Triton launch path
    with torch.no_grad():
        o1_tri = qed_dplr_prep(q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1], lam, use_qk_l2norm=use_l2)
        o1_ref = qed_dplr_prep_pytorch(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1], lam, use_qk_l2norm=use_l2
        )
        for a1, b1 in zip(o1_tri, o1_ref):
            tol = 1e-4 if dtype == torch.float32 else 2e-2
            torch.testing.assert_close(a1, b1, rtol=tol, atol=tol)


# ============================================================================ 4. Mocked FLA DPLR & FLA GDN-2 Dispatch


def test_mocked_fla_dplr_and_fla_gdn2_dispatch_and_no_double_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify exact dispatch contract for `fla` DPLR and `fla.ops.gdn2`:
      - T=1 no-grad -> `fused_recurrent_dplr_delta_rule` / `fused_recurrent_gdn2`
      - T=1 with grad -> `chunk_dplr_delta_rule` / `fla.ops.gdn2.chunk_gdn2` (never inference-only recurrent)
      - T>1 (grad or no-grad) -> `chunk_dplr_delta_rule` / `fla.ops.gdn2.chunk_gdn2`
      - No double-scaling of `q_hat` (`||q_hat||_2 ~= 1.0`, `scale == dk ** -0.5`), GQA expanded before `fla`.
    """
    calls: list[dict[str, Any]] = []

    def _fake_chunk_dplr(*, q, k, v, a, b, gk, scale=None, initial_state=None, output_final_state=False):
        calls.append({"fn": "chunk_dplr_delta_rule", "T": q.shape[1], "H": q.shape[2], "q": q.detach(), "b": b.detach(),
                      "k": k.detach(), "scale": scale, "grad": torch.is_grad_enabled()})
        return _dplr_exact_ref(
            q, k, v, a, b, gk, scale=scale, initial_state=initial_state, output_final_state=output_final_state
        )

    def _fake_rec_dplr(*, q, k, v, a, b, gk, scale=None, initial_state=None, output_final_state=False):
        if torch.is_grad_enabled():
            raise NotImplementedError("fused_recurrent_dplr_delta_rule is inference-only")
        calls.append({"fn": "fused_recurrent_dplr_delta_rule", "T": q.shape[1], "H": q.shape[2], "q": q.detach(),
                      "b": b.detach(), "k": k.detach(), "scale": scale, "grad": torch.is_grad_enabled()})
        return _dplr_exact_ref(
            q, k, v, a, b, gk, scale=scale, initial_state=initial_state, output_final_state=output_final_state
        )

    def _fake_chunk_gdn2(
        *, q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
    ):
        calls.append({"fn": "fla_chunk_gdn2", "T": q.shape[1], "H": q.shape[2], "scale": scale,
                      "use_l2": use_qk_l2norm_in_kernel, "grad": torch.is_grad_enabled()})
        return recurrent_gdn2(
            q, k, v, g, b, w, backend="pytorch", initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, scale=scale
        )

    def _fake_rec_gdn2(
        *, q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
    ):
        if torch.is_grad_enabled():
            raise NotImplementedError("fused_recurrent_gdn2 is inference-only")
        calls.append({"fn": "fla_fused_recurrent_gdn2", "T": q.shape[1], "H": q.shape[2], "scale": scale,
                      "use_l2": use_qk_l2norm_in_kernel, "grad": torch.is_grad_enabled()})
        return recurrent_gdn2(
            q, k, v, g, b, w, backend="pytorch", initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, scale=scale
        )

    # First verify CPU fallback without mocks works seamlessly even when backend="dplr" or "auto"
    torch.manual_seed(5)
    bsz, hq, hk, hv, dk, dv = 1, 2, 2, 4, 16, 16  # GQA 2 -> 4
    q = torch.randn(bsz, 5, hq, dk, dtype=torch.float32) * 3.0
    k = torch.randn(bsz, 5, hk, dk, dtype=torch.float32) * 3.0
    v = torch.randn(bsz, 5, hv, dv, dtype=torch.float32)
    g = -0.05 - 0.2 * torch.rand(bsz, 5, hv, dk, dtype=torch.float32)
    b = 0.1 + 1.3 * torch.rand(bsz, 5, hv, dk, dtype=torch.float32)
    w = 0.1 + 0.8 * torch.rand(bsz, 5, hv, dv, dtype=torch.float32)
    lam = torch.tensor([-0.25, 0.30, -0.10, 0.20], dtype=torch.float32)
    h0 = 0.2 * torch.randn(bsz, hv, dk, dv, dtype=torch.float32)

    out_cpu_fb, st_cpu_fb = chunk_gdn2(
        q, k, v, g, b, w, qed_lambda=lam, backend="auto", initial_state=h0,
        output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    out_py_ref, st_py_ref = recurrent_gdn2(
        q, k, v, g, b, w, qed_lambda=lam, backend="pytorch", initial_state=h0,
        output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    torch.testing.assert_close(out_cpu_fb, out_py_ref, rtol=1e-5, atol=1e-5)
    assert st_cpu_fb is not None and st_py_ref is not None
    torch.testing.assert_close(st_cpu_fb, st_py_ref, rtol=1e-5, atol=1e-5)

    # Now enable mocked FLA ops
    monkeypatch.setattr(gdn2_mod, "_is_fla_eligible_tensor", lambda t: t.dtype != torch.float64)
    monkeypatch.setattr(gdn2_mod, "_get_fla_dplr_ops", lambda: (_fake_chunk_dplr, _fake_rec_dplr))
    monkeypatch.setattr(gdn2_mod, "_get_fla_gdn2_ops", lambda: (_fake_chunk_gdn2, _fake_rec_gdn2))

    # 1) T=1 no-grad with QED -> `fused_recurrent_dplr_delta_rule`
    with torch.no_grad():
        o_t1, s_t1 = recurrent_gdn2(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
            qed_lambda=lam, backend="auto", initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True, scale=None,
        )
    assert calls[-1]["fn"] == "fused_recurrent_dplr_delta_rule"
    assert calls[-1]["H"] == hv
    assert calls[-1]["scale"] == pytest.approx(dk ** -0.5)
    # Verify NO double-scaling: `q` passed to FLA has unit L2 norm (~1.0), NOT `dk ** -0.5`!
    q_norms = calls[-1]["q"].norm(dim=-1)
    torch.testing.assert_close(q_norms, torch.ones_like(q_norms), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(calls[-1]["b"], calls[-1]["k"])
    o_t1_ref, s_t1_ref = recurrent_gdn2(
        q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
        qed_lambda=lam, backend="pytorch", initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, scale=None,
    )
    torch.testing.assert_close(o_t1, o_t1_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s_t1, s_t1_ref, rtol=1e-5, atol=1e-5)

    # 2) T=1 WITH grad -> `chunk_dplr_delta_rule` (NEVER `fused_recurrent_dplr_delta_rule`)
    q1 = q[:, :1].detach().clone().requires_grad_(True)
    lam1 = lam.detach().clone().requires_grad_(True)
    o_t1g, s_t1g = recurrent_gdn2(
        q1, k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
        qed_lambda=lam1, backend="auto", initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert calls[-1]["fn"] == "chunk_dplr_delta_rule"
    assert s_t1g is not None
    (o_t1g.sum() + s_t1g.sum()).backward()
    assert q1.grad is not None and torch.isfinite(q1.grad).all()
    assert lam1.grad is not None and torch.isfinite(lam1.grad).all()

    # 3) T>1 WITH grad -> `chunk_dplr_delta_rule`, exact parity with pure-PyTorch reference
    qg = q.detach().clone().requires_grad_(True)
    lamg = lam.detach().clone().requires_grad_(True)
    o_tg, s_tg = chunk_gdn2(
        qg, k, v, g, b, w, qed_lambda=lamg, backend="auto", initial_state=h0,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    assert calls[-1]["fn"] == "chunk_dplr_delta_rule"
    assert calls[-1]["scale"] == pytest.approx(dk ** -0.5)
    torch.testing.assert_close(o_tg, out_py_ref, rtol=1e-5, atol=1e-5)
    assert s_tg is not None and st_py_ref is not None
    torch.testing.assert_close(s_tg, st_py_ref, rtol=1e-5, atol=1e-5)
    (o_tg.sum() + s_tg.sum()).backward()
    assert qg.grad is not None and lamg.grad is not None

    # 4) No-QED `backend="auto"` dispatches to `fla.ops.gdn2` (`fused_recurrent_gdn2` for T=1 no-grad, else chunk)
    with torch.no_grad():
        o_nq1, _ = recurrent_gdn2(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
            qed_lambda=None, backend="auto", initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    assert calls[-1]["fn"] == "fla_fused_recurrent_gdn2"
    assert calls[-1]["H"] == hv
    assert calls[-1]["use_l2"] is True
    assert calls[-1]["scale"] == pytest.approx(dk ** -0.5)
    o_nq1_ref, _ = recurrent_gdn2(
        q[:, :1], k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
        qed_lambda=None, backend="pytorch", initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(o_nq1, o_nq1_ref, rtol=1e-5, atol=1e-5)

    o_nqg, _ = chunk_gdn2(
        q[:, :1].detach().clone().requires_grad_(True), k[:, :1], v[:, :1], g[:, :1], b[:, :1], w[:, :1],
        qed_lambda=None, backend="auto", initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert calls[-1]["fn"] == "fla_chunk_gdn2"
    o_nqg.sum().backward()

    # 5) `backend="fla_gdn2"` with QED safely redirects to `dplr`
    with torch.no_grad():
        chunk_gdn2(
            q, k, v, g, b, w, qed_lambda=lam, backend="fla_gdn2",
            use_qk_l2norm_in_kernel=True,
        )
    assert calls[-1]["fn"] == "chunk_dplr_delta_rule"

    # 6) float64 always falls back to pure-PyTorch even when FLA is installed
    n_before = len(calls)
    chunk_gdn2(
        q.double(), k.double(), v.double(), g.double(), b.double(), w.double(),
        qed_lambda=lam.double(), backend="auto", use_qk_l2norm_in_kernel=True,
    )
    assert len(calls) == n_before


# ============================================================================ 5. s3_v2 -> s2_v2 -> s4_v2 -> eval & Hub


class _TF524LinearLayerCache:
    """Mimics `transformers 5.24` `DynamicCache.layers[i]` for linear-attention layers."""

    def __init__(self) -> None:
        self.conv_states: torch.Tensor | None = None
        self.recurrent_states: torch.Tensor | None = None
        self.has_previous_state: bool = False


class _TF524HybridDynamicCache:
    """Minimal `transformers 5.24`-compatible cache (`layers[i]`, `get_conv_state`, `get_recurrent_state`)."""

    def __init__(self, config: ToyConfig) -> None:
        n = int(config.num_hidden_layers)
        self.layer_types = list(config.layer_types)
        self.key_cache: list[torch.Tensor | None] = [None] * n
        self.value_cache: list[torch.Tensor | None] = [None] * n
        self.conv_states: list[torch.Tensor | None] = [None] * n
        self.recurrent_states: list[torch.Tensor | None] = [None] * n
        self.layers = [_TF524LinearLayerCache() for _ in range(n)]
        self.first_full = self.layer_types.index("full_attention")

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = k, v
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        kc = self.key_cache[self.first_full if layer_idx is None else layer_idx]
        return 0 if kc is None else int(kc.shape[-2])

    def get_conv_state(self, layer_idx: int) -> torch.Tensor | None:
        return self.conv_states[layer_idx]

    def get_recurrent_state(self, layer_idx: int) -> torch.Tensor | None:
        return self.recurrent_states[layer_idx]


class _FakeHubAPI:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commits: list[str] = []

    def create_repo(self, **_: Any) -> None:
        return None

    def list_repo_files(self, **_: Any) -> list[str]:
        return sorted(self.files)

    def download_file(self, _repo: str, path_in_repo: str, dst: str) -> bool:
        if path_in_repo not in self.files:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(self.files[path_in_repo])
        return True

    def commit_bundle(self, _repo: str, additions: list[tuple[str, str | bytes]], msg: str) -> str:
        for rp, src in additions:
            if isinstance(src, bytes):
                self.files[rp] = src
            else:
                with open(src, "rb") as f:
                    self.files[rp] = f.read()
        self.commits.append(msg)
        return f"oid_{len(self.commits):06d}"


def _toy_stage_cfg(stage_name: str, *, checkpoint: bool = False) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "configs" / "stages" / f"{stage_name}.yaml")).to_dict()
    cfg["model"] = {
        "name": "toy",
        "toy": {
            "num_hidden_layers": 8,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    }
    cfg["loop"].update(
        {
            "prelude_end": 0,
            "core_end": 8,
            "num_passes": 3,
            "target_virtual_layers": 24,
            "bridge_rank": 8,
            "attn_adapter_rank": 4,
            "router_adapter_rank": 4,
            "checkpoint": checkpoint,
        }
    )
    if (cfg.get("gdn2") or {}).get("enabled"):
        cfg["gdn2"].update({"rank": 8, "chunk_size": 4, "segment_chunks": 2})
    cfg["lsa"].update(
        {
            "index_dim": 8,
            "index_on": "all",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
            "distill_query_frac": 1.0,
            "query_chunk": 16,
        }
    )
    cfg["checkpoint"] = {"sync": {"enabled": False}}
    return cfg


def test_s3_v2_to_s2_v2_gdn2_bptt_to_s4_v2_rl_eval_and_hub_e2e(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline continuity check:
    1) `s3_v2` saves checkpoint with NO `gdn2_adapter` keys (`stack.gdn2 == None`).
    2) `s2_v2_joint_bptt` loads `s3_v2` cleanly with NO strict-key crash, starts bit-identical at Step 0,
       runs 1 segmented BPTT step with 3-pass loop + TF 5.24 cache + GDN-2-1M+QED, saves `stack.gdn2` metadata.
    3) `s4_v2_rl` loads `s2_v2` GDN-2 weights cleanly, validates `lr_mult` & `slap_target_groups` isolation,
       and completes 1 GRPO step.
    4) `eval` (`_stack_with_ckpt` using `configs/eval/suite.yaml` + `_pipeline_stage_cmds`) and
       `load_qloop_from_hub` both restore GDN-2-1M+QED adapters automatically without dropping weights.
    """
    torch.manual_seed(0)

    # --- Stage 1: s3_v2_long_1m_lsa (no GDN-2)
    s3_cfg = _toy_stage_cfg("s3_v2_long_1m_lsa", checkpoint=True)
    s3_cfg["train"].update(
        {"init_from": None, "steps": 1, "micro_batch": 1, "grad_accum": 1, "log_every": 1,
         "ckpt_every": 1, "eval_every": 0, "prefetch_batches": 0}
    )
    s3_cfg["long_calib"].update({"enabled": False})
    s3_cfg["data"] = {
        "seq_len": 16,
        "respect_doc_boundaries": False,
        "sources": [{"name": "toy_text", "weight": 1.0}],
        "decontaminate": {"enabled": False},
    }
    s3_cfg["eval_quick"] = {"n_batches": 0}
    s3_dir = str(tmp_path / "s3_v2_long_1m_lsa")
    tr_s3 = LMTrainer(s3_cfg, s3_dir)
    assert tr_s3.stack.gdn2 is None and tr_s3.stack.hssd is None
    tr_s3.train()

    ck_s3 = CheckpointManager(s3_dir, keep_last=0).load()
    assert ck_s3 is not None
    assert not any("gdn2_adapter." in k for k in ck_s3["params"])
    assert ck_s3["state"]["stack"]["gdn2"] is None

    # Capture s3_v2 eval logits to verify exact Step-0 bit-identity of s2_v2 initialized from s3_v2
    probe_ids = torch.randint(4, 60, (1, 16))
    tr_s3.stack.model.eval()
    assert tr_s3.stack.loop is not None
    tr_s3.stack.loop.set_active_passes(3)
    with torch.no_grad():
        s3_eval_logits = tr_s3.stack.model(probe_ids).logits.detach().clone()

    # --- Stage 2: s2_v2_joint_bptt (GDN-2-1M + QED enabled, initialized from s3_v2!)
    s2_cfg = _toy_stage_cfg("s2_v2_joint_bptt", checkpoint=True)
    s2_cfg["train"].update(
        {"init_from": s3_dir, "steps": 1, "micro_batch": 1, "grad_accum": 1, "log_every": 1,
         "ckpt_every": 1, "eval_every": 0, "prefetch_batches": 0}
    )
    s2_cfg["bptt"].update(
        {"enabled": True, "n_segments": 4, "seg_len": 8, "replay_k": 4, "offload_prior_hidden": False,
         "query_sim_chunk": 8}
    )
    s2_cfg["data"] = {
        "seq_len": 32,
        "respect_doc_boundaries": True,
        "sources": [{"name": "toy_cot", "weight": 1.0}],
        "decontaminate": {"enabled": False},
    }
    s2_cfg["eval_quick"] = {"n_batches": 0}
    s2_dir = str(tmp_path / "s2_v2_joint_bptt")
    tr_s2 = LMTrainer(s2_cfg, s2_dir)
    assert tr_s2.stack.gdn2 is not None and tr_s2.stack.hssd is None
    tr_s2.maybe_resume()  # MUST NOT raise strict-key mismatch on missing gdn2_adapter.*!

    # Verify Step-0 bit-identity right after loading s3_v2 into s2_v2
    assert all(ad.is_zero() for ad in tr_s2.stack.gdn2.adapters.values())
    tr_s2.stack.model.eval()
    assert tr_s2.stack.loop is not None
    tr_s2.stack.loop.set_active_passes(3)
    with torch.no_grad():
        s2_step0_logits = tr_s2.stack.model(probe_ids).logits
    assert torch.equal(s2_step0_logits, s3_eval_logits), "s2_v2 with GDN-2-1M+QED must be bit-identical at step 0"

    # Verify optimizer lr_mult for gdn2_adapter is 5.0 in s2_v2
    gdn2_entries = [e for e in tr_s2.opt.entries.values() if "gdn2_adapter." in e.name]
    assert gdn2_entries
    expected_gdn2_lr = s2_cfg["train"]["optim"]["lr"] * 5.0
    assert all(e.base_lr == pytest.approx(expected_gdn2_lr) for e in gdn2_entries)

    # Also test SegmentedKVBPTTRunner directly with a TF 5.24-style hybrid DynamicCache
    import qloop.train.bptt as bptt_mod

    tr_s2.stack.model.train()
    with monkeypatch.context() as mp:
        mp.setattr(bptt_mod, "_maybe_create_initial_cache", lambda tm: _TF524HybridDynamicCache(tm.config))
        tf524_runner = SegmentedKVBPTTRunner(
            tr_s2.stack.model,
            cfg=BPTTConfig(
                enabled=True, n_segments=2, seg_len=8, replay_k=4, offload_prior_hidden=False, query_sim_chunk=8
            ),
        )
        bptt_ids = torch.randint(4, 60, (1, 16))
        bptt_res = tf524_runner.run_step(bptt_ids, labels=bptt_ids, backward=True)
    assert math.isfinite(bptt_res.total_loss)
    assert bptt_res.num_segments == 2
    for lid, ad in tr_s2.stack.gdn2.adapters.items():
        assert ad.qed_lambda_raw is not None and ad.qed_lambda_raw.grad is not None
        assert ad.neg_gamma_raw is not None and ad.neg_gamma_raw.grad is not None
        assert float(ad.qed_lambda_raw.grad.abs().max()) > 0.0, f"L{lid} qed_lambda_raw zero grad in BPTT"
        assert float(ad.neg_gamma_raw.grad.abs().max()) > 0.0, f"L{lid} neg_gamma_raw zero grad in BPTT"
    tr_s2.opt.zero_grad(set_to_none=True)

    # Run full LMTrainer 1-step BPTT training & checkpoint save
    out_s2 = tr_s2.train()
    assert out_s2["step"] == 1
    assert "gdn2/qed_lambda_abs_mean" in out_s2["last"]
    assert "gdn2/neg_eigval_scale_mean" in out_s2["last"]
    assert "gdn2/decay_horizon_mult_max" in out_s2["last"]

    ck_s2 = CheckpointManager(s2_dir, keep_last=0).load()
    assert ck_s2 is not None
    assert any("gdn2_adapter.qed_lambda_raw" in k for k in ck_s2["params"])
    assert any("gdn2_adapter.neg_gamma_raw" in k for k in ck_s2["params"])
    s2_gdn2_meta = ck_s2["state"]["stack"]["gdn2"]
    assert isinstance(s2_gdn2_meta, dict)
    assert s2_gdn2_meta["decay_log_min"] == pytest.approx(-3.0)
    assert s2_gdn2_meta["decay_log_max"] == pytest.approx(0.5)
    assert s2_gdn2_meta["qed_enabled"] is True
    assert s2_gdn2_meta["qed_max_lambda"] == pytest.approx(0.35)
    assert s2_gdn2_meta["neg_eigval_max_boost"] == pytest.approx(0.5)

    # Perturb saved s2_v2 qed_lambda_raw slightly so we verify non-trivial weight transfer to s4_v2_rl and eval
    saved_s2_dir = ck_s2["path"]
    s2_params = dict(ck_s2["params"])
    for k in list(s2_params):
        if k.endswith("gdn2_adapter.qed_lambda_raw"):
            s2_params[k] = torch.full_like(s2_params[k], 0.25)
        elif k.endswith("gdn2_adapter.neg_gamma_raw"):
            s2_params[k] = torch.full_like(s2_params[k], 0.30)
    torch.save(s2_params, os.path.join(saved_s2_dir, "trainable.pt"))

    # --- Stage 3: s4_v2_rl (loads s2_v2 GDN-2 weights, slap excludes gdn2_adapters, lr_mult=2.0)
    s4_cfg = _toy_stage_cfg("s4_v2_rl", checkpoint=False)
    s4_cfg["train"].update({"init_from": s2_dir, "steps": 1, "ckpt_every": 1})
    s4_cfg["rl"] = {
        "tasks": [{"source": "toy_arith", "n": 4, "verifier": "scorer:math", "weight": 1.0}],
        "group_size": 2,
        "prompts_per_step": 1,
        "max_new_tokens": 6,
        "micro_batch_seqs": 2,
        "verifier_workers": 1,
        "sirsi": copy.deepcopy(s4_cfg["rl"]["sirsi"]),
        "gate": {"every": 0},
    }
    s4_dir = str(tmp_path / "s4_v2_rl")
    rl_tr = GRPOTrainer(s4_cfg, s4_dir)
    assert rl_tr.stack.gdn2 is not None and rl_tr.stack.hssd is None
    # Verify surgical slap routing excludes gdn2_adapters (slap_target_groups == [loop_adapters, router])
    slap_params = resolve_slap_target_params(rl_tr.stack.named_trainable, rl_tr.sirsi.slap_target_groups)
    gdn2_param_ids = {id(p) for p in rl_tr.stack.gdn2.trainable_parameters()}
    assert not any(id(p) in gdn2_param_ids for p in slap_params)
    # Verify optim lr_mult for gdn2_adapter is 2.0
    s4_gdn2_entries = [e for e in rl_tr.opt.entries.values() if "gdn2_adapter." in e.name]
    assert s4_gdn2_entries
    assert all(e.base_lr == pytest.approx(s4_cfg["train"]["optim"]["lr"] * 2.0) for e in s4_gdn2_entries)

    rl_tr._resume()
    for ad in rl_tr.stack.gdn2.adapters.values():
        assert ad.qed_lambda_raw is not None and ad.neg_gamma_raw is not None
        torch.testing.assert_close(ad.qed_lambda_raw.detach(), torch.full_like(ad.qed_lambda_raw, 0.25))
        torch.testing.assert_close(ad.neg_gamma_raw.detach(), torch.full_like(ad.neg_gamma_raw, 0.30))

    out_s4 = rl_tr.train()
    assert out_s4["step"] == 1
    assert "gdn2/qed_lambda_abs_mean" in out_s4["last"]
    ck_s4 = CheckpointManager(s4_dir, keep_last=0).load()
    assert ck_s4 is not None
    assert ck_s4["state"]["stack"]["gdn2"]["qed_enabled"] is True

    # --- Stage 4: Eval `_stack_with_ckpt` & `_pipeline_stage_cmds` + Hub `load_qloop_from_hub`
    eval_cfg = _toy_stage_cfg("s4_v2_rl", checkpoint=False)
    # Deliberately mimic `configs/eval/suite.yaml` (gdn2.enabled=false, hssd.enabled=true) to verify auto-alignment!
    eval_cfg["gdn2"] = {"enabled": False}
    eval_cfg["hssd"] = {"enabled": True, "rank": 8, "chunk_size": 4, "window_chunks": 2, "num_pages": 4, "k_pages": 2}
    st_eval = _stack_with_ckpt(Cfg(eval_cfg), s4_dir)
    assert st_eval.hssd is None and st_eval.gdn2 is not None
    assert st_eval.gdn2.cfg.qed_enabled is True
    assert st_eval.gdn2.cfg.decay_log_min == pytest.approx(-3.0)
    assert st_eval.gdn2.cfg.neg_eigval_max_boost == pytest.approx(0.5)
    for lid, ad in st_eval.gdn2.adapters.items():
        ref_ad = rl_tr.stack.gdn2.adapters[lid]
        assert ad.qed_lambda_raw is not None and ref_ad.qed_lambda_raw is not None
        torch.testing.assert_close(ad.qed_lambda_raw, ref_ad.qed_lambda_raw)

    # `_pipeline_stage_cmds` for v2 pipeline `eval` enables GDN-2-1M + QED for s4_v2_rl, disables for s3_v2-only
    cmds_eval_s4 = _pipeline_stage_cmds(
        "eval",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=(*V3_GDN_PIPELINE_ORDER, "eval"),
    )
    flat_eval_s4 = " ".join(cmds_eval_s4[0])
    assert "gdn2.enabled=true" in flat_eval_s4
    assert "gdn2.qed_enabled=true" in flat_eval_s4
    assert "gdn2.decay_log_min=-3.0" in flat_eval_s4
    assert "gdn2.neg_eigval_max_boost=0.5" in flat_eval_s4
    assert "hssd.enabled=false" in flat_eval_s4

    cmds_eval_s3_only = _pipeline_stage_cmds(
        "eval",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=("s3_v2_long_1m_lsa", "eval"),
    )
    assert "gdn2.enabled=false" in " ".join(cmds_eval_s3_only[0])

    # Hub push + `load_qloop_from_hub` round-trip preserves full GDN-2-1M + QED metadata & weights
    api = _FakeHubAPI()
    hub = HubSyncManager("TassP0lar/test-gdn2-1m-qed", token="hf_test", api=api)
    oid = hub.push_checkpoint(
        "s4_v2_rl",
        ck_s4["path"],
        completed=True,
        target_steps=1,
        config_dict=s4_cfg,
    )
    assert oid is not None
    manifest = hub.fetch_manifest()
    assert manifest.architecture["linear_attention"] == "gdn2"
    assert manifest.architecture["gdn2_qed"] is True
    assert manifest.architecture["gdn2_neg_eigval_max_boost"] == pytest.approx(0.5)
    assert manifest.architecture["gdn2_decay_log_min"] == pytest.approx(-3.0)
    assert manifest.architecture["gdn2_decay_log_max"] == pytest.approx(0.5)

    st_hub = load_qloop_from_hub("TassP0lar/test-gdn2-1m-qed", stage_name="s4_v2_rl", api=api)
    assert st_hub.gdn2 is not None and st_hub.hssd is None
    assert st_hub.gdn2.cfg.qed_enabled is True
    assert st_hub.gdn2.cfg.decay_log_min == pytest.approx(-3.0)
    assert st_hub.gdn2.cfg.neg_eigval_max_boost == pytest.approx(0.5)
    for lid, ad in st_hub.gdn2.adapters.items():
        ref_ad = rl_tr.stack.gdn2.adapters[lid]
        assert ad.qed_lambda_raw is not None and ref_ad.qed_lambda_raw is not None
        torch.testing.assert_close(ad.qed_lambda_raw, ref_ad.qed_lambda_raw)


def test_real_stage_configs_validation_and_untouched_invariants() -> None:
    """Validate real production configs: `s2_v2` & `s4_v2` GDN-2-1M+QED; `s3_v2` & old HSSD untouched."""
    s3 = load_config(str(ROOT / "configs" / "stages" / "s3_v2_long_1m_lsa.yaml"))
    assert s3["gdn2"]["enabled"] is False
    assert s3["train"]["train_gdn2"] is False

    for name, expected_lr_mult in (("s2_v2_joint_bptt", 5.0), ("s4_v2_rl", 2.0)):
        cfg = load_config(str(ROOT / "configs" / "stages" / f"{name}.yaml"))
        g = cfg["gdn2"]
        assert g["enabled"] is True
        gcfg = GDN2Config.from_dict(g)
        gcfg.validate()
        assert gcfg.rank == 64
        assert gcfg.chunk_size == 64
        assert gcfg.decay_log_min == pytest.approx(-3.0)
        assert gcfg.decay_log_max == pytest.approx(0.5)
        assert gcfg.qed_enabled is True
        assert gcfg.qed_max_lambda == pytest.approx(0.35)
        assert gcfg.neg_eigval_max_boost == pytest.approx(0.5)
        assert gcfg.backend == "auto"
        assert cfg["train"]["train_gdn2"] is True
        assert "gdn2_adapters" in cfg["train"]["allowed_groups"]
        assert cfg["train"]["optim"]["lr_mult"]["gdn2_adapter"] == pytest.approx(expected_lr_mult)
        assert cfg["hssd"]["enabled"] is False

    s4 = load_config(str(ROOT / "configs" / "stages" / "s4_v2_rl.yaml"))
    assert s4["rl"]["sirsi"]["slap_target_groups"] == ["loop_adapters", "router"]

    # Old HSSD configs untouched
    for old_stage in ("s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        ocfg = load_config(str(ROOT / "configs" / "stages" / f"{old_stage}.yaml"))
        assert ocfg["hssd"]["enabled"] is True
        assert ocfg["gdn2"]["enabled"] is False
