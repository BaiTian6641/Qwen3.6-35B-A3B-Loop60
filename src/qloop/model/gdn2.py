"""NVIDIA Gated DeltaNet-2 (GDN-2) + 1M Asymmetric Slow-Decay + Bounded QED + Soft Negative Eigenvalues.

Reference:
    Ali Hatamizadeh, Yejin Choi, Jan Kautz (NVIDIA Research),
    "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791,
    `NVlabs/GatedDeltaNet-2`).

Why GDN-2-1M + Bounded QED + Soft Neg-Eigval upgrades stock Gated DeltaNet cleanly on Qwen3.6-35B-A3B:
  * Stock Gated DeltaNet ties memory erasure (key axis) and memory writing (value axis) to one
    shared scalar gate `beta_t in (0, 1)` per head alongside a scalar decay `g_t <= 0`:
        S_t = (I - beta_t k_t k_t^T) exp(g_t) S_{t-1} + beta_t k_t v_t^T
  * GDN-2 decouples these into three independent channel-wise vectors per head:
      1. Channel-wise key-side erase gate   b_t in (0, 1.5]^{d_k}  (soft negative eigenvalues via `b_scale`)
      2. Channel-wise value-side write gate w_t in (0, 1)^{d_v}
      3. Channel-wise log-decay             g_t in (-inf, 0]^{d_k}, D_t = Diag(exp(g_t))
         with optional `C^1`-continuous asymmetric `tanh` bound `[decay_log_min, decay_log_max]`
         (e.g. `[-3.0, +0.5]`, stretching memory horizon up to `exp(3) ~= 20.1x` for 1M contexts).
  * Bounded Query-Enhanced Erasure Direction (QED):
      With per-head bounded `lam_h = qed_max_lambda * tanh(qed_lambda_raw_h)` (`0.0` at step 0),
      the rank-1 erase direction adds a key-orthogonal query projection using UNSCALED unit-norm
      `(q_hat, k_hat)`:
          e_t = b_t * k_hat + lam_h * (b_t * q_hat - <k_hat, b_t * q_hat> * k_hat)
          S_t = (I - k_hat e_t^T) Diag(exp(g_t)) S_{t-1} + k_hat (w_t * v_t)^T
          o_t = S_t^T (q_hat * scale)
  * Strict generalization & 100% IEEE-754 step-0 bit-identity:
      All residual projections (`erase_up`, `write_up`, `decay_up`) and per-head scalars
      (`qed_lambda_raw`, `neg_gamma_raw`) start at exact zero, so at step 0 `b_t == beta_t`,
      `w_t == beta_t`, `g_t == g_base_t`, `lam_h == 0`, `b_scale == 1.0`, matching stock
      Qwen3.6 Gated DeltaNet bit-identically.
  * Hardware-efficient Blackwell `sm_120` execution via `fla` DPLR / `fla.ops.gdn2` + Fused Triton Prep:
      - `fla.ops.generalized_delta_rule.dplr.{chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule}`
        implements `S_t = (Diag(exp(gk_t)) + b_t a_t^T) S_{t-1} + k_t v_t^T`, `o_t = S_t^T (q_t * scale)`
        with exact 1-to-1 mapping:
            q = q_hat, k = k_hat, v = w_t * v_raw, a = -(e_t * exp(g_t)), b = k_hat, gk = g_t,
            scale = dk ** -0.5 (never double-scaling `q_hat`).
        `fused_recurrent_dplr_delta_rule` is inference-only in `fla 0.6.0` (raises `NotImplementedError`
        on backward), so it is dispatched ONLY when `T == 1` and `not torch.is_grad_enabled()`;
        `chunk_dplr_delta_rule` handles all `T > 1` and grad-enabled calls.
      - `fla.ops.gdn2.{chunk_gdn2, fused_recurrent_gdn2}` is dispatched for no-QED GDN-2 on CUDA.
      - A fused single-pass Triton prep autograd kernel computes `(q_hat, k_hat, v_w, a, gk)` and
        its exact analytical backward in minimal HBM passes (with clean PyTorch fallback on CPU,
        `float64`, `torch.compile`, or when Triton/`fla` is unavailable).
"""
from __future__ import annotations

import functools
import logging
import math
import sys
import types
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..debug import require
from ._dtype import upcast
from .introspect import LINEAR, attention_module, inspect_model

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU/macOS environments without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

log = logging.getLogger("qloop")

_VALID_BACKENDS = frozenset({"auto", "dplr", "fla_gdn2", "pytorch"})


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _is_compiling() -> bool:
    comp = getattr(torch, "compiler", None)
    fn = getattr(comp, "is_compiling", None) if comp is not None else None
    if callable(fn):
        return bool(fn())
    dyn = getattr(torch, "_dynamo", None)
    fn2 = getattr(dyn, "is_compiling", None) if dyn is not None else None
    return bool(fn2()) if callable(fn2) else False


# ============================================================================ config

@dataclass
class GDN2Config:
    rank: int = 64
    layers: list[int] | None = None  # physical linear-attention layers; None = all 30 of them
    chunk_size: int = 64
    segment_chunks: int = 16         # 16 * 64 = 1024 tokens per checkpointable WY macro-segment
    checkpoint_segments: bool = True
    fla_segment_len: int = 8192      # checkpointed macro-segment length for `fla` training (0 disables; >= 64)
    gate_logit_cap: float = 8.0      # smooth tanh bound on channel-wise erase/write logit shifts
    decay_log_cap: float = 1.0       # symmetric fallback bound on log-decay multiplier exp(+-cap)
    decay_log_min: float | None = None   # optional asymmetric min log-decay shift (< 0, e.g. -3.0 => 20.1x horizon)
    decay_log_max: float | None = None   # optional asymmetric max log-decay shift (> 0, e.g. +0.5)
    qed_enabled: bool = False        # bounded per-head Query-Enhanced Erasure Direction (QED)
    qed_max_lambda: float = 0.35     # smooth tanh bound on per-head QED coupling `lam_h`
    neg_eigval_max_boost: float = 0.0  # per-head erase scale `1 + boost*tanh(raw)` (0.5 => (0.5, 1.5))
    backend: str = "auto"            # "auto" | "dplr" | "fla_gdn2" | "pytorch"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> GDN2Config:
        clean = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        known = {f.name for f in fields(cls)}
        bad = set(clean) - known
        if bad:
            raise ValueError(f"unknown gdn2.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**clean)

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"gdn2.rank must be >= 1, got {self.rank}")
        if self.chunk_size < 1:
            raise ValueError(f"gdn2.chunk_size must be >= 1, got {self.chunk_size}")
        if self.segment_chunks < 1:
            raise ValueError(f"gdn2.segment_chunks must be >= 1, got {self.segment_chunks}")
        if (
            isinstance(self.fla_segment_len, bool)
            or not isinstance(self.fla_segment_len, int)
            or (self.fla_segment_len != 0 and self.fla_segment_len < 64)
        ):
            raise ValueError(f"gdn2.fla_segment_len must be 0 (disabled) or >= 64, got {self.fla_segment_len}")
        if self.gate_logit_cap <= 0 or self.decay_log_cap <= 0:
            raise ValueError("gdn2.gate_logit_cap and gdn2.decay_log_cap must be > 0")
        if self.decay_log_min is not None and float(self.decay_log_min) >= 0.0:
            raise ValueError(f"gdn2.decay_log_min must be < 0, got {self.decay_log_min}")
        if self.decay_log_max is not None and float(self.decay_log_max) <= 0.0:
            raise ValueError(f"gdn2.decay_log_max must be > 0, got {self.decay_log_max}")
        if float(self.qed_max_lambda) < 0.0:
            raise ValueError(f"gdn2.qed_max_lambda must be >= 0, got {self.qed_max_lambda}")
        if not (0.0 <= float(self.neg_eigval_max_boost) <= 1.0):
            raise ValueError(f"gdn2.neg_eigval_max_boost must be in [0, 1], got {self.neg_eigval_max_boost}")
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(f"gdn2.backend must be one of {sorted(_VALID_BACKENDS)}, got {self.backend!r}")


# ============================================================================ math primitives

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA / HF compatible L2 normalization accumulated in fp32/fp64."""
    xf = upcast(x)
    return xf * torch.rsqrt((xf * xf).sum(dim=dim, keepdim=True) + eps)


def asymmetric_tanh_bound(x: torch.Tensor, cap_min: float, cap_max: float) -> torch.Tensor:
    """`C^1`-continuous asymmetric `tanh` bound mapping `R -> (cap_min, cap_max)` (`cap_min < 0 < cap_max`).

    Uses `torch.where(x >= 0, cap_max * tanh(x / cap_max), (-cap_min) * tanh(x / (-cap_min)))`:
      * Bit-exact `0.0` at `x == 0.0` (step-0 identity).
      * Continuous unit derivative `f'(0+) == f'(0-) == 1.0`.
      * Stretches memory horizon up to `exp(-cap_min)` (e.g. `exp(3.0) ~= 20.1x` at `cap_min = -3.0`).
    """
    neg_span = -float(cap_min)
    pos_span = float(cap_max)
    return torch.where(
        x >= 0,
        pos_span * torch.tanh(x / pos_span),
        neg_span * torch.tanh(x / neg_span),
    )


def logistic_shift(beta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Exact singularity-free `sigmoid(logit(beta) + delta)` for `beta in [0, 1]`.

    Uses `beta + 2*beta*(1 - beta)*s / (1 + (2*beta - 1)*s)` with `s = tanh(delta / 2)`:
      * At `delta == 0`: `s == 0.0` -> returns `beta` bit-exactly in IEEE-754 floating point.
      * At `beta == 0` or `beta == 1` (e.g. MoR write-masked tokens): returns `0` or `1` with
        zero `log(0)` / `inf` / `NaN` singularities.
      * Derivative w.r.t. `delta` at 0 is `beta * (1 - beta)`, matching logistic shift identically.
    """
    s = torch.tanh(0.5 * delta)
    denom = (1.0 + (2.0 * beta - 1.0) * s).clamp_min(1e-12)
    return (beta + (2.0 * beta * (1.0 - beta) * s) / denom).clamp(0.0, 1.0)


def _broadcast_qed_lambda(lam: torch.Tensor, hv: int) -> torch.Tensor:
    if lam.ndim == 1:
        require(lam.shape[0] == hv, "qed_lambda length must equal num_v_heads", lam=lam, hv=hv)
        return lam.view(1, 1, hv, 1)
    if lam.ndim == 4:
        return lam
    if lam.ndim == 3:
        return lam.unsqueeze(-1)
    if lam.ndim == 0:
        return lam.view(1, 1, 1, 1)
    raise ValueError(f"unsupported qed_lambda shape {tuple(lam.shape)}; expected [H_v] or broadcastable to [B,T,H_v,1]")


def _compute_qed_erase(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    b_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None,
) -> torch.Tensor:
    """Compute GDN-2 + QED erase vector `e_t = b_t * k_hat + lam * (b_t * q_hat - <k_hat, b_t * q_hat> * k_hat)`.

    IMPORTANT: `q_hat` and `k_hat` MUST be unscaled unit-norm vectors (never pre-scaled by `dk ** -0.5`).
    """
    e_t = b_gate * k_hat
    if qed_lambda is None:
        return e_t
    lam = _broadcast_qed_lambda(upcast(qed_lambda).to(dtype=e_t.dtype, device=e_t.device), k_hat.shape[2])
    bq = b_gate * q_hat
    dot = (k_hat * bq).sum(dim=-1, keepdim=True)
    return e_t + lam * (bq - dot * k_hat)


def _prepare_qkv_unscaled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = upcast(query), upcast(key), upcast(value)
    hv = v.shape[2]
    if q.shape[2] != hv:
        require(hv % q.shape[2] == 0, "num_v_heads must be divisible by num_q_heads", q=q, v=v)
        q = q.repeat_interleave(hv // q.shape[2], dim=2)
    if k.shape[2] != hv:
        require(hv % k.shape[2] == 0, "num_v_heads must be divisible by num_k_heads", k=k, v=v)
        k = k.repeat_interleave(hv // k.shape[2], dim=2)
    if use_qk_l2norm:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    return q, k, v


# ============================================================================ fused QED+DPLR prep (Triton + PyTorch)

def qed_dplr_prep_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None = None,
    *,
    use_qk_l2norm: bool = False,
    out_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for QED+DPLR prep: returns `(q_hat, k_hat, v_w, a, gk)`.

    Maps GDN-2 + QED parameters onto `fla`'s DPLR recurrence
        `S_t = (Diag(exp(gk_t)) + b_t a_t^T) S_{t-1} + k_t v_t^T`
    with `b_t = k_hat`, `a_t = -(e_t * exp(gk_t))`, `v_w = w_t * v_raw`, `gk_t = clamp(g_vec, max=0.0)`,
    and unscaled unit-norm `q_hat, k_hat` (do NOT pre-scale `q_hat`).
    """
    dst_dt = query.dtype if out_dtype is None else out_dtype
    q_f, k_f, v_f = upcast(query), upcast(key), upcast(value)
    hv = v_f.shape[2]
    if q_f.shape[2] != hv:
        require(hv % q_f.shape[2] == 0, "num_v_heads must be divisible by num_q_heads", q=q_f, v=v_f)
        q_f = q_f.repeat_interleave(hv // q_f.shape[2], dim=2)
    if k_f.shape[2] != hv:
        require(hv % k_f.shape[2] == 0, "num_v_heads must be divisible by num_k_heads", k=k_f, v=v_f)
        k_f = k_f.repeat_interleave(hv // k_f.shape[2], dim=2)
    if use_qk_l2norm:
        q_hat = q_f * torch.rsqrt((q_f * q_f).sum(dim=-1, keepdim=True) + eps)
        k_hat = k_f * torch.rsqrt((k_f * k_f).sum(dim=-1, keepdim=True) + eps)
    else:
        q_hat = q_f
        k_hat = k_f

    gk = upcast(g_vec).clamp(max=0.0)
    bf = upcast(b_gate)
    wf = upcast(w_gate)
    e_t = _compute_qed_erase(q_hat, k_hat, bf, qed_lambda)
    a = -(e_t * gk.exp())
    v_w = wf * v_f
    gk_dt = torch.float64 if dst_dt == torch.float64 else torch.float32
    return (
        q_hat.to(dst_dt).contiguous(),
        k_hat.to(dst_dt).contiguous(),
        v_w.to(dst_dt).contiguous(),
        a.to(dst_dt).contiguous(),
        gk.to(gk_dt).contiguous(),
    )


def _qed_dplr_prep_bwd_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None,
    dq_hat: torch.Tensor | None,
    dk_hat: torch.Tensor | None,
    dv_w: torch.Tensor | None,
    da: torch.Tensor | None,
    dgk: torch.Tensor | None,
    *,
    use_qk_l2norm: bool = False,
    eps: float = 1e-6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Exact analytical backward matching `_qed_dplr_prep_bwd_kernel` line-for-line (for CPU/fp64 verification)."""
    q_f, k_f, v_f = upcast(query), upcast(key), upcast(value)
    bsz, seq_len, hq, dk = q_f.shape
    hk = k_f.shape[2]
    hv = v_f.shape[2]
    q_exp = q_f.repeat_interleave(hv // hq, dim=2) if hq != hv else q_f
    k_exp = k_f.repeat_interleave(hv // hk, dim=2) if hk != hv else k_f
    if use_qk_l2norm:
        r_q = torch.rsqrt((q_exp * q_exp).sum(dim=-1, keepdim=True) + eps)
        r_k = torch.rsqrt((k_exp * k_exp).sum(dim=-1, keepdim=True) + eps)
        qh = q_exp * r_q
        kh = k_exp * r_k
    else:
        r_q = None
        r_k = None
        qh = q_exp
        kh = k_exp

    gf = upcast(g_vec)
    bf = upcast(b_gate)
    wf = upcast(w_gate)

    has_qed = qed_lambda is not None
    if has_qed:
        assert qed_lambda is not None
        lam = _broadcast_qed_lambda(upcast(qed_lambda).to(dtype=qh.dtype, device=qh.device), hv)
        u = bf * qh
        c = (kh * u).sum(dim=-1, keepdim=True)
        p = u - c * kh
        e_val = bf * kh + lam * p
    else:
        lam = None
        u = None
        c = None
        p = None
        e_val = bf * kh

    g_le0 = gf <= 0.0
    g_clamped = torch.where(g_le0, gf, torch.zeros_like(gf))
    gamma = g_clamped.exp()
    a_val = -(e_val * gamma)

    dqh_in = torch.zeros_like(qh) if dq_hat is None else upcast(dq_hat)
    dkh_in = torch.zeros_like(kh) if dk_hat is None else upcast(dk_hat)
    da_in = torch.zeros_like(a_val) if da is None else upcast(da)
    dgk_in = torch.zeros_like(g_clamped) if dgk is None else upcast(dgk)
    dvw_in = torch.zeros_like(v_f) if dv_w is None else upcast(dv_w)

    de = -da_in * gamma
    dg_clamped = dgk_in + da_in * a_val
    dg = torch.where(g_le0, dg_clamped, torch.zeros_like(dg_clamped))

    if has_qed:
        assert lam is not None and u is not None and c is not None and p is not None and qed_lambda is not None
        dp = lam * de
        dc = -(dp * kh).sum(dim=-1, keepdim=True)
        du = dp + dc * kh
        db = de * kh + du * qh
        dkh_tot = dkh_in + de * bf - c * dp + dc * u
        dqh_tot = dqh_in + du * bf
        dlam_tok = (de * p).sum(dim=-1)  # [B, T, H_v]
        if qed_lambda.ndim == 1:
            dlam: torch.Tensor | None = dlam_tok.sum(dim=(0, 1)).to(qed_lambda.dtype)
        elif qed_lambda.ndim == 0:
            dlam = dlam_tok.sum().to(qed_lambda.dtype)
        else:
            dlam_4d = dlam_tok.unsqueeze(-1)
            reduce_dims = tuple(i for i, s in enumerate(qed_lambda.shape) if s == 1)
            dlam = (dlam_4d.sum(dim=reduce_dims, keepdim=True) if reduce_dims else dlam_4d).to(qed_lambda.dtype)
    else:
        db = de * kh
        dkh_tot = dkh_in + de * bf
        dqh_tot = dqh_in
        dlam = None

    if use_qk_l2norm:
        assert r_q is not None and r_k is not None
        dq_hv = r_q * (dqh_tot - (dqh_tot * qh).sum(dim=-1, keepdim=True) * qh)
        dk_hv = r_k * (dkh_tot - (dkh_tot * kh).sum(dim=-1, keepdim=True) * kh)
    else:
        dq_hv = dqh_tot
        dk_hv = dkh_tot

    dq = dq_hv.view(bsz, seq_len, hq, hv // hq, dk).sum(dim=3) if hq != hv else dq_hv
    dk_out = dk_hv.view(bsz, seq_len, hk, hv // hk, dk).sum(dim=3) if hk != hv else dk_hv
    dv = dvw_in * wf
    dw = dvw_in * v_f
    return (
        dq.to(query.dtype),
        dk_out.to(key.dtype),
        dv.to(value.dtype),
        dg.to(g_vec.dtype),
        db.to(b_gate.dtype),
        dw.to(w_gate.dtype),
        dlam,
    )


if _HAS_TRITON:

    @triton.jit
    def _qed_dplr_prep_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        G_ptr,
        B_ptr,
        W_ptr,
        LAM_ptr,
        QH_ptr,
        KH_ptr,
        VW_ptr,
        A_ptr,
        GK_ptr,
        EPS,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        H_V: tl.constexpr,
        Q_GROUP: tl.constexpr,
        K_GROUP: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
        HAS_QED: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)

        hq = pid_h // Q_GROUP
        hk = pid_h // K_GROUP

        offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = offs_k < DK

        base_q = (pid_bt * H_Q + hq) * DK
        base_k = (pid_bt * H_K + hk) * DK
        base_hv_k = (pid_bt * H_V + pid_h) * DK
        base_hv_v = (pid_bt * H_V + pid_h) * DV

        q_val = tl.load(Q_ptr + base_q + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_val = tl.load(K_ptr + base_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_val = tl.load(G_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        if USE_QK_L2NORM:
            r_q = tl.rsqrt(tl.sum(q_val * q_val, axis=0) + EPS)
            r_k = tl.rsqrt(tl.sum(k_val * k_val, axis=0) + EPS)
            qh = q_val * r_q
            kh = k_val * r_k
        else:
            qh = q_val
            kh = k_val

        if HAS_QED:
            lam = tl.load(LAM_ptr + pid_h).to(tl.float32)
            bq = b_val * qh
            dot_kbq = tl.sum(kh * bq, axis=0)
            e_val = b_val * kh + lam * (bq - dot_kbq * kh)
        else:
            e_val = b_val * kh

        g_clamped = tl.minimum(g_val, 0.0)
        a_val = -(e_val * tl.exp(g_clamped))

        tl.store(QH_ptr + base_hv_k + offs_k, qh.to(QH_ptr.dtype.element_ty), mask=mask_k)
        tl.store(KH_ptr + base_hv_k + offs_k, kh.to(KH_ptr.dtype.element_ty), mask=mask_k)
        tl.store(A_ptr + base_hv_k + offs_k, a_val.to(A_ptr.dtype.element_ty), mask=mask_k)
        tl.store(GK_ptr + base_hv_k + offs_k, g_clamped.to(GK_ptr.dtype.element_ty), mask=mask_k)

        offs_v = tl.arange(0, BLOCK_V).to(tl.int64)
        mask_v = offs_v < DV
        v_val = tl.load(V_ptr + base_hv_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        w_val = tl.load(W_ptr + base_hv_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        vw_val = w_val * v_val
        tl.store(VW_ptr + base_hv_v + offs_v, vw_val.to(VW_ptr.dtype.element_ty), mask=mask_v)

    @triton.jit
    def _qed_dplr_prep_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        G_ptr,
        B_ptr,
        W_ptr,
        LAM_ptr,
        DQH_ptr,
        DKH_ptr,
        DVW_ptr,
        DA_ptr,
        DGK_ptr,
        DQ_ptr,
        DK_ptr,
        DV_ptr,
        DG_ptr,
        DB_ptr,
        DW_ptr,
        DLAM_BUF_ptr,
        EPS,
        H_Q: tl.constexpr,
        H_K: tl.constexpr,
        H_V: tl.constexpr,
        Q_GROUP: tl.constexpr,
        K_GROUP: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
        HAS_QED: tl.constexpr,
        HAS_DQH: tl.constexpr,
        HAS_DKH: tl.constexpr,
        HAS_DVW: tl.constexpr,
        HAS_DA: tl.constexpr,
        HAS_DGK: tl.constexpr,
        NEED_DQ: tl.constexpr,
        NEED_DK: tl.constexpr,
        NEED_DV: tl.constexpr,
        NEED_DG: tl.constexpr,
        NEED_DB: tl.constexpr,
        NEED_DW: tl.constexpr,
        NEED_DLAM: tl.constexpr,
    ):
        pid_bt = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)

        hq = pid_h // Q_GROUP
        hk = pid_h // K_GROUP

        offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = offs_k < DK

        base_q = (pid_bt * H_Q + hq) * DK
        base_k = (pid_bt * H_K + hk) * DK
        base_hv_k = (pid_bt * H_V + pid_h) * DK
        base_hv_v = (pid_bt * H_V + pid_h) * DV

        q_val = tl.load(Q_ptr + base_q + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_val = tl.load(K_ptr + base_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g_val = tl.load(G_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        if USE_QK_L2NORM:
            r_q = tl.rsqrt(tl.sum(q_val * q_val, axis=0) + EPS)
            r_k = tl.rsqrt(tl.sum(k_val * k_val, axis=0) + EPS)
            qh = q_val * r_q
            kh = k_val * r_k
        else:
            r_q = 1.0
            r_k = 1.0
            qh = q_val
            kh = k_val

        if HAS_QED:
            lam = tl.load(LAM_ptr + pid_h).to(tl.float32)
            u = b_val * qh
            c = tl.sum(kh * u, axis=0)
            p = u - c * kh
            e_val = b_val * kh + lam * p
        else:
            lam = 0.0
            u = tl.zeros([BLOCK_K], dtype=tl.float32)
            c = 0.0
            p = tl.zeros([BLOCK_K], dtype=tl.float32)
            e_val = b_val * kh

        g_le0 = g_val <= 0.0
        g_clamped = tl.where(g_le0, g_val, 0.0)
        gamma = tl.exp(g_clamped)
        a_val = -(e_val * gamma)

        if HAS_DQH:
            dqh = tl.load(DQH_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        else:
            dqh = tl.zeros([BLOCK_K], dtype=tl.float32)
        if HAS_DKH:
            dkh = tl.load(DKH_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        else:
            dkh = tl.zeros([BLOCK_K], dtype=tl.float32)
        if HAS_DA:
            da = tl.load(DA_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        else:
            da = tl.zeros([BLOCK_K], dtype=tl.float32)
        if HAS_DGK:
            dgk = tl.load(DGK_ptr + base_hv_k + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        else:
            dgk = tl.zeros([BLOCK_K], dtype=tl.float32)

        de = -da * gamma
        dg_clamped = dgk + da * a_val
        dg = tl.where(g_le0, dg_clamped, 0.0)

        if HAS_QED:
            dp = lam * de
            dc = -tl.sum(dp * kh, axis=0)
            du = dp + dc * kh
            db = de * kh + du * qh
            dkh_tot = dkh + de * b_val - c * dp + dc * u
            dqh_tot = dqh + du * b_val
            if NEED_DLAM:
                dlam_tok = tl.sum(de * p, axis=0)
                tl.store(DLAM_BUF_ptr + pid_bt * H_V + pid_h, dlam_tok)
        else:
            db = de * kh
            dkh_tot = dkh + de * b_val
            dqh_tot = dqh

        if USE_QK_L2NORM:
            dq = r_q * (dqh_tot - tl.sum(dqh_tot * qh, axis=0) * qh)
            dk = r_k * (dkh_tot - tl.sum(dkh_tot * kh, axis=0) * kh)
        else:
            dq = dqh_tot
            dk = dkh_tot

        if NEED_DQ:
            tl.store(DQ_ptr + base_hv_k + offs_k, dq.to(DQ_ptr.dtype.element_ty), mask=mask_k)
        if NEED_DK:
            tl.store(DK_ptr + base_hv_k + offs_k, dk.to(DK_ptr.dtype.element_ty), mask=mask_k)
        if NEED_DG:
            tl.store(DG_ptr + base_hv_k + offs_k, dg.to(DG_ptr.dtype.element_ty), mask=mask_k)
        if NEED_DB:
            tl.store(DB_ptr + base_hv_k + offs_k, db.to(DB_ptr.dtype.element_ty), mask=mask_k)

        if NEED_DV or NEED_DW:
            offs_v = tl.arange(0, BLOCK_V).to(tl.int64)
            mask_v = offs_v < DV
            if HAS_DVW:
                dvw = tl.load(DVW_ptr + base_hv_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            else:
                dvw = tl.zeros([BLOCK_V], dtype=tl.float32)
            if NEED_DV:
                w_val = tl.load(W_ptr + base_hv_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                tl.store(DV_ptr + base_hv_v + offs_v, (dvw * w_val).to(DV_ptr.dtype.element_ty), mask=mask_v)
            if NEED_DW:
                v_val = tl.load(V_ptr + base_hv_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
                tl.store(DW_ptr + base_hv_v + offs_v, (dvw * v_val).to(DW_ptr.dtype.element_ty), mask=mask_v)

    def _launch_qed_dplr_prep_fwd(
        q_c: torch.Tensor,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        g_c: torch.Tensor,
        b_c: torch.Tensor,
        w_c: torch.Tensor,
        lam_1d: torch.Tensor | None,
        *,
        use_qk_l2norm: bool,
        out_dtype: torch.dtype,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, hq, dk = q_c.shape
        hk = k_c.shape[2]
        hv, dv = v_c.shape[2], v_c.shape[3]
        has_qed = lam_1d is not None
        lam_arg = lam_1d.to(torch.float32).contiguous() if has_qed else q_c.new_empty(1, dtype=torch.float32)

        q_hat = torch.empty((bsz, seq_len, hv, dk), device=q_c.device, dtype=out_dtype)
        k_hat = torch.empty((bsz, seq_len, hv, dk), device=q_c.device, dtype=out_dtype)
        v_w = torch.empty((bsz, seq_len, hv, dv), device=q_c.device, dtype=out_dtype)
        a_out = torch.empty((bsz, seq_len, hv, dk), device=q_c.device, dtype=out_dtype)
        gk_out = torch.empty((bsz, seq_len, hv, dk), device=q_c.device, dtype=torch.float32)

        block_k = max(16, triton.next_power_of_2(dk))
        block_v = max(16, triton.next_power_of_2(dv))
        num_warps = 4 if max(block_k, block_v) <= 256 else 8
        _qed_dplr_prep_fwd_kernel[(bsz * seq_len, hv)](
            q_c,
            k_c,
            v_c,
            g_c,
            b_c,
            w_c,
            lam_arg,
            q_hat,
            k_hat,
            v_w,
            a_out,
            gk_out,
            float(eps),
            H_Q=hq,
            H_K=hk,
            H_V=hv,
            Q_GROUP=hv // hq,
            K_GROUP=hv // hk,
            DK=dk,
            DV=dv,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            USE_QK_L2NORM=use_qk_l2norm,
            HAS_QED=has_qed,
            num_warps=num_warps,
            num_stages=1,
        )
        return q_hat, k_hat, v_w, a_out, gk_out

    class _TritonQEDDPLRPrepFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g_vec: torch.Tensor,
            b_gate: torch.Tensor,
            w_gate: torch.Tensor,
            qed_lambda: torch.Tensor | None,
            use_qk_l2norm: bool,
            out_dtype: torch.dtype,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            ctx.set_materialize_grads(False)
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            g_c = g_vec.contiguous()
            b_c = b_gate.contiguous()
            w_c = w_gate.contiguous()
            lam_1d = qed_lambda.reshape(v_c.shape[2]).contiguous() if qed_lambda is not None else None
            q_hat, k_hat, v_w, a_out, gk_out = _launch_qed_dplr_prep_fwd(
                q_c,
                k_c,
                v_c,
                g_c,
                b_c,
                w_c,
                lam_1d,
                use_qk_l2norm=use_qk_l2norm,
                out_dtype=out_dtype,
                eps=eps,
            )
            ctx.save_for_backward(q_c, k_c, v_c, g_c, b_c, w_c, lam_1d)
            ctx.use_qk_l2norm = bool(use_qk_l2norm)
            ctx.eps = float(eps)
            ctx.lam_shape = None if qed_lambda is None else tuple(qed_lambda.shape)
            ctx.lam_dtype = None if qed_lambda is None else qed_lambda.dtype
            return q_hat, k_hat, v_w, a_out, gk_out

        @staticmethod
        def backward(
            ctx: Any,
            dq_hat: torch.Tensor | None,
            dk_hat: torch.Tensor | None,
            dv_w: torch.Tensor | None,
            da: torch.Tensor | None,
            dgk: torch.Tensor | None,
        ) -> tuple[Any, ...]:
            need_dq, need_dk, need_dv, need_dg, need_db, need_dw, need_dlam = ctx.needs_input_grad[:7]
            if not any((need_dq, need_dk, need_dv, need_dg, need_db, need_dw, need_dlam)):
                return (None,) * 10

            q_c, k_c, v_c, g_c, b_c, w_c, lam_1d = ctx.saved_tensors
            bsz, seq_len, hq, dk = q_c.shape
            hk = k_c.shape[2]
            hv, dv = v_c.shape[2], v_c.shape[3]
            has_qed = lam_1d is not None
            need_dlam = bool(need_dlam and has_qed)

            dummy = q_c.new_empty(1, dtype=torch.float32)
            lam_arg = lam_1d.to(torch.float32).contiguous() if has_qed else dummy
            dqh_c = dq_hat.contiguous() if dq_hat is not None else dummy
            dkh_c = dk_hat.contiguous() if dk_hat is not None else dummy
            dvw_c = dv_w.contiguous() if dv_w is not None else dummy
            da_c = da.contiguous() if da is not None else dummy
            dgk_c = dgk.contiguous() if dgk is not None else dummy

            if need_dq:
                dq_buf = torch.empty(
                    (bsz, seq_len, hv, dk),
                    device=q_c.device,
                    dtype=torch.float32 if hq != hv else q_c.dtype,
                )
            else:
                dq_buf = dummy
            if need_dk:
                dk_buf = torch.empty(
                    (bsz, seq_len, hv, dk),
                    device=k_c.device,
                    dtype=torch.float32 if hk != hv else k_c.dtype,
                )
            else:
                dk_buf = dummy
            dv_out = torch.empty_like(v_c) if need_dv else dummy
            dg_out = torch.empty_like(g_c) if need_dg else dummy
            db_out = torch.empty_like(b_c) if need_db else dummy
            dw_out = torch.empty_like(w_c) if need_dw else dummy
            dlam_buf = torch.empty((bsz, seq_len, hv), device=q_c.device, dtype=torch.float32) if need_dlam else dummy

            block_k = max(16, triton.next_power_of_2(dk))
            block_v = max(16, triton.next_power_of_2(dv))
            num_warps = 4 if max(block_k, block_v) <= 256 else 8
            _qed_dplr_prep_bwd_kernel[(bsz * seq_len, hv)](
                q_c,
                k_c,
                v_c,
                g_c,
                b_c,
                w_c,
                lam_arg,
                dqh_c,
                dkh_c,
                dvw_c,
                da_c,
                dgk_c,
                dq_buf,
                dk_buf,
                dv_out,
                dg_out,
                db_out,
                dw_out,
                dlam_buf,
                float(ctx.eps),
                H_Q=hq,
                H_K=hk,
                H_V=hv,
                Q_GROUP=hv // hq,
                K_GROUP=hv // hk,
                DK=dk,
                DV=dv,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                USE_QK_L2NORM=ctx.use_qk_l2norm,
                HAS_QED=has_qed,
                HAS_DQH=dq_hat is not None,
                HAS_DKH=dk_hat is not None,
                HAS_DVW=dv_w is not None,
                HAS_DA=da is not None,
                HAS_DGK=dgk is not None,
                NEED_DQ=need_dq,
                NEED_DK=need_dk,
                NEED_DV=need_dv,
                NEED_DG=need_dg,
                NEED_DB=need_db,
                NEED_DW=need_dw,
                NEED_DLAM=need_dlam,
                num_warps=num_warps,
                num_stages=1,
            )

            dq_res: torch.Tensor | None = None
            if need_dq:
                dq_res = dq_buf.view(bsz, seq_len, hq, hv // hq, dk).sum(dim=3).to(q_c.dtype) if hq != hv else dq_buf
            dk_res: torch.Tensor | None = None
            if need_dk:
                dk_res = dk_buf.view(bsz, seq_len, hk, hv // hk, dk).sum(dim=3).to(k_c.dtype) if hk != hv else dk_buf
            dlam_res: torch.Tensor | None = None
            if need_dlam:
                assert ctx.lam_shape is not None and ctx.lam_dtype is not None
                dlam_res = dlam_buf.sum(dim=(0, 1)).to(ctx.lam_dtype).reshape(ctx.lam_shape)
            return (
                dq_res,
                dk_res,
                dv_out if need_dv else None,
                dg_out if need_dg else None,
                db_out if need_db else None,
                dw_out if need_dw else None,
                dlam_res,
                None,
                None,
                None,
            )


def _can_use_triton_prep(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> bool:
    if not _HAS_TRITON or _is_compiling() or not query.is_cuda:
        return False
    ok_dts = (torch.bfloat16, torch.float16, torch.float32)
    if out_dtype not in ok_dts:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    dev = query.device
    for t in (query, key, value, g_vec, b_gate, w_gate):
        if not t.is_cuda or t.device != dev or t.dtype not in ok_dts:
            return False
    bsz, seq_len, hq, dk = query.shape
    hk = key.shape[2]
    hv, dv = value.shape[2], value.shape[3]
    if bsz <= 0 or seq_len <= 0 or hq <= 0 or hk <= 0 or hv <= 0 or dk <= 0 or dv <= 0:
        return False
    if dk > 1024 or dv > 1024 or hv % hq != 0 or hv % hk != 0:
        return False
    if key.shape != (bsz, seq_len, hk, dk) or value.shape != (bsz, seq_len, hv, dv):
        return False
    if g_vec.shape != (bsz, seq_len, hv, dk) or b_gate.shape != (bsz, seq_len, hv, dk):
        return False
    if w_gate.shape != (bsz, seq_len, hv, dv):
        return False
    if qed_lambda is not None:
        if not qed_lambda.is_cuda or qed_lambda.device != dev or qed_lambda.dtype not in ok_dts:
            return False
        lam_shape = tuple(qed_lambda.shape)
        is_per_head = lam_shape in ((hv,), (1, 1, hv), (1, 1, hv, 1)) or (hv == 1 and qed_lambda.ndim == 0)
        if not is_per_head:
            return False
    return True


def qed_dplr_prep(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None = None,
    *,
    use_qk_l2norm: bool = False,
    out_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QED+DPLR prep returning `(q_hat, k_hat, v_w, a, gk)` via Triton on CUDA or PyTorch fallback."""
    dst_dt = query.dtype if out_dtype is None else out_dtype
    if _can_use_triton_prep(query, key, value, g_vec, b_gate, w_gate, qed_lambda, dst_dt):
        if not torch.is_grad_enabled():
            lam_1d = qed_lambda.reshape(value.shape[2]).contiguous() if qed_lambda is not None else None
            return _launch_qed_dplr_prep_fwd(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                g_vec.contiguous(),
                b_gate.contiguous(),
                w_gate.contiguous(),
                lam_1d,
                use_qk_l2norm=use_qk_l2norm,
                out_dtype=dst_dt,
                eps=eps,
            )
        return _TritonQEDDPLRPrepFn.apply(
            query,
            key,
            value,
            g_vec,
            b_gate,
            w_gate,
            qed_lambda,
            bool(use_qk_l2norm),
            dst_dt,
            float(eps),
        )
    return qed_dplr_prep_pytorch(
        query,
        key,
        value,
        g_vec,
        b_gate,
        w_gate,
        qed_lambda,
        use_qk_l2norm=use_qk_l2norm,
        out_dtype=dst_dt,
        eps=eps,
    )


# ============================================================================ FLA backend resolution & dispatch

_FLA_DPLR_CACHE: tuple[bool, Any, Any] | None = None
_FLA_GDN2_CACHE: tuple[bool, Any, Any] | None = None


def _get_fla_dplr_ops() -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    global _FLA_DPLR_CACHE
    if _FLA_DPLR_CACHE is None:
        try:
            from fla.ops.generalized_delta_rule.dplr import (  # type: ignore[import-not-found]
                chunk_dplr_delta_rule,
                fused_recurrent_dplr_delta_rule,
            )

            _FLA_DPLR_CACHE = (True, chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule)
        except Exception:
            _FLA_DPLR_CACHE = (False, None, None)
    ok, chunk_fn, rec_fn = _FLA_DPLR_CACHE
    return (chunk_fn, rec_fn) if ok else None


def _get_fla_gdn2_ops() -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    global _FLA_GDN2_CACHE
    if _FLA_GDN2_CACHE is None:
        try:
            from fla.ops.gdn2 import (  # type: ignore[import-not-found]
                chunk_gdn2 as fla_chunk_gdn2,
            )
            from fla.ops.gdn2 import (
                fused_recurrent_gdn2 as fla_fused_recurrent_gdn2,
            )

            _FLA_GDN2_CACHE = (True, fla_chunk_gdn2, fla_fused_recurrent_gdn2)
        except Exception:
            _FLA_GDN2_CACHE = (False, None, None)
    ok, chunk_fn, rec_fn = _FLA_GDN2_CACHE
    return (chunk_fn, rec_fn) if ok else None


def _is_fla_eligible_tensor(q: torch.Tensor) -> bool:
    return bool(q.is_cuda and q.dtype in (torch.bfloat16, torch.float16, torch.float32))


def _resolve_backend(backend: str, has_qed: bool, query: torch.Tensor) -> str:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"gdn2.backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}")
    if backend == "pytorch" or not _is_fla_eligible_tensor(query):
        return "pytorch"
    if backend == "dplr":
        return "dplr" if _get_fla_dplr_ops() is not None else "pytorch"
    if backend == "fla_gdn2":
        if has_qed:
            return "dplr" if _get_fla_dplr_ops() is not None else "pytorch"
        if _get_fla_gdn2_ops() is not None:
            return "fla_gdn2"
        if _get_fla_dplr_ops() is not None:
            return "dplr"
        return "pytorch"
    # backend == "auto"
    if has_qed:
        return "dplr" if _get_fla_dplr_ops() is not None else "pytorch"
    if _get_fla_gdn2_ops() is not None:
        return "fla_gdn2"
    if _get_fla_dplr_ops() is not None:
        return "dplr"
    return "pytorch"


def _run_fla_dplr(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    qed_lambda: torch.Tensor | None,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ops = _get_fla_dplr_ops()
    assert ops is not None
    chunk_dplr, rec_dplr = ops
    init_dtype = query.dtype
    seq_len = int(query.shape[1])
    dk = int(query.shape[-1])
    sc = (dk ** -0.5) if scale is None else float(scale)

    q_hat, k_hat, v_w, a, gk = qed_dplr_prep(
        query,
        key,
        value,
        g_vec,
        b_gate,
        w_gate,
        qed_lambda,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        out_dtype=init_dtype,
    )
    state_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    init_st = initial_state.to(state_dt).contiguous() if initial_state is not None else None
    # `fused_recurrent_dplr_delta_rule` is inference-only in fla 0.6.0 (raises NotImplementedError on backward).
    if seq_len == 1 and not torch.is_grad_enabled():
        out, final_state = rec_dplr(
            q=q_hat,
            k=k_hat,
            v=v_w,
            a=a,
            b=k_hat,
            gk=gk,
            scale=sc,
            initial_state=init_st,
            output_final_state=output_final_state,
        )
    else:
        out, final_state = chunk_dplr(
            q=q_hat,
            k=k_hat,
            v=v_w,
            a=a,
            b=k_hat,
            gk=gk,
            scale=sc,
            initial_state=init_st,
            output_final_state=output_final_state,
        )
    return out.to(init_dtype), (final_state if output_final_state else None)


def _run_fla_gdn2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ops = _get_fla_gdn2_ops()
    assert ops is not None
    fla_chunk, fla_rec = ops
    init_dtype = query.dtype
    seq_len = int(query.shape[1])
    hv = int(value.shape[2])
    dk = int(query.shape[-1])
    sc = (dk ** -0.5) if scale is None else float(scale)

    q = query
    k = key
    if q.shape[2] != hv:
        require(hv % q.shape[2] == 0, "num_v_heads must be divisible by num_q_heads", q=q, v=value)
        q = q.repeat_interleave(hv // q.shape[2], dim=2)
    if k.shape[2] != hv:
        require(hv % k.shape[2] == 0, "num_v_heads must be divisible by num_k_heads", k=k, v=value)
        k = k.repeat_interleave(hv // k.shape[2], dim=2)

    state_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    q_c = q.to(init_dtype).contiguous()
    k_c = k.to(init_dtype).contiguous()
    v_c = value.to(init_dtype).contiguous()
    g_c = g_vec.clamp(max=0.0).to(state_dt).contiguous()
    b_c = b_gate.to(init_dtype).contiguous()
    w_c = w_gate.to(init_dtype).contiguous()
    init_st = initial_state.to(state_dt).contiguous() if initial_state is not None else None

    if seq_len == 1 and not torch.is_grad_enabled():
        out, final_state = fla_rec(
            q=q_c,
            k=k_c,
            v=v_c,
            g=g_c,
            b=b_c,
            w=w_c,
            scale=sc,
            initial_state=init_st,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    else:
        out, final_state = fla_chunk(
            q=q_c,
            k=k_c,
            v=v_c,
            g=g_c,
            b=b_c,
            w=w_c,
            scale=sc,
            initial_state=init_st,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    return out.to(init_dtype), (final_state if output_final_state else None)


def _run_fla_segmented(
    resolved: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    qed_lambda: torch.Tensor | None,
    seg_len: int,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Checkpointed macro-segmented `fla` DPLR / `fla.ops.gdn2` execution over `T`.

    Splits `T` into `seg_len`-token sub-segments inside `checkpoint(..., use_reentrant=False)`,
    keeping `cur_state` connected in autograd across sub-segments for mathematically exact
    forward+backward gradients while bounding single-layer FLA scratch VRAM to `O(seg_len)`.
    """
    seq_len = int(query.shape[1])
    cur_state: torch.Tensor | None = initial_state
    seg_outs: list[torch.Tensor] = []

    for s0 in range(0, seq_len, seg_len):
        s1 = min(seq_len, s0 + seg_len)
        need_state_next = (s1 < seq_len) or output_final_state

        has_st = cur_state is not None
        has_lam = (resolved == "dplr") and (qed_lambda is not None)
        pos_args: list[torch.Tensor] = [
            query[:, s0:s1],
            key[:, s0:s1],
            value[:, s0:s1],
            g_vec[:, s0:s1],
            b_gate[:, s0:s1],
            w_gate[:, s0:s1],
        ]
        if has_st:
            assert cur_state is not None
            pos_args.append(cur_state)
        if has_lam:
            assert qed_lambda is not None
            lam_s = (
                qed_lambda[:, s0:s1]
                if (qed_lambda.ndim >= 3 and qed_lambda.shape[1] == seq_len)
                else qed_lambda
            )
            pos_args.append(lam_s)

        def _ckpt_seg(
            *t_args: torch.Tensor,
            _has_st: bool = has_st,
            _has_lam: bool = has_lam,
            _need_st: bool = need_state_next,
        ) -> Any:
            idx = 6
            st_in = t_args[idx] if _has_st else None
            if _has_st:
                idx += 1
            lam_in = t_args[idx] if _has_lam else None
            if resolved == "dplr":
                o_s, nxt_s = _run_fla_dplr(
                    t_args[0],
                    t_args[1],
                    t_args[2],
                    t_args[3],
                    t_args[4],
                    t_args[5],
                    lam_in,
                    initial_state=st_in,
                    output_final_state=_need_st,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                )
            else:
                o_s, nxt_s = _run_fla_gdn2(
                    t_args[0],
                    t_args[1],
                    t_args[2],
                    t_args[3],
                    t_args[4],
                    t_args[5],
                    initial_state=st_in,
                    output_final_state=_need_st,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                )
            return (o_s, nxt_s) if _need_st else o_s

        res = checkpoint(_ckpt_seg, *pos_args, use_reentrant=False)
        if need_state_next:
            o_seg, cur_state = res
        else:
            o_seg = res
            cur_state = None
        seg_outs.append(o_seg)

    out = seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)
    return out, (cur_state if output_final_state else None)


# ============================================================================ pure-PyTorch & public kernels

def _recurrent_gdn2_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    qed_lambda: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    init_dtype = query.dtype
    q_hat, k_hat, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    sc = (k_hat.shape[-1] ** -0.5) if scale is None else float(scale)
    q_read = q_hat * sc if sc != 1.0 else q_hat
    gf = upcast(g_vec).clamp(max=0.0)
    bf, wf = upcast(b_gate), upcast(w_gate)
    ef = _compute_qed_erase(q_hat, k_hat, bf, qed_lambda)
    bsz, seq_len, hv, dk = k_hat.shape
    dv = v.shape[-1]
    state = q_hat.new_zeros(bsz, hv, dk, dv) if initial_state is None else upcast(initial_state).clone()
    outs = []
    for i in range(seq_len):
        d_t = gf[:, i].exp()                                  # [B, H_v, d_k]
        state = state * d_t.unsqueeze(-1)                     # Diag(exp(g_t)) S_{t-1}
        k_t, v_t = k_hat[:, i], v[:, i]
        e_t = ef[:, i]                                        # [B, H_v, d_k]
        z_t = wf[:, i] * v_t                                  # [B, H_v, d_v]
        mem = (state * e_t.unsqueeze(-1)).sum(dim=-2)         # [B, H_v, d_v]
        u_t = z_t - mem
        state = state + k_t.unsqueeze(-1) * u_t.unsqueeze(-2)
        outs.append((state * q_read[:, i].unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=1).to(init_dtype)
    return out, (state if output_final_state else None)


def recurrent_gdn2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    qed_lambda: torch.Tensor | None = None,
    backend: str = "auto",
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequential O(1)-state Gated Delta Rule-2 (+ optional QED) kernel (decode & exact reference).

    Shapes:
        query, key: [B, T, H_k_or_v, d_k]
        value:      [B, T, H_v, d_v]
        g_vec:      [B, T, H_v, d_k]  (channel-wise log-decay <= 0)
        b_gate:     [B, T, H_v, d_k]  (channel-wise erase gate in [0, 2])
        w_gate:     [B, T, H_v, d_v]  (channel-wise write gate in [0, 1])
        qed_lambda: optional [H_v]    (bounded per-head QED coupling)
    """
    resolved = _resolve_backend(backend, qed_lambda is not None, query)
    if resolved == "dplr":
        return _run_fla_dplr(
            query,
            key,
            value,
            g_vec,
            b_gate,
            w_gate,
            qed_lambda,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
    if resolved == "fla_gdn2":
        return _run_fla_gdn2(
            query,
            key,
            value,
            g_vec,
            b_gate,
            w_gate,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
    return _recurrent_gdn2_pytorch(
        query,
        key,
        value,
        g_vec,
        b_gate,
        w_gate,
        qed_lambda=qed_lambda,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
    )


def _decayed_gram_fallback(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    gamma: torch.Tensor,
    strict_lower: bool,
) -> torch.Tensor:
    """Exact fallback for chunks whose single-chunk cumulative decay span exceeds fp32 exp range."""
    # lhs, rhs, gamma: [B, H, N_c, C, d_k]
    c = lhs.shape[-2]
    idx = torch.arange(c, device=lhs.device)
    invalid = (idx[:, None] <= idx[None, :]) if strict_lower else (idx[:, None] < idx[None, :])
    out_chunks = []
    for n in range(lhs.shape[2]):
        diff = gamma[:, :, n, :, None, :] - gamma[:, :, n, None, :, :]  # [B, H, C, C, d_k]
        diff = diff.masked_fill(invalid[None, None, :, :, None], float("-inf"))
        gram = (lhs[:, :, n, :, None, :] * rhs[:, :, n, None, :, :] * diff.exp()).sum(dim=-1)
        out_chunks.append(gram)
    return torch.stack(out_chunks, dim=2)


def _wy_segment(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    gf: torch.Tensor,
    bf: torch.Tensor,
    wf: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int,
    qed_lambda: torch.Tensor | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise WY GDN-2 (+ QED) forward over one padded macro-segment (`T_seg = n_chunks * chunk_size`).

    `q_hat` and `k_hat` are UNSCALED unit-norm vectors; `scale` is applied only to the readout query.
    Works identically for any erase direction `e_t` (including QED) and any `b_gate` in `[0, 2]`.
    """
    bsz, t_seg, hv, dk = k_hat.shape
    dv = v.shape[-1]
    nc = t_seg // chunk_size
    c = chunk_size

    ef = _compute_qed_erase(q_hat, k_hat, bf, qed_lambda)
    q_read = q_hat * scale if scale != 1.0 else q_hat

    # Reshape to [B, H_v, N_c, C, D] for batched Tensor-Core matmuls
    qc = q_read.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    kc = k_hat.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    vc = v.transpose(1, 2).reshape(bsz, hv, nc, c, dv)
    gc = gf.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    ec = ef.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    wc = wf.transpose(1, 2).reshape(bsz, hv, nc, c, dv)

    gamma = gc.cumsum(dim=-2)                                 # [B, H_v, N_c, C, d_k], <= 0
    zc = wc * vc                                              # gated write values
    exp_gamma = gamma.exp()
    e_bar = ec * exp_gamma
    q_bar = qc * exp_gamma

    # Midpoint-centered Tensor-Core factorization: exp(Gamma_i - mu) * exp(mu - Gamma_j)
    # Exact in IEEE-754 fp32 whenever chunk span (Gamma_0 - Gamma_{C-1}) <= 150 (< 2 * 85).
    span = (gamma[..., 0, :] - gamma[..., -1, :]).detach()
    if bool((span <= 150.0).all()):
        mu = 0.5 * (gamma[..., :1, :] + gamma[..., -1:, :])
        f_pos = (gamma - mu).exp()
        f_neg = (mu - gamma).exp()
        k_neg_t = (kc * f_neg).transpose(-1, -2)              # [B, H_v, N_c, d_k, C]
        l_mat = torch.tril((ec * f_pos) @ k_neg_t, diagonal=-1)
        m_mat = torch.tril((qc * f_pos) @ k_neg_t, diagonal=0)
    else:
        l_mat = _decayed_gram_fallback(ec, kc, gamma, strict_lower=True)
        m_mat = _decayed_gram_fallback(qc, kc, gamma, strict_lower=False)

    eye = torch.eye(c, device=q_hat.device, dtype=q_hat.dtype)
    rhs = torch.cat([e_bar, zc], dim=-1)                      # [B, H_v, N_c, C, d_k + d_v]
    wu = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)
    w_wy, u_wy = wu.split([dk, dv], dim=-1)

    k_tilde = kc * (gamma[..., -1:, :] - gamma).exp()         # gamma_{C-1} - gamma_j <= 0
    d_chunk = gamma[..., -1, :].exp()                         # [B, H_v, N_c, d_k]

    outs = []
    cur = state
    for n in range(nc):
        u_n = u_wy[:, :, n] - w_wy[:, :, n] @ cur             # [B, H_v, C, d_v]
        o_n = q_bar[:, :, n] @ cur + m_mat[:, :, n] @ u_n     # [B, H_v, C, d_v]
        outs.append(o_n)
        cur = cur * d_chunk[:, :, n, :, None] + k_tilde[:, :, n].transpose(-1, -2) @ u_n

    seg_out = torch.stack(outs, dim=2).reshape(bsz, hv, t_seg, dv).transpose(1, 2)
    return seg_out, cur


def chunk_gdn2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    qed_lambda: torch.Tensor | None = None,
    backend: str = "auto",
    chunk_size: int = 64,
    segment_chunks: int = 16,
    checkpoint_segments: bool = True,
    fla_segment_len: int = 8192,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Parallel chunkwise Gated DeltaNet-2 (+ QED) kernel (`fla` DPLR / `fla.ops.gdn2` / PyTorch WY)."""
    if isinstance(fla_segment_len, bool) or not isinstance(fla_segment_len, int) or fla_segment_len < 0:
        raise ValueError(f"fla_segment_len must be an int >= 0, got {fla_segment_len}")
    resolved = _resolve_backend(backend, qed_lambda is not None, query)
    if resolved in ("dplr", "fla_gdn2"):
        seq_len_fla = int(query.shape[1])
        if fla_segment_len > 0 and seq_len_fla > fla_segment_len and checkpoint_segments and torch.is_grad_enabled():
            return _run_fla_segmented(
                resolved,
                query,
                key,
                value,
                g_vec,
                b_gate,
                w_gate,
                qed_lambda=qed_lambda,
                seg_len=fla_segment_len,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
            )
        if resolved == "dplr":
            return _run_fla_dplr(
                query,
                key,
                value,
                g_vec,
                b_gate,
                w_gate,
                qed_lambda,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
            )
        return _run_fla_gdn2(
            query,
            key,
            value,
            g_vec,
            b_gate,
            w_gate,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )

    init_dtype = query.dtype
    seq_len = int(query.shape[1])
    if seq_len == 1:
        return _recurrent_gdn2_pytorch(
            query,
            key,
            value,
            g_vec,
            b_gate,
            w_gate,
            qed_lambda=qed_lambda,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )

    q_hat, k_hat, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    sc = (k_hat.shape[-1] ** -0.5) if scale is None else float(scale)
    gf = upcast(g_vec).clamp(max=0.0)
    bf, wf = upcast(b_gate), upcast(w_gate)

    c = min(int(chunk_size), max(1, ((seq_len + 15) // 16) * 16)) if seq_len < chunk_size else int(chunk_size)
    pad = (c - (seq_len % c)) % c
    if pad:
        # Zero-padding (q=0, k=0, v=0, g=0, b=0, w=0) leaves the recurrent state S_t invariant
        # for t >= seq_len even with non-zero qed_lambda: e_t = 0, k_t = 0, g_t = 0 => S_t = S_{t-1}.
        q_hat = F.pad(q_hat, (0, 0, 0, 0, 0, pad))
        k_hat = F.pad(k_hat, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        gf = F.pad(gf, (0, 0, 0, 0, 0, pad))
        bf = F.pad(bf, (0, 0, 0, 0, 0, pad))
        wf = F.pad(wf, (0, 0, 0, 0, 0, pad))

    bsz, t_pad, hv, dk = k_hat.shape
    dv = v.shape[-1]
    state = q_hat.new_zeros(bsz, hv, dk, dv) if initial_state is None else upcast(initial_state).clone()

    seg_len = max(c, int(segment_chunks) * c)
    grad_tensors = (q_hat, k_hat, v, gf, bf, wf, state) + ((qed_lambda,) if qed_lambda is not None else ())
    need_grad = any(t.requires_grad for t in grad_tensors)
    use_ckpt = bool(checkpoint_segments and torch.is_grad_enabled() and need_grad and t_pad > seg_len)

    seg_outs = []
    for s0 in range(0, t_pad, seg_len):
        s1 = min(t_pad, s0 + seg_len)
        base_args = (q_hat[:, s0:s1], k_hat[:, s0:s1], v[:, s0:s1], gf[:, s0:s1], bf[:, s0:s1], wf[:, s0:s1], state)
        if use_ckpt:
            if qed_lambda is not None:
                o_seg, state = checkpoint(
                    lambda qc, kc, vc, gc, bc, wc, st, lam, cs=c, s=sc: _wy_segment(
                        qc, kc, vc, gc, bc, wc, st, cs, qed_lambda=lam, scale=s
                    ),
                    *base_args,
                    qed_lambda,
                    use_reentrant=False,
                )
            else:
                o_seg, state = checkpoint(
                    lambda qc, kc, vc, gc, bc, wc, st, cs=c, s=sc: _wy_segment(
                        qc, kc, vc, gc, bc, wc, st, cs, qed_lambda=None, scale=s
                    ),
                    *base_args,
                    use_reentrant=False,
                )
        else:
            o_seg, state = _wy_segment(*base_args, chunk_size=c, qed_lambda=qed_lambda, scale=sc)
        seg_outs.append(o_seg)

    out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1))[:, :seq_len].to(init_dtype)
    return out, (state if output_final_state else None)


# ============================================================================ adapter & retrofit

class GDN2Adapter(nn.Module):
    """Zero-initialized low-rank channel-wise Erase / Write / Decay (+ QED + soft neg-eigval) projector."""

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        cfg: GDN2Config,
        layer_id: int,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.layer_id = layer_id
        self.hidden_size = int(hidden_size)
        self.num_v_heads = int(num_v_heads)
        self.head_k_dim = int(head_k_dim)
        self.head_v_dim = int(head_v_dim)
        eff_rank = max(1, min(int(cfg.rank), self.hidden_size))
        self.eff_rank = eff_rank

        self.down = nn.Linear(self.hidden_size, eff_rank, bias=False)
        self.erase_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.write_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.decay_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)

        if cfg.qed_enabled:
            self.qed_lambda_raw: nn.Parameter | None = nn.Parameter(torch.zeros(self.num_v_heads))
        else:
            self.register_parameter("qed_lambda_raw", None)

        if float(cfg.neg_eigval_max_boost) > 0.0:
            self.neg_gamma_raw: nn.Parameter | None = nn.Parameter(torch.zeros(self.num_v_heads))
        else:
            self.register_parameter("neg_gamma_raw", None)

        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        nn.init.zeros_(self.erase_up.weight)
        nn.init.zeros_(self.write_up.weight)
        nn.init.zeros_(self.decay_up.weight)
        self._last_stats: dict[str, float] = {}
        self._pending_stat_tensors: dict[str, torch.Tensor] = {}
        self._is_zero_cache: tuple[tuple[Any, ...], bool] | None = None

    @property
    def last_stats(self) -> dict[str, float]:
        if self._pending_stat_tensors:
            for k, t in self._pending_stat_tensors.items():
                self._last_stats[k] = float(t)
            self._pending_stat_tensors.clear()
        return self._last_stats

    @last_stats.setter
    def last_stats(self, value: dict[str, float]) -> None:
        self._pending_stat_tensors.clear()
        self._last_stats = dict(value)

    def _set_zero_stats(self) -> None:
        self._pending_stat_tensors.clear()
        self._last_stats["erase_write_l1"] = 0.0
        self._last_stats["decay_spread"] = 0.0
        self._last_stats["decay_horizon_mult_max"] = 1.0

    def _zero_check_params(self) -> tuple[nn.Parameter, ...]:
        params: list[nn.Parameter] = [self.erase_up.weight, self.write_up.weight, self.decay_up.weight]
        if self.qed_lambda_raw is not None:
            params.append(self.qed_lambda_raw)
        if self.neg_gamma_raw is not None:
            params.append(self.neg_gamma_raw)
        return tuple(params)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._is_zero_cache = None
        super()._load_from_state_dict(*args, **kwargs)

    def is_zero(self) -> bool:
        """Cached zero-weight check keyed by `(data_ptr, _version, shape, device, dtype)`.

        Avoids per-token GPU `.any()` reductions and D2H host syncs during `T=1` decode / `eval`,
        while automatically invalidating on any in-place mutation (`.fill_()`, `.zero_()`, `.copy_()`,
        `opt.step()`, `restore_checkpoint()`) or `.data` replacement.
        """
        params = self._zero_check_params()
        vkey = tuple((int(p.data_ptr()), int(p._version), tuple(p.shape), p.device, p.dtype) for p in params)
        cached = self._is_zero_cache
        if cached is not None and cached[0] == vkey:
            return cached[1]
        with torch.no_grad():
            res = not any(bool(p.any()) for p in params)
        self._is_zero_cache = (vkey, res)
        return res

    def qed_lambda(self) -> torch.Tensor | None:
        if self.qed_lambda_raw is None:
            return None
        return float(self.cfg.qed_max_lambda) * torch.tanh(upcast(self.qed_lambda_raw))

    def neg_eigval_scale(self) -> torch.Tensor | None:
        if self.neg_gamma_raw is None:
            return None
        return 1.0 + float(self.cfg.neg_eigval_max_boost) * torch.tanh(upcast(self.neg_gamma_raw))

    def compute_gates(
        self,
        x_in: torch.Tensor,
        g_base: torch.Tensor,
        beta_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map `(x_in [B,T,H_model], g_base [B,T,H_v], beta_base [B,T,H_v])` -> `(g_vec, b_gate, w_gate)`."""
        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        xf = upcast(x_in)
        h = F.silu(F.linear(xf, upcast(self.down.weight)))
        cap_g = float(self.cfg.gate_logit_cap)

        d_erase = (cap_g * torch.tanh(F.linear(h, upcast(self.erase_up.weight)))).view(bsz, seq_len, hv, dk)
        d_write = (cap_g * torch.tanh(F.linear(h, upcast(self.write_up.weight)))).view(bsz, seq_len, hv, dv)
        d_raw = F.linear(h, upcast(self.decay_up.weight)).view(bsz, seq_len, hv, dk)
        if self.cfg.decay_log_min is not None or self.cfg.decay_log_max is not None:
            cap_min = (
                float(self.cfg.decay_log_min)
                if self.cfg.decay_log_min is not None
                else -float(self.cfg.decay_log_cap)
            )
            cap_max = (
                float(self.cfg.decay_log_max)
                if self.cfg.decay_log_max is not None
                else float(self.cfg.decay_log_cap)
            )
            d_decay = asymmetric_tanh_bound(d_raw, cap_min, cap_max)
        else:
            d_decay = float(self.cfg.decay_log_cap) * torch.tanh(d_raw)

        beta_f = upcast(beta_base).clamp(0.0, 1.0).unsqueeze(-1)
        b_gate = logistic_shift(beta_f, d_erase)
        b_scale = self.neg_eigval_scale()
        if b_scale is not None:
            b_gate = b_gate * b_scale.to(dtype=b_gate.dtype, device=b_gate.device).view(1, 1, hv, 1)
        w_gate = logistic_shift(beta_f, d_write)
        g_vec = upcast(g_base).clamp(max=0.0).unsqueeze(-1) * d_decay.exp()

        # Defer host syncs until `stats()` is read; skip extra stat kernels on no-grad T=1 decode.
        if not _in_backward() and (seq_len > 1 or torch.is_grad_enabled()):
            with torch.no_grad():
                if dk == dv:
                    ew_diff = (b_gate.detach() - w_gate.detach()).abs().mean()
                else:
                    ew_diff = (b_gate.detach().mean(-1) - w_gate.detach().mean(-1)).abs().mean()
                d_std = g_vec.detach().exp().std(dim=-1, unbiased=False).mean()
                h_mult_max = (-d_decay.detach().min()).exp()
                self._pending_stat_tensors = {
                    "erase_write_l1": ew_diff,
                    "decay_spread": d_std,
                    "decay_horizon_mult_max": h_mult_max,
                }
        return g_vec, b_gate, w_gate


def _infer_gdn_dims(attn: nn.Module, layer_id: int) -> tuple[int, int, int, int]:
    """Structurally infer `(hidden_size, num_v_heads, head_k_dim, head_v_dim)` for any GDN layer."""
    out_proj = getattr(attn, "out_proj", None)
    require(isinstance(out_proj, nn.Linear), f"layer {layer_id}: linear_attn has no nn.Linear `out_proj`",
            children=[n for n, _ in attn.named_children()])
    hidden_size = int(out_proj.out_features)
    a_log = getattr(attn, "A_log", None)
    num_v_heads = (
        getattr(attn, "num_v_heads", None)
        or getattr(attn, "nh", None)
        or (int(a_log.numel()) if isinstance(a_log, torch.Tensor) else None)
    )
    require(isinstance(num_v_heads, int) and num_v_heads > 0, f"layer {layer_id}: cannot infer `num_v_heads`")
    assert isinstance(num_v_heads, int)
    head_v_dim = int(
        getattr(attn, "head_v_dim", None) or getattr(attn, "d", None) or (out_proj.in_features // num_v_heads)
    )
    head_k_dim = getattr(attn, "head_k_dim", None) or getattr(attn, "d", None)
    if head_k_dim is None:
        nk = int(getattr(attn, "num_k_heads", None) or num_v_heads)
        qkv_proj = getattr(attn, "in_proj_qkv", None) or getattr(attn, "in_proj_qkvz", None)
        if isinstance(qkv_proj, nn.Linear):
            rem = int(qkv_proj.out_features) - num_v_heads * head_v_dim
            if getattr(attn, "in_proj_qkvz", None) is qkv_proj:
                rem -= num_v_heads * head_v_dim
            head_k_dim = rem // max(1, 2 * nk)
        else:
            head_k_dim = head_v_dim
    return hidden_size, num_v_heads, int(head_k_dim), int(head_v_dim)


def _extract_gdr_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool, bool, float | None, int]:
    g = args[0] if len(args) >= 1 else kwargs.get("g")
    beta = args[1] if len(args) >= 2 else kwargs.get("beta")
    require(isinstance(g, torch.Tensor) and isinstance(beta, torch.Tensor),
            "GDN-2 wrapper expected Tensor `g` and `beta`", g=g, beta=beta)
    chunk_size = int(kwargs.get("chunk_size", default_chunk))
    initial_state = kwargs.get("initial_state")
    for extra in args[2:]:
        if isinstance(extra, int):
            chunk_size = extra
        elif isinstance(extra, torch.Tensor) and extra.ndim == 4 and initial_state is None:
            initial_state = extra
    output_final = bool(kwargs.get("output_final_state", False))
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    scale = kwargs.get("scale")
    return g, beta, initial_state, output_final, use_l2, scale, chunk_size


class GDN2Manager:
    """Manages GDN-2 adapters, hooks, and kernel wrappers across all upgraded linear-attention layers."""

    def __init__(
        self,
        model: nn.Module,
        cfg: GDN2Config,
        adapters: dict[int, GDN2Adapter],
        hooks: list[Any],
        restorers: list[Callable[[], None]],
    ):
        self.model = model
        self.cfg = cfg
        self.adapters = adapters
        self._hooks = hooks
        self._restorers = restorers

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters()]

    def stats(self) -> dict[str, float]:
        if not self.adapters:
            return {}
        ew_vals: list[float] = []
        ds_vals: list[float] = []
        hm_vals: list[float] = []
        qed_abs_vals: list[float] = []
        neg_scale_vals: list[float] = []
        norms: list[float] = []
        with torch.no_grad():
            for ad in self.adapters.values():
                ls = ad.last_stats
                if "erase_write_l1" in ls:
                    ew_vals.append(ls["erase_write_l1"])
                if "decay_spread" in ls:
                    ds_vals.append(ls["decay_spread"])
                if "decay_horizon_mult_max" in ls:
                    hm_vals.append(ls["decay_horizon_mult_max"])

                norm_parts = [
                    ad.erase_up.weight.float().norm(),
                    ad.write_up.weight.float().norm(),
                    ad.decay_up.weight.float().norm(),
                ]
                if ad.qed_lambda_raw is not None:
                    norm_parts.append(ad.qed_lambda_raw.float().norm())
                    lam_eff = float(ad.cfg.qed_max_lambda) * torch.tanh(ad.qed_lambda_raw.float())
                    qed_abs_vals.append(float(lam_eff.abs().mean()))
                else:
                    qed_abs_vals.append(0.0)
                if ad.neg_gamma_raw is not None:
                    norm_parts.append(ad.neg_gamma_raw.float().norm())
                    b_sc = 1.0 + float(ad.cfg.neg_eigval_max_boost) * torch.tanh(ad.neg_gamma_raw.float())
                    neg_scale_vals.append(float(b_sc.mean()))
                else:
                    neg_scale_vals.append(1.0)
                norms.append(float(sum(norm_parts) / len(norm_parts)))

        n = float(len(self.adapters))
        out: dict[str, float] = {
            "gdn2/adapter_norm": sum(norms) / n,
            "gdn2/qed_lambda_abs_mean": sum(qed_abs_vals) / n,
            "gdn2/neg_eigval_scale_mean": sum(neg_scale_vals) / n,
            "gdn2/decay_horizon_mult_max": max(hm_vals) if hm_vals else 1.0,
        }
        if ew_vals:
            out["gdn2/erase_write_l1"] = sum(ew_vals) / len(ew_vals)
        if ds_vals:
            out["gdn2/decay_spread"] = sum(ds_vals) / len(ds_vals)
        return out

    def uninstall(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for fn in reversed(self._restorers):
            fn()
        self._restorers.clear()
        log.info("GDN-2 uninstalled from %d linear-attention layers", len(self.adapters))


_CHUNK_RULE_ATTRS: tuple[str, ...] = ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule")
_REC_RULE_ATTRS: tuple[str, ...] = (
    "recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


def _run_adapter_fla_segmented(
    adapter: GDN2Adapter,
    resolved: str,
    x_in: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_base: torch.Tensor,
    beta_base: torch.Tensor,
    *,
    seg_len: int,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Checkpointed macro-segmented `compute_gates` + `fla` DPLR / `fla.ops.gdn2` over `T`.

    Running both `adapter.compute_gates` and `_run_fla_dplr` / `_run_fla_gdn2` inside each
    `checkpoint(..., use_reentrant=False)` sub-segment keeps fp32 gate, prep, and FLA scratch
    allocations bounded to `O(seg_len)` while keeping `cur_state` connected in autograd so
    gradients flow bit-identically across all `seq_len` tokens.
    """
    seq_len = int(query.shape[1])
    cur_state: torch.Tensor | None = initial_state
    seg_outs: list[torch.Tensor] = []

    record_stats = not _in_backward()
    ew_acc: torch.Tensor | None = None
    ds_acc: torch.Tensor | None = None
    hm_acc: torch.Tensor | None = None

    for s0 in range(0, seq_len, seg_len):
        s1 = min(seq_len, s0 + seg_len)
        seg_n = s1 - s0
        need_state_next = (s1 < seq_len) or output_final_state

        has_st = cur_state is not None
        pos_args: list[torch.Tensor] = [
            x_in[:, s0:s1],
            query[:, s0:s1],
            key[:, s0:s1],
            value[:, s0:s1],
            g_base[:, s0:s1],
            beta_base[:, s0:s1],
        ]
        if has_st:
            assert cur_state is not None
            pos_args.append(cur_state)

        def _ckpt_seg(
            *t_args: torch.Tensor,
            _has_st: bool = has_st,
            _need_st: bool = need_state_next,
        ) -> Any:
            st_in = t_args[6] if _has_st else None
            gv_s, bg_s, wg_s = adapter.compute_gates(t_args[0], t_args[4], t_args[5])
            if resolved == "dplr":
                o_s, nxt_s = _run_fla_dplr(
                    t_args[1],
                    t_args[2],
                    t_args[3],
                    gv_s,
                    bg_s,
                    wg_s,
                    adapter.qed_lambda(),
                    initial_state=st_in,
                    output_final_state=_need_st,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                )
            else:
                o_s, nxt_s = _run_fla_gdn2(
                    t_args[1],
                    t_args[2],
                    t_args[3],
                    gv_s,
                    bg_s,
                    wg_s,
                    initial_state=st_in,
                    output_final_state=_need_st,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                )
            return (o_s, nxt_s) if _need_st else o_s

        if record_stats:
            adapter._pending_stat_tensors = {}
        res = checkpoint(_ckpt_seg, *pos_args, use_reentrant=False)
        if need_state_next:
            o_seg, cur_state = res
        else:
            o_seg = res
            cur_state = None
        seg_outs.append(o_seg)

        if record_stats and adapter._pending_stat_tensors:
            with torch.no_grad():
                wt = float(seg_n) / float(seq_len)
                ew_i = adapter._pending_stat_tensors["erase_write_l1"] * wt
                ds_i = adapter._pending_stat_tensors["decay_spread"] * wt
                hm_i = adapter._pending_stat_tensors["decay_horizon_mult_max"]
                ew_acc = ew_i if ew_acc is None else (ew_acc + ew_i)
                ds_acc = ds_i if ds_acc is None else (ds_acc + ds_i)
                hm_acc = hm_i if hm_acc is None else torch.maximum(hm_acc, hm_i)

    if record_stats and ew_acc is not None and ds_acc is not None and hm_acc is not None:
        adapter._pending_stat_tensors = {
            "erase_write_l1": ew_acc,
            "decay_spread": ds_acc,
            "decay_horizon_mult_max": hm_acc,
        }

    out = seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)
    return out, (cur_state if output_final_state else None)


def _make_rule_wrapper(
    attn: nn.Module,
    adapter: GDN2Adapter,
    x_stack: list[torch.Tensor],
    orig_fn: Callable[..., Any],
    *,
    recurrent_only: bool,
    in_rule_box: list[int] | None = None,
) -> Callable[..., Any]:
    def wrapped(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any):
        if in_rule_box is not None and in_rule_box[0] > 0:
            return orig_fn(query, key, value, *args, **kwargs)
        if in_rule_box is not None:
            in_rule_box[0] += 1
        try:
            # Fast bit-exact passthrough in eval mode (or when frozen/no_grad) while zero-initialized.
            # Skip dynamic zero-check during `torch.compile` tracing to avoid data-dependent graph breaks.
            need_ad_grad = torch.is_grad_enabled() and any(p.requires_grad for p in adapter.parameters())
            if (not _is_compiling()) and (not attn.training or not need_ad_grad) and adapter.is_zero():
                adapter._set_zero_stats()
                return orig_fn(query, key, value, *args, **kwargs)

            require(bool(x_stack), f"GDN-2 layer {adapter.layer_id}: missing captured input hidden_states")
            x_in = x_stack[-1]
            bsz, seq_len, _ = x_in.shape
            g, beta, init_st, out_final, use_l2, scale, csize = _extract_gdr_args(args, kwargs, adapter.cfg.chunk_size)

            # Normalize head_first ([B, H, T, D]) vs token-first ([B, T, H, D]) layouts seamlessly
            head_first = bool(kwargs.get("head_first", False)) or (
                seq_len != adapter.num_v_heads and value.shape[1] == adapter.num_v_heads and value.shape[2] == seq_len
            )
            if head_first:
                query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                g, beta = g.transpose(1, 2), beta.transpose(1, 2)
            require(
                value.shape[:2] == (bsz, seq_len),
                f"GDN-2 layer {adapter.layer_id}: value shape {tuple(value.shape)} disagrees with input "
                f"{(bsz, seq_len)}",
            )

            backend = adapter.cfg.backend
            has_qed = adapter.qed_lambda_raw is not None
            resolved = _resolve_backend(backend, has_qed, query)
            fla_seg_len = int(adapter.cfg.fla_segment_len)
            if (
                resolved in ("dplr", "fla_gdn2")
                and fla_seg_len > 0
                and seq_len > fla_seg_len
                and torch.is_grad_enabled()
                and adapter.cfg.checkpoint_segments
            ):
                out, final_state = _run_adapter_fla_segmented(
                    adapter,
                    resolved,
                    x_in,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    seg_len=fla_seg_len,
                    initial_state=init_st,
                    output_final_state=out_final,
                    use_qk_l2norm_in_kernel=use_l2,
                    scale=scale,
                )
            else:
                g_vec, b_gate, w_gate = adapter.compute_gates(x_in, g, beta)
                qed_lam = adapter.qed_lambda()
                if (recurrent_only or seq_len == 1) and not torch.is_grad_enabled():
                    out, final_state = recurrent_gdn2(
                        query,
                        key,
                        value,
                        g_vec,
                        b_gate,
                        w_gate,
                        qed_lambda=qed_lam,
                        backend=backend,
                        initial_state=init_st,
                        output_final_state=out_final,
                        use_qk_l2norm_in_kernel=use_l2,
                        scale=scale,
                    )
                else:
                    out, final_state = chunk_gdn2(
                        query,
                        key,
                        value,
                        g_vec,
                        b_gate,
                        w_gate,
                        qed_lambda=qed_lam,
                        backend=backend,
                        chunk_size=csize,
                        segment_chunks=adapter.cfg.segment_chunks,
                        checkpoint_segments=adapter.cfg.checkpoint_segments,
                        fla_segment_len=fla_seg_len,
                        initial_state=init_st,
                        output_final_state=out_final,
                        use_qk_l2norm_in_kernel=use_l2,
                        scale=scale,
                    )
            if head_first:
                out = out.transpose(1, 2)
            return out, final_state
        finally:
            if in_rule_box is not None:
                in_rule_box[0] -= 1

    return wrapped


def install_gdn2(model: nn.Module, cfg: GDN2Config | dict[str, Any] | None = None) -> GDN2Manager:
    """Upgrade Qwen3.6 / Toy Gated DeltaNet layers in-place to NVIDIA Gated DeltaNet-2 (GDN-2-1M + QED).

    Install BEFORE `stable_names(model)` and `apply_loop(model, ...)` so `gdn2_adapter` parameters
    receive loop-invariant stable checkpoint names and share physical modules across core passes.
    """
    if not isinstance(cfg, GDN2Config):
        cfg = GDN2Config.from_dict(cfg)
    cfg.validate()

    parts = inspect_model(model)
    lin_indices = [i for i, lt in enumerate(parts.layer_types) if lt == LINEAR]
    target_layers = lin_indices if cfg.layers is None else [int(i) for i in cfg.layers]
    bad = [i for i in target_layers if i not in lin_indices]
    if bad:
        raise ValueError(f"gdn2.layers {bad} are not linear_attention layers (linear: {lin_indices})")

    ref = parts.text_model.embed_tokens.weight
    param_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
    adapters: dict[int, GDN2Adapter] = {}
    hooks: list[Any] = []
    restorers: list[Callable[[], None]] = []
    patched_mod_fns: set[tuple[str, str]] = set()
    active_attn_stack: list[tuple[nn.Module, GDN2Adapter, list[torch.Tensor], list[int], bool, bool]] = []

    rule_specs: list[tuple[str, bool]] = (
        [(attr, False) for attr in _CHUNK_RULE_ATTRS]
        + [(attr, True) for attr in _REC_RULE_ATTRS]
    )

    for i in target_layers:
        attn = attention_module(parts.layers[i], LINEAR)
        hs, hv, dk, dv = _infer_gdn_dims(attn, i)
        adapter = GDN2Adapter(hs, hv, dk, dv, cfg, i).to(device=ref.device, dtype=param_dtype)
        attn.gdn2_adapter = adapter
        adapters[i] = adapter

        x_stack: list[torch.Tensor] = []
        in_rule_box: list[int] = [0]

        def _del_adapter(a: nn.Module = attn, ad: GDN2Adapter = adapter) -> None:
            if getattr(a, "gdn2_adapter", None) is ad:
                delattr(a, "gdn2_adapter")

        restorers.append(_del_adapter)

        if "_kernel_repo" in attn.__dict__ and "forward" in attn.__dict__:
            saved_kfwd = attn.__dict__.pop("forward")
            saved_krepo = attn.__dict__.pop("_kernel_repo")

            def _restore_kernel_fwd(
                a: nn.Module = attn,
                kfwd: Any = saved_kfwd,
                krepo: Any = saved_krepo,
            ) -> None:
                a.forward = kfwd  # type: ignore[method-assign]
                a.__dict__["_kernel_repo"] = krepo

            restorers.append(_restore_kernel_fwd)

        wrapped_chunk = False
        wrapped_rec = False
        for attr, rec_only in rule_specs:
            orig_fn = getattr(attn, attr, None)
            if callable(orig_fn):
                if rec_only:
                    wrapped_rec = True
                else:
                    wrapped_chunk = True
                had_instance_attr = attr in attn.__dict__
                orig_inst_attr = attn.__dict__.get(attr)
                rule_w = _make_rule_wrapper(
                    attn, adapter, x_stack, orig_fn, recurrent_only=rec_only, in_rule_box=in_rule_box
                )
                if isinstance(orig_fn, types.MethodType) and getattr(orig_fn, "__self__", None) is attn:
                    raw_rfn = getattr(orig_fn, "__func__", orig_fn)

                    @functools.wraps(raw_rfn)
                    def _bound_rule(
                        _self: nn.Module,
                        *r_args: Any,
                        _rw: Callable[..., Any] = rule_w,
                        **r_kwargs: Any,
                    ) -> Any:
                        return _rw(*r_args, **r_kwargs)

                    rule_patch: Any = types.MethodType(_bound_rule, attn)
                else:
                    rule_patch = rule_w
                setattr(attn, attr, rule_patch)

                def _restore_attr(
                    a: nn.Module = attn,
                    name: str = attr,
                    fn: Any = orig_inst_attr,
                    had: bool = had_instance_attr,
                ) -> None:
                    if had:
                        setattr(a, name, fn)
                    else:
                        a.__dict__.pop(name, None)

                restorers.append(_restore_attr)

        def _make_pre(
            mod_ref: nn.Module,
            ad_ref: GDN2Adapter,
            st_ref: list[torch.Tensor],
            ir_ref: list[int],
            wc_flag: bool,
            wr_flag: bool,
        ):
            def pre_hook(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x_in = (
                    args[0] if args else (kwargs.get("hidden_states") if "hidden_states" in kwargs else kwargs.get("x"))
                )
                require(isinstance(x_in, torch.Tensor), "GDN-2 pre_hook could not locate input hidden_states")
                amask = kwargs.get("attention_mask", args[2] if len(args) >= 3 else None)
                if (
                    isinstance(amask, torch.Tensor)
                    and amask.ndim == 2
                    and amask.shape[0] == x_in.shape[0]
                    and amask.shape[1] >= x_in.shape[1]
                    and x_in.shape[1] > 1
                ):
                    x_in = x_in * amask[:, -x_in.shape[1]:, None].to(x_in.dtype)
                st_ref.append(x_in)
                active_attn_stack.append((mod_ref, ad_ref, st_ref, ir_ref, wc_flag, wr_flag))

            return pre_hook

        def _make_post(st_ref: list[torch.Tensor]):
            def post_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
                if st_ref:
                    st_ref.pop()
                if active_attn_stack:
                    active_attn_stack.pop()

            return post_hook

        hooks.append(
            attn.register_forward_pre_hook(
                _make_pre(attn, adapter, x_stack, in_rule_box, wrapped_chunk, wrapped_rec),
                with_kwargs=True,
            )
        )
        hooks.append(attn.register_forward_hook(_make_post(x_stack)))

        # Fallback if a transformers build calls module-level chunk or recurrent rule functions directly
        if not (wrapped_chunk and wrapped_rec):
            mod_name = type(attn).__module__
            host_mod = sys.modules.get(mod_name)
            if host_mod is not None:
                missing_specs: list[tuple[str, bool]] = []
                if not wrapped_chunk:
                    missing_specs.extend((nm, False) for nm in _CHUNK_RULE_ATTRS)
                if not wrapped_rec:
                    missing_specs.extend((nm, True) for nm in _REC_RULE_ATTRS)
                for fn_name, rec_only in missing_specs:
                    if (mod_name, fn_name) in patched_mod_fns:
                        continue
                    orig_mfn = getattr(host_mod, fn_name, None)
                    if callable(orig_mfn):
                        patched_mod_fns.add((mod_name, fn_name))

                        def _mod_dispatch(
                            query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            *args: Any,
                            _orig: Callable[..., Any] = orig_mfn,
                            _rec: bool = rec_only,
                            **kwargs: Any,
                        ):
                            if not active_attn_stack:
                                return _orig(query, key, value, *args, **kwargs)
                            cur_mod, cur_ad, cur_st, cur_ir, cur_wc, cur_wr = active_attn_stack[-1]
                            if cur_ir[0] > 0 or (_rec and cur_wr) or ((not _rec) and cur_wc):
                                return _orig(query, key, value, *args, **kwargs)
                            return _make_rule_wrapper(
                                cur_mod, cur_ad, cur_st, _orig, recurrent_only=_rec, in_rule_box=cur_ir
                            )(query, key, value, *args, **kwargs)

                        setattr(host_mod, fn_name, _mod_dispatch)
                        restorers.append(lambda hm=host_mod, n=fn_name, f=orig_mfn: setattr(hm, n, f))

    n_params = sum(p.numel() for p in (p for ad in adapters.values() for p in ad.parameters()))
    log.info(
        "GDN-2 installed on %d linear-attention layers (rank=%d, chunk=%d, decay=[%s,%s], qed=%s, "
        "neg_boost=%.2f, backend=%s, fla_seg=%d, params=%.2fM, step-0 bit-identical)",
        len(adapters),
        cfg.rank,
        cfg.chunk_size,
        cfg.decay_log_min if cfg.decay_log_min is not None else -cfg.decay_log_cap,
        cfg.decay_log_max if cfg.decay_log_max is not None else cfg.decay_log_cap,
        cfg.qed_enabled,
        cfg.neg_eigval_max_boost,
        cfg.backend,
        cfg.fla_segment_len,
        n_params / 1e6,
    )
    return GDN2Manager(model, cfg, adapters, hooks, restorers)


__all__ = [
    "GDN2Adapter",
    "GDN2Config",
    "GDN2Manager",
    "asymmetric_tanh_bound",
    "chunk_gdn2",
    "install_gdn2",
    "l2norm",
    "logistic_shift",
    "qed_dplr_prep",
    "qed_dplr_prep_pytorch",
    "recurrent_gdn2",
]
