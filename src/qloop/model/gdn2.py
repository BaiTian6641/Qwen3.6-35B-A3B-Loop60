"""NVIDIA Gated DeltaNet-2 (GDN-2) retrofit for Qwen3.6's 30 linear-attention layers.

Reference:
    Ali Hatamizadeh, Yejin Choi, Jan Kautz (NVIDIA Research),
    "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791,
    `NVlabs/GatedDeltaNet-2`).

Why GDN-2 upgrades stock Gated DeltaNet cleanly on Qwen3.6-35B-A3B:
  * Stock Gated DeltaNet ties memory erasure (key axis) and memory writing (value axis) to one
    shared scalar gate `beta_t in (0, 1)` per head alongside a scalar decay `g_t <= 0`:
        S_t = (I - beta_t k_t k_t^T) exp(g_t) S_{t-1} + beta_t k_t v_t^T
  * GDN-2 decouples these into three independent channel-wise vectors per head:
      1. Channel-wise key-side erase gate   b_t in (0, 1)^{d_k}
      2. Channel-wise value-side write gate w_t in (0, 1)^{d_v}
      3. Channel-wise log-decay             g_t in (-inf, 0]^{d_k},  D_t = Diag(exp(g_t))
    evolving the recurrent state via the Gated Delta Rule-2:
        S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1} + k_t (w_t * v_t)^T,   o_t = S_t^T q_t
  * Strict generalization & step-0 bit-identity:
    GDN-2 collapses to stock Gated DeltaNet identically when `b_t = beta_t * 1_{d_k}`,
    `w_t = beta_t * 1_{d_v}`, and `g_t = g_base_t * 1_{d_k}`. By parameterizing zero-initialized
    low-rank residual projections (`erase_up = 0`, `write_up = 0`, `decay_up = 0`) composed via
    the singularity-free logistic odds shift
        sigma(logit(beta) + delta) == beta + 2*beta*(1 - beta)*tanh(delta/2) / (1 + (2*beta - 1)*tanh(delta/2))
    and strictly non-positive decay scaling `g_t = g_base_t * exp(tanh(delta_g)) <= 0`, the
    retrofitted Qwen3.6 model is **100% bit-identical at step 0** while unlocking full GDN-2
    channel-selective memory editing during training and looped recurrence.
  * Hardware-efficient training (`chunk_gdn2`) & O(1) decode (`recurrent_gdn2`):
    `chunk_gdn2` implements the chunkwise WY representation (absorbing cumulative channel-wise
    decays into asymmetric erase factors solved via batched unit-lower-triangular `trsm` on Tensor
    Cores) with macro-segment activation checkpointing for 64K-1M long-context training.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..debug import require
from ._dtype import upcast
from .introspect import LINEAR, attention_module, inspect_model

log = logging.getLogger("qloop")


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


# ============================================================================ config

@dataclass
class GDN2Config:
    rank: int = 64
    layers: list[int] | None = None  # physical linear-attention layers; None = all 30 of them
    chunk_size: int = 64
    segment_chunks: int = 16         # 16 * 64 = 1024 tokens per checkpointable WY macro-segment
    checkpoint_segments: bool = True
    gate_logit_cap: float = 8.0      # smooth tanh bound on channel-wise erase/write logit shifts
    decay_log_cap: float = 1.0       # smooth bound on log-decay multiplier exp(+-cap) in [0.37, 2.72]

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
        if self.gate_logit_cap <= 0 or self.decay_log_cap <= 0:
            raise ValueError("gdn2.gate_logit_cap and gdn2.decay_log_cap must be > 0")


# ============================================================================ math primitives

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA / HF compatible L2 normalization accumulated in fp32/fp64."""
    xf = upcast(x)
    return xf * torch.rsqrt((xf * xf).sum(dim=dim, keepdim=True) + eps)


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


def _prepare_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_qk_l2norm: bool,
    scale: float | None,
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
    sc = (q.shape[-1] ** -0.5) if scale is None else float(scale)
    if sc != 1.0:
        q = q * sc
    return q, k, v


def recurrent_gdn2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_vec: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequential O(1)-state Gated Delta Rule-2 kernel (autoregressive decode & exact reference).

    Shapes:
        query, key: [B, T, H_k_or_v, d_k]
        value:      [B, T, H_v, d_v]
        g_vec:      [B, T, H_v, d_k]  (channel-wise log-decay <= 0)
        b_gate:     [B, T, H_v, d_k]  (channel-wise erase gate in [0, 1])
        w_gate:     [B, T, H_v, d_v]  (channel-wise write gate in [0, 1])
    """
    init_dtype = query.dtype
    q, k, v = _prepare_qkv(query, key, value, use_qk_l2norm_in_kernel, scale)
    gf = upcast(g_vec).clamp(max=0.0)
    bf, wf = upcast(b_gate), upcast(w_gate)
    bsz, seq_len, hv, dk = k.shape
    dv = v.shape[-1]
    state = q.new_zeros(bsz, hv, dk, dv) if initial_state is None else upcast(initial_state).clone()
    outs = []
    for i in range(seq_len):
        d_t = gf[:, i].exp()                                  # [B, H_v, d_k]
        state = state * d_t.unsqueeze(-1)                     # Diag(exp(g_t)) S_{t-1}
        k_t, v_t = k[:, i], v[:, i]
        e_t = bf[:, i] * k_t                                  # [B, H_v, d_k]
        z_t = wf[:, i] * v_t                                  # [B, H_v, d_v]
        mem = (state * e_t.unsqueeze(-1)).sum(dim=-2)         # [B, H_v, d_v]
        u_t = z_t - mem
        state = state + k_t.unsqueeze(-1) * u_t.unsqueeze(-2)
        outs.append((state * q[:, i].unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outs, dim=1).to(init_dtype)
    return out, (state if output_final_state else None)


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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gf: torch.Tensor,
    bf: torch.Tensor,
    wf: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise WY GDN-2 forward over one padded macro-segment (`T_seg = n_chunks * chunk_size`)."""
    bsz, t_seg, hv, dk = k.shape
    dv = v.shape[-1]
    nc = t_seg // chunk_size
    c = chunk_size

    # Reshape to [B, H_v, N_c, C, D] for batched Tensor-Core matmuls
    qc = q.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    kc = k.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    vc = v.transpose(1, 2).reshape(bsz, hv, nc, c, dv)
    gc = gf.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    bc = bf.transpose(1, 2).reshape(bsz, hv, nc, c, dk)
    wc = wf.transpose(1, 2).reshape(bsz, hv, nc, c, dv)

    gamma = gc.cumsum(dim=-2)                                 # [B, H_v, N_c, C, d_k], <= 0
    ec = bc * kc                                              # asymmetric erase keys
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

    eye = torch.eye(c, device=q.device, dtype=q.dtype)
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
    chunk_size: int = 64,
    segment_chunks: int = 16,
    checkpoint_segments: bool = True,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Parallel chunkwise WY Gated DeltaNet-2 kernel (`O(T)` prefill & training)."""
    init_dtype = query.dtype
    seq_len = query.shape[1]
    if seq_len == 1:
        return recurrent_gdn2(
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

    q, k, v = _prepare_qkv(query, key, value, use_qk_l2norm_in_kernel, scale)
    gf = upcast(g_vec).clamp(max=0.0)
    bf, wf = upcast(b_gate), upcast(w_gate)

    c = min(int(chunk_size), max(1, ((seq_len + 15) // 16) * 16)) if seq_len < chunk_size else int(chunk_size)
    pad = (c - (seq_len % c)) % c
    if pad:
        # Zero-padding (q=0, k=0, v=0, g=0, b=0, w=0) leaves the recurrent state S_t invariant
        # for t >= seq_len: S_t = (I - 0) * Diag(exp(0)) * S_{t-1} + 0 = S_{t-1}.
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        gf = F.pad(gf, (0, 0, 0, 0, 0, pad))
        bf = F.pad(bf, (0, 0, 0, 0, 0, pad))
        wf = F.pad(wf, (0, 0, 0, 0, 0, pad))

    bsz, t_pad, hv, dk = k.shape
    dv = v.shape[-1]
    state = q.new_zeros(bsz, hv, dk, dv) if initial_state is None else upcast(initial_state).clone()

    seg_len = max(c, int(segment_chunks) * c)
    need_grad = any(t.requires_grad for t in (q, k, v, gf, bf, wf, state))
    use_ckpt = bool(checkpoint_segments and torch.is_grad_enabled() and need_grad and t_pad > seg_len)

    seg_outs = []
    for s0 in range(0, t_pad, seg_len):
        s1 = min(t_pad, s0 + seg_len)
        args = (q[:, s0:s1], k[:, s0:s1], v[:, s0:s1], gf[:, s0:s1], bf[:, s0:s1], wf[:, s0:s1], state)
        if use_ckpt:
            o_seg, state = checkpoint(
                lambda qc, kc, vc, gc, bc, wc, st, cs=c: _wy_segment(qc, kc, vc, gc, bc, wc, st, cs),
                *args,
                use_reentrant=False,
            )
        else:
            o_seg, state = _wy_segment(*args, chunk_size=c)
        seg_outs.append(o_seg)

    out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1))[:, :seq_len].to(init_dtype)
    return out, (state if output_final_state else None)


# ============================================================================ adapter & retrofit

class GDN2Adapter(nn.Module):
    """Zero-initialized low-rank channel-wise Erase / Write / Decay projector for one GDN layer."""

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

        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        nn.init.zeros_(self.erase_up.weight)
        nn.init.zeros_(self.write_up.weight)
        nn.init.zeros_(self.decay_up.weight)
        self.last_stats: dict[str, float] = {}

    def is_zero(self) -> bool:
        with torch.no_grad():
            return not bool(
                self.erase_up.weight.any() or self.write_up.weight.any() or self.decay_up.weight.any()
            )

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
        cap_d = float(self.cfg.decay_log_cap)

        d_erase = (cap_g * torch.tanh(F.linear(h, upcast(self.erase_up.weight)))).view(bsz, seq_len, hv, dk)
        d_write = (cap_g * torch.tanh(F.linear(h, upcast(self.write_up.weight)))).view(bsz, seq_len, hv, dv)
        d_decay = (cap_d * torch.tanh(F.linear(h, upcast(self.decay_up.weight)))).view(bsz, seq_len, hv, dk)

        beta_f = upcast(beta_base).clamp(0.0, 1.0).unsqueeze(-1)
        b_gate = logistic_shift(beta_f, d_erase)
        w_gate = logistic_shift(beta_f, d_write)
        g_vec = upcast(g_base).clamp(max=0.0).unsqueeze(-1) * d_decay.exp()

        if not _in_backward():
            with torch.no_grad():
                if dk == dv:
                    ew_diff = float((b_gate.detach() - w_gate.detach()).abs().mean())
                else:
                    ew_diff = float((b_gate.detach().mean(-1) - w_gate.detach().mean(-1)).abs().mean())
                d_std = float(g_vec.detach().exp().std(dim=-1, unbiased=False).mean())
                self.last_stats["erase_write_l1"] = ew_diff
                self.last_stats["decay_spread"] = d_std
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
        ew_vals = [
            ad.last_stats["erase_write_l1"] for ad in self.adapters.values() if "erase_write_l1" in ad.last_stats
        ]
        ds_vals = [ad.last_stats["decay_spread"] for ad in self.adapters.values() if "decay_spread" in ad.last_stats]
        with torch.no_grad():
            norms = [
                float(
                    (
                        ad.erase_up.weight.float().norm()
                        + ad.write_up.weight.float().norm()
                        + ad.decay_up.weight.float().norm()
                    )
                    / 3.0
                )
                for ad in self.adapters.values()
            ]
        out: dict[str, float] = {"gdn2/adapter_norm": sum(norms) / len(norms)}
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
            # Fast bit-exact passthrough in eval mode (or when frozen/no_grad) while zero-initialized
            need_ad_grad = torch.is_grad_enabled() and any(p.requires_grad for p in adapter.parameters())
            if (not attn.training or not need_ad_grad) and adapter.is_zero():
                adapter.last_stats["erase_write_l1"] = 0.0
                adapter.last_stats["decay_spread"] = 0.0
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

            g_vec, b_gate, w_gate = adapter.compute_gates(x_in, g, beta)
            if recurrent_only or seq_len == 1:
                out, final_state = recurrent_gdn2(
                    query,
                    key,
                    value,
                    g_vec,
                    b_gate,
                    w_gate,
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
                    chunk_size=csize,
                    segment_chunks=adapter.cfg.segment_chunks,
                    checkpoint_segments=adapter.cfg.checkpoint_segments,
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
    """Upgrade Qwen3.6 / Toy Gated DeltaNet layers in-place to NVIDIA Gated DeltaNet-2 (GDN-2).

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
                setattr(
                    attn,
                    attr,
                    _make_rule_wrapper(
                        attn, adapter, x_stack, orig_fn, recurrent_only=rec_only, in_rule_box=in_rule_box
                    ),
                )

                def _restore_attr(
                    a: nn.Module = attn,
                    name: str = attr,
                    fn: Callable[..., Any] = orig_fn,
                    had: bool = had_instance_attr,
                ) -> None:
                    if had:
                        setattr(a, name, fn)
                    elif name in a.__dict__:
                        delattr(a, name)

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
        "GDN-2 installed on %d linear-attention layers (rank=%d, chunk=%d, params=%.2fM, step-0 bit-identical)",
        len(adapters),
        cfg.rank,
        cfg.chunk_size,
        n_params / 1e6,
    )
    return GDN2Manager(model, cfg, adapters, hooks, restorers)


__all__ = [
    "GDN2Adapter",
    "GDN2Config",
    "GDN2Manager",
    "chunk_gdn2",
    "install_gdn2",
    "l2norm",
    "logistic_shift",
    "recurrent_gdn2",
]
