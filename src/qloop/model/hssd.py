"""Horizon-Split Surprise-Gated Delta Memory (HSSD v3) retrofit for linear-attention layers.

Implements Track B (`HSSD v3`) of `Qwen3.6-35B-A6B-LoopHSSD` (`docs/ARCHITECTURE_SPEC.md` §5,
`docs/RESEARCH_ESSAY.md` §4 Theorems 1 & 2):

  * **Fix B1 (Strictly Disjoint Horizon Partition):** For 1-based token `t` in 0-based chunk
    `n = (t - 1) // C`, sets `k_far = max(0, n - W)` (`W = window_chunks`, default 8) and
    `far_cutoff = k_far * C`. Far-horizon memory reads exclusively from `S_boundary[k_far]`
    covering `I_far = [1, far_cutoff]`, while exact causal local SWA covers
    `I_loc = [far_cutoff + 1, t]` (`|I_loc| in [W*C + 1, (W + 1)*C]` for `t > W*C`, `[1, t]`
    for `t <= W*C`), guaranteeing zero double-counting and zero gap.
  * **Fix B5 (`PeakBoltzmannGate`):** Size-free hybrid log-partition + Gumbel extreme-value miss
    gate `stat = 0.5 * (LSE(z) - log(N_t)) + 0.5 * (max(z) - sqrt(2 * log(max(N_t, 2))))`,
    `p_hit = sigmoid(kappa_h * (stat - tau_h))`, blending `o = p_hit * o_loc + (1 - p_hit) * o_far`.
  * **Theorem 1 (2D Phase Rotation -> Real Symmetric DPLR KDA):** Learned per-pair phase step
    `theta_t = pi * tanh(W_theta(x_t))` (zero-init `W_theta`), cumulative `Theta_t = cumsum(theta_t) mod 2*pi`,
    counter-rotating `(q, k)` via `R(-Theta_t)` with over-relaxed erase gate `b_t in (0, 2)^{d_k}`
    (`2 * logistic_shift(0.5 * beta_base, delta_b)`).
  * **Fixes H2, B3, B4 (Smooth Chunk-Start Root-Surprise `s = g^2`):** Evaluates relative key-value
    prediction error against the frozen chunk-start primary state `S_{nC}^{(0)}` in parallel across
    all tokens of chunk `n` before the WY triangular solve, parameterizing `g in (0, 1)` directly
    without `sqrt(s)` so gradients at `g -> 0` are finite and `g = 0` yields exact `S_t == S_{t-1}`.
  * **Fixes B2 & B6 (Tiered `M=8` Multi-Page Router + Write-Mass Budget + Checkpointing):** Top-`k`
    (`k=2` of `M=8`) routed write/read pages with log-spaced decay priors, two-sided write-mass
    budget regularizer (`mu_max` & `mu_min = 0.2 * mu_max`), macro-segment activation checkpointing,
    and `O(1)`-memory `recurrent_hssd_step` ring-buffer decoding.

Clean-room Apache-2.0 / MIT PyTorch implementation.
"""
from __future__ import annotations

import importlib.util
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
from .gdn2 import _infer_gdn_dims, l2norm, logistic_shift
from .introspect import LINEAR, attention_module, find_text_model, inspect_model, layer_type_of

log = logging.getLogger("qloop")

DEFAULT_MU_MAX: tuple[float, ...] = (0.0025, 0.0025, 0.025, 0.025, 0.25, 0.25, 0.25, 0.25)


def has_fla() -> bool:
    """Return True if optional MIT `flash-linear-attention` (`fla`) package is importable."""
    return importlib.util.find_spec("fla") is not None


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


# ============================================================================ config

@dataclass
class HSSDConfig:
    rank: int = 64
    layers: list[int] | None = None
    chunk_size: int = 64
    window_chunks: int = 8
    num_pages: int = 8
    k_pages: int = 2
    top_k_pages: int | None = None
    segment_chunks: int = 16
    checkpoint_segments: bool = True
    gate_logit_cap: float = 8.0
    decay_log_cap: float = 2.0
    kappa_init: float = 2.0
    tau_init: float = 2.2
    surprise_slope_init: float = 3.0
    surprise_thresh_init: float = 0.85
    budget_weight: float = 1.0
    mu_max: tuple[float, ...] = DEFAULT_MU_MAX
    mu_min_ratio: float = 0.2
    decay_gamma_min: float = 1e-4
    decay_gamma_max: float = 5e-2

    def __post_init__(self) -> None:
        if self.top_k_pages is not None:
            self.k_pages = int(self.top_k_pages)
        else:
            self.top_k_pages = int(self.k_pages)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> HSSDConfig:
        clean = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        if "mu_max" in clean and isinstance(clean["mu_max"], list):
            clean["mu_max"] = tuple(float(x) for x in clean["mu_max"])
        known = {f.name for f in fields(cls)}
        bad = set(clean) - known
        if bad:
            raise ValueError(f"unknown hssd.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**clean)

    def resolved_mu_max(self) -> tuple[float, ...]:
        if len(self.mu_max) == self.num_pages:
            return tuple(float(x) for x in self.mu_max)
        if self.mu_max != DEFAULT_MU_MAX:
            raise ValueError(f"hssd.mu_max length {len(self.mu_max)} != num_pages {self.num_pages}")
        if self.num_pages == 1:
            return (0.25,)
        lo, hi = math.log(0.0025), math.log(0.25)
        return tuple(math.exp(lo + (hi - lo) * i / (self.num_pages - 1)) for i in range(self.num_pages))

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"hssd.rank must be >= 1, got {self.rank}")
        if self.chunk_size < 1:
            raise ValueError(f"hssd.chunk_size must be >= 1, got {self.chunk_size}")
        if self.window_chunks < 0:
            raise ValueError(f"hssd.window_chunks must be >= 0, got {self.window_chunks}")
        if self.num_pages < 1:
            raise ValueError(f"hssd.num_pages must be >= 1, got {self.num_pages}")
        if not 1 <= self.k_pages <= self.num_pages:
            raise ValueError(f"hssd.k_pages must be in [1, {self.num_pages}], got {self.k_pages}")
        if self.segment_chunks < 1:
            raise ValueError(f"hssd.segment_chunks must be >= 1, got {self.segment_chunks}")
        if self.gate_logit_cap <= 0 or self.decay_log_cap <= 0:
            raise ValueError("hssd.gate_logit_cap and hssd.decay_log_cap must be > 0")
        if self.kappa_init <= 0 or self.surprise_slope_init <= 0:
            raise ValueError("hssd.kappa_init and hssd.surprise_slope_init must be > 0")
        if not 0.0 <= self.mu_min_ratio <= 1.0:
            raise ValueError(f"hssd.mu_min_ratio must be in [0, 1], got {self.mu_min_ratio}")
        if self.decay_gamma_min <= 0 or self.decay_gamma_max < self.decay_gamma_min:
            raise ValueError("hssd decay_gamma bounds must satisfy 0 < min <= max")
        _ = self.resolved_mu_max()


# ============================================================================ math primitives

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Singularity-free `sqrt(max(x, 0))` with exact 0 value and finite 0 derivative at `x <= 0`."""
    pos = x > 0.0
    safe = torch.where(pos, x, torch.ones_like(x))
    return torch.where(pos, torch.sqrt(safe), torch.zeros_like(x))


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """L2 norm with finite 0 derivative at exact zero vectors."""
    return _safe_sqrt((x * x).sum(dim=dim, keepdim=keepdim))


def _rms_norm_scaled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = upcast(x)
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * upcast(weight)


def apply_2d_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Block-diagonal 2D phase rotation `R(theta)` on last dim `d = 2 * (d // 2)` (Theorem 1)."""
    require(x.shape[-1] % 2 == 0, "apply_2d_rotation requires an even last dimension", x=x)
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    return torch.stack([out0, out1], dim=-1).reshape_as(x)


def disjoint_horizon_bounds(t_1based: int, chunk_size: int = 64, window_chunks: int = 8) -> tuple[int, int, int]:
    """Return `(chunk_idx_n, k_far, far_cutoff)` for 1-based token index `t_1based` (Fix B1).

    `I_far = [1, far_cutoff]` and `I_loc = [far_cutoff + 1, t_1based]` partition `[1, t_1based]`
    with zero overlap and zero gap.
    """
    require(t_1based >= 1, "t_1based must be >= 1", t_1based=t_1based)
    n = (int(t_1based) - 1) // int(chunk_size)
    k_far = max(0, n - int(window_chunks))
    far_cutoff = k_far * int(chunk_size)
    return n, k_far, far_cutoff


class PeakBoltzmannGate(nn.Module):
    """Peak-augmented size-free Boltzmann local-hit gate (`Fix B5`).

    Computes:
        stat = 0.5 * (LSE(scores) - log(N_t)) + 0.5 * (max(scores) - sqrt(2 * log(max(N_t, 2))))
        p_hit = sigmoid(kappa_h * (stat - tau_h))
    """

    def __init__(self, num_heads: int = 1, kappa_init: float = 2.0, tau_init: float = 2.2):
        super().__init__()
        self.num_heads = int(num_heads)
        self.log_kappa = nn.Parameter(torch.full((self.num_heads,), math.log(float(kappa_init))))
        self.tau = nn.Parameter(torch.full((self.num_heads,), float(tau_init)))

    def forward(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        sf = upcast(scores)
        if valid_mask is None:
            n_val = float(sf.shape[-1])
            lse = torch.logsumexp(sf, dim=-1)
            max_sc = sf.max(dim=-1).values
            log_n = sf.new_tensor(math.log(max(n_val, 1.0)))
            gumbel = sf.new_tensor(math.sqrt(2.0 * math.log(max(n_val, 2.0))))
            has_valid: torch.Tensor | None = None
        else:
            vm = valid_mask.to(dtype=torch.bool).expand_as(sf)
            counts = vm.sum(dim=-1)
            has_valid = counts > 0
            safe_vm = torch.where(has_valid.unsqueeze(-1), vm, torch.ones_like(vm))
            masked = sf.masked_fill(~safe_vm, float("-inf"))
            lse = torch.logsumexp(masked, dim=-1)
            max_sc = masked.max(dim=-1).values
            n_eff = counts.clamp_min(1).to(sf.dtype)
            log_n = torch.log(n_eff)
            gumbel = torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0)))

        stat = 0.5 * (lse - log_n) + 0.5 * (max_sc - gumbel)
        kappa = upcast(self.log_kappa).exp().to(stat.dtype)
        tau = upcast(self.tau).to(stat.dtype)
        if self.num_heads == 1 or stat.ndim == 0:
            kappa_b = kappa[0]
            tau_b = tau[0]
        elif stat.ndim >= 2 and stat.shape[1] == self.num_heads:
            shape = [1, self.num_heads] + [1] * (stat.ndim - 2)
            kappa_b = kappa.view(*shape)
            tau_b = tau.view(*shape)
        elif stat.shape[-1] == self.num_heads:
            kappa_b = kappa
            tau_b = tau
        else:
            raise ValueError(f"PeakBoltzmannGate(num_heads={self.num_heads}) incompatible with stat shape {stat.shape}")

        p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
        if has_valid is not None:
            p_hit = torch.where(has_valid, p_hit, torch.zeros_like(p_hit))
        return p_hit.to(scores.dtype)


# ============================================================================ O(1) cache state

@dataclass(eq=False)
class HSSDCacheState:
    """Fixed-footprint `O(1)` incremental decode cache for one `(layer, pass)` HSSD slot.

    Maintains:
      * `page_states`: `[B, M, H_v, d_k, d_v]` live multi-page state after `step_count` tokens.
      * `s_chunk_start`: `[B, H_v, d_k, d_v]` frozen primary state `S_{nC}^{(0)}` at start of chunk `n`.
      * `theta_cum`: `[B, H_v, d_k // 2]` cumulative 2D phase rotation `Theta_t mod 2*pi`.
      * `boundary_ring`: length `W + 1` tuple of `[B, M, H_v, d_k, d_v]` boundary snapshots where
        `boundary_ring[0] == S_boundary[max(0, n - W)]` holds identically for every chunk `n >= 0`.
      * `swa_k_ring`, `swa_v_ring`, `swa_valid_ring`: circular buffer of size `(W + 1) * C` (`9 * C`).
    """

    step_count: int
    chunk_size: int
    window_chunks: int
    page_states: torch.Tensor
    s_chunk_start: torch.Tensor
    theta_cum: torch.Tensor
    boundary_ring: tuple[torch.Tensor, ...]
    swa_k_ring: torch.Tensor
    swa_v_ring: torch.Tensor
    swa_valid_ring: torch.Tensor

    @property
    def max_swa_len(self) -> int:
        return (self.window_chunks + 1) * self.chunk_size

    @property
    def shape(self) -> torch.Size:
        return self.page_states.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.page_states.dtype

    @property
    def device(self) -> torch.device:
        return self.page_states.device

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_pages: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        *,
        chunk_size: int = 64,
        window_chunks: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> HSSDCacheState:
        p_states = torch.zeros(batch_size, num_pages, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=dtype)
        s_cstart = p_states[:, 0].clone()
        theta = torch.zeros(batch_size, num_v_heads, head_k_dim // 2, device=device, dtype=dtype)
        b_ring = tuple(p_states for _ in range(window_chunks + 1))
        max_win = (window_chunks + 1) * chunk_size
        k_ring = torch.zeros(batch_size, max_win, num_v_heads, head_k_dim, device=device, dtype=dtype)
        v_ring = torch.zeros(batch_size, max_win, num_v_heads, head_v_dim, device=device, dtype=dtype)
        val_ring = torch.zeros(batch_size, max_win, device=device, dtype=torch.bool)
        return cls(
            step_count=0,
            chunk_size=int(chunk_size),
            window_chunks=int(window_chunks),
            page_states=p_states,
            s_chunk_start=s_cstart,
            theta_cum=theta,
            boundary_ring=b_ring,
            swa_k_ring=k_ring,
            swa_v_ring=v_ring,
            swa_valid_ring=val_ring,
        )

    @classmethod
    def from_tensor(
        cls,
        state_tensor: torch.Tensor,
        *,
        num_pages: int = 8,
        chunk_size: int = 64,
        window_chunks: int = 8,
    ) -> HSSDCacheState:
        st = upcast(state_tensor)
        if st.ndim == 4:
            bsz, hv, dk, dv = st.shape
            p_states = st.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone()
        elif st.ndim == 5:
            bsz, mp, hv, dk, dv = st.shape
            require(mp == num_pages, f"expected {num_pages} state pages, got {mp}", state=st)
            p_states = st.clone()
        else:
            raise ValueError(f"initial_state tensor must be 4D or 5D, got shape {tuple(st.shape)}")
        base = cls.zeros(
            bsz,
            num_pages,
            hv,
            dk,
            dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            device=p_states.device,
            dtype=p_states.dtype,
        )
        base.page_states = p_states
        base.s_chunk_start = p_states[:, 0].clone()
        base.boundary_ring = tuple(p_states for _ in range(window_chunks + 1))
        return base

    def clone(self) -> HSSDCacheState:
        return HSSDCacheState(
            step_count=self.step_count,
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=self.page_states.clone(),
            s_chunk_start=self.s_chunk_start.clone(),
            theta_cum=self.theta_cum.clone(),
            boundary_ring=tuple(t.clone() for t in self.boundary_ring),
            swa_k_ring=self.swa_k_ring.clone(),
            swa_v_ring=self.swa_v_ring.clone(),
            swa_valid_ring=self.swa_valid_ring.clone(),
        )

    def detach(self) -> HSSDCacheState:
        return HSSDCacheState(
            step_count=self.step_count,
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=self.page_states.detach(),
            s_chunk_start=self.s_chunk_start.detach(),
            theta_cum=self.theta_cum.detach(),
            boundary_ring=tuple(t.detach() for t in self.boundary_ring),
            swa_k_ring=self.swa_k_ring.detach(),
            swa_v_ring=self.swa_v_ring.detach(),
            swa_valid_ring=self.swa_valid_ring.detach(),
        )

    def to(self, *args: Any, **kwargs: Any) -> HSSDCacheState:
        p_states = self.page_states.to(*args, **kwargs)
        dev = p_states.device
        dt = p_states.dtype
        return HSSDCacheState(
            step_count=self.step_count,
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=p_states,
            s_chunk_start=self.s_chunk_start.to(device=dev, dtype=dt),
            theta_cum=self.theta_cum.to(device=dev, dtype=dt),
            boundary_ring=tuple(t.to(device=dev, dtype=dt) for t in self.boundary_ring),
            swa_k_ring=self.swa_k_ring.to(device=dev, dtype=dt),
            swa_v_ring=self.swa_v_ring.to(device=dev, dtype=dt),
            swa_valid_ring=self.swa_valid_ring.to(device=dev),
        )

    def state_equal(self, other: HSSDCacheState) -> bool:
        if not isinstance(other, HSSDCacheState):
            return False
        return (
            self.step_count == other.step_count
            and torch.equal(self.page_states, other.page_states)
            and torch.equal(self.s_chunk_start, other.s_chunk_start)
            and torch.equal(self.theta_cum, other.theta_cum)
            and len(self.boundary_ring) == len(other.boundary_ring)
            and all(torch.equal(a, b) for a, b in zip(self.boundary_ring, other.boundary_ring))
        )


# ============================================================================ core HSSD kernels

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


def _coerce_init_state(
    initial_state: HSSDCacheState | torch.Tensor | None,
    *,
    bsz: int,
    num_pages: int,
    hv: int,
    dk: int,
    dv: int,
    chunk_size: int,
    window_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HSSDCacheState:
    if initial_state is None:
        return HSSDCacheState.zeros(
            bsz,
            num_pages,
            hv,
            dk,
            dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            device=device,
            dtype=dtype,
        )
    if isinstance(initial_state, HSSDCacheState):
        require(
            initial_state.chunk_size == chunk_size and initial_state.window_chunks == window_chunks,
            "HSSDCacheState chunk_size/window_chunks mismatch",
            got=(initial_state.chunk_size, initial_state.window_chunks),
            want=(chunk_size, window_chunks),
        )
        return initial_state.to(device=device, dtype=dtype)
    if isinstance(initial_state, torch.Tensor):
        return HSSDCacheState.from_tensor(
            initial_state,
            num_pages=num_pages,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
        ).to(device=device, dtype=dtype)
    raise TypeError(f"Unsupported initial_state type: {type(initial_state).__name__}")


def _expand_gamma(gamma: torch.Tensor, num_pages: int) -> torch.Tensor:
    gf = upcast(gamma).clamp_min(0.0)
    if gf.ndim == 4:
        return gf.unsqueeze(2).expand(gf.shape[0], gf.shape[1], num_pages, gf.shape[2], gf.shape[3])
    require(gf.ndim == 5 and gf.shape[2] == num_pages, "gamma must be [B,T,H,D] or [B,T,M,H,D]", gamma=gf)
    return gf


def _strict_lower_decayed_gram_fallback(k_eff: torch.Tensor, cum_log_a: torch.Tensor) -> torch.Tensor:
    """Overflow-free fallback when single-chunk log-decay span exceeds IEEE-754 fp32 range."""
    # k_eff, cum_log_a: [B, M, H, C, d_k]
    c_len = k_eff.shape[-2]
    idx = torch.arange(c_len, device=k_eff.device)
    strict_lower = idx[:, None] > idx[None, :]
    diff = cum_log_a[..., :, None, :] - cum_log_a[..., None, :, :]
    safe_diff = torch.where(strict_lower.unsqueeze(-1), diff, torch.zeros_like(diff))
    gram = (k_eff[..., :, None, :] * k_eff[..., None, :, :] * safe_diff.exp()).sum(dim=-1)
    return torch.where(strict_lower, gram, torch.zeros_like(gram))


def _wy_kda_state_update(
    page_states: torch.Tensor,
    k_eff: torch.Tensor,
    v_eff: torch.Tensor,
    log_a: torch.Tensor,
) -> torch.Tensor:
    """Exact chunk-parallel WY state transition for symmetric DPLR KDA across `(B, M, H_v)`.

    Shapes:
        page_states: [B, M, H_v, d_k, d_v]
        k_eff:       [B, M, H_v, c_len, d_k]
        v_eff:       [B, M, H_v, c_len, d_v]
        log_a:       [B, M, H_v, c_len, d_k]  (<= 0)
    """
    c_len = k_eff.shape[-2]
    if c_len == 1:
        dec = log_a.squeeze(-2).exp().unsqueeze(-1) * page_states
        ke = k_eff.squeeze(-2)
        ve = v_eff.squeeze(-2)
        mem = (ke.unsqueeze(-2) @ dec).squeeze(-2)
        return dec + ke.unsqueeze(-1) * (ve - mem).unsqueeze(-2)

    cum_log_a = torch.cumsum(log_a, dim=-2)
    span = (cum_log_a[..., 0, :] - cum_log_a[..., -1, :]).detach()
    if bool((span <= 150.0).all()):
        mu = 0.5 * (cum_log_a[..., :1, :] + cum_log_a[..., -1:, :])
        f_pos = (cum_log_a - mu).exp()
        f_neg = (mu - cum_log_a).exp()
        l_mat = torch.tril((k_eff * f_pos) @ (k_eff * f_neg).transpose(-1, -2), diagonal=-1)
    else:
        l_mat = _strict_lower_decayed_gram_fallback(k_eff, cum_log_a)

    gamma_from_0 = cum_log_a.exp()
    rhs = v_eff - (gamma_from_0 * k_eff) @ page_states
    eye = torch.eye(c_len, dtype=k_eff.dtype, device=k_eff.device)
    u_wy = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)

    gamma_to_end = (cum_log_a[..., -1:, :] - cum_log_a).exp()
    d_chunk = cum_log_a[..., -1, :].exp().unsqueeze(-1)
    return d_chunk * page_states + (gamma_to_end * k_eff).transpose(-1, -2) @ u_wy


def _far_readout(
    q_hat_c: torch.Tensor,
    s_far: torch.Tensor,
    read_logits_c: torch.Tensor,
    top_k_pages: int,
) -> torch.Tensor:
    """Top-`k` page-routed readout from `S_boundary[k_far]` (`[B, c_len, H_v, d_v]`)."""
    bsz, c_len, hv, _dk = q_hat_c.shape
    dv = s_far.shape[-1]
    pi_read = F.softmax(read_logits_c, dim=-1)
    top_vals, top_idx = torch.topk(pi_read, k=top_k_pages, dim=-1)
    top_w = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    q_far = q_hat_c.permute(0, 2, 1, 3).unsqueeze(1)          # [B, 1, H_v, c_len, d_k]
    all_far = (q_far @ s_far).permute(0, 3, 1, 2, 4)          # [B, c_len, M, H_v, d_v]
    gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, c_len, top_k_pages, hv, dv)
    picked = torch.gather(all_far, dim=2, index=gather_idx)
    return (picked * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)


def _compute_root_surprise_and_routing(
    k_hat_c: torch.Tensor,
    v_c: torch.Tensor,
    s_chunk_start: torch.Tensor,
    sbias_c: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits_c: torch.Tensor,
    top_k_pages: int,
    wmask_c: torch.Tensor | None,
    g_override_c: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(g_page [B, c_len, M, H_v], u_sum [M], valid_tok_count [])` without `sqrt(0)`."""
    bsz, c_len, hv, _dk = k_hat_c.shape
    if g_override_c is not None:
        g_root = g_override_c
    else:
        pred_v = (k_hat_c.transpose(1, 2) @ s_chunk_start).transpose(1, 2)
        eps_err = _safe_norm(v_c - pred_v, dim=-1) / (_safe_norm(v_c, dim=-1) + 1e-6)
        slope = surprise_slope.view(1, 1, hv)
        thresh = surprise_thresh.view(1, 1, hv)
        g_root = torch.sigmoid(slope * (eps_err - thresh) + sbias_c)

    if wmask_c is not None:
        g_root = g_root * wmask_c.unsqueeze(-1)
        valid_toks = wmask_c.sum()
    else:
        valid_toks = g_root.new_tensor(float(bsz * c_len))

    pi_write = F.softmax(write_logits_c, dim=-1)
    g_sq_mean = (g_root * g_root).mean(dim=-1, keepdim=True)
    u_sum = (pi_write * g_sq_mean).sum(dim=(0, 1))

    top_vals, top_idx = torch.topk(pi_write, k=top_k_pages, dim=-1)
    top_probs = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    top_roots = torch.sqrt(top_probs.clamp_min(1e-30))
    write_root = torch.zeros_like(pi_write).scatter(-1, top_idx, top_roots)
    g_page = write_root.unsqueeze(-1) * g_root.unsqueeze(2)
    return g_page, u_sum, valid_toks


def compute_write_budget_loss(
    duty_vec: torch.Tensor,
    mu_max: torch.Tensor,
    mu_min_ratio: float = 0.2,
    budget_weight: float = 1.0,
) -> torch.Tensor:
    """Two-sided cumulative write-mass budget regularizer (`Fix B2`)."""
    target_max = mu_max.to(device=duty_vec.device, dtype=duty_vec.dtype)
    target_min = float(mu_min_ratio) * target_max
    over = F.relu(duty_vec - target_max)
    under = F.relu(target_min - duty_vec)
    return float(budget_weight) * (over * over + under * under).sum()


def recurrent_hssd_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    state: HSSDCacheState,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    top_k_pages: int = 2,
    write_mask: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    return_duty_sum: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState] | tuple[torch.Tensor, HSSDCacheState, torch.Tensor, torch.Tensor]:
    """Single-token `O(1)`-memory HSSD v3 autoregressive decode step.

    Accepts `[B, 1, ...]` (returning `[B, 1, H_v, d_v]`) or `[B, ...]` (returning `[B, H_v, d_v]`).
    """
    squeeze_time = query.ndim == 3
    if squeeze_time:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        gamma = gamma.unsqueeze(1)
        b_gate = b_gate.unsqueeze(1)
        w_gate = w_gate.unsqueeze(1)
        theta = theta.unsqueeze(1)
        surprise_bias = surprise_bias.unsqueeze(1)
        write_logits = write_logits.unsqueeze(1)
        read_logits = read_logits.unsqueeze(1)
        q_swa = q_swa.unsqueeze(1)
        k_swa = k_swa.unsqueeze(1)
        v_swa = v_swa.unsqueeze(1)
        if g_override is not None:
            g_override = g_override.unsqueeze(1)
    require(query.shape[1] == 1, "recurrent_hssd_step expects T == 1", query=query)

    init_dtype = query.dtype
    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    bsz, _one, _hv, dk = k.shape
    num_pages = state.page_states.shape[1]
    acc_dt = q.dtype
    st = state.to(device=q.device, dtype=acc_dt)

    gamma_f = _expand_gamma(gamma, num_pages).to(acc_dt)
    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    th = upcast(theta).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    qs = upcast(q_swa).to(acc_dt)
    ks = upcast(k_swa).to(acc_dt)
    vs = upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)

    if write_mask is not None:
        wm = write_mask.reshape(bsz, 1).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_tok = wm.squeeze(1) > 0.5
        th = th * wm.unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_tok = torch.ones(bsz, device=q.device, dtype=torch.bool)

    idx0 = st.step_count
    c_size = st.chunk_size
    w_chunks = st.window_chunks
    s_cstart = st.page_states[:, 0] if (idx0 % c_size == 0) else st.s_chunk_start

    theta_cum = torch.remainder(st.theta_cum + th.squeeze(1), 2.0 * math.pi)
    sc = (dk ** -0.5) if scale is None else float(scale)
    q_hat = apply_2d_rotation(q, -theta_cum.unsqueeze(1))
    if sc != 1.0:
        q_hat = q_hat * sc
    k_hat = apply_2d_rotation(k, -theta_cum.unsqueeze(1))

    # 1. Disjoint far-horizon readout from S_boundary[max(0, n - W)] == st.boundary_ring[0]
    o_far = _far_readout(q_hat, st.boundary_ring[0], rl, top_k_pages)

    # 2. Update O(1) circular SWA ring buffer at slot = idx0 % ((W + 1) * C) and attend over I_loc
    max_win = st.max_swa_len
    slot = idx0 % max_win
    slot_idx = torch.tensor([slot], device=q.device, dtype=torch.long)
    swa_k_ring = st.swa_k_ring.index_copy(1, slot_idx, ks)
    swa_v_ring = st.swa_v_ring.index_copy(1, slot_idx, vs)
    swa_valid_ring = st.swa_valid_ring.index_copy(1, slot_idx, valid_tok.unsqueeze(1))

    _n_chunk, _k_far, far_cutoff = disjoint_horizon_bounds(idx0 + 1, c_size, w_chunks)
    win_indices = torch.arange(far_cutoff, idx0 + 1, device=q.device) % max_win
    k_win = swa_k_ring.index_select(1, win_indices).transpose(1, 2)       # [B, H_v, N_t, d_k]
    v_win = swa_v_ring.index_select(1, win_indices).transpose(1, 2)       # [B, H_v, N_t, d_v]
    val_win = swa_valid_ring.index_select(1, win_indices)[:, None, None, :]  # [B, 1, 1, N_t]

    q_loc = qs.transpose(1, 2)                                            # [B, H_v, 1, d_k]
    scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)             # [B, H_v, 1, N_t]
    has_valid = val_win.any(dim=-1)
    safe_val = torch.where(has_valid.unsqueeze(-1), val_win, torch.ones_like(val_win))
    masked_sc = scores.masked_fill(~safe_val, float("-inf"))
    probs = torch.softmax(masked_sc, dim=-1)
    probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
    o_loc = (probs @ v_win).transpose(1, 2)                               # [B, 1, H_v, d_v]

    if boltzmann_gate is not None:
        p_hit = boltzmann_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)
    else:
        default_gate = PeakBoltzmannGate(1).to(device=q.device, dtype=acc_dt)
        p_hit = default_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)

    out = p_hit * o_loc + (1.0 - p_hit) * o_far

    # 3. Smooth chunk-start root-surprise `s = g^2` & single-step multi-page state update
    g_page, u_sum, valid_cnt = _compute_root_surprise_and_routing(
        k_hat, v, s_cstart, sb, slp, thr, wl, top_k_pages, wm, go
    )
    sqrt_b = _safe_sqrt(bf)
    k_eff = (g_page.unsqueeze(-1) * (sqrt_b * k_hat).unsqueeze(2)).permute(0, 2, 3, 1, 4)
    v_eff = (g_page.unsqueeze(-1) * (wf * v).unsqueeze(2)).permute(0, 2, 3, 1, 4)
    log_a = (-(g_page * g_page).unsqueeze(-1) * gamma_f).permute(0, 2, 3, 1, 4)
    new_pages = _wy_kda_state_update(st.page_states, k_eff, v_eff, log_a)

    new_step = idx0 + 1
    if new_step % c_size == 0:
        new_b_ring = (*st.boundary_ring[1:], new_pages)
        new_cstart = new_pages[:, 0]
    else:
        new_b_ring = st.boundary_ring
        new_cstart = s_cstart

    next_state = HSSDCacheState(
        step_count=new_step,
        chunk_size=c_size,
        window_chunks=w_chunks,
        page_states=new_pages,
        s_chunk_start=new_cstart,
        theta_cum=theta_cum,
        boundary_ring=new_b_ring,
        swa_k_ring=swa_k_ring,
        swa_v_ring=swa_v_ring,
        swa_valid_ring=swa_valid_ring,
    )
    out_cast = out.squeeze(1).to(init_dtype) if squeeze_time else out.to(init_dtype)
    if return_duty_sum:
        return out_cast, next_state, u_sum, valid_cnt
    return out_cast, next_state


def recurrent_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Token-by-token sequential HSSD v3 reference execution over `T` steps via `recurrent_hssd_step`."""
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    acc_dt = torch.float64 if query.dtype == torch.float64 else torch.float32
    state = _coerce_init_state(
        initial_state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        device=query.device,
        dtype=acc_dt,
    )
    gf = _expand_gamma(gamma, num_pages)
    outs = []
    u_total = query.new_zeros(num_pages, dtype=acc_dt)
    tok_total = query.new_zeros((), dtype=acc_dt)

    for i in range(seq_len):
        wm_i = None if write_mask is None else write_mask[:, i : i + 1]
        go_i = None if g_override is None else g_override[:, i : i + 1]
        step_res = recurrent_hssd_step(
            query[:, i : i + 1],
            key[:, i : i + 1],
            value[:, i : i + 1],
            gf[:, i : i + 1],
            b_gate[:, i : i + 1],
            w_gate[:, i : i + 1],
            theta[:, i : i + 1],
            surprise_bias[:, i : i + 1],
            surprise_slope,
            surprise_thresh,
            write_logits[:, i : i + 1],
            read_logits[:, i : i + 1],
            q_swa[:, i : i + 1],
            k_swa[:, i : i + 1],
            v_swa[:, i : i + 1],
            state,
            boltzmann_gate=boltzmann_gate,
            top_k_pages=top_k_pages,
            write_mask=wm_i,
            g_override=go_i,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            return_duty_sum=True,
        )
        o_i, state, u_i, cnt_i = step_res
        outs.append(o_i)
        u_total = u_total + u_i
        tok_total = tok_total + cnt_i

    out = torch.cat(outs, dim=1)
    final_st = state if output_final_state else None
    if return_aux:
        duty_vec = u_total / tok_total.clamp_min(1.0)
        return out, final_st, duty_vec
    return out, final_st


def _chunk_slices_for_span(step_offset: int, seq_len: int, chunk_size: int) -> list[tuple[int, int]]:
    """Partition local `[0, seq_len)` along global chunk boundaries `g // chunk_size`."""
    slices: list[tuple[int, int]] = []
    cur = 0
    while cur < seq_len:
        g_cur = step_offset + cur
        rem = chunk_size - (g_cur % chunk_size)
        nxt = min(seq_len, cur + rem)
        slices.append((cur, nxt))
        cur = nxt
    return slices


def _run_chunk_group(
    chunk_bounds: list[tuple[int, int]],
    *,
    step_offset: int,
    swa_base_global: int,
    chunk_size: int,
    window_chunks: int,
    top_k_pages: int,
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    gamma_f: torch.Tensor,
    bf: torch.Tensor,
    wf: torch.Tensor,
    sb: torch.Tensor,
    slp: torch.Tensor,
    thr: torch.Tensor,
    wl: torch.Tensor,
    rl: torch.Tensor,
    qs: torch.Tensor,
    all_ks: torch.Tensor,
    all_vs: torch.Tensor,
    all_valid_swa: torch.Tensor,
    wm: torch.Tensor | None,
    go: torch.Tensor | None,
    gate_log_kappa: torch.Tensor,
    gate_tau: torch.Tensor,
    page_states: torch.Tensor,
    s_chunk_start: torch.Tensor,
    boundary_ring: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Execute a contiguous group of global-aligned chunks (pure functional for checkpointing)."""
    dk = k_hat.shape[-1]
    hv = k_hat.shape[2]
    outs: list[torch.Tensor] = []
    u_acc = q_hat.new_zeros(wl.shape[-1])
    tok_acc = q_hat.new_zeros(())

    kappa = gate_log_kappa.exp()
    kappa_b = kappa[0] if kappa.numel() == 1 else kappa.view(1, hv, 1)
    tau_b = gate_tau[0] if gate_tau.numel() == 1 else gate_tau.view(1, hv, 1)

    p_cur = page_states
    sc_cur = s_chunk_start
    ring_cur = boundary_ring

    for l0, l1 in chunk_bounds:
        g0 = step_offset + l0
        g1 = step_offset + l1
        if g0 % chunk_size == 0:
            sc_cur = p_cur[:, 0]

        q_c = q_hat[:, l0:l1]
        k_c = k_hat[:, l0:l1]
        v_c = v[:, l0:l1]
        rl_c = rl[:, l0:l1]
        wl_c = wl[:, l0:l1]
        sb_c = sb[:, l0:l1]
        wm_c = None if wm is None else wm[:, l0:l1]
        go_c = None if go is None else go[:, l0:l1]

        # 1. Far-horizon readout strictly from S_boundary[max(0, n - W)] == ring_cur[0]
        o_far = _far_readout(q_c, ring_cur[0], rl_c, top_k_pages)

        # 2. Exact causal local SWA over I_loc(t) = [far_cutoff + 1, t] (0-based [far_cutoff, g1))
        _n_idx, _k_far, far_cutoff = disjoint_horizon_bounds(g0 + 1, chunk_size, window_chunks)
        s0 = far_cutoff - swa_base_global
        s1 = g1 - swa_base_global
        k_win = all_ks[:, s0:s1].transpose(1, 2)               # [B, H_v, win_len, d_k]
        v_win = all_vs[:, s0:s1].transpose(1, 2)               # [B, H_v, win_len, d_v]
        val_keys = all_valid_swa[:, s0:s1]                     # [B, win_len]

        q_pos = torch.arange(g0, g1, device=q_c.device)[:, None]
        k_pos = torch.arange(far_cutoff, g1, device=q_c.device)[None, :]
        valid_mask = (k_pos <= q_pos)[None, None, :, :] & val_keys[:, None, None, :]

        q_loc = qs[:, l0:l1].transpose(1, 2)                   # [B, H_v, c_len, d_k]
        scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)
        has_valid = valid_mask.any(dim=-1)
        safe_vm = torch.where(has_valid.unsqueeze(-1), valid_mask, torch.ones_like(valid_mask))
        masked_sc = scores.masked_fill(~safe_vm, float("-inf"))

        probs = torch.softmax(masked_sc, dim=-1)
        probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
        o_loc = (probs @ v_win).transpose(1, 2)

        counts = valid_mask.expand_as(scores).sum(dim=-1)
        has_v_sc = counts > 0
        n_eff = counts.clamp_min(1).to(scores.dtype)
        lse = torch.logsumexp(masked_sc, dim=-1)
        max_sc = masked_sc.max(dim=-1).values
        stat = 0.5 * (lse - torch.log(n_eff)) + 0.5 * (max_sc - torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0))))
        p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
        p_hit = torch.where(has_v_sc, p_hit, torch.zeros_like(p_hit)).transpose(1, 2).unsqueeze(-1)

        outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)

        # 3. Parallel chunk-start root-surprise + WY multi-page state update
        g_page, u_sum, cnt_c = _compute_root_surprise_and_routing(
            k_c, v_c, sc_cur, sb_c, slp, thr, wl_c, top_k_pages, wm_c, go_c
        )
        u_acc = u_acc + u_sum
        tok_acc = tok_acc + cnt_c

        sqrt_b = _safe_sqrt(bf[:, l0:l1])
        k_eff = (g_page.unsqueeze(-1) * (sqrt_b * k_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
        v_eff = (g_page.unsqueeze(-1) * (wf[:, l0:l1] * v_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
        log_a = (-(g_page * g_page).unsqueeze(-1) * gamma_f[:, l0:l1]).permute(0, 2, 3, 1, 4)
        p_cur = _wy_kda_state_update(p_cur, k_eff, v_eff, log_a)

        if g1 % chunk_size == 0:
            ring_cur = (*ring_cur[1:], p_cur)
            sc_cur = p_cur[:, 0]

    seg_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
    return seg_out, p_cur, sc_cur, u_acc, tok_acc, ring_cur


def chunk_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    segment_chunks: int = 4,
    checkpoint_segments: bool = True,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Parallel chunkwise WY + disjoint local SWA HSSD v3 kernel (`O(T)` prefill & training)."""
    init_dtype = query.dtype
    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    bsz, seq_len, hv, dk = k.shape
    dv = v.shape[-1]
    acc_dt = q.dtype

    st = _coerce_init_state(
        initial_state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        device=q.device,
        dtype=acc_dt,
    )
    if seq_len == 1:
        step_res = recurrent_hssd_step(
            query,
            key,
            value,
            _expand_gamma(gamma, num_pages),
            b_gate,
            w_gate,
            theta,
            surprise_bias,
            surprise_slope,
            surprise_thresh,
            write_logits,
            read_logits,
            q_swa,
            k_swa,
            v_swa,
            st,
            boltzmann_gate=boltzmann_gate,
            top_k_pages=top_k_pages,
            write_mask=write_mask,
            g_override=g_override,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            return_duty_sum=True,
        )
        o_step, st_next, u_sum, cnt_sum = step_res
        final_st = st_next if output_final_state else None
        if return_aux:
            return o_step, final_st, u_sum / cnt_sum.clamp_min(1.0)
        return o_step, final_st

    gamma_f = _expand_gamma(gamma, num_pages).to(acc_dt)
    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    th = upcast(theta).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    qs = upcast(q_swa).to(acc_dt)
    ks = upcast(k_swa).to(acc_dt)
    vs = upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)

    if write_mask is not None:
        wm = write_mask.reshape(bsz, seq_len).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_cur = wm > 0.5
        th = th * wm.unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_cur = torch.ones(bsz, seq_len, device=q.device, dtype=torch.bool)

    theta_cum_all = torch.remainder(st.theta_cum.unsqueeze(1) + torch.cumsum(th, dim=1), 2.0 * math.pi)
    sc = (dk ** -0.5) if scale is None else float(scale)
    q_hat = apply_2d_rotation(q, -theta_cum_all)
    if sc != 1.0:
        q_hat = q_hat * sc
    k_hat = apply_2d_rotation(k, -theta_cum_all)

    step_offset = st.step_count
    max_win = st.max_swa_len
    _, _, swa_base_global = disjoint_horizon_bounds(step_offset + 1, chunk_size, window_chunks)
    if step_offset > swa_base_global:
        prior_idx = torch.arange(swa_base_global, step_offset, device=q.device) % max_win
        all_ks = torch.cat([st.swa_k_ring.index_select(1, prior_idx), ks], dim=1)
        all_vs = torch.cat([st.swa_v_ring.index_select(1, prior_idx), vs], dim=1)
        all_valid = torch.cat([st.swa_valid_ring.index_select(1, prior_idx), valid_cur], dim=1)
    else:
        all_ks = ks
        all_vs = vs
        all_valid = valid_cur

    if boltzmann_gate is not None:
        gate_log_kappa = upcast(boltzmann_gate.log_kappa).to(acc_dt)
        gate_tau = upcast(boltzmann_gate.tau).to(acc_dt)
    else:
        gate_log_kappa = q.new_full((1,), math.log(2.0), dtype=acc_dt)
        gate_tau = q.new_full((1,), 2.2, dtype=acc_dt)

    all_chunks = _chunk_slices_for_span(step_offset, seq_len, chunk_size)
    seg_step = max(1, int(segment_chunks))
    need_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (
            q_hat,
            k_hat,
            v,
            gamma_f,
            bf,
            wf,
            sb,
            slp,
            thr,
            wl,
            rl,
            qs,
            all_ks,
            all_vs,
            gate_log_kappa,
            gate_tau,
            st.page_states,
            go,
        )
    )
    use_ckpt = bool(checkpoint_segments and torch.is_grad_enabled() and need_grad and len(all_chunks) > seg_step)

    p_cur = st.page_states
    sc_cur = st.s_chunk_start
    ring_cur = st.boundary_ring
    seg_outs: list[torch.Tensor] = []
    u_total = q.new_zeros(num_pages, dtype=acc_dt)
    tok_total = q.new_zeros((), dtype=acc_dt)

    for c_idx in range(0, len(all_chunks), seg_step):
        c_group = all_chunks[c_idx : c_idx + seg_step]
        l_start, l_end = c_group[0][0], c_group[-1][1]
        g_first = step_offset + l_start
        g_last = step_offset + l_end
        _, _, grp_far_cut = disjoint_horizon_bounds(g_first + 1, chunk_size, window_chunks)
        s_lo = grp_far_cut - swa_base_global
        s_hi = g_last - swa_base_global
        rel_bounds = [(a - l_start, b - l_start) for a, b in c_group]

        q_seg = q_hat[:, l_start:l_end]
        k_seg = k_hat[:, l_start:l_end]
        v_seg = v[:, l_start:l_end]
        gf_seg = gamma_f[:, l_start:l_end]
        bf_seg = bf[:, l_start:l_end]
        wf_seg = wf[:, l_start:l_end]
        sb_seg = sb[:, l_start:l_end]
        wl_seg = wl[:, l_start:l_end]
        rl_seg = rl[:, l_start:l_end]
        qs_seg = qs[:, l_start:l_end]
        ks_seg = all_ks[:, s_lo:s_hi]
        vs_seg = all_vs[:, s_lo:s_hi]
        val_seg = all_valid[:, s_lo:s_hi]
        wm_seg = None if wm is None else wm[:, l_start:l_end]
        go_seg = None if go is None else go[:, l_start:l_end]

        if use_ckpt:
            def _ckpt_fn(
                qc: torch.Tensor,
                kc: torch.Tensor,
                vc: torch.Tensor,
                gfc: torch.Tensor,
                bfc: torch.Tensor,
                wfc: torch.Tensor,
                sbc: torch.Tensor,
                slpc: torch.Tensor,
                thrc: torch.Tensor,
                wlc: torch.Tensor,
                rlc: torch.Tensor,
                qsc: torch.Tensor,
                ksc: torch.Tensor,
                vsc: torch.Tensor,
                glk: torch.Tensor,
                gtau: torch.Tensor,
                pst: torch.Tensor,
                scs: torch.Tensor,
                *ring_tensors: torch.Tensor,
                _rb: list[tuple[int, int]] = rel_bounds,
                _goff: int = g_first,
                _sbase: int = grp_far_cut,
                _val: torch.Tensor = val_seg,
                _wm: torch.Tensor | None = wm_seg,
                _go: torch.Tensor | None = go_seg,
            ):
                o_g, p_n, sc_n, u_g, tk_g, r_n = _run_chunk_group(
                    _rb,
                    step_offset=_goff,
                    swa_base_global=_sbase,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    top_k_pages=top_k_pages,
                    q_hat=qc,
                    k_hat=kc,
                    v=vc,
                    gamma_f=gfc,
                    bf=bfc,
                    wf=wfc,
                    sb=sbc,
                    slp=slpc,
                    thr=thrc,
                    wl=wlc,
                    rl=rlc,
                    qs=qsc,
                    all_ks=ksc,
                    all_vs=vsc,
                    all_valid_swa=_val,
                    wm=_wm,
                    go=_go,
                    gate_log_kappa=glk,
                    gate_tau=gtau,
                    page_states=pst,
                    s_chunk_start=scs,
                    boundary_ring=ring_tensors,
                )
                return (o_g, p_n, sc_n, u_g, tk_g, *r_n)

            ckpt_out = checkpoint(
                _ckpt_fn,
                q_seg,
                k_seg,
                v_seg,
                gf_seg,
                bf_seg,
                wf_seg,
                sb_seg,
                slp,
                thr,
                wl_seg,
                rl_seg,
                qs_seg,
                ks_seg,
                vs_seg,
                gate_log_kappa,
                gate_tau,
                p_cur,
                sc_cur,
                *ring_cur,
                use_reentrant=False,
            )
            o_grp, p_cur, sc_cur, u_grp, tok_grp = ckpt_out[:5]
            ring_cur = tuple(ckpt_out[5:])
        else:
            o_grp, p_cur, sc_cur, u_grp, tok_grp, ring_cur = _run_chunk_group(
                rel_bounds,
                step_offset=g_first,
                swa_base_global=grp_far_cut,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                top_k_pages=top_k_pages,
                q_hat=q_seg,
                k_hat=k_seg,
                v=v_seg,
                gamma_f=gf_seg,
                bf=bf_seg,
                wf=wf_seg,
                sb=sb_seg,
                slp=slp,
                thr=thr,
                wl=wl_seg,
                rl=rl_seg,
                qs=qs_seg,
                all_ks=ks_seg,
                all_vs=vs_seg,
                all_valid_swa=val_seg,
                wm=wm_seg,
                go=go_seg,
                gate_log_kappa=gate_log_kappa,
                gate_tau=gate_tau,
                page_states=p_cur,
                s_chunk_start=sc_cur,
                boundary_ring=ring_cur,
            )
        seg_outs.append(o_grp)
        u_total = u_total + u_grp
        tok_total = tok_total + tok_grp

    out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)).to(init_dtype)
    final_state: HSSDCacheState | None = None
    if output_final_state:
        end_step = step_offset + seq_len
        keep_from = max(swa_base_global, end_step - max_win)
        rel_from = keep_from - swa_base_global
        ring_slots = torch.arange(keep_from, end_step, device=q.device) % max_win
        new_k_ring = st.swa_k_ring.index_copy(1, ring_slots, all_ks[:, rel_from:])
        new_v_ring = st.swa_v_ring.index_copy(1, ring_slots, all_vs[:, rel_from:])
        new_val_ring = st.swa_valid_ring.index_copy(1, ring_slots, all_valid[:, rel_from:])
        final_state = HSSDCacheState(
            step_count=end_step,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            page_states=p_cur,
            s_chunk_start=sc_cur,
            theta_cum=theta_cum_all[:, -1],
            boundary_ring=ring_cur,
            swa_k_ring=new_k_ring,
            swa_v_ring=new_v_ring,
            swa_valid_ring=new_val_ring,
        )

    if return_aux:
        duty_vec = u_total / tok_total.clamp_min(1.0)
        return out, final_state, duty_vec
    return out, final_state


# ============================================================================ adapter & retrofit

class HSSDAdapter(nn.Module):
    """Per-layer `HSSD v3` module attached to each linear-attention (`GatedDeltaNet`) layer."""

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        cfg: HSSDConfig,
        layer_id: int,
    ):
        super().__init__()
        cfg.validate()
        require(head_k_dim % 2 == 0, f"layer {layer_id}: HSSD requires even head_k_dim, got {head_k_dim}")
        self.cfg = cfg
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.num_v_heads = int(num_v_heads)
        self.head_k_dim = int(head_k_dim)
        self.head_v_dim = int(head_v_dim)
        self.half_k_dim = self.head_k_dim // 2
        self.num_pages = int(cfg.num_pages)
        eff_rank = max(1, min(int(cfg.rank), self.hidden_size))
        self.eff_rank = eff_rank

        # Channel-wise Erase (over-relaxed (0, 2)), Write ([0, 1]), and paired Decay rates
        self.down = nn.Linear(self.hidden_size, eff_rank, bias=False)
        self.erase_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.write_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.decay_up = nn.Linear(eff_rank, self.num_v_heads * self.half_k_dim, bias=False)

        # Theorem 1: 2D phase rotation projector (zero-init => Theta_t == 0 at step 0)
        self.theta_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.half_k_dim, bias=False)

        # Fixes H2, B3, B4: Smooth chunk-start root-surprise `s = g^2`
        self.surprise_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        self.log_surprise_slope = nn.Parameter(torch.full((self.num_v_heads,), math.log(cfg.surprise_slope_init)))
        self.surprise_thresh = nn.Parameter(torch.full((self.num_v_heads,), cfg.surprise_thresh_init))

        # Fix B2: Tiered M=8 multi-page write/read routers & log-spaced decay priors
        self.write_router = nn.Linear(self.hidden_size, self.num_pages, bias=True)
        self.read_router = nn.Linear(self.num_v_heads * self.head_k_dim, self.num_pages, bias=True)
        self.log_page_gamma = nn.Parameter(torch.zeros(self.num_pages))

        # Fixes B1 & B5: Local sliding-window Q/K/V adapters + PeakBoltzmannGate
        self.swa_q_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_k_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_v_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.swa_q_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.swa_k_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.boltzmann_gate = PeakBoltzmannGate(
            self.num_v_heads, kappa_init=cfg.kappa_init, tau_init=cfg.tau_init
        )

        mu_max_vals = cfg.resolved_mu_max()
        self.register_buffer("mu_max", torch.tensor(mu_max_vals, dtype=torch.float32), persistent=False)
        if self.num_pages == 1:
            gamma_prior = torch.tensor([cfg.decay_gamma_min], dtype=torch.float32)
        else:
            gamma_prior = torch.exp(
                torch.linspace(math.log(cfg.decay_gamma_min), math.log(cfg.decay_gamma_max), self.num_pages)
            )
        self.register_buffer("page_gamma_prior", gamma_prior, persistent=False)

        self._reset_parameters()
        self._pending_aux: list[torch.Tensor] = []
        self.last_page_duty: torch.Tensor | None = None
        self.last_stats: dict[str, float] = {}

    def _reset_parameters(self) -> None:
        std_h = 1.0 / math.sqrt(self.hidden_size)
        std_q = 1.0 / math.sqrt(self.num_v_heads * self.head_k_dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=std_h)
        nn.init.zeros_(self.erase_up.weight)
        nn.init.zeros_(self.write_up.weight)
        nn.init.zeros_(self.decay_up.weight)
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.surprise_proj.weight)
        nn.init.zeros_(self.surprise_proj.bias)
        nn.init.normal_(self.write_router.weight, mean=0.0, std=std_h)
        nn.init.zeros_(self.write_router.bias)
        nn.init.normal_(self.read_router.weight, mean=0.0, std=std_q)
        nn.init.zeros_(self.read_router.bias)
        nn.init.zeros_(self.swa_q_up.weight)
        nn.init.zeros_(self.swa_k_up.weight)
        nn.init.zeros_(self.swa_v_up.weight)

    def clear_aux(self) -> None:
        self._pending_aux.clear()

    def project_inputs(
        self,
        x_in: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g_base: torch.Tensor,
        beta_base: torch.Tensor,
        *,
        use_qk_l2norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute all HSSD gate, router, phase, and local-SWA projections."""
        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        half_dk = self.half_k_dim

        q_prep, k_prep, v_prep = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm)
        xf = upcast(x_in)
        h = F.silu(F.linear(xf, upcast(self.down.weight)))

        cap_g = float(self.cfg.gate_logit_cap)
        cap_d = float(self.cfg.decay_log_cap)
        d_erase = (cap_g * torch.tanh(F.linear(h, upcast(self.erase_up.weight)))).view(bsz, seq_len, hv, dk)
        d_write = (cap_g * torch.tanh(F.linear(h, upcast(self.write_up.weight)))).view(bsz, seq_len, hv, dv)
        d_decay_pair = (cap_d * torch.tanh(F.linear(h, upcast(self.decay_up.weight)))).view(bsz, seq_len, hv, half_dk)

        beta_f = upcast(beta_base).clamp(0.0, 1.0).unsqueeze(-1)
        b_gate = 2.0 * logistic_shift(0.5 * beta_f, d_erase)
        w_gate = logistic_shift(beta_f, d_write)

        base_rate = 1.0 + 0.1 * (-upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
        chan_mod = (base_rate * d_decay_pair.exp()).repeat_interleave(2, dim=-1)    # [B, T, H_v, d_k]
        page_scale = upcast(self.page_gamma_prior) * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
        gamma = page_scale.view(1, 1, self.num_pages, 1, 1) * chan_mod.unsqueeze(2)

        theta = math.pi * torch.tanh(F.linear(xf, upcast(self.theta_proj.weight))).view(bsz, seq_len, hv, half_dk)
        surprise_bias = F.linear(xf, upcast(self.surprise_proj.weight), upcast(self.surprise_proj.bias))
        write_logits = F.linear(xf, upcast(self.write_router.weight), upcast(self.write_router.bias))
        q_flat = q_prep.reshape(bsz, seq_len, hv * dk)
        read_logits = F.linear(q_flat, upcast(self.read_router.weight), upcast(self.read_router.bias))

        root_dk = math.sqrt(dk)
        dq_swa = F.linear(h, upcast(self.swa_q_up.weight)).view(bsz, seq_len, hv, dk)
        dk_swa = F.linear(h, upcast(self.swa_k_up.weight)).view(bsz, seq_len, hv, dk)
        dv_swa = F.linear(h, upcast(self.swa_v_up.weight)).view(bsz, seq_len, hv, dv)
        q_swa = _rms_norm_scaled(root_dk * q_prep + dq_swa, self.swa_q_norm)
        k_swa = _rms_norm_scaled(root_dk * k_prep + dk_swa, self.swa_k_norm)
        v_swa = v_prep + dv_swa

        return {
            "query": q_prep.to(query.dtype),
            "key": k_prep.to(key.dtype),
            "value": v_prep.to(value.dtype),
            "gamma": gamma,
            "b_gate": b_gate,
            "w_gate": w_gate,
            "theta": theta,
            "surprise_bias": surprise_bias,
            "write_logits": write_logits,
            "read_logits": read_logits,
            "q_swa": q_swa,
            "k_swa": k_swa,
            "v_swa": v_swa,
        }

    def record_duty(self, duty_vec: torch.Tensor) -> None:
        if _in_backward():
            return
        bud_loss = compute_write_budget_loss(
            duty_vec,
            self.mu_max,
            mu_min_ratio=self.cfg.mu_min_ratio,
            budget_weight=self.cfg.budget_weight,
        )
        self._pending_aux.append(bud_loss)
        with torch.no_grad():
            det = duty_vec.detach()
            self.last_page_duty = det
            self.last_stats["budget_aux_loss"] = float(bud_loss.detach())
            self.last_stats["vault_duty"] = float(det[: min(2, det.numel())].mean())
            self.last_stats["scratch_duty"] = float(det[min(4, det.numel() - 1) :].mean())


def _extract_hssd_gdr_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, HSSDCacheState | torch.Tensor | None, bool, bool, float | None, int]:
    g = args[0] if len(args) >= 1 else kwargs.get("g")
    beta = args[1] if len(args) >= 2 else kwargs.get("beta")
    require(
        isinstance(g, torch.Tensor) and isinstance(beta, torch.Tensor),
        "HSSD wrapper expected Tensor `g` and `beta`",
        g=g,
        beta=beta,
    )
    chunk_size = int(kwargs.get("chunk_size", default_chunk))
    initial_state = kwargs.get("initial_state")
    for extra in args[2:]:
        if isinstance(extra, int):
            chunk_size = extra
        elif isinstance(extra, (HSSDCacheState, torch.Tensor)) and initial_state is None:
            initial_state = extra
    output_final = bool(kwargs.get("output_final_state", False))
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    scale = kwargs.get("scale")
    return g, beta, initial_state, output_final, use_l2, scale, chunk_size


def _active_mor_write_mask(text_model: nn.Module, bsz: int, seq_len: int) -> torch.Tensor | None:
    """Inspect `LoopController` (if attached via `apply_loop`) for Pass `r >= 2` MoR `keep` mask."""
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        return None
    ctl = getattr(layers[0], "_ctl", None)
    if ctl is None:
        return None
    cur = getattr(ctl, "_cur", None)
    if cur is None:
        return None
    ctx, r = cur
    spec = getattr(ctl, "spec", None)
    if r < 2 or spec is None or not (bool(getattr(spec, "mor", False)) and bool(getattr(spec, "write_mask", False))):
        return None
    keep = ctx.keep.get(r)
    if isinstance(keep, torch.Tensor) and keep.shape == (bsz, seq_len):
        return keep
    return None


class HSSDHandle:
    """Manages `HSSDAdapter` modules, forward hooks, aux write-budget losses, and clean removal."""

    def __init__(
        self,
        model: nn.Module,
        cfg: HSSDConfig,
        adapters: dict[int, HSSDAdapter],
        hooks: list[Any],
        restorers: list[Callable[[], None]],
    ):
        self.model = model
        self.cfg = cfg
        self.adapters = adapters
        self._hooks = hooks
        self._restorers = restorers
        self._removed = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters() if p.requires_grad]

    def parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters()]

    def clear_aux(self) -> None:
        for ad in self.adapters.values():
            ad.clear_aux()

    def aux_loss(self, *, clear: bool = True) -> torch.Tensor:
        """Aggregate two-sided cumulative write-mass budget loss across all HSSD layers & passes."""
        terms: list[torch.Tensor] = []
        for ad in self.adapters.values():
            if ad._pending_aux:
                terms.extend(ad._pending_aux)
                if clear:
                    ad.clear_aux()
        if not terms:
            if self.adapters:
                ref = next(iter(self.adapters.values())).log_surprise_slope
                return ref.new_zeros(())
            return torch.zeros(())
        return torch.stack(terms).mean()

    def page_duties(self) -> dict[int, torch.Tensor]:
        return {
            lid: ad.last_page_duty.clone()
            for lid, ad in self.adapters.items()
            if ad.last_page_duty is not None
        }

    def stats(self) -> dict[str, float]:
        if not self.adapters:
            return {}
        out: dict[str, float] = {}
        for key in ("budget_aux_loss", "vault_duty", "scratch_duty"):
            vals = [ad.last_stats[key] for ad in self.adapters.values() if key in ad.last_stats]
            if vals:
                out[f"hssd/{key}"] = sum(vals) / len(vals)
        return out

    def uninstall(self) -> None:
        if self._removed:
            return
        self._removed = True
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for fn in reversed(self._restorers):
            fn()
        self._restorers.clear()
        if getattr(self.model, "_hssd_handle", None) is self:
            delattr(self.model, "_hssd_handle")
        log.info("HSSD v3 uninstalled from %d linear-attention layers", len(self.adapters))

    def remove(self) -> None:
        self.uninstall()


HSSDManager = HSSDHandle


def _make_hssd_rule_wrapper(
    adapter: HSSDAdapter,
    ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None]],
    *,
    recurrent_only: bool,
) -> Callable[..., Any]:
    def wrapped(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any):
        require(bool(ctx_stack), f"HSSD layer {adapter.layer_id}: missing captured input hidden_states")
        x_in, wmask = ctx_stack[-1]
        bsz, seq_len, _ = x_in.shape
        g, beta, init_st, out_final, use_l2, scale, _csize = _extract_hssd_gdr_args(
            args, kwargs, adapter.cfg.chunk_size
        )

        head_first = bool(kwargs.get("head_first", False)) or (
            seq_len != adapter.num_v_heads and value.shape[1] == adapter.num_v_heads and value.shape[2] == seq_len
        )
        if head_first:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            g, beta = g.transpose(1, 2), beta.transpose(1, 2)
        require(
            value.shape[:2] == (bsz, seq_len),
            f"HSSD layer {adapter.layer_id}: value shape {tuple(value.shape)} disagrees with input {(bsz, seq_len)}",
        )

        proj = adapter.project_inputs(x_in, query, key, value, g, beta, use_qk_l2norm=use_l2)
        c_cfg = adapter.cfg
        if seq_len == 1:
            st0 = _coerce_init_state(
                init_st,
                bsz=bsz,
                num_pages=c_cfg.num_pages,
                hv=adapter.num_v_heads,
                dk=adapter.head_k_dim,
                dv=adapter.head_v_dim,
                chunk_size=c_cfg.chunk_size,
                window_chunks=c_cfg.window_chunks,
                device=query.device,
                dtype=torch.float64 if query.dtype == torch.float64 else torch.float32,
            )
            out, st_next, u_sum, cnt_sum = recurrent_hssd_step(
                proj["query"],
                proj["key"],
                proj["value"],
                proj["gamma"],
                proj["b_gate"],
                proj["w_gate"],
                proj["theta"],
                proj["surprise_bias"],
                adapter.log_surprise_slope.exp(),
                adapter.surprise_thresh,
                proj["write_logits"],
                proj["read_logits"],
                proj["q_swa"],
                proj["k_swa"],
                proj["v_swa"],
                st0,
                boltzmann_gate=adapter.boltzmann_gate,
                top_k_pages=c_cfg.k_pages,
                write_mask=wmask,
                use_qk_l2norm_in_kernel=False,
                scale=scale,
                return_duty_sum=True,
            )
            final_state = st_next if out_final else None
            duty_vec = u_sum / cnt_sum.clamp_min(1.0)
        elif recurrent_only:
            out, final_state, duty_vec = recurrent_hssd(
                proj["query"],
                proj["key"],
                proj["value"],
                proj["gamma"],
                proj["b_gate"],
                proj["w_gate"],
                proj["theta"],
                proj["surprise_bias"],
                adapter.log_surprise_slope.exp(),
                adapter.surprise_thresh,
                proj["write_logits"],
                proj["read_logits"],
                proj["q_swa"],
                proj["k_swa"],
                proj["v_swa"],
                boltzmann_gate=adapter.boltzmann_gate,
                chunk_size=c_cfg.chunk_size,
                window_chunks=c_cfg.window_chunks,
                num_pages=c_cfg.num_pages,
                top_k_pages=c_cfg.k_pages,
                initial_state=init_st,
                output_final_state=out_final,
                write_mask=wmask,
                use_qk_l2norm_in_kernel=False,
                scale=scale,
                return_aux=True,
            )
        else:
            out, final_state, duty_vec = chunk_hssd(
                proj["query"],
                proj["key"],
                proj["value"],
                proj["gamma"],
                proj["b_gate"],
                proj["w_gate"],
                proj["theta"],
                proj["surprise_bias"],
                adapter.log_surprise_slope.exp(),
                adapter.surprise_thresh,
                proj["write_logits"],
                proj["read_logits"],
                proj["q_swa"],
                proj["k_swa"],
                proj["v_swa"],
                boltzmann_gate=adapter.boltzmann_gate,
                chunk_size=c_cfg.chunk_size,
                window_chunks=c_cfg.window_chunks,
                num_pages=c_cfg.num_pages,
                top_k_pages=c_cfg.k_pages,
                segment_chunks=c_cfg.segment_chunks,
                checkpoint_segments=c_cfg.checkpoint_segments,
                initial_state=init_st,
                output_final_state=out_final,
                write_mask=wmask,
                use_qk_l2norm_in_kernel=False,
                scale=scale,
                return_aux=True,
            )

        adapter.record_duty(duty_vec)
        if head_first:
            out = out.transpose(1, 2)
        return out, final_state

    return wrapped


def _resolve_physical_layers(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, list[str]]:
    """Support installing HSSD either before `apply_loop` (standard) or on an already looped model."""
    tm = find_text_model(model)
    layers = getattr(tm, "layers", None)
    require(isinstance(layers, nn.ModuleList) and len(layers) > 0, "text model has no `.layers`")
    ctl = getattr(layers[0], "_ctl", None)
    if ctl is not None and hasattr(ctl, "_orig") and "layers" in ctl._orig:
        phys_layers = ctl._orig["layers"]
        cfg = getattr(tm, "config", None)
        types = [layer_type_of(layer, i, cfg) for i, layer in enumerate(phys_layers)]
        return tm, phys_layers, types
    parts = inspect_model(model)
    return parts.text_model, parts.layers, parts.layer_types


def apply_hssd(model: nn.Module, cfg: HSSDConfig | dict[str, Any] | None = None) -> HSSDHandle:
    """Retrofit Qwen3.6 / Toy linear-attention layers in-place to `HSSD v3`.

    Works seamlessly with `ToyForCausalLM` (`ToyHybridMoEForCausalLM`), HF Qwen3.6 GDN layers,
    `apply_loop` per-pass virtual cache slots, and MoR early-exit write masks.
    """
    if getattr(model, "_hssd_handle", None) is not None:
        raise RuntimeError("HSSD v3 is already installed on this model; call remove_hssd(model) first")
    if not isinstance(cfg, HSSDConfig):
        cfg = HSSDConfig.from_dict(cfg)
    cfg.validate()

    text_model, phys_layers, layer_types = _resolve_physical_layers(model)
    lin_indices = [i for i, lt in enumerate(layer_types) if lt == LINEAR]
    target_layers = lin_indices if cfg.layers is None else [int(i) for i in cfg.layers]
    bad = [i for i in target_layers if i not in lin_indices]
    if bad:
        raise ValueError(f"hssd.layers {bad} are not linear_attention layers (linear: {lin_indices})")

    ref = text_model.embed_tokens.weight
    param_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
    adapters: dict[int, HSSDAdapter] = {}
    hooks: list[Any] = []
    restorers: list[Callable[[], None]] = []
    patched_modules: set[str] = set()
    active_attn_stack: list[tuple[HSSDAdapter, list[tuple[torch.Tensor, torch.Tensor | None]]]] = []

    def _clear_model_aux(_mod: nn.Module, _args: tuple[Any, ...]) -> None:
        if not _in_backward():
            for ad in adapters.values():
                ad.clear_aux()

    hooks.append(text_model.register_forward_pre_hook(_clear_model_aux))

    for i in target_layers:
        attn = attention_module(phys_layers[i], LINEAR)
        hs, hv, dk, dv = _infer_gdn_dims(attn, i)
        adapter = HSSDAdapter(hs, hv, dk, dv, cfg, i).to(device=ref.device, dtype=param_dtype)
        attn.hssd_adapter = adapter
        adapters[i] = adapter

        ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None]] = []

        def _make_pre(
            ad_ref: HSSDAdapter,
            st_ref: list[tuple[torch.Tensor, torch.Tensor | None]],
        ):
            def pre_hook(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x_in = (
                    args[0] if args else (kwargs.get("hidden_states") if "hidden_states" in kwargs else kwargs.get("x"))
                )
                require(isinstance(x_in, torch.Tensor), "HSSD pre_hook could not locate input hidden_states")
                amask = kwargs.get("attention_mask", args[2] if len(args) >= 3 else None)
                wmask = _active_mor_write_mask(text_model, x_in.shape[0], x_in.shape[1])
                if (
                    isinstance(amask, torch.Tensor)
                    and amask.ndim == 2
                    and amask.shape[0] == x_in.shape[0]
                    and amask.shape[1] >= x_in.shape[1]
                ):
                    am_slice = amask[:, -x_in.shape[1] :].to(dtype=x_in.dtype)
                    if x_in.shape[1] > 1:
                        x_in = x_in * am_slice.unsqueeze(-1)
                    wmask = am_slice if wmask is None else (wmask.to(am_slice.dtype) * am_slice)
                if wmask is not None:
                    x_in = x_in * wmask.to(dtype=x_in.dtype).unsqueeze(-1)
                st_ref.append((x_in, wmask))
                active_attn_stack.append((ad_ref, st_ref))

            return pre_hook

        def _make_post(st_ref: list[tuple[torch.Tensor, torch.Tensor | None]]):
            def post_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
                if st_ref:
                    st_ref.pop()
                if active_attn_stack:
                    active_attn_stack.pop()

            return post_hook

        hooks.append(attn.register_forward_pre_hook(_make_pre(adapter, ctx_stack), with_kwargs=True))
        hooks.append(attn.register_forward_hook(_make_post(ctx_stack)))

        def _del_adapter(a: nn.Module = attn, ad: HSSDAdapter = adapter) -> None:
            if getattr(a, "hssd_adapter", None) is ad:
                delattr(a, "hssd_adapter")

        restorers.append(_del_adapter)

        wrapped_any = False
        for attr, rec_only in (("chunk_gated_delta_rule", False), ("fused_recurrent_gated_delta_rule", True)):
            orig_fn = getattr(attn, attr, None)
            if callable(orig_fn):
                wrapped_any = True
                had_instance_attr = attr in attn.__dict__
                setattr(attn, attr, _make_hssd_rule_wrapper(adapter, ctx_stack, recurrent_only=rec_only))

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

        if not wrapped_any:
            mod_name = type(attn).__module__
            host_mod = sys.modules.get(mod_name)
            if host_mod is not None and mod_name not in patched_modules:
                for fn_name, rec_only in (
                    ("chunk_gated_delta_rule", False),
                    ("torch_chunk_gated_delta_rule", False),
                    ("fused_recurrent_gated_delta_rule", True),
                    ("torch_recurrent_gated_delta_rule", True),
                ):
                    orig_mfn = getattr(host_mod, fn_name, None)
                    if callable(orig_mfn):
                        patched_modules.add(mod_name)

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
                            cur_ad, cur_st = active_attn_stack[-1]
                            return _make_hssd_rule_wrapper(cur_ad, cur_st, recurrent_only=_rec)(
                                query, key, value, *args, **kwargs
                            )

                        setattr(host_mod, fn_name, _mod_dispatch)
                        restorers.append(lambda hm=host_mod, n=fn_name, f=orig_mfn: setattr(hm, n, f))

    handle = HSSDHandle(model, cfg, adapters, hooks, restorers)
    object.__setattr__(model, "_hssd_handle", handle)
    n_params = sum(p.numel() for p in handle.parameters())
    log.info(
        "HSSD v3 installed on %d linear-attention layers (rank=%d, chunk=%d, window=%d, pages=%d, params=%.2fM)",
        len(adapters),
        cfg.rank,
        cfg.chunk_size,
        cfg.window_chunks,
        cfg.num_pages,
        n_params / 1e6,
    )
    return handle


install_hssd = apply_hssd


def remove_hssd(model_or_handle: nn.Module | HSSDHandle) -> None:
    """Cleanly uninstall `HSSD v3` from `model` (or via its `HSSDHandle`)."""
    if isinstance(model_or_handle, HSSDHandle):
        model_or_handle.uninstall()
        return
    handle = getattr(model_or_handle, "_hssd_handle", None)
    if isinstance(handle, HSSDHandle):
        handle.uninstall()
        return
    raise ValueError(f"No active HSSDHandle found on {type(model_or_handle).__name__}")


__all__ = [
    "DEFAULT_MU_MAX",
    "HSSDAdapter",
    "HSSDCacheState",
    "HSSDConfig",
    "HSSDHandle",
    "HSSDManager",
    "PeakBoltzmannGate",
    "apply_2d_rotation",
    "apply_hssd",
    "chunk_hssd",
    "compute_write_budget_loss",
    "disjoint_horizon_bounds",
    "has_fla",
    "install_hssd",
    "recurrent_hssd",
    "recurrent_hssd_step",
    "remove_hssd",
]
