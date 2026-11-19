"""Horizon-Split Surprise-Gated Delta Memory (HSSD v3.1) retrofit for linear-attention layers.

Implements Track B (`HSSD v3.1`) of `Qwen3.6-35B-A6B-LoopHSSD` (`docs/ARCHITECTURE_SPEC.md` §5,
`docs/RESEARCH_ESSAY.md` §4 Theorems 1 & 2), incorporating all external audit hardenings:

  * **Fix B1 (Strictly Disjoint Horizon Partition):** For 1-based token `t` in 0-based chunk
    `n = (t - 1) // C`, sets `k_far = max(0, n - W)` (`W = window_chunks`, default 8) and
    `far_cutoff = k_far * C`. Far-horizon memory reads exclusively from `S_boundary[k_far]`
    covering `I_far = [1, far_cutoff]`, while exact causal local SWA covers
    `I_loc = [far_cutoff + 1, t]` (`|I_loc| in [W*C + 1, (W + 1)*C]` for `t > W*C`, `[1, t]`
    for `t <= W*C`), guaranteeing zero double-counting and zero gap.
  * **Fix B5 (`PeakBoltzmannGate`):** Hybrid log-partition + Gumbel extreme-value branch gate
    `stat = 0.5 * (LSE(z) - log(N_t)) + 0.5 * (max(z) - sqrt(2 * log(max(N_t, 2))))`,
    `p_hit = sigmoid(kappa_h * (stat - tau_h))`, blending `o = p_hit * o_loc + (1 - p_hit) * o_far`.
  * **Theorem 1 (Low-Rank 2D Phase Rotation & `fp64` Phase Accumulator):** Pure linear low-rank
    phase step `theta_t = pi * tanh(W_up(W_down(x_t)))` (`theta_up` zero-init), accumulated in
    `torch.float64` (`Theta_t = cumsum(theta_t) mod 2*pi`, `< 2e-6 rad` drift at `1M` steps),
    counter-rotating `(q, k)` via `R(-Theta_t)` with over-relaxed erase `b_t in (0, 2)^{d_k}`
    and linear-`beta` write scaling `sqrt(w_t)`.
  * **Fixes H2, B3, B4 & 4.2 (Multi-Page Routed Chunk-Start Root-Surprise `s = g^2`):** Evaluates
    relative key-value prediction error against the top-`k` write-routed pages of `boundary_ring[-1]`
    (`S_{nC}`) in parallel across all tokens of chunk `n` before the WY triangular solve, with
    smooth gradients at `g -> 0` and bit-exact `S_t == S_{t-1}` at `g = 0`.
  * **Fixes B2, B6, 3A-3E, 4.1 (Factored Decay, Actual Sparse STE Write Budget, Compact Cache):**
    Factored `(gamma_page_scale[M], gamma_chan_pair[B, T, H_v, d_k // 2])` decay (`16x` smaller),
    actual sparse Top-`k` write-mass duty with Straight-Through Estimation (`pi_ste`), compact
    `HSSDCacheState` (`__torch_function__`, `HSSDHybridCache`, `state_dtype="bfloat16"` option),
    packed-document isolation (`cu_seqlens` / `seq_idx`), and 2D padding logical-clock invariance.

Clean-room Apache-2.0 / MIT PyTorch implementation.
"""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import logging
import math
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..debug import require
from ._dtype import upcast
from .gdn2 import _infer_gdn_dims, l2norm, logistic_shift
from .introspect import LINEAR, attention_module, find_text_model, inspect_model, layer_type_of

try:
    from transformers.cache_utils import Cache as _HFCacheBase  # type: ignore[import-not-found]
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

log = logging.getLogger("qloop")

DEFAULT_MU_MAX: tuple[float, ...] = (0.0025, 0.0025, 0.025, 0.025, 0.25, 0.25, 0.25, 0.25)

_VALID_STATE_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}
_VALID_SURPRISE_MODES = ("routed", "page0", "chunk_start", "segment_start", "none")
_VALID_WRITE_MODES = ("full", "split")


def has_fla() -> bool:
    """Return True if optional MIT `flash-linear-attention` (`fla`) package is importable."""
    return importlib.util.find_spec("fla") is not None


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    fn = getattr(compiler, "is_compiling", None) if compiler is not None else None
    return bool(fn()) if callable(fn) else False


def _in_outer_checkpoint() -> bool:
    """Return True when executing inside an outer `torch.utils.checkpoint(..., use_reentrant=False)` region.

    Under `VirtualLayer` (`use_reentrant=False`), PyTorch installs `_checkpoint_hook` during outer
    forward (already discarding cross-layer intermediates) and `_recomputation_hook` during outer
    backward recomputation. Skipping nested segment checkpointing consistently in both phases avoids
    a redundant 3rd segment forward pass while keeping forward/recompute saved-tensor metadata identical.
    """
    if _is_compiling():
        return False
    autograd_c = getattr(torch._C, "_autograd", None)
    fn = getattr(autograd_c, "_top_saved_tensors_default_hooks", None) if autograd_c is not None else None
    if not callable(fn):
        return False
    hooks = fn(False)
    if not hooks:
        return False
    pack_hook = hooks[0]
    if pack_hook is None:
        return False
    mod = getattr(pack_hook, "__module__", "") or ""
    qual = getattr(pack_hook, "__qualname__", "") or ""
    return "checkpoint" in mod or "_checkpoint_hook" in qual or "_recomputation_hook" in qual


def _compact(t: torch.Tensor) -> torch.Tensor:
    """Ensure `t` is contiguous and owns compact backing storage (`nbytes == numel * element_size`, offset 0)."""
    if not t.is_contiguous():
        return t.contiguous()
    if t.numel() == 0:
        if int(t.untyped_storage().nbytes()) != 0 or int(t.storage_offset()) != 0:
            return t.clone()
        return t
    if int(t.untyped_storage().nbytes()) != int(t.numel() * t.element_size()) or int(t.storage_offset()) != 0:
        return t.clone().contiguous()
    return t


def _is_hssd_state_duck(obj: Any) -> bool:
    return (
        not isinstance(obj, torch.Tensor)
        and hasattr(obj, "page_states")
        and hasattr(obj, "boundary_ring")
        and callable(getattr(obj, "_iter_tensors", None))
    )


def _ensure_dynamo_mark_static_address_compat() -> None:
    """Make `torch._dynamo.mark_static_address` accept `HSSDCacheState` / `_HSSDCacheState` (`transformers>=5.20`)."""
    dynamo = getattr(torch, "_dynamo", None)
    orig_fn = getattr(dynamo, "mark_static_address", None) if dynamo is not None else None
    if not callable(orig_fn) or getattr(orig_fn, "_qloop_hssd_aware", False):
        return

    @functools.wraps(orig_fn)
    def _hssd_mark_static_address(t: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_hssd_state_duck(t):
            seen: set[int] = set()
            for sub_t in t._iter_tensors():
                if isinstance(sub_t, torch.Tensor) and id(sub_t) not in seen:
                    seen.add(id(sub_t))
                    orig_fn(sub_t, *args, **kwargs)
            return None
        return orig_fn(t, *args, **kwargs)

    forbid = getattr(dynamo, "forbid_in_graph", None)
    wrapped = forbid(_hssd_mark_static_address) if callable(forbid) else _hssd_mark_static_address
    wrapped._qloop_hssd_aware = True  # type: ignore[attr-defined]
    wrapped._qloop_orig_mark_static_address = orig_fn  # type: ignore[attr-defined]
    dynamo.mark_static_address = wrapped  # type: ignore[attr-defined]


_ensure_dynamo_mark_static_address_compat()


def _resolve_state_dtype(state_dtype: str | torch.dtype | None, acc_dt: torch.dtype) -> torch.dtype:
    if state_dtype is None:
        return acc_dt
    if isinstance(state_dtype, torch.dtype):
        return state_dtype
    key = str(state_dtype).replace("torch.", "").lower()
    if key not in _VALID_STATE_DTYPES:
        raise ValueError(f"unknown hssd.state_dtype {state_dtype!r}; valid: {sorted(_VALID_STATE_DTYPES)}")
    resolved = _VALID_STATE_DTYPES[key]
    return acc_dt if resolved is None else resolved


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
    segment_chunks: int = 4
    checkpoint_segments: bool = True
    checkpoint_segments_min_len: int = 0
    outer_checkpoint_max_tokens: int = 2048
    gate_logit_cap: float = 8.0
    decay_log_cap: float = 2.0
    kappa_init: float = 2.0
    tau_init: float = 1.0
    surprise_slope_init: float = 3.0
    surprise_thresh_init: float = 0.85
    surprise_bias_init: float = 1.5
    budget_weight: float = 1.0
    mu_max: tuple[float, ...] = DEFAULT_MU_MAX
    mu_min_ratio: float = 0.2
    horizon_tokens: int | None = None
    decay_gamma_min: float = 1e-4
    decay_gamma_max: float = 5e-2
    use_stock_decay_init: bool = True
    theta_low_rank: bool = True
    state_dtype: str = "auto"
    paired_b_gate: bool = False
    use_swa: bool = True
    use_phase: bool = True
    use_surprise: bool = True
    surprise_mode: str = "routed"
    write_mode: str = "full"
    swa_recency_init: float = 0.0
    blend_init: float = 1.0
    blend_warmup_steps: int = 0
    compact_active_pages: bool = True
    compact_decode_cache: bool = True
    compile_chunks: bool = False

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

    def effective_mu_max(self) -> tuple[float, ...]:
        """Horizon-scaled upper write-duty budget (`H7`).

        Scales vault pages (`0, 1`) inversely as `min(base, 160.0 / horizon_tokens)` when
        `horizon_tokens is not None and horizon_tokens > 65536`.
        """
        base = list(self.resolved_mu_max())
        if self.horizon_tokens is not None and int(self.horizon_tokens) > 65536 and self.num_pages >= 2:
            vault_cap = 160.0 / float(self.horizon_tokens)
            n_vault = min(2, self.num_pages - 1)
            for idx in range(n_vault):
                base[idx] = min(base[idx], vault_cap)
        return tuple(base)

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
        if int(self.checkpoint_segments_min_len) < 0:
            raise ValueError(
                f"hssd.checkpoint_segments_min_len must be >= 0, got {self.checkpoint_segments_min_len}"
            )
        if int(self.outer_checkpoint_max_tokens) < 0:
            raise ValueError(
                f"hssd.outer_checkpoint_max_tokens must be >= 0, got {self.outer_checkpoint_max_tokens}"
            )
        if self.gate_logit_cap <= 0 or self.decay_log_cap <= 0:
            raise ValueError("hssd.gate_logit_cap and hssd.decay_log_cap must be > 0")
        if self.kappa_init <= 0 or self.surprise_slope_init <= 0:
            raise ValueError("hssd.kappa_init and hssd.surprise_slope_init must be > 0")
        if not 0.0 <= self.mu_min_ratio <= 1.0:
            raise ValueError(f"hssd.mu_min_ratio must be in [0, 1], got {self.mu_min_ratio}")
        if self.horizon_tokens is not None and int(self.horizon_tokens) < 1:
            raise ValueError(f"hssd.horizon_tokens must be >= 1 when set, got {self.horizon_tokens}")
        if self.decay_gamma_min <= 0 or self.decay_gamma_max < self.decay_gamma_min:
            raise ValueError("hssd decay_gamma bounds must satisfy 0 < min <= max")
        sdt_key = str(self.state_dtype).replace("torch.", "").lower()
        if sdt_key not in _VALID_STATE_DTYPES:
            raise ValueError(f"hssd.state_dtype must be one of {sorted(_VALID_STATE_DTYPES)}, got {self.state_dtype!r}")
        if self.surprise_mode not in _VALID_SURPRISE_MODES:
            raise ValueError(
                f"hssd.surprise_mode must be one of {_VALID_SURPRISE_MODES}, got {self.surprise_mode!r}"
            )
        if self.write_mode not in _VALID_WRITE_MODES:
            raise ValueError(f"hssd.write_mode must be one of {_VALID_WRITE_MODES}, got {self.write_mode!r}")
        if not 0.0 <= float(self.blend_init) <= 1.0:
            raise ValueError(f"hssd.blend_init must be in [0, 1], got {self.blend_init}")
        if int(self.blend_warmup_steps) < 0:
            raise ValueError(f"hssd.blend_warmup_steps must be >= 0, got {self.blend_warmup_steps}")
        _ = self.effective_mu_max()


# ============================================================================ math primitives

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Singularity-free `sqrt(max(x, 0))` with exact 0 value and bounded derivative near `x -> 0+`."""
    pos = x > 0.0
    x_pos = x if x.dtype == torch.float64 else x.clamp_min(1e-12)
    safe = torch.where(pos, x_pos, torch.ones_like(x))
    return torch.where(pos, torch.sqrt(safe), torch.zeros_like(x))


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """L2 norm with finite 0 derivative at exact zero vectors."""
    return _safe_sqrt((x * x).sum(dim=dim, keepdim=keepdim))


def _rms_norm_scaled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = upcast(x)
    r_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = xf * r_rms
    if x.dtype != torch.float64 and normed.requires_grad:
        s_bwd = (r_rms.clamp(max=10.0) / r_rms).detach()
        normed = normed.detach() + s_bwd * (normed - normed.detach())
    return normed * upcast(weight)


def compute_2d_trig(theta: torch.Tensor, target_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-evaluation modulo-reduced 2D rotation trig pair `(cos_t, sin_t)` (`H9`).

    Reduces `theta` modulo `2*pi` in `float64`; evaluates `cos`/`sin` in `float64` when
    `target_dtype == torch.float64` (preserving `< 1e-12` parity/gradcheck), and in `float32`
    otherwise before casting to `target_dtype`.
    """
    th_mod = torch.remainder(theta.to(dtype=torch.float64), 2.0 * math.pi)
    if target_dtype == torch.float64:
        return torch.cos(th_mod), torch.sin(th_mod)
    th32 = th_mod.to(dtype=torch.float32)
    return torch.cos(th32).to(dtype=target_dtype), torch.sin(th32).to(dtype=target_dtype)


def apply_2d_rotation_trig(x: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
    """Apply precomputed `(cos_t, sin_t)` 2D rotation to `x` (`H9`)."""
    require(x.shape[-1] % 2 == 0, "apply_2d_rotation requires an even last dimension", x=x)
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    c = cos_t if cos_t.dtype == x.dtype else cos_t.to(dtype=x.dtype)
    s = sin_t if sin_t.dtype == x.dtype else sin_t.to(dtype=x.dtype)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack([out0, out1], dim=-1).reshape_as(x)


def apply_2d_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Block-diagonal 2D phase rotation `R(theta)` on last dim `d = 2 * (d // 2)` (Theorem 1 & `H9`)."""
    require(x.shape[-1] % 2 == 0, "apply_2d_rotation requires an even last dimension", x=x)
    cos_t, sin_t = compute_2d_trig(theta, x.dtype)
    return apply_2d_rotation_trig(x, cos_t, sin_t)


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
    """Peak-augmented Boltzmann local-hit branch gate (`Fix B5`).

    Computes:
        stat = 0.5 * (LSE(scores) - log(N_t)) + 0.5 * (max(scores) - sqrt(2 * log(max(N_t, 2))))
        p_hit = sigmoid(kappa_h * (stat - tau_h))
    """

    def __init__(self, num_heads: int = 1, kappa_init: float = 2.0, tau_init: float = 1.0):
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
            log_n = sf.new_full((), math.log(max(n_val, 1.0)))
            gumbel = sf.new_full((), math.sqrt(2.0 * math.log(max(n_val, 2.0))))
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


# ============================================================================ O(1) cache state & hybrid cache

def _replay_chunk_on_bank(
    bank: torch.Tensor,
    replay_buf: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    chunk_idx: int,
    chunk_size: int,
    window_chunks: int,
    *,
    grad_enabled: bool = False,
    sync_free: bool = False,
) -> torch.Tensor:
    """Advance `bank` (`S_{chunk_idx * C}`) by one `chunk_size`-token WY update from `replay_buf` (`H5`)."""
    rep_ke, rep_ve, rep_g, rep_gam, rep_idx = replay_buf
    bsz, num_pages, hv, dk, dv = bank.shape
    c_slot = int(chunk_idx) % (int(window_chunks) + 1)
    s0 = c_slot * int(chunk_size)
    s1 = s0 + int(chunk_size)
    c_len = s1 - s0
    acc_dt = torch.float64 if bank.dtype == torch.float64 else torch.float32
    ctx = contextlib.nullcontext() if grad_enabled else torch.no_grad()

    with ctx:
        ke_base = rep_ke[:, :, s0:s1, :].to(dtype=acc_dt)                     # [B, H_v, C, d_k]
        ve_base = rep_ve[:, :, s0:s1, :].to(dtype=acc_dt)                     # [B, H_v, C, d_v]
        g_sel = rep_g[:, s0:s1].to(dtype=acc_dt)                              # [B, C, k, H_v]
        gam_sel = rep_gam[:, s0:s1].to(dtype=acc_dt)                          # [B, C, k, H_v, d_g]
        if gam_sel.shape[-1] == dk // 2:
            gam_sel = gam_sel.repeat_interleave(2, dim=-1)
        t_idx = rep_idx[:, s0:s1]                                             # [B, C, k]

        idx_g = t_idx.unsqueeze(-1).expand(bsz, c_len, t_idx.shape[-1], hv)
        g_page = torch.zeros((bsz, c_len, num_pages, hv), device=bank.device, dtype=acc_dt).scatter_add(
            2, idx_g, g_sel
        )
        idx_gam = t_idx[:, :, :, None, None].expand(bsz, c_len, t_idx.shape[-1], hv, dk)
        dec_rate = torch.zeros((bsz, c_len, num_pages, hv, dk), device=bank.device, dtype=acc_dt).scatter_add(
            2, idx_gam, (g_sel * g_sel).unsqueeze(-1) * gam_sel
        )

        g_perm = g_page.permute(0, 2, 3, 1).unsqueeze(-1)                     # [B, M, H_v, C, 1]
        k_eff = g_perm * ke_base.unsqueeze(1)                                 # [B, M, H_v, C, d_k]
        v_eff = g_perm * ve_base.unsqueeze(1)                                 # [B, M, H_v, C, d_v]
        log_a = (-dec_rate).permute(0, 2, 3, 1, 4)                            # [B, M, H_v, C, d_k]

        # Flatten (B, M) -> (B * M, 1) before upcasting so 5D (B, M, H_v, d_k, d_v) is never allocated in fp32
        bm = bsz * num_pages
        bank_flat = bank.reshape(bm, 1, hv, dk, dv).to(dtype=acc_dt)
        ke_flat = k_eff.reshape(bm, 1, hv, c_len, dk)
        ve_flat = v_eff.reshape(bm, 1, hv, c_len, dv)
        la_flat = log_a.reshape(bm, 1, hv, c_len, dk)

        if sync_free or _is_compiling():
            nxt_flat = _wy_kda_state_update(
                bank_flat,
                ke_flat,
                ve_flat,
                la_flat,
                active_mask=None,
                compact_active_pages=False,
            )
        else:
            act_mask = (
                torch.zeros((bsz, c_len, num_pages), dtype=torch.bool, device=bank.device)
                .scatter_(2, t_idx, True)
                .transpose(1, 2)
                .reshape(bm, 1, c_len)
            )
            nxt_flat = _wy_kda_state_update(
                bank_flat,
                ke_flat,
                ve_flat,
                la_flat,
                active_mask=act_mask,
                compact_active_pages=True,
            )
        nxt_cast = nxt_flat if nxt_flat.dtype == bank.dtype else nxt_flat.to(dtype=bank.dtype)
        return _compact(nxt_cast.reshape(bsz, num_pages, hv, dk, dv))


class HSSDCacheState:
    """Fixed-footprint `O(1)` incremental decode cache for one `(layer, pass)` HSSD slot (`H5` & `H8`).

    Supports two mathematically equivalent storage representations:
      1. **Compact Lagged-Replay Mode (`compact_decode_cache=True`, default):** Stores `page_states`,
         `s_far_bank` (`S_{max(0, n - W)*C}`), `s_chunk_start_bank` (`S_{nC}`), and a circular
         `(W + 1) * C` replay buffer of active post-rotation write inputs (`~2.5x-3.5x` smaller per
         layer than 9 boundary banks), synthesizing `.boundary_ring` on demand when inspected.
      2. **Explicit Ring Mode (`compact_decode_cache=False` or `distinct_ring_buffers=True`):**
         Stores all `W + 1` `[B, M, H_v, d_k, d_v]` boundary snapshots explicitly.

    Tracks `_uniform_step: int | None` in Python to eliminate GPU->CPU synchronizations in decode (`H8`).
    """

    def __init__(
        self,
        step_count: int | torch.Tensor,
        chunk_size: int,
        window_chunks: int,
        page_states: torch.Tensor,
        theta_cum: torch.Tensor,
        boundary_ring: tuple[torch.Tensor, ...] | None = None,
        swa_k_ring: torch.Tensor | None = None,
        swa_v_ring: torch.Tensor | None = None,
        swa_valid_ring: torch.Tensor | None = None,
        *,
        s_chunk_start: torch.Tensor | None = None,
        s_far_bank: torch.Tensor | None = None,
        s_chunk_start_bank: torch.Tensor | None = None,
        s_seg_start_bank: torch.Tensor | None = None,
        replay_buf: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        _uniform_step: int | None = None,
        _cached_mid_ring: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        del s_chunk_start
        require(
            swa_k_ring is not None and swa_v_ring is not None and swa_valid_ring is not None,
            "HSSDCacheState requires swa_k_ring, swa_v_ring, and swa_valid_ring",
        )
        assert swa_k_ring is not None and swa_v_ring is not None and swa_valid_ring is not None
        p_states = _compact(page_states)
        bsz = int(p_states.shape[0])
        dev = p_states.device
        inferred_uniform: int | None = _uniform_step
        if isinstance(step_count, torch.Tensor):
            if step_count.ndim == 0:
                u_val = int(step_count.item())
                sc_t = torch.full((bsz,), u_val, device=dev, dtype=torch.long)
                if inferred_uniform is None:
                    inferred_uniform = u_val
            else:
                require(
                    step_count.shape == (bsz,),
                    f"HSSDCacheState.step_count shape {tuple(step_count.shape)} != ({bsz},)",
                )
                sc_t = _compact(step_count.to(device=dev, dtype=torch.long))
        else:
            u_val = int(step_count)
            sc_t = torch.full((bsz,), u_val, device=dev, dtype=torch.long)
            if inferred_uniform is None:
                inferred_uniform = u_val

        th64 = _compact(theta_cum.to(device=dev, dtype=torch.float64))
        w_cnt = int(window_chunks)

        if boundary_ring is not None:
            b_ring = tuple(
                p_states if (t is page_states) else _compact(t.to(device=dev, dtype=p_states.dtype))
                for t in boundary_ring
            )
            require(
                len(b_ring) == w_cnt + 1,
                f"boundary_ring length {len(b_ring)} != window_chunks + 1 ({w_cnt + 1})",
            )
            self._boundary_ring: tuple[torch.Tensor, ...] | None = b_ring
            self._s_far_bank: torch.Tensor = b_ring[0]
            self._s_chunk_start_bank: torch.Tensor = b_ring[-1]
            self.replay_buf: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            self._cached_mid_ring: tuple[torch.Tensor, ...] | None = None
        else:
            far_t = p_states if (s_far_bank is None or s_far_bank is page_states) else _compact(
                s_far_bank.to(device=dev, dtype=p_states.dtype)
            )
            cstart_t = (
                p_states
                if (s_chunk_start_bank is None or s_chunk_start_bank is page_states)
                else (
                    far_t
                    if s_chunk_start_bank is s_far_bank
                    else _compact(s_chunk_start_bank.to(device=dev, dtype=p_states.dtype))
                )
            )
            self._boundary_ring = None
            self._s_far_bank = far_t
            self._s_chunk_start_bank = cstart_t
            if replay_buf is not None:
                rk, rv, rg, rgam, ridx = replay_buf
                self.replay_buf = (
                    _compact(rk.to(device=dev, dtype=p_states.dtype)),
                    _compact(rv.to(device=dev, dtype=p_states.dtype)),
                    _compact(rg.to(device=dev, dtype=p_states.dtype)),
                    _compact(rgam.to(device=dev, dtype=p_states.dtype)),
                    _compact(ridx.to(device=dev, dtype=torch.long)),
                )
            else:
                self.replay_buf = None
            self._cached_mid_ring = _cached_mid_ring

        if s_seg_start_bank is None:
            self._s_seg_start_bank: torch.Tensor | None = None
        elif s_seg_start_bank is page_states:
            self._s_seg_start_bank = p_states
        elif s_seg_start_bank is s_chunk_start_bank:
            self._s_seg_start_bank = self._s_chunk_start_bank
        else:
            self._s_seg_start_bank = _compact(s_seg_start_bank.to(device=dev, dtype=p_states.dtype))

        self.step_count: torch.Tensor = sc_t
        self._uniform_step: int | None = inferred_uniform
        self.chunk_size: int = int(chunk_size)
        self.window_chunks: int = w_cnt
        self.page_states: torch.Tensor = p_states
        self.theta_cum: torch.Tensor = th64
        self.swa_k_ring: torch.Tensor = _compact(swa_k_ring.to(device=dev, dtype=p_states.dtype))
        self.swa_v_ring: torch.Tensor = _compact(swa_v_ring.to(device=dev, dtype=p_states.dtype))
        self.swa_valid_ring: torch.Tensor = _compact(swa_valid_ring.to(device=dev, dtype=torch.bool))

    def __repr__(self) -> str:
        mode = "compact" if self._boundary_ring is None else "ring"
        return (
            f"HSSDCacheState(shape={tuple(self.shape)}, dtype={self.dtype}, mode={mode}, "
            f"theta_dtype={self.theta_cum.dtype}, step_count={self.step_count.tolist()}, "
            f"chunk_size={self.chunk_size}, window_chunks={self.window_chunks})"
        )

    @property
    def is_compact_ring(self) -> bool:
        return self._boundary_ring is None

    @property
    def s_far_bank(self) -> torch.Tensor:
        return self._boundary_ring[0] if self._boundary_ring is not None else self._s_far_bank

    @property
    def s_chunk_start_bank(self) -> torch.Tensor:
        """Full `M`-page chunk-start state bank `S_{nC}` (`[B, M, H_v, d_k, d_v]`)."""
        return self._boundary_ring[-1] if self._boundary_ring is not None else self._s_chunk_start_bank

    @property
    def s_seg_start_bank(self) -> torch.Tensor:
        return self._s_seg_start_bank if self._s_seg_start_bank is not None else self.s_chunk_start_bank

    @property
    def s_chunk_start(self) -> torch.Tensor:
        """Primary Page-0 chunk-start state `S_{nC}^{(0)}` (`[B, H_v, d_k, d_v]`)."""
        return self.s_chunk_start_bank[:, 0]

    @property
    def boundary_ring(self) -> tuple[torch.Tensor, ...]:
        if self._boundary_ring is not None:
            return self._boundary_ring
        w = self.window_chunks
        if w == 0:
            return (self._s_chunk_start_bank,)
        if self.replay_buf is None:
            return (self._s_far_bank, *([self._s_chunk_start_bank] * (w - 1)), self._s_chunk_start_bank)
        if self._cached_mid_ring is not None and len(self._cached_mid_ring) == w - 1:
            return (self._s_far_bank, *self._cached_mid_ring, self._s_chunk_start_bank)

        bsz = int(self.page_states.shape[0])
        if bsz > 1 and self._uniform_step is None and not bool((self.step_count == self.step_count[0]).all()):
            per_row = [self[b : b + 1].boundary_ring for b in range(bsz)]
            mid_cat = tuple(torch.cat([per_row[b][j] for b in range(bsz)], dim=0) for j in range(1, w))
            self._cached_mid_ring = mid_cat
            return (self._s_far_bank, *mid_cat, self._s_chunk_start_bank)

        step0 = (
            self._uniform_step
            if self._uniform_step is not None
            else (int(self.step_count[0].item()) if bsz > 0 else 0)
        )
        n_done = step0 // self.chunk_size
        k_far = max(0, n_done - w)
        replayed: dict[int, torch.Tensor] = {k_far: self._s_far_bank, n_done: self._s_chunk_start_bank}
        cur_bank = self._s_far_bank
        for c_idx in range(k_far, n_done - 1):
            cur_bank = _replay_chunk_on_bank(cur_bank, self.replay_buf, c_idx, self.chunk_size, w)
            replayed[c_idx + 1] = cur_bank
        mid_list: list[torch.Tensor] = []
        for j in range(1, w):
            target_k = max(0, n_done - (w - j))
            mid_list.append(replayed.get(target_k, self._s_far_bank))
        self._cached_mid_ring = tuple(mid_list)
        return (self._s_far_bank, *self._cached_mid_ring, self._s_chunk_start_bank)

    @boundary_ring.setter
    def boundary_ring(self, new_ring: tuple[torch.Tensor, ...]) -> None:
        dev = self.page_states.device
        dt = self.page_states.dtype
        b_ring = tuple(
            self.page_states if (t is self.page_states) else _compact(t.to(device=dev, dtype=dt))
            for t in new_ring
        )
        require(
            len(b_ring) == self.window_chunks + 1,
            f"boundary_ring length {len(b_ring)} != window_chunks + 1 ({self.window_chunks + 1})",
        )
        self._boundary_ring = b_ring
        self._s_far_bank = b_ring[0]
        self._s_chunk_start_bank = b_ring[-1]
        self.replay_buf = None
        self._cached_mid_ring = None

    def _banks_have_dtype(self, dt: torch.dtype) -> bool:
        if self._boundary_ring is not None:
            return all(t.dtype == dt for t in self._boundary_ring)
        if self._s_far_bank.dtype != dt or self._s_chunk_start_bank.dtype != dt:
            return False
        if self._s_seg_start_bank is not None and self._s_seg_start_bank.dtype != dt:
            return False
        if self.replay_buf is not None:
            rk, rv, rg, rgam, _ridx = self.replay_buf
            if rk.dtype != dt or rv.dtype != dt or rg.dtype != dt or rgam.dtype != dt:
                return False
        return True

    @property
    def step_count_int(self) -> int:
        if self._uniform_step is not None:
            return self._uniform_step
        if self.step_count.numel() == 0:
            return 0
        if self.step_count.numel() > 1:
            require(
                bool((self.step_count == self.step_count[0]).all()),
                "step_count_int requires uniform step_count across batch",
                step_count=self.step_count,
            )
        val = int(self.step_count[0].item())
        self._uniform_step = val
        return val

    @property
    def max_swa_len(self) -> int:
        return (self.window_chunks + 1) * self.chunk_size

    @property
    def shape(self) -> torch.Size:
        return self.page_states.shape

    def size(self, dim: int | None = None) -> torch.Size | int:
        return self.page_states.size() if dim is None else self.page_states.size(dim)

    @property
    def ndim(self) -> int:
        return self.page_states.ndim

    def dim(self) -> int:
        return self.page_states.dim()

    def numel(self) -> int:
        return self.page_states.numel()

    def __len__(self) -> int:
        return int(self.page_states.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.page_states.dtype

    @property
    def device(self) -> torch.device:
        return self.page_states.device

    def _iter_tensors(self) -> tuple[torch.Tensor, ...]:
        if self._boundary_ring is not None:
            seg_extra = (self._s_seg_start_bank,) if self._s_seg_start_bank is not None else ()
            return (
                self.step_count,
                self.page_states,
                self.theta_cum,
                *self._boundary_ring,
                *seg_extra,
                self.swa_k_ring,
                self.swa_v_ring,
                self.swa_valid_ring,
            )
        seg_extra = (self._s_seg_start_bank,) if self._s_seg_start_bank is not None else ()
        rep_extra = self.replay_buf if self.replay_buf is not None else ()
        return (
            self.step_count,
            self.page_states,
            self.theta_cum,
            self._s_far_bank,
            self._s_chunk_start_bank,
            *seg_extra,
            *rep_extra,
            self.swa_k_ring,
            self.swa_v_ring,
            self.swa_valid_ring,
        )

    def unique_storage_bytes(self) -> int:
        """Total unique backing-storage bytes across all tensors in this state (deduplicated by `data_ptr`)."""
        seen: set[int] = set()
        total = 0
        for t in self._iter_tensors():
            if t.numel() <= 0:
                continue
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                total += int(st.nbytes())
        return total

    def has_compact_storage(self) -> bool:
        """Return True iff every tensor is contiguous and owns exact-sized backing storage (zero view leaks)."""
        for t in self._iter_tensors():
            if not t.is_contiguous():
                return False
            if int(t.untyped_storage().nbytes()) != int(t.numel() * t.element_size()) or int(t.storage_offset()) != 0:
                return False
        return True

    def is_contiguous(self) -> bool:
        """Return True iff every tensor in `_iter_tensors()` is contiguous."""
        return all(t.is_contiguous() for t in self._iter_tensors())

    def contiguous(self) -> HSSDCacheState:
        """Return `self` if `has_compact_storage()` else a compacted copy."""
        if self.has_compact_storage():
            return self
        memo: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}

        def _cp(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                key = (
                    int(t.untyped_storage().data_ptr()),
                    int(t.storage_offset()),
                    tuple(t.shape),
                    tuple(t.stride()),
                )
                if key in memo:
                    return memo[key]
                out_t = _compact(t)
                memo[key] = out_t
                return out_t
            return _compact(t)

        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=_compact(self.step_count),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_cp(self.page_states),
                theta_cum=_compact(self.theta_cum),
                boundary_ring=tuple(_cp(t) for t in self._boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring),
                swa_v_ring=_compact(self.swa_v_ring),
                swa_valid_ring=_compact(self.swa_valid_ring),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _cp(self._s_seg_start_bank),
                _uniform_step=self._uniform_step,
            )
        rep_cp = (
            None
            if self.replay_buf is None
            else (
                _compact(self.replay_buf[0]),
                _compact(self.replay_buf[1]),
                _compact(self.replay_buf[2]),
                _compact(self.replay_buf[3]),
                _compact(self.replay_buf[4]),
            )
        )
        return HSSDCacheState(
            step_count=_compact(self.step_count),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_cp(self.page_states),
            theta_cum=_compact(self.theta_cum),
            boundary_ring=None,
            swa_k_ring=_compact(self.swa_k_ring),
            swa_v_ring=_compact(self.swa_v_ring),
            swa_valid_ring=_compact(self.swa_valid_ring),
            s_far_bank=_cp(self._s_far_bank),
            s_chunk_start_bank=_cp(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _cp(self._s_seg_start_bank),
            replay_buf=rep_cp,
            _uniform_step=self._uniform_step,
        )

    def compact(self) -> HSSDCacheState:
        return self.contiguous()

    def zero_(self) -> HSSDCacheState:
        """In-place zero all state buffers, unaliasing shared ring slots while preserving distinct static addresses."""
        self._cached_mid_ring = None
        if self._boundary_ring is None and self.replay_buf is None:
            # Aliased zero-init state (`cls.zeros(...)`) -> unalias into explicit W + 1 zero buffers
            self._boundary_ring = tuple(
                torch.zeros_like(self.page_states).contiguous() for _ in range(self.window_chunks + 1)
            )
            self._s_far_bank = self._boundary_ring[0]
            self._s_chunk_start_bank = self._boundary_ring[-1]
            if self._s_seg_start_bank is not None:
                self._s_seg_start_bank = torch.zeros_like(self.page_states).contiguous()
        elif self._boundary_ring is not None:
            seen_ptrs: set[int] = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            fixed_ring: list[torch.Tensor] = []
            unaliased = False
            for ring_t in self._boundary_ring:
                cur_t = ring_t
                if cur_t.numel() > 0:
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    if ptr in seen_ptrs:
                        cur_t = torch.zeros_like(cur_t).contiguous()
                        ptr = int(cur_t.untyped_storage().data_ptr())
                        unaliased = True
                    else:
                        cur_t.zero_()
                    seen_ptrs.add(ptr)
                else:
                    cur_t.zero_()
                fixed_ring.append(cur_t)
            if unaliased:
                self._boundary_ring = tuple(fixed_ring)
                self._s_far_bank = self._boundary_ring[0]
                self._s_chunk_start_bank = self._boundary_ring[-1]
            if self._s_seg_start_bank is not None:
                if (
                    self._s_seg_start_bank.numel() > 0
                    and int(self._s_seg_start_bank.untyped_storage().data_ptr()) in seen_ptrs
                ):
                    self._s_seg_start_bank = torch.zeros_like(self._s_seg_start_bank).contiguous()
                else:
                    self._s_seg_start_bank.zero_()
        else:
            # Compact lagged-replay mode (`self.replay_buf is not None`)
            seen_c: set[int] = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            if self._s_far_bank.numel() > 0:
                p_far = int(self._s_far_bank.untyped_storage().data_ptr())
                if p_far in seen_c:
                    self._s_far_bank = torch.zeros_like(self._s_far_bank).contiguous()
                    p_far = int(self._s_far_bank.untyped_storage().data_ptr())
                else:
                    self._s_far_bank.zero_()
                seen_c.add(p_far)
            else:
                self._s_far_bank.zero_()

            if self._s_chunk_start_bank.numel() > 0:
                p_cs = int(self._s_chunk_start_bank.untyped_storage().data_ptr())
                if p_cs in seen_c:
                    self._s_chunk_start_bank = torch.zeros_like(self._s_chunk_start_bank).contiguous()
                    p_cs = int(self._s_chunk_start_bank.untyped_storage().data_ptr())
                else:
                    self._s_chunk_start_bank.zero_()
                seen_c.add(p_cs)
            else:
                self._s_chunk_start_bank.zero_()

            if self._s_seg_start_bank is not None:
                if self._s_seg_start_bank.numel() > 0:
                    p_ss = int(self._s_seg_start_bank.untyped_storage().data_ptr())
                    if p_ss in seen_c:
                        self._s_seg_start_bank = torch.zeros_like(self._s_seg_start_bank).contiguous()
                    else:
                        self._s_seg_start_bank.zero_()
                else:
                    self._s_seg_start_bank.zero_()

            assert self.replay_buf is not None
            for rt in self.replay_buf:
                rt.zero_()

        self.step_count.zero_()
        self._uniform_step = 0
        self.page_states.zero_()
        self.theta_cum.zero_()
        self.swa_k_ring.zero_()
        self.swa_v_ring.zero_()
        self.swa_valid_ring.zero_()
        return self

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
        top_k_pages: int = 2,
        compact_decode_cache: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        distinct_ring_buffers: bool = False,
    ) -> HSSDCacheState:
        del top_k_pages
        p_states = torch.zeros(batch_size, num_pages, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=dtype)
        theta = torch.zeros(batch_size, num_v_heads, head_k_dim // 2, device=device, dtype=torch.float64)
        max_win = (window_chunks + 1) * chunk_size
        k_ring = torch.zeros(batch_size, max_win, num_v_heads, head_k_dim, device=device, dtype=dtype)
        v_ring = torch.zeros(batch_size, max_win, num_v_heads, head_v_dim, device=device, dtype=dtype)
        val_ring = torch.zeros(batch_size, max_win, device=device, dtype=torch.bool)
        steps = torch.zeros(batch_size, device=device, dtype=torch.long)
        if distinct_ring_buffers:
            b_ring = tuple(torch.zeros_like(p_states) for _ in range(window_chunks + 1))
            return cls(
                step_count=steps,
                chunk_size=int(chunk_size),
                window_chunks=int(window_chunks),
                page_states=p_states,
                theta_cum=theta,
                boundary_ring=b_ring,
                swa_k_ring=k_ring,
                swa_v_ring=v_ring,
                swa_valid_ring=val_ring,
                _uniform_step=0,
            )
        if not compact_decode_cache:
            return cls(
                step_count=steps,
                chunk_size=int(chunk_size),
                window_chunks=int(window_chunks),
                page_states=p_states,
                theta_cum=theta,
                boundary_ring=(p_states,) * (int(window_chunks) + 1),
                swa_k_ring=k_ring,
                swa_v_ring=v_ring,
                swa_valid_ring=val_ring,
                _uniform_step=0,
            )
        return cls(
            step_count=steps,
            chunk_size=int(chunk_size),
            window_chunks=int(window_chunks),
            page_states=p_states,
            theta_cum=theta,
            boundary_ring=None,
            swa_k_ring=k_ring,
            swa_v_ring=v_ring,
            swa_valid_ring=val_ring,
            s_far_bank=p_states,
            s_chunk_start_bank=p_states,
            replay_buf=None,
            _uniform_step=0,
        )

    @classmethod
    def _like_state(
        cls,
        other: HSSDCacheState,
        *,
        zero_fill: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HSSDCacheState:
        dev = other.device if device is None else torch.device(device)
        dt = other.dtype if dtype is None else dtype
        alloc = torch.zeros if zero_fill else torch.empty
        bsz, mp, hv, dk, dv = other.page_states.shape
        p_states = alloc(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
        theta = alloc(bsz, hv, dk // 2, device=dev, dtype=torch.float64)
        max_win = other.max_swa_len
        k_ring = alloc(bsz, max_win, hv, dk, device=dev, dtype=dt)
        v_ring = alloc(bsz, max_win, hv, dv, device=dev, dtype=dt)
        val_ring = alloc(bsz, max_win, device=dev, dtype=torch.bool)
        steps = alloc(bsz, device=dev, dtype=torch.long)
        seg_bank = (
            alloc(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
            if getattr(other, "_s_seg_start_bank", None) is not None
            else None
        )
        u_step = 0 if zero_fill else None

        if getattr(other, "_boundary_ring", None) is None and getattr(other, "replay_buf", None) is not None:
            assert other.replay_buf is not None
            k_sel = int(other.replay_buf[2].shape[2])
            gam_dim = int(other.replay_buf[3].shape[-1])
            far_b = alloc(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
            cs_b = alloc(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
            rep_b = (
                alloc(bsz, hv, max_win, dk, device=dev, dtype=dt),
                alloc(bsz, hv, max_win, dv, device=dev, dtype=dt),
                alloc(bsz, max_win, k_sel, hv, device=dev, dtype=dt),
                alloc(bsz, max_win, k_sel, hv, gam_dim, device=dev, dtype=dt),
                alloc(bsz, max_win, k_sel, device=dev, dtype=torch.long),
            )
            return cls(
                step_count=steps,
                chunk_size=other.chunk_size,
                window_chunks=other.window_chunks,
                page_states=p_states,
                theta_cum=theta,
                boundary_ring=None,
                swa_k_ring=k_ring,
                swa_v_ring=v_ring,
                swa_valid_ring=val_ring,
                s_far_bank=far_b,
                s_chunk_start_bank=cs_b,
                s_seg_start_bank=seg_bank,
                replay_buf=rep_b,
                _uniform_step=u_step,
            )

        b_ring = tuple(alloc(bsz, mp, hv, dk, dv, device=dev, dtype=dt) for _ in range(other.window_chunks + 1))
        return cls(
            step_count=steps,
            chunk_size=other.chunk_size,
            window_chunks=other.window_chunks,
            page_states=p_states,
            theta_cum=theta,
            boundary_ring=b_ring,
            swa_k_ring=k_ring,
            swa_v_ring=v_ring,
            swa_valid_ring=val_ring,
            s_seg_start_bank=seg_bank,
            _uniform_step=u_step,
        )

    @classmethod
    def empty_like_state(
        cls,
        other: HSSDCacheState,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HSSDCacheState:
        return cls._like_state(other, zero_fill=False, device=device, dtype=dtype)

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
            p_states = st.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone().contiguous()
        elif st.ndim == 5:
            bsz, mp, hv, dk, dv = st.shape
            require(mp == num_pages, f"expected {num_pages} state pages, got {mp}", state=st)
            p_states = st.clone().contiguous()
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
        base._s_far_bank = p_states
        base._s_chunk_start_bank = p_states
        return base

    def clone(self) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}

        def _cl(t: torch.Tensor) -> torch.Tensor:
            ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
            if ptr not in memo:
                memo[ptr] = t.clone().contiguous()
            return memo[ptr]

        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=self.step_count.clone().contiguous(),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_cl(self.page_states),
                theta_cum=self.theta_cum.clone().contiguous(),
                boundary_ring=tuple(_cl(t) for t in self._boundary_ring),
                swa_k_ring=self.swa_k_ring.clone().contiguous(),
                swa_v_ring=self.swa_v_ring.clone().contiguous(),
                swa_valid_ring=self.swa_valid_ring.clone().contiguous(),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _cl(self._s_seg_start_bank),
                _uniform_step=self._uniform_step,
            )
        rep_cl = (
            None
            if self.replay_buf is None
            else (
                self.replay_buf[0].clone().contiguous(),
                self.replay_buf[1].clone().contiguous(),
                self.replay_buf[2].clone().contiguous(),
                self.replay_buf[3].clone().contiguous(),
                self.replay_buf[4].clone().contiguous(),
            )
        )
        return HSSDCacheState(
            step_count=self.step_count.clone().contiguous(),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_cl(self.page_states),
            theta_cum=self.theta_cum.clone().contiguous(),
            boundary_ring=None,
            swa_k_ring=self.swa_k_ring.clone().contiguous(),
            swa_v_ring=self.swa_v_ring.clone().contiguous(),
            swa_valid_ring=self.swa_valid_ring.clone().contiguous(),
            s_far_bank=_cl(self._s_far_bank),
            s_chunk_start_bank=_cl(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _cl(self._s_seg_start_bank),
            replay_buf=rep_cl,
            _uniform_step=self._uniform_step,
        )

    def detach(self) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}

        def _dt(t: torch.Tensor) -> torch.Tensor:
            ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
            if ptr not in memo:
                memo[ptr] = _compact(t.detach())
            return memo[ptr]

        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=_compact(self.step_count.detach()),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_dt(self.page_states),
                theta_cum=_compact(self.theta_cum.detach()),
                boundary_ring=tuple(_dt(t) for t in self._boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.detach()),
                swa_v_ring=_compact(self.swa_v_ring.detach()),
                swa_valid_ring=_compact(self.swa_valid_ring.detach()),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _dt(self._s_seg_start_bank),
                _uniform_step=self._uniform_step,
            )
        rep_dt = (
            None
            if self.replay_buf is None
            else (
                _compact(self.replay_buf[0].detach()),
                _compact(self.replay_buf[1].detach()),
                _compact(self.replay_buf[2].detach()),
                _compact(self.replay_buf[3].detach()),
                _compact(self.replay_buf[4].detach()),
            )
        )
        return HSSDCacheState(
            step_count=_compact(self.step_count.detach()),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_dt(self.page_states),
            theta_cum=_compact(self.theta_cum.detach()),
            boundary_ring=None,
            swa_k_ring=_compact(self.swa_k_ring.detach()),
            swa_v_ring=_compact(self.swa_v_ring.detach()),
            swa_valid_ring=_compact(self.swa_valid_ring.detach()),
            s_far_bank=_dt(self._s_far_bank),
            s_chunk_start_bank=_dt(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _dt(self._s_seg_start_bank),
            replay_buf=rep_dt,
            _uniform_step=self._uniform_step,
        )

    def to(self, *args: Any, **kwargs: Any) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}
        non_blocking = bool(kwargs.get("non_blocking", False))
        p_states = _compact(self.page_states.to(*args, **kwargs))
        if self.page_states.numel() > 0:
            memo[int(self.page_states.untyped_storage().data_ptr())] = p_states
        dev = p_states.device
        dt = p_states.dtype

        def _conv(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))
                memo[ptr] = out_t
                return out_t
            return _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))

        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=_compact(self.step_count.to(device=dev, dtype=torch.long, non_blocking=non_blocking)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=p_states,
                theta_cum=_compact(self.theta_cum.to(device=dev, dtype=torch.float64, non_blocking=non_blocking)),
                boundary_ring=tuple(_conv(t) for t in self._boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
                swa_v_ring=_compact(self.swa_v_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
                swa_valid_ring=_compact(
                    self.swa_valid_ring.to(device=dev, dtype=torch.bool, non_blocking=non_blocking)
                ),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _conv(self._s_seg_start_bank),
                _uniform_step=self._uniform_step,
            )
        rep_cv = (
            None
            if self.replay_buf is None
            else (
                _compact(self.replay_buf[0].to(device=dev, dtype=dt, non_blocking=non_blocking)),
                _compact(self.replay_buf[1].to(device=dev, dtype=dt, non_blocking=non_blocking)),
                _compact(self.replay_buf[2].to(device=dev, dtype=dt, non_blocking=non_blocking)),
                _compact(self.replay_buf[3].to(device=dev, dtype=dt, non_blocking=non_blocking)),
                _compact(self.replay_buf[4].to(device=dev, dtype=torch.long, non_blocking=non_blocking)),
            )
        )
        return HSSDCacheState(
            step_count=_compact(self.step_count.to(device=dev, dtype=torch.long, non_blocking=non_blocking)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=p_states,
            theta_cum=_compact(self.theta_cum.to(device=dev, dtype=torch.float64, non_blocking=non_blocking)),
            boundary_ring=None,
            swa_k_ring=_compact(self.swa_k_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
            swa_v_ring=_compact(self.swa_v_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
            swa_valid_ring=_compact(self.swa_valid_ring.to(device=dev, dtype=torch.bool, non_blocking=non_blocking)),
            s_far_bank=_conv(self._s_far_bank),
            s_chunk_start_bank=_conv(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _conv(self._s_seg_start_bank),
            replay_buf=rep_cv,
            _uniform_step=self._uniform_step,
        )

    def copy_(self, other: Any) -> HSSDCacheState:
        if other is self:
            return self
        require(
            isinstance(other, HSSDCacheState) or _is_hssd_state_duck(other),
            f"copy_ expects HSSDCacheState, got {type(other).__name__}",
        )
        require(
            self.chunk_size == int(other.chunk_size) and self.window_chunks == int(other.window_chunks),
            "HSSDCacheState.copy_ chunk_size/window_chunks mismatch",
            dst=(self.chunk_size, self.window_chunks),
            src=(other.chunk_size, other.window_chunks),
        )
        self._cached_mid_ring = None
        other_compact = (
            isinstance(other, HSSDCacheState)
            and other._boundary_ring is None
            and other.replay_buf is not None
        )
        self_compact = self._boundary_ring is None and self.replay_buf is not None

        if self_compact and other_compact:
            seen_c: set[int] = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            if self._s_far_bank.numel() > 0:
                p_far = int(self._s_far_bank.untyped_storage().data_ptr())
                if p_far in seen_c:
                    self._s_far_bank = self._s_far_bank.clone().contiguous()
                    p_far = int(self._s_far_bank.untyped_storage().data_ptr())
                seen_c.add(p_far)
            if self._s_chunk_start_bank.numel() > 0:
                p_cs = int(self._s_chunk_start_bank.untyped_storage().data_ptr())
                if p_cs in seen_c:
                    self._s_chunk_start_bank = self._s_chunk_start_bank.clone().contiguous()
                    p_cs = int(self._s_chunk_start_bank.untyped_storage().data_ptr())
                seen_c.add(p_cs)

            self.step_count.copy_(other.step_count)
            self._uniform_step = other._uniform_step
            self.page_states.copy_(other.page_states)
            self.theta_cum.copy_(other.theta_cum)
            self._s_far_bank.copy_(other._s_far_bank)
            self._s_chunk_start_bank.copy_(other._s_chunk_start_bank)
            if other._s_seg_start_bank is not None:
                if self._s_seg_start_bank is None or (
                    self._s_seg_start_bank.numel() > 0
                    and int(self._s_seg_start_bank.untyped_storage().data_ptr()) in seen_c
                ):
                    self._s_seg_start_bank = other._s_seg_start_bank.to(
                        device=self.device, dtype=self.dtype
                    ).clone().contiguous()
                else:
                    self._s_seg_start_bank.copy_(other._s_seg_start_bank)
            else:
                self._s_seg_start_bank = None
            assert self.replay_buf is not None and other.replay_buf is not None
            for dst_r, src_r in zip(self.replay_buf, other.replay_buf):
                dst_r.copy_(src_r)
        else:
            if self._boundary_ring is None:
                self._boundary_ring = tuple(
                    self._s_far_bank if j == 0 else (
                        self._s_chunk_start_bank if j == self.window_chunks else self.page_states.clone().contiguous()
                    )
                    for j in range(self.window_chunks + 1)
                )
                self.replay_buf = None
            seen_ptrs: set[int] = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            fixed_ring: list[torch.Tensor] = []
            unaliased = False
            for ring_t in self._boundary_ring:
                cur_t = ring_t
                if cur_t.numel() > 0:
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    if ptr in seen_ptrs:
                        cur_t = cur_t.clone().contiguous()
                        ptr = int(cur_t.untyped_storage().data_ptr())
                        unaliased = True
                    seen_ptrs.add(ptr)
                fixed_ring.append(cur_t)
            if unaliased:
                self._boundary_ring = tuple(fixed_ring)
            self._s_far_bank = self._boundary_ring[0]
            self._s_chunk_start_bank = self._boundary_ring[-1]

            self.step_count.copy_(other.step_count)
            self._uniform_step = getattr(other, "_uniform_step", None)
            self.page_states.copy_(other.page_states)
            self.theta_cum.copy_(other.theta_cum)
            for dst, src in zip(self._boundary_ring, other.boundary_ring):
                dst.copy_(src)
            oth_seg = getattr(other, "_s_seg_start_bank", None)
            if oth_seg is not None:
                if self._s_seg_start_bank is None:
                    self._s_seg_start_bank = oth_seg.to(device=self.device, dtype=self.dtype).clone().contiguous()
                else:
                    self._s_seg_start_bank.copy_(oth_seg)
            else:
                self._s_seg_start_bank = None

        self.swa_k_ring.copy_(other.swa_k_ring)
        self.swa_v_ring.copy_(other.swa_v_ring)
        self.swa_valid_ring.copy_(other.swa_valid_ring)
        return self

    def index_select(self, dim: int, index: torch.Tensor) -> HSSDCacheState:
        d = dim if dim >= 0 else dim + self.ndim
        if d != 0:
            raise ValueError(f"HSSDCacheState.index_select only supports batch dim=0, got dim={dim}")
        idx = index.to(device=self.device, dtype=torch.long)
        memo: dict[int, torch.Tensor] = {}

        def _sel(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.index_select(0, idx))
                memo[ptr] = out_t
                return out_t
            return _compact(t.index_select(0, idx))

        u_step = self._uniform_step if self._uniform_step is not None else None
        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=_compact(self.step_count.index_select(0, idx)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_sel(self.page_states),
                theta_cum=_compact(self.theta_cum.index_select(0, idx)),
                boundary_ring=tuple(_sel(t) for t in self._boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.index_select(0, idx)),
                swa_v_ring=_compact(self.swa_v_ring.index_select(0, idx)),
                swa_valid_ring=_compact(self.swa_valid_ring.index_select(0, idx)),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _sel(self._s_seg_start_bank),
                _uniform_step=u_step,
            )
        rep_sel = (
            None
            if self.replay_buf is None
            else (
                _compact(self.replay_buf[0].index_select(0, idx)),
                _compact(self.replay_buf[1].index_select(0, idx)),
                _compact(self.replay_buf[2].index_select(0, idx)),
                _compact(self.replay_buf[3].index_select(0, idx)),
                _compact(self.replay_buf[4].index_select(0, idx)),
            )
        )
        return HSSDCacheState(
            step_count=_compact(self.step_count.index_select(0, idx)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_sel(self.page_states),
            theta_cum=_compact(self.theta_cum.index_select(0, idx)),
            boundary_ring=None,
            swa_k_ring=_compact(self.swa_k_ring.index_select(0, idx)),
            swa_v_ring=_compact(self.swa_v_ring.index_select(0, idx)),
            swa_valid_ring=_compact(self.swa_valid_ring.index_select(0, idx)),
            s_far_bank=_sel(self._s_far_bank),
            s_chunk_start_bank=_sel(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _sel(self._s_seg_start_bank),
            replay_buf=rep_sel,
            _uniform_step=u_step,
        )

    def __getitem__(self, idx: Any) -> HSSDCacheState:
        if isinstance(idx, tuple):
            rest = idx[1:]
            if any(x is not Ellipsis and x != slice(None) for x in rest):
                raise IndexError("HSSDCacheState indexing only supports batch dimension 0")
            idx = idx[0]
        if isinstance(idx, int):
            bsz = len(self)
            norm_i = idx if idx >= 0 else bsz + idx
            if not 0 <= norm_i < bsz:
                raise IndexError(f"batch index {idx} out of range for batch size {bsz}")
            return self.index_select(0, torch.tensor([norm_i], device=self.device, dtype=torch.long))
        if isinstance(idx, slice):
            indices = torch.arange(len(self), device=self.device, dtype=torch.long)[idx]
            return self.index_select(0, indices)
        if isinstance(idx, (list, tuple)):
            return self.index_select(0, torch.tensor(idx, device=self.device, dtype=torch.long))
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                require(idx.ndim == 1 and idx.shape[0] == len(self), "boolean batch mask must have shape [B]")
                indices = idx.to(device=self.device).nonzero(as_tuple=False).squeeze(-1)
                return self.index_select(0, indices)
            return self.index_select(0, idx)
        raise TypeError(f"Unsupported HSSDCacheState index type: {type(idx).__name__}")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> HSSDCacheState:
        return self.index_select(0, beam_idx)

    def repeat_interleave(
        self,
        repeats: int | torch.Tensor,
        dim: int | None = 0,
        *,
        output_size: int | None = None,
    ) -> HSSDCacheState:
        if dim is None:
            raise ValueError("HSSDCacheState.repeat_interleave requires explicit batch dim=0, got dim=None")
        d = dim if dim >= 0 else dim + self.ndim
        if d != 0:
            raise ValueError(f"HSSDCacheState.repeat_interleave only supports batch dim=0, got dim={dim}")
        rep = repeats.to(device=self.device) if isinstance(repeats, torch.Tensor) else repeats
        memo: dict[int, torch.Tensor] = {}

        def _ri(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))
                memo[ptr] = out_t
                return out_t
            return _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))

        if self._boundary_ring is not None:
            return HSSDCacheState(
                step_count=_compact(self.step_count.repeat_interleave(rep, dim=0, output_size=output_size)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_ri(self.page_states),
                theta_cum=_compact(self.theta_cum.repeat_interleave(rep, dim=0, output_size=output_size)),
                boundary_ring=tuple(_ri(t) for t in self._boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
                swa_v_ring=_compact(self.swa_v_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
                swa_valid_ring=_compact(self.swa_valid_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
                s_seg_start_bank=None if self._s_seg_start_bank is None else _ri(self._s_seg_start_bank),
                _uniform_step=self._uniform_step,
            )
        rep_ri = (
            None
            if self.replay_buf is None
            else (
                _compact(self.replay_buf[0].repeat_interleave(rep, dim=0, output_size=output_size)),
                _compact(self.replay_buf[1].repeat_interleave(rep, dim=0, output_size=output_size)),
                _compact(self.replay_buf[2].repeat_interleave(rep, dim=0, output_size=output_size)),
                _compact(self.replay_buf[3].repeat_interleave(rep, dim=0, output_size=output_size)),
                _compact(self.replay_buf[4].repeat_interleave(rep, dim=0, output_size=output_size)),
            )
        )
        return HSSDCacheState(
            step_count=_compact(self.step_count.repeat_interleave(rep, dim=0, output_size=output_size)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_ri(self.page_states),
            theta_cum=_compact(self.theta_cum.repeat_interleave(rep, dim=0, output_size=output_size)),
            boundary_ring=None,
            swa_k_ring=_compact(self.swa_k_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            swa_v_ring=_compact(self.swa_v_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            swa_valid_ring=_compact(self.swa_valid_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            s_far_bank=_ri(self._s_far_bank),
            s_chunk_start_bank=_ri(self._s_chunk_start_bank),
            s_seg_start_bank=None if self._s_seg_start_bank is None else _ri(self._s_seg_start_bank),
            replay_buf=rep_ri,
            _uniform_step=self._uniform_step,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kw = kwargs or {}
        if func is torch.zeros_like:
            st: HSSDCacheState = args[0]
            return cls._like_state(st, zero_fill=True, device=kw.get("device"), dtype=kw.get("dtype"))
        if func is torch.empty_like:
            st = args[0]
            return cls.empty_like_state(st, device=kw.get("device"), dtype=kw.get("dtype"))
        if func is torch.clone:
            st = args[0]
            return st.clone()
        if func is torch.detach:
            st = args[0]
            return st.detach()
        if func is torch.index_select:
            st = args[0]
            dim = kw["dim"] if "dim" in kw else args[1]
            index = kw["index"] if "index" in kw else args[2]
            return st.index_select(int(dim), index)
        if func is torch.repeat_interleave:
            st = args[0]
            repeats = kw["repeats"] if "repeats" in kw else args[1]
            dim = kw.get("dim", args[2] if len(args) >= 3 else None)
            return st.repeat_interleave(repeats, dim=dim, output_size=kw.get("output_size"))
        if func is torch.cat:
            states: Sequence[HSSDCacheState] = args[0]
            dim = int(args[1] if len(args) >= 2 else kw.get("dim", 0))
            if dim not in (0, -5):
                raise ValueError(f"torch.cat on HSSDCacheState only supports batch dim=0, got dim={dim}")
            if not states or not all(isinstance(s, cls) or _is_hssd_state_duck(s) for s in states):
                return NotImplemented
            ref = states[0]
            for s in states[1:]:
                require(
                    int(s.chunk_size) == ref.chunk_size and int(s.window_chunks) == ref.window_chunks,
                    "torch.cat HSSDCacheState chunk_size/window_chunks mismatch",
                )
            u_steps = [getattr(s, "_uniform_step", None) for s in states]
            cat_u = u_steps[0] if (u_steps[0] is not None and all(u == u_steps[0] for u in u_steps)) else None
            any_seg = any(getattr(s, "_s_seg_start_bank", None) is not None for s in states)
            seg_cat = (
                torch.cat([getattr(s, "s_seg_start_bank", s.s_chunk_start_bank) for s in states], dim=0)
                if any_seg
                else None
            )
            all_compact_rep = all(
                isinstance(s, cls)
                and s._boundary_ring is None
                and s.replay_buf is not None
                and s.replay_buf[2].shape[2] == states[0].replay_buf[2].shape[2]  # type: ignore[index]
                for s in states
            )
            if all_compact_rep:
                rep_cat = tuple(
                    torch.cat([s.replay_buf[j] for s in states], dim=0)  # type: ignore[index]
                    for j in range(5)
                )
                return cls(
                    step_count=torch.cat([s.step_count for s in states], dim=0),
                    chunk_size=ref.chunk_size,
                    window_chunks=ref.window_chunks,
                    page_states=torch.cat([s.page_states for s in states], dim=0),
                    theta_cum=torch.cat([s.theta_cum for s in states], dim=0),
                    boundary_ring=None,
                    swa_k_ring=torch.cat([s.swa_k_ring for s in states], dim=0),
                    swa_v_ring=torch.cat([s.swa_v_ring for s in states], dim=0),
                    swa_valid_ring=torch.cat([s.swa_valid_ring for s in states], dim=0),
                    s_far_bank=torch.cat([s._s_far_bank for s in states], dim=0),
                    s_chunk_start_bank=torch.cat([s._s_chunk_start_bank for s in states], dim=0),
                    s_seg_start_bank=seg_cat,
                    replay_buf=rep_cat,  # type: ignore[arg-type]
                    _uniform_step=cat_u,
                )
            all_compact_zero = all(
                isinstance(s, cls) and s._boundary_ring is None and s.replay_buf is None for s in states
            )
            if all_compact_zero:
                return cls(
                    step_count=torch.cat([s.step_count for s in states], dim=0),
                    chunk_size=ref.chunk_size,
                    window_chunks=ref.window_chunks,
                    page_states=torch.cat([s.page_states for s in states], dim=0),
                    theta_cum=torch.cat([s.theta_cum for s in states], dim=0),
                    boundary_ring=None,
                    swa_k_ring=torch.cat([s.swa_k_ring for s in states], dim=0),
                    swa_v_ring=torch.cat([s.swa_v_ring for s in states], dim=0),
                    swa_valid_ring=torch.cat([s.swa_valid_ring for s in states], dim=0),
                    s_far_bank=torch.cat([s._s_far_bank for s in states], dim=0),
                    s_chunk_start_bank=torch.cat([s._s_chunk_start_bank for s in states], dim=0),
                    s_seg_start_bank=seg_cat,
                    replay_buf=None,
                    _uniform_step=cat_u,
                )
            return cls(
                step_count=torch.cat([s.step_count for s in states], dim=0),
                chunk_size=ref.chunk_size,
                window_chunks=ref.window_chunks,
                page_states=torch.cat([s.page_states for s in states], dim=0),
                theta_cum=torch.cat([s.theta_cum for s in states], dim=0),
                boundary_ring=tuple(
                    torch.cat([s.boundary_ring[j] for s in states], dim=0)
                    for j in range(ref.window_chunks + 1)
                ),
                swa_k_ring=torch.cat([s.swa_k_ring for s in states], dim=0),
                swa_v_ring=torch.cat([s.swa_v_ring for s in states], dim=0),
                swa_valid_ring=torch.cat([s.swa_valid_ring for s in states], dim=0),
                s_seg_start_bank=seg_cat,
                _uniform_step=cat_u,
            )
        return NotImplemented

    def state_equal(self, other: Any, *, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
        if not (isinstance(other, HSSDCacheState) or _is_hssd_state_duck(other)):
            return False
        if self.chunk_size != int(other.chunk_size) or self.window_chunks != int(other.window_chunks):
            return False
        osc = (
            other.step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            if isinstance(other.step_count, torch.Tensor)
            else torch.full(self.step_count.shape, int(other.step_count), device=self.device, dtype=torch.long)
        )
        if self.step_count.shape != osc.shape or not bool(torch.equal(self.step_count, osc)):
            return False
        cmp_dt = (
            torch.float64
            if torch.float64 in (self.page_states.dtype, other.page_states.dtype)
            else torch.float32
        )
        if self.page_states.shape != other.page_states.shape or not bool(
            torch.allclose(
                self.page_states.to(cmp_dt),
                other.page_states.to(device=self.device, dtype=cmp_dt),
                atol=atol,
                rtol=rtol,
            )
        ):
            return False
        oth_th = other.theta_cum.to(device=self.device, dtype=torch.float64)
        if self.theta_cum.shape != oth_th.shape or not bool(
            torch.allclose(self.theta_cum, oth_th, atol=atol, rtol=rtol)
        ):
            return False
        if len(self.boundary_ring) != len(other.boundary_ring):
            return False
        for a, b in zip(self.boundary_ring, other.boundary_ring):
            if a.shape != b.shape or not bool(
                torch.allclose(a.to(cmp_dt), b.to(device=self.device, dtype=cmp_dt), atol=atol, rtol=rtol)
            ):
                return False
        oth_val = other.swa_valid_ring.to(device=self.device, dtype=torch.bool)
        if self.swa_valid_ring.shape != oth_val.shape or not bool(torch.equal(self.swa_valid_ring, oth_val)):
            return False
        if self.swa_k_ring.shape != other.swa_k_ring.shape or self.swa_v_ring.shape != other.swa_v_ring.shape:
            return False
        if bool(self.swa_valid_ring.any()):
            vm = self.swa_valid_ring.unsqueeze(-1).unsqueeze(-1)
            k_a = torch.where(vm, self.swa_k_ring.to(cmp_dt), torch.zeros_like(self.swa_k_ring, dtype=cmp_dt))
            k_b = torch.where(
                vm,
                other.swa_k_ring.to(device=self.device, dtype=cmp_dt),
                torch.zeros_like(self.swa_k_ring, dtype=cmp_dt),
            )
            if not bool(torch.allclose(k_a, k_b, atol=atol, rtol=rtol)):
                return False
            v_a = torch.where(vm, self.swa_v_ring.to(cmp_dt), torch.zeros_like(self.swa_v_ring, dtype=cmp_dt))
            v_b = torch.where(
                vm,
                other.swa_v_ring.to(device=self.device, dtype=cmp_dt),
                torch.zeros_like(self.swa_v_ring, dtype=cmp_dt),
            )
            if not bool(torch.allclose(v_a, v_b, atol=atol, rtol=rtol)):
                return False
        return True


class _GrowingSlotList(list):  # type: ignore[type-arg]
    """List that auto-extends with `None` when indexed at a virtual loop slot >= current length."""

    def _ensure(self, idx: int) -> None:
        if idx >= len(self):
            self.extend([None] * (idx + 1 - len(self)))

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int) and idx >= 0:
            self._ensure(idx)
        return super().__getitem__(idx)

    def __setitem__(self, idx: Any, value: Any) -> None:
        if isinstance(idx, int) and idx >= 0:
            self._ensure(idx)
        super().__setitem__(idx, value)


class _HybridHasPreviousState:
    """Dual-mode `has_previous_state`: truthy as a property (`HF 5.2-5.5`) and callable `(layer_idx)` (`HF 5.20+`)."""

    def __init__(self, cache: HSSDHybridCache) -> None:
        self._cache = cache

    def __bool__(self) -> bool:
        return self()

    def __call__(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        del state_idx
        c = self._cache
        if layer_idx is not None:
            cs = c.conv_states[layer_idx] if layer_idx < len(c.conv_states) else None
            rs = c.recurrent_states[layer_idx] if layer_idx < len(c.recurrent_states) else None
            return cs is not None or rs is not None
        last_lin = getattr(c, "last_linear_layer", None)
        if isinstance(last_lin, int) and 0 <= last_lin < len(c.conv_states):
            return c.conv_states[last_lin] is not None or c.recurrent_states[last_lin] is not None
        return any(x is not None for x in c.conv_states) or any(x is not None for x in c.recurrent_states)


class HSSDHybridCache(_HFCacheBase):  # type: ignore[misc]
    """Hugging Face `Cache`-compatible hybrid KV + `HSSDCacheState` container (`Bug 3D`).

    Supports full-attention KV caching (`update`), causal short-conv state slots (`conv_states`),
    independent per-(layer, loop-pass) `HSSDCacheState` slots (`recurrent_states`), beam-search
    `reorder_cache`, `reset`, `get_seq_length`, and deduplicated `unique_storage_bytes()`.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        num_layers: int | None = None,
        layer_types: Sequence[str] | None = None,
    ) -> None:
        if _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                pass
        if layer_types is not None:
            ltypes = [str(x) for x in layer_types]
        elif config is not None and getattr(config, "layer_types", None):
            ltypes = [str(x) for x in config.layer_types]
        else:
            n_cfg = int(num_layers or getattr(config, "num_hidden_layers", 0) or 0)
            ltypes = ["linear_attention"] * n_cfg
        n = int(num_layers) if num_layers is not None else len(ltypes)
        if len(ltypes) < n:
            ltypes.extend(["linear_attention"] * (n - len(ltypes)))
        self._initial_slots = n
        self.layer_types: list[str] = ltypes
        self.key_cache: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.value_cache: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.conv_states: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.recurrent_states: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.first_full: int | None = (
            self.layer_types.index("full_attention") if "full_attention" in self.layer_types else None
        )
        lin_indices = [idx for idx, lt in enumerate(self.layer_types) if lt == "linear_attention"]
        self.last_linear_layer: int | None = lin_indices[-1] if lin_indices else None

    @property
    def has_previous_state(self) -> _HybridHasPreviousState:
        return _HybridHasPreviousState(self)

    def is_recording_past(self, layer_idx: int = 0) -> bool:
        del layer_idx
        return False

    def __len__(self) -> int:
        return max(len(self.key_cache), len(self.recurrent_states))

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape[-2] == 0:
            kc = self.key_cache[layer_idx]
            vc = self.value_cache[layer_idx]
            if kc is not None and vc is not None:
                return kc, vc
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = _compact(key_states)
            self.value_cache[layer_idx] = _compact(value_states)
        else:
            self.key_cache[layer_idx] = _compact(torch.cat([self.key_cache[layer_idx], key_states], dim=-2))
            self.value_cache[layer_idx] = _compact(torch.cat([self.value_cache[layer_idx], value_states], dim=-2))
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            kc = self.key_cache[layer_idx] if layer_idx < len(self.key_cache) else None
            if isinstance(kc, torch.Tensor):
                return int(kc.shape[-2])
            rs = self.recurrent_states[layer_idx] if layer_idx < len(self.recurrent_states) else None
            if isinstance(rs, HSSDCacheState):
                return int(rs.step_count.max().item()) if rs.step_count.numel() > 0 else 0
            return 0
        if self.first_full is not None and self.first_full < len(self.key_cache):
            kc0 = self.key_cache[self.first_full]
            if isinstance(kc0, torch.Tensor):
                return int(kc0.shape[-2])
        for kc in self.key_cache:
            if isinstance(kc, torch.Tensor):
                return int(kc.shape[-2])
        for rs in self.recurrent_states:
            if isinstance(rs, HSSDCacheState) and rs.step_count.numel() > 0:
                return int(rs.step_count.max().item())
        return 0

    def get_max_cache_shape(self) -> int | None:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for seq in (self.key_cache, self.value_cache, self.conv_states):
            for i, item in enumerate(seq):
                if isinstance(item, torch.Tensor):
                    seq[i] = _compact(item.index_select(0, beam_idx.to(device=item.device, dtype=torch.long)))
        for i, st in enumerate(self.recurrent_states):
            if isinstance(st, HSSDCacheState) or _is_hssd_state_duck(st):
                self.recurrent_states[i] = st.reorder_cache(beam_idx)
            elif isinstance(st, torch.Tensor):
                self.recurrent_states[i] = _compact(
                    st.index_select(0, beam_idx.to(device=st.device, dtype=torch.long))
                )

    def reset(self) -> None:
        n = max(self._initial_slots, len(self))
        self.key_cache[:] = [None] * n
        self.value_cache[:] = [None] * n
        self.conv_states[:] = [None] * n
        self.recurrent_states[:] = [None] * n

    def slot_index(self, layer_idx: int, *, pass_idx: int = 0, stride: int | None = None) -> int:
        if pass_idx < 0:
            raise ValueError(f"pass_idx must be >= 0, got {pass_idx}")
        base = max(int(stride) if stride is not None else self._initial_slots, 1)
        return int(layer_idx) + int(pass_idx) * base

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        *,
        conv_kernel_size: int | None = None,
        pass_idx: int = 0,
        stride: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        prev = self.conv_states[idx]
        if prev is None:
            full_conv = conv_states
            if conv_kernel_size is not None and full_conv.shape[-1] < int(conv_kernel_size):
                full_conv = F.pad(full_conv, (int(conv_kernel_size) - full_conv.shape[-1], 0), value=0)
        else:
            full_conv = torch.cat([prev, conv_states], dim=-1)
        keep_w = int(conv_kernel_size) if conv_kernel_size is not None else full_conv.shape[-1]
        self.conv_states[idx] = _compact(full_conv[..., -keep_w:])
        return full_conv

    def get_conv_state(
        self,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
    ) -> torch.Tensor | None:
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        return self.conv_states[idx] if idx < len(self.conv_states) else None

    def update_recurrent_state(
        self,
        state: HSSDCacheState | torch.Tensor | None,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
        **kwargs: Any,
    ) -> HSSDCacheState | torch.Tensor | None:
        del kwargs
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        self.recurrent_states[idx] = state
        return state

    def get_recurrent_state(
        self,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
    ) -> HSSDCacheState | torch.Tensor | None:
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        return self.recurrent_states[idx] if idx < len(self.recurrent_states) else None

    def unique_storage_bytes(self) -> int:
        seen: set[int] = set()
        total = 0

        def _acc(t: torch.Tensor) -> None:
            nonlocal total
            if t.numel() <= 0:
                return
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                total += int(st.nbytes())

        for seq in (self.key_cache, self.value_cache, self.conv_states):
            for item in seq:
                if isinstance(item, torch.Tensor):
                    _acc(item)
        for rs in self.recurrent_states:
            if isinstance(rs, HSSDCacheState) or _is_hssd_state_duck(rs):
                for t in rs._iter_tensors():
                    _acc(t)
            elif isinstance(rs, torch.Tensor):
                _acc(rs)
        return total


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
    top_k_pages: int = 2,
    compact_decode_cache: bool = True,
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
            top_k_pages=top_k_pages,
            compact_decode_cache=compact_decode_cache,
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
        if (
            initial_state.device == device
            and initial_state.dtype == dtype
            and initial_state.theta_cum.dtype == torch.float64
            and initial_state.swa_k_ring.dtype == dtype
            and initial_state.swa_v_ring.dtype == dtype
            and initial_state._banks_have_dtype(dtype)
            and initial_state.has_compact_storage()
        ):
            return initial_state
        return initial_state.to(device=device, dtype=dtype)
    if _is_hssd_state_duck(initial_state):
        require(
            int(initial_state.chunk_size) == chunk_size and int(initial_state.window_chunks) == window_chunks,
            "HSSDCacheState chunk_size/window_chunks mismatch",
            got=(initial_state.chunk_size, initial_state.window_chunks),
            want=(chunk_size, window_chunks),
        )
        if (
            getattr(initial_state, "_boundary_ring", None) is None
            and getattr(initial_state, "replay_buf", None) is not None
        ):
            rk, rv, rg, rgam, ridx = initial_state.replay_buf
            return HSSDCacheState(
                step_count=initial_state.step_count,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                page_states=initial_state.page_states.to(device=device, dtype=dtype),
                theta_cum=initial_state.theta_cum,
                boundary_ring=None,
                swa_k_ring=initial_state.swa_k_ring.to(device=device, dtype=dtype),
                swa_v_ring=initial_state.swa_v_ring.to(device=device, dtype=dtype),
                swa_valid_ring=initial_state.swa_valid_ring,
                s_far_bank=initial_state.s_far_bank.to(device=device, dtype=dtype),
                s_chunk_start_bank=initial_state.s_chunk_start_bank.to(device=device, dtype=dtype),
                replay_buf=(
                    rk.to(device=device, dtype=dtype),
                    rv.to(device=device, dtype=dtype),
                    rg.to(device=device, dtype=dtype),
                    rgam.to(device=device, dtype=dtype),
                    ridx.to(device=device, dtype=torch.long),
                ),
                _uniform_step=getattr(initial_state, "_uniform_step", None),
            )
        return HSSDCacheState(
            step_count=initial_state.step_count,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            page_states=initial_state.page_states.to(device=device, dtype=dtype),
            theta_cum=initial_state.theta_cum,
            boundary_ring=tuple(t.to(device=device, dtype=dtype) for t in initial_state.boundary_ring),
            swa_k_ring=initial_state.swa_k_ring.to(device=device, dtype=dtype),
            swa_v_ring=initial_state.swa_v_ring.to(device=device, dtype=dtype),
            swa_valid_ring=initial_state.swa_valid_ring,
            _uniform_step=getattr(initial_state, "_uniform_step", None),
        )
    if isinstance(initial_state, torch.Tensor):
        return HSSDCacheState.from_tensor(
            initial_state,
            num_pages=num_pages,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
        ).to(device=device, dtype=dtype)
    raise TypeError(f"Unsupported initial_state type: {type(initial_state).__name__}")


def _cast_state_dtype(
    state: HSSDCacheState,
    state_dtype: str | torch.dtype | None,
    acc_dt: torch.dtype,
) -> HSSDCacheState:
    target_dt = _resolve_state_dtype(state_dtype, acc_dt)
    if state.dtype == target_dt and state.theta_cum.dtype == torch.float64 and state.has_compact_storage():
        return state
    return state.to(device=state.device, dtype=target_dt)


def _normalize_gamma_spec(
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    num_pages: int,
    dk: int,
    acc_dt: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Normalize `gamma` without materializing 5D `[B, T, M, H_v, d_k]`!

    Returns `(page_scale [M] | None, chan_or_full)`:
      * When `page_scale is not None`, `chan_or_full` is 4D `[B, T, H_v, d_k // 2 or d_k]`.
      * When `page_scale is None`, `chan_or_full` is legacy 5D `[B, T, M, H_v, d_k // 2 or d_k]`.
    """
    if isinstance(gamma, tuple):
        require(len(gamma) == 2, "factored gamma tuple must be (gamma_page_scale, gamma_chan_pair)")
        ps, chan = gamma
        ps_f = upcast(ps).to(device=device, dtype=acc_dt).clamp_min(0.0)
        require(ps_f.shape == (num_pages,), f"gamma_page_scale shape {tuple(ps_f.shape)} != ({num_pages},)")
        ch_f = upcast(chan).to(device=device, dtype=acc_dt).clamp_min(0.0)
        require(
            ch_f.ndim == 4 and ch_f.shape[-1] in (dk // 2, dk),
            "gamma_chan_pair must have shape [B, T, H_v, d_k // 2] or [B, T, H_v, d_k]",
            gamma_chan_pair=ch_f,
        )
        return ps_f, ch_f

    gf = upcast(gamma).to(device=device, dtype=acc_dt).clamp_min(0.0)
    if gf.ndim == 4:
        require(gf.shape[-1] in (dk // 2, dk), "4D gamma last dim must be d_k // 2 or d_k", gamma=gf)
        ones_p = torch.ones(num_pages, device=device, dtype=acc_dt)
        return ones_p, gf
    require(
        gf.ndim == 5 and gf.shape[2] == num_pages and gf.shape[-1] in (dk // 2, dk),
        "gamma must be factored (ps[M], chan[B,T,H,D//2]) or Tensor [B,T,H,D] / [B,T,M,H,D]",
        gamma=gf,
    )
    return None, gf


def _slice_gamma_input(
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    batch_sl: slice | None = None,
    time_sl: slice | torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    def _sl_tensor(t: torch.Tensor) -> torch.Tensor:
        out = t if batch_sl is None else t[batch_sl]
        if time_sl is None:
            return out
        if isinstance(time_sl, torch.Tensor):
            return out.index_select(1, time_sl)
        return out[:, time_sl]

    if isinstance(gamma, tuple):
        return (gamma[0], _sl_tensor(gamma[1]))
    return _sl_tensor(gamma)


def _materialize_chunk_gamma(
    page_scale: torch.Tensor | None,
    chan_c: torch.Tensor,
    *,
    num_pages: int,
    dk: int,
) -> torch.Tensor:
    """Expand `gamma` to `[B, c_len, M, H_v, d_k]` ONLY inside a single `c_len <= C` chunk."""
    if chan_c.shape[-1] == dk // 2:
        chan_c = chan_c.repeat_interleave(2, dim=-1)
    if page_scale is None:
        return chan_c
    return page_scale.view(1, 1, num_pages, 1, 1) * chan_c.unsqueeze(2)


_WY_BLOCK_META_CACHE: dict[
    tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_wy_block_meta(
    n_blk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(strict_lo_8, bi_idx, bj_idx, all_lin_idx)` on `device` without inference-tensor taint."""
    if _is_compiling():
        idx8 = torch.arange(8, device=device)
        strict_lo_8 = idx8[:, None] > idx8[None, :]
        diag_lin = torch.arange(n_blk, device=device, dtype=torch.long) * (n_blk + 1)
        if n_blk >= 2:
            tril = torch.tril_indices(n_blk, n_blk, offset=-1, device=device)
            bi_idx = tril[0]
            bj_idx = tril[1]
            all_lin_idx = torch.cat([diag_lin, bi_idx * n_blk + bj_idx], dim=0)
        else:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            bi_idx, bj_idx, all_lin_idx = empty_l, empty_l, diag_lin
        return strict_lo_8, bi_idx, bj_idx, all_lin_idx

    key = (device, n_blk)
    cached = _WY_BLOCK_META_CACHE.get(key)
    if cached is not None:
        return cached
    with torch.inference_mode(False), torch.no_grad():
        idx8 = torch.arange(8, device=device)
        strict_lo_8 = idx8[:, None] > idx8[None, :]
        diag_lin = torch.arange(n_blk, device=device, dtype=torch.long) * (n_blk + 1)
        if n_blk >= 2:
            tril = torch.tril_indices(n_blk, n_blk, offset=-1, device=device)
            bi_idx = tril[0]
            bj_idx = tril[1]
            all_lin_idx = torch.cat([diag_lin, bi_idx * n_blk + bj_idx], dim=0)
        else:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            bi_idx, bj_idx, all_lin_idx = empty_l, empty_l, diag_lin
    res = (strict_lo_8, bi_idx, bj_idx, all_lin_idx)
    _WY_BLOCK_META_CACHE[key] = res
    return res


def _build_wy_unit_lower(
    k_eff: torch.Tensor,
    log_a_neg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build overflow-free Block-8 strictly-lower WY interaction matrix `L` and `cum_log_a` (`[..., c_len, c_len]`)."""
    c_len = k_eff.shape[-2]
    b_sz = 8
    n_blk = (c_len + b_sz - 1) // b_sz
    c_pad = n_blk * b_sz
    pad_t = c_pad - c_len
    dk = k_eff.shape[-1]
    lead_shape = k_eff.shape[:-2]

    if pad_t > 0:
        k_pad = F.pad(k_eff, (0, 0, 0, pad_t), value=0.0)
        log_a_pad = F.pad(log_a_neg, (0, 0, 0, pad_t), value=0.0)
    else:
        k_pad = k_eff
        log_a_pad = log_a_neg

    cum_log_pad = torch.cumsum(log_a_pad, dim=-2)
    cum_log_a = cum_log_pad[..., :c_len, :] if pad_t > 0 else cum_log_pad

    k_blk = k_pad.reshape((*lead_shape, n_blk, b_sz, dk))
    g_blk = cum_log_pad.reshape((*lead_shape, n_blk, b_sz, dk))
    strict_lo_8, bi_idx, bj_idx, all_lin_idx = _get_wy_block_meta(n_blk, k_eff.device)

    diff_d = g_blk[..., :, None, :] - g_blk[..., None, :, :]
    safe_d = torch.where(strict_lo_8.unsqueeze(-1), diff_d, torch.zeros_like(diff_d))
    diag_blks = (k_blk[..., :, None, :] * k_blk[..., None, :, :] * safe_d.exp()).sum(dim=-1)
    diag_blks = torch.where(strict_lo_8, diag_blks, torch.zeros_like(diag_blks))

    if n_blk == 1:
        l_mat = diag_blks.squeeze(-3)[..., :c_len, :c_len]
    else:
        rhs_all = k_blk[..., :-1, :, :] * (g_blk[..., :-1, 7:8, :] - g_blk[..., :-1, :, :]).exp()
        rhs_k = rhs_all.index_select(-3, bj_idx)
        k_i_pairs = k_blk.index_select(-3, bi_idx)
        g_i_pairs = g_blk.index_select(-3, bi_idx)
        g_piv_pairs = g_blk[..., 7:8, :].index_select(-3, bj_idx)
        lhs = k_i_pairs * (g_i_pairs - g_piv_pairs).exp()
        off_blks = lhs @ rhs_k.transpose(-1, -2)

        active_blks = torch.cat([diag_blks, off_blks], dim=-3)
        n_act = active_blks.shape[-3]
        idx_exp = all_lin_idx.view((*(1,) * len(lead_shape), n_act, 1, 1)).expand(
            (*lead_shape, n_act, b_sz, b_sz)
        )
        grid = k_eff.new_zeros((*lead_shape, n_blk * n_blk, b_sz, b_sz)).scatter(-3, idx_exp, active_blks)
        l_pad = grid.view((*lead_shape, n_blk, n_blk, b_sz, b_sz)).transpose(-3, -2).reshape(
            (*lead_shape, c_pad, c_pad)
        )
        l_mat = l_pad[..., :c_len, :c_len] if pad_t > 0 else l_pad
    return l_mat, cum_log_a


def _solve_unit_lower_wy(l_mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve `(I + L) X = rhs` for unit-lower-triangular `I + L` via recursive block-doubling.

    For `C <= 64`, pads `C` to the next power-of-2 in `{2, 4, 8, 16, 32, 64}` and inverts `I + L`
    in `log2(C_pad)` batched Tensor-Core matmul stages (`b = 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64`):
      * Base `2x2` diagonal blocks: `(I + [[0, 0], [l_10, 0]])^{-1} = [[1, 0], [-l_10, 1]]` (0 FLOPs).
      * Doubling step (`b -> 2b`): `inv_10 = -(inv_11 @ l_10) @ inv_00`, assembling
        `[[inv_00, 0], [inv_10, inv_11]]` without raising `L` to high powers (immune to `fp32`
        cancellation even when over-relaxed `b_t in (0, 2)` makes `||L|| > 1`).
    Falls back to `torch.linalg.solve_triangular` only when `C > 64`.
    """
    c_len = l_mat.shape[-1]
    if c_len == 0:
        return rhs
    if c_len == 1:
        return rhs + l_mat * 0.0 if l_mat.requires_grad else rhs
    if c_len == 2:
        u0 = rhs[..., 0:1, :]
        u1 = rhs[..., 1:2, :] - l_mat[..., 1:2, 0:1] * u0
        return torch.cat([u0, u1], dim=-2)
    if c_len > 64:
        eye = torch.eye(c_len, dtype=l_mat.dtype, device=l_mat.device)
        return torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)

    if (
        _HAS_TRITON
        and l_mat.is_cuda
        and not _is_compiling()
        and c_len == 64
        and l_mat.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and rhs.dtype == l_mat.dtype
    ):
        return _TritonSolveUnitLower64Fn.apply(l_mat, rhs)

    c_pad = 1 << (c_len - 1).bit_length()
    pad_t = c_pad - c_len
    l_pad = F.pad(l_mat, (0, pad_t, 0, pad_t)) if pad_t > 0 else l_mat
    r_pad = F.pad(rhs, (0, 0, 0, pad_t)) if pad_t > 0 else rhs
    lead_shape = l_pad.shape[:-2]
    half_pad = c_pad // 2

    l_base_10 = l_pad.reshape(*lead_shape, half_pad, 2, half_pad, 2).diagonal(dim1=-4, dim2=-2)[..., 1, 0, :]
    z = torch.zeros_like(l_base_10)
    o = torch.ones_like(l_base_10)
    inv_diag = torch.stack(
        [
            torch.stack([o, z], dim=-1),
            torch.stack([-l_base_10, o], dim=-1),
        ],
        dim=-2,
    )

    b = 2
    while b < half_pad:
        half_blk = c_pad // (2 * b)
        inv_00 = inv_diag[..., 0::2, :, :]
        inv_11 = inv_diag[..., 1::2, :, :]
        l_10 = (
            l_pad.reshape(*lead_shape, half_blk, 2 * b, half_blk, 2 * b)
            .diagonal(dim1=-4, dim2=-2)[..., b:, :b, :]
            .movedim(-1, -3)
        )
        inv_10 = -(inv_11 @ l_10) @ inv_00
        zeros_01 = torch.zeros_like(inv_10)
        top = torch.cat([inv_00, zeros_01], dim=-1)
        bot = torch.cat([inv_10, inv_11], dim=-1)
        inv_diag = torch.cat([top, bot], dim=-2)
        b *= 2

    inv_00 = inv_diag[..., 0, :, :]
    inv_11 = inv_diag[..., 1, :, :]
    l_half = l_pad[..., half_pad:, :half_pad]
    r_0 = r_pad[..., :half_pad, :]
    r_1 = r_pad[..., half_pad:, :]
    u_0 = inv_00 @ r_0
    u_1 = inv_11 @ (r_1 - l_half @ u_0)
    u_pad = torch.cat([u_0, u_1], dim=-2)
    return u_pad[..., :c_len, :] if pad_t > 0 else u_pad


class _ActivePagePlan:
    """Precomputed active `(b, m)` page compaction metadata (reusable across surprise readout & WY write)."""

    __slots__ = ("_tok_pos", "_valid_mask", "act_flat", "b_i", "lin_bm", "m_i", "max_tok", "n_act")

    def __init__(
        self,
        lin_bm: torch.Tensor | None,
        act_flat: torch.Tensor | None,
        n_act: int,
        max_tok: int,
        num_pages: int,
    ) -> None:
        self.lin_bm = lin_bm
        if lin_bm is not None:
            self.b_i: torch.Tensor | None = torch.div(lin_bm, num_pages, rounding_mode="floor")
            self.m_i: torch.Tensor | None = lin_bm - self.b_i * num_pages
        else:
            self.b_i = None
            self.m_i = None
        self.act_flat = act_flat
        self.n_act = int(n_act)
        self.max_tok = int(max_tok)
        self._tok_pos: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def compaction_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(tok_pos, valid_mask)` of shape `[N_act, max_tok]` on device (computed once & cached)."""
        if self._tok_pos is None or self._valid_mask is None:
            assert self.lin_bm is not None and self.act_flat is not None
            c_len = self.act_flat.shape[-1]
            act_sub = self.act_flat.index_select(0, self.lin_bm)
            t_arange = torch.arange(c_len, device=act_sub.device).unsqueeze(0)
            sort_key = (~act_sub).to(dtype=torch.long) * c_len + t_arange
            tok_pos = torch.argsort(sort_key, dim=-1)[:, : self.max_tok]
            valid_mask = act_sub.gather(1, tok_pos)
            self._tok_pos = tok_pos
            self._valid_mask = valid_mask
        return self._tok_pos, self._valid_mask


def _build_active_page_plan(act_bm: torch.Tensor) -> _ActivePagePlan:
    """Build active `(b, m)` compaction plan from `act_bm` (`[B, M, c_len]` bool) in 1 tiny `.tolist()` transfer."""
    bsz, num_pages, c_len = act_bm.shape
    tot_bm = bsz * num_pages
    if tot_bm == 0 or c_len == 0:
        return _ActivePagePlan(None, None, 0, 0, max(num_pages, 1))
    act_flat = act_bm.reshape(tot_bm, c_len)
    cnt_flat: list[int] = act_flat.sum(dim=-1, dtype=torch.int32).tolist()
    active_lin: list[int] = []
    max_tok = 0
    for idx, cnt in enumerate(cnt_flat):
        if cnt > 0:
            active_lin.append(idx)
            max_tok = max(max_tok, cnt)
    n_act = len(active_lin)
    if n_act == 0 or (n_act == tot_bm and max_tok >= c_len):
        return _ActivePagePlan(None, None, n_act, max_tok, num_pages)
    lin_bm = torch.tensor(active_lin, dtype=torch.long, device=act_bm.device)
    return _ActivePagePlan(lin_bm, act_flat, n_act, max_tok, num_pages)


def _build_multi_span_active_plans(
    top_idx_spans: Sequence[torch.Tensor],
    num_pages: int,
) -> list[_ActivePagePlan]:
    """Batch-build `_ActivePagePlan`s across multiple `[B, c_len_s, k]` spans in 1 single `.tolist()` transfer."""
    n_spans = len(top_idx_spans)
    if n_spans == 0:
        return []
    device = top_idx_spans[0].device
    bsz = top_idx_spans[0].shape[0]
    tot_bm = bsz * num_pages
    if tot_bm == 0:
        return [_ActivePagePlan(None, None, 0, 0, max(num_pages, 1)) for _ in top_idx_spans]

    valid_pos: list[int] = []
    act_flats: list[torch.Tensor] = []
    cnt_tensors: list[torch.Tensor] = []
    span_lens: list[int] = []
    for s_i, t_idx in enumerate(top_idx_spans):
        c_len = int(t_idx.shape[1])
        if c_len <= 1:
            continue
        act_flat = (
            torch.zeros((bsz, c_len, num_pages), dtype=torch.bool, device=device)
            .scatter_(2, t_idx, True)
            .transpose(1, 2)
            .reshape(tot_bm, c_len)
        )
        valid_pos.append(s_i)
        act_flats.append(act_flat)
        cnt_tensors.append(act_flat.sum(dim=-1, dtype=torch.int32))
        span_lens.append(c_len)

    out_plans: list[_ActivePagePlan] = [
        _ActivePagePlan(None, None, 0, 0, num_pages) for _ in range(n_spans)
    ]
    if not cnt_tensors:
        return out_plans

    cnt_stack = cnt_tensors[0].unsqueeze(0) if len(cnt_tensors) == 1 else torch.stack(cnt_tensors, dim=0)
    all_cnts: list[list[int]] = cnt_stack.tolist()
    flat_active: list[int] = []
    specs: list[tuple[int, int, int, int]] = []
    for j, row in enumerate(all_cnts):
        c_len = span_lens[j]
        start = len(flat_active)
        max_tok = 0
        for idx, cnt in enumerate(row):
            if cnt > 0:
                flat_active.append(idx)
                max_tok = max(max_tok, cnt)
        n_act = len(flat_active) - start
        if n_act == 0 or (n_act == tot_bm and max_tok >= c_len):
            del flat_active[start:]
            specs.append((0, 0, n_act, max_tok))
        else:
            specs.append((start, n_act, n_act, max_tok))

    all_lin_t = torch.tensor(flat_active, dtype=torch.long, device=device) if flat_active else None
    for j, s_i in enumerate(valid_pos):
        offset, length, n_act, max_tok = specs[j]
        if length == 0 or all_lin_t is None:
            out_plans[s_i] = _ActivePagePlan(None, None, n_act, max_tok, num_pages)
        else:
            out_plans[s_i] = _ActivePagePlan(
                all_lin_t[offset : offset + length],
                act_flats[j],
                n_act,
                max_tok,
                num_pages,
            )
    return out_plans


def _wy_kda_state_update(
    page_states: torch.Tensor,
    k_eff: torch.Tensor,
    v_eff: torch.Tensor,
    log_a: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    compact_active_pages: bool = True,
    active_plan: _ActivePagePlan | None = None,
) -> torch.Tensor:
    """Exact overflow-free Block-8 WY state transition for DPLR KDA with optional active compaction (`5.2 #4`).

    When `active_plan` or `active_mask` (`[B, M, c_len]` bool) is provided and `compact_active_pages=True`,
    skips `(b, m)` slices with zero writes (`N_act <= B * M`) and compacts active write tokens per `(b, m)`
    in chronological order (`C_max <= c_len`), preserving exact `fp64` forward & backward equivalence
    since unselected tokens satisfy `g == 0 => alpha == 1, k_eff == 0, v_eff == 0` (exact identity).
    When `active_mask is None` and `active_plan is None`, executes zero-host-sync dense Block-8 WY.
    """
    c_len = k_eff.shape[-2]
    if (
        (active_plan is not None or active_mask is not None)
        and compact_active_pages
        and c_len > 1
        and not _is_compiling()
    ):
        bsz, num_pages, hv, dk, dv = page_states.shape
        if active_plan is not None:
            plan = active_plan
        else:
            assert active_mask is not None
            plan = _build_active_page_plan(active_mask)
        if plan.n_act == 0:
            return page_states
        if plan.lin_bm is not None:
            lin_bm = plan.lin_bm
            n_act = plan.n_act
            c_max = plan.max_tok
            tok_pos, valid_c = plan.compaction_indices()

            p_flat = page_states.reshape(bsz * num_pages, hv, dk, dv)
            p_sub = p_flat.index_select(0, lin_bm).unsqueeze(1)               # [N_act, 1, H_v, d_k, d_v]

            tp_k = tok_pos[:, None, None, :, None].expand(n_act, 1, hv, c_max, dk)
            tp_v = tok_pos[:, None, None, :, None].expand(n_act, 1, hv, c_max, dv)
            ke_sub = k_eff.reshape(bsz * num_pages, 1, hv, c_len, dk).index_select(0, lin_bm).gather(-2, tp_k)
            ve_sub = v_eff.reshape(bsz * num_pages, 1, hv, c_len, dv).index_select(0, lin_bm).gather(-2, tp_v)
            la_sub = log_a.reshape(bsz * num_pages, 1, hv, c_len, dk).index_select(0, lin_bm).gather(-2, tp_k)

            vm5 = valid_c[:, None, None, :, None]
            ke_sub = torch.where(vm5, ke_sub, torch.zeros_like(ke_sub))
            ve_sub = torch.where(vm5, ve_sub, torch.zeros_like(ve_sub))
            la_sub = torch.where(vm5, la_sub, torch.zeros_like(la_sub))

            p_upd = _wy_kda_state_update(
                p_sub,
                ke_sub,
                ve_sub,
                la_sub,
                active_mask=None,
                compact_active_pages=False,
            )
            idx_scatter = lin_bm[:, None, None, None].expand(n_act, hv, dk, dv)
            return p_flat.scatter(0, idx_scatter, p_upd.squeeze(1)).view(bsz, num_pages, hv, dk, dv)

    log_a_neg = log_a.clamp(max=0.0)
    if c_len == 1:
        dec = log_a_neg.squeeze(-2).exp().unsqueeze(-1) * page_states
        ke = k_eff.squeeze(-2)
        ve = v_eff.squeeze(-2)
        mem = (ke.unsqueeze(-2) @ dec).squeeze(-2)
        return dec + ke.unsqueeze(-1) * (ve - mem).unsqueeze(-2)

    l_mat, cum_log_a = _build_wy_unit_lower(k_eff, log_a_neg)
    gamma_from_0 = cum_log_a.exp()
    rhs = v_eff - (gamma_from_0 * k_eff) @ page_states
    u_wy = _solve_unit_lower_wy(l_mat, rhs)

    gamma_to_end = (cum_log_a[..., -1:, :] - cum_log_a).exp()
    d_chunk = cum_log_a[..., -1, :].exp().unsqueeze(-1)
    return d_chunk * page_states + (gamma_to_end * k_eff).transpose(-1, -2) @ u_wy


def _routed_bank_readout(
    vec_c: torch.Tensor,
    state_bank: torch.Tensor,
    top_idx: torch.Tensor,
    top_w: torch.Tensor,
    *,
    compact_active_pages: bool = True,
    active_plan: _ActivePagePlan | None = None,
) -> torch.Tensor:
    """Read `vec_c` (`[B, c_len, H_v, d_k]`) from `top_idx` (`[B, c_len, k]`) of `state_bank`.

    * Decode / short spans (`c_len <= 4`) when `k < M`: gathers ONLY the `k` active pages in
      `state_bank.dtype` and casts just those `k` pages before matmul (`4x` fewer state bytes & FLOPs).
    * Chunk spans (`c_len > 4`) when `compact_active_pages=True` and `k < M` (`5.2 #4`): skips
      untouched `(b, m)` pages (`N_act <= B * M`) and compacts active query tokens per `(b, m)` to
      `R_max <= c_len` using differentiable out-of-place `index_select` / `scatter_add`.
    """
    bsz, c_len, hv, dk = vec_c.shape
    num_pages = state_bank.shape[1]
    dv = state_bank.shape[-1]
    k_sel = top_idx.shape[-1]

    if k_sel < num_pages and c_len == 1:
        g_idx = top_idx[:, 0, :, None, None, None].expand(bsz, k_sel, hv, dk, dv)
        s_sel = torch.gather(state_bank, dim=1, index=g_idx)               # [B, k, H_v, d_k, d_v]
        if s_sel.dtype != vec_c.dtype:
            s_sel = s_sel.to(dtype=vec_c.dtype)
        v_q = vec_c.permute(0, 2, 1, 3).unsqueeze(1)                       # [B, 1, H_v, 1, d_k]
        picked_1 = (v_q @ s_sel).squeeze(-2)                               # [B, k, H_v, d_v]
        return (picked_1 * top_w[:, 0, :, None, None]).sum(dim=1, keepdim=True)

    if k_sel < num_pages and c_len <= 4:
        flat_idx = top_idx.reshape(bsz, c_len * k_sel, 1, 1, 1).expand(bsz, c_len * k_sel, hv, dk, dv)
        s_sel = torch.gather(state_bank, dim=1, index=flat_idx).view(bsz, c_len, k_sel, hv, dk, dv)
        if s_sel.dtype != vec_c.dtype:
            s_sel = s_sel.to(dtype=vec_c.dtype)
        v_q = vec_c.unsqueeze(2).unsqueeze(-2)                             # [B, c_len, 1, H_v, 1, d_k]
        picked_s = (v_q @ s_sel).squeeze(-2)                               # [B, c_len, k, H_v, d_v]
        return (picked_s * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

    if compact_active_pages and k_sel < num_pages and not _is_compiling():
        if active_plan is not None:
            plan = active_plan
        else:
            act_bm = (
                torch.zeros((bsz, c_len, num_pages), dtype=torch.bool, device=vec_c.device)
                .scatter_(2, top_idx, True)
                .transpose(1, 2)
            )                                                              # [B, M, c_len]
            plan = _build_active_page_plan(act_bm)
        if plan.n_act == 0:
            return vec_c.new_zeros(bsz, c_len, hv, dv)
        if plan.lin_bm is not None:
            assert plan.b_i is not None
            lin_bm = plan.lin_bm
            b_i = plan.b_i
            n_act = plan.n_act
            r_max = plan.max_tok
            tok_pos, valid_r = plan.compaction_indices()
            w_bm = (
                vec_c.new_zeros((bsz, c_len, num_pages))
                .scatter_add(2, top_idx, top_w)
                .transpose(1, 2)
                .reshape(bsz * num_pages, c_len)
            )

            lin_bt = b_i.unsqueeze(1) * c_len + tok_pos                    # [N_act, r_max]
            vec_flat = vec_c.reshape(bsz * c_len, hv, dk)
            q_sub = vec_flat.index_select(0, lin_bt.reshape(-1)).view(n_act, r_max, hv, dk).transpose(1, 2)

            bank_flat = state_bank.reshape(bsz * num_pages, hv, dk, dv)
            s_sub = bank_flat.index_select(0, lin_bm)
            if s_sub.dtype != vec_c.dtype:
                s_sub = s_sub.to(dtype=vec_c.dtype)

            out_sub = (q_sub @ s_sub).transpose(1, 2)                      # [N_act, r_max, H_v, d_v]
            w_sub = w_bm.index_select(0, lin_bm).gather(1, tok_pos)
            w_masked = torch.where(valid_r, w_sub, torch.zeros_like(w_sub))
            weighted = (out_sub * w_masked.unsqueeze(-1).unsqueeze(-1)).reshape(n_act * r_max, hv, dv)
            out_flat = vec_c.new_zeros((bsz * c_len, hv, dv)).scatter_add(
                0, lin_bt.reshape(-1, 1, 1).expand(n_act * r_max, hv, dv), weighted
            )
            return out_flat.view(bsz, c_len, hv, dv)

    bank_f = state_bank if state_bank.dtype == vec_c.dtype else state_bank.to(dtype=vec_c.dtype)
    v_all = vec_c.permute(0, 2, 1, 3).unsqueeze(1)                         # [B, 1, H_v, c_len, d_k]
    all_out = (v_all @ bank_f).permute(0, 3, 1, 2, 4)                      # [B, c_len, M, H_v, d_v]
    gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, c_len, k_sel, hv, dv)
    picked = torch.gather(all_out, dim=2, index=gather_idx)
    return (picked * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)


def _far_readout(
    q_hat_c: torch.Tensor,
    s_far: torch.Tensor,
    read_logits_c: torch.Tensor,
    top_k_pages: int,
    *,
    compact_active_pages: bool = True,
    precomputed_routing: tuple[torch.Tensor, torch.Tensor] | None = None,
    active_plan: _ActivePagePlan | None = None,
) -> torch.Tensor:
    """Top-`k` page-routed readout from `S_boundary[k_far]` (`[B, c_len, H_v, d_v]`)."""
    if precomputed_routing is not None:
        top_idx, top_w = precomputed_routing
    else:
        pi_read = F.softmax(read_logits_c, dim=-1)
        top_vals, top_idx = torch.topk(pi_read, k=top_k_pages, dim=-1)
        top_w = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return _routed_bank_readout(
        q_hat_c,
        s_far,
        top_idx,
        top_w,
        compact_active_pages=compact_active_pages,
        active_plan=active_plan,
    )


def _compute_root_surprise_and_routing(
    k_hat_c: torch.Tensor,
    v_c: torch.Tensor,
    s_chunk_start_bank: torch.Tensor,
    sbias_c: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits_c: torch.Tensor,
    top_k_pages: int,
    wmask_c: torch.Tensor | None,
    g_override_c: torch.Tensor | None,
    *,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    write_mode: str = "full",
    compact_active_pages: bool = True,
    precomputed_write_routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_plan: _ActivePagePlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(g_page, u_sum, valid_toks, top_idx, g_active)` (`H1`, `H6`, `Bug 3C`).

    * When `write_mode == "full"` (default): writes at full strength `1.0` to each selected page in
      `top_idx` using STE (`ste_active = 1 + (top_pi - top_pi.detach())`), eliminating `sqrt(pi)`
      read-write attenuation (`0.25*v1 + 0.5*v2` -> exact `v2`) while keeping `pred_v` normalized
      (`top_probs.sum(-1) == 1.0`) and accounting exact full-strength write mass `u_sum` (`pi_ste * g^2`).
    * When `write_mode == "split"`: legacy `sqrt(top_probs)` split-strength writes.
    """
    bsz, c_len, hv, _dk = k_hat_c.shape
    if precomputed_write_routing is not None:
        pi_write, top_vals, top_idx = precomputed_write_routing
    else:
        pi_write = F.softmax(write_logits_c, dim=-1)
        top_vals, top_idx = torch.topk(pi_write, k=top_k_pages, dim=-1)
    top_probs = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    if g_override_c is not None:
        g_root = (
            g_override_c.squeeze(-1)
            if (g_override_c.ndim == 4 and g_override_c.shape[-1] == 1)
            else g_override_c
        )
        if g_root.shape != (bsz, c_len, hv):
            g_root = g_root.expand(bsz, c_len, hv)
    elif (not use_surprise) or surprise_mode == "none":
        g_root = torch.ones(bsz, c_len, hv, device=k_hat_c.device, dtype=k_hat_c.dtype)
    else:
        if surprise_mode in ("routed", "chunk_start", "segment_start") and s_chunk_start_bank.ndim == 5:
            pred_v = _routed_bank_readout(
                k_hat_c,
                s_chunk_start_bank,
                top_idx,
                top_probs,
                compact_active_pages=compact_active_pages,
                active_plan=active_plan,
            )
        else:
            s0 = s_chunk_start_bank[:, 0] if s_chunk_start_bank.ndim == 5 else s_chunk_start_bank
            if s0.dtype != k_hat_c.dtype:
                s0 = s0.to(dtype=k_hat_c.dtype)
            pred_v = (k_hat_c.transpose(1, 2) @ s0).transpose(1, 2)
        eps_err = _safe_norm(v_c - pred_v, dim=-1) / (_safe_norm(v_c, dim=-1) + 1e-6)
        slope = surprise_slope.view(1, 1, hv)
        thresh = surprise_thresh.view(1, 1, hv)
        sb_eff = sbias_c.squeeze(-1) if (sbias_c.ndim == 4 and sbias_c.shape[-1] == 1) else sbias_c
        g_root = torch.sigmoid(slope * (eps_err - thresh) + sb_eff)

    if wmask_c is not None:
        g_root = g_root * wmask_c.unsqueeze(-1)
        valid_toks = wmask_c.sum()
    else:
        valid_toks = g_root.new_full((), float(bsz * c_len))

    g_sq_mean = (g_root * g_root).mean(dim=-1, keepdim=True)
    if write_mode == "full":
        ste_active = torch.ones_like(top_vals) + (top_vals - top_vals.detach())
        g_active = ste_active.unsqueeze(-1) * g_root.unsqueeze(2)
        write_root = torch.zeros_like(pi_write).scatter(-1, top_idx, ste_active)
        g_page = write_root.unsqueeze(-1) * g_root.unsqueeze(2)

        hard_sel = torch.zeros_like(pi_write).scatter(-1, top_idx, 1.0)
        pi_ste = hard_sel + float(top_k_pages) * (pi_write - pi_write.detach())
        u_sum = (pi_ste * g_sq_mean).sum(dim=(0, 1))
    elif write_mode == "split":
        pi_actual = torch.zeros_like(pi_write).scatter(-1, top_idx, top_probs)
        pi_ste = pi_actual + (pi_write - pi_write.detach())
        u_sum = (pi_ste * g_sq_mean).sum(dim=(0, 1))

        top_roots = torch.sqrt(top_probs.clamp_min(1e-30))
        g_active = top_roots.unsqueeze(-1) * g_root.unsqueeze(2)
        write_root = torch.zeros_like(pi_write).scatter(-1, top_idx, top_roots)
        g_page = write_root.unsqueeze(-1) * g_root.unsqueeze(2)
    else:
        raise ValueError(f"Unsupported write_mode {write_mode!r}; expected one of {_VALID_WRITE_MODES}")

    return g_page, u_sum, valid_toks, top_idx, g_active


def compute_write_budget_loss(
    duty_vec: torch.Tensor,
    mu_max: torch.Tensor | Sequence[float],
    mu_min_ratio: float = 0.2,
    budget_weight: float = 1.0,
) -> torch.Tensor:
    """Two-sided cumulative sparse write-mass budget regularizer (`Fixes B2 & 3C`)."""
    if isinstance(mu_max, torch.Tensor):
        target_max = mu_max.to(device=duty_vec.device, dtype=duty_vec.dtype)
    else:
        target_max = torch.tensor(list(mu_max), device=duty_vec.device, dtype=duty_vec.dtype)
    target_min = float(mu_min_ratio) * target_max
    over = F.relu(duty_vec - target_max)
    under = F.relu(target_min - duty_vec)
    return float(budget_weight) * (over * over + under * under).sum()


def _validate_packed_boundaries(
    *,
    bsz: int,
    seq_len: int,
    initial_state: Any,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
) -> list[list[tuple[int, int]]] | None:
    """Validate `cu_seqlens` / `seq_idx` (`Bug 3E`) and return per-row document spans `[s0, s1)`."""
    if cu_seqlens is None and seq_idx is None:
        return None
    if cu_seqlens is not None and seq_idx is not None:
        raise ValueError("Specify at most one of `cu_seqlens` or `seq_idx`, not both")

    if cu_seqlens is not None:
        if bsz != 1:
            raise ValueError(f"cu_seqlens requires batch size B == 1, got B == {bsz}")
        if initial_state is not None:
            raise ValueError("cu_seqlens cannot be combined with a non-None initial_state")
        if isinstance(cu_seqlens, torch.Tensor):
            if cu_seqlens.ndim != 1:
                raise ValueError(f"cu_seqlens tensor must be 1D, got shape {tuple(cu_seqlens.shape)}")
            cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        elif isinstance(cu_seqlens, Sequence):
            cu = [int(x) for x in cu_seqlens]
        else:
            raise ValueError(f"Unsupported cu_seqlens type: {type(cu_seqlens).__name__}")
        if len(cu) < 2:
            raise ValueError(f"cu_seqlens must have length >= 2, got {cu}")
        if cu[0] != 0 or cu[-1] != seq_len:
            raise ValueError(f"cu_seqlens must start at 0 and end at seq_len={seq_len}, got {cu}")
        for i in range(len(cu) - 1):
            if cu[i] > cu[i + 1]:
                raise ValueError(f"cu_seqlens must be monotonic non-decreasing, got {cu}")
        spans = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1) if cu[i + 1] > cu[i]]
        return [spans]

    assert seq_idx is not None
    if not isinstance(seq_idx, torch.Tensor) or seq_idx.ndim != 2 or seq_idx.shape != (bsz, seq_len):
        got = tuple(seq_idx.shape) if isinstance(seq_idx, torch.Tensor) else type(seq_idx).__name__
        raise ValueError(f"seq_idx must be a 2D Tensor of shape {(bsz, seq_len)}, got {got}")
    if seq_idx.is_floating_point() or seq_idx.is_complex():
        raise ValueError(f"seq_idx must be an integer Tensor, got dtype {seq_idx.dtype}")
    s_cpu = seq_idx.detach().cpu()
    if seq_len > 0 and bool((s_cpu < 0).any()):
        raise ValueError("seq_idx entries must be non-negative")
    if seq_len > 1 and bool((s_cpu[:, 1:] < s_cpu[:, :-1]).any()):
        raise ValueError("seq_idx must be monotonic non-decreasing along dim=1")
    if initial_state is not None and seq_len > 0 and bool((s_cpu[:, 0] != s_cpu[:, -1]).any()):
        raise ValueError("multi-document seq_idx cannot be combined with a non-None initial_state")

    row_spans: list[list[tuple[int, int]]] = []
    for b in range(bsz):
        if seq_len == 0:
            row_spans.append([])
            continue
        _vals, counts = torch.unique_consecutive(s_cpu[b], return_counts=True)
        cur = 0
        b_list: list[tuple[int, int]] = []
        for cnt in counts.tolist():
            nxt = cur + int(cnt)
            b_list.append((cur, nxt))
            cur = nxt
        row_spans.append(b_list)
    return row_spans


def _pages_alias_boundary(st: HSSDCacheState) -> bool:
    if st.page_states.numel() <= 0:
        return False
    p_ptr = int(st.page_states.untyped_storage().data_ptr())
    if st._boundary_ring is not None:
        if any(b.numel() > 0 and int(b.untyped_storage().data_ptr()) == p_ptr for b in st._boundary_ring):
            return True
    else:
        if st._s_far_bank.numel() > 0 and int(st._s_far_bank.untyped_storage().data_ptr()) == p_ptr:
            return True
        if st._s_chunk_start_bank.numel() > 0 and int(st._s_chunk_start_bank.untyped_storage().data_ptr()) == p_ptr:
            return True
    return bool(
        st._s_seg_start_bank is not None
        and st._s_seg_start_bank.numel() > 0
        and int(st._s_seg_start_bank.untyped_storage().data_ptr()) == p_ptr
    )


def recurrent_hssd_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
    swa_recency_slope: torch.Tensor | None = None,
    top_k_pages: int = 2,
    segment_chunks: int = 4,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    write_mode: str = "full",
    compact_active_pages: bool = True,
    compact_decode_cache: bool = True,
    inplace_state: bool = False,
    return_duty_sum: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState] | tuple[torch.Tensor, HSSDCacheState, torch.Tensor, torch.Tensor]:
    """Single-token `O(1)`-memory HSSD v3.2 autoregressive decode step (`H5`, `H8`, `H9`, `H3`).

    Zero GPU->CPU host synchronizations on uniform-step decode (`_uniform_step`), single-eval
    `compute_2d_trig`, fixed `(W+1)*C`-slot masked SWA with optional `swa_recency_slope`, `O(k)`
    active-page update, and compact lagged-replay decode cache (`~2.5x-3.5x` smaller per layer).
    """
    squeeze_time = query.ndim == 3
    if squeeze_time:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        gamma = (gamma[0], gamma[1].unsqueeze(1)) if isinstance(gamma, tuple) else gamma.unsqueeze(1)
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
    bsz = query.shape[0]
    hv = value.shape[2]
    dk = key.shape[-1]
    dv = value.shape[-1]
    num_pages = state.page_states.shape[1]
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    target_dt = _resolve_state_dtype(state_dtype, acc_dt)

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, 1).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())

    u_step = getattr(state, "_uniform_step", None)
    if u_step is not None:
        inhomogeneous_steps = False
    else:
        inhomogeneous_steps = bsz > 1 and not bool((state.step_count == state.step_count[0]).all())

    if has_pad_zeros or inhomogeneous_steps:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_acc = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_acc = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = state[b : b + 1]
            if pm_bool is not None and not bool(pm_bool[b, 0]):
                row_outs.append(query.new_zeros(1, 1, hv, dv, dtype=init_dtype))
                row_states.append(_cast_state_dtype(st_b, target_dt, acc_dt))
                continue
            sl_b = slice(b, b + 1)
            o_b, nxt_b, u_b, c_b = recurrent_hssd_step(
                query[sl_b],
                key[sl_b],
                value[sl_b],
                _slice_gamma_input(gamma, batch_sl=sl_b),
                b_gate[sl_b],
                w_gate[sl_b],
                theta[sl_b],
                surprise_bias[sl_b],
                surprise_slope,
                surprise_thresh,
                write_logits[sl_b],
                read_logits[sl_b],
                q_swa[sl_b],
                k_swa[sl_b],
                v_swa[sl_b],
                st_b,
                boltzmann_gate=boltzmann_gate,
                swa_recency_slope=swa_recency_slope,
                top_k_pages=top_k_pages,
                segment_chunks=segment_chunks,
                write_mask=None if write_mask is None else write_mask.reshape(bsz, 1)[sl_b],
                pad_mask=None,
                g_override=None if g_override is None else g_override[sl_b],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=target_dt,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_active_pages=compact_active_pages,
                compact_decode_cache=compact_decode_cache,
                inplace_state=False,
                return_duty_sum=True,
            )
            row_outs.append(o_b)
            row_states.append(nxt_b)
            u_acc = u_acc + u_b
            cnt_acc = cnt_acc + c_b
        out_cat = torch.cat(row_outs, dim=0)
        st_cat = torch.cat(row_states, dim=0)
        out_ret = out_cat.squeeze(1) if squeeze_time else out_cat
        if return_duty_sum:
            return out_ret, st_cat, u_acc, cnt_acc
        return out_ret, st_cat

    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    st = _coerce_init_state(
        state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=int(state.chunk_size),
        window_chunks=int(state.window_chunks),
        device=q.device,
        dtype=target_dt,
    )
    can_inplace = bool(inplace_state and not torch.is_grad_enabled() and st is state)

    ps_gamma, ch_gamma = _normalize_gamma_spec(
        gamma, num_pages=num_pages, dk=dk, acc_dt=acc_dt, device=q.device
    )

    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    qs = upcast(q_swa).to(acc_dt)
    ks = upcast(k_swa).to(acc_dt)
    vs = upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)

    th64 = theta.to(dtype=torch.float64)
    if write_mask is not None:
        wm = write_mask.reshape(bsz, 1).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_tok = wm.squeeze(1) > 0.5
        th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_tok = torch.ones(bsz, device=q.device, dtype=torch.bool)
    if not use_phase:
        th64 = torch.zeros_like(th64)

    if st._uniform_step is not None:
        idx0 = st._uniform_step
    else:
        idx0 = int(st.step_count[0].item()) if st.step_count.numel() > 0 else 0
        st._uniform_step = idx0

    c_size = st.chunk_size
    w_chunks = st.window_chunks
    s_ref_bank = st.s_seg_start_bank if surprise_mode == "segment_start" else st.s_chunk_start_bank

    if can_inplace:
        st.theta_cum.add_(th64.squeeze(1))
        torch.remainder(st.theta_cum, 2.0 * math.pi, out=st.theta_cum)
        theta_cum = st.theta_cum
    else:
        theta_cum = _compact(torch.remainder(st.theta_cum + th64.squeeze(1), 2.0 * math.pi))

    sc = (dk ** -0.5) if scale is None else float(scale)
    if use_phase:
        cos_t, sin_t = compute_2d_trig(-theta_cum.unsqueeze(1), q.dtype)
        q_hat = apply_2d_rotation_trig(q, cos_t, sin_t)
        k_hat = apply_2d_rotation_trig(k, cos_t, sin_t)
    else:
        q_hat = q
        k_hat = k
    if sc != 1.0:
        q_hat = q_hat * sc

    # 1. Disjoint far-horizon readout from S_boundary[max(0, n - W)] == st.s_far_bank (zero ring replay!)
    o_far = _far_readout(q_hat, st.s_far_bank, rl, top_k_pages, compact_active_pages=False)

    # 2. Update O(1) circular SWA ring buffer at slot = idx0 % ((W + 1) * C) and attend over fixed ring (H8 & H3)
    max_win = st.max_swa_len
    slot = idx0 % max_win
    ks_ring = ks if ks.dtype == target_dt else ks.to(dtype=target_dt)
    vs_ring = vs if vs.dtype == target_dt else vs.to(dtype=target_dt)

    if use_swa:
        _n_chunk, _k_far, far_cutoff = disjoint_horizon_bounds(idx0 + 1, c_size, w_chunks)
        ring_pos = torch.arange(max_win, device=q.device)
        age = (idx0 - ring_pos) % max_win                                     # [max_win], exact q_pos - k_pos
        in_horizon = age < (idx0 - far_cutoff + 1)                            # [max_win]

        # Build [B, H_v, max_win, d_k/d_v] in acc_dt WITHOUT ever allocating 4D (B, max_win, H_v, d_k) in fp32!
        k_prior_t = st.swa_k_ring.transpose(1, 2)
        v_prior_t = st.swa_v_ring.transpose(1, 2)
        k_win = k_prior_t.clone() if k_prior_t.dtype == acc_dt else k_prior_t.to(dtype=acc_dt)
        v_win = v_prior_t.clone() if v_prior_t.dtype == acc_dt else v_prior_t.to(dtype=acc_dt)
        k_win[:, :, slot : slot + 1, :] = ks.transpose(1, 2)
        v_win[:, :, slot : slot + 1, :] = vs.transpose(1, 2)

        if can_inplace:
            st.swa_k_ring[:, slot : slot + 1] = ks_ring
            st.swa_v_ring[:, slot : slot + 1] = vs_ring
            st.swa_valid_ring[:, slot : slot + 1] = valid_tok.unsqueeze(1)
            swa_k_ring = st.swa_k_ring
            swa_v_ring = st.swa_v_ring
            swa_valid_ring = st.swa_valid_ring
        else:
            swa_k_ring = st.swa_k_ring.clone()
            swa_k_ring[:, slot : slot + 1] = ks_ring
            swa_v_ring = st.swa_v_ring.clone()
            swa_v_ring[:, slot : slot + 1] = vs_ring
            swa_valid_ring = st.swa_valid_ring.clone()
            swa_valid_ring[:, slot : slot + 1] = valid_tok.unsqueeze(1)

        val_slots = swa_valid_ring & in_horizon.unsqueeze(0)                  # [B, max_win]
        val_win = val_slots[:, None, None, :]                                 # [B, 1, 1, max_win]
        v_win = torch.where(val_slots[:, None, :, None], v_win, torch.zeros_like(v_win))

        q_loc = qs.transpose(1, 2)                                            # [B, H_v, 1, d_k]
        scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)             # [B, H_v, 1, max_win]
        if swa_recency_slope is not None:
            slp_swa = upcast(swa_recency_slope).to(device=q.device, dtype=acc_dt).view(1, hv, 1, 1)
            dist_norm = age.to(dtype=acc_dt).view(1, 1, 1, max_win) * (1.0 / float(c_size))
            scores = scores - slp_swa * dist_norm

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
    else:
        if can_inplace:
            st.swa_k_ring[:, slot : slot + 1] = ks_ring
            st.swa_v_ring[:, slot : slot + 1] = vs_ring
            st.swa_valid_ring[:, slot : slot + 1] = valid_tok.unsqueeze(1)
            swa_k_ring = st.swa_k_ring
            swa_v_ring = st.swa_v_ring
            swa_valid_ring = st.swa_valid_ring
        else:
            swa_k_ring = st.swa_k_ring.clone()
            swa_k_ring[:, slot : slot + 1] = ks_ring
            swa_v_ring = st.swa_v_ring.clone()
            swa_v_ring[:, slot : slot + 1] = vs_ring
            swa_valid_ring = st.swa_valid_ring.clone()
            swa_valid_ring[:, slot : slot + 1] = valid_tok.unsqueeze(1)
        out = o_far

    # 3. Routed root-surprise `s = g^2` & `O(k)` active-page state update (H1, H6, H8)
    _g_page, u_sum, valid_cnt, top_idx, g_active = _compute_root_surprise_and_routing(
        k_hat,
        v,
        s_ref_bank,
        sb,
        slp,
        thr,
        wl,
        top_k_pages,
        wm,
        go,
        use_surprise=use_surprise,
        surprise_mode=("routed" if surprise_mode == "segment_start" else surprise_mode),
        write_mode=write_mode,
        compact_active_pages=False,
    )
    sqrt_b = _safe_sqrt(bf[:, 0])
    sqrt_w = _safe_sqrt(wf[:, 0])
    g_sel = g_active[:, 0]                                                    # [B, k, H_v]
    t_idx = top_idx[:, 0]                                                     # [B, k]
    base_pages = st.page_states
    p_idx = t_idx[:, :, None, None, None].expand(bsz, top_k_pages, hv, dk, dv)
    s_sel = torch.gather(base_pages, dim=1, index=p_idx)                      # [B, k, H_v, d_k, d_v]
    if s_sel.dtype != acc_dt:
        s_sel = s_sel.to(dtype=acc_dt)

    if ps_gamma is not None:
        ch0 = ch_gamma[:, 0]                                                  # [B, H_v, d_g]
        d_g = ch0.shape[-1]
        ps_sel = ps_gamma.index_select(0, t_idx.reshape(-1)).view(bsz, top_k_pages, 1, 1)
        gam_sel_raw = ps_sel * ch0.unsqueeze(1)                               # [B, k, H_v, d_g]
        gam_sel = gam_sel_raw.repeat_interleave(2, dim=-1) if d_g == dk // 2 else gam_sel_raw
    else:
        ch5 = ch_gamma[:, 0]                                                  # [B, M, H_v, d_g]
        d_g = ch5.shape[-1]
        g_idx_gam = t_idx[:, :, None, None].expand(bsz, top_k_pages, hv, d_g)
        gam_sel_raw = torch.gather(ch5, dim=1, index=g_idx_gam)               # [B, k, H_v, d_g]
        gam_sel = gam_sel_raw.repeat_interleave(2, dim=-1) if d_g == dk // 2 else gam_sel_raw

    ke_base_0 = sqrt_b * k_hat[:, 0]                                          # [B, H_v, d_k]
    ve_base_0 = sqrt_w * v[:, 0]                                              # [B, H_v, d_v]
    dec_sel = (-(g_sel * g_sel).unsqueeze(-1) * gam_sel).clamp(max=0.0).exp().unsqueeze(-1) * s_sel
    ke_sel = g_sel.unsqueeze(-1) * ke_base_0.unsqueeze(1)
    ve_sel = g_sel.unsqueeze(-1) * ve_base_0.unsqueeze(1)
    mem_sel = (ke_sel.unsqueeze(-2) @ dec_sel).squeeze(-2)
    updated_sel = dec_sel + ke_sel.unsqueeze(-1) * (ve_sel - mem_sel).unsqueeze(-2)
    updated_sel_cast = updated_sel if updated_sel.dtype == target_dt else updated_sel.to(dtype=target_dt)

    if can_inplace:
        if _pages_alias_boundary(st):
            st.page_states = st.page_states.clone().contiguous()
        st.page_states.scatter_(1, p_idx, updated_sel_cast)
        new_pages = st.page_states
    else:
        new_pages = _compact(base_pages.scatter(1, p_idx, updated_sel_cast))

    new_step_int = idx0 + 1
    seg_tokens = max(1, int(segment_chunks)) * c_size
    if surprise_mode == "segment_start":
        if new_step_int % seg_tokens == 0:
            if (
                can_inplace
                and st._s_seg_start_bank is not None
                and st._s_seg_start_bank.data_ptr() != new_pages.data_ptr()
                and st._s_seg_start_bank.data_ptr() != st._s_far_bank.data_ptr()
                and st._s_seg_start_bank.data_ptr() != st._s_chunk_start_bank.data_ptr()
            ):
                st._s_seg_start_bank.copy_(new_pages)
                new_seg_bank: torch.Tensor | None = st._s_seg_start_bank
            else:
                new_seg_bank = new_pages.clone().contiguous()
        else:
            new_seg_bank = st.s_seg_start_bank
    else:
        new_seg_bank = st._s_seg_start_bank

    seg_ptr = (
        int(new_seg_bank.untyped_storage().data_ptr())
        if (new_seg_bank is not None and new_seg_bank.numel() > 0)
        else -1
    )
    use_explicit_ring = (st._boundary_ring is not None) or (not compact_decode_cache)
    if use_explicit_ring:
        cur_ring = st._boundary_ring if st._boundary_ring is not None else st.boundary_ring
        if can_inplace:
            if new_step_int % c_size == 0:
                ring_ptrs = [int(t.untyped_storage().data_ptr()) for t in cur_ring]
                p_ptr = int(new_pages.untyped_storage().data_ptr())
                if len(set(ring_ptrs)) == len(cur_ring) and p_ptr not in ring_ptrs and seg_ptr not in ring_ptrs:
                    for r_j in range(w_chunks):
                        cur_ring[r_j].copy_(cur_ring[r_j + 1])
                    cur_ring[w_chunks].copy_(new_pages)
                    new_b_ring = cur_ring
                else:
                    new_b_ring = (*cur_ring[1:], new_pages.clone().contiguous())
            else:
                new_b_ring = cur_ring
            st.step_count.add_(1)
            st._uniform_step = new_step_int
            st._boundary_ring = new_b_ring
            st._s_far_bank = new_b_ring[0]
            st._s_chunk_start_bank = new_b_ring[-1]
            st._s_seg_start_bank = new_seg_bank
            st.replay_buf = None
            st._cached_mid_ring = None
            next_state = st
        else:
            new_b_ring = (*cur_ring[1:], new_pages.clone().contiguous()) if new_step_int % c_size == 0 else cur_ring
            next_state = HSSDCacheState(
                step_count=st.step_count + 1,
                chunk_size=c_size,
                window_chunks=w_chunks,
                page_states=new_pages,
                theta_cum=theta_cum,
                boundary_ring=new_b_ring,
                swa_k_ring=swa_k_ring,
                swa_v_ring=swa_v_ring,
                swa_valid_ring=swa_valid_ring,
                s_seg_start_bank=new_seg_bank,
                _uniform_step=new_step_int,
            )
    else:
        # Compact Lagged-Replay Decode Cache (H5)
        new_replay_buf: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        if w_chunks > 0:
            ke_tok = ke_base_0.unsqueeze(2)
            ve_tok = ve_base_0.unsqueeze(2)
            g_tok = g_sel.unsqueeze(1)
            gam_tok = gam_sel_raw.unsqueeze(1)
            ke_tok_c = ke_tok if ke_tok.dtype == target_dt else ke_tok.to(dtype=target_dt)
            ve_tok_c = ve_tok if ve_tok.dtype == target_dt else ve_tok.to(dtype=target_dt)
            g_tok_c = g_tok if g_tok.dtype == target_dt else g_tok.to(dtype=target_dt)
            gam_tok_c = gam_tok if gam_tok.dtype == target_dt else gam_tok.to(dtype=target_dt)
            idx_tok = t_idx.unsqueeze(1)

            if (
                st.replay_buf is None
                or st.replay_buf[2].shape[2] != top_k_pages
                or st.replay_buf[3].shape[-1] != d_g
            ):
                rk = torch.zeros((bsz, hv, max_win, dk), device=q.device, dtype=target_dt)
                rv = torch.zeros((bsz, hv, max_win, dv), device=q.device, dtype=target_dt)
                rg = torch.zeros((bsz, max_win, top_k_pages, hv), device=q.device, dtype=target_dt)
                rgam = torch.zeros((bsz, max_win, top_k_pages, hv, d_g), device=q.device, dtype=target_dt)
                ridx = torch.zeros((bsz, max_win, top_k_pages), device=q.device, dtype=torch.long)
            elif can_inplace:
                rk, rv, rg, rgam, ridx = st.replay_buf
            else:
                rk = st.replay_buf[0].clone()
                rv = st.replay_buf[1].clone()
                rg = st.replay_buf[2].clone()
                rgam = st.replay_buf[3].clone()
                ridx = st.replay_buf[4].clone()

            rk[:, :, slot : slot + 1, :] = ke_tok_c
            rv[:, :, slot : slot + 1, :] = ve_tok_c
            rg[:, slot : slot + 1, :, :] = g_tok_c
            rgam[:, slot : slot + 1, :, :, :] = gam_tok_c
            ridx[:, slot : slot + 1, :] = idx_tok
            new_replay_buf = (rk, rv, rg, rgam, ridx)
        else:
            new_replay_buf = None

        if can_inplace:
            if new_step_int % c_size == 0:
                n_cur = idx0 // c_size
                p_ptr = int(new_pages.untyped_storage().data_ptr())
                far_ptr = int(st._s_far_bank.untyped_storage().data_ptr())
                cs_ptr = int(st._s_chunk_start_bank.untyped_storage().data_ptr())
                far_distinct = far_ptr not in (p_ptr, cs_ptr, seg_ptr)
                cs_distinct = cs_ptr not in (p_ptr, far_ptr, seg_ptr)
                if w_chunks == 0:
                    if far_distinct:
                        st._s_far_bank.copy_(new_pages)
                    else:
                        st._s_far_bank = new_pages.clone().contiguous()
                elif n_cur >= w_chunks:
                    assert new_replay_buf is not None
                    upd_far = _replay_chunk_on_bank(
                        st._s_far_bank,
                        new_replay_buf,
                        n_cur - w_chunks,
                        c_size,
                        w_chunks,
                        grad_enabled=torch.is_grad_enabled(),
                        sync_free=True,
                    )
                    if far_distinct:
                        st._s_far_bank.copy_(upd_far)
                    else:
                        st._s_far_bank = upd_far
                if cs_distinct:
                    st._s_chunk_start_bank.copy_(new_pages)
                else:
                    st._s_chunk_start_bank = new_pages.clone().contiguous()
                next_cached_mid: tuple[torch.Tensor, ...] | None = None
            else:
                next_cached_mid = st._cached_mid_ring
            st.step_count.add_(1)
            st._uniform_step = new_step_int
            st._boundary_ring = None
            st._s_seg_start_bank = new_seg_bank
            st.replay_buf = new_replay_buf
            st._cached_mid_ring = next_cached_mid
            next_state = st
        else:
            if new_step_int % c_size == 0:
                n_cur = idx0 // c_size
                new_cstart_bank = new_pages.clone().contiguous()
                if w_chunks == 0:
                    new_far_bank = new_cstart_bank
                elif n_cur >= w_chunks:
                    assert new_replay_buf is not None
                    new_far_bank = _replay_chunk_on_bank(
                        st.s_far_bank,
                        new_replay_buf,
                        n_cur - w_chunks,
                        c_size,
                        w_chunks,
                        grad_enabled=torch.is_grad_enabled(),
                        sync_free=True,
                    )
                else:
                    new_far_bank = st.s_far_bank
                next_cached_mid = None
            else:
                new_far_bank = st.s_far_bank
                new_cstart_bank = st.s_chunk_start_bank
                next_cached_mid = st._cached_mid_ring
            next_state = HSSDCacheState(
                step_count=st.step_count + 1,
                chunk_size=c_size,
                window_chunks=w_chunks,
                page_states=new_pages,
                theta_cum=theta_cum,
                boundary_ring=None,
                swa_k_ring=swa_k_ring,
                swa_v_ring=swa_v_ring,
                swa_valid_ring=swa_valid_ring,
                s_far_bank=new_far_bank,
                s_chunk_start_bank=new_cstart_bank,
                s_seg_start_bank=new_seg_bank,
                replay_buf=new_replay_buf,
                _uniform_step=new_step_int,
                _cached_mid_ring=next_cached_mid,
            )

    out_cast = out.squeeze(1).to(init_dtype) if squeeze_time else out.to(init_dtype)
    if return_duty_sum:
        return out_cast, next_state, u_sum, valid_cnt
    return out_cast, next_state


def _recurrent_hssd_raw(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
    boltzmann_gate: PeakBoltzmannGate | None,
    swa_recency_slope: torch.Tensor | None,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    segment_chunks: int,
    initial_state: HSSDCacheState | torch.Tensor | None,
    output_final_state: bool,
    write_mask: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
    g_override: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
    state_dtype: str | torch.dtype,
    use_swa: bool,
    use_phase: bool,
    use_surprise: bool,
    surprise_mode: str,
    write_mode: str,
    compact_active_pages: bool,
    compact_decode_cache: bool,
) -> tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor, torch.Tensor]:
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    init_dtype = query.dtype
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    target_dt = _resolve_state_dtype(state_dtype, acc_dt)

    doc_spans = _validate_packed_boundaries(
        bsz=bsz, seq_len=seq_len, initial_state=initial_state, cu_seqlens=cu_seqlens, seq_idx=seq_idx
    )
    if doc_spans is not None:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_tot = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot = query.new_zeros((), dtype=acc_dt)
        for b, spans_b in enumerate(doc_spans):
            sl_b = slice(b, b + 1)
            doc_outs: list[torch.Tensor] = []
            last_st: HSSDCacheState | None = None
            for s0, s1 in spans_b:
                sl_t = slice(s0, s1)
                o_d, st_d, u_d, c_d = _recurrent_hssd_raw(
                    query[sl_b, sl_t],
                    key[sl_b, sl_t],
                    value[sl_b, sl_t],
                    _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=sl_t),
                    b_gate[sl_b, sl_t],
                    w_gate[sl_b, sl_t],
                    theta[sl_b, sl_t],
                    surprise_bias[sl_b, sl_t],
                    surprise_slope,
                    surprise_thresh,
                    write_logits[sl_b, sl_t],
                    read_logits[sl_b, sl_t],
                    q_swa[sl_b, sl_t],
                    k_swa[sl_b, sl_t],
                    v_swa[sl_b, sl_t],
                    boltzmann_gate=boltzmann_gate,
                    swa_recency_slope=swa_recency_slope,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    segment_chunks=segment_chunks,
                    initial_state=None,
                    output_final_state=output_final_state,
                    write_mask=None if write_mask is None else write_mask[sl_b, sl_t],
                    pad_mask=None if pad_mask is None else pad_mask[sl_b, sl_t],
                    cu_seqlens=None,
                    seq_idx=None,
                    g_override=None if g_override is None else g_override[sl_b, sl_t],
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                    state_dtype=target_dt,
                    use_swa=use_swa,
                    use_phase=use_phase,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                    write_mode=write_mode,
                    compact_active_pages=compact_active_pages,
                    compact_decode_cache=compact_decode_cache,
                )
                doc_outs.append(o_d)
                last_st = st_d
                u_tot = u_tot + u_d
                cnt_tot = cnt_tot + c_d
            row_outs.append(
                torch.cat(doc_outs, dim=1)
                if doc_outs
                else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
            )
            if output_final_state:
                if last_st is None:
                    last_st = HSSDCacheState.zeros(
                        1, num_pages, hv, dk, dv,
                        chunk_size=chunk_size, window_chunks=window_chunks,
                        device=query.device, dtype=target_dt,
                    )
                row_states.append(last_st)
        out_all = row_outs[0] if len(row_outs) == 1 else torch.cat(row_outs, dim=0)
        st_all = (
            (row_states[0] if len(row_states) == 1 else torch.cat(row_states, dim=0))
            if output_final_state
            else None
        )
        return out_all, st_all, u_tot, cnt_tot

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
        dtype=target_dt,
    )

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, seq_len).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())
    u_step = getattr(state, "_uniform_step", None)
    inhomogeneous_steps = (
        False if u_step is not None else (bsz > 1 and not bool((state.step_count == state.step_count[0]).all()))
    )

    if has_pad_zeros or inhomogeneous_steps:
        row_outs_p: list[torch.Tensor] = []
        row_states_p: list[HSSDCacheState] = []
        u_tot_p = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot_p = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = state[b : b + 1]
            sl_b = slice(b, b + 1)
            idx_b = pm_bool[b].nonzero(as_tuple=False).squeeze(-1) if pm_bool is not None else None
            if idx_b is not None and idx_b.numel() == 0:
                row_outs_p.append(query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype))
                if output_final_state:
                    row_states_p.append(_cast_state_dtype(st_b, target_dt, acc_dt))
                continue
            tsl: slice | torch.Tensor | None = (
                idx_b if (idx_b is not None and idx_b.numel() < seq_len) else None
            )
            q_b = query[sl_b] if tsl is None else query[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            k_b = key[sl_b] if tsl is None else key[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            v_b = value[sl_b] if tsl is None else value[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            gam_b = _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=tsl)
            bg_b = b_gate[sl_b] if tsl is None else b_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wg_b = w_gate[sl_b] if tsl is None else w_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            th_b = theta[sl_b] if tsl is None else theta[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            sb_b = surprise_bias[sl_b] if tsl is None else surprise_bias[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wl_b = write_logits[sl_b] if tsl is None else write_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            rl_b = read_logits[sl_b] if tsl is None else read_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            qs_b = q_swa[sl_b] if tsl is None else q_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            ks_b = k_swa[sl_b] if tsl is None else k_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            vs_b = v_swa[sl_b] if tsl is None else v_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wm_b = None
            if write_mask is not None:
                wm_row = write_mask.reshape(bsz, seq_len)[sl_b]
                wm_b = wm_row if tsl is None else wm_row.index_select(1, idx_b)  # type: ignore[arg-type]
            go_b = None
            if g_override is not None:
                go_b = g_override[sl_b] if tsl is None else g_override[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]

            o_valid, st_next_b, u_b, c_b = _recurrent_hssd_raw(
                q_b,
                k_b,
                v_b,
                gam_b,
                bg_b,
                wg_b,
                th_b,
                sb_b,
                surprise_slope,
                surprise_thresh,
                wl_b,
                rl_b,
                qs_b,
                ks_b,
                vs_b,
                boltzmann_gate=boltzmann_gate,
                swa_recency_slope=swa_recency_slope,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                segment_chunks=segment_chunks,
                initial_state=st_b,
                output_final_state=output_final_state,
                write_mask=wm_b,
                pad_mask=None,
                cu_seqlens=None,
                seq_idx=None,
                g_override=go_b,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=target_dt,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_active_pages=compact_active_pages,
                compact_decode_cache=compact_decode_cache,
            )
            if tsl is None:
                row_outs_p.append(o_valid)
            else:
                assert idx_b is not None
                n_val = int(idx_b.numel())
                idx_exp = idx_b.view(1, n_val, 1, 1).expand(1, n_val, hv, dv)
                out_row = o_valid.new_zeros(1, seq_len, hv, dv).scatter(1, idx_exp, o_valid)
                row_outs_p.append(out_row)
            if output_final_state:
                assert st_next_b is not None
                row_states_p.append(st_next_b)
            u_tot_p = u_tot_p + u_b
            cnt_tot_p = cnt_tot_p + c_b

        out_p = row_outs_p[0] if len(row_outs_p) == 1 else torch.cat(row_outs_p, dim=0)
        st_p = (
            (row_states_p[0] if len(row_states_p) == 1 else torch.cat(row_states_p, dim=0))
            if output_final_state
            else None
        )
        return out_p, st_p, u_tot_p, cnt_tot_p

    outs: list[torch.Tensor] = []
    u_total = query.new_zeros(num_pages, dtype=acc_dt)
    tok_total = query.new_zeros((), dtype=acc_dt)
    allow_inplace = not torch.is_grad_enabled()

    for i in range(seq_len):
        sl_i = slice(i, i + 1)
        wm_i = None if write_mask is None else write_mask[:, sl_i]
        go_i = None if g_override is None else g_override[:, sl_i]
        step_inplace = allow_inplace and not (i == 0 and initial_state is not None and state is initial_state)
        o_i, state, u_i, cnt_i = recurrent_hssd_step(
            query[:, sl_i],
            key[:, sl_i],
            value[:, sl_i],
            _slice_gamma_input(gamma, time_sl=sl_i),
            b_gate[:, sl_i],
            w_gate[:, sl_i],
            theta[:, sl_i],
            surprise_bias[:, sl_i],
            surprise_slope,
            surprise_thresh,
            write_logits[:, sl_i],
            read_logits[:, sl_i],
            q_swa[:, sl_i],
            k_swa[:, sl_i],
            v_swa[:, sl_i],
            state,
            boltzmann_gate=boltzmann_gate,
            swa_recency_slope=swa_recency_slope,
            top_k_pages=top_k_pages,
            segment_chunks=segment_chunks,
            write_mask=wm_i,
            pad_mask=None,
            g_override=go_i,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            state_dtype=target_dt,
            use_swa=use_swa,
            use_phase=use_phase,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
            write_mode=write_mode,
            compact_active_pages=compact_active_pages,
            compact_decode_cache=compact_decode_cache,
            inplace_state=step_inplace,
            return_duty_sum=True,
        )
        outs.append(o_i)
        u_total = u_total + u_i
        tok_total = tok_total + cnt_i

    out = torch.cat(outs, dim=1) if outs else query.new_zeros(bsz, 0, hv, dv)
    final_st = _cast_state_dtype(state, target_dt, acc_dt) if output_final_state else None
    return out, final_st, u_total, tok_total


def recurrent_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
    swa_recency_slope: torch.Tensor | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    segment_chunks: int = 4,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    cu_seqlens: Any = None,
    seq_idx: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    write_mode: str = "full",
    compact_active_pages: bool = True,
    compact_decode_cache: bool = True,
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Token-by-token sequential HSSD v3.2 reference execution over `T` steps via `recurrent_hssd_step`."""
    out, final_st, u_total, tok_total = _recurrent_hssd_raw(
        query,
        key,
        value,
        gamma,
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
        boltzmann_gate=boltzmann_gate,
        swa_recency_slope=swa_recency_slope,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        segment_chunks=segment_chunks,
        initial_state=initial_state,
        output_final_state=output_final_state,
        write_mask=write_mask,
        pad_mask=pad_mask,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        g_override=g_override,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        state_dtype=state_dtype,
        use_swa=use_swa,
        use_phase=use_phase,
        use_surprise=use_surprise,
        surprise_mode=surprise_mode,
        write_mode=write_mode,
        compact_active_pages=compact_active_pages,
        compact_decode_cache=compact_decode_cache,
    )
    if return_aux:
        return out, final_st, u_total / tok_total.clamp_min(1.0)
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


def _segment_groups_for_chunks(
    all_chunks: list[tuple[int, int]],
    *,
    step_offset: int,
    chunk_size: int,
    seg_step: int,
    align_global_segments: bool,
) -> list[tuple[int, int]]:
    """Return `(start_chunk_idx, end_chunk_idx)` groups over `all_chunks`."""
    n_chunks = len(all_chunks)
    if n_chunks == 0:
        return []
    if not align_global_segments:
        return [(i, min(n_chunks, i + seg_step)) for i in range(0, n_chunks, seg_step)]
    groups: list[tuple[int, int]] = []
    i = 0
    while i < n_chunks:
        g0 = step_offset + all_chunks[i][0]
        seg_id = (g0 // chunk_size) // seg_step
        j = i + 1
        while j < n_chunks:
            gj = step_offset + all_chunks[j][0]
            if (gj // chunk_size) // seg_step != seg_id:
                break
            j += 1
        groups.append((i, j))
        i = j
    return groups


_COMPILED_CHUNK_GROUP_FN: Callable[..., Any] | None = None


def _run_chunk_group(
    chunk_bounds: list[tuple[int, int]],
    *,
    pre_chunk_map: list[tuple[int, int] | None],
    fut_src_locals: list[int],
    fut_chunk_bounds: list[tuple[int, int]],
    save_boundary_locals: list[int],
    collect_replay: bool,
    step_offset: int,
    swa_base_global: int,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    use_swa: bool,
    use_surprise: bool,
    surprise_mode: str,
    write_mode: str,
    compact_active_pages: bool,
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    gamma_page_scale: torch.Tensor | None,
    gamma_chan_seg: torch.Tensor,
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
    swa_recency_slope: torch.Tensor | None,
    o_far_pre: torch.Tensor,
    q_fut_seg: torch.Tensor,
    rl_fut_seg: torch.Tensor,
    page_states: torch.Tensor,
    first_surprise_bank: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """Execute a contiguous group of global-aligned chunks with read-ahead far readout (`H4`)."""
    bsz = q_hat.shape[0]
    dk = k_hat.shape[-1]
    dv = v.shape[-1]
    hv = k_hat.shape[2]
    outs: list[torch.Tensor] = []
    u_acc = q_hat.new_zeros(wl.shape[-1])
    tok_acc = q_hat.new_zeros(())

    kappa = gate_log_kappa.exp()
    kappa_b = kappa[0] if kappa.numel() == 1 else kappa.view(1, hv, 1)
    tau_b = gate_tau[0] if gate_tau.numel() == 1 else gate_tau.view(1, hv, 1)

    n_grp = len(chunk_bounds)
    base_l0 = chunk_bounds[0][0] if n_grp > 0 else 0
    g_start = step_offset + base_l0
    win_prior = window_chunks * chunk_size
    is_uniform_full_chunks = (
        n_grp > 0
        and g_start % chunk_size == 0
        and all(
            chunk_bounds[c][0] == base_l0 + c * chunk_size
            and chunk_bounds[c][1] == base_l0 + (c + 1) * chunk_size
            for c in range(n_grp)
        )
    )
    can_batch_swa = (
        use_swa
        and n_grp > 1
        and is_uniform_full_chunks
        and swa_base_global <= max(0, g_start - win_prior)
    )

    o_loc_batched: torch.Tensor | None = None
    p_hit_batched: torch.Tensor | None = None
    if can_batch_swa:
        win_full = (window_chunks + 1) * chunk_size
        buf_len = (window_chunks + n_grp) * chunk_size
        want_start = g_start - win_prior
        if want_start >= swa_base_global:
            s_off = want_start - swa_base_global
            ks_buf = all_ks[:, s_off : s_off + buf_len]
            vs_buf = all_vs[:, s_off : s_off + buf_len]
            val_buf = all_valid_swa[:, s_off : s_off + buf_len]
        else:
            pad_left = swa_base_global - want_start
            take_len = buf_len - pad_left
            ks_buf = F.pad(all_ks[:, :take_len], (0, 0, 0, 0, pad_left, 0), value=0.0)
            vs_buf = F.pad(all_vs[:, :take_len], (0, 0, 0, 0, pad_left, 0), value=0.0)
            val_buf = F.pad(all_valid_swa[:, :take_len], (pad_left, 0), value=False)

        k_win_t = ks_buf.unfold(1, win_full, chunk_size)                      # [B, n_grp, H_v, d_k, win_full]
        v_win_b = vs_buf.unfold(1, win_full, chunk_size).transpose(-1, -2)    # [B, n_grp, H_v, win_full, d_v]
        val_win_b = val_buf.unfold(1, win_full, chunk_size)                   # [B, n_grp, win_full]

        q_loc_b = (
            qs[:, base_l0 : base_l0 + n_grp * chunk_size]
            .reshape(bsz, n_grp, chunk_size, hv, dk)
            .transpose(2, 3)
        )                                                                     # [B, n_grp, H_v, C, d_k]
        scores_b = (q_loc_b @ k_win_t).to(dtype=q_hat.dtype) * (dk ** -0.5)

        r_idx = torch.arange(chunk_size, device=qs.device)[:, None]
        w_idx = torch.arange(win_full, device=qs.device)[None, :]
        if swa_recency_slope is not None and swa_recency_slope.numel() > 0:
            dist_b = (win_prior + r_idx - w_idx).clamp_min(0).to(dtype=scores_b.dtype) * (
                1.0 / float(chunk_size)
            )
            slp_b = swa_recency_slope.view(1, 1, hv, 1, 1)
            scores_b = scores_b - slp_b * dist_b[None, None, None, :, :]

        causal_b = w_idx <= (win_prior + r_idx)
        valid_mask_b = causal_b[None, None, None, :, :] & val_win_b[:, :, None, None, :]

        has_valid_b = valid_mask_b.any(dim=-1)
        safe_vm_b = torch.where(has_valid_b.unsqueeze(-1), valid_mask_b, torch.ones_like(valid_mask_b))
        masked_sc_b = scores_b.masked_fill(~safe_vm_b, float("-inf"))

        probs_b = torch.softmax(masked_sc_b, dim=-1)
        probs_b = torch.where(has_valid_b.unsqueeze(-1), probs_b, torch.zeros_like(probs_b))
        o_loc_batched = (probs_b.to(dtype=v_win_b.dtype) @ v_win_b).to(dtype=q_hat.dtype).transpose(2, 3)

        counts_b = valid_mask_b.expand_as(scores_b).sum(dim=-1)
        has_v_sc_b = counts_b > 0
        n_eff_b = counts_b.clamp_min(1).to(scores_b.dtype)
        lse_b = torch.logsumexp(masked_sc_b, dim=-1)
        max_sc_b = masked_sc_b.max(dim=-1).values
        stat_b = 0.5 * (lse_b - torch.log(n_eff_b)) + 0.5 * (
            max_sc_b - torch.sqrt(2.0 * torch.log(n_eff_b.clamp_min(2.0)))
        )
        p_hit_b = torch.sigmoid(kappa_b * (stat_b - tau_b))
        p_hit_batched = torch.where(has_v_sc_b, p_hit_b, torch.zeros_like(p_hit_b)).transpose(2, 3).unsqueeze(-1)

    p_cur = page_states
    s_init_ref = page_states if first_surprise_bank is None else first_surprise_bank

    fut_map: dict[int, tuple[int, int]] = {
        src_c: fb for src_c, fb in zip(fut_src_locals, fut_chunk_bounds)
    }

    # Precompute segment write/read Top-k routing + 1-sync batched active-page plans (`0` per-chunk host syncs)
    seg_pi_write: torch.Tensor | None = None
    seg_w_vals: torch.Tensor | None = None
    seg_w_idx: torch.Tensor | None = None
    intra_read_routing: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    fut_read_routing: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    seg_write_plan: _ActivePagePlan | None = None
    chunk_write_plans: list[_ActivePagePlan | None] = [None] * n_grp
    intra_read_plans: dict[int, _ActivePagePlan | None] = {}
    fut_read_plans: dict[int, _ActivePagePlan | None] = {}

    if n_grp > 0:
        seg_pi_write = F.softmax(wl, dim=-1)
        seg_w_vals, seg_w_idx = torch.topk(seg_pi_write, k=top_k_pages, dim=-1)

        ti_rfut: torch.Tensor | None = None
        tw_rfut: torch.Tensor | None = None
        if rl_fut_seg.shape[1] > 0:
            pi_rfut = F.softmax(rl_fut_seg, dim=-1)
            tv_rfut, ti_rfut = torch.topk(pi_rfut, k=top_k_pages, dim=-1)
            tw_rfut = tv_rfut / tv_rfut.sum(dim=-1, keepdim=True).clamp_min(1e-30)

        for c_i, (_l0, l1) in enumerate(chunk_bounds):
            if (step_offset + l1) % chunk_size == 0:
                c_intra_dst = c_i + window_chunks + 1
                if c_intra_dst < n_grp:
                    fl0, fl1 = chunk_bounds[c_intra_dst]
                    pi_ri = F.softmax(rl[:, fl0:fl1], dim=-1)
                    tv_ri, ti_ri = torch.topk(pi_ri, k=top_k_pages, dim=-1)
                    tw_ri = tv_ri / tv_ri.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                    intra_read_routing[c_i] = (ti_ri, tw_ri)
                elif c_i in fut_map and ti_rfut is not None and tw_rfut is not None:
                    fl0, fl1 = fut_map[c_i]
                    fut_read_routing[c_i] = (ti_rfut[:, fl0:fl1], tw_rfut[:, fl0:fl1])

        if compact_active_pages and top_k_pages < num_pages and not _is_compiling():
            plan_spans: list[torch.Tensor] = []
            seg_plan_pos = -1
            chunk_plan_start = -1
            if surprise_mode == "segment_start":
                seg_plan_pos = len(plan_spans)
                plan_spans.append(seg_w_idx)
                if not (n_grp > 1 and is_uniform_full_chunks):
                    chunk_plan_start = len(plan_spans)
                    for l0, l1 in chunk_bounds:
                        plan_spans.append(seg_w_idx[:, l0:l1])
            else:
                chunk_plan_start = len(plan_spans)
                for l0, l1 in chunk_bounds:
                    plan_spans.append(seg_w_idx[:, l0:l1])

            intra_pos_map: dict[int, int] = {}
            for c_i, (ti_ri, _tw_ri) in intra_read_routing.items():
                intra_pos_map[c_i] = len(plan_spans)
                plan_spans.append(ti_ri)

            fut_pos_map: dict[int, int] = {}
            for c_i, (ti_rf, _tw_rf) in fut_read_routing.items():
                fut_pos_map[c_i] = len(plan_spans)
                plan_spans.append(ti_rf)

            built_plans = _build_multi_span_active_plans(plan_spans, num_pages)
            if seg_plan_pos >= 0:
                seg_write_plan = built_plans[seg_plan_pos]
            if chunk_plan_start >= 0:
                for c_i in range(n_grp):
                    chunk_write_plans[c_i] = built_plans[chunk_plan_start + c_i]
            for c_i, pos in intra_pos_map.items():
                intra_read_plans[c_i] = built_plans[pos]
            for c_i, pos in fut_pos_map.items():
                fut_read_plans[c_i] = built_plans[pos]

    # Optional segment-batched surprise + WY solve (`Section 5.2 #6`: surprise_mode == "segment_start")
    seg_g_page: torch.Tensor | None = None
    seg_top_idx: torch.Tensor | None = None
    seg_g_act: torch.Tensor | None = None
    wy_w_batched: torch.Tensor | None = None
    wy_u0_batched: torch.Tensor | None = None
    wy_d_batched: torch.Tensor | None = None
    wy_kt_batched: torch.Tensor | None = None
    if surprise_mode == "segment_start" and n_grp > 0:
        assert seg_pi_write is not None and seg_w_vals is not None and seg_w_idx is not None
        seg_g_page, u_seg, cnt_seg, seg_top_idx, seg_g_act = _compute_root_surprise_and_routing(
            k_hat,
            v,
            s_init_ref,
            sb,
            slp,
            thr,
            wl,
            top_k_pages,
            wm,
            go,
            use_surprise=use_surprise,
            surprise_mode="segment_start",
            write_mode=write_mode,
            compact_active_pages=compact_active_pages,
            precomputed_write_routing=(seg_pi_write, seg_w_vals, seg_w_idx),
            active_plan=seg_write_plan,
        )
        u_acc = u_acc + u_seg
        tok_acc = tok_acc + cnt_seg

        if n_grp > 1 and is_uniform_full_chunks:
            gamma_all = _materialize_chunk_gamma(
                gamma_page_scale, gamma_chan_seg, num_pages=num_pages, dk=dk
            )
            sqrt_b_all = _safe_sqrt(bf)
            sqrt_w_all = _safe_sqrt(wf)
            ke_all = (
                (seg_g_page.unsqueeze(-1) * (sqrt_b_all * k_hat).unsqueeze(2))
                .reshape(bsz, n_grp, chunk_size, num_pages, hv, dk)
                .permute(0, 1, 3, 4, 2, 5)
            )                                                                 # [B, n_grp, M, H_v, C, d_k]
            ve_all = (
                (seg_g_page.unsqueeze(-1) * (sqrt_w_all * v).unsqueeze(2))
                .reshape(bsz, n_grp, chunk_size, num_pages, hv, dv)
                .permute(0, 1, 3, 4, 2, 5)
            )                                                                 # [B, n_grp, M, H_v, C, d_v]
            la_all = (
                (-(seg_g_page * seg_g_page).unsqueeze(-1) * gamma_all)
                .clamp(max=0.0)
                .reshape(bsz, n_grp, chunk_size, num_pages, hv, dk)
                .permute(0, 1, 3, 4, 2, 5)
            )                                                                 # [B, n_grp, M, H_v, C, d_k]
            if compact_active_pages and top_k_pages < num_pages and not _is_compiling():
                act_bcm = (
                    torch.zeros(
                        (bsz, n_grp * chunk_size, num_pages),
                        dtype=torch.bool,
                        device=ke_all.device,
                    )
                    .scatter_(2, seg_top_idx, True)
                    .reshape(bsz, n_grp, chunk_size, num_pages)
                    .permute(0, 1, 3, 2)
                )                                                             # [B, n_grp, M, C]
                tot_bcm = bsz * n_grp * num_pages
                active_bcm = [
                    idx for idx, flag in enumerate(act_bcm.any(dim=-1).reshape(-1).tolist()) if flag
                ]
                n_act = len(active_bcm)
                if n_act == 0:
                    wy_w_batched = ke_all.new_zeros((bsz, n_grp, num_pages, hv, chunk_size, dk))
                    wy_u0_batched = ve_all.new_zeros((bsz, n_grp, num_pages, hv, chunk_size, dv))
                    wy_d_batched = ke_all.new_ones((bsz, n_grp, num_pages, hv, dk, 1))
                    wy_kt_batched = ke_all.new_zeros((bsz, n_grp, num_pages, hv, dk, chunk_size))
                elif n_act < tot_bcm:
                    lin_bcm = torch.tensor(active_bcm, dtype=torch.long, device=ke_all.device)
                    ke_s = ke_all.reshape(tot_bcm, hv, chunk_size, dk).index_select(0, lin_bcm)
                    ve_s = ve_all.reshape(tot_bcm, hv, chunk_size, dv).index_select(0, lin_bcm)
                    la_s = la_all.reshape(tot_bcm, hv, chunk_size, dk).index_select(0, lin_bcm)
                    l_s, cum_s = _build_wy_unit_lower(ke_s, la_s)
                    eb_s = cum_s.exp() * ke_s
                    wv_s = _solve_unit_lower_wy(l_s, torch.cat([eb_s, ve_s], dim=-1))
                    w_s, u0_s = wv_s.split([dk, dv], dim=-1)
                    d_s = cum_s[..., -1, :].exp().unsqueeze(-1)
                    kt_s = ((cum_s[..., -1:, :] - cum_s).exp() * ke_s).transpose(-1, -2)

                    idx_w = lin_bcm[:, None, None, None].expand(n_act, hv, chunk_size, dk)
                    idx_u = lin_bcm[:, None, None, None].expand(n_act, hv, chunk_size, dv)
                    idx_d = lin_bcm[:, None, None, None].expand(n_act, hv, dk, 1)
                    idx_kt = lin_bcm[:, None, None, None].expand(n_act, hv, dk, chunk_size)
                    wy_w_batched = ke_all.new_zeros((tot_bcm, hv, chunk_size, dk)).scatter(0, idx_w, w_s).view(
                        bsz, n_grp, num_pages, hv, chunk_size, dk
                    )
                    wy_u0_batched = ve_all.new_zeros((tot_bcm, hv, chunk_size, dv)).scatter(0, idx_u, u0_s).view(
                        bsz, n_grp, num_pages, hv, chunk_size, dv
                    )
                    wy_d_batched = ke_all.new_ones((tot_bcm, hv, dk, 1)).scatter(0, idx_d, d_s).view(
                        bsz, n_grp, num_pages, hv, dk, 1
                    )
                    wy_kt_batched = ke_all.new_zeros((tot_bcm, hv, dk, chunk_size)).scatter(0, idx_kt, kt_s).view(
                        bsz, n_grp, num_pages, hv, dk, chunk_size
                    )
            if wy_w_batched is None:
                l_all, cum_all = _build_wy_unit_lower(ke_all, la_all)
                eb_all = cum_all.exp() * ke_all
                wv_all = _solve_unit_lower_wy(l_all, torch.cat([eb_all, ve_all], dim=-1))
                wy_w_batched, wy_u0_batched = wv_all.split([dk, dv], dim=-1)
                wy_d_batched = cum_all[..., -1, :].exp().unsqueeze(-1)
                wy_kt_batched = ((cum_all[..., -1:, :] - cum_all).exp() * ke_all).transpose(-1, -2)

    o_far_intra: dict[int, torch.Tensor] = {}
    save_set = set(save_boundary_locals)
    saved_map: dict[int, torch.Tensor] = {}
    fut_outs: list[torch.Tensor] = []

    rep_ke_list: list[torch.Tensor] = []
    rep_ve_list: list[torch.Tensor] = []
    rep_g_list: list[torch.Tensor] = []
    rep_gam_list: list[torch.Tensor] = []
    rep_idx_list: list[torch.Tensor] = []

    for c_i, (l0, l1) in enumerate(chunk_bounds):
        g0 = step_offset + l0
        g1 = step_offset + l1
        c_len = l1 - l0

        q_c = q_hat[:, l0:l1]
        k_c = k_hat[:, l0:l1]
        v_c = v[:, l0:l1]
        wl_c = wl[:, l0:l1]
        sb_c = sb[:, l0:l1]
        wm_c = None if wm is None else wm[:, l0:l1]
        go_c = None if go is None else go[:, l0:l1]

        # 1. Far-horizon readout from precomputed / read-ahead `o_far` (`H4`)
        pre_sl = pre_chunk_map[c_i]
        o_far = o_far_pre[:, pre_sl[0] : pre_sl[1]] if pre_sl is not None else o_far_intra.pop(c_i)

        # 2. Exact causal local SWA over I_loc(t) = [far_cutoff + 1, t] (0-based [far_cutoff, g1)) (`H3`)
        if o_loc_batched is not None and p_hit_batched is not None:
            o_loc = o_loc_batched[:, c_i]
            p_hit = p_hit_batched[:, c_i]
            outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)
        elif use_swa:
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
            scores = (q_loc @ k_win.transpose(-1, -2)).to(dtype=q_hat.dtype) * (dk ** -0.5)
            if swa_recency_slope is not None and swa_recency_slope.numel() > 0:
                dist_c = (q_pos - k_pos).clamp_min(0).to(dtype=scores.dtype) * (1.0 / float(chunk_size))
                scores = scores - swa_recency_slope.view(1, hv, 1, 1) * dist_c[None, None, :, :]
            has_valid = valid_mask.any(dim=-1)
            safe_vm = torch.where(has_valid.unsqueeze(-1), valid_mask, torch.ones_like(valid_mask))
            masked_sc = scores.masked_fill(~safe_vm, float("-inf"))

            probs = torch.softmax(masked_sc, dim=-1)
            probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
            o_loc = (probs.to(dtype=v_win.dtype) @ v_win).to(dtype=q_hat.dtype).transpose(1, 2)

            counts = valid_mask.expand_as(scores).sum(dim=-1)
            has_v_sc = counts > 0
            n_eff = counts.clamp_min(1).to(scores.dtype)
            lse = torch.logsumexp(masked_sc, dim=-1)
            max_sc = masked_sc.max(dim=-1).values
            stat = 0.5 * (lse - torch.log(n_eff)) + 0.5 * (max_sc - torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0))))
            p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
            p_hit = torch.where(has_v_sc, p_hit, torch.zeros_like(p_hit)).transpose(1, 2).unsqueeze(-1)
            outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)
        else:
            outs.append(o_far)

        # 3. Parallel routed root-surprise + WY multi-page state update (`H1`, `H6`, `5.2 #4`, `5.2 #6`)
        w_plan_c = chunk_write_plans[c_i]
        if surprise_mode == "segment_start":
            assert seg_g_page is not None and seg_top_idx is not None and seg_g_act is not None
            g_page = seg_g_page[:, l0:l1]
            top_idx = seg_top_idx[:, l0:l1]
            g_act = seg_g_act[:, l0:l1]
        else:
            assert seg_pi_write is not None and seg_w_vals is not None and seg_w_idx is not None
            s_surp_c = s_init_ref if c_i == 0 else p_cur
            g_page, u_sum, cnt_c, top_idx, g_act = _compute_root_surprise_and_routing(
                k_c,
                v_c,
                s_surp_c,
                sb_c,
                slp,
                thr,
                wl_c,
                top_k_pages,
                wm_c,
                go_c,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_active_pages=compact_active_pages,
                precomputed_write_routing=(
                    seg_pi_write[:, l0:l1],
                    seg_w_vals[:, l0:l1],
                    seg_w_idx[:, l0:l1],
                ),
                active_plan=w_plan_c,
            )
            u_acc = u_acc + u_sum
            tok_acc = tok_acc + cnt_c

        sqrt_b = _safe_sqrt(bf[:, l0:l1])
        sqrt_w = _safe_sqrt(wf[:, l0:l1])
        ke_base_c = sqrt_b * k_c                                              # [B, c_len, H_v, d_k]
        ve_base_c = sqrt_w * v_c                                              # [B, c_len, H_v, d_v]

        if collect_replay:
            rep_ke_list.append(ke_base_c.transpose(1, 2))                     # [B, H_v, c_len, d_k]
            rep_ve_list.append(ve_base_c.transpose(1, 2))                     # [B, H_v, c_len, d_v]
            rep_g_list.append(g_act)                                          # [B, c_len, k, H_v]
            ch_c = gamma_chan_seg[:, l0:l1]
            if gamma_page_scale is not None:
                d_g = ch_c.shape[-1]
                ps_sel = gamma_page_scale.index_select(0, top_idx.reshape(-1)).view(
                    bsz, c_len, top_k_pages, 1, 1
                )
                gam_act = ps_sel * ch_c.unsqueeze(2)                          # [B, c_len, k, H_v, d_g]
            else:
                d_g = ch_c.shape[-1]
                g_idx_gam = top_idx[:, :, :, None, None].expand(bsz, c_len, top_k_pages, hv, d_g)
                gam_act = torch.gather(ch_c, dim=2, index=g_idx_gam)
            rep_gam_list.append(gam_act)
            rep_idx_list.append(top_idx)

        if (
            wy_w_batched is not None
            and wy_u0_batched is not None
            and wy_d_batched is not None
            and wy_kt_batched is not None
        ):
            u_eff_c = wy_u0_batched[:, c_i] - wy_w_batched[:, c_i] @ p_cur
            p_cur = wy_d_batched[:, c_i] * p_cur + wy_kt_batched[:, c_i] @ u_eff_c
        else:
            gamma_c = _materialize_chunk_gamma(
                gamma_page_scale, gamma_chan_seg[:, l0:l1], num_pages=num_pages, dk=dk
            )
            k_eff = (g_page.unsqueeze(-1) * ke_base_c.unsqueeze(2)).permute(0, 2, 3, 1, 4)
            v_eff = (g_page.unsqueeze(-1) * ve_base_c.unsqueeze(2)).permute(0, 2, 3, 1, 4)
            log_a = (-(g_page * g_page).unsqueeze(-1) * gamma_c).permute(0, 2, 3, 1, 4)
            act_mask: torch.Tensor | None = None
            if (
                w_plan_c is None
                and compact_active_pages
                and top_k_pages < num_pages
                and not _is_compiling()
            ):
                act_mask = (
                    torch.zeros((bsz, c_len, num_pages), dtype=torch.bool, device=k_c.device)
                    .scatter_(2, top_idx, True)
                    .transpose(1, 2)
                )
            p_cur = _wy_kda_state_update(
                p_cur,
                k_eff,
                v_eff,
                log_a,
                active_mask=act_mask,
                compact_active_pages=compact_active_pages,
                active_plan=w_plan_c,
            )

        if g1 % chunk_size == 0:
            if c_i in save_set:
                saved_map[c_i] = p_cur
            c_intra_dst = c_i + window_chunks + 1
            if c_intra_dst < n_grp:
                fl0, fl1 = chunk_bounds[c_intra_dst]
                o_far_intra[c_intra_dst] = _far_readout(
                    q_hat[:, fl0:fl1],
                    p_cur,
                    rl[:, fl0:fl1],
                    top_k_pages,
                    compact_active_pages=compact_active_pages,
                    precomputed_routing=intra_read_routing.get(c_i),
                    active_plan=intra_read_plans.get(c_i),
                )
            elif c_i in fut_map:
                fl0, fl1 = fut_map[c_i]
                fut_outs.append(
                    _far_readout(
                        q_fut_seg[:, fl0:fl1],
                        p_cur,
                        rl_fut_seg[:, fl0:fl1],
                        top_k_pages,
                        compact_active_pages=compact_active_pages,
                        precomputed_routing=fut_read_routing.get(c_i),
                        active_plan=fut_read_plans.get(c_i),
                    )
                )

    seg_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
    if fut_outs:
        o_far_fut = fut_outs[0] if len(fut_outs) == 1 else torch.cat(fut_outs, dim=1)
    else:
        o_far_fut = q_hat.new_zeros((bsz, 0, hv, dv))
    saved_banks = [saved_map[idx] for idx in save_boundary_locals if idx in saved_map]
    replay_seg: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if collect_replay and rep_ke_list:
        replay_seg = (
            rep_ke_list[0] if len(rep_ke_list) == 1 else torch.cat(rep_ke_list, dim=2),
            rep_ve_list[0] if len(rep_ve_list) == 1 else torch.cat(rep_ve_list, dim=2),
            rep_g_list[0] if len(rep_g_list) == 1 else torch.cat(rep_g_list, dim=1),
            rep_gam_list[0] if len(rep_gam_list) == 1 else torch.cat(rep_gam_list, dim=1),
            rep_idx_list[0] if len(rep_idx_list) == 1 else torch.cat(rep_idx_list, dim=1),
        )
    return seg_out, p_cur, u_acc, tok_acc, o_far_fut, saved_banks, replay_seg


def _get_compiled_chunk_group() -> Callable[..., Any]:
    global _COMPILED_CHUNK_GROUP_FN
    if _COMPILED_CHUNK_GROUP_FN is None:
        if hasattr(torch, "compile"):
            raw_compiled = torch.compile(_run_chunk_group, dynamic=True)

            def _safe_compiled(*args: Any, **kwargs: Any) -> Any:
                global _COMPILED_CHUNK_GROUP_FN
                try:
                    return raw_compiled(*args, **kwargs)
                except Exception as exc:
                    if "CppCompileError" in type(exc).__name__ or "CppCompileError" in str(exc):
                        _COMPILED_CHUNK_GROUP_FN = _run_chunk_group
                        return _run_chunk_group(*args, **kwargs)
                    raise

            _COMPILED_CHUNK_GROUP_FN = _safe_compiled
        else:
            _COMPILED_CHUNK_GROUP_FN = _run_chunk_group
    return _COMPILED_CHUNK_GROUP_FN


# ============================================================================ Triton CUDA fast-path kernels

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - optional on CPU / macOS dev hosts
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _phase_chunk_sum_fwd_kernel(
        Theta_ptr,
        WM_ptr,
        ChunkSum_ptr,
        stride_tb,
        stride_tt,
        stride_th,
        stride_td,
        stride_wmb,
        stride_wmt,
        stride_csb,
        stride_csc,
        stride_csh,
        stride_csd,
        T_len,
        H,
        CHUNK_C: tl.constexpr,
        D_HALF: tl.constexpr,
        HAS_WM: tl.constexpr,
    ):
        c_id = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        t0 = c_id * CHUNK_C
        offs_t = t0 + tl.arange(0, CHUNK_C)
        mask_t = offs_t < T_len
        offs_d = tl.arange(0, D_HALF)

        th = tl.load(
            Theta_ptr + b * stride_tb + offs_t[:, None] * stride_tt + h * stride_th + offs_d[None, :] * stride_td,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float64)
        if HAS_WM:
            wm = tl.load(WM_ptr + b * stride_wmb + offs_t * stride_wmt, mask=mask_t, other=0.0).to(tl.float64)
            th = th * wm[:, None]
        csum = tl.sum(th, axis=0)
        tl.store(ChunkSum_ptr + b * stride_csb + c_id * stride_csc + h * stride_csh + offs_d * stride_csd, csum)

    @triton.jit
    def _phase_chunk_scan_fwd_kernel(
        InitTheta_ptr,
        ChunkSum_ptr,
        ChunkPrefix_ptr,
        TermTheta_ptr,
        stride_ib,
        stride_ih,
        stride_id,
        stride_csb,
        stride_csc,
        stride_csh,
        stride_csd,
        N_chunks,
        H,
        D_HALF: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        offs_d = tl.arange(0, D_HALF)
        two_pi = tl.full([D_HALF], 6.283185307179586476925286766559, dtype=tl.float64)

        acc = tl.load(InitTheta_ptr + b * stride_ib + h * stride_ih + offs_d * stride_id).to(tl.float64)
        for c in range(N_chunks):
            tl.store(ChunkPrefix_ptr + b * stride_csb + c * stride_csc + h * stride_csh + offs_d * stride_csd, acc)
            cs = tl.load(ChunkSum_ptr + b * stride_csb + c * stride_csc + h * stride_csh + offs_d * stride_csd)
            acc = acc + cs
        q = tl.floor(acc / two_pi)
        term = acc - q * two_pi
        tl.store(TermTheta_ptr + b * stride_ib + h * stride_ih + offs_d * stride_id, term)

    @triton.jit
    def _phase_trig_emit_fwd_kernel(
        Theta_ptr,
        WM_ptr,
        ChunkPrefix_ptr,
        Cos_ptr,
        Sin_ptr,
        ModF32_ptr,
        stride_tb,
        stride_tt,
        stride_th,
        stride_td,
        stride_wmb,
        stride_wmt,
        stride_cpb,
        stride_cpc,
        stride_cph,
        stride_cpd,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        T_len,
        H,
        CHUNK_C: tl.constexpr,
        D_HALF: tl.constexpr,
        HAS_WM: tl.constexpr,
        SAVE_MOD: tl.constexpr,
    ):
        c_id = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        t0 = c_id * CHUNK_C
        offs_t = t0 + tl.arange(0, CHUNK_C)
        mask_t = offs_t < T_len
        offs_d = tl.arange(0, D_HALF)
        two_pi = tl.full([1, D_HALF], 6.283185307179586476925286766559, dtype=tl.float64)

        th = tl.load(
            Theta_ptr + b * stride_tb + offs_t[:, None] * stride_tt + h * stride_th + offs_d[None, :] * stride_td,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float64)
        if HAS_WM:
            wm = tl.load(WM_ptr + b * stride_wmb + offs_t * stride_wmt, mask=mask_t, other=0.0).to(tl.float64)
            th = th * wm[:, None]

        pref = tl.load(ChunkPrefix_ptr + b * stride_cpb + c_id * stride_cpc + h * stride_cph + offs_d * stride_cpd)
        th_cum = pref[None, :] + tl.cumsum(th, axis=0)
        q_pos = tl.floor(th_cum / two_pi)
        th_mod = th_cum - q_pos * two_pi
        neg_th = -th_mod
        q_neg = tl.floor(neg_th / two_pi)
        neg_mod = (neg_th - q_neg * two_pi).to(tl.float32)

        cos_v = tl.cos(neg_mod)
        sin_v = tl.sin(neg_mod)
        o_ptrs = b * stride_ob + offs_t[:, None] * stride_ot + h * stride_oh + offs_d[None, :] * stride_od
        tl.store(Cos_ptr + o_ptrs, cos_v.to(Cos_ptr.dtype.element_ty), mask=mask_t[:, None])
        tl.store(Sin_ptr + o_ptrs, sin_v.to(Sin_ptr.dtype.element_ty), mask=mask_t[:, None])
        if SAVE_MOD:
            tl.store(ModF32_ptr + o_ptrs, neg_mod, mask=mask_t[:, None])

    @triton.jit
    def _phase_bwd_chunk_sum_kernel(
        DCos_ptr,
        DSin_ptr,
        ModF32_ptr,
        DTerm_ptr,
        DChunkSum_ptr,
        DLocal_ptr,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_ib,
        stride_ih,
        stride_id,
        stride_csb,
        stride_csc,
        stride_csh,
        stride_csd,
        T_len,
        H,
        CHUNK_C: tl.constexpr,
        D_HALF: tl.constexpr,
        HAS_DTERM: tl.constexpr,
    ):
        c_id = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        t0 = c_id * CHUNK_C
        offs_t = t0 + tl.arange(0, CHUNK_C)
        mask_t = offs_t < T_len
        offs_d = tl.arange(0, D_HALF)

        o_ptrs = b * stride_ob + offs_t[:, None] * stride_ot + h * stride_oh + offs_d[None, :] * stride_od
        dcos = tl.load(DCos_ptr + o_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
        dsin = tl.load(DSin_ptr + o_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
        neg_mod = tl.load(ModF32_ptr + o_ptrs, mask=mask_t[:, None], other=0.0)
        dth_cum = tl.where(mask_t[:, None], dcos * tl.sin(neg_mod) - dsin * tl.cos(neg_mod), 0.0)
        if HAS_DTERM:
            is_last = offs_t == (T_len - 1)
            dterm = tl.load(DTerm_ptr + b * stride_ib + h * stride_ih + offs_d * stride_id).to(tl.float32)
            dth_cum = dth_cum + tl.where(is_last[:, None], dterm[None, :], 0.0)

        tl.store(DLocal_ptr + o_ptrs, dth_cum, mask=mask_t[:, None])
        tl.store(
            DChunkSum_ptr + b * stride_csb + c_id * stride_csc + h * stride_csh + offs_d * stride_csd,
            tl.sum(dth_cum, axis=0),
        )

    @triton.jit
    def _phase_bwd_scan_suffix_kernel(
        DChunkSum_ptr,
        DSuffix_ptr,
        DInit_ptr,
        stride_csb,
        stride_csc,
        stride_csh,
        stride_csd,
        stride_ib,
        stride_ih,
        stride_id,
        N_chunks,
        H,
        D_HALF: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        offs_d = tl.arange(0, D_HALF)
        acc = tl.zeros([D_HALF], dtype=tl.float32)
        for c_rev in range(N_chunks):
            c = N_chunks - 1 - c_rev
            tl.store(DSuffix_ptr + b * stride_csb + c * stride_csc + h * stride_csh + offs_d * stride_csd, acc)
            acc += tl.load(DChunkSum_ptr + b * stride_csb + c * stride_csc + h * stride_csh + offs_d * stride_csd)
        tl.store(DInit_ptr + b * stride_ib + h * stride_ih + offs_d * stride_id, acc.to(tl.float64))

    @triton.jit
    def _phase_bwd_emit_kernel(
        DLocal_ptr,
        DSuffix_ptr,
        Theta_ptr,
        WM_ptr,
        DTheta_ptr,
        DWM_ptr,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_csb,
        stride_csc,
        stride_csh,
        stride_csd,
        stride_tb,
        stride_tt,
        stride_th,
        stride_td,
        stride_wmb,
        stride_wmt,
        T_len,
        H,
        CHUNK_C: tl.constexpr,
        D_HALF: tl.constexpr,
        HAS_WM: tl.constexpr,
        NEED_DWM: tl.constexpr,
    ):
        c_id = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        t0 = c_id * CHUNK_C
        offs_t = t0 + tl.arange(0, CHUNK_C)
        mask_t = offs_t < T_len
        offs_d = tl.arange(0, D_HALF)

        o_ptrs = b * stride_ob + offs_t[:, None] * stride_ot + h * stride_oh + offs_d[None, :] * stride_od
        dloc = tl.load(DLocal_ptr + o_ptrs, mask=mask_t[:, None], other=0.0)
        suff = tl.load(DSuffix_ptr + b * stride_csb + c_id * stride_csc + h * stride_csh + offs_d * stride_csd)
        tot_c = tl.sum(dloc, axis=0)
        rev_c = tot_c[None, :] - tl.cumsum(dloc, axis=0) + dloc + suff[None, :]
        rev_c = tl.where(mask_t[:, None], rev_c, 0.0)

        if HAS_WM:
            wm = tl.load(WM_ptr + b * stride_wmb + offs_t * stride_wmt, mask=mask_t, other=0.0).to(tl.float32)
            if NEED_DWM:
                th = tl.load(
                    Theta_ptr
                    + b * stride_tb
                    + offs_t[:, None] * stride_tt
                    + h * stride_th
                    + offs_d[None, :] * stride_td,
                    mask=mask_t[:, None],
                    other=0.0,
                ).to(tl.float32)
                dwm_h = tl.sum(rev_c * th, axis=1)
                tl.atomic_add(DWM_ptr + b * stride_wmb + offs_t * stride_wmt, dwm_h, mask=mask_t)
            rev_c = rev_c * wm[:, None]

        tl.store(
            DTheta_ptr + b * stride_tb + offs_t[:, None] * stride_tt + h * stride_th + offs_d[None, :] * stride_td,
            rev_c.to(DTheta_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )

    class _TritonPhaseTrigFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta, init_theta_cum, wm, out_dtype: torch.dtype):
            ctx.set_materialize_grads(False)
            B, T, H, D_half = theta.shape
            chunk_c = 64
            n_chunks = (T + chunk_c - 1) // chunk_c
            th_c = theta.contiguous()
            init_c = init_theta_cum.to(torch.float64).contiguous()
            has_wm = wm is not None
            wm_c = wm.float().contiguous() if has_wm else th_c.new_empty((1, 1), dtype=torch.float32)

            csum = torch.empty((B, n_chunks, H, D_half), device=th_c.device, dtype=torch.float64)
            cpref = torch.empty((B, n_chunks, H, D_half), device=th_c.device, dtype=torch.float64)
            term_th = torch.empty((B, H, D_half), device=th_c.device, dtype=torch.float64)
            cos_t = torch.empty((B, T, H, D_half), device=th_c.device, dtype=out_dtype)
            sin_t = torch.empty((B, T, H, D_half), device=th_c.device, dtype=out_dtype)
            need_bwd = any(ctx.needs_input_grad)
            mod_f32 = (
                torch.empty((B, T, H, D_half), device=th_c.device, dtype=torch.float32)
                if need_bwd
                else cos_t.new_empty(0, dtype=torch.float32)
            )

            _phase_chunk_sum_fwd_kernel[(n_chunks, B * H)](
                th_c,
                wm_c,
                csum,
                *th_c.stride(),
                *wm_c.stride(),
                *csum.stride(),
                T,
                H,
                CHUNK_C=chunk_c,
                D_HALF=D_half,
                HAS_WM=has_wm,
                num_warps=4,
                num_stages=1,
            )
            _phase_chunk_scan_fwd_kernel[(B * H,)](
                init_c,
                csum,
                cpref,
                term_th,
                *init_c.stride(),
                *csum.stride(),
                n_chunks,
                H,
                D_HALF=D_half,
                num_warps=2,
                num_stages=1,
            )
            _phase_trig_emit_fwd_kernel[(n_chunks, B * H)](
                th_c,
                wm_c,
                cpref,
                cos_t,
                sin_t,
                mod_f32,
                *th_c.stride(),
                *wm_c.stride(),
                *cpref.stride(),
                *cos_t.stride(),
                T,
                H,
                CHUNK_C=chunk_c,
                D_HALF=D_half,
                HAS_WM=has_wm,
                SAVE_MOD=need_bwd,
                num_warps=4,
                num_stages=1,
            )
            if need_bwd:
                ctx.save_for_backward(th_c, wm_c, mod_f32)
                ctx.meta = (has_wm, B, T, H, D_half)
            return cos_t, sin_t, term_th

        @staticmethod
        def backward(ctx, dcos, dsin, dterm):
            th_c, wm_c, mod_f32 = ctx.saved_tensors
            has_wm, B, T, H, D_half = ctx.meta
            chunk_c = 64
            n_chunks = (T + chunk_c - 1) // chunk_c
            dcos_c = (
                dcos.contiguous()
                if dcos is not None
                else torch.zeros((B, T, H, D_half), device=th_c.device, dtype=torch.float32)
            )
            dsin_c = (
                dsin.contiguous()
                if dsin is not None
                else torch.zeros((B, T, H, D_half), device=th_c.device, dtype=torch.float32)
            )
            dterm_c = (
                dterm.contiguous() if dterm is not None else mod_f32.new_zeros((B, H, D_half), dtype=torch.float64)
            )

            dcsum = torch.empty((B, n_chunks, H, D_half), device=th_c.device, dtype=torch.float32)
            dsuff = torch.empty((B, n_chunks, H, D_half), device=th_c.device, dtype=torch.float32)
            dloc = torch.empty((B, T, H, D_half), device=th_c.device, dtype=torch.float32)
            dinit = torch.empty((B, H, D_half), device=th_c.device, dtype=torch.float64)
            dtheta = torch.empty_like(th_c)
            need_dwm = bool(has_wm and ctx.needs_input_grad[2])
            dwm = torch.zeros_like(wm_c) if need_dwm else wm_c.new_empty(1)

            _phase_bwd_chunk_sum_kernel[(n_chunks, B * H)](
                dcos_c,
                dsin_c,
                mod_f32,
                dterm_c,
                dcsum,
                dloc,
                *mod_f32.stride(),
                *dterm_c.stride(),
                *dcsum.stride(),
                T,
                H,
                CHUNK_C=chunk_c,
                D_HALF=D_half,
                HAS_DTERM=(dterm is not None),
                num_warps=4,
                num_stages=1,
            )
            _phase_bwd_scan_suffix_kernel[(B * H,)](
                dcsum,
                dsuff,
                dinit,
                *dcsum.stride(),
                *dinit.stride(),
                n_chunks,
                H,
                D_HALF=D_half,
                num_warps=2,
                num_stages=1,
            )
            _phase_bwd_emit_kernel[(n_chunks, B * H)](
                dloc,
                dsuff,
                th_c,
                wm_c,
                dtheta,
                dwm,
                *dloc.stride(),
                *dsuff.stride(),
                *th_c.stride(),
                *wm_c.stride(),
                T,
                H,
                CHUNK_C=chunk_c,
                D_HALF=D_half,
                HAS_WM=has_wm,
                NEED_DWM=need_dwm,
                num_warps=4,
                num_stages=1,
            )
            return (
                dtheta if ctx.needs_input_grad[0] else None,
                dinit if ctx.needs_input_grad[1] else None,
                dwm if need_dwm else None,
                None,
            )

    @triton.jit
    def _scan_prep_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        BGate_ptr,
        WGate_ptr,
        Cos_ptr,
        Sin_ptr,
        QHat_ptr,
        KHat_ptr,
        VTc_ptr,
        KBase_ptr,
        VBase_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_bgb,
        stride_bgt,
        stride_bgh,
        stride_bgd,
        stride_wgb,
        stride_wgt,
        stride_wgh,
        stride_wgd,
        stride_cb,
        stride_ct,
        stride_ch,
        stride_cd,
        stride_qhb,
        stride_qht,
        stride_qhh,
        stride_qhd,
        stride_vtb,
        stride_vtt,
        stride_vth,
        stride_vtd,
        T_len,
        H,
        scale,
        D_HALF: tl.constexpr,
        D_V: tl.constexpr,
        USE_PHASE: tl.constexpr,
        USE_L2: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len

        offs_p = tl.arange(0, D_HALF)
        d0 = 2 * offs_p
        d1 = d0 + 1

        q_base = Q_ptr + b * stride_qb + t * stride_qt + h * stride_qh
        k_base_p = K_ptr + b * stride_kb + t * stride_kt + h * stride_kh
        q0 = tl.load(q_base + d0 * stride_qd).to(tl.float32)
        q1 = tl.load(q_base + d1 * stride_qd).to(tl.float32)
        k0 = tl.load(k_base_p + d0 * stride_kd).to(tl.float32)
        k1 = tl.load(k_base_p + d1 * stride_kd).to(tl.float32)

        if USE_L2:
            rq = tl.rsqrt(tl.sum(q0 * q0 + q1 * q1, axis=0) + 1e-6)
            rk = tl.rsqrt(tl.sum(k0 * k0 + k1 * k1, axis=0) + 1e-6)
            q0 = q0 * rq
            q1 = q1 * rq
            k0 = k0 * rk
            k1 = k1 * rk

        if USE_PHASE:
            c_base = b * stride_cb + t * stride_ct + h * stride_ch + offs_p * stride_cd
            cs = tl.load(Cos_ptr + c_base).to(tl.float32)
            sn = tl.load(Sin_ptr + c_base).to(tl.float32)
            qr0 = q0 * cs - q1 * sn
            qr1 = q0 * sn + q1 * cs
            kr0 = k0 * cs - k1 * sn
            kr1 = k0 * sn + k1 * cs
        else:
            qr0 = q0
            qr1 = q1
            kr0 = k0
            kr1 = k1

        bg_p = BGate_ptr + b * stride_bgb + t * stride_bgt + h * stride_bgh
        bg0 = tl.load(bg_p + d0 * stride_bgd).to(tl.float32)
        bg1 = tl.load(bg_p + d1 * stride_bgd).to(tl.float32)
        sb0 = tl.where(bg0 > 0.0, tl.sqrt(tl.where(bg0 > 0.0, bg0, 1.0)), 0.0)
        sb1 = tl.where(bg1 > 0.0, tl.sqrt(tl.where(bg1 > 0.0, bg1, 1.0)), 0.0)

        out_k_off = b * stride_qhb + t * stride_qht + h * stride_qhh
        out_dt = QHat_ptr.dtype.element_ty
        tl.store(QHat_ptr + out_k_off + d0 * stride_qhd, (qr0 * scale).to(out_dt))
        tl.store(QHat_ptr + out_k_off + d1 * stride_qhd, (qr1 * scale).to(out_dt))
        tl.store(KHat_ptr + out_k_off + d0 * stride_qhd, kr0.to(out_dt))
        tl.store(KHat_ptr + out_k_off + d1 * stride_qhd, kr1.to(out_dt))
        tl.store(KBase_ptr + out_k_off + d0 * stride_qhd, (sb0 * kr0).to(out_dt))
        tl.store(KBase_ptr + out_k_off + d1 * stride_qhd, (sb1 * kr1).to(out_dt))

        offs_v = tl.arange(0, D_V)
        vv = tl.load(V_ptr + b * stride_vb + t * stride_vt + h * stride_vh + offs_v * stride_vd).to(tl.float32)
        wg = tl.load(WGate_ptr + b * stride_wgb + t * stride_wgt + h * stride_wgh + offs_v * stride_wgd).to(tl.float32)
        sw = tl.where(wg > 0.0, tl.sqrt(tl.where(wg > 0.0, wg, 1.0)), 0.0)
        out_v_off = b * stride_vtb + t * stride_vtt + h * stride_vth + offs_v * stride_vtd
        tl.store(VTc_ptr + out_v_off, vv.to(out_dt))
        tl.store(VBase_ptr + out_v_off, (sw * vv).to(out_dt))

    @triton.jit
    def _scan_prep_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        BGate_ptr,
        WGate_ptr,
        Cos_ptr,
        Sin_ptr,
        DQHat_ptr,
        DKHat_ptr,
        DVTc_ptr,
        DKBase_ptr,
        DVBase_ptr,
        DQ_ptr,
        DK_ptr,
        DV_ptr,
        DBGate_ptr,
        DWGate_ptr,
        DCos_ptr,
        DSin_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_bgb,
        stride_bgt,
        stride_bgh,
        stride_bgd,
        stride_wgb,
        stride_wgt,
        stride_wgh,
        stride_wgd,
        stride_cb,
        stride_ct,
        stride_ch,
        stride_cd,
        stride_qhb,
        stride_qht,
        stride_qhh,
        stride_qhd,
        stride_vtb,
        stride_vtt,
        stride_vth,
        stride_vtd,
        T_len,
        H,
        scale,
        D_HALF: tl.constexpr,
        D_V: tl.constexpr,
        USE_PHASE: tl.constexpr,
        USE_L2: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len

        offs_p = tl.arange(0, D_HALF)
        d0 = 2 * offs_p
        d1 = d0 + 1

        q_base = Q_ptr + b * stride_qb + t * stride_qt + h * stride_qh
        k_base_p = K_ptr + b * stride_kb + t * stride_kt + h * stride_kh
        q0_raw = tl.load(q_base + d0 * stride_qd).to(tl.float32)
        q1_raw = tl.load(q_base + d1 * stride_qd).to(tl.float32)
        k0_raw = tl.load(k_base_p + d0 * stride_kd).to(tl.float32)
        k1_raw = tl.load(k_base_p + d1 * stride_kd).to(tl.float32)

        if USE_L2:
            rq = tl.rsqrt(tl.sum(q0_raw * q0_raw + q1_raw * q1_raw, axis=0) + 1e-6)
            rk = tl.rsqrt(tl.sum(k0_raw * k0_raw + k1_raw * k1_raw, axis=0) + 1e-6)
            q0 = q0_raw * rq
            q1 = q1_raw * rq
            k0 = k0_raw * rk
            k1 = k1_raw * rk
        else:
            rq = 1.0
            rk = 1.0
            q0 = q0_raw
            q1 = q1_raw
            k0 = k0_raw
            k1 = k1_raw

        if USE_PHASE:
            c_base = b * stride_cb + t * stride_ct + h * stride_ch + offs_p * stride_cd
            cs = tl.load(Cos_ptr + c_base).to(tl.float32)
            sn = tl.load(Sin_ptr + c_base).to(tl.float32)
            kr0 = k0 * cs - k1 * sn
            kr1 = k0 * sn + k1 * cs
        else:
            cs = tl.zeros([D_HALF], dtype=tl.float32) + 1.0
            sn = tl.zeros([D_HALF], dtype=tl.float32)
            kr0 = k0
            kr1 = k1

        bg_p = BGate_ptr + b * stride_bgb + t * stride_bgt + h * stride_bgh
        bg0 = tl.load(bg_p + d0 * stride_bgd).to(tl.float32)
        bg1 = tl.load(bg_p + d1 * stride_bgd).to(tl.float32)
        sb0 = tl.where(bg0 > 0.0, tl.sqrt(tl.where(bg0 > 0.0, bg0, 1.0)), 0.0)
        sb1 = tl.where(bg1 > 0.0, tl.sqrt(tl.where(bg1 > 0.0, bg1, 1.0)), 0.0)

        out_k_off = b * stride_qhb + t * stride_qht + h * stride_qhh
        dqh0 = tl.load(DQHat_ptr + out_k_off + d0 * stride_qhd).to(tl.float32) * scale
        dqh1 = tl.load(DQHat_ptr + out_k_off + d1 * stride_qhd).to(tl.float32) * scale
        dkh0 = tl.load(DKHat_ptr + out_k_off + d0 * stride_qhd).to(tl.float32)
        dkh1 = tl.load(DKHat_ptr + out_k_off + d1 * stride_qhd).to(tl.float32)
        dkb0 = tl.load(DKBase_ptr + out_k_off + d0 * stride_qhd).to(tl.float32)
        dkb1 = tl.load(DKBase_ptr + out_k_off + d1 * stride_qhd).to(tl.float32)

        dbg0 = tl.where(bg0 > 0.0, 0.5 * dkb0 * kr0 / tl.maximum(sb0, 1e-6), 0.0)
        dbg1 = tl.where(bg1 > 0.0, 0.5 * dkb1 * kr1 / tl.maximum(sb1, 1e-6), 0.0)
        dbg_p = DBGate_ptr + b * stride_bgb + t * stride_bgt + h * stride_bgh
        tl.store(dbg_p + d0 * stride_bgd, dbg0.to(DBGate_ptr.dtype.element_ty))
        tl.store(dbg_p + d1 * stride_bgd, dbg1.to(DBGate_ptr.dtype.element_ty))

        dkr0 = dkh0 + sb0 * dkb0
        dkr1 = dkh1 + sb1 * dkb1

        if USE_PHASE:
            dcos = (dqh0 * q0 + dqh1 * q1) + (dkr0 * k0 + dkr1 * k1)
            dsin = (-dqh0 * q1 + dqh1 * q0) + (-dkr0 * k1 + dkr1 * k0)
            c_base = b * stride_cb + t * stride_ct + h * stride_ch + offs_p * stride_cd
            tl.store(DCos_ptr + c_base, dcos.to(DCos_ptr.dtype.element_ty))
            tl.store(DSin_ptr + c_base, dsin.to(DSin_ptr.dtype.element_ty))
            dqn0 = dqh0 * cs + dqh1 * sn
            dqn1 = -dqh0 * sn + dqh1 * cs
            dkn0 = dkr0 * cs + dkr1 * sn
            dkn1 = -dkr0 * sn + dkr1 * cs
        else:
            dqn0 = dqh0
            dqn1 = dqh1
            dkn0 = dkr0
            dkn1 = dkr1

        if USE_L2:
            q_dot = tl.sum(dqn0 * q0 + dqn1 * q1, axis=0)
            k_dot = tl.sum(dkn0 * k0 + dkn1 * k1, axis=0)
            dq0 = rq * (dqn0 - q0 * q_dot)
            dq1 = rq * (dqn1 - q1 * q_dot)
            dk0 = rk * (dkn0 - k0 * k_dot)
            dk1 = rk * (dkn1 - k1 * k_dot)
        else:
            dq0 = dqn0
            dq1 = dqn1
            dk0 = dkn0
            dk1 = dkn1

        dq_p = DQ_ptr + b * stride_qb + t * stride_qt + h * stride_qh
        dk_p = DK_ptr + b * stride_kb + t * stride_kt + h * stride_kh
        tl.store(dq_p + d0 * stride_qd, dq0.to(DQ_ptr.dtype.element_ty))
        tl.store(dq_p + d1 * stride_qd, dq1.to(DQ_ptr.dtype.element_ty))
        tl.store(dk_p + d0 * stride_kd, dk0.to(DK_ptr.dtype.element_ty))
        tl.store(dk_p + d1 * stride_kd, dk1.to(DK_ptr.dtype.element_ty))

        offs_v = tl.arange(0, D_V)
        vv = tl.load(V_ptr + b * stride_vb + t * stride_vt + h * stride_vh + offs_v * stride_vd).to(tl.float32)
        wg = tl.load(WGate_ptr + b * stride_wgb + t * stride_wgt + h * stride_wgh + offs_v * stride_wgd).to(tl.float32)
        sw = tl.where(wg > 0.0, tl.sqrt(tl.where(wg > 0.0, wg, 1.0)), 0.0)
        out_v_off = b * stride_vtb + t * stride_vtt + h * stride_vth + offs_v * stride_vtd
        dvtc = tl.load(DVTc_ptr + out_v_off).to(tl.float32)
        dvb = tl.load(DVBase_ptr + out_v_off).to(tl.float32)
        dv_tot = dvtc + sw * dvb
        dwg = tl.where(wg > 0.0, 0.5 * dvb * vv / tl.maximum(sw, 1e-6), 0.0)
        tl.store(
            DV_ptr + b * stride_vb + t * stride_vt + h * stride_vh + offs_v * stride_vd,
            dv_tot.to(DV_ptr.dtype.element_ty),
        )
        tl.store(
            DWGate_ptr + b * stride_wgb + t * stride_wgt + h * stride_wgh + offs_v * stride_wgd,
            dwg.to(DWGate_ptr.dtype.element_ty),
        )

    class _TritonScanPrepFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            b_gate,
            w_gate,
            cos_t,
            sin_t,
            scale: float,
            use_phase: bool,
            use_l2: bool,
            tc_dt: torch.dtype,
        ):
            ctx.set_materialize_grads(False)
            B, T, H, D_k = q.shape
            D_v = v.shape[-1]
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            bg_c = b_gate.contiguous()
            wg_c = w_gate.contiguous()
            cos_c = cos_t.contiguous() if use_phase else q_c.new_empty((1, 1, 1, 1), dtype=torch.float32)
            sin_c = sin_t.contiguous() if use_phase else q_c.new_empty((1, 1, 1, 1), dtype=torch.float32)

            q_hat = torch.empty((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            k_hat = torch.empty((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            v_tc = torch.empty((B, T, H, D_v), device=q_c.device, dtype=tc_dt)
            k_base = torch.empty((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            v_base = torch.empty((B, T, H, D_v), device=q_c.device, dtype=tc_dt)

            _scan_prep_fwd_kernel[(B * T, H)](
                q_c,
                k_c,
                v_c,
                bg_c,
                wg_c,
                cos_c,
                sin_c,
                q_hat,
                k_hat,
                v_tc,
                k_base,
                v_base,
                *q_c.stride(),
                *k_c.stride(),
                *v_c.stride(),
                *bg_c.stride(),
                *wg_c.stride(),
                *cos_c.stride(),
                *q_hat.stride(),
                *v_tc.stride(),
                T,
                H,
                scale,
                D_HALF=D_k // 2,
                D_V=D_v,
                USE_PHASE=use_phase,
                USE_L2=use_l2,
                num_warps=4,
                num_stages=1,
            )
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(q_c, k_c, v_c, bg_c, wg_c, cos_c, sin_c)
                ctx.meta = (scale, use_phase, use_l2, tc_dt)
            return q_hat, k_hat, v_tc, k_base, v_base

        @staticmethod
        def backward(ctx, dq_hat, dk_hat, dv_tc, dk_base, dv_base):
            q_c, k_c, v_c, bg_c, wg_c, cos_c, sin_c = ctx.saved_tensors
            scale, use_phase, use_l2, tc_dt = ctx.meta
            B, T, H, D_k = q_c.shape
            D_v = v_c.shape[-1]

            dqh_c = (
                dq_hat.contiguous()
                if dq_hat is not None
                else torch.zeros((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            )
            dkh_c = (
                dk_hat.contiguous()
                if dk_hat is not None
                else torch.zeros((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            )
            dvtc_c = (
                dv_tc.contiguous() if dv_tc is not None else torch.zeros((B, T, H, D_v), device=q_c.device, dtype=tc_dt)
            )
            dkb_c = (
                dk_base.contiguous()
                if dk_base is not None
                else torch.zeros((B, T, H, D_k), device=q_c.device, dtype=tc_dt)
            )
            dvb_c = (
                dv_base.contiguous()
                if dv_base is not None
                else torch.zeros((B, T, H, D_v), device=q_c.device, dtype=tc_dt)
            )

            dq = torch.empty_like(q_c)
            dk = torch.empty_like(k_c)
            dv = torch.empty_like(v_c)
            dbg = torch.empty_like(bg_c)
            dwg = torch.empty_like(wg_c)
            dcos = torch.empty_like(cos_c) if use_phase else cos_c
            dsin = torch.empty_like(sin_c) if use_phase else sin_c

            _scan_prep_bwd_kernel[(B * T, H)](
                q_c,
                k_c,
                v_c,
                bg_c,
                wg_c,
                cos_c,
                sin_c,
                dqh_c,
                dkh_c,
                dvtc_c,
                dkb_c,
                dvb_c,
                dq,
                dk,
                dv,
                dbg,
                dwg,
                dcos,
                dsin,
                *q_c.stride(),
                *k_c.stride(),
                *v_c.stride(),
                *bg_c.stride(),
                *wg_c.stride(),
                *cos_c.stride(),
                *dqh_c.stride(),
                *dvtc_c.stride(),
                T,
                H,
                scale,
                D_HALF=D_k // 2,
                D_V=D_v,
                USE_PHASE=use_phase,
                USE_L2=use_l2,
                num_warps=4,
                num_stages=1,
            )
            return (
                dq if ctx.needs_input_grad[0] else None,
                dk if ctx.needs_input_grad[1] else None,
                dv if ctx.needs_input_grad[2] else None,
                dbg if ctx.needs_input_grad[3] else None,
                dwg if ctx.needs_input_grad[4] else None,
                dcos if (use_phase and ctx.needs_input_grad[5]) else None,
                dsin if (use_phase and ctx.needs_input_grad[6]) else None,
                None,
                None,
                None,
                None,
            )

    @triton.jit
    def _boltzmann_blend_fwd_kernel(
        OLoc_ptr,
        OFar_ptr,
        LSE_ptr,
        MaxSc_ptr,
        Counts_ptr,
        LogKappa_ptr,
        Tau_ptr,
        Out_ptr,
        stride_olb,
        stride_olt,
        stride_olh,
        stride_old,
        stride_ofb,
        stride_oft,
        stride_ofh,
        stride_ofd,
        stride_lb,
        stride_lt,
        stride_lh,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        T_len,
        H,
        SCALAR_GATE: tl.constexpr,
        D_V: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len

        l_off = b * stride_lb + t * stride_lt + h * stride_lh
        cnt = tl.load(Counts_ptr + l_off)
        offs_v = tl.arange(0, D_V)
        o_far = tl.load(OFar_ptr + b * stride_ofb + t * stride_oft + h * stride_ofh + offs_v * stride_ofd).to(
            tl.float32
        )

        if cnt > 0:
            lse = tl.load(LSE_ptr + l_off).to(tl.float32)
            msc = tl.load(MaxSc_ptr + l_off).to(tl.float32)
            n_eff = cnt.to(tl.float32)
            n_gum = tl.maximum(n_eff, 2.0)
            stat = 0.5 * (lse - tl.log(n_eff)) + 0.5 * (msc - tl.sqrt(2.0 * tl.log(n_gum)))
            g_idx = 0 if SCALAR_GATE else h
            kappa = tl.exp(tl.load(LogKappa_ptr + g_idx).to(tl.float32))
            tau = tl.load(Tau_ptr + g_idx).to(tl.float32)
            p_hit = tl.sigmoid(kappa * (stat - tau))
            o_loc = tl.load(OLoc_ptr + b * stride_olb + t * stride_olt + h * stride_olh + offs_v * stride_old).to(
                tl.float32
            )
            res = p_hit * o_loc + (1.0 - p_hit) * o_far
        else:
            res = o_far

        tl.store(
            Out_ptr + b * stride_ob + t * stride_ot + h * stride_oh + offs_v * stride_od,
            res.to(Out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _boltzmann_blend_bwd_kernel(
        OLoc_ptr,
        OFar_ptr,
        LSE_ptr,
        MaxSc_ptr,
        Counts_ptr,
        LogKappa_ptr,
        Tau_ptr,
        DOut_ptr,
        DOLoc_ptr,
        DOFar_ptr,
        DLSE_ptr,
        DMaxSc_ptr,
        DLogKappaPart_ptr,
        DTauPart_ptr,
        stride_olb,
        stride_olt,
        stride_olh,
        stride_old,
        stride_ofb,
        stride_oft,
        stride_ofh,
        stride_ofd,
        stride_lb,
        stride_lt,
        stride_lh,
        stride_dob,
        stride_dot,
        stride_doh,
        stride_dod,
        T_len,
        H,
        SCALAR_GATE: tl.constexpr,
        D_V: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len

        l_off = b * stride_lb + t * stride_lt + h * stride_lh
        cnt = tl.load(Counts_ptr + l_off)
        offs_v = tl.arange(0, D_V)
        do = tl.load(DOut_ptr + b * stride_dob + t * stride_dot + h * stride_doh + offs_v * stride_dod).to(tl.float32)
        o_far = tl.load(OFar_ptr + b * stride_ofb + t * stride_oft + h * stride_ofh + offs_v * stride_ofd).to(
            tl.float32
        )
        part_idx = bt * H + h

        if cnt > 0:
            lse = tl.load(LSE_ptr + l_off).to(tl.float32)
            msc = tl.load(MaxSc_ptr + l_off).to(tl.float32)
            n_eff = cnt.to(tl.float32)
            n_gum = tl.maximum(n_eff, 2.0)
            stat = 0.5 * (lse - tl.log(n_eff)) + 0.5 * (msc - tl.sqrt(2.0 * tl.log(n_gum)))
            g_idx = 0 if SCALAR_GATE else h
            kappa = tl.exp(tl.load(LogKappa_ptr + g_idx).to(tl.float32))
            tau = tl.load(Tau_ptr + g_idx).to(tl.float32)
            p_hit = tl.sigmoid(kappa * (stat - tau))
            o_loc = tl.load(OLoc_ptr + b * stride_olb + t * stride_olt + h * stride_olh + offs_v * stride_old).to(
                tl.float32
            )
            do_loc = p_hit * do
            do_far = (1.0 - p_hit) * do
            dp_hit = tl.sum(do * (o_loc - o_far), axis=0)
            dz = dp_hit * p_hit * (1.0 - p_hit)
            dstat = dz * kappa
            tl.store(DLSE_ptr + l_off, 0.5 * dstat)
            tl.store(DMaxSc_ptr + l_off, 0.5 * dstat)
            tl.store(DLogKappaPart_ptr + part_idx, dz * kappa * (stat - tau))
            tl.store(DTauPart_ptr + part_idx, -dz * kappa)
        else:
            do_loc = tl.zeros([D_V], dtype=tl.float32)
            do_far = do
            tl.store(DLSE_ptr + l_off, 0.0)
            tl.store(DMaxSc_ptr + l_off, 0.0)
            tl.store(DLogKappaPart_ptr + part_idx, 0.0)
            tl.store(DTauPart_ptr + part_idx, 0.0)

        tl.store(
            DOLoc_ptr + b * stride_olb + t * stride_olt + h * stride_olh + offs_v * stride_old,
            do_loc.to(DOLoc_ptr.dtype.element_ty),
        )
        tl.store(DOFar_ptr + b * stride_ofb + t * stride_oft + h * stride_ofh + offs_v * stride_ofd, do_far)

    class _TritonBoltzmannBlendFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, o_loc, o_far, lse, max_sc, counts, log_kappa, tau, out_dtype: torch.dtype):
            ctx.set_materialize_grads(False)
            B, T, H, D_v = o_far.shape
            ol_c = o_loc.contiguous()
            of_c = o_far.contiguous()
            lse_c = lse.contiguous()
            msc_c = max_sc.contiguous()
            cnt_c = counts.contiguous()
            lk_c = log_kappa.float().contiguous()
            tau_c = tau.float().contiguous()
            scalar_gate = lk_c.numel() == 1

            out = torch.empty((B, T, H, D_v), device=of_c.device, dtype=out_dtype)
            _boltzmann_blend_fwd_kernel[(B * T, H)](
                ol_c,
                of_c,
                lse_c,
                msc_c,
                cnt_c,
                lk_c,
                tau_c,
                out,
                *ol_c.stride(),
                *of_c.stride(),
                *lse_c.stride(),
                *out.stride(),
                T,
                H,
                SCALAR_GATE=scalar_gate,
                D_V=D_v,
                num_warps=4,
                num_stages=1,
            )
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(ol_c, of_c, lse_c, msc_c, cnt_c, lk_c, tau_c)
                ctx.meta = (scalar_gate, log_kappa.dtype, tau.dtype)
            return out

        @staticmethod
        def backward(ctx, dout):
            ol_c, of_c, lse_c, msc_c, cnt_c, lk_c, tau_c = ctx.saved_tensors
            scalar_gate, lk_dt, tau_dt = ctx.meta
            B, T, H, D_v = of_c.shape
            if dout is None:
                return None, None, None, None, None, None, None, None
            do_c = dout.contiguous()
            do_loc = torch.empty_like(ol_c)
            do_far = torch.empty_like(of_c)
            dlse = torch.empty_like(lse_c)
            dmsc = torch.empty_like(msc_c)
            dlk_part = torch.empty((B * T * H,), device=of_c.device, dtype=torch.float32)
            dtau_part = torch.empty((B * T * H,), device=of_c.device, dtype=torch.float32)

            _boltzmann_blend_bwd_kernel[(B * T, H)](
                ol_c,
                of_c,
                lse_c,
                msc_c,
                cnt_c,
                lk_c,
                tau_c,
                do_c,
                do_loc,
                do_far,
                dlse,
                dmsc,
                dlk_part,
                dtau_part,
                *ol_c.stride(),
                *of_c.stride(),
                *lse_c.stride(),
                *do_c.stride(),
                T,
                H,
                SCALAR_GATE=scalar_gate,
                D_V=D_v,
                num_warps=4,
                num_stages=1,
            )
            if scalar_gate:
                dlk = dlk_part.sum().view_as(lk_c).to(lk_dt)
                dtau = dtau_part.sum().view_as(tau_c).to(tau_dt)
            else:
                dlk = dlk_part.view(B * T, H).sum(dim=0).to(lk_dt)
                dtau = dtau_part.view(B * T, H).sum(dim=0).to(tau_dt)
            return (
                do_loc if ctx.needs_input_grad[0] else None,
                do_far if ctx.needs_input_grad[1] else None,
                dlse if ctx.needs_input_grad[2] else None,
                dmsc if ctx.needs_input_grad[3] else None,
                None,
                dlk if ctx.needs_input_grad[5] else None,
                dtau if ctx.needs_input_grad[6] else None,
                None,
            )

    @triton.jit
    def _proj_gates_swa_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Beta_ptr,
        AllUp_ptr,
        QNormW_ptr,
        KNormW_ptr,
        QOut_ptr,
        KOut_ptr,
        BGate_ptr,
        WGate_ptr,
        QSwa_ptr,
        KSwa_ptr,
        VSwa_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_bb,
        stride_bt,
        stride_bh,
        stride_aub,
        stride_aut,
        stride_aud,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_ovb,
        stride_ovt,
        stride_ovh,
        stride_ovd,
        T_len,
        H_v,
        gqa_q,
        gqa_k,
        cap_g,
        root_dk,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        USE_L2: tl.constexpr,
        PAIRED_B: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len
        hq = h // gqa_q
        hk = h // gqa_k

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)

        qf = tl.load(Q_ptr + b * stride_qb + t * stride_qt + hq * stride_qh + offs_dk * stride_qd).to(tl.float32)
        kf = tl.load(K_ptr + b * stride_kb + t * stride_kt + hk * stride_kh + offs_dk * stride_kd).to(tl.float32)
        vf = tl.load(V_ptr + b * stride_vb + t * stride_vt + h * stride_vh + offs_dv * stride_vd).to(tl.float32)

        if USE_L2:
            qf = qf * tl.rsqrt(tl.sum(qf * qf, axis=0) + 1e-6)
            kf = kf * tl.rsqrt(tl.sum(kf * kf, axis=0) + 1e-6)

        au_base = AllUp_ptr + b * stride_aub + t * stride_aut
        er_off = h * D_K + offs_dk
        wr_off = H_v * D_K + h * D_V + offs_dv
        sq_off = H_v * (D_K + D_V) + h * D_K + offs_dk
        sk_off = H_v * (2 * D_K + D_V) + h * D_K + offs_dk
        sv_off = H_v * (3 * D_K + D_V) + h * D_V + offs_dv

        raw_er = tl.load(au_base + er_off * stride_aud).to(tl.float32)
        raw_wr = tl.load(au_base + wr_off * stride_aud).to(tl.float32)
        dq_s = tl.load(au_base + sq_off * stride_aud).to(tl.float32)
        dk_s = tl.load(au_base + sk_off * stride_aud).to(tl.float32)
        dv_s = tl.load(au_base + sv_off * stride_aud).to(tl.float32)

        # tanh(x) = 2 * sigmoid(2 * x) - 1 (immune to exp overflow on large inputs!)
        th_er = 2.0 * tl.sigmoid(2.0 * raw_er) - 1.0
        th_wr = 2.0 * tl.sigmoid(2.0 * raw_wr) - 1.0
        d_er = cap_g * th_er
        d_wr = cap_g * th_wr

        beta = tl.minimum(
            tl.maximum(tl.load(Beta_ptr + b * stride_bb + t * stride_bt + h * stride_bh).to(tl.float32), 0.0), 1.0
        )
        bh_val = 0.5 * beta
        s_e = 2.0 * tl.sigmoid(d_er) - 1.0
        den_e = tl.maximum(1.0 + (2.0 * bh_val - 1.0) * s_e, 1e-12)
        bg = 2.0 * tl.minimum(tl.maximum(bh_val + (2.0 * bh_val * (1.0 - bh_val) * s_e) / den_e, 0.0), 1.0)
        if PAIRED_B:
            bg_2d = tl.reshape(bg, [D_K // 2, 2])
            bg_m = tl.sum(bg_2d, axis=1) * 0.5
            bg = tl.reshape(tl.broadcast_to(bg_m[:, None], [D_K // 2, 2]), [D_K])

        s_w = 2.0 * tl.sigmoid(d_wr) - 1.0
        den_w = tl.maximum(1.0 + (2.0 * beta - 1.0) * s_w, 1e-12)
        wg = tl.minimum(tl.maximum(beta + (2.0 * beta * (1.0 - beta) * s_w) / den_w, 0.0), 1.0)

        qx = root_dk * qf + dq_s
        kx = root_dk * kf + dk_s
        rq_s = tl.rsqrt(tl.sum(qx * qx, axis=0) * (1.0 / D_K) + 1e-6)
        rk_s = tl.rsqrt(tl.sum(kx * kx, axis=0) * (1.0 / D_K) + 1e-6)
        qnw = tl.load(QNormW_ptr + offs_dk).to(tl.float32)
        knw = tl.load(KNormW_ptr + offs_dk).to(tl.float32)
        q_swa = qx * rq_s * qnw
        k_swa = kx * rk_s * knw
        v_swa = vf + dv_s

        out_dt = QOut_ptr.dtype.element_ty
        o_k = b * stride_ob + t * stride_ot + h * stride_oh + offs_dk * stride_od
        o_v = b * stride_ovb + t * stride_ovt + h * stride_ovh + offs_dv * stride_ovd
        tl.store(QOut_ptr + o_k, qf.to(out_dt))
        tl.store(KOut_ptr + o_k, kf.to(out_dt))
        tl.store(BGate_ptr + o_k, bg.to(BGate_ptr.dtype.element_ty))
        tl.store(WGate_ptr + o_v, wg.to(WGate_ptr.dtype.element_ty))
        tl.store(QSwa_ptr + o_k, q_swa.to(out_dt))
        tl.store(KSwa_ptr + o_k, k_swa.to(out_dt))
        tl.store(VSwa_ptr + o_v, v_swa.to(out_dt))

    @triton.jit
    def _proj_gates_swa_bwd_kernel(
        Q_ptr,
        K_ptr,
        Beta_ptr,
        AllUp_ptr,
        QNormW_ptr,
        KNormW_ptr,
        DQOut_ptr,
        DKOut_ptr,
        DBGate_ptr,
        DWGate_ptr,
        DQSwa_ptr,
        DKSwa_ptr,
        DVSwa_ptr,
        DQFull_ptr,
        DKFull_ptr,
        DVSub_ptr,
        DBeta_ptr,
        DAllUp_ptr,
        DQNormPart_ptr,
        DKNormPart_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_bb,
        stride_bt,
        stride_bh,
        stride_aub,
        stride_aut,
        stride_aud,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_ovb,
        stride_ovt,
        stride_ovh,
        stride_ovd,
        T_len,
        H_v,
        gqa_q,
        gqa_k,
        cap_g,
        root_dk,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        USE_L2: tl.constexpr,
        PAIRED_B: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // T_len
        t = bt % T_len
        hq = h // gqa_q
        hk = h // gqa_k

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)

        qf_raw = tl.load(Q_ptr + b * stride_qb + t * stride_qt + hq * stride_qh + offs_dk * stride_qd).to(tl.float32)
        kf_raw = tl.load(K_ptr + b * stride_kb + t * stride_kt + hk * stride_kh + offs_dk * stride_kd).to(tl.float32)
        if USE_L2:
            rq_l2 = tl.rsqrt(tl.sum(qf_raw * qf_raw, axis=0) + 1e-6)
            rk_l2 = tl.rsqrt(tl.sum(kf_raw * kf_raw, axis=0) + 1e-6)
            qf = qf_raw * rq_l2
            kf = kf_raw * rk_l2
        else:
            rq_l2 = 1.0
            rk_l2 = 1.0
            qf = qf_raw
            kf = kf_raw

        au_base = AllUp_ptr + b * stride_aub + t * stride_aut
        er_off = h * D_K + offs_dk
        wr_off = H_v * D_K + h * D_V + offs_dv
        sq_off = H_v * (D_K + D_V) + h * D_K + offs_dk
        sk_off = H_v * (2 * D_K + D_V) + h * D_K + offs_dk
        sv_off = H_v * (3 * D_K + D_V) + h * D_V + offs_dv

        raw_er = tl.load(au_base + er_off * stride_aud).to(tl.float32)
        raw_wr = tl.load(au_base + wr_off * stride_aud).to(tl.float32)
        dq_s = tl.load(au_base + sq_off * stride_aud).to(tl.float32)
        dk_s = tl.load(au_base + sk_off * stride_aud).to(tl.float32)

        o_k = b * stride_ob + t * stride_ot + h * stride_oh + offs_dk * stride_od
        o_v = b * stride_ovb + t * stride_ovt + h * stride_ovh + offs_dv * stride_ovd
        dq_o = tl.load(DQOut_ptr + o_k).to(tl.float32)
        dk_o = tl.load(DKOut_ptr + o_k).to(tl.float32)
        dbg = tl.load(DBGate_ptr + o_k).to(tl.float32)
        dwg = tl.load(DWGate_ptr + o_v).to(tl.float32)
        dqs = tl.load(DQSwa_ptr + o_k).to(tl.float32)
        dks = tl.load(DKSwa_ptr + o_k).to(tl.float32)
        dvs = tl.load(DVSwa_ptr + o_v).to(tl.float32)

        if PAIRED_B:
            dbg_2d = tl.reshape(dbg, [D_K // 2, 2])
            dbg_m = tl.sum(dbg_2d, axis=1) * 0.5
            dbg = tl.reshape(tl.broadcast_to(dbg_m[:, None], [D_K // 2, 2]), [D_K])

        # Exact derivatives of 2*logistic_shift(0.5*beta, d_er) and logistic_shift(beta, d_wr)
        beta_raw = tl.load(Beta_ptr + b * stride_bb + t * stride_bt + h * stride_bh).to(tl.float32)
        in_b_range = (beta_raw > 0.0) & (beta_raw < 1.0)
        beta = tl.minimum(tl.maximum(beta_raw, 0.0), 1.0)
        bh_val = 0.5 * beta

        th_er = 2.0 * tl.sigmoid(2.0 * raw_er) - 1.0
        d_er = cap_g * th_er
        s_e = 2.0 * tl.sigmoid(d_er) - 1.0
        den_e = tl.maximum(1.0 + (2.0 * bh_val - 1.0) * s_e, 1e-12)
        qe = (1.0 - s_e * s_e) / (den_e * den_e)
        # bg = 2.0 * ls_e => d(ls_e) = 2.0 * dbg; d(bh_val) = 0.5 * d(beta) => factor 2.0 * 0.5 == 1.0!
        dd_er = (2.0 * dbg) * bh_val * (1.0 - bh_val) * qe
        draw_er = dd_er * cap_g * (1.0 - th_er * th_er)
        dbeta_e = tl.sum(dbg * qe, axis=0)

        th_wr = 2.0 * tl.sigmoid(2.0 * raw_wr) - 1.0
        d_wr = cap_g * th_wr
        s_w = 2.0 * tl.sigmoid(d_wr) - 1.0
        den_w = tl.maximum(1.0 + (2.0 * beta - 1.0) * s_w, 1e-12)
        qw = (1.0 - s_w * s_w) / (den_w * den_w)
        dd_wr = dwg * beta * (1.0 - beta) * qw
        draw_wr = dd_wr * cap_g * (1.0 - th_wr * th_wr)
        dbeta_w = tl.sum(dwg * qw, axis=0)
        dbeta_tot = tl.where(in_b_range, dbeta_e + dbeta_w, 0.0)
        tl.store(DBeta_ptr + b * stride_bb + t * stride_bt + h * stride_bh, dbeta_tot.to(DBeta_ptr.dtype.element_ty))

        # SWA RMSNorm backward
        qx = root_dk * qf + dq_s
        kx = root_dk * kf + dk_s
        inv_dk = 1.0 / D_K
        rq_s = tl.rsqrt(tl.sum(qx * qx, axis=0) * inv_dk + 1e-6)
        rk_s = tl.rsqrt(tl.sum(kx * kx, axis=0) * inv_dk + 1e-6)
        qn = qx * rq_s
        kn = kx * rk_s
        qnw = tl.load(QNormW_ptr + offs_dk).to(tl.float32)
        knw = tl.load(KNormW_ptr + offs_dk).to(tl.float32)

        row_id = bt * H_v + h
        tl.store(DQNormPart_ptr + row_id * D_K + offs_dk, dqs * qn)
        tl.store(DKNormPart_ptr + row_id * D_K + offs_dk, dks * kn)

        dqn = dqs * qnw
        dkn = dks * knw
        rq_s_bwd = tl.minimum(rq_s, 10.0)
        rk_s_bwd = tl.minimum(rk_s, 10.0)
        ddq_s = rq_s_bwd * (dqn - qn * (tl.sum(dqn * qn, axis=0) * inv_dk))
        ddk_s = rk_s_bwd * (dkn - kn * (tl.sum(dkn * kn, axis=0) * inv_dk))

        dau_base = DAllUp_ptr + b * stride_aub + t * stride_aut
        tl.store(dau_base + er_off * stride_aud, draw_er)
        tl.store(dau_base + wr_off * stride_aud, draw_wr)
        tl.store(dau_base + sq_off * stride_aud, ddq_s)
        tl.store(dau_base + sk_off * stride_aud, ddk_s)
        tl.store(dau_base + sv_off * stride_aud, dvs)

        dq_prep = dq_o + root_dk * ddq_s
        dk_prep = dk_o + root_dk * ddk_s
        if USE_L2:
            dq_h = rq_l2 * (dq_prep - qf * tl.sum(dq_prep * qf, axis=0))
            dk_h = rk_l2 * (dk_prep - kf * tl.sum(dk_prep * kf, axis=0))
        else:
            dq_h = dq_prep
            dk_h = dk_prep

        tl.store(DQFull_ptr + o_k, dq_h)
        tl.store(DKFull_ptr + o_k, dk_h)
        tl.store(DVSub_ptr + o_v, dvs)

    class _FusedAdapterProjFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q_raw,
            k_raw,
            v_raw,
            h_f,
            beta_base,
            w_er,
            w_wr,
            w_sq,
            w_sk,
            w_sv,
            q_norm_w,
            k_norm_w,
            hv: int,
            cap_g: float,
            use_l2: bool,
            paired_b: bool,
        ):
            ctx.set_materialize_grads(False)
            B, T, H_q, D_k = q_raw.shape
            H_k = k_raw.shape[2]
            D_v = v_raw.shape[-1]
            out_dt = q_raw.dtype

            cat_w = torch.cat([w_er.float(), w_wr.float(), w_sq.float(), w_sk.float(), w_sv.float()], dim=0)
            all_up = F.linear(h_f, cat_w).contiguous()

            qc = q_raw.contiguous()
            kc = k_raw.contiguous()
            vc = v_raw.contiguous()
            bc = beta_base.contiguous()
            qnw_c = q_norm_w.float().contiguous()
            knw_c = k_norm_w.float().contiguous()

            q_out = torch.empty((B, T, hv, D_k), device=qc.device, dtype=out_dt)
            k_out = torch.empty((B, T, hv, D_k), device=qc.device, dtype=out_dt)
            b_gate = torch.empty((B, T, hv, D_k), device=qc.device, dtype=torch.float32)
            w_gate = torch.empty((B, T, hv, D_v), device=qc.device, dtype=torch.float32)
            q_swa = torch.empty((B, T, hv, D_k), device=qc.device, dtype=out_dt)
            k_swa = torch.empty((B, T, hv, D_k), device=qc.device, dtype=out_dt)
            v_swa = torch.empty((B, T, hv, D_v), device=qc.device, dtype=out_dt)

            _proj_gates_swa_fwd_kernel[(B * T, hv)](
                qc,
                kc,
                vc,
                bc,
                all_up,
                qnw_c,
                knw_c,
                q_out,
                k_out,
                b_gate,
                w_gate,
                q_swa,
                k_swa,
                v_swa,
                *qc.stride(),
                *kc.stride(),
                *vc.stride(),
                *bc.stride(),
                *all_up.stride(),
                *q_out.stride(),
                *w_gate.stride(),
                T,
                hv,
                hv // H_q,
                hv // H_k,
                float(cap_g),
                float(math.sqrt(D_k)),
                D_K=D_k,
                D_V=D_v,
                USE_L2=use_l2,
                PAIRED_B=paired_b,
                num_warps=4,
                num_stages=1,
            )
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(qc, kc, h_f, bc, w_er, w_wr, w_sq, w_sk, w_sv, qnw_c, knw_c)
                ctx.meta = (hv, cap_g, use_l2, paired_b, v_raw.dtype, q_norm_w.dtype, k_norm_w.dtype)
            return q_out, k_out, b_gate, w_gate, q_swa, k_swa, v_swa

        @staticmethod
        def backward(ctx, dq_out, dk_out, db_gate, dw_gate, dq_swa, dk_swa, dv_swa):
            qc, kc, h_f, bc, w_er, w_wr, w_sq, w_sk, w_sv, qnw_c, knw_c = ctx.saved_tensors
            hv, cap_g, use_l2, paired_b, v_dt, qnw_dt, knw_dt = ctx.meta
            B, T, H_q, D_k = qc.shape
            H_k = kc.shape[2]
            D_v = w_wr.shape[0] // hv
            out_dt = qc.dtype

            cat_w = torch.cat([w_er.float(), w_wr.float(), w_sq.float(), w_sk.float(), w_sv.float()], dim=0)
            all_up = F.linear(h_f, cat_w).contiguous()

            def _z_k() -> torch.Tensor:
                return torch.zeros((B, T, hv, D_k), device=qc.device, dtype=out_dt)

            def _z_v() -> torch.Tensor:
                return torch.zeros((B, T, hv, D_v), device=qc.device, dtype=out_dt)

            dqo_c = dq_out.contiguous() if dq_out is not None else _z_k()
            dko_c = dk_out.contiguous() if dk_out is not None else _z_k()
            dbg_c = (
                db_gate.contiguous()
                if db_gate is not None
                else torch.zeros((B, T, hv, D_k), device=qc.device, dtype=torch.float32)
            )
            dwg_c = (
                dw_gate.contiguous()
                if dw_gate is not None
                else torch.zeros((B, T, hv, D_v), device=qc.device, dtype=torch.float32)
            )
            dqs_c = dq_swa.contiguous() if dq_swa is not None else _z_k()
            dks_c = dk_swa.contiguous() if dk_swa is not None else _z_k()
            dvs_c = dv_swa.contiguous() if dv_swa is not None else _z_v()

            dq_full = torch.empty((B, T, hv, D_k), device=qc.device, dtype=torch.float32)
            dk_full = torch.empty((B, T, hv, D_k), device=qc.device, dtype=torch.float32)
            dv_sub = torch.empty((B, T, hv, D_v), device=qc.device, dtype=torch.float32)
            dbeta = torch.empty_like(bc)
            d_all_up = torch.empty_like(all_up)
            dqn_part = torch.empty((B * T * hv, D_k), device=qc.device, dtype=torch.float32)
            dkn_part = torch.empty((B * T * hv, D_k), device=qc.device, dtype=torch.float32)

            _proj_gates_swa_bwd_kernel[(B * T, hv)](
                qc,
                kc,
                bc,
                all_up,
                qnw_c,
                knw_c,
                dqo_c,
                dko_c,
                dbg_c,
                dwg_c,
                dqs_c,
                dks_c,
                dvs_c,
                dq_full,
                dk_full,
                dv_sub,
                dbeta,
                d_all_up,
                dqn_part,
                dkn_part,
                *qc.stride(),
                *kc.stride(),
                *bc.stride(),
                *all_up.stride(),
                *dqo_c.stride(),
                *dwg_c.stride(),
                T,
                hv,
                hv // H_q,
                hv // H_k,
                float(cap_g),
                float(math.sqrt(D_k)),
                D_K=D_k,
                D_V=D_v,
                USE_L2=use_l2,
                PAIRED_B=paired_b,
                num_warps=4,
                num_stages=1,
            )

            gqa_q = hv // H_q
            gqa_k = hv // H_k
            dq_raw = dq_full.view(B, T, H_q, gqa_q, D_k).sum(dim=3).to(out_dt) if gqa_q > 1 else dq_full.to(out_dt)
            dk_raw = dk_full.view(B, T, H_k, gqa_k, D_k).sum(dim=3).to(out_dt) if gqa_k > 1 else dk_full.to(out_dt)
            dv_raw = dv_sub.to(v_dt)

            dau_2d = d_all_up.view(B * T, -1)
            hf_2d = h_f.reshape(B * T, -1)
            dh = (dau_2d @ cat_w).view_as(h_f) if ctx.needs_input_grad[3] else None
            dcat_w = dau_2d.t() @ hf_2d
            dw_er, dw_wr, dw_sq, dw_sk, dw_sv = dcat_w.split([hv * D_k, hv * D_v, hv * D_k, hv * D_k, hv * D_v], dim=0)
            dqnw = dqn_part.sum(dim=0).to(qnw_dt) if ctx.needs_input_grad[10] else None
            dknw = dkn_part.sum(dim=0).to(knw_dt) if ctx.needs_input_grad[11] else None

            return (
                dq_raw if ctx.needs_input_grad[0] else None,
                dk_raw if ctx.needs_input_grad[1] else None,
                dv_raw if ctx.needs_input_grad[2] else None,
                dh,
                dbeta if ctx.needs_input_grad[4] else None,
                dw_er.to(w_er.dtype) if ctx.needs_input_grad[5] else None,
                dw_wr.to(w_wr.dtype) if ctx.needs_input_grad[6] else None,
                dw_sq.to(w_sq.dtype) if ctx.needs_input_grad[7] else None,
                dw_sk.to(w_sk.dtype) if ctx.needs_input_grad[8] else None,
                dw_sv.to(w_sv.dtype) if ctx.needs_input_grad[9] else None,
                dqnw,
                dknw,
                None,
                None,
                None,
                None,
            )

    @triton.jit
    def _disjoint_swa_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Valid_ptr,
        Slope_ptr,
        Out_ptr,
        LSE_ptr,
        Max_ptr,
        Argmax_ptr,
        Counts_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_val_b,
        stride_val_t,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_lb,
        stride_lt,
        stride_lh,
        T_q,
        H,
        scale,
        inv_c,
        CHUNK_C: tl.constexpr,
        WIN_CHUNKS: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        HAS_SLOPE: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        chunk_n = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_start = chunk_n * CHUNK_C
        offs_m = q_start + tl.arange(0, CHUNK_C)
        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        mask_m = offs_m < T_q

        q = tl.load(
            Q_ptr + b * stride_qb + offs_m[:, None] * stride_qt + h * stride_qh + offs_dk[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        slope_h = tl.load(Slope_ptr + h).to(tl.float32) if HAS_SLOPE else 0.0

        m_i = tl.full([CHUNK_C], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([CHUNK_C], dtype=tl.float32)
        acc = tl.zeros([CHUNK_C, D_V], dtype=tl.float32)
        argmax_i = tl.zeros([CHUNK_C], dtype=tl.int32)
        cnt_i = tl.zeros([CHUNK_C], dtype=tl.int32)

        k_far = tl.maximum(0, chunk_n - WIN_CHUNKS)
        for c_k in range(k_far, chunk_n + 1):
            k_start = c_k * CHUNK_C
            offs_n = k_start + tl.arange(0, CHUNK_C)
            mask_n = offs_n < T_q
            if HAS_VALID:
                val_n = tl.load(Valid_ptr + b * stride_val_b + offs_n * stride_val_t, mask=mask_n, other=False).to(
                    tl.int1
                )
                mask_n = mask_n & val_n

            k_blk = tl.load(
                K_ptr + b * stride_kb + offs_n[None, :] * stride_kt + h * stride_kh + offs_dk[:, None] * stride_kd,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk = tl.dot(q, k_blk).to(tl.float32) * scale
            if HAS_SLOPE:
                dist = tl.maximum(0, offs_m[:, None] - offs_n[None, :]).to(tl.float32) * inv_c
                qk = qk - slope_h * dist

            valid = mask_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            cnt_i += tl.sum(valid.to(tl.int32), axis=1)
            qk = tl.where(valid, qk, -float("inf"))

            blk_max, blk_idx = tl.max(qk, axis=1, return_indices=True)
            better = blk_max > m_i
            argmax_i = tl.where(better, k_start + blk_idx.to(tl.int32), argmax_i)

            m_ij = tl.maximum(m_i, blk_max)
            has_any = m_ij > -float("inf")
            safe_m_ij = tl.where(has_any, m_ij, 0.0)
            safe_m_i = tl.where(m_i > -float("inf"), m_i, 0.0)
            p = tl.where(valid, tl.exp(qk - safe_m_ij[:, None]), 0.0)
            alpha = tl.where(m_i > -float("inf"), tl.exp(safe_m_i - safe_m_ij), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_blk = tl.load(
                V_ptr + b * stride_vb + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
            m_i = m_ij

        has_v = cnt_i > 0
        safe_l = tl.where(has_v, l_i, 1.0)
        out = tl.where(has_v[:, None], acc / safe_l[:, None], 0.0)
        safe_m = tl.where(has_v, m_i, 0.0)
        lse = tl.where(has_v, safe_m + tl.log(safe_l), 0.0)

        tl.store(
            Out_ptr + b * stride_ob + offs_m[:, None] * stride_ot + h * stride_oh + offs_dv[None, :] * stride_od,
            out.to(Out_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
        l_base = b * stride_lb + offs_m * stride_lt + h * stride_lh
        tl.store(LSE_ptr + l_base, lse, mask=mask_m)
        tl.store(Max_ptr + l_base, safe_m, mask=mask_m)
        tl.store(Argmax_ptr + l_base, tl.where(has_v, argmax_i, -1), mask=mask_m)
        tl.store(Counts_ptr + l_base, cnt_i, mask=mask_m)

    @triton.jit
    def _disjoint_swa_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Valid_ptr,
        Slope_ptr,
        DO_ptr,
        Delta_ptr,
        DLSE_ptr,
        DMax_ptr,
        LSE_ptr,
        Argmax_ptr,
        Counts_ptr,
        DQ_ptr,
        DSlope_part_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_val_b,
        stride_val_t,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_lb,
        stride_lt,
        stride_lh,
        T_q,
        H,
        scale,
        inv_c,
        CHUNK_C: tl.constexpr,
        WIN_CHUNKS: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        HAS_SLOPE: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        chunk_n = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        q_start = chunk_n * CHUNK_C
        offs_m = q_start + tl.arange(0, CHUNK_C)
        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        mask_m = offs_m < T_q

        l_base = b * stride_lb + offs_m * stride_lt + h * stride_lh
        cnt_i = tl.load(Counts_ptr + l_base, mask=mask_m, other=0)
        active_m = mask_m & (cnt_i > 0)

        q = tl.load(
            Q_ptr + b * stride_qb + offs_m[:, None] * stride_qt + h * stride_qh + offs_dk[None, :] * stride_qd,
            mask=active_m[:, None],
            other=0.0,
        )
        do = tl.load(
            DO_ptr + b * stride_ob + offs_m[:, None] * stride_ot + h * stride_oh + offs_dv[None, :] * stride_od,
            mask=active_m[:, None],
            other=0.0,
        )
        lse = tl.load(LSE_ptr + l_base, mask=active_m, other=0.0)
        delta = tl.load(Delta_ptr + l_base, mask=active_m, other=0.0)
        dlse = tl.load(DLSE_ptr + l_base, mask=active_m, other=0.0)
        dmax = tl.load(DMax_ptr + l_base, mask=active_m, other=0.0)
        argmax_i = tl.load(Argmax_ptr + l_base, mask=active_m, other=-1)
        slope_h = tl.load(Slope_ptr + h).to(tl.float32) if HAS_SLOPE else 0.0

        dq_acc = tl.zeros([CHUNK_C, D_K], dtype=tl.float32)
        dslope_acc = 0.0

        k_far = tl.maximum(0, chunk_n - WIN_CHUNKS)
        for c_k in range(k_far, chunk_n + 1):
            k_start = c_k * CHUNK_C
            offs_n = k_start + tl.arange(0, CHUNK_C)
            mask_n = offs_n < T_q
            if HAS_VALID:
                val_n = tl.load(Valid_ptr + b * stride_val_b + offs_n * stride_val_t, mask=mask_n, other=False).to(
                    tl.int1
                )
                mask_n = mask_n & val_n

            k_blk = tl.load(
                K_ptr + b * stride_kb + offs_n[:, None] * stride_kt + h * stride_kh + offs_dk[None, :] * stride_kd,
                mask=mask_n[:, None],
                other=0.0,
            )
            v_blk = tl.load(
                V_ptr + b * stride_vb + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * scale
            dist = tl.maximum(0, offs_m[:, None] - offs_n[None, :]).to(tl.float32) * inv_c
            if HAS_SLOPE:
                qk = qk - slope_h * dist

            valid = active_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v_blk)).to(tl.float32)
            is_max = valid & (offs_n[None, :] == argmax_i[:, None])
            ds = p * (dp - delta[:, None] + dlse[:, None]) + tl.where(is_max, dmax[:, None], 0.0)
            if HAS_SLOPE:
                dslope_acc -= tl.sum(ds * dist)
            dq_acc += tl.dot(ds.to(k_blk.dtype), k_blk)

        dq_acc = dq_acc * scale
        tl.store(
            DQ_ptr + b * stride_qb + offs_m[:, None] * stride_qt + h * stride_qh + offs_dk[None, :] * stride_qd,
            dq_acc.to(DQ_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )
        if HAS_SLOPE:
            tl.store(DSlope_part_ptr + chunk_n * tl.num_programs(1) + bh, dslope_acc)

    @triton.jit
    def _disjoint_swa_bwd_dkv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Valid_ptr,
        Slope_ptr,
        DO_ptr,
        Delta_ptr,
        DLSE_ptr,
        DMax_ptr,
        LSE_ptr,
        Argmax_ptr,
        Counts_ptr,
        DK_ptr,
        DV_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_val_b,
        stride_val_t,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_lb,
        stride_lt,
        stride_lh,
        T_q,
        N_chunks,
        H,
        scale,
        inv_c,
        CHUNK_C: tl.constexpr,
        WIN_CHUNKS: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        HAS_SLOPE: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        chunk_k = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        k_start = chunk_k * CHUNK_C
        offs_n = k_start + tl.arange(0, CHUNK_C)
        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        mask_n_raw = offs_n < T_q
        mask_n = mask_n_raw
        if HAS_VALID:
            val_n = tl.load(Valid_ptr + b * stride_val_b + offs_n * stride_val_t, mask=mask_n_raw, other=False).to(
                tl.int1
            )
            mask_n = mask_n_raw & val_n

        k_blk = tl.load(
            K_ptr + b * stride_kb + offs_n[:, None] * stride_kt + h * stride_kh + offs_dk[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v_blk = tl.load(
            V_ptr + b * stride_vb + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        slope_h = tl.load(Slope_ptr + h).to(tl.float32) if HAS_SLOPE else 0.0

        dk_acc = tl.zeros([CHUNK_C, D_K], dtype=tl.float32)
        dv_acc = tl.zeros([CHUNK_C, D_V], dtype=tl.float32)

        q_end = tl.minimum(N_chunks, chunk_k + WIN_CHUNKS + 1)
        for c_q in range(chunk_k, q_end):
            q_start = c_q * CHUNK_C
            offs_m = q_start + tl.arange(0, CHUNK_C)
            mask_m = offs_m < T_q
            l_base = b * stride_lb + offs_m * stride_lt + h * stride_lh
            cnt_i = tl.load(Counts_ptr + l_base, mask=mask_m, other=0)
            active_m = mask_m & (cnt_i > 0)

            q = tl.load(
                Q_ptr + b * stride_qb + offs_m[:, None] * stride_qt + h * stride_qh + offs_dk[None, :] * stride_qd,
                mask=active_m[:, None],
                other=0.0,
            )
            do = tl.load(
                DO_ptr + b * stride_ob + offs_m[:, None] * stride_ot + h * stride_oh + offs_dv[None, :] * stride_od,
                mask=active_m[:, None],
                other=0.0,
            )
            lse = tl.load(LSE_ptr + l_base, mask=active_m, other=0.0)
            delta = tl.load(Delta_ptr + l_base, mask=active_m, other=0.0)
            dlse = tl.load(DLSE_ptr + l_base, mask=active_m, other=0.0)
            dmax = tl.load(DMax_ptr + l_base, mask=active_m, other=0.0)
            argmax_i = tl.load(Argmax_ptr + l_base, mask=active_m, other=-1)

            qk = tl.dot(q, tl.trans(k_blk)).to(tl.float32) * scale
            if HAS_SLOPE:
                dist = tl.maximum(0, offs_m[:, None] - offs_n[None, :]).to(tl.float32) * inv_c
                qk = qk - slope_h * dist

            valid = active_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)
            dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v_blk)).to(tl.float32)
            is_max = valid & (offs_n[None, :] == argmax_i[:, None])
            ds = p * (dp - delta[:, None] + dlse[:, None]) + tl.where(is_max, dmax[:, None], 0.0)
            dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

        dk_acc = dk_acc * scale
        tl.store(
            DK_ptr + b * stride_kb + offs_n[:, None] * stride_kt + h * stride_kh + offs_dk[None, :] * stride_kd,
            dk_acc.to(DK_ptr.dtype.element_ty),
            mask=mask_n_raw[:, None],
        )
        tl.store(
            DV_ptr + b * stride_vb + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :] * stride_vd,
            dv_acc.to(DV_ptr.dtype.element_ty),
            mask=mask_n_raw[:, None],
        )

    class _DisjointFlashSWAFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, q_swa, k_swa, v_swa, valid_swa, slope, chunk_size: int, window_chunks: int, has_valid_mask: bool
        ):
            B, T, H, D_k = q_swa.shape
            D_v = v_swa.shape[-1]
            q_c = q_swa.contiguous()
            k_c = k_swa.contiguous()
            v_c = v_swa.contiguous()
            val_c = valid_swa.contiguous() if has_valid_mask else q_c.new_empty((1, 1), dtype=torch.bool)
            has_slope = slope is not None and slope.numel() > 0
            slp_c = slope.detach().float().contiguous() if has_slope else q_c.new_empty(1, dtype=torch.float32)

            out = torch.empty((B, T, H, D_v), device=q_c.device, dtype=q_c.dtype)
            lse = torch.empty((B, T, H), device=q_c.device, dtype=torch.float32)
            max_sc = torch.empty((B, T, H), device=q_c.device, dtype=torch.float32)
            argmax = torch.empty((B, T, H), device=q_c.device, dtype=torch.int32)
            counts = torch.empty((B, T, H), device=q_c.device, dtype=torch.int32)

            n_chunks = (T + chunk_size - 1) // chunk_size
            scale = D_k**-0.5
            inv_c = 1.0 / float(chunk_size)

            _disjoint_swa_fwd_kernel[(n_chunks, B * H)](
                q_c,
                k_c,
                v_c,
                val_c,
                slp_c,
                out,
                lse,
                max_sc,
                argmax,
                counts,
                *q_c.stride(),
                *k_c.stride(),
                *v_c.stride(),
                *val_c.stride(),
                *out.stride(),
                *lse.stride(),
                T,
                H,
                scale,
                inv_c,
                CHUNK_C=chunk_size,
                WIN_CHUNKS=window_chunks,
                D_K=D_k,
                D_V=D_v,
                HAS_SLOPE=has_slope,
                HAS_VALID=has_valid_mask,
                num_warps=4,
                num_stages=2,
            )
            ctx.set_materialize_grads(False)
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(q_c, k_c, v_c, val_c, slp_c, out, lse, argmax, counts)
                ctx.meta = (chunk_size, window_chunks, has_slope, has_valid_mask, slope.dtype if has_slope else None)
            return out, lse, max_sc, counts

        @staticmethod
        def backward(ctx, do, dlse, dmax, _dcounts):
            q_c, k_c, v_c, val_c, slp_c, out, lse, argmax, counts = ctx.saved_tensors
            chunk_size, window_chunks, has_slope, has_valid_mask, slp_dt = ctx.meta
            B, T, H, D_k = q_c.shape
            D_v = v_c.shape[-1]
            do_c = do.to(v_c.dtype).contiguous() if do is not None else torch.zeros_like(v_c)
            dlse_c = dlse.float().contiguous() if dlse is not None else torch.zeros_like(lse)
            dmax_c = dmax.float().contiguous() if dmax is not None else torch.zeros_like(lse)
            delta = (do.float() * out.float()).sum(dim=-1).contiguous() if do is not None else torch.zeros_like(lse)

            dq = torch.empty_like(q_c)
            dk = torch.empty_like(k_c)
            dv = torch.empty_like(v_c)
            n_chunks = (T + chunk_size - 1) // chunk_size
            dslope_part = (
                torch.empty((n_chunks, B * H), device=q_c.device, dtype=torch.float32)
                if has_slope
                else q_c.new_empty(1, dtype=torch.float32)
            )
            scale = D_k**-0.5
            inv_c = 1.0 / float(chunk_size)

            _disjoint_swa_bwd_dq_kernel[(n_chunks, B * H)](
                q_c,
                k_c,
                v_c,
                val_c,
                slp_c,
                do_c,
                delta,
                dlse_c,
                dmax_c,
                lse,
                argmax,
                counts,
                dq,
                dslope_part,
                *q_c.stride(),
                *k_c.stride(),
                *v_c.stride(),
                *val_c.stride(),
                *do_c.stride(),
                *lse.stride(),
                T,
                H,
                scale,
                inv_c,
                CHUNK_C=chunk_size,
                WIN_CHUNKS=window_chunks,
                D_K=D_k,
                D_V=D_v,
                HAS_SLOPE=has_slope,
                HAS_VALID=has_valid_mask,
                num_warps=4,
                num_stages=1,
            )
            _disjoint_swa_bwd_dkv_kernel[(n_chunks, B * H)](
                q_c,
                k_c,
                v_c,
                val_c,
                slp_c,
                do_c,
                delta,
                dlse_c,
                dmax_c,
                lse,
                argmax,
                counts,
                dk,
                dv,
                *q_c.stride(),
                *k_c.stride(),
                *v_c.stride(),
                *val_c.stride(),
                *do_c.stride(),
                *lse.stride(),
                T,
                n_chunks,
                H,
                scale,
                inv_c,
                CHUNK_C=chunk_size,
                WIN_CHUNKS=window_chunks,
                D_K=D_k,
                D_V=D_v,
                HAS_SLOPE=has_slope,
                HAS_VALID=has_valid_mask,
                num_warps=4,
                num_stages=1,
            )
            dslope = dslope_part.view(n_chunks, B, H).sum(dim=(0, 1)).to(slp_dt) if has_slope else None
            return dq, dk, dv, None, dslope, None, None, None

    @triton.jit
    def _routed_readout_bwd_vec_w_kernel(
        Vec_ptr,
        Bank_ptr,
        TopIdx_ptr,
        TopW_ptr,
        DOut_ptr,
        DVec_ptr,
        DW_part_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_dwp_bh,
        stride_dwp_t,
        stride_dwp_k,
        T_len,
        M,
        H,
        K_SEL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T_len
        offs_dk = tl.arange(0, D_K)
        offs_64 = tl.arange(0, 64)

        vec = tl.load(
            Vec_ptr + b * stride_vb + offs_t[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        )
        dvec_acc = tl.zeros([BLOCK_T, D_K], dtype=tl.float32)

        # Zero DW_part for this (bh, offs_t)
        for r in tl.static_range(0, K_SEL):
            tl.store(DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k, 0.0, mask=mask_t)

        for m in range(M):
            any_sel = tl.zeros([BLOCK_T], dtype=tl.int32)
            wm = tl.zeros([BLOCK_T], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(TopIdx_ptr + b * stride_ib + offs_t * stride_it + r * stride_ik, mask=mask_t, other=-1)
                w_r = tl.load(TopW_ptr + b * stride_wb + offs_t * stride_wt + r * stride_wk, mask=mask_t, other=0.0).to(
                    tl.float32
                )
                sel_r = idx_r == m
                any_sel += sel_r.to(tl.int32)
                wm += tl.where(sel_r, w_r, 0.0)

            if tl.max(any_sel) > 0:
                dot_m = tl.zeros([BLOCK_T], dtype=tl.float32)
                for v_off in range(0, D_V, 64):
                    v_cols = v_off + offs_64
                    dout_tile = tl.load(
                        DOut_ptr
                        + b * stride_ob
                        + offs_t[:, None] * stride_ot
                        + h * stride_oh
                        + v_cols[None, :] * stride_od,
                        mask=mask_t[:, None],
                        other=0.0,
                    )
                    s_tile = tl.load(
                        Bank_ptr
                        + b * stride_sb
                        + m * stride_sm
                        + h * stride_sh
                        + offs_dk[:, None] * stride_sk
                        + v_cols[None, :] * stride_sv
                    )
                    if USE_BF16:
                        prod = tl.dot(vec.to(tl.bfloat16), s_tile.to(tl.bfloat16)).to(tl.float32)
                        dvec_acc += wm[:, None] * tl.dot(
                            dout_tile.to(tl.bfloat16), tl.trans(s_tile.to(tl.bfloat16))
                        ).to(tl.float32)
                    else:
                        prod = tl.dot(vec.to(tl.float32), s_tile.to(tl.float32))
                        dvec_acc += wm[:, None] * tl.dot(dout_tile.to(tl.float32), tl.trans(s_tile.to(tl.float32)))
                    dot_m += tl.sum(dout_tile.to(tl.float32) * prod, axis=1)

                for r in tl.static_range(0, K_SEL):
                    idx_r = tl.load(
                        TopIdx_ptr + b * stride_ib + offs_t * stride_it + r * stride_ik, mask=mask_t, other=-1
                    )
                    prev_dw = tl.load(
                        DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k,
                        mask=mask_t,
                        other=0.0,
                    )
                    tl.store(
                        DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k,
                        prev_dw + tl.where(idx_r == m, dot_m, 0.0),
                        mask=mask_t,
                    )

        tl.store(
            DVec_ptr + b * stride_vb + offs_t[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            dvec_acc.to(DVec_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )

    @triton.jit
    def _routed_readout_bwd_ds_kernel(
        Vec_ptr,
        TopIdx_ptr,
        TopW_ptr,
        DOut_ptr,
        DBank_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        T_len,
        M,
        H,
        K_SEL: tl.constexpr,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        v_blk = tl.program_id(1)
        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M

        offs_64 = tl.arange(0, 64)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)

        # Fast early-exit for single-chunk T_len <= 64 if page m is untouched!
        if T_len <= 64:
            t_mask0 = offs_64 < T_len
            wm0 = tl.zeros([64], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(
                    TopIdx_ptr + b * stride_ib + offs_64 * stride_it + r * stride_ik, mask=t_mask0, other=-1
                )
                w_r = tl.load(
                    TopW_ptr + b * stride_wb + offs_64 * stride_wt + r * stride_wk, mask=t_mask0, other=0.0
                ).to(tl.float32)
                wm0 += tl.where(idx_r == m, w_r, 0.0)
            if tl.max(tl.abs(wm0)) == 0.0:
                return
            dout_sub = tl.load(
                DOut_ptr + b * stride_ob + offs_64[:, None] * stride_ot + h * stride_oh + offs_bv[None, :] * stride_od,
                mask=t_mask0[:, None],
                other=0.0,
            ).to(tl.float32)
            for k_off in range(0, D_K, 64):
                dk_sub = k_off + offs_64
                vec_sub = tl.load(
                    Vec_ptr
                    + b * stride_vb
                    + offs_64[:, None] * stride_vt
                    + h * stride_vh
                    + dk_sub[None, :] * stride_vd,
                    mask=t_mask0[:, None],
                    other=0.0,
                ).to(tl.float32)
                w_vec = wm0[:, None] * vec_sub
                if USE_BF16:
                    ds_acc = tl.dot(tl.trans(w_vec.to(tl.bfloat16)), dout_sub.to(tl.bfloat16)).to(tl.float32)
                else:
                    ds_acc = tl.dot(tl.trans(w_vec), dout_sub)
                ds_ptrs = (
                    DBank_ptr
                    + b * stride_sb
                    + m * stride_sm
                    + h * stride_sh
                    + dk_sub[:, None] * stride_sk
                    + offs_bv[None, :] * stride_sv
                )
                tl.store(ds_ptrs, tl.load(ds_ptrs) + ds_acc)
            return

        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            ds_acc = tl.zeros([64, B_V], dtype=tl.float32)
            has_any = 0
            for t0 in range(0, T_len, 64):
                t_offs = t0 + offs_64
                t_mask = t_offs < T_len
                wm = tl.zeros([64], dtype=tl.float32)
                for r in tl.static_range(0, K_SEL):
                    idx_r = tl.load(
                        TopIdx_ptr + b * stride_ib + t_offs * stride_it + r * stride_ik, mask=t_mask, other=-1
                    )
                    w_r = tl.load(
                        TopW_ptr + b * stride_wb + t_offs * stride_wt + r * stride_wk, mask=t_mask, other=0.0
                    ).to(tl.float32)
                    wm += tl.where(idx_r == m, w_r, 0.0)
                if tl.max(tl.abs(wm)) > 0.0:
                    has_any = 1
                    vec_sub = tl.load(
                        Vec_ptr
                        + b * stride_vb
                        + t_offs[:, None] * stride_vt
                        + h * stride_vh
                        + dk_sub[None, :] * stride_vd,
                        mask=t_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    dout_sub = tl.load(
                        DOut_ptr
                        + b * stride_ob
                        + t_offs[:, None] * stride_ot
                        + h * stride_oh
                        + offs_bv[None, :] * stride_od,
                        mask=t_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    w_vec = wm[:, None] * vec_sub
                    if USE_BF16:
                        ds_acc += tl.dot(tl.trans(w_vec.to(tl.bfloat16)), dout_sub.to(tl.bfloat16)).to(tl.float32)
                    else:
                        ds_acc += tl.dot(tl.trans(w_vec), dout_sub)
            if has_any == 1:
                ds_ptrs = (
                    DBank_ptr
                    + b * stride_sb
                    + m * stride_sm
                    + h * stride_sh
                    + dk_sub[:, None] * stride_sk
                    + offs_bv[None, :] * stride_sv
                )
                tl.store(ds_ptrs, tl.load(ds_ptrs) + ds_acc)

    @triton.jit
    def _invert_unit_lower_16x16_fast(L_blk):
        offs = tl.arange(0, 16)
        r = offs[:, None]
        c = offs[None, :]
        L_s = tl.where(r > c, L_blk, 0.0)
        inv = (r == c).to(tl.float32) - tl.where((r == c + 1) & ((c & 1) == 0), L_s, 0.0)
        L_s1 = tl.where(((r >> 2) == (c >> 2)) & ((r >> 1) > (c >> 1)), L_s, 0.0)
        inv = inv - tl.dot(tl.dot(inv, L_s1, input_precision="ieee"), inv, input_precision="ieee")
        L_s2 = tl.where(((r >> 3) == (c >> 3)) & ((r >> 2) > (c >> 2)), L_s, 0.0)
        inv = inv - tl.dot(tl.dot(inv, L_s2, input_precision="ieee"), inv, input_precision="ieee")
        L_s3 = tl.where((r >> 3) > (c >> 3), L_s, 0.0)
        inv = inv - tl.dot(tl.dot(inv, L_s3, input_precision="ieee"), inv, input_precision="ieee")
        return inv

    @triton.jit
    def _wy_merge_inv_fwd_kernel(
        Act_ptr,
        L_work_ptr,
        A_inv_ptr,
        stride_l_bmh,
        stride_l_r,
        stride_l_c,
        stride_a_bmh,
        stride_a_r,
        stride_a_c,
    ):
        bmh = tl.program_id(0)
        if tl.load(Act_ptr + bmh) == 0:
            return
        offs_16 = tl.arange(0, 16)
        l_base = L_work_ptr + bmh * stride_l_bmh
        a_base = A_inv_ptr + bmh * stride_a_bmh

        inv00 = tl.load(l_base + (0 + offs_16[:, None]) * stride_l_r + (0 + offs_16[None, :]) * stride_l_c)
        inv11 = tl.load(l_base + (16 + offs_16[:, None]) * stride_l_r + (16 + offs_16[None, :]) * stride_l_c)
        inv22 = tl.load(l_base + (32 + offs_16[:, None]) * stride_l_r + (32 + offs_16[None, :]) * stride_l_c)
        inv33 = tl.load(l_base + (48 + offs_16[:, None]) * stride_l_r + (48 + offs_16[None, :]) * stride_l_c)
        L10 = tl.load(l_base + (16 + offs_16[:, None]) * stride_l_r + (0 + offs_16[None, :]) * stride_l_c)
        L20 = tl.load(l_base + (32 + offs_16[:, None]) * stride_l_r + (0 + offs_16[None, :]) * stride_l_c)
        L21 = tl.load(l_base + (32 + offs_16[:, None]) * stride_l_r + (16 + offs_16[None, :]) * stride_l_c)
        L30 = tl.load(l_base + (48 + offs_16[:, None]) * stride_l_r + (0 + offs_16[None, :]) * stride_l_c)
        L31 = tl.load(l_base + (48 + offs_16[:, None]) * stride_l_r + (16 + offs_16[None, :]) * stride_l_c)
        L32 = tl.load(l_base + (48 + offs_16[:, None]) * stride_l_r + (32 + offs_16[None, :]) * stride_l_c)

        inv10 = -tl.dot(tl.dot(inv11, L10, input_precision="ieee"), inv00, input_precision="ieee")
        inv32 = -tl.dot(tl.dot(inv33, L32, input_precision="ieee"), inv22, input_precision="ieee")
        M20 = tl.dot(L20, inv00, input_precision="ieee") + tl.dot(L21, inv10, input_precision="ieee")
        M21 = tl.dot(L21, inv11, input_precision="ieee")
        M30 = tl.dot(L30, inv00, input_precision="ieee") + tl.dot(L31, inv10, input_precision="ieee")
        M31 = tl.dot(L31, inv11, input_precision="ieee")
        inv20 = -tl.dot(inv22, M20, input_precision="ieee")
        inv21 = -tl.dot(inv22, M21, input_precision="ieee")
        inv30 = -(tl.dot(inv32, M20, input_precision="ieee") + tl.dot(inv33, M30, input_precision="ieee"))
        inv31 = -(tl.dot(inv32, M21, input_precision="ieee") + tl.dot(inv33, M31, input_precision="ieee"))
        z16 = tl.zeros([16, 16], dtype=tl.float32)

        tl.store(a_base + (0 + offs_16[:, None]) * stride_a_r + (0 + offs_16[None, :]) * stride_a_c, inv00)
        tl.store(a_base + (0 + offs_16[:, None]) * stride_a_r + (16 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (0 + offs_16[:, None]) * stride_a_r + (32 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (0 + offs_16[:, None]) * stride_a_r + (48 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (16 + offs_16[:, None]) * stride_a_r + (0 + offs_16[None, :]) * stride_a_c, inv10)
        tl.store(a_base + (16 + offs_16[:, None]) * stride_a_r + (16 + offs_16[None, :]) * stride_a_c, inv11)
        tl.store(a_base + (16 + offs_16[:, None]) * stride_a_r + (32 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (16 + offs_16[:, None]) * stride_a_r + (48 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (32 + offs_16[:, None]) * stride_a_r + (0 + offs_16[None, :]) * stride_a_c, inv20)
        tl.store(a_base + (32 + offs_16[:, None]) * stride_a_r + (16 + offs_16[None, :]) * stride_a_c, inv21)
        tl.store(a_base + (32 + offs_16[:, None]) * stride_a_r + (32 + offs_16[None, :]) * stride_a_c, inv22)
        tl.store(a_base + (32 + offs_16[:, None]) * stride_a_r + (48 + offs_16[None, :]) * stride_a_c, z16)
        tl.store(a_base + (48 + offs_16[:, None]) * stride_a_r + (0 + offs_16[None, :]) * stride_a_c, inv30)
        tl.store(a_base + (48 + offs_16[:, None]) * stride_a_r + (16 + offs_16[None, :]) * stride_a_c, inv31)
        tl.store(a_base + (48 + offs_16[:, None]) * stride_a_r + (32 + offs_16[None, :]) * stride_a_c, inv32)
        tl.store(a_base + (48 + offs_16[:, None]) * stride_a_r + (48 + offs_16[None, :]) * stride_a_c, inv33)

    @triton.jit
    def _wy_L_blocks_batched_kernel(
        K_base_ptr,
        G_page_ptr,
        CumG_ptr,
        Act_ptr,
        L_work_ptr,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_gp_c,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_cg_cbmh,
        stride_cg_t,
        stride_cg_d,
        stride_l_cbmh,
        stride_l_r,
        stride_l_c,
        T_total,
        c_start,
        BMH,
        chunk_size: tl.constexpr,
        M,
        H,
        D_K: tl.constexpr,
        USE_BF16_DOT: tl.constexpr,
    ):
        slot = tl.program_id(0)
        if tl.load(Act_ptr + slot) == 0:
            return
        blk_id = tl.program_id(1)
        loc = slot // BMH
        bmh = slot % BMH
        n = c_start + loc
        t0 = n * chunk_size
        c_len = tl.minimum(chunk_size, T_total - t0)

        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        offs_16 = tl.arange(0, 16)
        offs_dk = tl.arange(0, D_K)

        gp_base = G_page_ptr + loc * stride_gp_c + b * stride_gp_b + m * stride_gp_m + h * stride_gp_h
        kb_base = K_base_ptr + b * stride_kb_b + h * stride_kb_h
        cg_base = CumG_ptr + slot * stride_cg_cbmh

        if blk_id < 4:
            br = blk_id
            r_offs = br * 16 + offs_16
            r_mask = r_offs < c_len
            strict_lo = offs_16[:, None] > offs_16[None, :]
            g_r = tl.load(gp_base + r_offs * stride_gp_t, mask=r_mask, other=0.0).to(tl.float32)
            cg_0 = tl.load(cg_base + (br * 16) * stride_cg_t + offs_dk * stride_cg_d)
            cg_15 = tl.load(cg_base + (br * 16 + 15) * stride_cg_t + offs_dk * stride_cg_d)
            if tl.max(cg_0 - cg_15) < 12.0:
                kb_r = tl.load(
                    kb_base + (t0 + r_offs[:, None]) * stride_kb_t + offs_dk[None, :] * stride_kb_d,
                    mask=r_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cg_r = tl.load(cg_base + r_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d)
                ke_r = g_r[:, None] * kb_r
                lhs_d = ke_r * tl.exp(cg_r - cg_0[None, :])
                rhs_d = ke_r * tl.exp(cg_0[None, :] - cg_r)
                L_diag = tl.where(strict_lo, tl.dot(lhs_d, tl.trans(rhs_d), input_precision="ieee"), 0.0)
            else:
                L_diag = tl.zeros([16, 16], dtype=tl.float32)
                for d0 in range(0, D_K, 8):
                    d_offs = d0 + tl.arange(0, 8)
                    kb_s = tl.load(
                        kb_base + (t0 + r_offs[:, None]) * stride_kb_t + d_offs[None, :] * stride_kb_d,
                        mask=r_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    ke_s = g_r[:, None] * kb_s
                    cg_s = tl.load(cg_base + r_offs[:, None] * stride_cg_t + d_offs[None, :] * stride_cg_d)
                    diff = tl.where(strict_lo[:, :, None], cg_s[:, None, :] - cg_s[None, :, :], 0.0)
                    L_diag += tl.sum(ke_s[:, None, :] * ke_s[None, :, :] * tl.exp(diff), axis=2)
                L_diag = tl.where(strict_lo, L_diag, 0.0)
            inv_b = _invert_unit_lower_16x16_fast(L_diag)
            tl.store(
                L_work_ptr + slot * stride_l_cbmh + r_offs[:, None] * stride_l_r + r_offs[None, :] * stride_l_c, inv_b
            )
        else:
            if blk_id == 4:
                br, bc = 1, 0
            elif blk_id == 5:
                br, bc = 2, 0
            elif blk_id == 6:
                br, bc = 2, 1
            elif blk_id == 7:
                br, bc = 3, 0
            elif blk_id == 8:
                br, bc = 3, 1
            else:
                br, bc = 3, 2
            r_offs = br * 16 + offs_16
            c_offs = bc * 16 + offs_16
            r_mask = r_offs < c_len
            c_mask = c_offs < c_len
            piv_row = bc * 16 + 15
            g_r = tl.load(gp_base + r_offs * stride_gp_t, mask=r_mask, other=0.0).to(tl.float32)
            g_c = tl.load(gp_base + c_offs * stride_gp_t, mask=c_mask, other=0.0).to(tl.float32)
            kb_r = tl.load(
                kb_base + (t0 + r_offs[:, None]) * stride_kb_t + offs_dk[None, :] * stride_kb_d,
                mask=r_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            kb_c = tl.load(
                kb_base + (t0 + c_offs[:, None]) * stride_kb_t + offs_dk[None, :] * stride_kb_d,
                mask=c_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_r = tl.load(cg_base + r_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d)
            cg_c = tl.load(cg_base + c_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d)
            cg_piv = tl.load(cg_base + piv_row * stride_cg_t + offs_dk * stride_cg_d)
            lhs = (g_r[:, None] * kb_r) * tl.exp(cg_r - cg_piv[None, :])
            rhs = (g_c[:, None] * kb_c) * tl.exp(cg_piv[None, :] - cg_c)
            blk = tl.dot(lhs, tl.trans(rhs), input_precision="ieee")
            tl.store(
                L_work_ptr + slot * stride_l_cbmh + r_offs[:, None] * stride_l_r + c_offs[None, :] * stride_l_c, blk
            )

    @triton.jit
    def _batched_chunk_readout_fwd_kernel(
        Vec_ptr,
        Banks_ptr,
        TopIdx_ptr,
        TopW_ptr,
        Out_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_sc,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        T_span,
        chunk_size: tl.constexpr,
        M,
        H,
        K_SEL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        bh = tl.program_id(1)
        v_blk = tl.program_id(2)
        b = bh // H
        h = bh % H

        offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T_span
        c_loc = (t_blk * BLOCK_T) // chunk_size
        offs_dk = tl.arange(0, D_K)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)

        vec = tl.load(
            Vec_ptr + b * stride_vb + offs_t[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        )
        acc = tl.zeros([BLOCK_T, B_V], dtype=tl.float32)
        s_base = Banks_ptr + c_loc * stride_sc + b * stride_sb + h * stride_sh

        for m in range(M):
            wm = tl.zeros([BLOCK_T], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(TopIdx_ptr + b * stride_ib + offs_t * stride_it + r * stride_ik, mask=mask_t, other=-1)
                w_r = tl.load(TopW_ptr + b * stride_wb + offs_t * stride_wt + r * stride_wk, mask=mask_t, other=0.0).to(
                    tl.float32
                )
                wm += tl.where(idx_r == m, w_r, 0.0)
            if tl.max(tl.abs(wm)) > 0.0:
                s_tile = tl.load(s_base + m * stride_sm + offs_dk[:, None] * stride_sk + offs_bv[None, :] * stride_sv)
                if USE_BF16:
                    prod = tl.dot(vec.to(tl.bfloat16), s_tile.to(tl.bfloat16)).to(tl.float32)
                else:
                    prod = tl.dot(vec.to(tl.float32), s_tile.to(tl.float32))
                acc += wm[:, None] * prod

        tl.store(
            Out_ptr + b * stride_ob + offs_t[:, None] * stride_ot + h * stride_oh + offs_bv[None, :] * stride_od,
            acc.to(Out_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )

    @triton.jit
    def _batched_chunk_readout_bwd_vec_w_kernel(
        Vec_ptr,
        Banks_ptr,
        TopIdx_ptr,
        TopW_ptr,
        DOut_ptr,
        DVec_ptr,
        DW_part_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_sc,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_dwp_bh,
        stride_dwp_t,
        stride_dwp_k,
        T_span,
        chunk_size: tl.constexpr,
        M,
        H,
        K_SEL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T_span
        c_loc = (t_blk * BLOCK_T) // chunk_size
        offs_dk = tl.arange(0, D_K)
        offs_64 = tl.arange(0, 64)

        vec = tl.load(
            Vec_ptr + b * stride_vb + offs_t[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        )
        dvec_acc = tl.zeros([BLOCK_T, D_K], dtype=tl.float32)
        s_base = Banks_ptr + c_loc * stride_sc + b * stride_sb + h * stride_sh

        for r in tl.static_range(0, K_SEL):
            tl.store(DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k, 0.0, mask=mask_t)

        for m in range(M):
            any_sel = tl.zeros([BLOCK_T], dtype=tl.int32)
            wm = tl.zeros([BLOCK_T], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(TopIdx_ptr + b * stride_ib + offs_t * stride_it + r * stride_ik, mask=mask_t, other=-1)
                w_r = tl.load(TopW_ptr + b * stride_wb + offs_t * stride_wt + r * stride_wk, mask=mask_t, other=0.0).to(
                    tl.float32
                )
                sel_r = idx_r == m
                any_sel += sel_r.to(tl.int32)
                wm += tl.where(sel_r, w_r, 0.0)
            if tl.max(any_sel) > 0:
                dot_m = tl.zeros([BLOCK_T], dtype=tl.float32)
                for v_off in range(0, D_V, 64):
                    v_cols = v_off + offs_64
                    dout_tile = tl.load(
                        DOut_ptr
                        + b * stride_ob
                        + offs_t[:, None] * stride_ot
                        + h * stride_oh
                        + v_cols[None, :] * stride_od,
                        mask=mask_t[:, None],
                        other=0.0,
                    )
                    s_tile = tl.load(
                        s_base + m * stride_sm + offs_dk[:, None] * stride_sk + v_cols[None, :] * stride_sv
                    )
                    if USE_BF16:
                        prod = tl.dot(vec.to(tl.bfloat16), s_tile.to(tl.bfloat16)).to(tl.float32)
                        dvec_acc += wm[:, None] * tl.dot(
                            dout_tile.to(tl.bfloat16), tl.trans(s_tile.to(tl.bfloat16))
                        ).to(tl.float32)
                    else:
                        prod = tl.dot(vec.to(tl.float32), s_tile.to(tl.float32))
                        dvec_acc += wm[:, None] * tl.dot(dout_tile.to(tl.float32), tl.trans(s_tile.to(tl.float32)))
                    dot_m += tl.sum(dout_tile.to(tl.float32) * prod, axis=1)
                for r in tl.static_range(0, K_SEL):
                    idx_r = tl.load(
                        TopIdx_ptr + b * stride_ib + offs_t * stride_it + r * stride_ik, mask=mask_t, other=-1
                    )
                    prev_dw = tl.load(
                        DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k,
                        mask=mask_t,
                        other=0.0,
                    )
                    tl.store(
                        DW_part_ptr + bh * stride_dwp_bh + offs_t * stride_dwp_t + r * stride_dwp_k,
                        prev_dw + tl.where(idx_r == m, dot_m, 0.0),
                        mask=mask_t,
                    )

        tl.store(
            DVec_ptr + b * stride_vb + offs_t[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            dvec_acc.to(DVec_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )

    @triton.jit
    def _routed_readout_off_fwd_kernel(
        Vec_ptr,
        Bank_ptr,
        TopIdx_ptr,
        TopW_ptr,
        Out_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        vec_t0,
        out_t0,
        c_len,
        M,
        H,
        K_SEL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        bh = tl.program_id(1)
        v_blk = tl.program_id(2)
        b = bh // H
        h = bh % H

        offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < c_len
        t_in = vec_t0 + offs_t
        t_out = out_t0 + offs_t
        offs_dk = tl.arange(0, D_K)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)

        vec = tl.load(
            Vec_ptr + b * stride_vb + t_in[:, None] * stride_vt + h * stride_vh + offs_dk[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        )
        acc = tl.zeros([BLOCK_T, B_V], dtype=tl.float32)

        for m in range(M):
            wm = tl.zeros([BLOCK_T], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(TopIdx_ptr + b * stride_ib + t_in * stride_it + r * stride_ik, mask=mask_t, other=-1)
                w_r = tl.load(TopW_ptr + b * stride_wb + t_in * stride_wt + r * stride_wk, mask=mask_t, other=0.0).to(
                    tl.float32
                )
                wm += tl.where(idx_r == m, w_r, 0.0)
            if tl.max(tl.abs(wm)) > 0.0:
                s_tile = tl.load(
                    Bank_ptr
                    + b * stride_sb
                    + m * stride_sm
                    + h * stride_sh
                    + offs_dk[:, None] * stride_sk
                    + offs_bv[None, :] * stride_sv
                )
                if USE_BF16:
                    prod = tl.dot(vec.to(tl.bfloat16), s_tile.to(tl.bfloat16)).to(tl.float32)
                else:
                    prod = tl.dot(vec.to(tl.float32), s_tile.to(tl.float32))
                acc += wm[:, None] * prod

        tl.store(
            Out_ptr + b * stride_ob + t_out[:, None] * stride_ot + h * stride_oh + offs_bv[None, :] * stride_od,
            acc.to(Out_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )

    @triton.jit
    def _surprise_and_cumg_fwd_kernel(
        V_ptr,
        PredV_ptr,
        SBias_ptr,
        Slope_ptr,
        Thresh_ptr,
        WM_ptr,
        TopIdx_ptr,
        SteW_ptr,
        PS_ptr,
        Chan_ptr,
        GRoot_ptr,
        GPage_ptr,
        CumG_ptr,
        Act_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_pvb,
        stride_pvt,
        stride_pvh,
        stride_pvd,
        stride_sbb,
        stride_sbt,
        stride_sbh,
        stride_wmb,
        stride_wmt,
        stride_ib,
        stride_it,
        stride_ik,
        stride_swb,
        stride_swt,
        stride_swk,
        stride_ch_b,
        stride_ch_t,
        stride_ch_h,
        stride_ch_d,
        stride_grb,
        stride_grt,
        stride_grh,
        stride_gpb,
        stride_gpt,
        stride_gpm,
        stride_gph,
        stride_cg_bmh,
        stride_cg_t,
        stride_cg_d,
        t0,
        pv_t0,
        c_len,
        M: tl.constexpr,
        H,
        K_SEL: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        HAS_WM: tl.constexpr,
        USE_SURPRISE: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        offs_64 = tl.arange(0, 64)
        mask_t = offs_64 < c_len
        t_glob = t0 + offs_64
        t_pv = pv_t0 + offs_64

        if USE_SURPRISE:
            offs_dv = tl.arange(0, D_V)
            v_f = tl.load(
                V_ptr + b * stride_vb + t_glob[:, None] * stride_vt + h * stride_vh + offs_dv[None, :] * stride_vd,
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            pv_f = tl.load(
                PredV_ptr
                + b * stride_pvb
                + t_pv[:, None] * stride_pvt
                + h * stride_pvh
                + offs_dv[None, :] * stride_pvd,
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            diff = v_f - pv_f
            err_sq = tl.sum(diff * diff, axis=1)
            v_sq = tl.sum(v_f * v_f, axis=1)
            err_norm = tl.where(err_sq > 0.0, tl.sqrt(err_sq), 0.0)
            v_norm = tl.where(v_sq > 0.0, tl.sqrt(v_sq), 0.0)
            eps_err = err_norm / (v_norm + 1e-6)
            slope_h = tl.load(Slope_ptr + h).to(tl.float32)
            thresh_h = tl.load(Thresh_ptr + h).to(tl.float32)
            sb = tl.load(SBias_ptr + b * stride_sbb + t_glob * stride_sbt + h * stride_sbh, mask=mask_t, other=0.0).to(
                tl.float32
            )
            g_root = tl.where(mask_t, tl.sigmoid(slope_h * (eps_err - thresh_h) + sb), 0.0)
        else:
            g_root = tl.where(mask_t, 1.0, 0.0)

        if HAS_WM:
            wm = tl.load(WM_ptr + b * stride_wmb + t_glob * stride_wmt, mask=mask_t, other=0.0).to(tl.float32)
            g_root = g_root * wm

        tl.store(GRoot_ptr + b * stride_grb + t_glob * stride_grt + h * stride_grh, g_root, mask=mask_t)

        offs_dk = tl.arange(0, D_K)
        offs_half = offs_dk // 2
        ch_val = tl.load(
            Chan_ptr
            + b * stride_ch_b
            + t_glob[:, None] * stride_ch_t
            + h * stride_ch_h
            + offs_half[None, :] * stride_ch_d,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)

        for m in range(M):
            wm_m = tl.zeros([64], dtype=tl.float32)
            for r in tl.static_range(0, K_SEL):
                idx_r = tl.load(TopIdx_ptr + b * stride_ib + t_glob * stride_it + r * stride_ik, mask=mask_t, other=-1)
                sw_r = tl.load(
                    SteW_ptr + b * stride_swb + t_glob * stride_swt + r * stride_swk, mask=mask_t, other=0.0
                ).to(tl.float32)
                wm_m += tl.where(idx_r == m, sw_r, 0.0)
            g_vec = wm_m * g_root
            tl.store(GPage_ptr + b * stride_gpb + offs_64 * stride_gpt + m * stride_gpm + h * stride_gph, g_vec)
            bmh = (b * M + m) * H + h
            if tl.max(tl.abs(g_vec)) == 0.0:
                tl.store(Act_ptr + bmh, 0)
            else:
                tl.store(Act_ptr + bmh, 1)
                ps_m = tl.load(PS_ptr + m).to(tl.float32)
                log_a = tl.minimum(-(g_vec[:, None] * g_vec[:, None]) * (ps_m * ch_val), 0.0)
                cum_g = tl.cumsum(log_a, axis=0)
                tl.store(
                    CumG_ptr + bmh * stride_cg_bmh + offs_64[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d,
                    cum_g,
                )

    @triton.jit
    def _wy_L_blocks_off_fwd_kernel(
        K_base_ptr,
        G_page_ptr,
        CumG_ptr,
        Act_ptr,
        L_work_ptr,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_cg_bmh,
        stride_cg_t,
        stride_cg_d,
        stride_l_bmh,
        stride_l_r,
        stride_l_c,
        t0,
        c_len,
        M,
        H,
        D_K: tl.constexpr,
        USE_BF16_DOT: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        if tl.load(Act_ptr + bmh) == 0:
            return
        blk_id = tl.program_id(1)
        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        offs_16 = tl.arange(0, 16)
        offs_dk = tl.arange(0, D_K)

        if blk_id < 4:
            br = blk_id
            r_offs = br * 16 + offs_16
            r_mask = r_offs < c_len
            strict_lo = offs_16[:, None] > offs_16[None, :]
            g_r = tl.load(
                G_page_ptr + b * stride_gp_b + r_offs * stride_gp_t + m * stride_gp_m + h * stride_gp_h,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
            cg_0 = tl.load(CumG_ptr + bmh * stride_cg_bmh + (br * 16) * stride_cg_t + offs_dk * stride_cg_d)
            cg_15 = tl.load(CumG_ptr + bmh * stride_cg_bmh + (br * 16 + 15) * stride_cg_t + offs_dk * stride_cg_d)
            if tl.max(cg_0 - cg_15) < 12.0:
                kb_r = tl.load(
                    K_base_ptr
                    + b * stride_kb_b
                    + (t0 + r_offs[:, None]) * stride_kb_t
                    + h * stride_kb_h
                    + offs_dk[None, :] * stride_kb_d,
                    mask=r_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cg_r = tl.load(
                    CumG_ptr + bmh * stride_cg_bmh + r_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d
                )
                ke_r = g_r[:, None] * kb_r
                L_diag = tl.where(
                    strict_lo,
                    tl.dot(
                        ke_r * tl.exp(cg_r - cg_0[None, :]),
                        tl.trans(ke_r * tl.exp(cg_0[None, :] - cg_r)),
                        input_precision="ieee",
                    ),
                    0.0,
                )
            else:
                L_diag = tl.zeros([16, 16], dtype=tl.float32)
                for d0 in range(0, D_K, 8):
                    d_offs = d0 + tl.arange(0, 8)
                    kb_s = tl.load(
                        K_base_ptr
                        + b * stride_kb_b
                        + (t0 + r_offs[:, None]) * stride_kb_t
                        + h * stride_kb_h
                        + d_offs[None, :] * stride_kb_d,
                        mask=r_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    ke_s = g_r[:, None] * kb_s
                    cg_s = tl.load(
                        CumG_ptr + bmh * stride_cg_bmh + r_offs[:, None] * stride_cg_t + d_offs[None, :] * stride_cg_d
                    )
                    diff = tl.where(strict_lo[:, :, None], cg_s[:, None, :] - cg_s[None, :, :], 0.0)
                    L_diag += tl.sum(ke_s[:, None, :] * ke_s[None, :, :] * tl.exp(diff), axis=2)
                L_diag = tl.where(strict_lo, L_diag, 0.0)
            inv_b = _invert_unit_lower_16x16_fast(L_diag)
            tl.store(
                L_work_ptr + bmh * stride_l_bmh + r_offs[:, None] * stride_l_r + r_offs[None, :] * stride_l_c, inv_b
            )
        else:
            if blk_id == 4:
                br, bc = 1, 0
            elif blk_id == 5:
                br, bc = 2, 0
            elif blk_id == 6:
                br, bc = 2, 1
            elif blk_id == 7:
                br, bc = 3, 0
            elif blk_id == 8:
                br, bc = 3, 1
            else:
                br, bc = 3, 2
            r_offs = br * 16 + offs_16
            c_offs = bc * 16 + offs_16
            r_mask = r_offs < c_len
            c_mask = c_offs < c_len
            piv_row = bc * 16 + 15
            g_r = tl.load(
                G_page_ptr + b * stride_gp_b + r_offs * stride_gp_t + m * stride_gp_m + h * stride_gp_h,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
            g_c = tl.load(
                G_page_ptr + b * stride_gp_b + c_offs * stride_gp_t + m * stride_gp_m + h * stride_gp_h,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
            kb_r = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + (t0 + r_offs[:, None]) * stride_kb_t
                + h * stride_kb_h
                + offs_dk[None, :] * stride_kb_d,
                mask=r_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            kb_c = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + (t0 + c_offs[:, None]) * stride_kb_t
                + h * stride_kb_h
                + offs_dk[None, :] * stride_kb_d,
                mask=c_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_r = tl.load(
                CumG_ptr + bmh * stride_cg_bmh + r_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d
            )
            cg_c = tl.load(
                CumG_ptr + bmh * stride_cg_bmh + c_offs[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d
            )
            cg_piv = tl.load(CumG_ptr + bmh * stride_cg_bmh + piv_row * stride_cg_t + offs_dk * stride_cg_d)
            lhs = (g_r[:, None] * kb_r) * tl.exp(cg_r - cg_piv[None, :])
            rhs = (g_c[:, None] * kb_c) * tl.exp(cg_piv[None, :] - cg_c)
            blk = tl.dot(lhs, tl.trans(rhs), input_precision="ieee")
            tl.store(L_work_ptr + bmh * stride_l_bmh + r_offs[:, None] * stride_l_r + c_offs[None, :] * stride_l_c, blk)

    @triton.jit
    def _wy_w2_off_fwd_kernel(
        K_base_ptr,
        V_base_ptr,
        G_page_ptr,
        A_inv_ptr,
        CumG_ptr,
        Act_ptr,
        S_in_ptr,
        S_out_ptr,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_vb_b,
        stride_vb_t,
        stride_vb_h,
        stride_vb_d,
        stride_gp_c,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_a_bmh,
        stride_a_r,
        stride_a_c,
        stride_cg_bmh,
        stride_cg_t,
        stride_cg_d,
        stride_s_c,
        stride_s_bmh,
        stride_s_k,
        stride_s_v,
        t0,
        loc,
        slot0,
        s_in_idx,
        s_out_idx,
        c_len,
        M,
        H,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        INPLACE_S: tl.constexpr,
        USE_BF16_DOT: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        v_blk = tl.program_id(1)
        slot = slot0 + bmh
        act = tl.load(Act_ptr + slot)
        offs_64 = tl.arange(0, 64)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)
        s_in_base = S_in_ptr + s_in_idx * stride_s_c + bmh * stride_s_bmh
        s_out_base = S_out_ptr + s_out_idx * stride_s_c + bmh * stride_s_bmh
        if act == 0:
            if not INPLACE_S:
                for k0_off in range(0, D_K, 64):
                    dk_sub = k0_off + offs_64
                    tl.store(
                        s_out_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v,
                        tl.load(s_in_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v),
                    )
            return
        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        mask_64 = offs_64 < c_len
        t_glob = t0 + offs_64
        g_vec = tl.load(
            G_page_ptr
            + loc * stride_gp_c
            + b * stride_gp_b
            + offs_64 * stride_gp_t
            + m * stride_gp_m
            + h * stride_gp_h,
            mask=mask_64,
            other=0.0,
        ).to(tl.float32)
        v_base = tl.load(
            V_base_ptr
            + b * stride_vb_b
            + t_glob[:, None] * stride_vb_t
            + h * stride_vb_h
            + offs_bv[None, :] * stride_vb_d,
            mask=mask_64[:, None],
            other=0.0,
        ).to(tl.float32)
        rhs = g_vec[:, None] * v_base
        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            kb_sub = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + t_glob[:, None] * stride_kb_t
                + h * stride_kb_h
                + dk_sub[None, :] * stride_kb_d,
                mask=mask_64[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_sub = tl.load(
                CumG_ptr + slot * stride_cg_bmh + offs_64[:, None] * stride_cg_t + dk_sub[None, :] * stride_cg_d
            )
            k0_sub = tl.exp(cg_sub) * (g_vec[:, None] * kb_sub)
            s_sub = tl.load(s_in_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(tl.float32)
            if USE_BF16_DOT:
                rhs -= tl.dot(k0_sub.to(tl.bfloat16), s_sub.to(tl.bfloat16)).to(tl.float32)
            else:
                rhs -= tl.dot(k0_sub, s_sub)
        a_mat = tl.load(A_inv_ptr + slot * stride_a_bmh + offs_64[:, None] * stride_a_r + offs_64[None, :] * stride_a_c)
        u_wy = tl.where(mask_64[:, None], tl.dot(a_mat, rhs, input_precision="ieee"), 0.0)
        last_idx = c_len - 1
        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            kb_sub = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + t_glob[:, None] * stride_kb_t
                + h * stride_kb_h
                + dk_sub[None, :] * stride_kb_d,
                mask=mask_64[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_sub = tl.load(
                CumG_ptr + slot * stride_cg_bmh + offs_64[:, None] * stride_cg_t + dk_sub[None, :] * stride_cg_d
            )
            cg_end = tl.load(CumG_ptr + slot * stride_cg_bmh + last_idx * stride_cg_t + dk_sub * stride_cg_d)
            kend_sub = tl.where(mask_64[:, None], tl.exp(cg_end[None, :] - cg_sub) * (g_vec[:, None] * kb_sub), 0.0)
            s_sub = tl.load(s_in_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(tl.float32)
            if USE_BF16_DOT:
                upd = tl.dot(tl.trans(kend_sub.to(tl.bfloat16)), u_wy.to(tl.bfloat16)).to(tl.float32)
            else:
                upd = tl.dot(tl.trans(kend_sub), u_wy)
            tl.store(
                s_out_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v,
                tl.exp(cg_end)[:, None] * s_sub + upd,
            )

    @triton.jit
    def _wy_cumg_from_routes_batched_kernel(
        TWI_ptr,
        SteW_ptr,
        GRoot_ptr,
        PS_ptr,
        Chan_ptr,
        G_page_ptr,
        CumG_ptr,
        Act_ptr,
        stride_ib,
        stride_it,
        stride_ik,
        stride_swb,
        stride_swt,
        stride_swk,
        stride_grb,
        stride_grt,
        stride_grh,
        stride_ch_b,
        stride_ch_t,
        stride_ch_h,
        stride_ch_d,
        stride_gp_c,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_cg_cbmh,
        stride_cg_t,
        stride_cg_d,
        T_total,
        c_start,
        chunk_size: tl.constexpr,
        M,
        H,
        K_SEL: tl.constexpr,
        D_K: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        loc = tl.program_id(1)
        n = c_start + loc
        t0 = n * chunk_size
        c_len = tl.minimum(chunk_size, T_total - t0)

        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        slot = loc * tl.num_programs(0) + bmh

        offs_64 = tl.arange(0, 64)
        mask_64 = offs_64 < c_len
        t_glob = t0 + offs_64

        wm = tl.zeros([64], dtype=tl.float32)
        for r in tl.static_range(0, K_SEL):
            idx_r = tl.load(TWI_ptr + b * stride_ib + t_glob * stride_it + r * stride_ik, mask=mask_64, other=-1)
            sw_r = tl.load(
                SteW_ptr + b * stride_swb + t_glob * stride_swt + r * stride_swk, mask=mask_64, other=0.0
            ).to(tl.float32)
            wm += tl.where(idx_r == m, sw_r, 0.0)
        gr = tl.load(GRoot_ptr + b * stride_grb + t_glob * stride_grt + h * stride_grh, mask=mask_64, other=0.0).to(
            tl.float32
        )
        g_vec = wm * gr
        tl.store(
            G_page_ptr
            + loc * stride_gp_c
            + b * stride_gp_b
            + offs_64 * stride_gp_t
            + m * stride_gp_m
            + h * stride_gp_h,
            g_vec,
        )

        if tl.max(tl.abs(g_vec)) == 0.0:
            tl.store(Act_ptr + slot, 0)
            return
        tl.store(Act_ptr + slot, 1)
        ps_m = tl.load(PS_ptr + m).to(tl.float32)
        offs_dk = tl.arange(0, D_K)
        offs_half = offs_dk // 2
        ch_val = tl.load(
            Chan_ptr
            + b * stride_ch_b
            + t_glob[:, None] * stride_ch_t
            + h * stride_ch_h
            + offs_half[None, :] * stride_ch_d,
            mask=mask_64[:, None],
            other=0.0,
        ).to(tl.float32)
        log_a = tl.minimum(-(g_vec[:, None] * g_vec[:, None]) * (ps_m * ch_val), 0.0)
        cum_g = tl.cumsum(log_a, axis=0)
        tl.store(
            CumG_ptr + slot * stride_cg_cbmh + offs_64[:, None] * stride_cg_t + offs_dk[None, :] * stride_cg_d, cum_g
        )

    @triton.jit
    def _routed_readout_off_bwd_ds_kernel(
        Vec_ptr,
        TopIdx_ptr,
        TopW_ptr,
        DOut_ptr,
        DBank_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_it,
        stride_ik,
        stride_wb,
        stride_wt,
        stride_wk,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_sb,
        stride_sm,
        stride_sh,
        stride_sk,
        stride_sv,
        vec_t0,
        dout_t0,
        c_len,
        M,
        H,
        K_SEL: tl.constexpr,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        v_blk = tl.program_id(1)
        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M

        offs_64 = tl.arange(0, 64)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)
        t_mask = offs_64 < c_len
        t_in = vec_t0 + offs_64
        t_do = dout_t0 + offs_64

        wm0 = tl.zeros([64], dtype=tl.float32)
        for r in tl.static_range(0, K_SEL):
            idx_r = tl.load(TopIdx_ptr + b * stride_ib + t_in * stride_it + r * stride_ik, mask=t_mask, other=-1)
            w_r = tl.load(TopW_ptr + b * stride_wb + t_in * stride_wt + r * stride_wk, mask=t_mask, other=0.0).to(
                tl.float32
            )
            wm0 += tl.where(idx_r == m, w_r, 0.0)
        if tl.max(tl.abs(wm0)) == 0.0:
            return
        dout_sub = tl.load(
            DOut_ptr + b * stride_ob + t_do[:, None] * stride_ot + h * stride_oh + offs_bv[None, :] * stride_od,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            vec_sub = tl.load(
                Vec_ptr + b * stride_vb + t_in[:, None] * stride_vt + h * stride_vh + dk_sub[None, :] * stride_vd,
                mask=t_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            w_vec = wm0[:, None] * vec_sub
            if USE_BF16:
                ds_acc = tl.dot(tl.trans(w_vec.to(tl.bfloat16)), dout_sub.to(tl.bfloat16)).to(tl.float32)
            else:
                ds_acc = tl.dot(tl.trans(w_vec), dout_sub)
            ds_ptrs = (
                DBank_ptr
                + b * stride_sb
                + m * stride_sm
                + h * stride_sh
                + dk_sub[:, None] * stride_sk
                + offs_bv[None, :] * stride_sv
            )
            tl.store(ds_ptrs, tl.load(ds_ptrs) + ds_acc)

    @triton.jit
    def _wy_bwd_b1_off_kernel(
        K_base_ptr,
        V_base_ptr,
        G_page_ptr,
        A_inv_ptr,
        CumG_ptr,
        Act_ptr,
        S_seg_ptr,
        DS_cur_ptr,
        DL_part_ptr,
        DK0_part_ptr,
        DKEnd_part_ptr,
        DG63_part_ptr,
        DGV_part_ptr,
        DV_m_ptr,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_vb_b,
        stride_vb_t,
        stride_vb_h,
        stride_vb_d,
        stride_gp_c,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_a_bmh,
        stride_a_r,
        stride_a_c,
        stride_cg_bmh,
        stride_cg_t,
        stride_cg_d,
        stride_s_c,
        stride_s_bmh,
        stride_s_k,
        stride_s_v,
        stride_dl_bmh,
        stride_dl_vb,
        stride_dl_r,
        stride_dl_c,
        stride_dkp_bmh,
        stride_dkp_vb,
        stride_dkp_t,
        stride_dkp_d,
        stride_dg63_bmh,
        stride_dg63_vb,
        stride_dg63_d,
        stride_dgv_bmh,
        stride_dgv_vb,
        stride_dgv_t,
        stride_dvm_bmh,
        stride_dvm_t,
        stride_dvm_v,
        t0,
        loc,
        slot0,
        c_len,
        M,
        H,
        D_K: tl.constexpr,
        B_V: tl.constexpr,
        USE_BF16_DOT: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        v_blk = tl.program_id(1)
        slot = slot0 + bmh
        act = tl.load(Act_ptr + slot)
        offs_64 = tl.arange(0, 64)
        offs_bv = v_blk * B_V + tl.arange(0, B_V)

        if act == 0:
            tl.store(
                DV_m_ptr + slot * stride_dvm_bmh + offs_64[:, None] * stride_dvm_t + offs_bv[None, :] * stride_dvm_v,
                0.0,
            )
            tl.store(DGV_part_ptr + slot * stride_dgv_bmh + v_blk * stride_dgv_vb + offs_64 * stride_dgv_t, 0.0)
            return

        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        mask_64 = offs_64 < c_len
        last_idx = c_len - 1
        t_glob = t0 + offs_64

        g_vec = tl.load(
            G_page_ptr
            + loc * stride_gp_c
            + b * stride_gp_b
            + offs_64 * stride_gp_t
            + m * stride_gp_m
            + h * stride_gp_h,
            mask=mask_64,
            other=0.0,
        ).to(tl.float32)
        v_base = tl.load(
            V_base_ptr
            + b * stride_vb_b
            + t_glob[:, None] * stride_vb_t
            + h * stride_vb_h
            + offs_bv[None, :] * stride_vb_d,
            mask=mask_64[:, None],
            other=0.0,
        ).to(tl.float32)

        s_prev_base = S_seg_ptr + loc * stride_s_c + bmh * stride_s_bmh
        ds_base = DS_cur_ptr + bmh * stride_s_bmh

        rhs = g_vec[:, None] * v_base
        du_wy = tl.zeros([64, B_V], dtype=tl.float32)
        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            kb_sub = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + t_glob[:, None] * stride_kb_t
                + h * stride_kb_h
                + dk_sub[None, :] * stride_kb_d,
                mask=mask_64[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_sub = tl.load(
                CumG_ptr + slot * stride_cg_bmh + offs_64[:, None] * stride_cg_t + dk_sub[None, :] * stride_cg_d
            )
            cg_end = tl.load(CumG_ptr + slot * stride_cg_bmh + last_idx * stride_cg_t + dk_sub * stride_cg_d)
            ke_sub = g_vec[:, None] * kb_sub
            k0_sub = tl.exp(cg_sub) * ke_sub
            kend_sub = tl.where(mask_64[:, None], tl.exp(cg_end[None, :] - cg_sub) * ke_sub, 0.0)
            s_prev_sub = tl.load(s_prev_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(
                tl.float32
            )
            ds_in_sub = tl.load(ds_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(tl.float32)
            if USE_BF16_DOT:
                rhs -= tl.dot(k0_sub.to(tl.bfloat16), s_prev_sub.to(tl.bfloat16)).to(tl.float32)
                du_wy += tl.dot(kend_sub.to(tl.bfloat16), ds_in_sub.to(tl.bfloat16)).to(tl.float32)
            else:
                rhs -= tl.dot(k0_sub, s_prev_sub)
                du_wy += tl.dot(kend_sub, ds_in_sub)

        a_mat = tl.load(A_inv_ptr + slot * stride_a_bmh + offs_64[:, None] * stride_a_r + offs_64[None, :] * stride_a_c)
        u_wy = tl.where(mask_64[:, None], tl.dot(a_mat, rhs, input_precision="ieee"), 0.0)
        du_wy = tl.where(mask_64[:, None], du_wy, 0.0)
        dr = tl.where(mask_64[:, None], tl.dot(tl.trans(a_mat), du_wy, input_precision="ieee"), 0.0)

        dl_part = -tl.dot(dr, tl.trans(u_wy), input_precision="ieee")
        tl.store(
            DL_part_ptr
            + slot * stride_dl_bmh
            + v_blk * stride_dl_vb
            + offs_64[:, None] * stride_dl_r
            + offs_64[None, :] * stride_dl_c,
            dl_part,
        )
        tl.store(
            DV_m_ptr + slot * stride_dvm_bmh + offs_64[:, None] * stride_dvm_t + offs_bv[None, :] * stride_dvm_v,
            g_vec[:, None] * dr,
            mask=mask_64[:, None],
        )
        tl.store(
            DGV_part_ptr + slot * stride_dgv_bmh + v_blk * stride_dgv_vb + offs_64 * stride_dgv_t,
            tl.sum(dr * v_base, axis=1),
        )

        for k_off in range(0, D_K, 64):
            dk_sub = k_off + offs_64
            kb_sub = tl.load(
                K_base_ptr
                + b * stride_kb_b
                + t_glob[:, None] * stride_kb_t
                + h * stride_kb_h
                + dk_sub[None, :] * stride_kb_d,
                mask=mask_64[:, None],
                other=0.0,
            ).to(tl.float32)
            cg_sub = tl.load(
                CumG_ptr + slot * stride_cg_bmh + offs_64[:, None] * stride_cg_t + dk_sub[None, :] * stride_cg_d
            )
            cg_end = tl.load(CumG_ptr + slot * stride_cg_bmh + last_idx * stride_cg_t + dk_sub * stride_cg_d)
            k0_sub = tl.exp(cg_sub) * (g_vec[:, None] * kb_sub)

            s_prev_sub = tl.load(s_prev_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(
                tl.float32
            )
            ds_in_sub = tl.load(ds_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v).to(tl.float32)

            if USE_BF16_DOT:
                dk0_sub = -tl.dot(dr.to(tl.bfloat16), tl.trans(s_prev_sub.to(tl.bfloat16))).to(tl.float32)
                dkend_sub = tl.dot(u_wy.to(tl.bfloat16), tl.trans(ds_in_sub.to(tl.bfloat16))).to(tl.float32)
                ds_upd = tl.dot(tl.trans(k0_sub.to(tl.bfloat16)), dr.to(tl.bfloat16)).to(tl.float32)
            else:
                dk0_sub = -tl.dot(dr, tl.trans(s_prev_sub))
                dkend_sub = tl.dot(u_wy, tl.trans(ds_in_sub))
                ds_upd = tl.dot(tl.trans(k0_sub), dr)

            dec_end = tl.exp(cg_end)
            dg63_sub = tl.sum((dec_end[:, None] * s_prev_sub) * ds_in_sub, axis=1)
            ds_prev_sub = dec_end[:, None] * ds_in_sub - ds_upd

            tl.store(ds_base + dk_sub[:, None] * stride_s_k + offs_bv[None, :] * stride_s_v, ds_prev_sub)
            tl.store(
                DK0_part_ptr
                + slot * stride_dkp_bmh
                + v_blk * stride_dkp_vb
                + offs_64[:, None] * stride_dkp_t
                + dk_sub[None, :] * stride_dkp_d,
                dk0_sub,
            )
            tl.store(
                DKEnd_part_ptr
                + slot * stride_dkp_bmh
                + v_blk * stride_dkp_vb
                + offs_64[:, None] * stride_dkp_t
                + dk_sub[None, :] * stride_dkp_d,
                dkend_sub,
            )
            tl.store(DG63_part_ptr + slot * stride_dg63_bmh + v_blk * stride_dg63_vb + dk_sub * stride_dg63_d, dg63_sub)

    @triton.jit
    def _wy_bwd_b2_off_kernel(
        K_base_ptr,
        G_page_ptr,
        PS_ptr,
        Chan_ptr,
        CumG_ptr,
        Act_ptr,
        DL_part_ptr,
        DK0_part_ptr,
        DKEnd_part_ptr,
        DG63_part_ptr,
        DK_m_ptr,
        DGam_m_ptr,
        DGK_part_ptr,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_gp_c,
        stride_gp_b,
        stride_gp_t,
        stride_gp_m,
        stride_gp_h,
        stride_ch_b,
        stride_ch_t,
        stride_ch_h,
        stride_ch_d,
        stride_cg_cbmh,
        stride_cg_t,
        stride_cg_d,
        stride_dl_cbmh,
        stride_dl_vb,
        stride_dl_r,
        stride_dl_c,
        stride_dkp_cbmh,
        stride_dkp_vb,
        stride_dkp_t,
        stride_dkp_d,
        stride_dg63_cbmh,
        stride_dg63_vb,
        stride_dg63_d,
        stride_dkm_cbmh,
        stride_dkm_t,
        stride_dkm_d,
        stride_dgk_cbmh,
        stride_dgk_kb,
        stride_dgk_t,
        t0,
        loc,
        slot0,
        c_len,
        M,
        H,
        N_VB: tl.constexpr,
        B_K: tl.constexpr,
    ):
        bmh = tl.program_id(0)
        k_blk = tl.program_id(1)
        slot = slot0 + bmh
        offs_64 = tl.arange(0, 64)
        offs_bk = k_blk * B_K + tl.arange(0, B_K)
        if tl.load(Act_ptr + slot) == 0:
            tl.store(
                DK_m_ptr + slot * stride_dkm_cbmh + offs_64[:, None] * stride_dkm_t + offs_bk[None, :] * stride_dkm_d,
                0.0,
            )
            tl.store(
                DGam_m_ptr + slot * stride_dkm_cbmh + offs_64[:, None] * stride_dkm_t + offs_bk[None, :] * stride_dkm_d,
                0.0,
            )
            tl.store(DGK_part_ptr + slot * stride_dgk_cbmh + k_blk * stride_dgk_kb + offs_64 * stride_dgk_t, 0.0)
            return

        h = bmh % H
        bm = bmh // H
        m = bm % M
        b = bm // M
        offs_16 = tl.arange(0, 16)
        offs_half = offs_bk // 2
        mask_64 = offs_64 < c_len
        last_idx = c_len - 1

        dk0 = tl.zeros([64, B_K], dtype=tl.float32)
        dkend = tl.zeros([64, B_K], dtype=tl.float32)
        dg63_st = tl.zeros([B_K], dtype=tl.float32)
        for vb in tl.static_range(0, N_VB):
            dk0 += tl.load(
                DK0_part_ptr
                + slot * stride_dkp_cbmh
                + vb * stride_dkp_vb
                + offs_64[:, None] * stride_dkp_t
                + offs_bk[None, :] * stride_dkp_d
            )
            dkend += tl.load(
                DKEnd_part_ptr
                + slot * stride_dkp_cbmh
                + vb * stride_dkp_vb
                + offs_64[:, None] * stride_dkp_t
                + offs_bk[None, :] * stride_dkp_d
            )
            dg63_st += tl.load(DG63_part_ptr + slot * stride_dg63_cbmh + vb * stride_dg63_vb + offs_bk * stride_dg63_d)

        strict_16 = offs_16[:, None] > offs_16[None, :]
        gp_base = G_page_ptr + loc * stride_gp_c + b * stride_gp_b + m * stride_gp_m + h * stride_gp_h
        kb_base = K_base_ptr + b * stride_kb_b + h * stride_kb_h
        cg_base = CumG_ptr + slot * stride_cg_cbmh
        dl_base = DL_part_ptr + slot * stride_dl_cbmh

        for b_idx in range(0, 4):
            idx_offs = b_idx * 16 + offs_16
            idx_mask = idx_offs < c_len
            acc_lo = tl.zeros([16, B_K], dtype=tl.float32)
            acc_up = tl.zeros([16, B_K], dtype=tl.float32)

            dl_diag = tl.zeros([16, 16], dtype=tl.float32)
            for vb in tl.static_range(0, N_VB):
                dl_diag += tl.load(
                    dl_base + vb * stride_dl_vb + idx_offs[:, None] * stride_dl_r + idx_offs[None, :] * stride_dl_c
                )
            dl_diag = tl.where(strict_16 & idx_mask[:, None] & idx_mask[None, :], dl_diag, 0.0)

            g_b = tl.load(gp_base + idx_offs * stride_gp_t, mask=idx_mask, other=0.0).to(tl.float32)
            kb_b = tl.load(
                kb_base + (t0 + idx_offs[:, None]) * stride_kb_t + offs_bk[None, :] * stride_kb_d,
                mask=idx_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            ke_b = g_b[:, None] * kb_b
            cg_b = tl.load(cg_base + idx_offs[:, None] * stride_cg_t + offs_bk[None, :] * stride_cg_d)
            cg_0 = tl.load(cg_base + (b_idx * 16) * stride_cg_t + offs_bk * stride_cg_d)
            cg_15 = tl.load(cg_base + (b_idx * 16 + 15) * stride_cg_t + offs_bk * stride_cg_d)

            if tl.max(cg_0 - cg_15) < 12.0:
                acc_lo += tl.exp(cg_b - cg_0[None, :]) * tl.dot(
                    dl_diag, tl.exp(cg_0[None, :] - cg_b) * ke_b, input_precision="ieee"
                )
                acc_up += tl.exp(cg_0[None, :] - cg_b) * tl.dot(
                    tl.trans(dl_diag), tl.exp(cg_b - cg_0[None, :]) * ke_b, input_precision="ieee"
                )
            else:
                for j_loc in range(0, 16):
                    j_glob = b_idx * 16 + j_loc
                    cg_j = tl.load(cg_base + j_glob * stride_cg_t + offs_bk * stride_cg_d)
                    kb_j = tl.load(
                        kb_base + (t0 + j_glob) * stride_kb_t + offs_bk * stride_kb_d, mask=(j_glob < c_len), other=0.0
                    ).to(tl.float32)
                    g_j = tl.load(gp_base + j_glob * stride_gp_t, mask=(j_glob < c_len), other=0.0).to(tl.float32)
                    ke_j = g_j * kb_j
                    dl_col = tl.sum(tl.where((offs_16 == j_loc)[None, :], dl_diag, 0.0), axis=1)
                    m_gt = offs_16 > j_loc
                    ed = tl.where(m_gt[:, None], tl.exp(tl.where(m_gt[:, None], cg_b - cg_j[None, :], 0.0)), 0.0)
                    acc_lo += dl_col[:, None] * ed * ke_j[None, :]
                    dl_row = tl.sum(tl.where((offs_16 == j_loc)[:, None], dl_diag, 0.0), axis=0)
                    m_lt = offs_16 < j_loc
                    ed_u = tl.where(m_lt[:, None], tl.exp(tl.where(m_lt[:, None], cg_j[None, :] - cg_b, 0.0)), 0.0)
                    acc_up += dl_row[:, None] * ed_u * ke_j[None, :]

            for bc in range(0, b_idx):
                c_offs = bc * 16 + offs_16
                c_mask = c_offs < c_len
                piv = bc * 16 + 15
                dl_blk = tl.zeros([16, 16], dtype=tl.float32)
                for vb in tl.static_range(0, N_VB):
                    dl_blk += tl.load(
                        dl_base + vb * stride_dl_vb + idx_offs[:, None] * stride_dl_r + c_offs[None, :] * stride_dl_c
                    )
                dl_blk = tl.where(idx_mask[:, None] & c_mask[None, :], dl_blk, 0.0)
                g_c = tl.load(gp_base + c_offs * stride_gp_t, mask=c_mask, other=0.0).to(tl.float32)
                kb_c = tl.load(
                    kb_base + (t0 + c_offs[:, None]) * stride_kb_t + offs_bk[None, :] * stride_kb_d,
                    mask=c_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cg_c = tl.load(cg_base + c_offs[:, None] * stride_cg_t + offs_bk[None, :] * stride_cg_d)
                cg_piv = tl.load(cg_base + piv * stride_cg_t + offs_bk * stride_cg_d)
                acc_lo += tl.exp(cg_b - cg_piv[None, :]) * tl.dot(
                    dl_blk, (g_c[:, None] * kb_c) * tl.exp(cg_piv[None, :] - cg_c), input_precision="ieee"
                )

            piv_self = b_idx * 16 + 15
            cg_piv_self = tl.load(cg_base + piv_self * stride_cg_t + offs_bk * stride_cg_d)
            scale_self = tl.exp(cg_piv_self[None, :] - cg_b)
            for br in range(b_idx + 1, 4):
                r_offs = br * 16 + offs_16
                r_mask = r_offs < c_len
                dl_blk = tl.zeros([16, 16], dtype=tl.float32)
                for vb in tl.static_range(0, N_VB):
                    dl_blk += tl.load(
                        dl_base + vb * stride_dl_vb + r_offs[:, None] * stride_dl_r + idx_offs[None, :] * stride_dl_c
                    )
                dl_blk = tl.where(r_mask[:, None] & idx_mask[None, :], dl_blk, 0.0)
                g_r = tl.load(gp_base + r_offs * stride_gp_t, mask=r_mask, other=0.0).to(tl.float32)
                kb_r = tl.load(
                    kb_base + (t0 + r_offs[:, None]) * stride_kb_t + offs_bk[None, :] * stride_kb_d,
                    mask=r_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                cg_r = tl.load(cg_base + r_offs[:, None] * stride_cg_t + offs_bk[None, :] * stride_cg_d)
                acc_up += scale_self * tl.dot(
                    tl.trans(dl_blk),
                    (g_r[:, None] * kb_r) * tl.exp(cg_r - cg_piv_self[None, :]),
                    input_precision="ieee",
                )

            tl.store(
                DK0_part_ptr
                + slot * stride_dkp_cbmh
                + 0 * stride_dkp_vb
                + idx_offs[:, None] * stride_dkp_t
                + offs_bk[None, :] * stride_dkp_d,
                acc_lo,
            )
            tl.store(
                DKEnd_part_ptr
                + slot * stride_dkp_cbmh
                + 0 * stride_dkp_vb
                + idx_offs[:, None] * stride_dkp_t
                + offs_bk[None, :] * stride_dkp_d,
                acc_up,
            )

        dk_lo = tl.load(
            DK0_part_ptr
            + slot * stride_dkp_cbmh
            + 0 * stride_dkp_vb
            + offs_64[:, None] * stride_dkp_t
            + offs_bk[None, :] * stride_dkp_d
        )
        dk_up = tl.load(
            DKEnd_part_ptr
            + slot * stride_dkp_cbmh
            + 0 * stride_dkp_vb
            + offs_64[:, None] * stride_dkp_t
            + offs_bk[None, :] * stride_dkp_d
        )
        g_vec = tl.load(gp_base + offs_64 * stride_gp_t, mask=mask_64, other=0.0).to(tl.float32)
        kb_full = tl.load(
            kb_base + (t0 + offs_64[:, None]) * stride_kb_t + offs_bk[None, :] * stride_kb_d,
            mask=mask_64[:, None],
            other=0.0,
        ).to(tl.float32)
        ke_full = g_vec[:, None] * kb_full
        cg_full = tl.load(cg_base + offs_64[:, None] * stride_cg_t + offs_bk[None, :] * stride_cg_d)
        cg_end = tl.load(cg_base + last_idx * stride_cg_t + offs_bk * stride_cg_d)

        exp_cg = tl.exp(cg_full)
        exp_to_end = tl.where(mask_64[:, None], tl.exp(cg_end[None, :] - cg_full), 0.0)
        k0_full = exp_cg * ke_full
        kend_full = exp_to_end * ke_full

        dk_eff = tl.where(mask_64[:, None], dk_lo + dk_up + exp_cg * dk0 + exp_to_end * dkend, 0.0)
        dG = ke_full * (dk_lo - dk_up) + k0_full * dk0 - kend_full * dkend
        end_extra = dg63_st + tl.sum(kend_full * dkend, axis=0)
        dG = tl.where(mask_64[:, None], dG + tl.where((offs_64 == last_idx)[:, None], end_extra[None, :], 0.0), 0.0)
        tot_dG = tl.sum(dG, axis=0)
        d_loga = tl.where(mask_64[:, None], tot_dG[None, :] - tl.cumsum(dG, axis=0) + dG, 0.0)

        ps_m = tl.load(PS_ptr + m).to(tl.float32)
        ch_val = tl.load(
            Chan_ptr
            + b * stride_ch_b
            + (t0 + offs_64[:, None]) * stride_ch_t
            + h * stride_ch_h
            + offs_half[None, :] * stride_ch_d,
            mask=mask_64[:, None],
            other=0.0,
        ).to(tl.float32)
        gam_full = ps_m * ch_val

        tl.store(
            DK_m_ptr + slot * stride_dkm_cbmh + offs_64[:, None] * stride_dkm_t + offs_bk[None, :] * stride_dkm_d,
            tl.where(mask_64[:, None], g_vec[:, None] * dk_eff, 0.0),
        )
        tl.store(
            DGam_m_ptr + slot * stride_dkm_cbmh + offs_64[:, None] * stride_dkm_t + offs_bk[None, :] * stride_dkm_d,
            tl.where(mask_64[:, None], -(g_vec[:, None] * g_vec[:, None]) * d_loga, 0.0),
        )
        tl.store(
            DGK_part_ptr + slot * stride_dgk_cbmh + k_blk * stride_dgk_kb + offs_64 * stride_dgk_t,
            tl.where(mask_64, tl.sum(dk_eff * kb_full, axis=1) - 2.0 * g_vec * tl.sum(d_loga * gam_full, axis=1), 0.0),
        )

    @triton.jit
    def _surprise_gate_off_bwd_kernel(
        V_ptr,
        PredV_ptr,
        SBias_ptr,
        Slope_ptr,
        Thresh_ptr,
        WM_ptr,
        TopIdx_ptr,
        SteW_ptr,
        GRoot_ptr,
        DGV_ptr,
        DGK_ptr,
        DGRootAux_ptr,
        DPredV_ptr,
        DVSurp_ptr,
        DSBias_ptr,
        DSlopePart_ptr,
        DThreshPart_ptr,
        DSteWPart_ptr,
        DWM_ptr,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_pvb,
        stride_pvt,
        stride_pvh,
        stride_pvd,
        stride_sbb,
        stride_sbt,
        stride_sbh,
        stride_wmb,
        stride_wmt,
        stride_ib,
        stride_it,
        stride_ik,
        stride_swb,
        stride_swt,
        stride_swk,
        stride_grb,
        stride_grt,
        stride_grh,
        stride_dgv_bmh,
        stride_dgv_vb,
        stride_dgv_t,
        stride_dgk_bmh,
        stride_dgk_kb,
        stride_dgk_t,
        stride_dswp_bh,
        stride_dswp_t,
        stride_dswp_k,
        t0,
        l_off,
        slot0,
        c_len,
        M: tl.constexpr,
        H,
        K_SEL: tl.constexpr,
        D_V: tl.constexpr,
        N_VB: tl.constexpr,
        N_KB: tl.constexpr,
        HAS_WM: tl.constexpr,
        NEED_DWM: tl.constexpr,
        USE_SURPRISE: tl.constexpr,
    ):
        bt = tl.program_id(0)
        h = tl.program_id(1)
        b = bt // c_len
        t = bt % c_len
        bh = b * H + h
        t_glob = t0 + t
        t_loc = l_off + t

        g_root = tl.load(GRoot_ptr + b * stride_grb + t_glob * stride_grt + h * stride_grh).to(tl.float32)
        dg_root = tl.load(DGRootAux_ptr + b * stride_grb + t_glob * stride_grt + h * stride_grh).to(tl.float32)

        for r in tl.static_range(0, K_SEL):
            idx_r = tl.load(TopIdx_ptr + b * stride_ib + t_glob * stride_it + r * stride_ik)
            sw_r = tl.load(SteW_ptr + b * stride_swb + t_glob * stride_swt + r * stride_swk).to(tl.float32)
            slot_r = slot0 + (b * M + idx_r) * H + h
            dgp_r = 0.0
            for vb in tl.static_range(0, N_VB):
                dgp_r += tl.load(DGV_ptr + slot_r * stride_dgv_bmh + vb * stride_dgv_vb + t * stride_dgv_t).to(
                    tl.float32
                )
            for kb in tl.static_range(0, N_KB):
                dgp_r += tl.load(DGK_ptr + slot_r * stride_dgk_bmh + kb * stride_dgk_kb + t * stride_dgk_t).to(
                    tl.float32
                )
            dg_root += sw_r * dgp_r
            tl.store(DSteWPart_ptr + bh * stride_dswp_bh + t_loc * stride_dswp_t + r * stride_dswp_k, g_root * dgp_r)

        if not USE_SURPRISE:
            if HAS_WM and NEED_DWM:
                tl.atomic_add(DWM_ptr + b * stride_wmb + t_glob * stride_wmt, dg_root)
            return

        if HAS_WM:
            wm = tl.load(WM_ptr + b * stride_wmb + t_glob * stride_wmt).to(tl.float32)
            dsig = dg_root * wm
        else:
            dsig = dg_root

        offs_dv = tl.arange(0, D_V)
        v_f = tl.load(V_ptr + b * stride_vb + t_glob * stride_vt + h * stride_vh + offs_dv * stride_vd).to(tl.float32)
        pv_f = tl.load(PredV_ptr + b * stride_pvb + t_loc * stride_pvt + h * stride_pvh + offs_dv * stride_pvd).to(
            tl.float32
        )
        diff = v_f - pv_f
        err_sq = tl.sum(diff * diff, axis=0)
        v_sq = tl.sum(v_f * v_f, axis=0)
        err_norm = tl.where(err_sq > 0.0, tl.sqrt(err_sq), 0.0)
        v_norm = tl.where(v_sq > 0.0, tl.sqrt(v_sq), 0.0)
        denom = v_norm + 1e-6
        eps_err = err_norm / denom

        slope_h = tl.load(Slope_ptr + h).to(tl.float32)
        thresh_h = tl.load(Thresh_ptr + h).to(tl.float32)
        sb = tl.load(SBias_ptr + b * stride_sbb + t_glob * stride_sbt + h * stride_sbh).to(tl.float32)
        sig = tl.sigmoid(slope_h * (eps_err - thresh_h) + sb)
        if HAS_WM and NEED_DWM:
            tl.atomic_add(DWM_ptr + b * stride_wmb + t_glob * stride_wmt, dg_root * sig)

        dz = dsig * sig * (1.0 - sig)
        tl.store(DSBias_ptr + b * stride_sbb + t_glob * stride_sbt + h * stride_sbh, dz)
        out_idx = (b * stride_grb // H + t_glob) * H + h
        tl.store(DSlopePart_ptr + out_idx, dz * (eps_err - thresh_h))
        tl.store(DThreshPart_ptr + out_idx, -dz * slope_h)

        deps = dz * slope_h
        d_err_norm = deps / denom
        d_diff = tl.where(err_sq > 0.0, (d_err_norm / tl.where(err_sq > 0.0, err_norm, 1.0)) * diff, 0.0)
        d_v_norm = -deps * eps_err / denom
        dv_denom = tl.where(v_sq > 0.0, (d_v_norm / tl.where(v_sq > 0.0, v_norm, 1.0)) * v_f, 0.0)

        tl.store(DPredV_ptr + b * stride_pvb + t_loc * stride_pvt + h * stride_pvh + offs_dv * stride_pvd, -d_diff)
        tl.store(
            DVSurp_ptr + b * stride_vb + t_glob * stride_vt + h * stride_vh + offs_dv * stride_vd,
            (d_diff + dv_denom).to(DVSurp_ptr.dtype.element_ty),
        )

    @triton.jit
    def _wy_seg_reduce_noatomic_bwd_kernel(
        DK_m_ptr,
        DV_m_ptr,
        DGam_m_ptr,
        PS_ptr,
        Chan_ptr,
        DK_base_ptr,
        DV_base_ptr,
        DChan_ptr,
        DPS_part_ptr,
        stride_dkm_cbmh,
        stride_dkm_t,
        stride_dkm_d,
        stride_dvm_cbmh,
        stride_dvm_t,
        stride_dvm_v,
        stride_ch_b,
        stride_ch_t,
        stride_ch_h,
        stride_ch_d,
        stride_kb_b,
        stride_kb_t,
        stride_kb_h,
        stride_kb_d,
        stride_vb_b,
        stride_vb_t,
        stride_vb_h,
        stride_vb_d,
        seg_t0,
        seg_len,
        BMH,
        chunk_size: tl.constexpr,
        M: tl.constexpr,
        H,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_16 = tl.arange(0, 16)
        l_offs = t_blk * 16 + offs_16
        mask_t = l_offs < seg_len
        loc = (t_blk * 16) // chunk_size
        c_offs = (t_blk * 16) % chunk_size + offs_16
        t_glob = seg_t0 + l_offs

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_half = tl.arange(0, D_K // 2)

        dk_acc = tl.zeros([16, D_K], dtype=tl.float32)
        dv_acc = tl.zeros([16, D_V], dtype=tl.float32)
        dch_acc = tl.zeros([16, D_K // 2], dtype=tl.float32)

        ch_val = tl.load(
            Chan_ptr
            + b * stride_ch_b
            + t_glob[:, None] * stride_ch_t
            + h * stride_ch_h
            + offs_half[None, :] * stride_ch_d,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)

        part_row = t_blk * (tl.num_programs(1)) + bh
        for m in range(M):
            slot = loc * BMH + (b * M + m) * H + h
            dk_acc += tl.load(
                DK_m_ptr + slot * stride_dkm_cbmh + c_offs[:, None] * stride_dkm_t + offs_dk[None, :] * stride_dkm_d,
                mask=mask_t[:, None],
                other=0.0,
            )
            dv_acc += tl.load(
                DV_m_ptr + slot * stride_dvm_cbmh + c_offs[:, None] * stride_dvm_t + offs_dv[None, :] * stride_dvm_v,
                mask=mask_t[:, None],
                other=0.0,
            )
            dg0 = tl.load(
                DGam_m_ptr
                + slot * stride_dkm_cbmh
                + c_offs[:, None] * stride_dkm_t
                + (2 * offs_half[None, :]) * stride_dkm_d,
                mask=mask_t[:, None],
                other=0.0,
            )
            dg1 = tl.load(
                DGam_m_ptr
                + slot * stride_dkm_cbmh
                + c_offs[:, None] * stride_dkm_t
                + (2 * offs_half[None, :] + 1) * stride_dkm_d,
                mask=mask_t[:, None],
                other=0.0,
            )
            dg_pair = dg0 + dg1
            ps_m = tl.load(PS_ptr + m).to(tl.float32)
            dch_acc += dg_pair * ps_m
            tl.store(DPS_part_ptr + part_row * M + m, tl.sum(dg_pair * ch_val))

        tl.store(
            DK_base_ptr
            + b * stride_kb_b
            + t_glob[:, None] * stride_kb_t
            + h * stride_kb_h
            + offs_dk[None, :] * stride_kb_d,
            dk_acc.to(DK_base_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )
        tl.store(
            DV_base_ptr
            + b * stride_vb_b
            + t_glob[:, None] * stride_vb_t
            + h * stride_vb_h
            + offs_dv[None, :] * stride_vb_d,
            dv_acc.to(DV_base_ptr.dtype.element_ty),
            mask=mask_t[:, None],
        )
        tl.store(
            DChan_ptr
            + b * stride_ch_b
            + t_glob[:, None] * stride_ch_t
            + h * stride_ch_h
            + offs_half[None, :] * stride_ch_d,
            dch_acc,
            mask=mask_t[:, None],
        )

    @triton.jit
    def _reduce_head_weights_kernel(
        Part_ptr,
        Out_ptr,
        stride_p_bh,
        stride_p_t,
        stride_p_k,
        stride_o_b,
        stride_o_t,
        stride_o_k,
        out_t0,
        T_span,
        H: tl.constexpr,
        K_SEL: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        t_blk = tl.program_id(0)
        b = tl.program_id(1)
        offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T_span
        for r in tl.static_range(0, K_SEL):
            acc = tl.zeros([BLOCK_T], dtype=tl.float32)
            for h in range(H):
                acc += tl.load(
                    Part_ptr + (b * H + h) * stride_p_bh + offs_t * stride_p_t + r * stride_p_k, mask=mask_t, other=0.0
                )
            tl.store(Out_ptr + b * stride_o_b + (out_t0 + offs_t) * stride_o_t + r * stride_o_k, acc, mask=mask_t)

    class _TritonHSSDScanFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q_hat,
            k_hat,
            v,
            k_base,
            v_base,
            ps_gamma,
            ch_gamma,
            sbias,
            slope,
            thresh,
            wm,
            top_w_idx,
            top_w_probs,
            ste_w,
            top_r_idx,
            top_r_w,
            s_init,
            chunk_size: int,
            window_chunks: int,
            ckpt_every: int,
            use_surprise: bool,
            surprise_mode: str,
            use_bf16_dot: bool,
            need_boundaries=None,
            boundary_sink=None,
            init_far_ring=None,
        ):
            ctx.set_materialize_grads(False)
            B, T, H, D_k = q_hat.shape
            D_v = v.shape[-1]
            M = int(ps_gamma.shape[0])
            K_sel = int(top_w_idx.shape[-1])
            BMH = B * M * H
            BH = B * H
            n_chunks = (T + chunk_size - 1) // chunk_size
            n_segs = (n_chunks + ckpt_every - 1) // ckpt_every

            qh_c = q_hat.contiguous()
            kh_c = k_hat.contiguous()
            v_c = v.contiguous()
            kb_c = k_base.contiguous()
            vb_c = v_base.contiguous()
            ps_c = ps_gamma.float().contiguous()
            ch_c = ch_gamma.float().contiguous()
            sb_c = sbias.float().contiguous()
            slp_c = slope.float().contiguous()
            thr_c = thresh.float().contiguous()
            has_wm = wm is not None
            wm_c = wm.float().contiguous() if has_wm else sb_c.new_empty((1, 1))

            twi_c = top_w_idx.to(torch.int32).contiguous()
            twp_c = top_w_probs.float().contiguous()
            stew_c = ste_w.float().contiguous()
            tri_c = top_r_idx.to(torch.int32).contiguous()
            trw_c = top_r_w.float().contiguous()

            is_page0 = surprise_mode == "page0"
            if is_page0:
                surp_idx = torch.zeros((B, T, 1), device=q_hat.device, dtype=torch.int32)
                surp_prob = torch.ones((B, T, 1), device=q_hat.device, dtype=torch.float32)
                K_surp = 1
            else:
                surp_idx = twi_c
                surp_prob = twp_c
                K_surp = K_sel

            need_bwd = any(ctx.needs_input_grad)
            s_ckpts = (
                torch.empty((n_segs, B, M, H, D_k, D_v), device=q_hat.device, dtype=torch.float32)
                if need_bwd
                else q_hat.new_empty(0, dtype=torch.float32)
            )
            # Slot 0 is rolling state when no future readout; slots 1..ckpt_every hold s_after_seg!
            s_ring = torch.empty((ckpt_every + 1, B, M, H, D_k, D_v), device=q_hat.device, dtype=torch.float32)
            s_ring[0].copy_(s_init)
            s_ring_views = [s_ring[i] for i in range(ckpt_every + 1)]
            cur_idx = 0

            o_far = torch.empty((B, T, H, D_v), device=q_hat.device, dtype=torch.float32)
            g_root = torch.empty((B, T, H), device=q_hat.device, dtype=torch.float32)

            is_seg_surp = surprise_mode == "segment_start"
            do_surp = use_surprise and surprise_mode != "none"
            pv_len = ckpt_every * chunk_size if is_seg_surp else chunk_size
            pred_v_buf = torch.empty((B, pv_len, H, D_v), device=q_hat.device, dtype=torch.float32)
            g_page_buf = torch.zeros((1, B, 64, M, H), device=q_hat.device, dtype=torch.float32)
            gp0_view = g_page_buf[0]
            cg_buf = torch.empty((BMH, 64, D_k), device=q_hat.device, dtype=torch.float32)
            act_buf = torch.empty((BMH,), device=q_hat.device, dtype=torch.int32)
            l_work = torch.empty((BMH, 64, 64), device=q_hat.device, dtype=torch.float32)
            a_buf = torch.empty((BMH, 64, 64), device=q_hat.device, dtype=torch.float32)

            # Hoist ALL strides ONCE before the loops!
            st_qh = qh_c.stride()
            st_kh = kh_c.stride()
            st_v = v_c.stride()
            st_kb = kb_c.stride()
            st_vb = vb_c.stride()
            st_ch = ch_c.stride()
            st_sb = sb_c.stride()
            st_wm = wm_c.stride()
            st_twi = twi_c.stride()
            st_stew = stew_c.stride()
            st_si = surp_idx.stride()
            st_sp = surp_prob.stride()
            st_tri = tri_c.stride()
            st_trw = trw_c.stride()
            st_of = o_far.stride()
            st_gr = g_root.stride()
            st_pv = pred_v_buf.stride()
            st_gp5 = g_page_buf.stride()
            st_gp4 = g_page_buf[0].stride()
            st_cg = cg_buf.stride()
            st_lw = l_work.stride()
            st_a = a_buf.stride()
            st_sr6 = s_ring.stride()
            st_sr5 = s_ring[0].stride()
            s_c_stride = BMH * D_k * D_v
            s_bmh_stride = D_k * D_v
            nv_32 = D_v // 32
            nv_64 = D_v // 64

            n_init_far = min(n_chunks, window_chunks + 1)
            init_toks = min(T, n_init_far * chunk_size)
            if init_toks > 0:
                if init_far_ring is None:
                    _routed_readout_off_fwd_kernel[((init_toks + 15) // 16, BH, nv_64)](
                        qh_c,
                        s_ring[0],
                        tri_c,
                        trw_c,
                        o_far,
                        *st_qh,
                        *st_sr5,
                        *st_tri,
                        *st_trw,
                        *st_of,
                        0,
                        0,
                        init_toks,
                        M,
                        H,
                        K_SEL=K_sel,
                        BLOCK_T=16,
                        D_K=D_k,
                        B_V=64,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    for m in range(n_init_far):
                        m_t0 = m * chunk_size
                        m_len = min(chunk_size, T - m_t0)
                        if m_len <= 0:
                            break
                        ring_m = init_far_ring[m].to(device=q_hat.device, dtype=torch.float32).contiguous()
                        _routed_readout_off_fwd_kernel[((m_len + 15) // 16, BH, nv_64)](
                            qh_c,
                            ring_m,
                            tri_c,
                            trw_c,
                            o_far,
                            *st_qh,
                            *ring_m.stride(),
                            *st_tri,
                            *st_trw,
                            *st_of,
                            m_t0,
                            m_t0,
                            m_len,
                            M,
                            H,
                            K_SEL=K_sel,
                            BLOCK_T=16,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )

            for seg_i in range(n_segs):
                if need_bwd:
                    s_ckpts[seg_i].copy_(s_ring[cur_idx])
                c_start = seg_i * ckpt_every
                c_end = min(n_chunks, c_start + ckpt_every)
                s_cnt = c_end - c_start
                s_seg_ref = s_ring[cur_idx].clone() if is_seg_surp else None

                if do_surp and is_seg_surp:
                    seg_t0 = c_start * chunk_size
                    seg_len = min(T, c_end * chunk_size) - seg_t0
                    _routed_readout_off_fwd_kernel[((seg_len + 15) // 16, BH, nv_64)](
                        kh_c,
                        s_seg_ref,
                        surp_idx,
                        surp_prob,
                        pred_v_buf,
                        *st_kh,
                        *st_sr5,
                        *st_si,
                        *st_sp,
                        *st_pv,
                        seg_t0,
                        0,
                        seg_len,
                        M,
                        H,
                        K_SEL=K_surp,
                        BLOCK_T=16,
                        D_K=D_k,
                        B_V=64,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )

                n_fut_valid = 0
                for loc in range(s_cnt):
                    n = c_start + loc
                    t0 = n * chunk_size
                    c_len = min(chunk_size, T - t0)

                    if do_surp and not is_seg_surp:
                        _routed_readout_off_fwd_kernel[((c_len + 15) // 16, BH, nv_64)](
                            kh_c,
                            s_ring_views[cur_idx],
                            surp_idx,
                            surp_prob,
                            pred_v_buf,
                            *st_kh,
                            *st_sr5,
                            *st_si,
                            *st_sp,
                            *st_pv,
                            t0,
                            0,
                            c_len,
                            M,
                            H,
                            K_SEL=K_surp,
                            BLOCK_T=16,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )

                    pv_t0 = loc * chunk_size if is_seg_surp else 0
                    _surprise_and_cumg_fwd_kernel[(BH,)](
                        v_c,
                        pred_v_buf,
                        sb_c,
                        slp_c,
                        thr_c,
                        wm_c,
                        twi_c,
                        stew_c,
                        ps_c,
                        ch_c,
                        g_root,
                        gp0_view,
                        cg_buf,
                        act_buf,
                        *st_v,
                        *st_pv,
                        *st_sb,
                        *st_wm,
                        *st_twi,
                        *st_stew,
                        *st_ch,
                        *st_gr,
                        *st_gp4,
                        *st_cg,
                        t0,
                        pv_t0,
                        c_len,
                        M=M,
                        H=H,
                        K_SEL=K_sel,
                        D_K=D_k,
                        D_V=D_v,
                        HAS_WM=has_wm,
                        USE_SURPRISE=do_surp,
                        num_warps=4,
                        num_stages=1,
                    )
                    _wy_L_blocks_off_fwd_kernel[(BMH, 10)](
                        kb_c,
                        gp0_view,
                        cg_buf,
                        act_buf,
                        l_work,
                        *st_kb,
                        *st_gp4,
                        *st_cg,
                        *st_lw,
                        t0,
                        c_len,
                        M,
                        H,
                        D_K=D_k,
                        USE_BF16_DOT=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    _wy_merge_inv_fwd_kernel[(BMH,)](
                        act_buf,
                        l_work,
                        a_buf,
                        *st_lw,
                        *st_a,
                        num_warps=4,
                        num_stages=1,
                    )

                    has_fut = (n + window_chunks + 1 < n_chunks) and (c_len == chunk_size)
                    nxt_idx = (loc + 1) if has_fut else 0
                    _wy_w2_off_fwd_kernel[(BMH, nv_32)](
                        kb_c,
                        vb_c,
                        g_page_buf,
                        a_buf,
                        cg_buf,
                        act_buf,
                        s_ring,
                        s_ring,
                        *st_kb,
                        *st_vb,
                        *st_gp5,
                        *st_a,
                        *st_cg,
                        s_c_stride,
                        s_bmh_stride,
                        D_v,
                        1,
                        t0,
                        0,
                        0,
                        cur_idx,
                        nxt_idx,
                        c_len,
                        M,
                        H,
                        D_K=D_k,
                        B_V=32,
                        INPLACE_S=(cur_idx == nxt_idx),
                        USE_BF16_DOT=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    cur_idx = nxt_idx
                    if has_fut:
                        n_fut_valid = loc + 1
                    if (
                        boundary_sink is not None
                        and need_boundaries
                        and c_len == chunk_size
                        and (n + 1) in need_boundaries
                    ):
                        boundary_sink[n + 1] = s_ring_views[cur_idx].clone()

                if n_fut_valid > 0:
                    f0 = (c_start + window_chunks + 1) * chunk_size
                    f1 = min(T, (c_start + n_fut_valid + window_chunks + 1) * chunk_size)
                    f_span = f1 - f0
                    _batched_chunk_readout_fwd_kernel[((f_span + 15) // 16, BH, nv_64)](
                        qh_c[:, f0:f1],
                        s_ring[1 : 1 + n_fut_valid],
                        tri_c[:, f0:f1],
                        trw_c[:, f0:f1],
                        o_far[:, f0:f1],
                        *st_qh,
                        *st_sr6,
                        *st_tri,
                        *st_trw,
                        *st_of,
                        f_span,
                        chunk_size=chunk_size,
                        M=M,
                        H=H,
                        K_SEL=K_sel,
                        BLOCK_T=16,
                        D_K=D_k,
                        B_V=64,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )

            s_final = s_ring[cur_idx].clone()
            if need_bwd:
                has_init_far = bool(init_far_ring is not None and init_toks > 0)
                init_far_t = (
                    torch.stack(
                        [init_far_ring[m].to(device=q_hat.device) for m in range(n_init_far)], dim=0
                    ).contiguous()
                    if has_init_far and init_far_ring is not None
                    else s_ckpts.new_empty(0)
                )
                ctx.save_for_backward(
                    qh_c,
                    kh_c,
                    v_c,
                    kb_c,
                    vb_c,
                    ps_c,
                    ch_c,
                    sb_c,
                    slp_c,
                    thr_c,
                    wm_c,
                    twi_c,
                    twp_c,
                    stew_c,
                    tri_c,
                    trw_c,
                    g_root,
                    s_ckpts,
                    init_far_t,
                )
                wm_dt = wm.dtype if has_wm else None
                ctx.meta = (
                    chunk_size,
                    window_chunks,
                    ckpt_every,
                    use_surprise,
                    surprise_mode,
                    use_bf16_dot,
                    has_wm,
                    wm_dt,
                    has_init_far,
                )
            return o_far, g_root, s_final

        @staticmethod
        def backward(ctx, do_far, dg_root_aux, ds_final):
            (
                qh_c,
                kh_c,
                v_c,
                kb_c,
                vb_c,
                ps_c,
                ch_c,
                sb_c,
                slp_c,
                thr_c,
                wm_c,
                twi_c,
                twp_c,
                stew_c,
                tri_c,
                trw_c,
                g_root,
                s_ckpts,
                init_far_t,
            ) = ctx.saved_tensors
            (
                chunk_size,
                window_chunks,
                ckpt_every,
                use_surprise,
                surprise_mode,
                use_bf16_dot,
                has_wm,
                wm_dt,
                has_init_far,
            ) = ctx.meta

            B, T, H, D_k = qh_c.shape
            D_v = v_c.shape[-1]
            M = int(ps_c.shape[0])
            K_sel = int(twi_c.shape[-1])
            BMH = B * M * H
            BH = B * H
            n_chunks = (T + chunk_size - 1) // chunk_size
            n_segs = (n_chunks + ckpt_every - 1) // ckpt_every

            is_page0 = surprise_mode == "page0"
            is_seg_surp = surprise_mode == "segment_start"
            do_surp = use_surprise and surprise_mode != "none"
            if is_page0:
                surp_idx = torch.zeros((B, T, 1), device=qh_c.device, dtype=torch.int32)
                surp_prob = torch.ones((B, T, 1), device=qh_c.device, dtype=torch.float32)
                K_surp = 1
            else:
                surp_idx = twi_c
                surp_prob = twp_c
                K_surp = K_sel

            dof_c = (
                do_far.float().contiguous()
                if do_far is not None
                else torch.zeros((B, T, H, D_v), device=qh_c.device, dtype=torch.float32)
            )
            dgr_c = (
                dg_root_aux.float().contiguous()
                if dg_root_aux is not None
                else torch.zeros((B, T, H), device=qh_c.device, dtype=torch.float32)
            )
            ds_cur = (
                ds_final.float().clone().contiguous()
                if ds_final is not None
                else torch.zeros((B, M, H, D_k, D_v), device=qh_c.device, dtype=torch.float32)
            )

            dq_hat = torch.zeros((B, T, H, D_k), device=qh_c.device, dtype=qh_c.dtype)
            dk_hat = torch.zeros((B, T, H, D_k), device=qh_c.device, dtype=kh_c.dtype)
            dv_surp = torch.zeros((B, T, H, D_v), device=qh_c.device, dtype=v_c.dtype)
            dk_base = torch.empty((B, T, H, D_k), device=qh_c.device, dtype=kb_c.dtype)
            dv_base = torch.empty((B, T, H, D_v), device=qh_c.device, dtype=vb_c.dtype)
            dps_acc = torch.zeros((M,), device=qh_c.device, dtype=torch.float32)
            dch_gamma = torch.empty((B, T, H, D_k // 2), device=qh_c.device, dtype=torch.float32)
            dsbias = torch.zeros((B, T, H), device=qh_c.device, dtype=torch.float32)
            dtw_probs = torch.zeros((B, T, K_sel), device=qh_c.device, dtype=torch.float32)
            dste_w = torch.empty((B, T, K_sel), device=qh_c.device, dtype=torch.float32)
            dtr_w = torch.zeros((B, T, K_sel), device=qh_c.device, dtype=torch.float32)

            s_seg_states = torch.empty((ckpt_every + 1, B, M, H, D_k, D_v), device=qh_c.device, dtype=torch.float32)
            gp_seg = torch.empty((ckpt_every, B, 64, M, H), device=qh_c.device, dtype=torch.float32)
            cg_seg = torch.empty((ckpt_every * BMH, 64, D_k), device=qh_c.device, dtype=torch.float32)
            act_seg = torch.empty((ckpt_every * BMH,), device=qh_c.device, dtype=torch.int32)
            l_work_seg = torch.empty((ckpt_every * BMH, 64, 64), device=qh_c.device, dtype=torch.float32)
            a_seg = torch.empty((ckpt_every * BMH, 64, 64), device=qh_c.device, dtype=torch.float32)

            nv_64 = D_v // 64
            nv_32 = D_v // 32
            nk_32 = D_k // 32
            dl_seg = torch.empty((ckpt_every * BMH, nv_64, 64, 64), device=qh_c.device, dtype=torch.float32)
            dk0_seg = torch.empty((ckpt_every * BMH, nv_64, 64, D_k), device=qh_c.device, dtype=torch.float32)
            dkend_seg = torch.empty((ckpt_every * BMH, nv_64, 64, D_k), device=qh_c.device, dtype=torch.float32)
            dg63_seg = torch.empty((ckpt_every * BMH, nv_64, D_k), device=qh_c.device, dtype=torch.float32)
            dgv_seg = torch.empty((ckpt_every * BMH, nv_64, 64), device=qh_c.device, dtype=torch.float32)
            dgk_seg = torch.empty((ckpt_every * BMH, nk_32, 64), device=qh_c.device, dtype=torch.float32)
            dv_m_seg = torch.empty((ckpt_every * BMH, 64, D_v), device=qh_c.device, dtype=torch.float32)
            dk_m_seg = torch.empty((ckpt_every * BMH, 64, D_k), device=qh_c.device, dtype=torch.float32)
            dgam_m_seg = torch.empty((ckpt_every * BMH, 64, D_k), device=qh_c.device, dtype=torch.float32)

            max_seg_toks = max((window_chunks + 1) * chunk_size, ckpt_every * chunk_size)
            pred_v_seg = torch.empty((B, ckpt_every * chunk_size, H, D_v), device=qh_c.device, dtype=torch.float32)
            dpred_v_seg = torch.empty((B, ckpt_every * chunk_size, H, D_v), device=qh_c.device, dtype=torch.float32)
            dslp_part = torch.zeros((B * T * H,), device=qh_c.device, dtype=torch.float32)
            dthr_part = torch.zeros((B * T * H,), device=qh_c.device, dtype=torch.float32)
            need_dwm = bool(has_wm and ctx.needs_input_grad[10])
            dwm_acc = torch.zeros_like(wm_c) if need_dwm else wm_c.new_empty(1)
            dsw_part = torch.empty((BH, ckpt_every * chunk_size, K_sel), device=qh_c.device, dtype=torch.float32)
            dw_read_part = torch.empty((BH, max_seg_toks, max(K_sel, K_surp)), device=qh_c.device, dtype=torch.float32)
            dps_part = torch.empty(
                (((ckpt_every * chunk_size + 15) // 16) * BH, M), device=qh_c.device, dtype=torch.float32
            )

            # Hoist ALL strides ONCE!
            st_qh = qh_c.stride()
            st_kh = kh_c.stride()
            st_v = v_c.stride()
            st_kb = kb_c.stride()
            st_vb = vb_c.stride()
            st_ch = ch_c.stride()
            st_sb = sb_c.stride()
            st_wm = wm_c.stride()
            st_twi = twi_c.stride()
            st_stew = stew_c.stride()
            st_si = surp_idx.stride()
            st_sp = surp_prob.stride()
            st_tri = tri_c.stride()
            st_trw = trw_c.stride()
            st_dof = dof_c.stride()
            st_gr = g_root.stride()
            st_ds = ds_cur.stride()
            st_gp = gp_seg.stride()
            st_cg = cg_seg.stride()
            st_lw = l_work_seg.stride()
            st_a = a_seg.stride()
            st_ss6 = s_seg_states.stride()
            st_ss5 = s_seg_states[0].stride()
            st_dl = dl_seg.stride()
            st_dkp = dk0_seg.stride()
            st_dg63 = dg63_seg.stride()
            st_dgv = dgv_seg.stride()
            st_dgk = dgk_seg.stride()
            st_dvm = dv_m_seg.stride()
            st_dkm = dk_m_seg.stride()
            st_pv = pred_v_seg.stride()
            st_dpv = dpred_v_seg.stride()
            st_dsw = dsw_part.stride()
            st_dwr = dw_read_part.stride()
            st_dtrw = dtr_w.stride()
            st_dtwp = dtw_probs.stride()
            st_dstew = dste_w.stride()
            s_c_stride = BMH * D_k * D_v
            s_bmh_stride = D_k * D_v

            for seg_i in range(n_segs - 1, -1, -1):
                c_start = seg_i * ckpt_every
                c_end = min(n_chunks, c_start + ckpt_every)
                s_cnt = c_end - c_start
                seg_t0 = c_start * chunk_size
                seg_t1 = min(T, c_end * chunk_size)
                seg_len = seg_t1 - seg_t0
                tot_slots = s_cnt * BMH

                _wy_cumg_from_routes_batched_kernel[(BMH, s_cnt)](
                    twi_c,
                    stew_c,
                    g_root,
                    ps_c,
                    ch_c,
                    gp_seg,
                    cg_seg,
                    act_seg,
                    *st_twi,
                    *st_stew,
                    *st_gr,
                    *st_ch,
                    *st_gp,
                    *st_cg,
                    T,
                    c_start,
                    chunk_size=chunk_size,
                    M=M,
                    H=H,
                    K_SEL=K_sel,
                    D_K=D_k,
                    num_warps=4,
                    num_stages=1,
                )
                _wy_L_blocks_batched_kernel[(tot_slots, 10)](
                    kb_c,
                    gp_seg,
                    cg_seg,
                    act_seg,
                    l_work_seg,
                    *st_kb,
                    *st_gp,
                    *st_cg,
                    *st_lw,
                    T,
                    c_start,
                    BMH,
                    chunk_size=chunk_size,
                    M=M,
                    H=H,
                    D_K=D_k,
                    USE_BF16_DOT=use_bf16_dot,
                    num_warps=4,
                    num_stages=1,
                )
                _wy_merge_inv_fwd_kernel[(tot_slots,)](
                    act_seg,
                    l_work_seg,
                    a_seg,
                    *st_lw,
                    *st_a,
                    num_warps=4,
                    num_stages=1,
                )

                s_seg_states[0].copy_(s_ckpts[seg_i])
                n_fut_valid = 0
                for loc in range(s_cnt):
                    n = c_start + loc
                    t0 = n * chunk_size
                    c_len = min(chunk_size, T - t0)
                    _wy_w2_off_fwd_kernel[(BMH, nv_32)](
                        kb_c,
                        vb_c,
                        gp_seg,
                        a_seg,
                        cg_seg,
                        act_seg,
                        s_seg_states,
                        s_seg_states,
                        *st_kb,
                        *st_vb,
                        *st_gp,
                        *st_a,
                        *st_cg,
                        s_c_stride,
                        s_bmh_stride,
                        D_v,
                        1,
                        t0,
                        loc,
                        loc * BMH,
                        loc,
                        loc + 1,
                        c_len,
                        M,
                        H,
                        D_K=D_k,
                        B_V=32,
                        INPLACE_S=False,
                        USE_BF16_DOT=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    if n + window_chunks + 1 < n_chunks and c_len == chunk_size:
                        n_fut_valid = loc + 1

                if n_fut_valid > 0:
                    f0 = (c_start + window_chunks + 1) * chunk_size
                    f1 = min(T, (c_start + n_fut_valid + window_chunks + 1) * chunk_size)
                    f_span = f1 - f0
                    _batched_chunk_readout_bwd_vec_w_kernel[((f_span + 15) // 16, BH)](
                        qh_c[:, f0:f1],
                        s_seg_states[1 : 1 + n_fut_valid],
                        tri_c[:, f0:f1],
                        trw_c[:, f0:f1],
                        dof_c[:, f0:f1],
                        dq_hat[:, f0:f1],
                        dw_read_part,
                        *st_qh,
                        *st_ss6,
                        *st_tri,
                        *st_trw,
                        *st_dof,
                        *st_dwr,
                        f_span,
                        chunk_size=chunk_size,
                        M=M,
                        H=H,
                        K_SEL=K_sel,
                        BLOCK_T=16,
                        D_K=D_k,
                        D_V=D_v,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    _reduce_head_weights_kernel[((f_span + 63) // 64, B)](
                        dw_read_part,
                        dtr_w,
                        *st_dwr,
                        *st_dtrw,
                        f0,
                        f_span,
                        H=H,
                        K_SEL=K_sel,
                        BLOCK_T=64,
                        num_warps=4,
                        num_stages=1,
                    )

                if do_surp:
                    if is_seg_surp:
                        _routed_readout_off_fwd_kernel[((seg_len + 15) // 16, BH, nv_64)](
                            kh_c,
                            s_seg_states[0],
                            surp_idx,
                            surp_prob,
                            pred_v_seg,
                            *st_kh,
                            *st_ss5,
                            *st_si,
                            *st_sp,
                            *st_pv,
                            seg_t0,
                            0,
                            seg_len,
                            M,
                            H,
                            K_SEL=K_surp,
                            BLOCK_T=16,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )
                    else:
                        _batched_chunk_readout_fwd_kernel[((seg_len + 15) // 16, BH, nv_64)](
                            kh_c[:, seg_t0:seg_t1],
                            s_seg_states[:s_cnt],
                            surp_idx[:, seg_t0:seg_t1],
                            surp_prob[:, seg_t0:seg_t1],
                            pred_v_seg,
                            *st_kh,
                            *st_ss6,
                            *st_si,
                            *st_sp,
                            *st_pv,
                            seg_len,
                            chunk_size=chunk_size,
                            M=M,
                            H=H,
                            K_SEL=K_surp,
                            BLOCK_T=16,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )

                # Zero-slice per-chunk state backward!
                ds_seg_start = torch.zeros_like(ds_cur) if is_seg_surp else None
                ds_surp_target = ds_seg_start if is_seg_surp else ds_cur
                for loc in range(s_cnt - 1, -1, -1):
                    n = c_start + loc
                    t0 = n * chunk_size
                    c_len = min(chunk_size, T - t0)
                    l_off = loc * chunk_size
                    slot0 = loc * BMH

                    if loc < n_fut_valid:
                        f0 = (n + window_chunks + 1) * chunk_size
                        f_len = min(chunk_size, T - f0)
                        _routed_readout_off_bwd_ds_kernel[(BMH, nv_64)](
                            qh_c,
                            tri_c,
                            trw_c,
                            dof_c,
                            ds_cur,
                            *st_qh,
                            *st_tri,
                            *st_trw,
                            *st_dof,
                            *st_ds,
                            f0,
                            f0,
                            f_len,
                            M,
                            H,
                            K_SEL=K_sel,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )

                    _wy_bwd_b1_off_kernel[(BMH, nv_64)](
                        kb_c,
                        vb_c,
                        gp_seg,
                        a_seg,
                        cg_seg,
                        act_seg,
                        s_seg_states,
                        ds_cur,
                        dl_seg,
                        dk0_seg,
                        dkend_seg,
                        dg63_seg,
                        dgv_seg,
                        dv_m_seg,
                        *st_kb,
                        *st_vb,
                        *st_gp,
                        *st_a,
                        *st_cg,
                        s_c_stride,
                        s_bmh_stride,
                        D_v,
                        1,
                        *st_dl,
                        *st_dkp,
                        *st_dg63,
                        *st_dgv,
                        *st_dvm,
                        t0,
                        loc,
                        slot0,
                        c_len,
                        M,
                        H,
                        D_K=D_k,
                        B_V=64,
                        USE_BF16_DOT=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    _wy_bwd_b2_off_kernel[(BMH, nk_32)](
                        kb_c,
                        gp_seg,
                        ps_c,
                        ch_c,
                        cg_seg,
                        act_seg,
                        dl_seg,
                        dk0_seg,
                        dkend_seg,
                        dg63_seg,
                        dk_m_seg,
                        dgam_m_seg,
                        dgk_seg,
                        *st_kb,
                        *st_gp,
                        *st_ch,
                        *st_cg,
                        *st_dl,
                        *st_dkp,
                        *st_dg63,
                        *st_dkm,
                        *st_dgk,
                        t0,
                        loc,
                        slot0,
                        c_len,
                        M,
                        H,
                        N_VB=nv_64,
                        B_K=32,
                        num_warps=4,
                        num_stages=1,
                    )
                    _surprise_gate_off_bwd_kernel[(B * c_len, H)](
                        v_c,
                        pred_v_seg,
                        sb_c,
                        slp_c,
                        thr_c,
                        wm_c,
                        twi_c,
                        stew_c,
                        g_root,
                        dgv_seg,
                        dgk_seg,
                        dgr_c,
                        dpred_v_seg,
                        dv_surp,
                        dsbias,
                        dslp_part,
                        dthr_part,
                        dsw_part,
                        dwm_acc,
                        *st_v,
                        *st_pv,
                        *st_sb,
                        *st_wm,
                        *st_twi,
                        *st_stew,
                        *st_gr,
                        *st_dgv,
                        *st_dgk,
                        *st_dsw,
                        t0,
                        l_off,
                        slot0,
                        c_len,
                        M=M,
                        H=H,
                        K_SEL=K_sel,
                        D_V=D_v,
                        N_VB=nv_64,
                        N_KB=nk_32,
                        HAS_WM=has_wm,
                        NEED_DWM=need_dwm,
                        USE_SURPRISE=do_surp,
                        num_warps=4,
                        num_stages=1,
                    )
                    if do_surp:
                        _routed_readout_off_bwd_ds_kernel[(BMH, nv_64)](
                            kh_c,
                            surp_idx,
                            surp_prob,
                            dpred_v_seg,
                            ds_surp_target,
                            *st_kh,
                            *st_si,
                            *st_sp,
                            *st_dpv,
                            *st_ds,
                            t0,
                            l_off,
                            c_len,
                            M,
                            H,
                            K_SEL=K_surp,
                            D_K=D_k,
                            B_V=64,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )

                if is_seg_surp and ds_seg_start is not None:
                    ds_cur.add_(ds_seg_start)

                if do_surp:
                    if is_seg_surp:
                        _routed_readout_bwd_vec_w_kernel[((seg_len + 15) // 16, BH)](
                            kh_c[:, seg_t0:seg_t1],
                            s_seg_states[0],
                            surp_idx[:, seg_t0:seg_t1],
                            surp_prob[:, seg_t0:seg_t1],
                            dpred_v_seg[:, :seg_len],
                            dk_hat[:, seg_t0:seg_t1],
                            dw_read_part,
                            *st_kh,
                            *st_ss5,
                            *st_si,
                            *st_sp,
                            *st_dpv,
                            *st_dwr,
                            seg_len,
                            M,
                            H,
                            K_SEL=K_surp,
                            BLOCK_T=16,
                            D_K=D_k,
                            D_V=D_v,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )
                    else:
                        _batched_chunk_readout_bwd_vec_w_kernel[((seg_len + 15) // 16, BH)](
                            kh_c[:, seg_t0:seg_t1],
                            s_seg_states[:s_cnt],
                            surp_idx[:, seg_t0:seg_t1],
                            surp_prob[:, seg_t0:seg_t1],
                            dpred_v_seg[:, :seg_len],
                            dk_hat[:, seg_t0:seg_t1],
                            dw_read_part,
                            *st_kh,
                            *st_ss6,
                            *st_si,
                            *st_sp,
                            *st_dpv,
                            *st_dwr,
                            seg_len,
                            chunk_size=chunk_size,
                            M=M,
                            H=H,
                            K_SEL=K_surp,
                            BLOCK_T=16,
                            D_K=D_k,
                            D_V=D_v,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )
                    if not is_page0:
                        _reduce_head_weights_kernel[((seg_len + 63) // 64, B)](
                            dw_read_part,
                            dtw_probs,
                            *st_dwr,
                            *st_dtwp,
                            seg_t0,
                            seg_len,
                            H=H,
                            K_SEL=K_sel,
                            BLOCK_T=64,
                            num_warps=4,
                            num_stages=1,
                        )

                _reduce_head_weights_kernel[((seg_len + 63) // 64, B)](
                    dsw_part,
                    dste_w,
                    *st_dsw,
                    *st_dstew,
                    seg_t0,
                    seg_len,
                    H=H,
                    K_SEL=K_sel,
                    BLOCK_T=64,
                    num_warps=4,
                    num_stages=1,
                )
                n_t_blks = (seg_len + 15) // 16
                _wy_seg_reduce_noatomic_bwd_kernel[(n_t_blks, BH)](
                    dk_m_seg,
                    dv_m_seg,
                    dgam_m_seg,
                    ps_c,
                    ch_c,
                    dk_base,
                    dv_base,
                    dch_gamma,
                    dps_part,
                    *st_dkm,
                    *st_dvm,
                    *st_ch,
                    *st_kb,
                    *st_vb,
                    seg_t0,
                    seg_len,
                    BMH,
                    chunk_size=chunk_size,
                    M=M,
                    H=H,
                    D_K=D_k,
                    D_V=D_v,
                    num_warps=4,
                    num_stages=1,
                )
                dps_acc.add_(dps_part[: n_t_blks * BH].sum(dim=0))

            dslope_acc = (
                dslp_part.view(B * T, H).sum(dim=0)
                if do_surp
                else torch.zeros((H,), device=qh_c.device, dtype=torch.float32)
            )
            dthresh_acc = (
                dthr_part.view(B * T, H).sum(dim=0)
                if do_surp
                else torch.zeros((H,), device=qh_c.device, dtype=torch.float32)
            )

            n_init_far = min(n_chunks, window_chunks + 1)
            init_toks = min(T, n_init_far * chunk_size)
            if init_toks > 0:
                if not has_init_far:
                    s0_bank = s_ckpts[0]
                    _routed_readout_bwd_vec_w_kernel[((init_toks + 15) // 16, BH)](
                        qh_c[:, :init_toks],
                        s0_bank,
                        tri_c[:, :init_toks],
                        trw_c[:, :init_toks],
                        dof_c[:, :init_toks],
                        dq_hat[:, :init_toks],
                        dw_read_part,
                        *st_qh,
                        *st_ss5,
                        *st_tri,
                        *st_trw,
                        *st_dof,
                        *st_dwr,
                        init_toks,
                        M,
                        H,
                        K_SEL=K_sel,
                        BLOCK_T=16,
                        D_K=D_k,
                        D_V=D_v,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                    _reduce_head_weights_kernel[((init_toks + 63) // 64, B)](
                        dw_read_part,
                        dtr_w,
                        *st_dwr,
                        *st_dtrw,
                        0,
                        init_toks,
                        H=H,
                        K_SEL=K_sel,
                        BLOCK_T=64,
                        num_warps=4,
                        num_stages=1,
                    )
                    _routed_readout_bwd_ds_kernel[(BMH, nv_64)](
                        qh_c[:, :init_toks],
                        tri_c[:, :init_toks],
                        trw_c[:, :init_toks],
                        dof_c[:, :init_toks],
                        ds_cur,
                        *st_qh,
                        *st_tri,
                        *st_trw,
                        *st_dof,
                        *st_ds,
                        init_toks,
                        M,
                        H,
                        K_SEL=K_sel,
                        D_K=D_k,
                        B_V=64,
                        USE_BF16=use_bf16_dot,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    for m in range(n_init_far):
                        t0_m = m * chunk_size
                        t1_m = min(T, t0_m + chunk_size)
                        len_m = t1_m - t0_m
                        if len_m <= 0:
                            break
                        sm_bank = init_far_t[m].float().contiguous()
                        _routed_readout_bwd_vec_w_kernel[((len_m + 15) // 16, BH)](
                            qh_c[:, t0_m:t1_m],
                            sm_bank,
                            tri_c[:, t0_m:t1_m],
                            trw_c[:, t0_m:t1_m],
                            dof_c[:, t0_m:t1_m],
                            dq_hat[:, t0_m:t1_m],
                            dw_read_part,
                            *st_qh,
                            *sm_bank.stride(),
                            *st_tri,
                            *st_trw,
                            *st_dof,
                            *st_dwr,
                            len_m,
                            M,
                            H,
                            K_SEL=K_sel,
                            BLOCK_T=16,
                            D_K=D_k,
                            D_V=D_v,
                            USE_BF16=use_bf16_dot,
                            num_warps=4,
                            num_stages=1,
                        )
                        _reduce_head_weights_kernel[((len_m + 63) // 64, B)](
                            dw_read_part,
                            dtr_w,
                            *st_dwr,
                            *st_dtrw,
                            t0_m,
                            len_m,
                            H=H,
                            K_SEL=K_sel,
                            BLOCK_T=64,
                            num_warps=4,
                            num_stages=1,
                        )

            return (
                dq_hat,
                dk_hat,
                dv_surp,
                dk_base,
                dv_base,
                dps_acc,
                dch_gamma,
                dsbias,
                dslope_acc,
                dthresh_acc,
                dwm_acc.to(wm_dt) if (need_dwm and wm_dt is not None) else None,
                None,
                dtw_probs,
                dste_w,
                None,
                dtr_w,
                ds_cur,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    @triton.jit
    def _invert_unit_lower_16x16_ieee(L_blk):
        return _invert_unit_lower_16x16_fast(L_blk)

    @triton.jit
    def _invert_unit_lower_64_contig_kernel(L_ptr, AInv_ptr):
        pid = tl.program_id(0)
        offs_16 = tl.arange(0, 16)
        l_base = L_ptr + pid * 4096

        L00 = tl.load(l_base + (0 + offs_16[:, None]) * 64 + (0 + offs_16[None, :])).to(tl.float32)
        L11 = tl.load(l_base + (16 + offs_16[:, None]) * 64 + (16 + offs_16[None, :])).to(tl.float32)
        L22 = tl.load(l_base + (32 + offs_16[:, None]) * 64 + (32 + offs_16[None, :])).to(tl.float32)
        L33 = tl.load(l_base + (48 + offs_16[:, None]) * 64 + (48 + offs_16[None, :])).to(tl.float32)
        inv00 = _invert_unit_lower_16x16_ieee(L00)
        inv11 = _invert_unit_lower_16x16_ieee(L11)
        inv22 = _invert_unit_lower_16x16_ieee(L22)
        inv33 = _invert_unit_lower_16x16_ieee(L33)

        L10 = tl.load(l_base + (16 + offs_16[:, None]) * 64 + (0 + offs_16[None, :])).to(tl.float32)
        L20 = tl.load(l_base + (32 + offs_16[:, None]) * 64 + (0 + offs_16[None, :])).to(tl.float32)
        L21 = tl.load(l_base + (32 + offs_16[:, None]) * 64 + (16 + offs_16[None, :])).to(tl.float32)
        L30 = tl.load(l_base + (48 + offs_16[:, None]) * 64 + (0 + offs_16[None, :])).to(tl.float32)
        L31 = tl.load(l_base + (48 + offs_16[:, None]) * 64 + (16 + offs_16[None, :])).to(tl.float32)
        L32 = tl.load(l_base + (48 + offs_16[:, None]) * 64 + (32 + offs_16[None, :])).to(tl.float32)

        inv10 = -tl.dot(tl.dot(inv11, L10, input_precision="ieee"), inv00, input_precision="ieee")
        inv32 = -tl.dot(tl.dot(inv33, L32, input_precision="ieee"), inv22, input_precision="ieee")
        M20 = tl.dot(L20, inv00, input_precision="ieee") + tl.dot(L21, inv10, input_precision="ieee")
        M21 = tl.dot(L21, inv11, input_precision="ieee")
        M30 = tl.dot(L30, inv00, input_precision="ieee") + tl.dot(L31, inv10, input_precision="ieee")
        M31 = tl.dot(L31, inv11, input_precision="ieee")
        inv20 = -tl.dot(inv22, M20, input_precision="ieee")
        inv21 = -tl.dot(inv22, M21, input_precision="ieee")
        inv30 = -(tl.dot(inv32, M20, input_precision="ieee") + tl.dot(inv33, M30, input_precision="ieee"))
        inv31 = -(tl.dot(inv32, M21, input_precision="ieee") + tl.dot(inv33, M31, input_precision="ieee"))

        a_base = AInv_ptr + pid * 4096
        z16 = tl.zeros([16, 16], dtype=tl.float32)
        tl.store(a_base + (0 + offs_16[:, None]) * 64 + (0 + offs_16[None, :]), inv00)
        tl.store(a_base + (0 + offs_16[:, None]) * 64 + (16 + offs_16[None, :]), z16)
        tl.store(a_base + (0 + offs_16[:, None]) * 64 + (32 + offs_16[None, :]), z16)
        tl.store(a_base + (0 + offs_16[:, None]) * 64 + (48 + offs_16[None, :]), z16)
        tl.store(a_base + (16 + offs_16[:, None]) * 64 + (0 + offs_16[None, :]), inv10)
        tl.store(a_base + (16 + offs_16[:, None]) * 64 + (16 + offs_16[None, :]), inv11)
        tl.store(a_base + (16 + offs_16[:, None]) * 64 + (32 + offs_16[None, :]), z16)
        tl.store(a_base + (16 + offs_16[:, None]) * 64 + (48 + offs_16[None, :]), z16)
        tl.store(a_base + (32 + offs_16[:, None]) * 64 + (0 + offs_16[None, :]), inv20)
        tl.store(a_base + (32 + offs_16[:, None]) * 64 + (16 + offs_16[None, :]), inv21)
        tl.store(a_base + (32 + offs_16[:, None]) * 64 + (32 + offs_16[None, :]), inv22)
        tl.store(a_base + (32 + offs_16[:, None]) * 64 + (48 + offs_16[None, :]), z16)
        tl.store(a_base + (48 + offs_16[:, None]) * 64 + (0 + offs_16[None, :]), inv30)
        tl.store(a_base + (48 + offs_16[:, None]) * 64 + (16 + offs_16[None, :]), inv31)
        tl.store(a_base + (48 + offs_16[:, None]) * 64 + (32 + offs_16[None, :]), inv32)
        tl.store(a_base + (48 + offs_16[:, None]) * 64 + (48 + offs_16[None, :]), inv33)

    class _TritonSolveUnitLower64Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, l_mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            ctx.set_materialize_grads(False)
            lead_shape = l_mat.shape[:-2]
            d_rhs = int(rhs.shape[-1])
            n_batch = int(l_mat.numel() // 4096)
            l_c = l_mat.contiguous().view(n_batch, 64, 64)
            r_3d = rhs.reshape(n_batch, 64, d_rhs)
            a_inv = torch.empty((n_batch, 64, 64), device=l_mat.device, dtype=torch.float32)
            _invert_unit_lower_64_contig_kernel[(n_batch,)](l_c, a_inv, num_warps=4, num_stages=1)
            if rhs.dtype == torch.float32:
                prev_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
                if prev_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    u_3d = torch.bmm(a_inv, r_3d)
                finally:
                    if prev_tf32:
                        torch.backends.cuda.matmul.allow_tf32 = True
            else:
                u_3d = torch.bmm(a_inv.to(rhs.dtype), r_3d)
            if any(ctx.needs_input_grad):
                ctx.save_for_backward(a_inv, u_3d)
                ctx.meta = (lead_shape, d_rhs, l_mat.dtype, rhs.dtype)
            return u_3d.view(*lead_shape, 64, d_rhs)

        @staticmethod
        def backward(ctx, du: torch.Tensor | None):
            a_inv, u_3d = ctx.saved_tensors
            lead_shape, d_rhs, l_dt, r_dt = ctx.meta
            if du is None:
                return None, None
            n_batch = int(a_inv.shape[0])
            need_dl = bool(ctx.needs_input_grad[0])
            need_dr = bool(ctx.needs_input_grad[1])
            if all(s == 0 for s in du.stride()):
                scale_c = du.reshape(-1)[:1].float().view(1, 1, 1)
                dr_col = a_inv.sum(dim=-2, keepdim=True).transpose(-1, -2) * scale_c
                dr_out = (
                    (dr_col.to(r_dt) if r_dt != torch.float32 else dr_col)
                    .view(*lead_shape, 64, 1)
                    .expand(*lead_shape, 64, d_rhs)
                )
                if need_dl:
                    u_row = u_3d.float().sum(dim=-1, keepdim=True).transpose(-1, -2)
                    dl_f = (-dr_col * u_row).tril_(-1)
                    dl_out = (dl_f.to(l_dt) if l_dt != torch.float32 else dl_f).view(*lead_shape, 64, 64)
                else:
                    dl_out = None
                return dl_out, dr_out if need_dr else None

            du_3d = du.reshape(n_batch, 64, d_rhs)
            if r_dt == torch.float32:
                prev_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
                if prev_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    dr_3d = torch.bmm(a_inv.transpose(-1, -2), du_3d)
                    dl_3d = (
                        torch.bmm(dr_3d, u_3d.transpose(-1, -2)).neg_().tril_(-1)
                        if need_dl
                        else None
                    )
                finally:
                    if prev_tf32:
                        torch.backends.cuda.matmul.allow_tf32 = True
            else:
                dr_f = torch.bmm(a_inv.transpose(-1, -2), du_3d.float())
                dr_3d = dr_f.to(r_dt) if need_dr else None
                dl_3d = (
                    torch.bmm(dr_f, u_3d.float().transpose(-1, -2)).neg_().tril_(-1).to(l_dt)
                    if need_dl
                    else None
                )
            return (
                dl_3d.view(*lead_shape, 64, 64) if need_dl and dl_3d is not None else None,
                dr_3d.view(*lead_shape, 64, d_rhs) if need_dr and dr_3d is not None else None,
            )


_ORIG_RUN_CHUNK_GROUP = _run_chunk_group
_ORIG_CHECKPOINT = checkpoint


def _chunk_hssd_triton_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ps_gamma: torch.Tensor,
    ch_gamma: torch.Tensor,
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
    st: HSSDCacheState,
    boltzmann_gate: PeakBoltzmannGate | None,
    swa_recency_slope: torch.Tensor | None,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    segment_chunks: int,
    output_final_state: bool,
    write_mask: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
    state_dtype: str | torch.dtype,
    use_swa: bool,
    use_phase: bool,
    use_surprise: bool,
    surprise_mode: str,
    write_mode: str,
    compact_decode_cache: bool,
    detach_final_state: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor, torch.Tensor]:
    init_dtype = query.dtype
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    acc_dt = torch.float32
    use_bf16_dot = init_dtype in (torch.bfloat16, torch.float16)
    tc_dt = torch.bfloat16 if use_bf16_dot else torch.float32

    q_in = query
    k_in = key
    if q_in.shape[2] != hv:
        require(hv % q_in.shape[2] == 0, "num_v_heads must be divisible by num_q_heads", q=q_in, v=value)
        q_in = q_in.repeat_interleave(hv // q_in.shape[2], dim=2)
    if k_in.shape[2] != hv:
        require(hv % k_in.shape[2] == 0, "num_v_heads must be divisible by num_k_heads", k=k_in, v=value)
        k_in = k_in.repeat_interleave(hv // k_in.shape[2], dim=2)

    if write_mask is not None:
        wm_f: torch.Tensor | None = (
            write_mask.reshape(bsz, seq_len).to(device=query.device, dtype=acc_dt).clamp(0.0, 1.0)
        )
        valid_cur = wm_f > 0.5
    else:
        wm_f = None
        valid_cur = (
            torch.ones(bsz, seq_len, device=query.device, dtype=torch.bool)
            if output_final_state
            else torch.empty(0, device=query.device, dtype=torch.bool)
        )

    if use_phase:
        cos_t, sin_t, term_theta = _TritonPhaseTrigFn.apply(theta, st.theta_cum, wm_f, acc_dt)
    else:
        cos_t = query.new_empty(0, dtype=acc_dt)
        sin_t = query.new_empty(0, dtype=acc_dt)
        term_theta = st.theta_cum.to(dtype=torch.float64).clone().contiguous()

    sc = float(dk ** -0.5) if scale is None else float(scale)
    q_hat, k_hat, v_tc, k_base, v_base = _TritonScanPrepFn.apply(
        q_in, k_in, value, b_gate, w_gate, cos_t, sin_t, sc, use_phase, use_qk_l2norm_in_kernel, tc_dt
    )

    wl_f = write_logits.float()
    rl_f = read_logits.float()
    pi_w = F.softmax(wl_f, dim=-1)
    tw_vals, tw_idx = torch.topk(pi_w, k=top_k_pages, dim=-1)
    tw_probs = tw_vals / tw_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    if write_mode == "full":
        ste_w = torch.ones_like(tw_vals) + (tw_vals - tw_vals.detach())
        hard_sel = torch.zeros_like(pi_w).scatter(-1, tw_idx, 1.0)
        pi_ste = hard_sel + float(top_k_pages) * (pi_w - pi_w.detach())
    else:
        ste_w = torch.sqrt(tw_probs.clamp_min(1e-30))
        pi_act = torch.zeros_like(pi_w).scatter(-1, tw_idx, tw_probs)
        pi_ste = pi_act + (pi_w - pi_w.detach())

    pi_r = F.softmax(rl_f, dim=-1)
    tr_vals, tr_idx = torch.topk(pi_r, k=top_k_pages, dim=-1)
    tr_w = tr_vals / tr_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    seg_step = max(1, int(segment_chunks))
    seg_tokens = seg_step * chunk_size
    ckpt_every = seg_step if surprise_mode == "segment_start" else max(seg_step, 8)

    step_offset = st._uniform_step if st._uniform_step is not None else int(st.step_count[0].item())
    n_0 = step_offset // chunk_size
    end_step = step_offset + seq_len
    n_end = end_step // chunk_size
    k_far_end = max(0, n_end - window_chunks)
    want_final_now = bool(output_final_state and not (detach_final_state and _in_backward()))
    can_build_compact = bool(want_final_now and compact_decode_cache)
    need_local_boundaries: set[int] = set()
    produced_boundaries: dict[int, torch.Tensor] = {}
    if want_final_now:
        if can_build_compact:
            if k_far_end > n_0:
                need_local_boundaries.add(k_far_end - n_0)
            if n_end > n_0 and end_step % chunk_size != 0:
                need_local_boundaries.add(n_end - n_0)
            if surprise_mode == "segment_start":
                seg_end_k = (end_step // seg_tokens) * seg_step
                if seg_end_k > n_0 and end_step % seg_tokens != 0:
                    need_local_boundaries.add(seg_end_k - n_0)
        else:
            for k_b in range(max(n_0 + 1, n_end - window_chunks), n_end + 1):
                need_local_boundaries.add(k_b - n_0)
            if surprise_mode == "segment_start":
                seg_end_k = (end_step // seg_tokens) * seg_step
                if seg_end_k > n_0 and end_step % seg_tokens != 0:
                    need_local_boundaries.add(seg_end_k - n_0)

    init_far_ring = st.boundary_ring if step_offset > 0 else None

    sb_in = (
        surprise_bias.squeeze(-1)
        if (surprise_bias.ndim == 4 and surprise_bias.shape[-1] == 1)
        else surprise_bias
    )
    o_far, g_root, p_cur = _TritonHSSDScanFn.apply(
        q_hat,
        k_hat,
        v_tc,
        k_base,
        v_base,
        ps_gamma,
        ch_gamma,
        sb_in,
        surprise_slope,
        surprise_thresh,
        wm_f,
        tw_idx,
        tw_probs,
        ste_w,
        tr_idx,
        tr_w,
        st.page_states,
        chunk_size,
        window_chunks,
        ckpt_every,
        use_surprise,
        surprise_mode,
        use_bf16_dot,
        need_local_boundaries if want_final_now else None,
        produced_boundaries if want_final_now else None,
        init_far_ring,
    )

    if use_swa:
        if boltzmann_gate is not None:
            gate_log_kappa = upcast(boltzmann_gate.log_kappa).to(acc_dt)
            gate_tau = upcast(boltzmann_gate.tau).to(acc_dt)
        else:
            gate_log_kappa = query.new_full((1,), math.log(2.0), dtype=acc_dt)
            gate_tau = query.new_full((1,), 1.0, dtype=acc_dt)
        qs_tc = q_swa if q_swa.dtype == tc_dt else q_swa.to(dtype=tc_dt)
        ks_tc = k_swa if k_swa.dtype == tc_dt else k_swa.to(dtype=tc_dt)
        vs_tc = v_swa if v_swa.dtype == tc_dt else v_swa.to(dtype=tc_dt)
        if step_offset > 0 and window_chunks > 0:
            max_win_swa = st.max_swa_len
            _, _, swa_base_global = disjoint_horizon_bounds(step_offset + 1, chunk_size, window_chunks)
            prior_len = step_offset - swa_base_global
            if prior_len > 0:
                prior_idx = torch.arange(swa_base_global, step_offset, device=query.device) % max_win_swa
                p_ks = st.swa_k_ring.index_select(1, prior_idx).to(dtype=tc_dt)
                p_vs = st.swa_v_ring.index_select(1, prior_idx).to(dtype=tc_dt)
                p_val = st.swa_valid_ring.index_select(1, prior_idx)
                p_qs = qs_tc.new_zeros((bsz, prior_len, qs_tc.shape[2], qs_tc.shape[3]))
                qs_in = torch.cat([p_qs, qs_tc], dim=1)
                ks_in = torch.cat([p_ks, ks_tc], dim=1)
                vs_in = torch.cat([p_vs, vs_tc], dim=1)
                need_val_mask = (wm_f is not None) or (not bool(p_val.all()))
                if need_val_mask:
                    vc_use = (
                        valid_cur
                        if valid_cur.numel() > 0
                        else torch.ones((bsz, seq_len), device=query.device, dtype=torch.bool)
                    )
                    val_in = torch.cat([p_val, vc_use], dim=1)
                else:
                    val_in = valid_cur
                o_loc_p, lse_p, max_sc_p, counts_p = _DisjointFlashSWAFn.apply(
                    qs_in, ks_in, vs_in, val_in, swa_recency_slope, chunk_size, window_chunks, need_val_mask
                )
                o_loc = o_loc_p[:, prior_len:]
                lse = lse_p[:, prior_len:]
                max_sc = max_sc_p[:, prior_len:]
                counts = counts_p[:, prior_len:]
            else:
                o_loc, lse, max_sc, counts = _DisjointFlashSWAFn.apply(
                    qs_tc, ks_tc, vs_tc, valid_cur, swa_recency_slope, chunk_size, window_chunks, wm_f is not None
                )
        else:
            o_loc, lse, max_sc, counts = _DisjointFlashSWAFn.apply(
                qs_tc, ks_tc, vs_tc, valid_cur, swa_recency_slope, chunk_size, window_chunks, wm_f is not None
            )
        out = _TritonBoltzmannBlendFn.apply(
            o_loc, o_far, lse, max_sc, counts, gate_log_kappa, gate_tau, init_dtype
        )
    else:
        out = o_far.to(dtype=init_dtype)

    g_sq_mean = (g_root * g_root).mean(dim=-1, keepdim=True)
    u_total = (pi_ste * g_sq_mean).sum(dim=(0, 1))
    tok_total = wm_f.sum() if wm_f is not None else g_root.new_full((), float(bsz * seq_len))

    final_state: HSSDCacheState | None = None
    if want_final_now:
        with torch.no_grad() if detach_final_state else contextlib.nullcontext():
            max_win = st.max_swa_len
            write_from_g = max(step_offset, end_step - max_win)
            write_from_l = write_from_g - step_offset
            ring_slots = torch.arange(write_from_g, end_step, device=query.device) % max_win
            new_k_ring = _compact(
                st.swa_k_ring.index_copy(1, ring_slots, k_swa[:, write_from_l:].to(dtype=st.swa_k_ring.dtype))
            )
            new_v_ring = _compact(
                st.swa_v_ring.index_copy(1, ring_slots, v_swa[:, write_from_l:].to(dtype=st.swa_v_ring.dtype))
            )
            new_val_ring = _compact(st.swa_valid_ring.index_copy(1, ring_slots, valid_cur[:, write_from_l:]))
            p_comp = _compact(p_cur)
            new_steps = st.step_count + seq_len

            final_seg_bank: torch.Tensor | None = None
            if surprise_mode == "segment_start":
                if end_step % seg_tokens == 0:
                    final_seg_bank = p_comp
                else:
                    seg_end_k = (end_step // seg_tokens) * seg_step
                    if seg_end_k > n_0 and (seg_end_k - n_0) in produced_boundaries:
                        final_seg_bank = _compact(produced_boundaries[seg_end_k - n_0])
                    elif seg_end_k == n_0:
                        final_seg_bank = _compact(st.page_states)
                    elif st.s_seg_start_bank is not None:
                        final_seg_bank = _compact(st.s_seg_start_bank)
                    else:
                        final_seg_bank = _compact(st.s_chunk_start_bank)

            if can_build_compact:
                if k_far_end == 0 and n_0 == 0:
                    s_far_final = _compact(st.s_far_bank)
                elif k_far_end == n_0:
                    s_far_final = _compact(st.page_states)
                else:
                    s_far_final = _compact(produced_boundaries[k_far_end - n_0])
                if end_step % chunk_size == 0:
                    s_cstart_final = p_comp
                elif n_end > n_0:
                    s_cstart_final = _compact(produced_boundaries[n_end - n_0])
                else:
                    s_cstart_final = _compact(st.s_chunk_start_bank)

                rep_d_g = int(ch_gamma.shape[-1])
                new_rep_ke = query.new_zeros((bsz, hv, max_win, dk), dtype=acc_dt)
                new_rep_ve = query.new_zeros((bsz, hv, max_win, dv), dtype=acc_dt)
                new_rep_g = query.new_zeros((bsz, max_win, top_k_pages, hv), dtype=acc_dt)
                new_rep_gam = query.new_zeros((bsz, max_win, top_k_pages, hv, rep_d_g), dtype=acc_dt)
                new_rep_idx = torch.zeros((bsz, max_win, top_k_pages), device=query.device, dtype=torch.long)

                c_from_g = max(step_offset, k_far_end * chunk_size)
                c_from_l = c_from_g - step_offset
                if c_from_g < end_step:
                    tail_len = end_step - c_from_g
                    r_slots = torch.arange(c_from_g, end_step, device=query.device) % max_win
                    k_tail = k_in[:, c_from_l:].float()
                    if use_qk_l2norm_in_kernel:
                        k_tail = l2norm(k_tail, dim=-1, eps=1e-6)
                    if use_phase:
                        k_tail = apply_2d_rotation_trig(
                            k_tail, cos_t[:, c_from_l:].float(), sin_t[:, c_from_l:].float()
                        )
                    ke_tail = (_safe_sqrt(b_gate[:, c_from_l:].float()) * k_tail).transpose(1, 2)
                    ve_tail = (_safe_sqrt(w_gate[:, c_from_l:].float()) * value[:, c_from_l:].float()).transpose(1, 2)
                    g_tail = ste_w[:, c_from_l:].unsqueeze(-1) * g_root[:, c_from_l:].unsqueeze(2)
                    ps_sel = ps_gamma.index_select(0, tw_idx[:, c_from_l:].reshape(-1)).view(
                        bsz, tail_len, top_k_pages, 1, 1
                    )
                    gam_tail = ps_sel * ch_gamma[:, c_from_l:].unsqueeze(2)
                    new_rep_ke = new_rep_ke.index_copy(2, r_slots, ke_tail)
                    new_rep_ve = new_rep_ve.index_copy(2, r_slots, ve_tail)
                    new_rep_g = new_rep_g.index_copy(1, r_slots, g_tail)
                    new_rep_gam = new_rep_gam.index_copy(1, r_slots, gam_tail)
                    new_rep_idx = new_rep_idx.index_copy(1, r_slots, tw_idx[:, c_from_l:])

                final_state = _cast_state_dtype(
                    HSSDCacheState(
                        step_count=new_steps,
                        chunk_size=chunk_size,
                        window_chunks=window_chunks,
                        page_states=p_comp,
                        theta_cum=term_theta,
                        boundary_ring=None,
                        swa_k_ring=new_k_ring,
                        swa_v_ring=new_v_ring,
                        swa_valid_ring=new_val_ring,
                        s_far_bank=s_far_final,
                        s_chunk_start_bank=s_cstart_final,
                        s_seg_start_bank=final_seg_bank,
                        replay_buf=(
                            _compact(new_rep_ke),
                            _compact(new_rep_ve),
                            _compact(new_rep_g),
                            _compact(new_rep_gam),
                            _compact(new_rep_idx),
                        ),
                        _uniform_step=end_step,
                    ),
                    state_dtype,
                    acc_dt,
                )
            else:
                ring_list: list[torch.Tensor] = []
                for r_pos in range(window_chunks + 1):
                    k_want = n_end - window_chunks + r_pos
                    if k_want <= n_0:
                        ring_list.append(_compact(st.page_states))
                    elif k_want == n_end and end_step % chunk_size == 0:
                        ring_list.append(p_comp)
                    else:
                        ring_list.append(_compact(produced_boundaries[k_want - n_0]))
                final_state = _cast_state_dtype(
                    HSSDCacheState(
                        step_count=new_steps,
                        chunk_size=chunk_size,
                        window_chunks=window_chunks,
                        page_states=p_comp,
                        theta_cum=term_theta,
                        boundary_ring=tuple(ring_list),
                        swa_k_ring=new_k_ring,
                        swa_v_ring=new_v_ring,
                        swa_valid_ring=new_val_ring,
                        s_seg_start_bank=final_seg_bank,
                        _uniform_step=end_step,
                    ),
                    state_dtype,
                    acc_dt,
                )

    return out, final_state, u_total, tok_total


def _chunk_hssd_raw(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
    boltzmann_gate: PeakBoltzmannGate | None,
    swa_recency_slope: torch.Tensor | None,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    segment_chunks: int,
    checkpoint_segments: bool,
    checkpoint_segments_min_len: int,
    outer_checkpoint_max_tokens: int,
    initial_state: HSSDCacheState | torch.Tensor | None,
    output_final_state: bool,
    write_mask: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
    g_override: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
    state_dtype: str | torch.dtype,
    use_swa: bool,
    use_phase: bool,
    use_surprise: bool,
    surprise_mode: str,
    write_mode: str,
    compact_active_pages: bool,
    compact_decode_cache: bool,
    compile_chunks: bool,
    detach_final_state: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor, torch.Tensor]:
    init_dtype = query.dtype
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32

    doc_spans = _validate_packed_boundaries(
        bsz=bsz, seq_len=seq_len, initial_state=initial_state, cu_seqlens=cu_seqlens, seq_idx=seq_idx
    )
    if doc_spans is not None:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_tot = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot = query.new_zeros((), dtype=acc_dt)
        for b, spans_b in enumerate(doc_spans):
            sl_b = slice(b, b + 1)
            doc_outs: list[torch.Tensor] = []
            last_st: HSSDCacheState | None = None
            for s0, s1 in spans_b:
                sl_t = slice(s0, s1)
                o_d, st_d, u_d, c_d = _chunk_hssd_raw(
                    query[sl_b, sl_t],
                    key[sl_b, sl_t],
                    value[sl_b, sl_t],
                    _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=sl_t),
                    b_gate[sl_b, sl_t],
                    w_gate[sl_b, sl_t],
                    theta[sl_b, sl_t],
                    surprise_bias[sl_b, sl_t],
                    surprise_slope,
                    surprise_thresh,
                    write_logits[sl_b, sl_t],
                    read_logits[sl_b, sl_t],
                    q_swa[sl_b, sl_t],
                    k_swa[sl_b, sl_t],
                    v_swa[sl_b, sl_t],
                    boltzmann_gate=boltzmann_gate,
                    swa_recency_slope=swa_recency_slope,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    segment_chunks=segment_chunks,
                    checkpoint_segments=checkpoint_segments,
                    checkpoint_segments_min_len=checkpoint_segments_min_len,
                    outer_checkpoint_max_tokens=outer_checkpoint_max_tokens,
                    initial_state=None,
                    output_final_state=output_final_state,
                    write_mask=None if write_mask is None else write_mask[sl_b, sl_t],
                    pad_mask=None if pad_mask is None else pad_mask[sl_b, sl_t],
                    cu_seqlens=None,
                    seq_idx=None,
                    g_override=None if g_override is None else g_override[sl_b, sl_t],
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                    state_dtype=state_dtype,
                    use_swa=use_swa,
                    use_phase=use_phase,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                    write_mode=write_mode,
                    compact_active_pages=compact_active_pages,
                    compact_decode_cache=compact_decode_cache,
                    compile_chunks=compile_chunks,
                )
                doc_outs.append(o_d)
                last_st = st_d
                u_tot = u_tot + u_d
                cnt_tot = cnt_tot + c_d
            row_outs.append(
                torch.cat(doc_outs, dim=1)
                if doc_outs
                else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
            )
            if output_final_state:
                if last_st is None:
                    last_st = _cast_state_dtype(
                        HSSDCacheState.zeros(
                            1, num_pages, hv, dk, dv,
                            chunk_size=chunk_size, window_chunks=window_chunks,
                            top_k_pages=top_k_pages,
                            compact_decode_cache=compact_decode_cache,
                            device=query.device, dtype=acc_dt,
                        ),
                        state_dtype,
                        acc_dt,
                    )
                row_states.append(last_st)
        out_all = row_outs[0] if len(row_outs) == 1 else torch.cat(row_outs, dim=0)
        st_all = (
            (row_states[0] if len(row_states) == 1 else torch.cat(row_states, dim=0))
            if output_final_state
            else None
        )
        return out_all, st_all, u_tot, cnt_tot

    target_dt = _resolve_state_dtype(state_dtype, acc_dt)
    if seq_len == 0:
        st0 = _coerce_init_state(
            initial_state,
            bsz=bsz,
            num_pages=num_pages,
            hv=hv,
            dk=dk,
            dv=dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            top_k_pages=top_k_pages,
            compact_decode_cache=compact_decode_cache,
            device=query.device,
            dtype=target_dt,
        )
        empty_out = query.new_zeros(bsz, 0, hv, dv, dtype=init_dtype)
        final_0 = _cast_state_dtype(st0, target_dt, acc_dt) if output_final_state else None
        return empty_out, final_0, query.new_zeros(num_pages, dtype=acc_dt), query.new_zeros((), dtype=acc_dt)

    if seq_len == 1:
        st1 = _coerce_init_state(
            initial_state,
            bsz=bsz,
            num_pages=num_pages,
            hv=hv,
            dk=dk,
            dv=dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            top_k_pages=top_k_pages,
            compact_decode_cache=compact_decode_cache,
            device=query.device,
            dtype=target_dt,
        )
        o_step, st_next, u_sum, cnt_sum = recurrent_hssd_step(
            query,
            key,
            value,
            gamma,
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
            st1,
            boltzmann_gate=boltzmann_gate,
            swa_recency_slope=swa_recency_slope,
            top_k_pages=top_k_pages,
            segment_chunks=segment_chunks,
            write_mask=write_mask,
            pad_mask=pad_mask,
            g_override=g_override,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            state_dtype=target_dt,
            use_swa=use_swa,
            use_phase=use_phase,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
            write_mode=write_mode,
            compact_decode_cache=compact_decode_cache,
            return_duty_sum=True,
        )
        return o_step, (st_next if output_final_state else None), u_sum, cnt_sum

    st = _coerce_init_state(
        initial_state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        top_k_pages=top_k_pages,
        compact_decode_cache=compact_decode_cache,
        device=query.device,
        dtype=acc_dt,
    )

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, seq_len).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())
    inhomogeneous_steps = (
        bsz > 1
        and st._uniform_step is None
        and not bool((st.step_count == st.step_count[0]).all())
    )

    if has_pad_zeros or inhomogeneous_steps:
        row_outs_p: list[torch.Tensor] = []
        row_states_p: list[HSSDCacheState] = []
        u_tot_p = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot_p = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = st[b : b + 1]
            sl_b = slice(b, b + 1)
            idx_b = pm_bool[b].nonzero(as_tuple=False).squeeze(-1) if pm_bool is not None else None
            if idx_b is not None and idx_b.numel() == 0:
                row_outs_p.append(query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype))
                if output_final_state:
                    row_states_p.append(_cast_state_dtype(st_b, state_dtype, acc_dt))
                continue
            tsl: slice | torch.Tensor | None = (
                idx_b if (idx_b is not None and idx_b.numel() < seq_len) else None
            )
            q_b = query[sl_b] if tsl is None else query[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            k_b = key[sl_b] if tsl is None else key[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            v_b = value[sl_b] if tsl is None else value[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            gam_b = _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=tsl)
            bg_b = b_gate[sl_b] if tsl is None else b_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wg_b = w_gate[sl_b] if tsl is None else w_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            th_b = theta[sl_b] if tsl is None else theta[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            sb_b = surprise_bias[sl_b] if tsl is None else surprise_bias[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wl_b = write_logits[sl_b] if tsl is None else write_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            rl_b = read_logits[sl_b] if tsl is None else read_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            qs_b = q_swa[sl_b] if tsl is None else q_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            ks_b = k_swa[sl_b] if tsl is None else k_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            vs_b = v_swa[sl_b] if tsl is None else v_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wm_b = None
            if write_mask is not None:
                wm_row = write_mask.reshape(bsz, seq_len)[sl_b]
                wm_b = wm_row if tsl is None else wm_row.index_select(1, idx_b)  # type: ignore[arg-type]
            go_b = None
            if g_override is not None:
                go_b = g_override[sl_b] if tsl is None else g_override[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]

            o_valid, st_next_b, u_b, c_b = _chunk_hssd_raw(
                q_b,
                k_b,
                v_b,
                gam_b,
                bg_b,
                wg_b,
                th_b,
                sb_b,
                surprise_slope,
                surprise_thresh,
                wl_b,
                rl_b,
                qs_b,
                ks_b,
                vs_b,
                boltzmann_gate=boltzmann_gate,
                swa_recency_slope=swa_recency_slope,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                segment_chunks=segment_chunks,
                checkpoint_segments=checkpoint_segments,
                checkpoint_segments_min_len=checkpoint_segments_min_len,
                outer_checkpoint_max_tokens=outer_checkpoint_max_tokens,
                initial_state=st_b,
                output_final_state=output_final_state,
                write_mask=wm_b,
                pad_mask=None,
                cu_seqlens=None,
                seq_idx=None,
                g_override=go_b,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=state_dtype,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_active_pages=compact_active_pages,
                compact_decode_cache=compact_decode_cache,
                compile_chunks=compile_chunks,
            )
            if tsl is None:
                row_outs_p.append(o_valid)
            else:
                assert idx_b is not None
                n_val = int(idx_b.numel())
                idx_exp = idx_b.view(1, n_val, 1, 1).expand(1, n_val, hv, dv)
                out_row = o_valid.new_zeros(1, seq_len, hv, dv).scatter(1, idx_exp, o_valid)
                row_outs_p.append(out_row)
            if output_final_state:
                assert st_next_b is not None
                row_states_p.append(st_next_b)
            u_tot_p = u_tot_p + u_b
            cnt_tot_p = cnt_tot_p + c_b

        out_p = row_outs_p[0] if len(row_outs_p) == 1 else torch.cat(row_outs_p, dim=0)
        st_p = (
            (row_states_p[0] if len(row_states_p) == 1 else torch.cat(row_states_p, dim=0))
            if output_final_state
            else None
        )
        return out_p, st_p, u_tot_p, cnt_tot_p

    step_offset_0 = st._uniform_step if st._uniform_step is not None else int(st.step_count[0].item())
    ps_gamma, ch_gamma = _normalize_gamma_spec(
        gamma, num_pages=num_pages, dk=dk, acc_dt=acc_dt, device=query.device
    )
    if (
        _HAS_TRITON
        and query.is_cuda
        and not _is_compiling()
        and not compile_chunks
        and _run_chunk_group is _ORIG_RUN_CHUNK_GROUP
        and checkpoint is _ORIG_CHECKPOINT
        and init_dtype in (torch.bfloat16, torch.float16)
        and (
            step_offset_0 == 0
            or (
                step_offset_0 % chunk_size == 0
                and seq_len >= window_chunks * chunk_size
                and surprise_mode != "segment_start"
            )
        )
        and chunk_size == 64
        and dk in (64, 128)
        and dv in (64, 128)
        and g_override is None
        and 1 <= top_k_pages <= num_pages
        and window_chunks >= 0
        and ps_gamma is not None
        and ch_gamma.shape[-1] == dk // 2
    ):
        any_grad_in = torch.is_grad_enabled() and any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in (
                query,
                key,
                value,
                ps_gamma,
                ch_gamma,
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
                st.page_states,
                write_mask,
                swa_recency_slope,
                None if boltzmann_gate is None else boltzmann_gate.log_kappa,
                None if boltzmann_gate is None else boltzmann_gate.tau,
            )
        )
        st_detached = not (
            st.page_states.requires_grad
            or st.s_far_bank.requires_grad
            or st.swa_k_ring.requires_grad
            or st.swa_v_ring.requires_grad
            or st.theta_cum.requires_grad
        )
        if (
            not any_grad_in
            or (not output_final_state and step_offset_0 == 0)
            or (st_detached and (not output_final_state or detach_final_state))
        ):
            return _chunk_hssd_triton_fast(
                query,
                key,
                value,
                ps_gamma,
                ch_gamma,
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
                st=st,
                boltzmann_gate=boltzmann_gate,
                swa_recency_slope=swa_recency_slope,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                segment_chunks=segment_chunks,
                output_final_state=output_final_state,
                write_mask=write_mask,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=state_dtype,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_decode_cache=compact_decode_cache,
                detach_final_state=detach_final_state,
            )

    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    swa_dt = init_dtype if (q.is_cuda and init_dtype in (torch.bfloat16, torch.float16)) else acc_dt
    qs = q_swa.to(swa_dt) if swa_dt != acc_dt else upcast(q_swa).to(acc_dt)
    ks = k_swa.to(swa_dt) if swa_dt != acc_dt else upcast(k_swa).to(acc_dt)
    vs = v_swa.to(swa_dt) if swa_dt != acc_dt else upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)
    swa_slp = None if swa_recency_slope is None else upcast(swa_recency_slope).to(device=q.device, dtype=acc_dt)

    th64 = theta.to(dtype=torch.float64)
    if write_mask is not None:
        wm = write_mask.reshape(bsz, seq_len).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_cur = wm > 0.5
        th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_cur = torch.ones(bsz, seq_len, device=q.device, dtype=torch.bool)
    if not use_phase:
        th64 = torch.zeros_like(th64)

    # Bug 4.3 + H9: Accumulate phase in float64; evaluate modulo + cos/sin ONCE for both Q and K
    theta_cum_all = torch.remainder(
        st.theta_cum.to(dtype=torch.float64).unsqueeze(1) + torch.cumsum(th64, dim=1),
        2.0 * math.pi,
    )
    sc = (dk ** -0.5) if scale is None else float(scale)
    if use_phase:
        cos_t, sin_t = compute_2d_trig(-theta_cum_all, q.dtype)
        q_hat = apply_2d_rotation_trig(q, cos_t, sin_t)
        k_hat = apply_2d_rotation_trig(k, cos_t, sin_t)
    else:
        q_hat = q
        k_hat = k
    if sc != 1.0:
        q_hat = q_hat * sc

    step_offset = st._uniform_step if st._uniform_step is not None else int(st.step_count[0].item())
    end_step = step_offset + seq_len
    max_win = st.max_swa_len
    _, _, swa_base_global = disjoint_horizon_bounds(step_offset + 1, chunk_size, window_chunks)
    if step_offset > swa_base_global:
        prior_idx = torch.arange(swa_base_global, step_offset, device=q.device) % max_win
        all_ks = torch.cat([st.swa_k_ring.index_select(1, prior_idx).to(swa_dt), ks], dim=1)
        all_vs = torch.cat([st.swa_v_ring.index_select(1, prior_idx).to(swa_dt), vs], dim=1)
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
        gate_tau = q.new_full((1,), 1.0, dtype=acc_dt)

    all_chunks = _chunk_slices_for_span(step_offset, seq_len, chunk_size)
    n_chunks = len(all_chunks)
    seg_step = max(1, int(segment_chunks))
    seg_tokens = seg_step * chunk_size
    seg_groups = _segment_groups_for_chunks(
        all_chunks,
        step_offset=step_offset,
        chunk_size=chunk_size,
        seg_step=seg_step,
        align_global_segments=(surprise_mode == "segment_start"),
    )

    eff_compact_pages = bool(
        compact_active_pages and not compile_chunks and not (q.is_cuda and bsz * num_pages <= 32)
    )
    group_runner = _get_compiled_chunk_group() if compile_chunks else _run_chunk_group

    # H4: Precompute o_far for initial chunks `m <= W` so `st.boundary_ring` NEVER enters segment checkpoints
    o_far_slots: list[torch.Tensor | None] = [None] * n_chunks
    n_init_pre = min(n_chunks, window_chunks + 1)
    if n_init_pre > 0:
        init_end_tok = all_chunks[n_init_pre - 1][1]
        pi_r_init = F.softmax(rl[:, :init_end_tok], dim=-1)
        tv_r_init, ti_r_init = torch.topk(pi_r_init, k=top_k_pages, dim=-1)
        tw_r_init = tv_r_init / tv_r_init.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        init_plans: list[_ActivePagePlan | None]
        if eff_compact_pages and top_k_pages < num_pages and not _is_compiling():
            init_plans = list(
                _build_multi_span_active_plans(
                    [ti_r_init[:, a_m:b_m] for a_m, b_m in all_chunks[:n_init_pre]],
                    num_pages,
                )
            )
        else:
            init_plans = [None] * n_init_pre

        if step_offset == 0:
            for m in range(n_init_pre):
                a_m, b_m = all_chunks[m]
                o_far_slots[m] = _far_readout(
                    q_hat[:, a_m:b_m],
                    st.page_states,
                    rl[:, a_m:b_m],
                    top_k_pages,
                    compact_active_pages=eff_compact_pages,
                    precomputed_routing=(ti_r_init[:, a_m:b_m], tw_r_init[:, a_m:b_m]),
                    active_plan=init_plans[m],
                )
        elif n_init_pre == 1:
            a_0, b_0 = all_chunks[0]
            o_far_slots[0] = _far_readout(
                q_hat[:, a_0:b_0],
                st.s_far_bank,
                rl[:, a_0:b_0],
                top_k_pages,
                compact_active_pages=eff_compact_pages,
                precomputed_routing=(ti_r_init[:, a_0:b_0], tw_r_init[:, a_0:b_0]),
                active_plan=init_plans[0],
            )
        else:
            init_ring = st.boundary_ring
            for m in range(n_init_pre):
                a_m, b_m = all_chunks[m]
                o_far_slots[m] = _far_readout(
                    q_hat[:, a_m:b_m],
                    init_ring[m],
                    rl[:, a_m:b_m],
                    top_k_pages,
                    compact_active_pages=eff_compact_pages,
                    precomputed_routing=(ti_r_init[:, a_m:b_m], tw_r_init[:, a_m:b_m]),
                    active_plan=init_plans[m],
                )

    # Determine which trailing chunk boundaries / replay slices are needed if output_final_state=True (`H5`)
    n_0 = step_offset // chunk_size
    n_end = end_step // chunk_size
    k_far_end = max(0, n_end - window_chunks)
    can_build_compact = bool(
        output_final_state
        and compact_decode_cache
        and (st.replay_buf is not None or step_offset <= k_far_end * chunk_size)
    )
    need_global_boundaries: set[int] = set()
    replay_start_global = end_step
    if output_final_state:
        if can_build_compact:
            if k_far_end > n_0:
                need_global_boundaries.add(k_far_end)
            if n_end > n_0 and end_step % chunk_size != 0:
                need_global_boundaries.add(n_end)
            if surprise_mode == "segment_start":
                seg_end_k = (end_step // seg_tokens) * seg_step
                if seg_end_k > n_0 and end_step % seg_tokens != 0:
                    need_global_boundaries.add(seg_end_k)
            replay_start_global = max(step_offset, k_far_end * chunk_size)
        else:
            for k_b in range(max(n_0 + 1, n_end - window_chunks), n_end + 1):
                need_global_boundaries.add(k_b)
            if surprise_mode == "segment_start":
                seg_end_k = (end_step // seg_tokens) * seg_step
                if seg_end_k > n_0 and end_step % seg_tokens != 0:
                    need_global_boundaries.add(seg_end_k)

    need_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (
            q_hat,
            k_hat,
            v,
            ps_gamma,
            ch_gamma,
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
            swa_slp,
            st.page_states,
            go,
        )
    )
    skip_nested_ckpt = (
        int(outer_checkpoint_max_tokens) > 0
        and bsz * seq_len <= int(outer_checkpoint_max_tokens)
        and (_in_backward() or _in_outer_checkpoint())
    )
    use_ckpt = bool(
        checkpoint_segments
        and seq_len >= int(checkpoint_segments_min_len)
        and torch.is_grad_enabled()
        and need_grad
        and len(seg_groups) > 1
        and not skip_nested_ckpt
    )

    p_cur = st.page_states
    seg_outs: list[torch.Tensor] = []
    u_total = q.new_zeros(num_pages, dtype=acc_dt)
    tok_total = q.new_zeros((), dtype=acc_dt)
    produced_boundaries: dict[int, torch.Tensor] = {}
    replay_pieces: list[
        tuple[int, int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = []

    for seg_i, (c_idx, c_end) in enumerate(seg_groups):
        c_group = all_chunks[c_idx:c_end]
        n_grp = len(c_group)
        l_start, l_end = c_group[0][0], c_group[-1][1]
        g_first = step_offset + l_start
        g_last = step_offset + l_end
        _, _, grp_far_cut = disjoint_horizon_bounds(g_first + 1, chunk_size, window_chunks)
        s_lo = grp_far_cut - swa_base_global
        s_hi = g_last - swa_base_global
        rel_bounds = [(a - l_start, b - l_start) for a, b in c_group]

        # Build precomputed `o_far_pre` for chunks in `c_group` whose far bank came from before `c_idx`
        pre_tensors: list[torch.Tensor] = []
        pre_chunk_map: list[tuple[int, int] | None] = []
        p_cursor = 0
        for c_local in range(n_grp):
            m_glob = c_idx + c_local
            slot_t = o_far_slots[m_glob]
            if slot_t is not None:
                t_len = slot_t.shape[1]
                pre_tensors.append(slot_t)
                pre_chunk_map.append((p_cursor, p_cursor + t_len))
                p_cursor += t_len
                o_far_slots[m_glob] = None
            else:
                pre_chunk_map.append(None)
        if pre_tensors:
            o_far_pre = pre_tensors[0] if len(pre_tensors) == 1 else torch.cat(pre_tensors, dim=1)
        else:
            o_far_pre = q_hat.new_zeros((bsz, 0, hv, dv))

        # Build future read-ahead slices `q_fut_seg`, `rl_fut_seg` for chunks `m_fut >= c_end` fed by `c_group`
        fut_src_locals: list[int] = []
        fut_dst_globals: list[int] = []
        for c_local in range(n_grp):
            g1_c = step_offset + c_group[c_local][1]
            if g1_c % chunk_size == 0:
                m_fut = c_idx + c_local + window_chunks + 1
                if c_end <= m_fut < n_chunks:
                    fut_src_locals.append(c_local)
                    fut_dst_globals.append(m_fut)
        if fut_dst_globals:
            fut_l0 = all_chunks[fut_dst_globals[0]][0]
            fut_l1 = all_chunks[fut_dst_globals[-1]][1]
            q_fut_seg = q_hat[:, fut_l0:fut_l1]
            rl_fut_seg = rl[:, fut_l0:fut_l1]
            fut_chunk_bounds = [
                (all_chunks[m_f][0] - fut_l0, all_chunks[m_f][1] - fut_l0)
                for m_f in fut_dst_globals
            ]
        else:
            q_fut_seg = q_hat.new_zeros((bsz, 0, hv, dk))
            rl_fut_seg = rl.new_zeros((bsz, 0, num_pages))
            fut_chunk_bounds = []

        # Trailing boundary / replay capture (ONLY non-empty near the end when output_final_state=True)
        save_boundary_locals: list[int] = []
        save_boundary_globals: list[int] = []
        if need_global_boundaries:
            for c_local in range(n_grp):
                g1_c = step_offset + c_group[c_local][1]
                if g1_c % chunk_size == 0:
                    k_done = g1_c // chunk_size
                    if k_done in need_global_boundaries:
                        save_boundary_locals.append(c_local)
                        save_boundary_globals.append(k_done)
        collect_replay = bool(can_build_compact and g_last > replay_start_global)

        first_surp_bank: torch.Tensor | None = None
        if seg_i == 0:
            if surprise_mode == "segment_start" and g_first % seg_tokens != 0:
                first_surp_bank = (
                    st.s_seg_start_bank if st.s_seg_start_bank is not None else st.s_chunk_start_bank
                )
            elif surprise_mode != "segment_start" and g_first % chunk_size != 0:
                first_surp_bank = st.s_chunk_start_bank

        q_seg = q_hat[:, l_start:l_end]
        k_seg = k_hat[:, l_start:l_end]
        v_seg = v[:, l_start:l_end]
        ch_seg = ch_gamma[:, l_start:l_end]
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

        # When output_final_state=False, `save_boundary_locals` is empty and `collect_replay` is False,
        # so `_ckpt_fn` crosses segment boundaries with ONLY ONE bank `pst = p_cur` (`H4` ~8-10x state cut)!
        if use_ckpt and not collect_replay and first_surp_bank is None:
            ps_arg = ps_gamma if ps_gamma is not None else q.new_empty(0)
            has_ps = ps_gamma is not None
            swaslp_arg = swa_slp if swa_slp is not None else q.new_empty(0)
            has_swaslp = swa_slp is not None

            def _ckpt_fn(
                qc: torch.Tensor,
                kc: torch.Tensor,
                vc: torch.Tensor,
                psc: torch.Tensor,
                chc: torch.Tensor,
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
                swaslpc: torch.Tensor,
                ofpre: torch.Tensor,
                qfut: torch.Tensor,
                rlfut: torch.Tensor,
                pst: torch.Tensor,
                _rb: list[tuple[int, int]] = rel_bounds,
                _pcm: list[tuple[int, int] | None] = pre_chunk_map,
                _fsl: list[int] = fut_src_locals,
                _fcb: list[tuple[int, int]] = fut_chunk_bounds,
                _sbl: list[int] = save_boundary_locals,
                _goff: int = g_first,
                _sbase: int = grp_far_cut,
                _val: torch.Tensor = val_seg,
                _wm: torch.Tensor | None = wm_seg,
                _go: torch.Tensor | None = go_seg,
                _hps: bool = has_ps,
                _hsw: bool = has_swaslp,
            ):
                o_g, p_n, u_g, tk_g, of_fut, sv_b, _rep = group_runner(
                    _rb,
                    pre_chunk_map=_pcm,
                    fut_src_locals=_fsl,
                    fut_chunk_bounds=_fcb,
                    save_boundary_locals=_sbl,
                    collect_replay=False,
                    step_offset=_goff,
                    swa_base_global=_sbase,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    use_swa=use_swa,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                    write_mode=write_mode,
                    compact_active_pages=eff_compact_pages,
                    q_hat=qc,
                    k_hat=kc,
                    v=vc,
                    gamma_page_scale=(psc if _hps else None),
                    gamma_chan_seg=chc,
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
                    swa_recency_slope=(swaslpc if _hsw else None),
                    o_far_pre=ofpre,
                    q_fut_seg=qfut,
                    rl_fut_seg=rlfut,
                    page_states=pst,
                    first_surprise_bank=None,
                )
                return (o_g, p_n, u_g, tk_g, of_fut, *sv_b)

            ckpt_out = checkpoint(
                _ckpt_fn,
                q_seg,
                k_seg,
                v_seg,
                ps_arg,
                ch_seg,
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
                swaslp_arg,
                o_far_pre,
                q_fut_seg,
                rl_fut_seg,
                p_cur,
                use_reentrant=False,
            )
            o_grp, p_cur, u_grp, tok_grp, o_far_fut = ckpt_out[:5]
            saved_banks = list(ckpt_out[5:])
            replay_seg = None
        else:
            o_grp, p_cur, u_grp, tok_grp, o_far_fut, saved_banks, replay_seg = group_runner(
                rel_bounds,
                pre_chunk_map=pre_chunk_map,
                fut_src_locals=fut_src_locals,
                fut_chunk_bounds=fut_chunk_bounds,
                save_boundary_locals=save_boundary_locals,
                collect_replay=collect_replay,
                step_offset=g_first,
                swa_base_global=grp_far_cut,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                use_swa=use_swa,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                write_mode=write_mode,
                compact_active_pages=eff_compact_pages,
                q_hat=q_seg,
                k_hat=k_seg,
                v=v_seg,
                gamma_page_scale=ps_gamma,
                gamma_chan_seg=ch_seg,
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
                swa_recency_slope=swa_slp,
                o_far_pre=o_far_pre,
                q_fut_seg=q_fut_seg,
                rl_fut_seg=rl_fut_seg,
                page_states=p_cur,
                first_surprise_bank=first_surp_bank,
            )

        for m_dst, (fl0, fl1) in zip(fut_dst_globals, fut_chunk_bounds):
            o_far_slots[m_dst] = o_far_fut[:, fl0:fl1]
        for k_glob, b_tensor in zip(save_boundary_globals, saved_banks):
            produced_boundaries[k_glob] = b_tensor
        if replay_seg is not None:
            replay_pieces.append((g_first, g_last, replay_seg))

        seg_outs.append(o_grp)
        u_total = u_total + u_grp
        tok_total = tok_total + tok_grp

    out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)).to(init_dtype)
    final_state: HSSDCacheState | None = None
    if output_final_state:
        keep_from = max(swa_base_global, end_step - max_win)
        rel_from = keep_from - swa_base_global
        ring_slots = torch.arange(keep_from, end_step, device=q.device) % max_win
        new_k_ring = _compact(
            st.swa_k_ring.index_copy(1, ring_slots, all_ks[:, rel_from:].to(st.swa_k_ring.dtype))
        )
        new_v_ring = _compact(
            st.swa_v_ring.index_copy(1, ring_slots, all_vs[:, rel_from:].to(st.swa_v_ring.dtype))
        )
        new_val_ring = _compact(st.swa_valid_ring.index_copy(1, ring_slots, all_valid[:, rel_from:]))
        # Bug 3A: Explicitly clone & compact terminal theta_cum so [B, T, H_v, d_k // 2] is freed!
        term_theta = theta_cum_all[:, -1].clone().contiguous()
        p_comp = _compact(p_cur)
        new_steps = st.step_count + seq_len

        final_seg_bank: torch.Tensor | None = None
        if surprise_mode == "segment_start":
            if end_step % seg_tokens == 0:
                final_seg_bank = p_comp
            else:
                seg_end_k = (end_step // seg_tokens) * seg_step
                if seg_end_k in produced_boundaries:
                    final_seg_bank = _compact(produced_boundaries[seg_end_k])
                elif st.s_seg_start_bank is not None:
                    final_seg_bank = _compact(st.s_seg_start_bank)
                else:
                    final_seg_bank = _compact(st.s_chunk_start_bank)

        if can_build_compact:
            k_far_0 = max(0, n_0 - window_chunks)
            if k_far_end == k_far_0:
                s_far_final = _compact(st.s_far_bank)
            elif k_far_end <= n_0:
                if st._boundary_ring is not None:
                    s_far_final = _compact(st._boundary_ring[window_chunks - (n_0 - k_far_end)])
                else:
                    assert st.replay_buf is not None
                    cur_f = st.s_far_bank
                    for c_adv in range(k_far_0, k_far_end):
                        cur_f = _replay_chunk_on_bank(
                            cur_f,
                            st.replay_buf,
                            c_adv,
                            chunk_size,
                            window_chunks,
                            grad_enabled=torch.is_grad_enabled(),
                        )
                    s_far_final = _compact(cur_f)
            else:
                s_far_final = _compact(produced_boundaries[k_far_end])

            if end_step % chunk_size == 0:
                # Share storage with `p_comp` at exact chunk boundaries (`test_hssd_audit_regressions`)
                s_cstart_final = p_comp
            elif n_end > n_0:
                s_cstart_final = _compact(produced_boundaries[n_end])
            else:
                s_cstart_final = _compact(st.s_chunk_start_bank)

            rep_d_g = int(replay_pieces[0][2][3].shape[-1]) if replay_pieces else (dk // 2)
            if st.replay_buf is not None and int(st.replay_buf[3].shape[-1]) == rep_d_g:
                r_ke, r_ve, r_g, r_gam, r_idx = st.replay_buf
                new_rep_ke = r_ke.to(dtype=acc_dt).clone()
                new_rep_ve = r_ve.to(dtype=acc_dt).clone()
                new_rep_g = r_g.to(dtype=acc_dt).clone()
                new_rep_gam = r_gam.to(dtype=acc_dt).clone()
                new_rep_idx = r_idx.clone()
            else:
                new_rep_ke = q.new_zeros((bsz, hv, max_win, dk), dtype=acc_dt)
                new_rep_ve = q.new_zeros((bsz, hv, max_win, dv), dtype=acc_dt)
                new_rep_g = q.new_zeros((bsz, max_win, top_k_pages, hv), dtype=acc_dt)
                new_rep_gam = q.new_zeros((bsz, max_win, top_k_pages, hv, rep_d_g), dtype=acc_dt)
                new_rep_idx = torch.zeros((bsz, max_win, top_k_pages), device=q.device, dtype=torch.long)

            for g_a, g_b, (p_ke, p_ve, p_g, p_gam, p_idx) in replay_pieces:
                c_from = max(g_a, k_far_end * chunk_size)
                if c_from >= g_b:
                    continue
                off = c_from - g_a
                r_slots = torch.arange(c_from, g_b, device=q.device) % max_win
                new_rep_ke = new_rep_ke.index_copy(2, r_slots, p_ke[:, :, off:])
                new_rep_ve = new_rep_ve.index_copy(2, r_slots, p_ve[:, :, off:])
                new_rep_g = new_rep_g.index_copy(1, r_slots, p_g[:, off:])
                new_rep_gam = new_rep_gam.index_copy(1, r_slots, p_gam[:, off:])
                new_rep_idx = new_rep_idx.index_copy(1, r_slots, p_idx[:, off:])

            final_state = _cast_state_dtype(
                HSSDCacheState(
                    step_count=new_steps,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    page_states=p_comp,
                    theta_cum=term_theta,
                    boundary_ring=None,
                    swa_k_ring=new_k_ring,
                    swa_v_ring=new_v_ring,
                    swa_valid_ring=new_val_ring,
                    s_far_bank=s_far_final,
                    s_chunk_start_bank=s_cstart_final,
                    s_seg_start_bank=final_seg_bank,
                    replay_buf=(
                        _compact(new_rep_ke),
                        _compact(new_rep_ve),
                        _compact(new_rep_g),
                        _compact(new_rep_gam),
                        _compact(new_rep_idx),
                    ),
                    _uniform_step=end_step,
                ),
                state_dtype,
                acc_dt,
            )
        else:
            init_ring_end = st.boundary_ring if n_end - n_0 < window_chunks + 1 else ()
            ring_list: list[torch.Tensor] = []
            for r_pos in range(window_chunks + 1):
                k_want = n_end - window_chunks + r_pos
                if k_want <= 0:
                    if n_0 == 0:
                        ring_list.append(_compact(st.page_states))
                    elif k_want <= n_0 - window_chunks:
                        ring_list.append(_compact(init_ring_end[0]))
                    else:
                        ring_list.append(_compact(init_ring_end[k_want - (n_0 - window_chunks)]))
                elif k_want <= n_0:
                    ring_list.append(_compact(init_ring_end[window_chunks - (n_0 - k_want)]))
                elif k_want == n_end and end_step % chunk_size == 0:
                    ring_list.append(p_comp)
                else:
                    ring_list.append(_compact(produced_boundaries[k_want]))
            final_state = _cast_state_dtype(
                HSSDCacheState(
                    step_count=new_steps,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    page_states=p_comp,
                    theta_cum=term_theta,
                    boundary_ring=tuple(ring_list),
                    swa_k_ring=new_k_ring,
                    swa_v_ring=new_v_ring,
                    swa_valid_ring=new_val_ring,
                    s_seg_start_bank=final_seg_bank,
                    _uniform_step=end_step,
                ),
                state_dtype,
                acc_dt,
            )

    return out, final_state, u_total, tok_total


def chunk_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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
    swa_recency_slope: torch.Tensor | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    segment_chunks: int = 4,
    checkpoint_segments: bool = True,
    checkpoint_segments_min_len: int = 0,
    outer_checkpoint_max_tokens: int = 2048,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    cu_seqlens: Any = None,
    seq_idx: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    write_mode: str = "full",
    compact_active_pages: bool = True,
    compact_decode_cache: bool = True,
    compile_chunks: bool = False,
    detach_final_state: bool = False,
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Parallel chunkwise WY + disjoint local SWA HSSD v3.2 kernel (`O(T)` prefill & training)."""
    out, final_state, u_total, tok_total = _chunk_hssd_raw(
        query,
        key,
        value,
        gamma,
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
        boltzmann_gate=boltzmann_gate,
        swa_recency_slope=swa_recency_slope,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        segment_chunks=segment_chunks,
        checkpoint_segments=checkpoint_segments,
        checkpoint_segments_min_len=checkpoint_segments_min_len,
        outer_checkpoint_max_tokens=outer_checkpoint_max_tokens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        write_mask=write_mask,
        pad_mask=pad_mask,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        g_override=g_override,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        state_dtype=state_dtype,
        use_swa=use_swa,
        use_phase=use_phase,
        use_surprise=use_surprise,
        surprise_mode=surprise_mode,
        write_mode=write_mode,
        compact_active_pages=compact_active_pages,
        compact_decode_cache=compact_decode_cache,
        compile_chunks=compile_chunks,
        detach_final_state=detach_final_state,
    )
    if return_aux:
        return out, final_state, u_total / tok_total.clamp_min(1.0)
    return out, final_state


# ============================================================================ adapter & retrofit

class HSSDAdapter(nn.Module):
    """Per-layer `HSSD v3.2` module attached to each linear-attention (`GatedDeltaNet`) layer."""

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        cfg: HSSDConfig,
        layer_id: int = 0,
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

        # Theorem 1 & Issue 5: Low-rank 2D phase rotation projector (pure linear down->up, zero-init up)
        if cfg.theta_low_rank:
            self.theta_down: nn.Linear | None = nn.Linear(self.hidden_size, eff_rank, bias=False)
            self.theta_up: nn.Linear | None = nn.Linear(
                eff_rank, self.num_v_heads * self.half_k_dim, bias=False
            )
            self.theta_proj: nn.Linear | None = None
        else:
            self.theta_down = None
            self.theta_up = None
            self.theta_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.half_k_dim, bias=False)

        # Fixes H2, B3, B4: Smooth routed chunk-start root-surprise `s = g^2`
        self.surprise_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        self.log_surprise_slope = nn.Parameter(torch.full((self.num_v_heads,), math.log(cfg.surprise_slope_init)))
        self.surprise_thresh = nn.Parameter(torch.full((self.num_v_heads,), cfg.surprise_thresh_init))

        # Fix B2 & H2: Tiered M=8 multi-page write/read routers & log-spaced decay priors
        self.write_router = nn.Linear(self.hidden_size, self.num_pages, bias=True)
        self.read_router = nn.Linear(self.num_v_heads * self.head_k_dim, self.num_pages, bias=True)
        self.log_page_gamma = nn.Parameter(torch.zeros(self.num_pages))

        # Fixes B1, B5 & H3: Local sliding-window Q/K/V adapters + recency slope + PeakBoltzmannGate
        self.swa_q_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_k_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_v_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.swa_q_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.swa_k_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.swa_recency_slope = nn.Parameter(torch.full((self.num_v_heads,), float(cfg.swa_recency_init)))
        self.boltzmann_gate = PeakBoltzmannGate(
            self.num_v_heads, kappa_init=cfg.kappa_init, tau_init=cfg.tau_init
        )

        mu_max_vals = cfg.effective_mu_max()
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
        self.last_dense_router_prob: torch.Tensor | None = None
        self._last_stats: dict[str, float] = {}
        self._pending_stats_tensor: torch.Tensor | None = None

    @property
    def last_stats(self) -> dict[str, float]:
        if self._pending_stats_tensor is not None:
            vals = self._pending_stats_tensor.tolist()
            self._pending_stats_tensor = None
            self._last_stats = {
                "budget_aux_loss": float(vals[0]),
                "vault_duty": float(vals[1]),
                "scratch_duty": float(vals[2]),
                "dense_router_prob": float(vals[3]),
                "dead_page_count": float(vals[4]),
            }
        return self._last_stats

    @last_stats.setter
    def last_stats(self, value: dict[str, float]) -> None:
        self._pending_stats_tensor = None
        self._last_stats = dict(value)

    def _reset_parameters(self) -> None:
        std_h = 1.0 / math.sqrt(self.hidden_size)
        std_q = 1.0 / math.sqrt(self.num_v_heads * self.head_k_dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=std_h)
        nn.init.zeros_(self.erase_up.weight)
        nn.init.zeros_(self.write_up.weight)
        nn.init.zeros_(self.decay_up.weight)
        if self.theta_down is not None and self.theta_up is not None:
            nn.init.kaiming_uniform_(self.theta_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.theta_up.weight)
        elif self.theta_proj is not None:
            nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.surprise_proj.weight)
        nn.init.constant_(self.surprise_proj.bias, float(self.cfg.surprise_bias_init))
        # H2: Matched step-0 page biases favoring working pages `{M-2, M-1}` + small weights
        nn.init.normal_(self.write_router.weight, mean=0.0, std=0.01 * std_h)
        nn.init.normal_(self.read_router.weight, mean=0.0, std=0.01 * std_q)
        if self.num_pages > 1:
            with torch.no_grad():
                page_bias = torch.linspace(
                    -0.5,
                    0.5,
                    self.num_pages,
                    device=self.write_router.bias.device,
                    dtype=self.write_router.bias.dtype,
                )
                self.write_router.bias.copy_(page_bias)
                self.read_router.bias.copy_(page_bias)
        else:
            nn.init.zeros_(self.write_router.bias)
            nn.init.zeros_(self.read_router.bias)
        nn.init.zeros_(self.swa_q_up.weight)
        nn.init.zeros_(self.swa_k_up.weight)
        nn.init.zeros_(self.swa_v_up.weight)
        nn.init.constant_(self.swa_recency_slope, float(self.cfg.swa_recency_init))

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
        """Compute HSSD gate, router, phase, and SWA projections with factored decay (`Bug 3B`, `H2`)."""
        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        half_dk = self.half_k_dim

        if (
            _HAS_TRITON
            and x_in.is_cuda
            and not _is_compiling()
            and query.dtype in (torch.bfloat16, torch.float16)
            and key.dtype == query.dtype
            and value.dtype == query.dtype
            and seq_len >= 16
            and dk in (64, 128)
            and dv in (64, 128)
            and hv % query.shape[2] == 0
            and hv % key.shape[2] == 0
        ):
            xf = upcast(x_in)
            h = F.silu(F.linear(xf, upcast(self.down.weight)))
            cap_g = float(self.cfg.gate_logit_cap)
            cap_d = float(self.cfg.decay_log_cap)
            q_out, k_out, b_gate, w_gate, q_swa, k_swa, v_swa = _FusedAdapterProjFn.apply(
                query,
                key,
                value,
                h,
                beta_base,
                self.erase_up.weight,
                self.write_up.weight,
                self.swa_q_up.weight,
                self.swa_k_up.weight,
                self.swa_v_up.weight,
                self.swa_q_norm,
                self.swa_k_norm,
                hv,
                cap_g,
                use_qk_l2norm,
                bool(self.cfg.paired_b_gate),
            )
            d_decay_pair = (cap_d * torch.tanh(F.linear(h, upcast(self.decay_up.weight)))).view(
                bsz, seq_len, hv, half_dk
            )
            if self.cfg.use_stock_decay_init:
                rel_prior = upcast(self.page_gamma_prior) / float(self.cfg.decay_gamma_max)
                gamma_page_scale = rel_prior * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
                stock_gamma = (-upcast(g_base).clamp(max=0.0)).unsqueeze(-1)
                stock_base = torch.where(
                    stock_gamma > 0.0,
                    stock_gamma.clamp(float(self.cfg.decay_gamma_min), float(self.cfg.decay_gamma_max)),
                    torch.full_like(stock_gamma, float(self.cfg.decay_gamma_max)),
                )
                gamma_chan_pair = stock_base * d_decay_pair.exp()
            else:
                base_rate = 1.0 + 0.1 * (-upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
                gamma_chan_pair = base_rate * d_decay_pair.exp()
                gamma_page_scale = (
                    upcast(self.page_gamma_prior) * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
                )
            if not self.cfg.use_phase:
                theta = xf.new_zeros(bsz, seq_len, hv, half_dk)
            elif self.theta_down is not None and self.theta_up is not None:
                theta_raw = F.linear(
                    F.linear(xf, upcast(self.theta_down.weight)),
                    upcast(self.theta_up.weight),
                )
                theta = math.pi * torch.tanh(theta_raw).view(bsz, seq_len, hv, half_dk)
            else:
                assert self.theta_proj is not None
                theta = math.pi * torch.tanh(F.linear(xf, upcast(self.theta_proj.weight))).view(
                    bsz, seq_len, hv, half_dk
                )
            if theta.requires_grad and seq_len > self.cfg.chunk_size:
                # Normalize cumulative-phase backward spectral norm ||cumsum_T||_2 = O(T) to chunk scale O(C).
                theta_grad_scale = float(self.cfg.chunk_size) / float(seq_len)
                theta = theta.detach() + theta_grad_scale * (theta - theta.detach())
            surprise_bias = F.linear(xf, upcast(self.surprise_proj.weight), upcast(self.surprise_proj.bias))
            write_logits = F.linear(xf, upcast(self.write_router.weight), upcast(self.write_router.bias))
            q_flat = q_out.float().reshape(bsz, seq_len, hv * dk)
            read_logits = F.linear(q_flat, upcast(self.read_router.weight), upcast(self.read_router.bias))
            return {
                "query": q_out,
                "key": k_out,
                "value": value,
                "gamma_page_scale": gamma_page_scale,
                "gamma_chan_pair": gamma_chan_pair,
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
        if self.cfg.paired_b_gate:
            b_gate = (
                b_gate.view(bsz, seq_len, hv, half_dk, 2)
                .mean(dim=-1, keepdim=True)
                .expand(bsz, seq_len, hv, half_dk, 2)
                .reshape(bsz, seq_len, hv, dk)
            )
        w_gate = logistic_shift(beta_f, d_write)

        # Bug 3B & H2: Factored decay `gamma_page_scale` [M] * `gamma_chan_pair` [B, T, H_v, d_k // 2]
        if self.cfg.use_stock_decay_init:
            rel_prior = upcast(self.page_gamma_prior) / float(self.cfg.decay_gamma_max)
            gamma_page_scale = rel_prior * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
            stock_gamma = (-upcast(g_base).clamp(max=0.0)).unsqueeze(-1)
            stock_base = torch.where(
                stock_gamma > 0.0,
                stock_gamma.clamp(float(self.cfg.decay_gamma_min), float(self.cfg.decay_gamma_max)),
                torch.full_like(stock_gamma, float(self.cfg.decay_gamma_max)),
            )
            gamma_chan_pair = stock_base * d_decay_pair.exp()
        else:
            base_rate = 1.0 + 0.1 * (-upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
            gamma_chan_pair = base_rate * d_decay_pair.exp()
            gamma_page_scale = upcast(self.page_gamma_prior) * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()

        if not self.cfg.use_phase:
            theta = xf.new_zeros(bsz, seq_len, hv, half_dk)
        elif self.theta_down is not None and self.theta_up is not None:
            theta_raw = F.linear(
                F.linear(xf, upcast(self.theta_down.weight)),
                upcast(self.theta_up.weight),
            )
            theta = math.pi * torch.tanh(theta_raw).view(bsz, seq_len, hv, half_dk)
        else:
            assert self.theta_proj is not None
            theta = math.pi * torch.tanh(F.linear(xf, upcast(self.theta_proj.weight))).view(bsz, seq_len, hv, half_dk)
        if theta.requires_grad and seq_len > self.cfg.chunk_size:
            theta_grad_scale = float(self.cfg.chunk_size) / float(seq_len)
            theta = theta.detach() + theta_grad_scale * (theta - theta.detach())

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
            "gamma_page_scale": gamma_page_scale,
            "gamma_chan_pair": gamma_chan_pair,
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

    def record_duty(
        self,
        duty_vec: torch.Tensor,
        *,
        write_logits: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> None:
        bud_loss = compute_write_budget_loss(
            duty_vec,
            self.mu_max,
            mu_min_ratio=self.cfg.mu_min_ratio,
            budget_weight=self.cfg.budget_weight,
        )
        if _in_backward():
            return
        self._pending_aux.append(bud_loss)
        if _is_compiling():
            return
        with torch.no_grad():
            det = duty_vec.detach()
            det_f = det.float()
            self.last_page_duty = det
            dead_cnt_t = (det_f < 1e-6).sum(dtype=torch.float32)
            if write_logits is not None and write_logits.numel() > 0:
                probs_d = F.softmax(write_logits.detach().float(), dim=-1)
                if pad_mask is not None and pad_mask.shape == probs_d.shape[:2]:
                    pm_f = (pad_mask.detach() > 0.5).float().unsqueeze(-1)
                    dense_vec = (probs_d * pm_f).sum(dim=(0, 1)) / pm_f.sum().clamp_min(1.0)
                else:
                    dense_vec = probs_d.mean(dim=(0, 1))
                self.last_dense_router_prob = dense_vec
                dense_prob_t = dense_vec.min()
            else:
                dense_prob_t = det_f.mean() if det_f.numel() > 0 else det_f.new_zeros(())
            vault_t = det_f[: min(2, det_f.numel())].mean() if det_f.numel() > 0 else det_f.new_zeros(())
            scratch_t = det_f[min(4, det_f.numel() - 1) :].mean() if det_f.numel() > 0 else det_f.new_zeros(())
            self._pending_stats_tensor = torch.stack(
                [bud_loss.detach().float(), vault_t, scratch_t, dense_prob_t, dead_cnt_t]
            )


def _extract_hssd_gdr_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_chunk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    HSSDCacheState | torch.Tensor | None,
    bool,
    bool,
    float | None,
    int,
    Any,
    torch.Tensor | None,
]:
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
        elif initial_state is None and (
            isinstance(extra, (HSSDCacheState, torch.Tensor)) or _is_hssd_state_duck(extra)
        ):
            initial_state = extra
    output_final = bool(kwargs.get("output_final_state", False))
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    scale = kwargs.get("scale")
    cu_seqlens = kwargs.get("cu_seqlens")
    seq_idx = kwargs.get("seq_idx")
    return g, beta, initial_state, output_final, use_l2, scale, chunk_size, cu_seqlens, seq_idx


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


@dataclass
class _HSSDRuntimeState:
    """Shared dynamic blend & teacher-capture control state across an `HSSDHandle`'s wrappers (`H2`)."""

    blend: float = 1.0
    bypass_to_orig: bool = False
    teacher_callbacks: dict[int, Callable[..., Any]] = field(default_factory=dict)
    next_cb_id: int = 0


class _CallbackHandle:
    """Removable handle returned by `HSSDHandle.register_teacher_step_callback`."""

    def __init__(self, callbacks: dict[int, Callable[..., Any]], cb_id: int):
        self._callbacks = callbacks
        self._cb_id = cb_id

    def remove(self) -> None:
        self._callbacks.pop(self._cb_id, None)


class HSSDHandle:
    """Manages `HSSDAdapter` modules, forward hooks, aux write-budget losses, blend, and teacher hooks."""

    def __init__(
        self,
        model: nn.Module,
        cfg: HSSDConfig,
        adapters: dict[int, HSSDAdapter],
        hooks: list[Any],
        restorers: list[Callable[[], None]],
        runtime: _HSSDRuntimeState | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.adapters = adapters
        self._hooks = hooks
        self._restorers = restorers
        self._runtime = runtime if runtime is not None else _HSSDRuntimeState(blend=float(cfg.blend_init))
        self._removed = False

    @property
    def blend(self) -> float:
        return float(self._runtime.blend)

    def set_blend(self, lam: float) -> None:
        lam_f = float(lam)
        if not -1e-6 <= lam_f <= 1.0 + 1e-6:
            raise ValueError(f"HSSD blend lambda must be in [0, 1], got {lam}")
        self._runtime.blend = max(0.0, min(1.0, lam_f))

    def step_blend(self, global_step: int) -> float:
        w_steps = int(self.cfg.blend_warmup_steps)
        if w_steps <= 0:
            lam = 1.0
        else:
            prog = max(0.0, min(1.0, float(global_step) / float(w_steps)))
            b0 = float(self.cfg.blend_init)
            lam = b0 + prog * (1.0 - b0)
        self.set_blend(lam)
        return self.blend

    @property
    def bypass_to_orig(self) -> bool:
        return bool(self._runtime.bypass_to_orig)

    def set_bypass_to_orig(self, bypass: bool) -> None:
        self._runtime.bypass_to_orig = bool(bypass)

    @contextlib.contextmanager
    def teacher_mode(self):
        """Force all wrapped linear-attention layers to execute stock `orig_fn` (zero-extra-VRAM teacher)."""
        prev = self._runtime.bypass_to_orig
        self._runtime.bypass_to_orig = True
        try:
            yield self
        finally:
            self._runtime.bypass_to_orig = prev

    def register_teacher_step_callback(self, fn: Callable[..., Any]) -> _CallbackHandle:
        """Register `fn(layer_idx, adapter, orig_out, query, key, value, g, beta, x_in, mor_wmask, attention_mask)`."""
        cid = self._runtime.next_cb_id
        self._runtime.next_cb_id += 1
        self._runtime.teacher_callbacks[cid] = fn
        return _CallbackHandle(self._runtime.teacher_callbacks, cid)

    @contextlib.contextmanager
    def teacher_layerwise_Hook(self, callback: Callable[..., Any]):
        """Execute `teacher_mode()` while invoking `callback` at every wrapped linear-attention layer (`5.2 #2`)."""
        cb_handle = self.register_teacher_step_callback(callback)
        try:
            with self.teacher_mode():
                yield self
        finally:
            cb_handle.remove()

    teacher_layerwise_hook = teacher_layerwise_Hook

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters() if p.requires_grad]

    def parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters()]

    def clear_aux(self) -> None:
        for ad in self.adapters.values():
            ad.clear_aux()

    def aux_loss(self, *, clear: bool = True) -> torch.Tensor:
        """Aggregate two-sided cumulative actual-sparse write-mass budget loss across layers & passes."""
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
        pending_ads = [ad for ad in self.adapters.values() if ad._pending_stats_tensor is not None]
        if pending_ads:
            by_dev: dict[torch.device, list[HSSDAdapter]] = {}
            for ad in pending_ads:
                assert ad._pending_stats_tensor is not None
                by_dev.setdefault(ad._pending_stats_tensor.device, []).append(ad)
            for dev_ads in by_dev.values():
                rows = torch.stack([ad._pending_stats_tensor for ad in dev_ads], dim=0).tolist()  # type: ignore[misc]
                for ad, vals in zip(dev_ads, rows):
                    ad._pending_stats_tensor = None
                    ad._last_stats = {
                        "budget_aux_loss": float(vals[0]),
                        "vault_duty": float(vals[1]),
                        "scratch_duty": float(vals[2]),
                        "dense_router_prob": float(vals[3]),
                        "dead_page_count": float(vals[4]),
                    }
        out: dict[str, float] = {}
        stat_keys = (
            "budget_aux_loss",
            "vault_duty",
            "scratch_duty",
            "dense_router_prob",
            "dead_page_count",
        )
        for key in stat_keys:
            vals = [ad.last_stats[key] for ad in self.adapters.values() if key in ad.last_stats]
            if vals:
                out[f"hssd/{key}"] = sum(vals) / len(vals)
        return out

    def uninstall(self) -> None:
        if self._removed:
            return
        self._removed = True
        self._runtime.teacher_callbacks.clear()
        self._runtime.bypass_to_orig = False
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for fn in reversed(self._restorers):
            fn()
        self._restorers.clear()
        if getattr(self.model, "_hssd_handle", None) is self:
            delattr(self.model, "_hssd_handle")
        log.info("HSSD v3.2 uninstalled from %d linear-attention layers", len(self.adapters))

    def remove(self) -> None:
        self.uninstall()


HSSDManager = HSSDHandle


_HSSD_CHUNK_RULE_ATTRS: tuple[str, ...] = ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule")
_HSSD_REC_RULE_ATTRS: tuple[str, ...] = (
    "recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


def _make_hssd_rule_wrapper(
    adapter: HSSDAdapter,
    ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]],
    *,
    recurrent_only: bool,
    orig_fn: Callable[..., Any] | None = None,
    in_rule_box: list[int] | None = None,
    runtime: _HSSDRuntimeState | None = None,
) -> Callable[..., Any]:
    def wrapped(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any):
        if in_rule_box is not None and in_rule_box[0] > 0:
            if orig_fn is not None:
                return orig_fn(query, key, value, *args, **kwargs)
            raise RuntimeError(f"HSSD layer {adapter.layer_id}: unexpected re-entrant rule call without orig_fn")
        if in_rule_box is not None:
            in_rule_box[0] += 1
        try:
            if runtime is not None and runtime.bypass_to_orig:
                if orig_fn is None:
                    raise RuntimeError(f"HSSD layer {adapter.layer_id}: teacher_mode requires orig_fn")
                orig_res = orig_fn(query, key, value, *args, **kwargs)
                if runtime.teacher_callbacks and ctx_stack:
                    x_in_cb, mor_wmask_cb, pad_mask_cb, _dfs_cb = ctx_stack[-1]
                    _bsz_cb, seq_len_cb, _ = x_in_cb.shape
                    g_cb, beta_cb, _ist, _of, _ul2, _sc, _cs, _cu, _si = _extract_hssd_gdr_args(
                        args, kwargs, adapter.cfg.chunk_size
                    )
                    orig_out = orig_res[0] if isinstance(orig_res, tuple) else orig_res
                    hf_cb = bool(kwargs.get("head_first", False)) or (
                        seq_len_cb != adapter.num_v_heads
                        and value.shape[1] == adapter.num_v_heads
                        and value.shape[2] == seq_len_cb
                    )
                    if hf_cb:
                        q_t, k_t, v_t = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                        g_t, b_t = g_cb.transpose(1, 2), beta_cb.transpose(1, 2)
                        o_t = orig_out.transpose(1, 2)
                    else:
                        q_t, k_t, v_t = query, key, value
                        g_t, b_t = g_cb, beta_cb
                        o_t = orig_out
                    for cb in list(runtime.teacher_callbacks.values()):
                        cb(
                            adapter.layer_id,
                            adapter,
                            o_t,
                            q_t,
                            k_t,
                            v_t,
                            g_t,
                            b_t,
                            x_in_cb,
                            mor_wmask_cb,
                            pad_mask_cb,
                        )
                return orig_res

            require(bool(ctx_stack), f"HSSD layer {adapter.layer_id}: missing captured input hidden_states")
            x_in, mor_wmask, pad_mask, detach_fs = ctx_stack[-1]
            bsz, seq_len, _ = x_in.shape
            g, beta, init_st, out_final, use_l2, scale, _csize, cu_seqlens, seq_idx = _extract_hssd_gdr_args(
                args, kwargs, adapter.cfg.chunk_size
            )

            orig_q, orig_k, orig_v = query, key, value
            head_first = bool(kwargs.get("head_first", False)) or (
                seq_len != adapter.num_v_heads and value.shape[1] == adapter.num_v_heads and value.shape[2] == seq_len
            )
            if head_first:
                query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                g, beta = g.transpose(1, 2), beta.transpose(1, 2)
            require(
                value.shape[:2] == (bsz, seq_len),
                f"HSSD layer {adapter.layer_id}: value shape {tuple(value.shape)} disagrees with input "
                f"{(bsz, seq_len)}",
            )

            proj = adapter.project_inputs(x_in, query, key, value, g, beta, use_qk_l2norm=use_l2)
            factored_gamma = (proj["gamma_page_scale"], proj["gamma_chan_pair"])
            c_cfg = adapter.cfg

            if seq_len == 1 and cu_seqlens is None and seq_idx is None:
                acc_dt = torch.float64 if query.dtype == torch.float64 else torch.float32
                target_dt = _resolve_state_dtype(c_cfg.state_dtype, acc_dt)
                st0 = _coerce_init_state(
                    init_st,
                    bsz=bsz,
                    num_pages=c_cfg.num_pages,
                    hv=adapter.num_v_heads,
                    dk=adapter.head_k_dim,
                    dv=adapter.head_v_dim,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    top_k_pages=c_cfg.k_pages,
                    compact_decode_cache=c_cfg.compact_decode_cache,
                    device=query.device,
                    dtype=target_dt,
                )
                out, st_next, u_sum, cnt_sum = recurrent_hssd_step(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
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
                    swa_recency_slope=adapter.swa_recency_slope,
                    top_k_pages=c_cfg.k_pages,
                    segment_chunks=c_cfg.segment_chunks,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=target_dt,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    write_mode=c_cfg.write_mode,
                    compact_decode_cache=c_cfg.compact_decode_cache,
                    inplace_state=(not torch.is_grad_enabled() and out_final and isinstance(init_st, HSSDCacheState)),
                    return_duty_sum=True,
                )
                final_state = st_next if out_final else None
                duty_vec = u_sum / cnt_sum.clamp_min(1.0)
            elif recurrent_only:
                out, final_state, duty_vec = recurrent_hssd(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
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
                    swa_recency_slope=adapter.swa_recency_slope,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    num_pages=c_cfg.num_pages,
                    top_k_pages=c_cfg.k_pages,
                    segment_chunks=c_cfg.segment_chunks,
                    initial_state=init_st,
                    output_final_state=out_final,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    cu_seqlens=cu_seqlens,
                    seq_idx=seq_idx,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=c_cfg.state_dtype,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    write_mode=c_cfg.write_mode,
                    compact_active_pages=c_cfg.compact_active_pages,
                    compact_decode_cache=c_cfg.compact_decode_cache,
                    return_aux=True,
                )
            else:
                out, final_state, duty_vec = chunk_hssd(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
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
                    swa_recency_slope=adapter.swa_recency_slope,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    num_pages=c_cfg.num_pages,
                    top_k_pages=c_cfg.k_pages,
                    segment_chunks=c_cfg.segment_chunks,
                    checkpoint_segments=c_cfg.checkpoint_segments,
                    checkpoint_segments_min_len=c_cfg.checkpoint_segments_min_len,
                    outer_checkpoint_max_tokens=c_cfg.outer_checkpoint_max_tokens,
                    initial_state=init_st,
                    output_final_state=out_final,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    cu_seqlens=cu_seqlens,
                    seq_idx=seq_idx,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=c_cfg.state_dtype,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    write_mode=c_cfg.write_mode,
                    compact_active_pages=c_cfg.compact_active_pages,
                    compact_decode_cache=c_cfg.compact_decode_cache,
                    compile_chunks=c_cfg.compile_chunks,
                    detach_final_state=detach_fs,
                    return_aux=True,
                )

            adapter.record_duty(duty_vec, write_logits=proj["write_logits"], pad_mask=pad_mask)

            # H2: Scheduled stock-to-HSSD output blend (stateless training forward; skipped when lam >= 1.0).
            # Keep live gradients through `(1 - lam) * stock_out` so multi-layer backward Jacobians match the
            # actual forward computation instead of substituting an amplified `0.1 * J_hssd` surrogate.
            lam = 1.0 if runtime is None else float(runtime.blend)
            if lam < 1.0 and init_st is None and not out_final and orig_fn is not None:
                need_stock_grad = bool(
                    torch.is_grad_enabled()
                    and out.requires_grad
                    and (
                        orig_q.requires_grad
                        or orig_k.requires_grad
                        or orig_v.requires_grad
                        or g.requires_grad
                        or beta.requires_grad
                    )
                )
                with torch.enable_grad() if need_stock_grad else torch.no_grad():
                    stock_res = orig_fn(orig_q, orig_k, orig_v, *args, **kwargs)
                    stock_out = stock_res[0] if isinstance(stock_res, tuple) else stock_res
                    if head_first:
                        stock_out = stock_out.transpose(1, 2)
                    stock_out = stock_out.to(dtype=out.dtype)
                if lam <= 0.0:
                    if out.requires_grad:
                        ste_eps = 1.0 / float(max(int(c_cfg.blend_warmup_steps), 100))
                        out = stock_out + ste_eps * (out - out.detach())
                    else:
                        out = stock_out
                else:
                    out = ((1.0 - lam) * upcast(stock_out) + lam * upcast(out)).to(dtype=out.dtype)

            if head_first:
                out = out.transpose(1, 2)
            return out, final_state
        finally:
            if in_rule_box is not None:
                in_rule_box[0] -= 1

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
    """Retrofit Qwen3.6 / Toy linear-attention layers in-place to `HSSD v3.2`.

    Works seamlessly with `ToyForCausalLM` (`ToyHybridMoEForCausalLM`), HF Qwen3.6 GDN layers,
    `apply_loop` per-pass virtual cache slots, MoR early-exit write masks, 2D `attention_mask`
    padding invariance, and packed `cu_seqlens` / `seq_idx` document boundaries.
    """
    if getattr(model, "_hssd_handle", None) is not None:
        raise RuntimeError("HSSD v3 is already installed on this model; call remove_hssd(model) first")
    _ensure_dynamo_mark_static_address_compat()
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
    runtime = _HSSDRuntimeState(blend=float(cfg.blend_init))
    adapters: dict[int, HSSDAdapter] = {}
    hooks: list[Any] = []
    restorers: list[Callable[[], None]] = []
    patched_mod_fns: set[tuple[str, str]] = set()
    active_attn_stack: list[
        tuple[
            HSSDAdapter,
            list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]],
            list[int],
            bool,
            bool,
        ]
    ] = []
    amask_stack: list[torch.Tensor | None] = []

    rule_specs: list[tuple[str, bool]] = (
        [(attr, False) for attr in _HSSD_CHUNK_RULE_ATTRS]
        + [(attr, True) for attr in _HSSD_REC_RULE_ATTRS]
    )

    def _capture_model_amask(_mod: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
        am = kw.get("attention_mask")
        if am is None and len(a) >= 2 and isinstance(a[1], torch.Tensor) and a[1].ndim == 2:
            am = a[1]
        if isinstance(am, torch.Tensor) and am.ndim == 2:
            amask_stack.append(am)
        elif amask_stack:
            amask_stack.append(amask_stack[-1])
        else:
            amask_stack.append(None)

    def _pop_model_amask(_mod: nn.Module, _inp: Any, _out: Any) -> None:
        if amask_stack:
            amask_stack.pop()

    def _clear_model_aux(_mod: nn.Module, _args: tuple[Any, ...]) -> None:
        if not _in_backward():
            for ad in adapters.values():
                ad.clear_aux()

    hooks.append(model.register_forward_pre_hook(_capture_model_amask, with_kwargs=True))
    hooks.append(model.register_forward_hook(_pop_model_amask))
    if text_model is not model:
        hooks.append(text_model.register_forward_pre_hook(_capture_model_amask, with_kwargs=True))
        hooks.append(text_model.register_forward_hook(_pop_model_amask))
    hooks.append(text_model.register_forward_pre_hook(_clear_model_aux))

    for i in target_layers:
        attn = attention_module(phys_layers[i], LINEAR)
        hs, hv, dk, dv = _infer_gdn_dims(attn, i)
        adapter = HSSDAdapter(hs, hv, dk, dv, cfg, i).to(device=ref.device, dtype=param_dtype)
        attn.hssd_adapter = adapter
        adapters[i] = adapter

        ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]] = []
        in_rule_box: list[int] = [0]

        def _del_adapter(a: nn.Module = attn, ad: HSSDAdapter = adapter) -> None:
            if getattr(a, "hssd_adapter", None) is ad:
                delattr(a, "hssd_adapter")

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
                rule_w = _make_hssd_rule_wrapper(
                    adapter,
                    ctx_stack,
                    recurrent_only=rec_only,
                    orig_fn=orig_fn,
                    in_rule_box=in_rule_box,
                    runtime=runtime,
                )
                if isinstance(orig_fn, types.MethodType) and getattr(orig_fn, "__self__", None) is attn:
                    raw_rfn = getattr(orig_fn, "__func__", orig_fn)

                    @functools.wraps(raw_rfn)
                    def _bound_hssd_rule(
                        _self: nn.Module,
                        *r_args: Any,
                        _rw: Callable[..., Any] = rule_w,
                        **r_kwargs: Any,
                    ) -> Any:
                        return _rw(*r_args, **r_kwargs)

                    rule_patch: Any = types.MethodType(_bound_hssd_rule, attn)
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
            ad_ref: HSSDAdapter,
            st_ref: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]],
            ir_ref: list[int],
            wc_flag: bool,
            wr_flag: bool,
        ):
            def pre_hook(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                x_in = (
                    args[0]
                    if args
                    else (kwargs.get("hidden_states") if "hidden_states" in kwargs else kwargs.get("x"))
                )
                require(isinstance(x_in, torch.Tensor), "HSSD pre_hook could not locate input hidden_states")
                amask = kwargs.get("attention_mask", args[2] if len(args) >= 3 else None)
                if amask is None and amask_stack:
                    amask = amask_stack[-1]
                cache_p = kwargs.get("cache_params")
                if cache_p is None:
                    cache_p = kwargs.get("past_key_values")
                if cache_p is None:
                    for _arg in args[1:]:
                        if getattr(_arg, "_detach_recurrent_state", False):
                            cache_p = _arg
                            break
                detach_fs = bool(getattr(cache_p, "_detach_recurrent_state", False))
                mor_wmask = _active_mor_write_mask(text_model, x_in.shape[0], x_in.shape[1])
                pad_mask: torch.Tensor | None = None
                ret_modified = False
                if (
                    isinstance(amask, torch.Tensor)
                    and amask.ndim == 2
                    and amask.shape[0] == x_in.shape[0]
                    and amask.shape[1] >= x_in.shape[1]
                ):
                    pad_mask = amask[:, -x_in.shape[1] :]
                    if not runtime.bypass_to_orig and not bool((pad_mask > 0.5).all()):
                        x_in = x_in * pad_mask.to(dtype=x_in.dtype).unsqueeze(-1)
                        ret_modified = True
                x_proj = x_in * mor_wmask.to(dtype=x_in.dtype).unsqueeze(-1) if mor_wmask is not None else x_in
                st_ref.append((x_proj, mor_wmask, pad_mask, detach_fs))
                active_attn_stack.append((ad_ref, st_ref, ir_ref, wc_flag, wr_flag))
                if ret_modified:
                    if args:
                        return (x_in, *args[1:]), kwargs
                    new_kw = dict(kwargs)
                    if "hidden_states" in new_kw:
                        new_kw["hidden_states"] = x_in
                    else:
                        new_kw["x"] = x_in
                    return args, new_kw
                return None

            return pre_hook

        def _make_post(st_ref: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]]):
            def post_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
                if st_ref:
                    st_ref.pop()
                if active_attn_stack:
                    active_attn_stack.pop()

            return post_hook

        hooks.append(
            attn.register_forward_pre_hook(
                _make_pre(adapter, ctx_stack, in_rule_box, wrapped_chunk, wrapped_rec),
                with_kwargs=True,
            )
        )
        hooks.append(attn.register_forward_hook(_make_post(ctx_stack)))

        if not (wrapped_chunk and wrapped_rec):
            mod_name = type(attn).__module__
            host_mod = sys.modules.get(mod_name)
            if host_mod is not None:
                missing_specs: list[tuple[str, bool]] = []
                if not wrapped_chunk:
                    missing_specs.extend((nm, False) for nm in _HSSD_CHUNK_RULE_ATTRS)
                if not wrapped_rec:
                    missing_specs.extend((nm, True) for nm in _HSSD_REC_RULE_ATTRS)
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
                            cur_ad, cur_st, cur_ir, cur_wc, cur_wr = active_attn_stack[-1]
                            if cur_ir[0] > 0 or (_rec and cur_wr) or ((not _rec) and cur_wc):
                                return _orig(query, key, value, *args, **kwargs)
                            return _make_hssd_rule_wrapper(
                                cur_ad,
                                cur_st,
                                recurrent_only=_rec,
                                orig_fn=_orig,
                                in_rule_box=cur_ir,
                                runtime=runtime,
                            )(query, key, value, *args, **kwargs)

                        setattr(host_mod, fn_name, _mod_dispatch)
                        restorers.append(lambda hm=host_mod, n=fn_name, f=orig_mfn: setattr(hm, n, f))

    handle = HSSDHandle(model, cfg, adapters, hooks, restorers, runtime=runtime)
    object.__setattr__(model, "_hssd_handle", handle)
    n_params = sum(p.numel() for p in handle.parameters())
    log.info(
        "HSSD v3.2 installed on %d linear-attention layers "
        "(rank=%d, low_rank_theta=%s, chunk=%d, window=%d, pages=%d, write_mode=%s, state_dtype=%s, params=%.2fM)",
        len(adapters),
        cfg.rank,
        cfg.theta_low_rank,
        cfg.chunk_size,
        cfg.window_chunks,
        cfg.num_pages,
        cfg.write_mode,
        cfg.state_dtype,
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


def profile_hssd_layer(
    *,
    hidden_size: int = 2048,
    num_v_heads: int = 32,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    seq_len: int = 1024,
    stream_chunk_len: int = 256,
    state_dtype: str = "bfloat16",
    theta_low_rank: bool = True,
    rank: int = 64,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> dict[str, Any]:
    """Re-export of `qloop.model.probes.profile_hssd_layer`."""
    from .probes import profile_hssd_layer as _profile_fn

    return _profile_fn(
        hidden_size=hidden_size,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        seq_len=seq_len,
        stream_chunk_len=stream_chunk_len,
        state_dtype=state_dtype,
        theta_low_rank=theta_low_rank,
        rank=rank,
        device=device,
        dtype=dtype,
        **kwargs,
    )


def hssd_ablation_matrix() -> dict[str, dict[str, Any]]:
    """Re-export of `qloop.model.probes.hssd_ablation_matrix`."""
    from .probes import hssd_ablation_matrix as _ablation_fn

    return _ablation_fn()


__all__ = [
    "DEFAULT_MU_MAX",
    "HSSDAdapter",
    "HSSDCacheState",
    "HSSDConfig",
    "HSSDHandle",
    "HSSDHybridCache",
    "HSSDManager",
    "PeakBoltzmannGate",
    "apply_2d_rotation",
    "apply_2d_rotation_trig",
    "apply_hssd",
    "chunk_hssd",
    "compute_2d_trig",
    "compute_write_budget_loss",
    "disjoint_horizon_bounds",
    "has_fla",
    "hssd_ablation_matrix",
    "install_hssd",
    "profile_hssd_layer",
    "recurrent_hssd",
    "recurrent_hssd_step",
    "remove_hssd",
]
